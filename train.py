import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler  # <--- IMPORTANTE

from drive_connection import auth_drive
from helpers.chbmit_helpers import get_patient_folder_id, list_patient_edfs
from readers.chbmit_reader import build_windows_and_labels
from processors.wavelet import extract_features_wavelet
from models.hybrid_model import build_cnn_lstm_model

import matplotlib.pyplot as plt

def plot_training_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Treino (Acc)', color='blue')
    plt.plot(epochs_range, val_acc, label='Validação (Acc)', color='orange', linestyle='--')
    plt.legend(loc='lower right')
    plt.title('Acurácia')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Treino (Loss)', color='blue')
    plt.plot(epochs_range, val_loss, label='Validação (Loss)', color='red', linestyle='--')
    plt.legend(loc='upper right')
    plt.title('Loss')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('resultado_treino.png')
    print("\n[INFO] Gráfico salvo como 'resultado_treino.png'")

# --- CONFIGURAÇÕES ---
FOLDER_ID = "1nJm3E6XnYVVFz2itBBdC-qtSab6GZmLO"
PATIENT = "chb01"
EPOCHS = 50        # Aumentei para dar tempo de convergir com LR baixo
BATCH_SIZE = 16    

def main():
    print("--- INICIANDO TREINAMENTO ROBUSTO ---")
    service = auth_drive()
    patient_id = get_patient_folder_id(service, FOLDER_ID, PATIENT)
    
    edfs = list_patient_edfs(service, patient_id)
    seizure_files = [f for f in edfs if f['has_seizures_file']]
    
    if not seizure_files: return

    all_X = []
    all_y = []

    print(f">> Processando {len(seizure_files)} arquivos...")
    for edf_row in seizure_files:
        try:
            raw, windows, y = build_windows_and_labels(
                service, FOLDER_ID, PATIENT, edf_row["name"],
                window_s=2.0, step_s=0.5
            )
            if len(windows) > 0:
                X = extract_features_wavelet(raw, windows)
                all_X.append(X)
                all_y.append(y)
        except Exception as e:
            print(f"  [Skip] {edf_row['name']}: {e}")
            continue

    X_raw = np.concatenate(all_X, axis=0)
    y_raw = np.concatenate(all_y, axis=0)
    
    # --- BALANCEAMENTO ---
    idx_seizure = np.where(y_raw == 1)[0]
    idx_normal = np.where(y_raw == 0)[0]
    
    n_samples = len(idx_seizure)
    # Undersampling dos normais
    if len(idx_normal) > n_samples:
        np.random.shuffle(idx_normal)
        selected_normals = idx_normal[:n_samples] 
    else:
        selected_normals = idx_normal

    idx_final = np.concatenate([idx_seizure, selected_normals])
    np.random.shuffle(idx_final)
    
    X_final = X_raw[idx_final]
    y_final = y_raw[idx_final]
    
    print(f"[Dataset] Shape: {X_final.shape} (Balanceado)")

    # --- SPLIT ---
    # Stratify garante que a proporção de crises seja igual no treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X_final, y_final, test_size=0.2, random_state=42, stratify=y_final
    )

    # --- NORMALIZAÇÃO (CRÍTICO) ---
    # StandardScaler funciona em 2D, precisamos achatar, escalar e voltar pra 3D
    scaler = StandardScaler()
    
    N, T, F = X_train.shape
    X_train_reshaped = X_train.reshape(-1, F)
    X_test_reshaped = X_test.reshape(-1, F)
    
    # Ajusta o scaler SÓ no treino para evitar vazamento de dados
    X_train_scaled = scaler.fit_transform(X_train_reshaped).reshape(N, T, F)
    X_test_scaled = scaler.transform(X_test_reshaped).reshape(X_test.shape[0], T, F)
    
    print(">> Dados Normalizados.")

    # --- MODELO ---
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights_dict = dict(enumerate(class_weights))

    model = build_cnn_lstm_model((T, F))
    
    # Otimizador com Learning Rate menor para evitar pular o mínimo
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    
    model.compile(
        optimizer=optimizer, 
        loss='binary_crossentropy', 
        metrics=['accuracy']
    )

    # Paciência maior
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True
    )

    history = model.fit(
        X_train_scaled, y_train, # Usando dados SCALED
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test_scaled, y_test),
        class_weight=class_weights_dict,
        callbacks=[early_stop],
        verbose=1
    )

    plot_training_history(history)
    model.save("modelo_final_epilepsia.keras")

    # --- AVALIAÇÃO ---
    print("\n--- RESULTADOS FINAIS ---")
    y_pred = (model.predict(X_test_scaled) > 0.5).astype("int32")
    
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Crise'], zero_division=0))
    print("Matriz de Confusão:")
    print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    main()