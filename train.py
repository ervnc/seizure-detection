import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from drive_connection import auth_drive
from helpers.chbmit_helpers import get_patient_folder_id, list_patient_edfs
from readers.chbmit_reader import build_windows_and_labels
from processors.wavelet import extract_features_wavelet
from models.hybrid_model import build_cnn_lstm_model

import matplotlib.pyplot as plt


def plot_training_history(history):
    """
    Gera gráficos de Acurácia e Loss lado a lado e salva como imagem.
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(14, 6))

    # Gráfico 1: Acurácia
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Treino (Acc)', color='blue')
    plt.plot(epochs_range, val_acc, label='Validação (Acc)', color='orange', linestyle='--')
    plt.legend(loc='lower right')
    plt.title('Evolução da Acurácia')
    plt.xlabel('Épocas')
    plt.ylabel('Acurácia')
    plt.grid(True, alpha=0.3)

    # Gráfico 2: Loss (Erro)
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Treino (Loss)', color='blue')
    plt.plot(epochs_range, val_loss, label='Validação (Loss)', color='red', linestyle='--')
    plt.legend(loc='upper right')
    plt.title('Evolução do Erro (Loss)')
    plt.xlabel('Épocas')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('resultado_treino.png') # Salva no disco automaticamente
    print("\n[INFO] Gráfico salvo como 'resultado_treino.png'")
    plt.show() # Abre a janela pra você ver

    
# --- CONFIGURAÇÕES ---
FOLDER_ID = "1nJm3E6XnYVVFz2itBBdC-qtSab6GZmLO"
PATIENT = "chb01"
EPOCHS = 30        # Mais épocas para dar tempo de aprender
BATCH_SIZE = 8     # Batch menor ajuda a generalizar melhor com poucos dados

def main():
    print("--- INICIANDO TREINAMENTO MULTI-ARQUIVO ---")
    
    service = auth_drive()
    patient_id = get_patient_folder_id(service, FOLDER_ID, PATIENT)
    
    # 1. PEGAR TODOS OS ARQUIVOS COM CRISE (Não apenas o primeiro)
    edfs = list_patient_edfs(service, patient_id)
    seizure_files = [f for f in edfs if f['has_seizures_file']]
    
    if not seizure_files:
        print("ERRO: Nenhum arquivo de crise encontrado.")
        return

    print(f">> Encontrados {len(seizure_files)} arquivos com crise. Processando todos...")

    all_X = []
    all_y = []

    # Loop para acumular dados de todos os arquivos
    for edf_row in seizure_files:
        print(f"  -> Processando {edf_row['name']}...")
        try:
            raw, windows, y = build_windows_and_labels(
                service, FOLDER_ID, PATIENT, edf_row["name"],
                window_s=2.0, step_s=0.5
            )
            # Extrai apenas se tiver janelas
            if len(windows) > 0:
                X = extract_features_wavelet(raw, windows)
                all_X.append(X)
                all_y.append(y)
        except Exception as e:
            print(f"  [ERRO] Falha ao ler {edf_row['name']}: {e}")
            continue

    # Junta tudo num array gigante
    X_raw = np.concatenate(all_X, axis=0)
    y_raw = np.concatenate(all_y, axis=0)
    
    print(f"\n>> Total acumulado: {len(y_raw)} janelas.")

    # 2. BALANCEAMENTO INTELIGENTE
    idx_seizure = np.where(y_raw == 1)[0]
    idx_normal = np.where(y_raw == 0)[0]
    
    print(f"   Crises Totais: {len(idx_seizure)}")
    print(f"   Normais Totais: {len(idx_normal)}")
    
    # Estratégia: Manter proporção 1:1 estrita para forçar aprendizado
    n_samples = len(idx_seizure)
    if len(idx_normal) > n_samples:
        np.random.shuffle(idx_normal)
        selected_normals = idx_normal[:n_samples] # Pega exatamente a mesma quantidade
    else:
        selected_normals = idx_normal

    idx_final = np.concatenate([idx_seizure, selected_normals])
    np.random.shuffle(idx_final)
    
    X_final = X_raw[idx_final]
    y_final = y_raw[idx_final]
    
    print(f"[Dataset Final 50/50] Shape: {X_final.shape}")

    # 3. SPLIT
    X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.2, random_state=42)

    # 4. CALCULAR PESOS DE CLASSE (Class Weights)
    # Isso garante que se a rede errar uma crise, a penalidade é alta
    class_weights = compute_class_weight(
        class_weight='balanced', 
        classes=np.unique(y_train), 
        y=y_train
    )
    class_weights_dict = dict(enumerate(class_weights))
    print(f"Pesos das Classes: {class_weights_dict}")

    # 5. MODELO E TREINO
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_cnn_lstm_model(input_shape)

    # Callbacks para parar se melhorar
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        class_weight=class_weights_dict, 
        callbacks=[early_stop],
        verbose=1
    )

    plot_training_history(history)
    
    # E ESTA PARA SALVAR O MODELO:
    model.save("modelo_final_epilepsia.keras")
    print("[INFO] Modelo salvo como 'modelo_final_epilepsia.keras'")


    # 6. AVALIAÇÃO
    print("\n--- RESULTADOS FINAIS ---")
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Crise'], zero_division=0))
    print("Matriz de Confusão:")
    print(confusion_matrix(y_test, y_pred))

    model.save("modelo_chb01_97acc.keras")
    print("Modelo salvo com sucesso!")

if __name__ == "__main__":
    main()