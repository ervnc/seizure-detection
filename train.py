import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Seus módulos
from drive_connection import auth_drive
from helpers.chbmit_helpers import get_patient_folder_id, list_patient_edfs
from readers.chbmit_reader import build_windows_and_labels
from processors.wavelet import extract_features_wavelet
from models.hybrid_model import build_cnn_lstm_model

# Configurações
FOLDER_ID = "1nJm3E6XnYVVFz2itBBdC-qtSab6GZmLO"
PATIENT = "chb01" # Comece com um paciente só para testar!

def main():
    service = auth_drive()
    patient_id = get_patient_folder_id(service, FOLDER_ID, PATIENT)
    
    # Pega o primeiro arquivo que tem crise (pra garantir que temos dados positivos)
    edfs = list_patient_edfs(service, patient_id)
    seizure_files = [f for f in edfs if f['has_seizures_file']]
    
    if not seizure_files:
        print("Nenhum arquivo com crise encontrado para esse paciente!")
        return

    # Vamos usar apenas o PRIMEIRO arquivo com crise para o teste inicial (economizar tempo)
    target_edf = seizure_files[0]
    print(f"--- Carregando {target_edf['name']} ---")

    # 1. Pipeline de Dados
    raw, windows, y = build_windows_and_labels(
        service, FOLDER_ID, PATIENT, target_edf["name"],
        window_s=2.0, step_s=0.5
    )
    
    # 2. Pipeline de Processamento (Wavelet)
    # Se seu PC travar, diminua o número de janelas aqui: ex windows[:500]
    print("Extraindo features...")
    X = extract_features_wavelet(raw, windows)

    # 3. Balanceamento de Dados (CRÍTICO)
    # Seleciona indices onde tem crise (1) e onde não tem (0)
    idx_seizure = np.where(y == 1)[0]
    idx_normal = np.where(y == 0)[0]
    
    print(f"Total Crises: {len(idx_seizure)} | Total Normal: {len(idx_normal)}")
    
    # Corta o excesso de normais para ficar 50/50 (ou próximo disso)
    # Isso é "Undersampling"
    if len(idx_normal) > len(idx_seizure):
        np.random.shuffle(idx_normal)
        idx_normal = idx_normal[:len(idx_seizure) * 2] # Pega 2x mais normais que crises (proporção 1:2)
    
    # Junta tudo
    idx_final = np.concatenate([idx_seizure, idx_normal])
    X_final = X[idx_final]
    y_final = y[idx_final]
    
    # 4. Split Treino/Teste
    X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.2, random_state=42)

    # 5. Construir e Treinar Modelo
    input_shape = (X_train.shape[1], X_train.shape[2]) # (TimeSteps, Features)
    model = build_cnn_lstm_model(input_shape)
    
    print("\n--- Iniciando Treinamento ---")
    model.fit(
        X_train, y_train, 
        epochs=10,             # Aumente se tiver tempo
        batch_size=32, 
        validation_data=(X_test, y_test),
        verbose=1
    )

    # 6. Resultados
    print("\n--- Avaliação Final ---")
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    print(classification_report(y_test, y_pred))
    print("Matriz de Confusão:")
    print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    main()