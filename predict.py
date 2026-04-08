import mne
import numpy as np
import tensorflow as tf
import joblib
import argparse
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Importa as mesmas funções usadas no treino para garantir consistência
from processors.wavelet import extract_features_wavelet
from helpers.chbmit_helpers import make_windows

def predict_pipeline(edf_path, model_path='modelo_final_epilepsia.keras', scaler_path='scaler_treinado.pkl'):
    # 1. Validação de Arquivos
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print("ERRO CRÍTICO: Você precisa treinar o modelo primeiro (rode train.py).")
        print("Certifique-se de que 'modelo_final_epilepsia.keras' e 'scaler_treinado.pkl' existem.")
        return

    print(f"--- Carregando Artefatos ---")
    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    print("Modelo e Scaler carregados.")

    # 2. Leitura e Pré-processamento do EDF (Igual ao chbmit_reader.py)
    print(f"--- Lendo {edf_path} ---")
    # Carrega em memória
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    
    # Filtro de Banda (0.5 - 45 Hz) - Essencial para remover ruído DC e alta frequência
    raw.filter(l_freq=0.5, h_freq=45.0, fir_design="firwin", verbose=False)
    
    # Resample para 256 Hz (A rede espera essa densidade de dados)
    if raw.info['sfreq'] != 256:
        print(f"Reamostrando de {raw.info['sfreq']} Hz para 256 Hz...")
        raw.resample(256, npad="auto")

    # Seleciona apenas canais EEG (remove ECG, etc se houver mix)
    # Nota: Se os canais forem diferentes do treino, a Wavelet vai quebrar. 
    # Assumindo consistência do CHB-MIT.
    if 'eeg' in raw:
        raw.pick_types(eeg=True)

    # 3. Janelamento (2s janela, 0.5s passo)
    sf = raw.info['sfreq']
    windows = make_windows(raw.n_times, sf, window_s=2.0, step_s=0.5)
    print(f"Geradas {len(windows)} janelas de análise.")

    # 4. Feature Extraction (Wavelet db4)
    # Isso transforma o sinal bruto em tensores que a rede entende
    X = extract_features_wavelet(raw, windows, wavelet='db4', level=4)
    
    # 5. Normalização (CRUCIAL)
    # Achatamos para 2D -> aplicamos a régua do treino -> voltamos para 3D
    N, T, F = X.shape
    X_flat = X.reshape(-1, F)
    X_scaled = scaler.transform(X_flat).reshape(N, T, F)

    # 6. Inferência (Predição)
    print("--- Analisando Atividade Cerebral ---")
    # verbose=1 mostra barra de progresso
    probs = model.predict(X_scaled, verbose=1)
    
    # Limiar de decisão (padrão 0.5, mas ajustável)
    #threshold = 0.5
    THRESHOLD_CONFIDENCE = 0.85
    #predictions = (probs > threshold).astype(int)
    raw_predictions = (probs > THRESHOLD_CONFIDENCE).astype(int).flatten()


    MIN_CONSECUTIVE_WINDOWS = 15  # ~7 a 8 segundos contínuos
    print(f">> Aplicando filtro: Mínimo de {MIN_CONSECUTIVE_WINDOWS} janelas consecutivas com confiança > {THRESHOLD_CONFIDENCE*100}%")

    final_detections = []
    current_streak = 0
    start_idx = -1
    
    for i, pred in enumerate(raw_predictions):
        if pred == 1:
            if current_streak == 0:
                start_idx = i
            current_streak += 1
        else:
            # Se a sequência quebrou, verificamos se ela foi longa o suficiente
            if current_streak >= MIN_CONSECUTIVE_WINDOWS:
                final_detections.append((start_idx, i - 1))
            current_streak = 0
            start_idx = -1
            
    # Caso a crise vá até o final do arquivo
    if current_streak >= MIN_CONSECUTIVE_WINDOWS:
         final_detections.append((start_idx, len(raw_predictions) - 1))

    # 7. Relatório Final Filtrado
    if len(final_detections) == 0:
        print("\n>>> RESULTADO FINAL: Normal (Nenhuma crise sustentada detectada).")
        print(f"    (Nota: O modelo pode ter visto {np.sum(raw_predictions)} janelas suspeitas isoladas, mas foram descartadas como ruído).")
    else:
        print(f"\n>>> ALERTA CONFIRMADO: Detectados {len(final_detections)} eventos epilépticos sustentados.")
        
        for start_win, end_win in final_detections:
            # Converter índice de janela para segundos
            # Janela = 2s, Step = 0.5s.
            # Tempo = indice * 0.5
            t_start = start_win * 0.5
            t_end = (end_win * 0.5) + 2.0 # +2.0 pela duração da última janela
            duration = t_end - t_start
            
            print(f"  [EVENTO] {t_start:.2f}s até {t_end:.2f}s (Duração: {duration:.2f}s)")

    plt.figure(figsize=(18, 5))
    plt.plot(probs.flatten(), label="Probabilidade de Crise", color='blue')
    plt.plot(raw_predictions * 1.05, label="Predição Binária (> limiar)", color='red', alpha=0.5)
    
    for (start_win, end_win) in final_detections:
        plt.axvspan(start_win, end_win, color='orange', alpha=0.3, label='Crise Detectada' if start_win == final_detections[0][0] else None)

    plt.axhline(THRESHOLD_CONFIDENCE, color='green', linestyle='--', label=f"Limiar = {THRESHOLD_CONFIDENCE}")
    plt.title("Probabilidade de Crise por Janela")
    plt.xlabel("Janela (cada passo = 0.5s)")
    plt.ylabel("Probabilidade")
    plt.legend(loc='upper right')
    plt.grid(alpha=0.3)

    plt.savefig("predicao_epilepsia.png")

if __name__ == "__main__":
    # Uso via linha de comando
    parser = argparse.ArgumentParser(description='Detector de Epilepsia em Arquivos EDF')
    parser.add_argument('edf_file', type=str, help='Caminho para o arquivo .edf')
    args = parser.parse_args()
    
    predict_pipeline(args.edf_file)