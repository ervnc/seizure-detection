import numpy as np
import pywt
from tqdm import tqdm

def extract_features_wavelet(raw, windows, wavelet='db4', level=4):
    """
    Recorta as janelas do sinal EEG bruto e aplica Transformada Wavelet.
    
    Args:
        raw: Objeto MNE carregado com os dados.
        windows: Array numpy (N, 2) com índices [start, end].
        wavelet: Nome da wavelet (ex: 'db4', 'sym5').
        level: Nível de decomposição.
        
    Returns:
        X: Array 3D (N_Janelas, Time_Steps_Reduzido, N_Canais) pronto para LSTM.
    """
    # Carrega dados para memória RAM para ser rápido (se tiver RAM suficiente)
    # Se der erro de memória, avise que mudamos para leitura sob demanda
    print("Carregando dados brutos para memória...")
    data, times = raw.get_data(return_times=True)
    
    X_list = []
    
    print(f"Processando {len(windows)} janelas com Wavelet '{wavelet}'...")
    
    for start, end in tqdm(windows, desc="DWT Feature Extraction"):
        # 1. Recorte (Slicing)
        # Shape: (n_channels, n_samples_na_janela)
        segment = data[:, start:end]
        
        # 2. Transformada Wavelet Discreta (DWT)
        # Aplicamos ao longo do eixo do tempo (axis=-1)
        # coeffs é uma lista: [cA_n, cD_n, cD_n-1, ..., cD_1]
        try:
            coeffs = pywt.wavedec(segment, wavelet, level=level, axis=-1)
        except ValueError as e:
            # Se a janela for muito pequena para o nível, reduz o nível
            # Isso evita crash em janelas finais quebradas
            print(f"[WARN] Erro na Wavelet: {e}. Ignorando janela.")
            continue

        # 3. Feature Engineering (Truque para Deep Learning)
        # Pegamos a Aproximação (cA - frequências baixas importantes)
        # e o Detalhe mais "forte" (cD - frequências altas/ruído/transientes)
        cA = coeffs[0] 
        cD = coeffs[1] 
        
        # Concatenamos no eixo do tempo. 
        # Isso reduz o tamanho original do sinal mas mantem a "assinatura"
        features = np.concatenate([cA, cD], axis=-1)
        
        # 4. Transposição para LSTM
        # LSTM espera: (Samples, TimeSteps, Features/Channels)
        # Nosso 'segment' era (Channels, Time), então features é (Channels, NewTime)
        # Precisamos transpor para (NewTime, Channels)
        X_list.append(features.T)
        
    return np.array(X_list)