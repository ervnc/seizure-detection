import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization

def build_cnn_lstm_model(input_shape):
    """
    Constrói o modelo híbrido conforme o paper (adaptado para funcionar de primeira).
    
    Args:
        input_shape: Tupla (TimeSteps, Features). 
                     Ex: Se sua wavelet reduziu para 60 pontos e tem 23 canais -> (60, 23)
    """
    model = Sequential()

    # --- BLOCO 1: CNN (Extração de Features Espaciais/Locais) ---
    # O paper usa CNN 1D para processar a sequência temporal extraída das wavelets
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(BatchNormalization()) # Ajuda a convergir mais rápido (essencial pra prazo curto)
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3)) # Evita decorar os dados (Overfitting)

    # --- BLOCO 2: LSTM (Dependência Temporal) ---
    # Entende a evolução da crise ao longo do tempo
    model.add(LSTM(64, return_sequences=False)) # False = queremos apenas uma decisão final
    model.add(Dropout(0.3))

    # --- BLOCO 3: Classificador ---
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid')) # Saída binária: 0 (Normal) ou 1 (Crise)

    # Compilação
    model.compile(
        optimizer='adam', 
        loss='binary_crossentropy', 
        metrics=['accuracy']
    )
    
    return model


if __name__ == "__main__":
    # Teste rápido de sanidade para ver se o modelo monta sem erros
    # Simula um input de (50 passos de tempo, 23 canais)
    model = build_cnn_lstm_model((50, 23))
    model.summary()
    print("Modelo construído com sucesso!")