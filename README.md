# Detecção de Crises Epilépticas em EEG usando Deep Learning Híbrido (CNN-LSTM) e Transformada Wavelet

Este projeto implementa um sistema de detecção automática de crises epilépticas (seizures) em sinais de Eletroencefalograma (EEG) do dataset **CHB-MIT**. A arquitetura combina Processamento Digital de Sinais (DSP) avançado com Deep Learning para capturar tanto características espectrais quanto dependências temporais.

> **Status:** Concluído (Acurácia: \~99% | Recall de Crise: \>95%)

## 📋 Metodologia

O pipeline de processamento foi projetado para lidar com a natureza não-estacionária dos sinais de EEG.

### 1\. Pré-processamento e Engenharia de Features

  * **Dataset:** CHB-MIT Scalp EEG Database (focado no Paciente `chb01`).
  * **Janelamento (Windowing):** Janelas de 2 segundos com sobreposição (overlap) de 0.5 segundos.
  * **Extração de Features (Wavelet):**
      * Utilização da **Transformada Wavelet Discreta (DWT)**.
      * **Wavelet Mãe:** Daubechies 4 (`db4`).
      * **Decomposição:** O sinal é decomposto para extrair coeficientes de aproximação e detalhe, reduzindo a dimensionalidade de 512 pontos para **76 features** temporais, mantendo a informação de frequência essencial.
  * **Normalização (Crítico):** Aplicação de `StandardScaler` (Z-Score) para normalizar as features (`mean=0`, `std=1`), essencial para a convergência da rede LSTM.

### 2\. Arquitetura da Rede Neural (Híbrida)

O modelo `CNN-LSTM` foi construído para aprender padrões espaciais e temporais simultaneamente:

1.  **Input Layer:** Recebe as features Wavelet `(Timesteps, Channels)`.
2.  **Conv1D + BatchNormalization:** Extrai padrões locais e frequências específicas (filtros espaciais).
3.  **MaxPooling1D:** Reduz a dimensionalidade e foca nas ativações mais fortes.
4.  **LSTM (Long Short-Term Memory):** Captura a evolução temporal da crise (dependência de longo prazo).
5.  **Dense + Dropout:** Camadas totalmente conectadas com regularização para evitar overfitting.
6.  **Output:** Ativação Sigmoid (Classificação Binária: Normal vs. Crise).

## 📊 Resultados

O modelo foi treinado utilizando uma estratégia agressiva de balanceamento de dados (Undersampling 1:1) e pesos de classe (`class_weights`), atingindo convergência estável.

| Métrica | Performance (Conjunto de Teste) |
| :--- | :--- |
| **Acurácia Global** | **\~99%** |
| **Precision (Crise)** | **\~0.98** |
| **Recall (Sensibilidade)** | **\~0.97** |
| **F1-Score** | **0.97** |

*Os resultados demonstram que o modelo é capaz de distinguir crises de atividade normal com extrema precisão, sem apresentar viés para a classe majoritária.*

## 📂 Estrutura do Projeto

```text
seizure-detection/
├── models/
│   └── hybrid_model.py       # Arquitetura CNN-LSTM (Keras/TensorFlow)
├── processors/
│   └── wavelet.py            # Extração de features com PyWavelets (DWT)
├── readers/
│   └── chbmit_reader.py      # Leitura, parsing e janelamento de arquivos .EDF
├── helpers/                  # Funções auxiliares de string/regex
├── utils/                    # Utilitários de conexão com Google Drive
├── train.py                  # Script principal de treinamento e avaliação
├── drive_connection.py       # Autenticação OAuth2
├── requirements.txt          # Dependências
└── README.md                 # Documentação
```

## 🚀 Como Executar

### Pré-requisitos

  * Python 3.9+
  * Conta no Google Cloud Platform (para acesso à API do Drive, se necessário) ou arquivos locais.

### Instalação

```bash
pip install -r requirements.txt
```

*(Certifique-se de instalar: `tensorflow`, `mne`, `numpy`, `scipy`, `PyWavelets`, `scikit-learn`, `matplotlib`)*

### Configuração

1.  Gere o arquivo `credentials.json` no Google Cloud Console (OAuth 2.0 Client ID).
2.  Coloque-o na raiz do projeto.

### Treinamento

Para iniciar o pipeline completo (Download -\> Processamento -\> Treino -\> Avaliação):

```bash
python train.py
```

O script irá:

1.  Processar os arquivos `.edf` do paciente configurado.
2.  Gerar os gráficos de convergência (`resultado_treino.png`).
3.  Salvar o modelo treinado em `modelo_final_epilepsia.keras`.

## 🛠️ Tecnologias Utilizadas

  * **Linguagem:** Python
  * **Deep Learning:** TensorFlow / Keras
  * **Processamento de Sinais:** PyWavelets, MNE-Python
  * **Manipulação de Dados:** NumPy, Scikit-Learn

-----

**Autor:** Evandro Risso e João Calisto  
**Curso:** Bacharelado em Física Computacional - IFSC/USP