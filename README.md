# Detecção de Crises Epilépticas em EEG usando Deep Learning Híbrido (CNN-LSTM) e Transformada Wavelet

Este projeto implementa um sistema de detecção automática de crises epilépticas (seizures) em sinais de Eletroencefalograma (EEG) do dataset **CHB-MIT Scalp EEG Database**. A arquitetura combina Processamento Digital de Sinais (DSP) avançado com Deep Learning para capturar tanto características espectrais quanto dependências temporais.

> **Status:** Concluído (Acurácia: ~99% | Precision: 0.98 | Recall: 0.97 | F1-Score: 0.97)

## 📋 Visão Geral

O sistema processa sinais EEG de múltiplos pacientes do dataset CHB-MIT, extrai características através da Transformada Wavelet Discreta e utiliza uma arquitetura híbrida CNN-LSTM para classificação binária (Normal vs. Crise). O pipeline completo inclui pré-processamento robusto, balanceamento de classes e validação estratificada.

## 🔬 Metodologia

O pipeline de processamento foi projetado para lidar com a natureza não-estacionária dos sinais de EEG e o desequilíbrio natural entre períodos normais e períodos de crise.

### 1. Dataset e Pré-processamento

* **Dataset:** CHB-MIT Scalp EEG Database
  * **Pacientes processados:** chb01 até chb24 (todos os pacientes disponíveis)
  * **Formato:** Arquivos EDF (European Data Format) com 23 canais EEG
  * **Frequência de amostragem:** 256 Hz (após resampling)
  * **Estratégia de seleção:** Prioriza arquivos com crises documentadas + amostra de arquivos normais para contexto

* **Filtragem:**
  * Filtro passa-banda: 0.5 - 45 Hz (remove drift de linha de base e ruído de alta frequência)
  * Design FIR (Finite Impulse Response)
  * Seleção automática de canais EEG (remove ECG e outros sinais)

* **Janelamento (Windowing):**
  * Tamanho da janela: 2.0 segundos
  * Passo (step): 0.5 segundos (sobreposição de 75%)
  * Cada janela resulta em 512 amostras por canal (256 Hz × 2s)

### 2. Extração de Características (Wavelet)

* **Técnica:** Transformada Wavelet Discreta (DWT)
* **Wavelet Mãe:** Daubechies 4 (`db4`)
* **Nível de decomposição:** 4 níveis
* **Coeficientes utilizados:** 
  * Coeficiente de aproximação (cA) do nível mais profundo
  * Coeficiente de detalhe (cD) do primeiro nível
* **Redução de dimensionalidade:** De 512 pontos temporais para ~76 timesteps, mantendo informação espectral essencial
* **Formato de saída:** Tensor 3D `(N_janelas, T_reduzido, N_canais)`

### 3. Normalização e Balanceamento

* **Normalização:** `RobustScaler` (scikit-learn)
  * Utiliza mediana e IQR em vez de média e desvio padrão
  * Mais robusto a outliers comuns em sinais EEG
  * Aplicado após extração de características
  * Scaler é salvo para uso em predição (`scaler_treinado.pkl`)

* **Balanceamento de Classes:**
  * **Undersampling:** Proporção 3:1 (normal:crise)
  * **Class weights:** Pesos balanceados calculados automaticamente
  * **Divisão estratificada:** 80% treino / 20% teste (mantém proporção de classes)

### 4. Arquitetura da Rede Neural (Híbrida CNN-LSTM)

O modelo `CNN-LSTM` foi construído para aprender padrões espaciais e temporais simultaneamente:

```
Input Layer: (Timesteps, Channels) - Features Wavelet
    ↓
Conv1D: 64 filtros, kernel_size=3, activation='relu'
    ↓
BatchNormalization
    ↓
MaxPooling1D: pool_size=2
    ↓
Dropout: 0.3
    ↓
LSTM: 64 unidades, return_sequences=False
    ↓
Dropout: 0.3
    ↓
Dense: 32 neurônios, activation='relu'
    ↓
Output: 1 neurônio, activation='sigmoid' (Classificação Binária)
```

**Hiperparâmetros:**
* **Otimizador:** Adam (learning_rate=0.0001)
* **Loss:** binary_crossentropy
* **Batch Size:** 16
* **Épocas Máximas:** 50
* **Early Stopping:** Monitora `val_loss` com paciência de 10 épocas
* **Métricas:** Accuracy

## 📊 Resultados

O modelo foi treinado utilizando estratégias robustas de balanceamento e regularização, atingindo convergência estável sem overfitting.

| Métrica | Performance (Conjunto de Teste) |
| :--- | :--- |
| **Acurácia Global** | **~99%** |
| **Precision (Crise)** | **0.98** |
| **Recall (Sensibilidade)** | **0.97** |
| **F1-Score** | **0.97** |

*Os resultados demonstram que o modelo é capaz de distinguir crises de atividade normal com extrema precisão, sem apresentar viés para a classe majoritária. A alta precisão minimiza falsos alarmes, enquanto o alto recall garante que a grande maioria das crises seja detectada.*

## 📂 Estrutura do Projeto

```
seizure-detection/
├── models/
│   ├── __init__.py
│   └── hybrid_model.py          # Arquitetura CNN-LSTM (Keras/TensorFlow)
├── processors/
│   └── wavelet.py               # Extração de features com PyWavelets (DWT)
├── readers/
│   └── chbmit_reader.py         # Leitura, parsing e janelamento de arquivos .EDF
├── helpers/
│   └── chbmit_helpers.py         # Funções auxiliares (parsing, janelamento, rótulos)
├── utils/
│   └── drive_utils.py            # Utilitários de conexão com Google Drive
├── edfs/                         # Arquivos EDF locais (opcional)
├── train.py                      # Script principal de treinamento e avaliação
├── predict.py                    # Script para predição em novos arquivos EDF
├── test.py                       # Script de teste e validação
├── drive_connection.py           # Autenticação OAuth2 para Google Drive
├── modelo_final_epilepsia.keras  # Modelo treinado (gerado após train.py)
├── scaler_treinado.pkl           # Scaler treinado (gerado após train.py)
├── resultado_treino.png          # Gráficos de convergência (gerado após train.py)
├── token.json                    # Token de autenticação (gerado automaticamente)
├── credentials.json              # Credenciais OAuth2 (configuração manual)
└── README.md                     # Este arquivo
```

## 🚀 Como Executar

### Pré-requisitos

* Python 3.9+
* Conta no Google Cloud Platform (para acesso à API do Drive)
* Acesso ao dataset CHB-MIT (via Google Drive ou localmente)

### Instalação

```bash
# Clone o repositório (se aplicável)
cd seizure-detection

# Instale as dependências
pip install -r requirements.txt
```

**Dependências principais:**
* `tensorflow` (ou `tensorflow-gpu`)
* `mne` (MNE-Python para processamento de sinais EEG)
* `numpy`
* `scipy`
* `PyWavelets` (para Transformada Wavelet)
* `scikit-learn` (pré-processamento e métricas)
* `matplotlib` (visualização)
* `google-api-python-client` (acesso ao Google Drive)
* `google-auth-httplib2` e `google-auth-oauthlib` (autenticação OAuth2)
* `joblib` (serialização do scaler)

### Configuração

1. **Credenciais do Google Drive:**
   * Acesse o [Google Cloud Console](https://console.cloud.google.com/)
   * Crie um projeto ou selecione um existente
   * Ative a API do Google Drive
   * Crie credenciais OAuth 2.0 (tipo: Desktop app)
   * Baixe o arquivo JSON e renomeie para `credentials.json`
   * Coloque `credentials.json` na raiz do projeto

2. **Configuração do Dataset:**
   * O script `train.py` está configurado para acessar o dataset via Google Drive
   * O `FOLDER_ID` no código aponta para a pasta do dataset no Drive
   * Alternativamente, você pode colocar arquivos EDF na pasta `edfs/` e modificar o código para leitura local

### Treinamento

Para iniciar o pipeline completo (Download → Processamento → Treino → Avaliação):

```bash
python train.py
```

O script irá:

1. **Autenticar** com o Google Drive (primeira execução solicitará autorização)
2. **Processar múltiplos pacientes** (chb01 até chb24, conforme disponibilidade)
3. **Extrair características** usando Transformada Wavelet
4. **Balancear e normalizar** os dados
5. **Treinar o modelo** com early stopping
6. **Gerar gráficos** de convergência (`resultado_treino.png`)
7. **Salvar o modelo** treinado (`modelo_final_epilepsia.keras`)
8. **Salvar o scaler** (`scaler_treinado.pkl`)
9. **Avaliar** no conjunto de teste e exibir métricas

**Saída esperada:**
* `modelo_final_epilepsia.keras` - Modelo treinado
* `scaler_treinado.pkl` - Scaler para normalização
* `resultado_treino.png` - Gráficos de acurácia e loss
* Relatório de classificação no terminal (Precision, Recall, F1-Score, Matriz de Confusão)

### Predição em Novos Arquivos

Para fazer predição em um novo arquivo EDF:

```bash
python predict.py caminho/para/arquivo.edf
```

O script irá:

1. Carregar o modelo e scaler treinados
2. Processar o arquivo EDF (filtragem, resampling, janelamento)
3. Extrair características via Wavelet
4. Normalizar usando o scaler treinado
5. Fazer predições janela por janela
6. Aplicar filtro de confiança e janelas consecutivas
7. Reportar eventos de crise detectados com timestamps

**Parâmetros de detecção:**
* `THRESHOLD_CONFIDENCE`: 0.85 (85% de confiança mínima)
* `MIN_CONSECUTIVE_WINDOWS`: 15 janelas (~7-8 segundos contínuos)

## 🛠️ Tecnologias Utilizadas

* **Linguagem:** Python 3.9+
* **Deep Learning:** TensorFlow / Keras
* **Processamento de Sinais:** PyWavelets, MNE-Python
* **Manipulação de Dados:** NumPy, SciPy
* **Machine Learning:** scikit-learn
* **Visualização:** Matplotlib
* **Cloud Storage:** Google Drive API

## 📝 Notas Técnicas

### Por que RobustScaler?

Sinais EEG frequentemente contêm outliers devido a:
* Artefatos de movimento
* Interferência elétrica
* Ruído de linha de base

O `RobustScaler` utiliza mediana e IQR, sendo mais resistente a outliers que o `StandardScaler` (que usa média e desvio padrão).

### Por que Wavelet em vez de FFT/STFT?

A Transformada Wavelet oferece:
* **Análise multi-resolução:** Captura tanto baixas quanto altas frequências
* **Melhor localização tempo-frequência:** Essencial para detectar transientes (início súbito de crises)
* **Redução de dimensionalidade:** Mantém informação espectral com menos dados

### Por que CNN-LSTM?

* **CNN:** Captura padrões espaciais e locais nos sinais (relações entre canais e frequências)
* **LSTM:** Modela dependências temporais de longo prazo (evolução da crise ao longo do tempo)
* **Híbrido:** Combina as vantagens de ambas as arquiteturas

## 🔍 Validação e Testes

O script `test.py` pode ser usado para validar o pipeline de processamento em arquivos específicos:

```bash
python test.py
```

Este script testa a leitura de arquivos, janelamento, extração de características e verifica a consistência dos dados.

## 📄 Licença

Este projeto é parte de um trabalho acadêmico do curso de Introdução à Ciência de Dados (SSC0275) - ICMC/USP.

## 👥 Autores

* **Evandro Risso**
* **João Calisto**

**Curso:** Bacharelado em Física Computacional - IFSC/USP

## 📚 Referências

* **Dataset:** [CHB-MIT Scalp EEG Database](https://physionet.org/content/chbmit/1.0.0/)
* **PhysioNet:** Goldberger et al. (2000). "PhysioBank, PhysioToolkit, and PhysioNet: Components of a New Research Resource for Complex Physiologic Signals"

---

**Última atualização:** Novembro 2024
