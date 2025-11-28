import mne
import numpy as np
from joblib import load

from helpers.chbmit_helpers import make_windows
from helpers.features_chbmit import extract_features_all


def preprocess_raw_generic(raw,
                          l_freq=0.5,
                          h_freq=45.0,
                          resample_hz=256.0):
  """
  Aplica o mesmo pré-processamento que foi usado nos dados do CHB-MIT:
  - filtro passa-faixa
  - reamostragem
  - seleção de canais EEG
  """
  raw = raw.copy()

  if l_freq is not None or h_freq is not None:
    raw.filter(l_freq=l_freq, h_freq=h_freq, fir_design="firwin", verbose=False)

  if resample_hz is not None:
    raw.resample(resample_hz, npad="auto")

  raw.pick_types(eeg=True)
  return raw


def load_edf_local(path,
                  l_freq=0.5,
                  h_freq=45.0,
                  resample_hz=256.0):
  """
  Lê um EDF local e aplica o pré-processamento padrão.
  """
  raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
  raw = preprocess_raw_generic(raw, l_freq=l_freq, h_freq=h_freq, resample_hz=resample_hz)
  return raw


def detect_seizures_raw(raw,
                      model,
                      window_s=2.0,
                      step_s=0.5,
                      threshold=0.5):
  """
  Dado um Raw (já pré-processado) e um modelo,
  retorna:
    windows: (n_windows, 2) indices em amostras
    proba:  (n_windows,) probabilidade de crise
    pred:   (n_windows,) 0/1 para cada janela
  """
  sf = float(raw.info["sfreq"])
  windows = make_windows(raw.n_times, sf, window_s=window_s, step_s=step_s)
  X = extract_features_all(raw, windows)
  proba = model.predict_proba(X)[:, 1]
  pred = (proba >= threshold).astype(int)
  return windows, proba, pred


def load_model(model_path):
  """
  Carrega um modelo salvo (.joblib).
  """
  return load(model_path)
