import os
import numpy as np
import pandas as pd

from infer.infer_seizures import (
  load_edf_local,
  load_model,
  detect_seizures_raw,
)

# caminhos
MODEL_PATH = "models/rf_chb01.joblib"   # modelo treinado
EDF_PATH = "/home/ervnc/Downloads/chb03_02.edf"  # TODO: troque aqui
OUTPUT_CSV = "inference_output.csv"

WINDOW_S = 2.0
STEP_S = 0.5
THRESHOLD = 0.5  # limiar de decisão: prob >= 0.5 => crise


if __name__ == "__main__":
  assert os.path.exists(MODEL_PATH), f"Modelo não encontrado em {MODEL_PATH}"
  assert os.path.exists(EDF_PATH), f"EDF não encontrado em {EDF_PATH}"

  print(f"Lendo EDF: {EDF_PATH}")
  raw = load_edf_local(EDF_PATH)

  raw.pick_types(eeg=True)
  if len(raw.ch_names) > 23:
    raw.pick_channels(raw.ch_names[:23])
  print("N canais usados na inferência:", len(raw.ch_names))

  print(f"Carregando modelo: {MODEL_PATH}")
  model = load_model(MODEL_PATH)

  print("Rodando detecção de crises...")
  windows, proba, pred = detect_seizures_raw(
    raw,
    model,
    window_s=WINDOW_S,
    step_s=STEP_S,
    threshold=THRESHOLD,
  )

  sf = float(raw.info["sfreq"])
  times_start = windows[:, 0] / sf
  times_end = windows[:, 1] / sf

  # resumo simples
  n_pos = int(pred.sum())
  print(f"\nTotal de janelas: {len(pred)}")
  print(f"Janelas positivas (prováveis crises): {n_pos}")

  # salvar em CSV para você analisar depois (ou plotar em notebook)
  df = pd.DataFrame({
    "t_start_s": times_start,
    "t_end_s": times_end,
    "prob_seizure": proba,
    "pred_seizure": pred,
  })
  df.to_csv(OUTPUT_CSV, index=False)
  print(f"\n[SAÍDA] salva em {OUTPUT_CSV}")

  # mostrar alguns intervalos detectados
  print("\nAlguns intervalos com alta probabilidade (> 0.8):")
  mask = proba >= 0.8
  for ts, te, p in zip(times_start[mask][:10], times_end[mask][:10], proba[mask][:10]):
    print(f"  {ts:8.1f}–{te:8.1f} s | prob={p:.3f}")
