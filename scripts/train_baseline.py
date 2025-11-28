import json
from joblib import dump
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from drive_connection import auth_drive
from helpers.chbmit_helpers import get_patient_folder_id, list_patient_edfs
from readers.chbmit_reader import build_windows_and_labels
from helpers.features_chbmit import extract_features_all

# ===========================
# Configurações
# ===========================
FOLDER_ID = "1nJm3E6XnYVVFz2itBBdC-qtSab6GZmLO"
PATIENT = "chb01"
CACHE_DIR = "cache"

os.makedirs(CACHE_DIR, exist_ok=True)

# ===========================
# Conexão e seleção de EDFs
# ===========================
service = auth_drive()
patient_id = get_patient_folder_id(service, FOLDER_ID, PATIENT)
assert patient_id, f"Paciente {PATIENT} não encontrado dentro de {FOLDER_ID}"

edfs = list_patient_edfs(service, patient_id)
assert edfs, f"Nenhum EDF encontrado para {PATIENT}"

all_X = []
all_y = []

for row in edfs:
  edf_name = row["name"]
  cache_path = os.path.join(
    CACHE_DIR,
    f"{PATIENT}_{edf_name.replace('.edf', '')}.npz"
  )

  if os.path.exists(cache_path):
    print(f"[CACHE] Carregando {edf_name} de {cache_path}")
    data = np.load(cache_path)
    X_edf = data["X"]
    y_edf = data["y"]
  else:
    print(f"[PROCESSANDO] {edf_name}")
    raw, windows, y = build_windows_and_labels(
      service, FOLDER_ID, PATIENT, edf_name,
      window_s=2.0, step_s=0.5, prediction_horizon_s=0.0
    )

    X_edf = extract_features_all(raw, windows)
    y_edf = y.astype(np.int64)

    np.savez_compressed(cache_path, X=X_edf, y=y_edf)
    print(f"[CACHE] Salvo em {cache_path}")

  all_X.append(X_edf)
  all_y.append(y_edf)

X = np.vstack(all_X)
y = np.hstack(all_y)

print("Shape final X:", X.shape, "| y:", y.shape)
print("Proporção de positivos:", y.mean())

# ===========================
# Train / Test split
# ===========================
X_train, X_test, y_train, y_test = train_test_split(
  X, y,
  test_size=0.25,
  stratify=y,
  random_state=42
)

# ===========================
# Modelo baseline
# ===========================
clf = RandomForestClassifier(
  n_estimators=200,
  max_depth=None,
  class_weight="balanced",
  n_jobs=-1,
  random_state=42,
)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("\nMatriz de confusão:")
print(confusion_matrix(y_test, y_pred))

print("\nRelatório de classificação:")
print(classification_report(y_test, y_pred, digits=3))

# ==========================
# Salvando o modelo e config
# ==========================
os.makedirs("models", exist_ok=True)
model_path = os.path.join("models", f"rf_{PATIENT}.joblib")
dump(clf, model_path)
print(f"\n[MODELO] salvo em {model_path}")

config = {
  "patient": PATIENT,
  "folder_id": FOLDER_ID,
  "windows_s": 2.0,
  "step_s": 0.5,
  "sfreq": float(X.shape[1]),
  "note": "Modelo RandomForest baseline, features_chbmit.extract_features_all"
}

with open(os.path.join("models", f"config_{PATIENT}.json"), "w") as f:
  json.dump(config, f, indent=2)
print("[CONFIG] salva em models/config_*.json")