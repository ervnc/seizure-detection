from drive_connection import auth_drive
from helpers.chbmit_helpers import list_patient_edfs, get_patient_folder_id, get_intervals_from_drive
from readers.chbmit_reader import build_windows_and_labels

FOLDER_ID = "1nJm3E6XnYVVFz2itBBdC-qtSab6GZmLO" # ID da pasta '1.0.0' no Google Drive
PATIENT = "chb01"

if __name__ == "__main__":
  service = auth_drive()
  patient_id = get_patient_folder_id(service, FOLDER_ID, PATIENT)
  assert patient_id, f"Paciente {PATIENT} não encontrado em {FOLDER_ID}"

  edfs = list_patient_edfs(service, patient_id)
  assert edfs, f"Nenhum EDF na pasta {PATIENT}"

  edf_with = next((r for r in edfs if r["has_seizures_file"]), edfs[0])
  edf_without = next((r for r in edfs if not r["has_seizures_file"]), edfs[min(1, len(edfs)-1)])

  for edf_row in [edf_with, edf_without]:
    print(f"\n>> Testando {edf_row['name']}  | has_seizures_file={edf_row['has_seizures_file']}")
    raw, windows, y = build_windows_and_labels(
      service, FOLDER_ID, PATIENT, edf_row["name"],
      window_s=2.0, step_s=0.5, prediction_horizon_s=0.0
    )
    sf = float(raw.info["sfreq"])

    patient_id = get_patient_folder_id(service, FOLDER_ID, PATIENT)
    intervals = get_intervals_from_drive(service, patient_id, edf_row["name"])
    print("intervalos de crise:", intervals)

    print(f"sfreq={sf} Hz | windows={windows.shape[0]} | positivos={int(y.sum())}")

    