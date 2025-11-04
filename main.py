from dataset.drive_connection import auth_drive
from utils.gdrive_edf import read_raw_edf_from_drive, find_child_folder_id, find_file_id_by_name
from utils.chbmit_labels import get_seizure_intervals_for_edf, find_file_by_name
import mne

ROOT_FOLDER_ID = "1nJm3E6XnYVVFz2itBBdC-qtSab6GZmLO"
PATIENT = "chb01"
EDF_NAME = "chb01_03.edf"

service = auth_drive()

patient_id = find_child_folder_id(service, ROOT_FOLDER_ID, PATIENT)
edf_meta = find_file_by_name(service, patient_id, EDF_NAME)
if not edf_meta:
    raise SystemExit(f"{EDF_NAME} não encontrado em {PATIENT}.")
raw = read_raw_edf_from_drive(service, edf_meta["id"])
print(raw)

intervals = get_seizure_intervals_for_edf(service, patient_id, PATIENT, EDF_NAME)
print(f"Intervalos de crise para {EDF_NAME}: {intervals}")

if intervals:
    onsets   = [a for a, b in intervals]
    durations= [b - a for a, b in intervals]
    ann = mne.Annotations(onset=onsets, duration=durations, description=["seizure"]*len(intervals))
    raw.set_annotations(ann)
    print(raw.annotations)  # deve listar as anotações com onsets/durações
else:
    print("⚠ Nenhuma crise anotada nesse arquivo.")
