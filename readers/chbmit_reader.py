import mne
import os
import tempfile
from typing import Tuple
import numpy as np

from utils.drive_utils import stream_file_bytes
from helpers.chbmit_helpers import get_patient_folder_id, list_patient_edfs, make_windows, get_intervals_from_drive, label_windows

# ==============================================================================
# Faz download em memória, grava em um arquivo temporário .edf
# e passa o caminho para o MNE (compatível com versões que não aceitam BytesIO).
# ==============================================================================
def read_edf_from_drive(service, edf_file_id: str,
                        l_freq: float = 0.5, h_freq: float = 45.0,
                        resample_hz: float | None = 256.0) -> mne.io.BaseRaw:
    data_bytes = stream_file_bytes(service, edf_file_id)

    # cria arquivo temporário
    with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as tmp:
        tmp_path = tmp.name
        tmp.write(data_bytes)

    try:
        raw = mne.io.read_raw_edf(tmp_path, preload=True, verbose=False)
    finally:
        # remove o arquivo temporário depois de carregar
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    if l_freq is not None or h_freq is not None:
        raw.filter(l_freq=l_freq, h_freq=h_freq, fir_design="firwin", verbose=False)
    if resample_hz is not None:
        raw.resample(resample_hz, npad="auto")
    raw.pick_types(eeg=True)
    return raw


# ==============================================================
# Constrói janelas e rótulos a partir do EDF e arquivos de anota
# ==============================================================
def build_windows_and_labels(service, root_folder_id: str, patient: str, edf_name: str, window_s: float = 2.0, step_s = 0.5, prediction_horizon_s: float = 0.0) -> Tuple[mne.io.BaseRaw, np.ndarray, np.ndarray]:
  patient_id = get_patient_folder_id(service, root_folder_id, patient)
  assert patient_id, f"Paciente {patient} não encontrado em {root_folder_id}"

  # Encontra o arquivo
  edfs = list_patient_edfs(service, patient_id)
  hit = next((r for r in edfs if r["name"] == edf_name), None)
  assert hit, f"EDF {edf_name} não encontrado para o paciente {patient}"

  # Leitura + janelas + rótulos
  raw = read_edf_from_drive(service, hit["id"])
  sf = float(raw.info["sfreq"])
  windows = make_windows(raw.n_times, sf, window_s=window_s, step_s=step_s)
  intervals = get_intervals_from_drive(service, patient_id, edf_name)
  y = label_windows(windows, sf, intervals, prediction_horizon_s=prediction_horizon_s)
  return raw, windows, y