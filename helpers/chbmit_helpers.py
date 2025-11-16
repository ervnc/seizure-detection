from typing import Optional, List, Dict, Tuple
from utils.drive_utils import list_children, find_by_name_in_folder, read_text_file
import re
import numpy as np

FOLDER_MIMETYPE = "application/vnd.google-apps.folder"

# ------------------ Navegação no Drive ------------------ #


# =========================================
# Obtém o ID da pasta do paciente pelo nome
# =========================================
def get_patient_folder_id(service, root_folder_id: str, patient_name: str) -> Optional[str]:
  children = list_children(service, root_folder_id, q_extra=f"name='{patient_name}' and mimeType='{FOLDER_MIMETYPE}'")
  if children:
    return children[0]["id"]
  return None


# =========================================================
# Lista todos os arquivos .edf e seus arquivos de anotações
# =========================================================
def list_patient_edfs(service, patient_folder_id: str) -> List[Dict]:
  items = list_children(service, patient_folder_id)
  edfs = [it for it in items if it.get("name", "").endswith(".edf")]
  seizures = { (it["name"].replace(".seizures", "")): it["id"]
              for it in items if it.get("name", "").endswith(".edf.seizures") }
  
  out = []
  for f in edfs:
    name = f["name"]
    out.append({
      "id": f["id"],
      "name": name,
      "has_seizures_file": name in seizures,
      "seizures_id": seizures.get(name)
    })
  out.sort(key=lambda x: x["name"])
  return out


# ------------------ Parsers de intervalos ------------------ #


# ===============================
# Parser de intervalos de strings
# ===============================
def _merge_intervals(intervals: List[Tuple[float,float]]) -> List[Tuple[float,float]]:
  if not intervals:
    return []
  
  intervals = sorted(intervals)
  merged = [list(intervals[0])]
  for s,e in intervals[1:]:
    if s <= merged[-1][1] + 1e-6:
      merged[-1][1] = max(merged[-1][1], e)
    else:
      merged.append([s,e])
  return [(float(a), float(b)) for a,b in merged]


# ===================================================================
# Parser para arquivos .edf.seizures do CHB-MIT.
# Formatos comuns:
#   - 'Seizure Start Time: 2996 seconds'
#   - 'Seizure End Time: 3036 seconds'
# Fallback: padrões genéricos com 'start' / 'end' e pares de números.
# ===================================================================
def parse_edf_seizures_text(txt: str) -> List[Tuple[float,float]]:
  t = txt.lower()

  # 1) Padrão específico CHB-MIT: "Seizure Start Time: 2996 seconds"
  starts = [float(x) for x in re.findall(r'seizure\s+\d*\s*start time:\s*([0-9]+(?:\.[0-9]+)?)', t)]
  ends   = [float(x) for x in re.findall(r'seizure\s+\d*\s*end time:\s*([0-9]+(?:\.[0-9]+)?)', t)]
  intervals = [(s, e) for s, e in zip(starts, ends) if e > s]

  # 2) Se não achou nada, tenta padrão genérico 'start ... <num>' / 'end ... <num>'
  if not intervals:
    starts = [float(x) for x in re.findall(r'start[^0-9]*([0-9]+(?:\.[0-9]+)?)', t)]
    ends   = [float(x) for x in re.findall(r'end[^0-9]*([0-9]+(?:\.[0-9]+)?)', t)]
    intervals = [(s, e) for s, e in zip(starts, ends) if e > s]

  # 3) Fallback extremo: pares de números por linha
  if not intervals:
    for line in t.splitlines():
      nums = [float(x) for x in re.findall(r'([0-9]+(?:\.[0-9]+)?)', line)]
      if len(nums) >= 2 and nums[1] > nums[0]:
        intervals.append((nums[0], nums[1]))

  return _merge_intervals(intervals)


# ===========================================================
# Converte string 'hh:mm:ss' ou 'mm:ss' ou 'ss' para segundos
# ===========================================================
def _hms_to_seconds(hms: str) -> Optional[float]:
  parts = hms.split(':')
  try:
    parts = list(map(float, parts))
  except Exception:
    return None
  
  if len(parts) == 3:
    h, m, s = parts; return h*3600 + m*60 + s
  if len(parts) == 2:
    m, s = parts; return m*60 + s
  if len(parts) == 1:
    return parts[0]
  return None


# ===============================
# Parser de SUMMARYs de pacientes
# ===============================
def parse_patient_summary_text(txt: str) -> Dict[str, List[Tuple[float,float]]]:
  """
  Retorna dict: { 'chb01_03.edf': [(start_sec, end_sec), ...], ... }
  usando o formato oficial do CHB-MIT, por ex.:

    File Name: chb01_03.edf
    File Start Time: 13:43:04
    File End Time: 14:43:04
    Number of Seizures in File: 1
    Seizure Start Time: 2996 seconds
    Seizure End Time: 3036 seconds
  """

  lines = txt.lower().splitlines()
  current_file = None
  blocks: Dict[str, List[str]] = {}

  for line in lines:
    line = line.strip()
    # casa "file name: chb01_03.edf"
    m = re.search(r'file name:\s*([\w\-.]+\.edf)', line)
    if m:
      current_file = m.group(1)
      blocks.setdefault(current_file, [])
      continue
    if current_file is not None:
      blocks[current_file].append(line)

  out: Dict[str, List[Tuple[float,float]]] = {}
  for fname, flines in blocks.items():
    file_text = "\n".join(flines)
    # pega "Seizure Start Time: 2996 seconds"
    starts = [
      float(x) for x in re.findall(
        r'seizure start time:\s*([0-9]+(?:\.[0-9]+)?)\s*seconds',
        file_text
        )
      ]
    ends = [
      float(x) for x in re.findall(
        r'seizure end time:\s*([0-9]+(?:\.[0-9]+)?)\s*seconds',
        file_text
      )
    ]
    intervals = [(s, e) for s, e in zip(starts, ends) if e > s]
    out[fname] = _merge_intervals(intervals)

  return out


# =========================================================================
# Preferência: <edf_name>.seizures; fallback: SUMMARY da pasta do paciente.
# =========================================================================
def get_intervals_from_drive(service, patient_folder_id: str, edf_name: str) -> List[Tuple[float,float]]:
  intervals: List[Tuple[float,float]] = []

  # 1) tentar .edf.seizures
  seiz = find_by_name_in_folder(service, patient_folder_id, edf_name + ".seizures")
  if seiz:
    try:
      txt = read_text_file(service, seiz["id"])
      intervals = parse_edf_seizures_text(txt)
    except Exception as e:
      print(f"[WARN] erro ao ler {edf_name}.seizures: {e}")

  # se deu certo, já retorna
  if intervals:
    return intervals

  # 2) fallback: SUMMARY (ex.: chb01-summary.txt)
  summaries = list_children(service, patient_folder_id, q_extra="name contains 'summary'")
  if summaries:
    txt = read_text_file(service, summaries[0]["id"])
    mapping = parse_patient_summary_text(txt)
    return mapping.get(edf_name, [])

  return []


# ------------------ Janelação + rótulos ------------------ #


# =====================================================
# Gera janelas deslizantes sobre os dados (início, fim)
# =====================================================
def make_windows(n_samples: int, sfreq: float, window_s: float = 2.0, step_s: float = 0.5) -> np.ndarray:
  w = int(round(window_s * sfreq))
  s = int(round(step_s * sfreq))
  starts = np.arange(0, n_samples - w + 1, s, dtype=int)
  return np.stack([starts, starts + w], axis= 1)


# ====================================================================
# Rotula janelas com base em intervalos de eventos (1 = contém evento)
# ====================================================================
def label_windows(windows: np.ndarray, sfreq: float, intervals: List[Tuple[float,float]], prediction_horizon_s: float = 0.0) -> np.ndarray:
  if not intervals:
    return np.zeros(len(windows), dtype=int)
  
  ints = [(int(round(s*sfreq)), int(round(e*sfreq))) for s,e in intervals]
  horizon = int(round(prediction_horizon_s * sfreq))
  y = np.zeros(len(windows), dtype=int)
  for i, (a,b) in enumerate(windows):
    b_h = b + horizon
    for s,e in ints:
      if not (b_h <= s or a >= e):
        y[i] = 1
        break
  return y