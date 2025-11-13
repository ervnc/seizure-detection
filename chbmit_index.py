from typing import List, Dict
from drive_connection import auth_drive
from utils.drive_utils import list_children

FOLDER_MIME_TYPE = "application/vnd.google-apps.folder"

# ==========================================================================
# Lista todos os arquivos e pastas recursivamente a partir de uma pasta raiz
# ==========================================================================
def list_all_recursive(service, root_folder_id: str) -> List[Dict]:
  """Varre recursivamente a árvore a partir de root_folder_id."""
  stack: List[str] = [root_folder_id]
  all_items: List[Dict] = []  # <-- INICIALIZA AQUI (dentro da função)

  while stack:
    fid = stack.pop()
    try:
      children = list_children(service, fid)
    except Exception as e:
      print(f"[ERRO] Falha ao listar filhos de {fid}: {e}")
      children = []

    for it in children:
      all_items.append(it)
      if it.get("mimeType") == FOLDER_MIME_TYPE:
        stack.append(it["id"])

  return all_items


# ==========================================================
# Constrói o índice dos arquivos .edf e .edf.seizures no CHB
# ==========================================================
def build_chbmit_index(service, root_folder_id: str) -> List[Dict]:
  items = list_all_recursive(service, root_folder_id)
  folders = {it["id"]: it for it in items if it["mimeType"] == FOLDER_MIME_TYPE}
  files = [it for it in items if it["mimeType"] != FOLDER_MIME_TYPE]

  rows = []
  for f in files:
    name = f["name"]
    if not (name.endswith(".edf") or name.endswith(".edf.seizures")):
      continue
    parent_id = f.get("parents", [None])[0]
    patient = None
    if parent_id and parent_id in folders:
      pname = folders[parent_id]["name"]
      if pname.lower().startswith("chb"):
        patient = pname

    kind = "edf" if name.endswith(".edf") else "seizures"
    base = name if kind == "edf" else name.replace(".seizures", "")
    rows.append({
      "patient": patient,
      "name": name,
      "id": f["id"],
      "kind": kind,
      "base": base,
      "parent_id": parent_id
    })

  # Marca se a .edf tem par .seizures na mesma pasta
  edfs = [r for r in rows if r["kind"] == "edf"]
  seiz_map = {(r["parent_id"], r["base"]): True for r in rows if r["kind"] == "seizures"}
  for r in edfs:
    r["has_seizures_file"] = bool(seiz_map.get((r["parent_id"], r["name"]), False))

  edfs.sort(key=lambda r: ((r["patient"] or ""), r["name"]))
  return edfs

if __name__ == "__main__":
  service = auth_drive()
  FOLDER_ID = "1nJm3E6XnYVVFz2itBBdC-qtSab6GZmLO" # ID da pasta '1.0.0' no Google Drive

  index = build_chbmit_index(service, FOLDER_ID)

  # Resumo por paciente
  from collections import defaultdict
  per_patient = defaultdict(lambda: {"total": 0, "with_seizures": 0})
  for r in index:
    per_patient[r["patient"]]["total"] += 1
    per_patient[r["patient"]]["with_seizures"] += int(r["has_seizures_file"])

  print("\nResumo por paciente:")
  for p in sorted(per_patient):
    t = per_patient[p]["total"]
    w = per_patient[p]["with_seizures"]
    print(f"Paciente {p}: {t} arquivos .edf, {w} com arquivo .seizures")

  print("\nAmostras:")
  for r in index[:8]:
    print(f"{r['patient']} | {r['name']} | seizures={r['has_seizures_file']} | id={r['id']}")
