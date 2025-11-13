from typing import List, Dict, Optional
import io
from googleapiclient.http import MediaIoBaseDownload

# ===============================================================================
# Lista todos os arquivos e pastas dentro de uma pasta específica no Google Drive
# ===============================================================================
def list_children(service, folder_id: str, q_extra: str = ""):
  q = f"'{folder_id}' in parents and trashed = false"
  if q_extra:
      q += f" and {q_extra}"
  try:
    res = service.files().list(
      q=q,
      fields="nextPageToken, files(id,name,mimeType,parents)"
    ).execute()
  except Exception as e:
    print(f"[ERRO list_children] Falha ao listar filhos de {folder_id}: {e}")
    return []

  items = res.get("files", [])
  while res.get("nextPageToken"):
    try:
      res = service.files().list(
        q=q,
        pageToken=res["nextPageToken"],
        fields="nextPageToken, files(id,name,mimeType,parents)"
      ).execute()
      items.extend(res.get("files", []))
    except Exception as e:
      print(f"[ERRO paginação] {e}")
      break
  return items

# =====================================================================================
# Encontra um arquivo ou pasta pelo nome dentro de uma pasta específica no Google Drive
# ====================================================================================
def find_by_name_in_folder(service, folder_id: str, name: str) -> Optional[Dict]:
  q = f"'{folder_id}' in parents and name = '{name}' and trashed = false"
  res = service.files().list(q=q, fields="files(id, name, mimeType, parents)").execute()
  files = res.get("files", [])
  return files[0] if files else None

# =================================================================
# Faz o download de um arquivo do Google Drive e retorna seus bytes
# =================================================================
def stream_file_bytes(service, file_id: str) -> bytes:
  req = service.files().getMedia(fileId=file_id)
  buf = io.BytesIO()
  down = MediaIoBaseDownload(buf, req)
  done = False
  while not done:
    _, done = down.next_chunk()
  buf.seek(0)
  return buf.read()

# =========================================================================
# Lê um arquivo de texto do Google Drive e retorna seu conteúdo como string
# =========================================================================
def read_text_file(service, file_id: str, encoding="utf-8") -> str:
  data = stream_file_bytes(service, file_id)
  try:
    return data.decode(encoding)
  except Exception:
    return data.decore("latin-1", errors="ignore")