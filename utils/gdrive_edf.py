import tempfile, io, os
import mne

from dataset.drive_connection import auth_drive
from googleapiclient.http import MediaIoBaseDownload
from googleapiclient.discovery import Resource

def find_child_folder_id(service: Resource, parent_folder_id: str, child_name: str) -> str | None:
    q = (
        f"'{parent_folder_id}' in parents and "
        f"name = '{child_name}' and "
        "mimeType = 'application/vnd.google-apps.folder' and trashed = false"
    )
    res = service.files().list(
        q=q, fields="files(id,name)", pageSize=1,
        supportsAllDrives=True, includeItemsFromAllDrives=True
    ).execute()
    files = res.get("files", [])
    return files[0]["id"] if files else None

def find_file_id_by_name(service: Resource, parent_folder_id: str, filename: str) -> str | None:
    q = (
        f"'{parent_folder_id}' in parents and "
        f"name = '{filename}' and trashed = false"
    )
    res = service.files().list(
        q=q, fields="files(id,name)", pageSize=1,
        supportsAllDrives=True, includeItemsFromAllDrives=True
    ).execute()
    files = res.get("files", [])
    return files[0]["id"] if files else None

def download_text_from_drive(service, file_id) -> str:
    """Baixa conteúdo textual, funcionando para arquivo normal e Google Docs."""
    meta = service.files().get(
        fileId=file_id,
        fields="id,name,mimeType,shortcutDetails",
        supportsAllDrives=True
    ).execute()

    # Se for atalho, resolve para o alvo real
    if meta.get("shortcutDetails"):
        target_id = meta["shortcutDetails"]["targetId"]
        return download_text_from_drive(service, target_id)

    mime = meta.get("mimeType", "")
    # Google Docs → exporta como texto
    if mime.startswith("application/vnd.google-apps."):
        request = service.files().export_media(fileId=file_id, mimeType="text/plain")
    else:
        request = service.files().get_media(fileId=file_id, supportsAllDrives=True)

    buf = io.BytesIO()
    dl = MediaIoBaseDownload(buf, request)
    done = False
    while not done:
        _, done = dl.next_chunk()
    buf.seek(0)
    return buf.read().decode("utf-8", errors="ignore")


def read_raw_edf_from_drive(service, file_id, *, preload=True):
    with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as tmp:
        tmp_path = tmp.name
        request = service.files().get_media(fileId=file_id, supportsAllDrives=True)
        downloader = MediaIoBaseDownload(io.FileIO(tmp_path, "wb"), request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
            if status:  # evita None na última iteração
                print(f"Download {int(status.progress()*100)}%.")

    try:
        raw = mne.io.read_raw_edf(tmp_path, preload=preload, verbose=False)
    finally:
        try: os.remove(tmp_path)
        except OSError: pass
    return raw

if __name__ == "__main__":
    service = auth_drive()
    file_id = "1xhgG35prvEyf9YvB0h3OnX3amzXt7utX"
    raw = read_raw_edf_from_drive(service, file_id)
    print(raw)