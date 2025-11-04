import re
from typing import List, Tuple
from googleapiclient.discovery import Resource
from .gdrive_edf import download_text_from_drive
from .gdrive_edf import find_file_id_by_name

def find_file_by_name(service, parent_folder_id: str, filename: str):
    res = service.files().list(
        q=f"'{parent_folder_id}' in parents and name = '{filename}' and trashed = false",
        fields="files(id,name,mimeType,shortcutDetails)",
        pageSize=1, supportsAllDrives=True, includeItemsFromAllDrives=True
    ).execute()
    files = res.get("files", [])
    return files[0] if files else None

def _extract_block_for_file(txt: str, edf_name: str) -> str:
    lines = txt.splitlines()
    out, grab = [], False
    for ln in lines:
        low = ln.lower()
        if "file name:" in low:
            grab = (edf_name.lower() in low)
            continue
        if grab and "file name:" in low:
            break
        if grab:
            out.append(ln)
    return "\n".join(out)

def _parse_intervals_from_text(txt: str) -> List[Tuple[float, float]]:
    L = txt.lower()

    # Padrão explícito:
    pairs = re.findall(
        r"seizure start time:\s*([0-9]+(?:\.[0-9]+)?)\s*seconds?.*?"
        r"seizure end time:\s*([0-9]+(?:\.[0-9]+)?)\s*seconds?",
        L, flags=re.S
    )
    ints = [(float(a), float(b)) for a, b in pairs if float(b) > float(a)]
    if ints: return ints

    # Fallbacks
    starts = [float(x) for x in re.findall(r"start[^0-9]+([0-9]+(?:\.[0-9]+)?)", L)]
    ends   = [float(x) for x in re.findall(r"end[^0-9]+([0-9]+(?:\.[0-9]+)?)", L)]
    ints = [(a, b) for a, b in zip(starts, ends) if b > a]
    if ints: return ints

    for ln in L.splitlines():
        if "seiz" in ln:
            nums = re.findall(r"([0-9]+(?:\.[0-9]+)?)", ln)
            if len(nums) >= 2:
                a, b = map(float, nums[:2])
                if b > a:
                    ints.append((a, b))
    return ints

def get_seizure_intervals_for_edf(service: Resource, patient_folder_id: str,
                                  patient_name: str, edf_name: str) -> List[Tuple[float, float]]:
    # 1) tenta .edf.seizures
    seiz_meta = find_file_by_name(service, patient_folder_id, f"{edf_name}.seizures")
    if seiz_meta:
        txt = download_text_from_drive(service, seiz_meta["id"])
        ints = _parse_intervals_from_text(txt)
        print(f"[labels] source={seiz_meta['name']} mime={seiz_meta.get('mimeType')} parsed={ints}")
        if ints: return ints

    # 2) fallback: summary
    summ_meta = find_file_by_name(service, patient_folder_id, f"{patient_name}-summary.txt")
    if summ_meta:
        full_txt = download_text_from_drive(service, summ_meta["id"])
        block = _extract_block_for_file(full_txt, edf_name)
        ints = _parse_intervals_from_text(block)
        print(f"[labels] source={summ_meta['name']} mime={summ_meta.get('mimeType')} parsed={ints}")
        return ints

    print("[labels] nenhuma fonte encontrada (.edf.seizures nem summary)")
    return []