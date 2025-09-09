# convert_to_ljspeech.py
from pathlib import Path

IN  = Path("my-tts-ds/metadata.csv")            # 기존 2열: 0001.wav|텍스트
OUT = Path("my-tts-ds/metadata_ljs.csv")        # 새 3열: 0001|텍스트|텍스트

lines_out = []
for ln in IN.read_text(encoding="utf-8").splitlines():
    if not ln.strip(): 
        continue
    f, *rest = ln.split("|", 1)
    text = (rest[0] if rest else "").strip()
    id_ = f.strip().removesuffix(".wav")
    # 3열로: id | text | text  (정규화 텍스트가 따로 없으면 동일값 복제)
    lines_out.append(f"{id_}|{text}|{text}")

OUT.write_text("\n".join(lines_out) + "\n", encoding="utf-8")
print(f"[OK] wrote {OUT} (rows={len(lines_out)})")
