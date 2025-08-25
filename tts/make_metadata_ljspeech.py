#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, csv, os, re, shutil
from pathlib import Path

def normalize_text(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\[[^\]]{1,40}\]", " ", s)   # [웃음] 같은 무대지시 제거
    s = re.sub(r"\([^\)]{1,40}\)", " ", s)   # (한숨) 같은 표기 제거
    s = re.sub(r"<[^>]+>", " ", s)           # <i> 태그 등 제거
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def delimiter_join(parts):
    return ", ".join([p.strip() for p in parts if p is not None])

def load_mapping_from_csv(csv_path: Path):
    mapping = {}
    for delim in [",", "|", "\t"]:
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                for row in csv.reader(f, delimiter=delim):
                    if not row or len(row) < 2:
                        continue
                    fname = row[0].strip()
                    text = delimiter_join(row[1:]).strip()
                    if fname and text:
                        mapping[fname] = text
            if mapping:
                return mapping
        except Exception:
            continue
    raise RuntimeError(f"Failed to read mapping from {csv_path}. Supported delimiters: comma, pipe, tab.")

def load_mapping_from_txt_dir(txt_dir: Path):
    mapping = {}
    for p in sorted(Path(txt_dir).glob("*.txt")):
        mapping[p.stem + ".wav"] = p.read_text(encoding="utf-8", errors="ignore")
    if not mapping:
        raise RuntimeError(f"No .txt files found in {txt_dir}")
    return mapping

def main():
    ap = argparse.ArgumentParser(description="Make LJSpeech-style metadata.csv")
    ap.add_argument("--wavs", required=True, help="Folder with WAV chunks (e.g., dataset/chunks)")
    ap.add_argument("--txt", help="Folder with per-file TXT transcripts matching WAV basenames")
    ap.add_argument("--csv", help="CSV/TSV/Pipe file with mapping: filename,text  OR filename|text")
    ap.add_argument("--out", required=True, help="Output dataset root (creates <out>/wavs + metadata.csv)")
    ap.add_argument("--copy", action="store_true", help="Copy WAVs instead of symlinking")
    args = ap.parse_args()

    wav_dir = Path(args.wavs)
    out_root = Path(args.out)
    out_wavs = out_root / "wavs"
    out_root.mkdir(parents=True, exist_ok=True)
    out_wavs.mkdir(parents=True, exist_ok=True)

    if args.csv:
        mapping = load_mapping_from_csv(Path(args.csv))
    elif args.txt:
        mapping = load_mapping_from_txt_dir(Path(args.txt))
    else:
        raise SystemExit("Provide transcripts with --csv or --txt")

    lines, missing = [], []
    for wav in sorted(wav_dir.glob("*.wav")):
        key = wav.name
        text = mapping.get(key)
        if not text:
            missing.append(key)
            continue
        text = normalize_text(text)
        dst = out_wavs / wav.name
        if not dst.exists():
            try:
                os.symlink(wav.resolve(), dst)
            except Exception:
                shutil.copy2(wav, dst)
        lines.append(f"{wav.name}|{text}")

    (out_root / "metadata.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if missing:
        (out_root / "missing.txt").write_text("\n".join(missing) + "\n", encoding="utf-8")
        print(f"[WARN] Missing transcripts for {len(missing)} files. See: {out_root/'missing.txt'}")
    print(f"[OK] Wrote {out_root/'metadata.csv'}  (rows: {len(lines)})")
    print(f"WAVs placed at: {out_wavs} (symlinked by default; use --copy to copy files)")

if __name__ == "__main__":
    main()
