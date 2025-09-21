from pathlib import Path

from faster_whisper import WhisperModel

AUDIO_DIR = Path("dataset/chunks")
OUT_DIR = Path("dataset/transcripts"); OUT_DIR.mkdir(exist_ok=True)

# device: "metal" 권장(M2). 실패 시 "cpu". compute_type는 int8이 제일 빠름.
try:
    model = WhisperModel("small", device="metal", compute_type="int8")
except Exception:
    model = WhisperModel("small", device="cpu", compute_type="int8")

for wav in sorted(AUDIO_DIR.glob("*.wav")):
    try:
        segments, info = model.transcribe(
            str(wav), language="ko",
            vad_filter=False,
            beam_size=1, temperature=0.0,
            word_timestamps=False
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        (OUT_DIR / f"{wav.stem}.txt").write_text(text, encoding="utf-8")
        # print("OK:", wav.name)
    except Exception as e:
        print("FAIL:", wav.name, "-", e)

