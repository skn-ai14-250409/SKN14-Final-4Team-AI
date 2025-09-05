from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

AUDIO_FILE = Path("tts/look-1_tts.mp3")  # 네 mp3 파일 경로

with AUDIO_FILE.open("rb") as f:
    transcript = client.audio.transcriptions.create(
        model="gpt-4o-mini-transcribe",  # Whisper 최신 모델
        file=f,
        response_format="text"
    )

print("음성 내용:")
print(transcript.strip())
