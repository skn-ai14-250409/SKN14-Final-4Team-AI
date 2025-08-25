from TTS.api import TTS
SNAP = "/Users/yun-iseo/tmp_vits_eval" 

# 폴더만 넘기기 (폴더 안에 config.json + best_model_*.pth 있어야 함)
try:
    tts = TTS(model_path=SNAP)
except Exception:
    # 방법 B: 가중치와 설정을 명시적으로 지정
    tts = TTS(
        model_path=f"{SNAP}/best_model_96.pth",   # 파일명이 다르면 맞게 바꾸세요
        config_path=f"{SNAP}/config.json"
    )

tts.tts_to_file(
    text="테스트 문장을 읽어봅니다. 학습 중간 모델입니다.",
    file_path="sample_out.wav"
)
print("saved: sample_out.wav")
