from __future__ import annotations
import os, re, json, uuid, time, datetime
import requests, boto3

from dataclasses import dataclass
from typing import Any, Dict, Optional
from openai import OpenAI


_llm_client = OpenAI()
_CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

def summarize_p_text_via_llm(html_str: str) -> str:
    """
    HTML을 LLM에 그대로 보내서:
    1) 내부 텍스트만 추출
    2) 한국어 50~100자 요약
    """
    if not isinstance(html_str, str) or not html_str.strip():
        return ""

    messages = [
        {
            "role": "system",
            "content": (
                "너는 HTML 문서에서 본문을 요약하는 도우미야. "
                "반드시 다음 순서를 지켜:\n"
                "1) 제공된 HTML에서 내부 텍스트만 순서대로 추출해.\n"
                "2) 그 텍스트만 기반으로 한국어로 50~100자 사이로 간결하게 요약해.\n"
                "3) HTML/마크업/따옴표/접두사 없이 순수 문장만 출력해.\n"
            )
        },
        {
            "role": "user",
            "content": (
                "다음은 HTML이야. 위 지침을 따라서 출력해:\n\n"
                f"{html_str}"
            )
        }
    ]

    resp = _llm_client.chat.completions.create(
        model=_CHAT_MODEL,
        messages=messages,
        temperature=0.2,
        max_tokens=150,
    )
    return (resp.choices[0].message.content or "").strip()

# --------- S3 업로드 유틸 ---------------------------------------------------------
@dataclass
class S3Config:
    bucket       : str
    region       : str = os.getenv("AWS_S3_REGION", "ap-northeast-2")
    prefix       : str = os.getenv("AWS_S3_PREFIX", "model_tts")
    public_read : bool = (os.getenv("AWS_S3_PUBLIC_READ", "1").lower() in ("1","true","yes"))

class S3Uploader:
    def __init__(self, cfg: S3Config):
        self.cfg = cfg
        self.s3  = boto3.client("s3", region_name=cfg.region)

    def build_key(self, filename: str) -> str:
        stamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        return f"{self.cfg.prefix}/{stamp}/{filename}"

    def put_bytes(self, data: bytes, key: str, content_type: str = "audio/mpeg") -> Dict[str, Any]:
        extra = {"ContentType": content_type}
        self.s3.put_object(Bucket=self.cfg.bucket, Key=key, Body=data, **extra)

        url   = f"https://{self.cfg.bucket}.s3.{self.cfg.region}.amazonaws.com/{key}"
        return {"url": url, "key": key}


# --------- ElevenLabs 호출 --------------------------------------------------------
@dataclass
class ElevenConfig:
    api_key   : str
    model_id  : str = os.getenv("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2")

def _resolve_voice_id(persona: Optional[str]) -> Optional[str]:
    """
    1) ELEVENLABS_VOICE_MAP에 persona→voice_id JSON이 있으면 우선 사용
       예: {"2":"<voice-id-abc>", "HongJinkyeong":"<voice-id-xyz>"}
    2) ELEVENLABS_VOICE_ID_{PERSONA} 환경변수
    3) ELEVENLABS_DEFAULT_VOICE_ID
    """
    # 1) JSON 매핑
    vm = os.getenv("ELEVENLABS_VOICE_MAP")
    if vm:
        try:
            m = json.loads(vm)
            if persona in m and m[persona]:
                return m[persona]
        except Exception:
            pass
    # 2) 개별 키
    if persona:
        vid = os.getenv(f"ELEVENLABS_VOICE_ID_{persona}")
        if vid:
            return vid
    # 3) 기본
    return os.getenv("ELEVENLABS_DEFAULT_VOICE_ID")

class ElevenLabsTTSClient:
    """
    RunPod 클라이언트와 인터페이스를 맞춘 ElevenLabs 버전.
    - run_tts(text, persona) -> { "raw": {...}, "status": "COMPLETED", "url": "https://..." }
    - ElevenLabs 응답(오디오 바이트)을 받아 S3에 업로드 후 URL 반환
    """
    def __init__(self, cfg: Optional[ElevenConfig] = None, s3cfg: Optional[S3Config] = None):
        api_key = os.getenv("ELEVENLABS_API_KEY")
        if not api_key:
            raise RuntimeError("ELEVENLABS_API_KEY 가 설정되어 있지 않습니다 (.env).")

        bucket = os.getenv("AWS_S3_BUCKET_NAME")
        if not bucket:
            raise RuntimeError("AWS_S3_BUCKET_NAME이 설정되어 있지 않습니다.")

        self.cfg     = cfg or ElevenConfig(api_key=api_key)
        self.s3_up   = S3Uploader(s3cfg or S3Config(bucket=bucket))
        self.session = requests.Session()
        self.session.headers.update({"xi-api-key": api_key, "Accept":"audio/mpeg", "Content-Type":"application/json"})

        # 안전장치: 너무 긴 텍스트는 잘라내기 (ElevenLabs 일반 한도 ~5k자 정도)
        self.max_chars = int(os.getenv("ELEVENLABS_MAX_CHARS", "4800"))

        # 음성 합성 기본 옵션
        self.stability         = float(os.getenv("ELEVENLABS_STABILITY", "0.25"))
        self.similarity_bo     = float(os.getenv("ELEVENLABS_SIMILARITY_BOOST", "0.5"))
        self.style             = float(os.getenv("ELEVENLABS_STYLE", "0.0"))
        self.use_speaker_boost = (os.getenv("ELEVENLABS_USE_SPEAKER_BOOST", "1").lower() in ("1","true","yes"))

        self.last_input_text = ""

    def run_tts(self, *, text: str, persona: Optional[str] = None) -> Dict[str, Any]:
        # 라우터에서 가끔 HTML이 그대로 넘어오므로 방지 차원에서 한번 더 정리
        text = (text or "").strip()
        if len(text) > self.max_chars:
            text = text[: self.max_chars]
        self.last_input_text = text

        voice_id = _resolve_voice_id(persona)
        if not voice_id:
            raise RuntimeError(
                "ElevenLabs voice_id를 찾을 수 없습니다. "
                "ELEVENLABS_VOICE_MAP 또는 ELEVENLABS_VOICE_ID_{persona} 또는 ELEVENLABS_DEFAULT_VOICE_ID 를 설정하세요."
            )

        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        payload = {
            "model_id": self.cfg.model_id,
            "text": text,
            "voice_settings": {
                "stability": self.stability,
                "similarity_boost": self.similarity_bo,
                "style": self.style,
                "use_speaker_boost": self.use_speaker_boost
            }
        }

        resp = self.session.post(url, json=payload, timeout=120)

        if resp.status_code >= 400:
            # API 에러 메시지 전달
            try:
                err = resp.json()
            except Exception:
                err = resp.text[:500]
            raise RuntimeError(f"ElevenLabs HTTP {resp.status_code}: {err}")

        audio_bytes = resp.content  # 기본은 audio/mpeg
        # 파일명 생성 후 S3 업로드
        key = self.s3_up.build_key(f"tts_{uuid.uuid4().hex}.mp3")
        put = self.s3_up.put_bytes(audio_bytes, key, content_type=resp.headers.get("Content-Type", "audio/mpeg"))

        # 반환
        return {
            "voice_id":voice_id,
            "s3_url":put["url"]
        }

    def __str__(self):
        return self.last_input_text