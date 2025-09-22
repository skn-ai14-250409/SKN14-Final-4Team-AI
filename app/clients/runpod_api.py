from __future__ import annotations
import os, requests

from dataclasses import dataclass
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

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


@dataclass
class RunpodConfig:
    """
    RunPod 호출에 필요한 설정값 묶음.
    - api_key: RunPod API 인증 토큰
    - endpoint_id: 배포한 엔드포인트 ID
    - base_url: API 베이스 URL (보통 고정)
    - timeout_sec: runsync가 완료될 때까지 기다리는 최대 시간
    """
    api_key    : str
    endpoint_id: str
    base_url   : str = "https://api.runpod.ai/v2"
    timeout_sec: int = 120  # runsync는 완료까지 기다리므로 적당히 넉넉히

def _prefer_s3_url(data:dict) -> Optional[str]:
    try:
        return data["output"]["output"]["s3_url"]
    except Exception:
        return None

class RunpodTTSClient:
    """
    RunPod 'runsync' 엔드포인트로 TTS 작업을 던지고,
    완료 응답에서 결과 URL을 뽑아오는 간단한 클라이언트.
    """
    def __init__(self, config: Optional[RunpodConfig] = None):
        api_key     = os.getenv("RUNPOD_API_KEY") if config is None else config.api_key
        endpoint_id = os.getenv("RUNPOD_ENDPOINT_ID") if config is None else config.endpoint_id
        if not api_key:
            raise RuntimeError("RUNPOD_API_KEY 가 설정되어 있지 않습니다 (.env).")
        if not endpoint_id:
            raise RuntimeError("RUNPOD_ENDPOINT_ID 가 설정되어 있지 않습니다 (.env).")

        self.cfg = config or RunpodConfig(api_key=api_key, endpoint_id=endpoint_id)

    def run_tts(self, *, text: str, persona: str) -> Dict[str, Any]:
        """
        RunPod runsync 호출 → 완료 응답을 그대로 dict로 반환.
        실패 시 예외 발생.
        """
        url     = f"{self.cfg.base_url}/{self.cfg.endpoint_id}/runsync"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.cfg.api_key}",
        }
        payload = {"input": {"text": text, "persona": str(persona)}}

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=self.cfg.timeout_sec)
        except requests.Timeout as e:
            raise RuntimeError(f"RunPod runsync 타임아웃 ({self.cfg.timeout_sec}s)") from e
        except requests.RequestException as e:
            raise RuntimeError("RunPod runsync 요청 중 네트워크 오류") from e

        if resp.status_code >= 400:
            raise RuntimeError(f"RunPod runsync HTTP {resp.status_code}: {resp.text[:500]}")

        try:
            data = resp.json()
        except Exception:
            raise RuntimeError(f"RunPod runsync 응답이 JSON이 아닙니다: {resp.text[:500]}")

        status  = data.get("status")
        s3_url  = _prefer_s3_url(data)
        return {"raw": data, "status": status, "s3_url":s3_url}
    
    def run_tts_from_html(self, *, html:str, persona:str) -> Dict[str, Any]:
        summarized = summarize_p_text_via_llm(html)
        return self.run_tts(text=summarized, persona=persona)


# def build_autoplay_audio_tag(src_url: str) -> str:
#     """
#     브라우저 자동재생 정책을 고려해 autoplay/playsinline + JS 보조 시도
#     """
#     return (
#         f'<audio id="ttsAudio" controls autoplay preload="auto" playsinline src="{src_url}"></audio>'
#         "<script>"
#         "  (function(){"
#         "    var a=document.getElementById('ttsAudio');"
#         "    if(a){"
#         "      var play=()=>a.play().catch(()=>{});"
#         "      if(document.readyState==='complete'){play();}"
#         "      else {window.addEventListener('load', play, {once:true});}"
#         "    }"
#         "  })();"
#         "</script>"
#     )