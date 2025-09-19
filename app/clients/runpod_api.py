from __future__ import annotations
import os, re, json, time
from dataclasses import dataclass
from typing import Any, Dict, Optional, List
import requests

# 응답에 있는 URL 형태 찾기위한 정규식
_URL_RE = re.compile(r"https?://[^\s\"'<>]+", re.IGNORECASE)


def _extract_first_url(obj: Any) -> Optional[str]:
    """
    RunPod 응답의 output/logs 어디에 URL이 와도 최대한 찾아서 반환.
    - dict면 흔한 키(s3_url/audio_url/url/file_url) 우선 확인 → 없으면 값들을 재귀적으로 탐색
    - list/tuple이면 각 원소를 재귀적으로 탐색
    - str이면 정규식으로 URL 패턴을 추출
    - 아무것도 없으면 None
    """
    # 1) 딕셔너리: 보편적인 키 먼저 체크
    if isinstance(obj, dict):
        for k in ("s3_url", "audio_url", "url", "file_url"):
            if k in obj and isinstance(obj[k], str) and obj[k].startswith("http"):
                return obj[k]
        # 값들 안쪽도 검사
        for v in obj.values():
            found = _extract_first_url(v)
            if found:
                return found

    # 2) 리스트/튜플: 각 원소 재귀
    if isinstance(obj, (list, tuple)):
        for item in obj:
            found = _extract_first_url(item)
            if found:
                return found

    # 3) 문자열: 정규식으로 URL 추출
    if isinstance(obj, str):
        m = _URL_RE.search(obj)
        if m:
            return m.group(0)

    return None


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

        status = data.get("status")
        if status not in ("COMPLETED", "IN_QUEUE", "IN_PROGRESS"):  # 드물게 변형된 상태값 대비
            # 그래도 output이 있을 수 있으니 URL을 먼저 시도
            pass

        # 결과 URL 추출 시도
        url_out = _extract_first_url(data.get("output")) or _extract_first_url(data)
        return {
            "raw": data,
            "status": status,
            "url": url_out,
        }
