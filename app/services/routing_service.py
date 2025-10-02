import json, re, os, logging
from typing import Any, Dict, Optional, Tuple, List
from pydantic import BaseModel, field_validator, ValidationError
from app.adapters.llm import LLM
from app.prompts.router_prompt import prompt_router
from app.prompts.ask_prompt import prompt_ask_system, build_ask_user_prompt

logger = logging.getLogger(__name__)

class RouterOut(BaseModel):
    tool: str
    slots: Dict[str, Any]

    @field_validator("tool")
    @classmethod
    def _valid_tool(cls, v):
        allow = {"CERT_VERIFY","PRODUCT_FIND","OUTFIT_RECO","MATERIAL_EXPLAIN","FALLBACK"}
        if v not in allow:
            raise ValueError(f"invalid tool: {v}")
        return v

ALLOWED = {"CERT_VERIFY","PRODUCT_FIND","OUTFIT_RECO","MATERIAL_EXPLAIN","FALLBACK"}

async def route_and_ask(query: str, profile: Dict[str, Any]) -> Tuple[str, Dict[str, Any], Optional[str], Optional[List[str]], str]:
    q = re.sub(r"\s+", " ", (query or "").strip())

    # 1) 인텐트/슬롯: 프롬프트 반영 + JSON 모드 호출 (프롬프트 준수율↑)
    raw = await LLM.call(
        messages=[{"role":"system","content": prompt_router},
                  {"role":"user","content": q}],
        json_mode=True,              # 내부에서 response_format/json 설정
        temperature=0                # (LLM 어댑터가 지원하면 전달)
    )

    # 2) JSON 파싱 + Pydantic 검증
    try:
        data = json.loads(raw)
        tool  = (data.get("tool") or "FALLBACK").strip().upper()
        slots = data.get("slots") or {}
        _ = RouterOut(tool=tool, slots=slots)      # 검증
        tool  = _.tool
        slots = _.slots
    except Exception as e:
        logger.warning(f"[router] invalid json: {e}; raw={raw[:200]}")
        tool, slots = "FALLBACK", {}

    # 3) 성별 자동 보정
    gender = (profile or {}).get("gender")
    if gender and not slots.get("성별"):
        slots["성별"] = "남성" if str(gender).lower().startswith("m") else "여성"

    # 4) ask 생성 (필요할 때만)
    ask, targets = None, None
    try:
        ask_raw = await LLM.call(
            messages=[
                {"role":"system","content": prompt_ask_system},
                {"role":"user","content": build_ask_user_prompt(tool, q, profile or {}, slots)}
            ],
            json_mode=True,
            temperature=0
        )
        obj = json.loads(ask_raw)
        ask, targets = obj.get("ask"), obj.get("targets")
    except Exception as e:
        logger.debug(f"[ask] skip: {e}")

    return tool, slots, ask, targets, "ok"
