import inspect
from starlette.responses import Response

from typing import Any, Dict, List, Optional
from app.intent.registry import get_intent_handler
from app.services.routing_service import route_and_ask

# 슬롯 기반 교정 헬퍼
OUTFIT_HINTS = {"룩", "코디", "스타일", "스타일링", "착장", "매치", "하객룩", "오피스룩", "캠핑룩", "데이트룩", "데일리룩"}


def _normalize_tool_with_slots(tool: str, slots: Dict[str, Any], query: str) -> str:
    slots = slots or {}
    q = (query or "").lower()
    category  = slots.get("카테고리") or slots.get("category")
    situation = slots.get("상황") or slots.get("situation")
    # cert      = slots.get("인증") or slots.get("cert")

    # if tool == "CERT_VERIFY" or cert:
    #     return "CERT_VERIFY"
    if category and not situation:
        return "PRODUCT_FIND"
    if situation and not category and any(h in q for h in OUTFIT_HINTS):
        return "OUTFIT_RECO"
    if category and situation:
        return "OUTFIT_RECO" if any(h in q for h in OUTFIT_HINTS) else "PRODUCT_FIND"
    return tool

def _to_payload(res):
    if isinstance(res, Response):
        # 그대로 반환하려면 상위에서 처리; 여기서는 dict로 표준화 안 함
        return {"text": "", "html": "", "slots": {}, "meta": {}, "__response__": res}
    if isinstance(res, dict):
        return {
            "text":  res.get("text","") or "",
            "html":  res.get("html","") or "",
            "slots": res.get("slots",{}) or {},
            "meta":  res.get("meta",{}) or {},
        }
    # 객체 속성 접근도 허용 (예: result.text)
    if any(hasattr(res, k) for k in ("text","html","slots","meta")):
        return {
            "text":  getattr(res,"text","") or "",
            "html":  getattr(res,"html","") or "",
            "slots": getattr(res,"slots",{}) or {},
            "meta":  getattr(res,"meta",{}) or {},
        }
    # 마지막: 문자열 등은 text로 래핑
    return {"text": str(res), "html": "", "slots": {}, "meta": {}}

async def run_pipeline(query: str, profile: Dict[str, Any]) -> Dict[str, Any]:
    tool, slots, ask, targets, note = await route_and_ask(query, profile)
    tool = _normalize_tool_with_slots(tool, slots, query)

    handler = get_intent_handler(tool)

    # ✅ run()이 있으면 사용, 없으면 __call__로 폴백
    if hasattr(handler, "run"):
        result = await handler.run(query=query, slots=slots, profile=profile) \
                 if inspect.iscoroutinefunction(handler.run) \
                 else handler.run(query=query, slots=slots, profile=profile)
    elif callable(handler):
        result = handler(query=query, slots=slots, profile=profile)
    else:
        raise AttributeError(f"{handler.__class__.__name__} has no 'run' or '__call__'")

    payload = _to_payload(result)
    # Response 객체면 그대로 반환하도록 상위에서 분기할 수도 있습니다.
    return {
        "tool":    tool,
        "slots":   payload["slots"],
        "text":    payload["text"],
        "html":    payload["html"],
        "ask":     ask,
        "targets": targets,
        "note":    note,
        "meta":    payload["meta"],
        # "__response__": payload.get("__response__")  # 필요 시 노출
    }