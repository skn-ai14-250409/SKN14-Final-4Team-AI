# intent/base.py
from __future__ import annotations
from typing import Any, Dict, Optional, Literal, Final, Type, Protocol
from pydantic import BaseModel, Field, ValidationError
from starlette.responses import HTMLResponse

ToolLiteral = Literal["CERT_VERIFY","PRODUCT_FIND","OUTFIT_RECO","MATERIAL_EXPLAIN","FALLBACK"]
ALLOWED_TOOLS: Final[set[str]] = set(ToolLiteral.__args__)  # type: ignore

def normalize_tool(v: str | None) -> ToolLiteral:
    vv = (v or "FALLBACK").strip().upper()
    return vv if vv in ALLOWED_TOOLS else "FALLBACK"

class IntentResult(BaseModel):
    """모든 인텐트가 공통으로 반환하는 Envelope"""
    tool: ToolLiteral = Field(..., description="실행된 툴 라벨")
    slots: Dict[str, Any] = Field(default_factory=dict, description="보정/추가된 슬롯")
    text: str = Field(..., description="메인 응답 텍스트(프론트 표시용)")
    payload: Optional[Dict[str, Any]] = Field(
        default=None,
        description="툴별 상세 데이터(추천 리스트, 인증 정보 등). 스키마는 별도 모듈에 정의"
    )
    meta: Optional[Dict[str, Any]] = Field(default=None, description="디버깅/추적용")

# 스키마 검증 헬퍼(툴별 페이로드 모델을 주입받아 검증)
def validate_payload(tool: ToolLiteral, payload: Dict[str, Any] | None, model: Type[BaseModel] | None) -> Dict[str, Any] | None:
    if payload is None or model is None:
        return payload
    try:
        return model.model_validate(payload).model_dump()  # v2
    except ValidationError as e:
        # 운영에서는 로깅 후 payload None 처리 또는 축약 반환
        return None

class BaseIntent(Protocol):
    def __call__(self, **kwargs) -> Any:
        kwargs["with_voice"] = True
        result = self.ask_llm(**kwargs)
        return HTMLResponse(result, status_code=200)

__all__ = ["ToolLiteral","ALLOWED_TOOLS","normalize_tool","IntentResult","validate_payload","BaseIntent"]