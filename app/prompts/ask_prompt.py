# 누락 슬롯 유도 질문(ASK) 생성용 시스템 프롬프트 + 헬퍼

from __future__ import annotations
from typing import Dict, Any, List, Optional
import json

# intent별 필수/보조 우선순위
ASK_PRIORITIES: Dict[str, Dict[str, List[str]]] = {
    "OUTFIT_RECO": {
        "required": ["상황", "성별"],  # 성별은 프로필에서 자동 수집(질문 금지)
        "optional_priority": ["계절","스타일","팔레트톤","예산","컬러","소재","카테고리","인증"],
    },
    "PRODUCT_FIND": {
        "required": ["카테고리", "성별"],
        "optional_priority": ["상황","스타일","예산","컬러","소재","계절","팔레트톤","인증"],
    },
    "CERT_VERIFY": {
        "required": ["인증"],
        "optional_priority": ["상황","스타일","카테고리","성별","예산","컬러","소재","계절","팔레트톤"],
    },
    "MATERIAL_EXPLAIN": {
        "required": ["소재"],
        "optional_priority": ["상황","스타일","카테고리","성별","예산","컬러","계절","팔레트톤","인증"],
    },
    "FALLBACK": {
        "required": [],
        "optional_priority": ["상황"],  # 친환경 관심 유도 + 상황 확보
    },
}

prompt_ask_system: str = (
    "당신은 한국어 UX 라이터입니다. 목적은 ‘부족한 슬롯 정보를 직접적 표현 없이(간접 질문)’ 자연스럽게 묻는 것입니다.\n"
    "입력: intent, slots(현재값), profile.gender, priorities, query.\n"
    "규칙:\n"
    "1) 성별은 profile.gender로 자동 수집되므로 절대 묻지 마세요.\n"
    "2) intent의 required 중 null이 있으면 → 그 슬롯(들)을 먼저 겨냥해 1~2문장 간접 질문.\n"
    "3) 필수 다 채워졌다면 → optional_priority 순서대로 누락된 슬롯 상위 1~2개만 간접 질문.\n"
    "4) 슬롯 용어를 직접 말하지 말고 맥락/선호/상황을 묻는 말로 표현. 한 문장 130자 이내, 총 1~2문장.\n"
    "5) FALLBACK에서는 친환경 제품 관심을 부드럽게 끌면서 OUTFIT_RECO의 필수(상황)를 자연스럽게 확보.\n"
    "출력은 반드시 JSON 하나의 객체만: {\"ask\": \"문장\", \"targets\": [\"슬롯1\",\"슬롯2\"]}\n"
)

def build_ask_user_prompt(intent: str, query: str, profile: Dict[str, Any], slots: Dict[str, Any]) -> str:
    """
    ASK 생성을 위해 LLM에 전달할 user 프롬프트를 구성합니다.
    """
    return (
        f"intent: {intent}\n"
        f'query: "{query}"\n'
        f"profile: {json.dumps(profile, ensure_ascii=False)}\n"
        f"slots: {json.dumps(slots, ensure_ascii=False)}\n"
        f"priorities: {json.dumps(ASK_PRIORITIES, ensure_ascii=False)}\n\n"
        "요청:\n"
        "- 위 시스템 지침에 따라 누락 슬롯을 간접적으로 유도하는 ask를 생성하세요.\n"
        "- 규칙: 필수 누락 우선 → 없으면 보조 우선순위 상위 1–2개만.\n"
        "- 성별은 profile에서 자동 수집되므로 묻지 마세요.\n"
        "- JSON 객체 하나만 반환하세요."
    )
