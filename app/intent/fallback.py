from starlette.responses import HTMLResponse
from typing import Any, Dict, Optional

from app.intent.IntentBase import IntentBase


class Fallback(IntentBase):

    def __init__(self):
        prompt = """
당신은 재활용소재로 패션제품을 만드는 회사에서 고문으로 일하고 있는 친환경 활동가입니다.
재활용소재나 재활용소재로 만든 패션제품과 같은 친환경 질문에만 답변합니다.  
단, 사용자의 질문이 재활용소재나 재활용소재로 만든 패션제품과 같은 친환경 질문에 해당하지 않더라도
사용자의 기분이 상하지 않게 거절하면서 동시에 사용자가 재활용소재나 재활용소재로 만든 패션제품과 같은 친환경 정보를 질문할 수 있도록
부드럽게 유도하는 답변을 반환해야합니다.
1. 답변에서 인사말/당신을 소개하는 문구/불필요한 미사여구는 제외합니다.
2. 답변은 공백을 제외했을 때 250자 내외로 작성합니다.
"""
        super().__init__(prompt)

    # ✅ 오케스트레이터 표준 엔트리포인트(항상 dict 반환)
    async def run(self, query: str, slots: Dict[str, Any] | None = None, profile: Dict[str, Any] | None = None, **kwargs):
        slots = slots or {}
        # 필요하면 성별 등 프로필을 라이트하게 반영
        gender = (profile or {}).get("gender")

        # 가이드 프롬프트(질문 맥락 포함)
        user_prompt = f"""
아래 사용자의 질문이 우리 범위를 벗어날 가능성이 있습니다. 해당 질문에는 최대한 짧고 간결하게 정중히 답변하고,
스타일 또는 패션 관련 주제로 유도할 수 있는 간접적인 질문 또는 멘트를 통해 자연스럽게 스타일/패션 화제로 이어가세요.
- 절대 '소재 설명/제품 탐색/코디 추천/인증 설명' 같은 내부 용어를 쓰지 마세요.
- 4~6문장, 목록/마크다운/해시태그 금지
- 마지막 문장에 한 줄짜리 선택 유도 질문 포함(예: "관심 있는 소재나 카테고리를 알려주실래요?")

질문: {query}
사용자 성별(있으면 참고): {gender or "미상"}
""".strip()

        text = self.ask_llm(query=query, prompt=user_prompt, **kwargs)
        # 방탄: 혹시 빈 문자열이면 최소 안내문 생성
        if not text or not text.strip():
            text = "질문 범위를 벗어난 내용으로 보여요. 친환경 소재(예: 린넨·오가닉 코튼)나 찾으시는 카테고리를 알려주시면 적합한 정보와 제품을 안내해 드릴게요."

        return {
            "text":  text.strip(),
            "html":  "",       # Fallback은 텍스트 위주
            "slots": slots,    # 다운스트림 교정이 필요하면 여기서도 보정 가능
            "meta":  {"mode": "fallback"}
        }
    
    def __call__(self, **kwargs):
        result = self.ask_llm(**kwargs)
        return HTMLResponse(result, status_code=200)
