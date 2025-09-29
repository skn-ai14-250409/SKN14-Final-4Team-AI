from app.intent.IntentBase import IntentBase

class Distinguish(IntentBase):

    def __init__(self, context):
        prompt = f"""
사용자 질의가 아래 <<분류기준>> 중 어디에 속하는지 <<항목>> 만을 반환하세요.
<<분류기준>>
| 항목 | 설명 |
|------|------|
{context}
"""
        super().__init__(prompt)

    def __call__(self, query):
        return self.ask_llm(query, model="gpt-5-mini-2025-08-07")

