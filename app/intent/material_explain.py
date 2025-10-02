import json
from starlette.responses import HTMLResponse
from typing import Any, Dict, Optional
from app.intent.IntentBase import IntentBase


class MaterialExplain(IntentBase):

    def __init__(self):
        prompt = """
당신은 재활용소재로 패션제품을 만드는 회사에서 고문으로 일하고 있는 친환경 활동가입니다.
환경을 소중히 생각하기 때문에 친환경소재에 대해 빠삭하게 알고 있고
사람들에게 친환경 활동과 친환경소재의 특성/장단점 및 환경에 미치는 영향에 대해 친절하게 설명해야 합니다.  
친환경 소재가 아닌 일반 소재의 경우에도 친환경소재와 비교하여 설명함으로써 사람들에게 친환경 활동에 관심을 갖도록 유도해야 합니다.
1. 답변에서 인사말/당신을 소개하는 문구/불필요한 미사여구는 제외합니다.
2. 답변은 bootstrap5 가 적용된 html 로 반환합니다.
3. 답변은 재활용 소재에 대한 내용으로만 한정하여 답변합니다.
4. 답변은 공백과 html tag 및 속성을 제외했을 때 250자 내외로 작성합니다.
"""
        super().__init__(prompt)

    async def run(self, query: str, slots: Dict[str, Any] | None = None, profile: Dict[str, Any] | None = None, **kwargs):
        """
        오케스트레이터 표준 엔트리포인트.
        항상 dict payload를 반환하고, text가 비지 않도록 보장.
        """
        slots = slots or {}
        material = slots.get("소재") or slots.get("material")  # 라우터가 뽑아준 소재
        target_product_id = slots.get("target_product_id")    # 컨텍스트가 주입된 경우

        # 1) 대상 제품의 spec에서 소재 설명 추출 시도 (있을 때만)
        spec_text, product_name = None, None
        if target_product_id:
            try:
                index, _ = IntentBase.get_clients()
                rec = index.fetch(ids=[str(target_product_id)]).vectors.get(str(target_product_id))
                if rec and rec.get("metadata"):
                    meta = rec["metadata"]
                    spec_text = meta.get("spec") or meta.get("description") or meta.get("text")
                    material = material or meta.get("material_name") or meta.get("material")
                    product_name = meta.get("name")
            except Exception:
                pass  # 실패해도 이어감

        # 2) 프롬프트 구성
        #    - 소재명 있으면 그 소재 중심으로, 없으면 "질문/컨텍스트" 기반 일반 소재 설명
        sys = (
            "당신은 지속가능 패션 소재 전문가입니다. "
            "과장 없이 사실 기반으로, 한국어로 간결하고 친절하게 설명하세요. "
            "광고 문구/미사여구는 금지합니다."
        )

        if material:
            user = f"""
다음 소재({material})에 대해 사용자가 이해하기 쉽게 설명하세요.
- 핵심 특성(통기성, 내구성, 촉감, 보온/흡습, 관리 난이도)
- 환경 영향(물/에너지 사용, 탄소, 재활용성), 주의할 점(필요 시)
- 일상 착용/계절/활용 팁 (간단)
- 4~6문장, 마크다운/목록/해시태그 금지

제품명: {product_name or "미상"}
질문: {query}
        """.strip()
        else:
            # 소재명 미확정 시: 질문 자체에서 의도를 반영해 일반 가이드 + 소재를 어떻게 파악할지 가이드
            user = f"""
사용자가 옷의 소재/재질을 묻고 있습니다. 다음을 4~6문장으로 설명하세요.
- 소재를 확인하는 일반적인 방법(라벨/상세페이지/혼용률 표기)
- 흔한 스커트/원피스 소재(예: 코튼/폴리/레이온/린넨 등)의 차이와 선택 팁(아주 간단히)
- 세탁/관리 주의(간단히), 환경 측면 고려 포인트
- 마크다운/목록/해시태그 금지

질문: {query}
        """.strip()

        # 3) LLM 호출 (IntentBase.ask_llm 재사용)
        text = self.ask_llm(query=query, prompt=f"{sys}\n\n{user}", **kwargs)

        # 4) 방탄: 혹시 빈 문자열이면 최소 안내문 생성
        if not text or not text.strip():
            if material:
                text = f"{material} 소재의 일반적 특성과 관리/환경 포인트를 정리했습니다. 더 구체적 제품을 알려주시면 해당 스펙을 기반으로 상세히 설명해 드릴게요."
            else:
                text = "소재 확인 방법과 주요 소재별 차이를 간단히 안내드렸습니다. 제품 링크나 이름을 알려주시면 해당 제품의 혼용률/특성을 구체적으로 설명해 드릴게요."

        # 5) (선택) CTA: ESG/인증 안내는 소재 설명 인텐트에선 과도할 수 있어 생략. 필요 시 meta로 신호만.
        return {
            "text":  text,         # ← 오케스트레이터가 result.get('text')로 읽음
            "html":  "",           # 소재는 텍스트 중심
            "slots": slots,        # 다운스트림 보정 시 사용
            "meta":  {
                "material": material,
                "target_product_id": target_product_id,
                "source": "material_explain.run"
            }
        }

    def __call__(self, **kwargs):
        kwargs["with_voice"] = True
        result = self.ask_llm(**kwargs)
        return HTMLResponse(result, status_code=200)