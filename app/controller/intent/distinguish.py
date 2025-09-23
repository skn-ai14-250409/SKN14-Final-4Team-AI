from .IntentBase import IntentBase
from .cert_verify import CertVerify
from .fallback import Fallback
from .material_explain import MaterialExplain
from .outfit_repo import OutfitReco
from .product_find import ProductFind
from .show_composition import ShowComposition


class Distinguish(IntentBase):

    def __init__(self):
        self.intents = {
            "material_explain": {
                "description": "친환경 소재",
                "questions"  : ["재활용 소재를 쓰는 이유가 뭐야?", "재활용 소재에 뭐가 있어?"],
                "process"    : MaterialExplain()
            },
            "product_find"    : {
                "description": "의류/옷/상의/하의/의복/제품",
                "questions"  : ["여성 재킷 보여줘.", "Ecolabel 인증을 받은 제품을 찾아줘.", "흰 바지에 어울리는 옷을 찾아줘."],
                "process"    : ProductFind()
            },
            "outfit_reco"     : {
                "description": "룩/스타일/스타일링/옷차림",
                "questions"  : ["결혼식에 어울리는 옷차림을 추천해줘.", "레이어드룩", "데일리룩 보여줘."],
                "process"    : OutfitReco()
            },
            "show_composition": {
                "description": "착샷/착용/조합",
                "questions"  : ["레이어드룩 제품을 입은 모습", "이 제품들 조합해서 보고싶어."],
                "process"    : ShowComposition()
            },
            "cert_verify"     : {
                "description": "친환경인증/친환경 인증정보/친환경 인증마크/친환경 인증기호",
                "questions"  : ["친환경인증 종류는 뭐가 있어?", "RCS 는 뭐하는 마크야?"],
                "process"    : CertVerify()
            },
            "fallback"        : {
                "description": "일상질문/일상대화/잡담",
                "questions"  : ["오늘 날씨 어때?", "심심해."],
                "process"    : Fallback()
            }
        }
        context = "\n".join([f"| {catg} | {info['description']} | {', '.join(info['questions'])} |" for catg, info in self.intents.items()])
        prompt = f"""
사용자 질의가 아래 <<분류기준>> 중 어디에 속하는지 <<항목>> 만을 반환하세요.
<<분류기준>>
| 항목 | 설명 | 예시 |
|------|------|------|
{context}
"""
        super().__init__(prompt)

    def __call__(self, query):
        intent = self.ask_llm(query, model="gpt-5-mini-2025-08-07")
        return self.intents[intent]

