from fastapi import APIRouter, Body

from app.controller.intent.cert_verify import CertVerify
from app.controller.intent.distinguish import Distinguish
from app.controller.intent.fallback import Fallback
from app.controller.intent.material_explain import MaterialExplain
from app.controller.intent.outfit_repo import OutfitReco
from app.controller.intent.product_find import ProductFind
from app.controller.intent.show_composition import ShowComposition

#################################################### FastAPI api Routing
router = APIRouter(prefix="/api", tags=["API"], responses={404: {"description": "Not found"}} )

#################################################### 사용자 질의의 목적 구분
# INTENTS = {
#     "material_explain" : {
#         "description" : "제품의 소재/그 소재의 장단점/그 소재가 환경에 미치는 영향과 같이, 소재나 환경정보를 묻는 질문인 경우.",
#         "process"     : MaterialExplain()
#     },
#     "product_find" : {
#         "description" : "의류제품에 대한 질문인 경우.",
#         "process"     : ProductFind()
#     },
#     "outfit_reco" : {
#         "description" : "스타일링 추천을 요청하는 경우.",
#         "process"     : OutfitReco()
#     },
#     "show_composition" : {
#         "description" : "제품들을 합성한 이미지나 착용샷, 착샷을 보고싶어하는 경우.",
#         "process"     : ShowComposition()
#     },
#     "cert_verify" : {
#         "description" : "GRS/RCS 등의 식별자를 기반으로 친환경 인증정보나 사실에 대해 질의하는 경우.",
#         "process"     : CertVerify()
#     },
#     "fallback" : {
#         "description" : "material_explain, product_find, outfit_reco, cert_verify 로 분류되지 않는 경우.",
#         "process"     : Fallback()
#     }
# }
# context = "\n".join([f"| {catg} | {info['description']} |" for catg,info in INTENTS.items()])
# distinguish = Distinguish(context)
distinguish = Distinguish()

####################################################
@router.post("/ask")
def api_ask(param:dict = Body(None, examples=[{
        "query"   : "폴리에스터는 왜 재활용하는거야?",
        "user_id" : 1,
        "ai_id"   : 1
    }])):
    query   = param.get("query")
    user_id = param.get("user_id")
    ai_id   = param.get("ai_id")

    intent  = distinguish(query)                          # 질의를 사전에 정해놓은 분류대로 나누기
    # return intent
    # print(f"{intent=}")
    process = intent["process"]                    # 수행할 함수 확인
    result  = process(query=query, user_id=user_id, ai_id=ai_id)  # 분류에 맞게 사용자 질의를 재정의하여 데이터 전달하기
    return result