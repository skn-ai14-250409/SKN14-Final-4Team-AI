import time

from fastapi import APIRouter, Body
from app.controller.intent.distinguish import Distinguish


#################################################### FastAPI api Routing
router = APIRouter(prefix="/api", tags=["API"], responses={404: {"description": "Not found"}} )

#################################################### 사용자 질의의 목적 구분
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

    start = time.time()
    intent  = distinguish(query)                          # 질의를 사전에 정해놓은 분류대로 나누기
    end = time.time()
    print(f"Distinguish :: Time spend : {end - start}")
    # return intent
    # print(f"{intent=}")

    start = time.time()
    process = intent["process"]                    # 수행할 함수 확인
    result  = process(query=query, user_id=user_id, ai_id=ai_id)  # 분류에 맞게 사용자 질의를 재정의하여 데이터 전달하기
    end = time.time()
    print(f"Process-{process} :: Time spend : {end - start}")
    return result