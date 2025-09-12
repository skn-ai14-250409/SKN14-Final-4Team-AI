import json

from fastapi import APIRouter, HTTPException, Query, Path, Body
from openai import OpenAI
from pinecone import Pinecone
from pydantic import BaseModel, Field
import os
from sqlalchemy import text

from app.database import SessionLocal

router = APIRouter(
    prefix="/api",
    tags=["API"],
    responses={404: {"description": "Not found"}}
)

BODY_EXAMPLE:dict = Body(
    None,
    examples=[ {
            "query"   : "가을날씨에 어울리는 출근복장 스타일 추천해줘.",
            "user_id" : 1,
        }
    ]
)

# 사용자 질의의 목적 구분 ###########################################
def __distinguish(client, query:str):
    context = "\n".join([f"| {catg} | {info['description']} |" for catg,info in INTENTS.items()])
    message = f"""
<<사용자 질의>> 를 보고 아래 <<분류기준>>에 따라 어디에 속하는지 <<항목>> 만을 반환하세요.
<<분류기준>>
| 항목 | 설명 |
|------|------|
{context}

<<사용자 질의>>
{query}
"""
    resp = client.chat.completions.create(
        model=os.getenv("CHAT_MODEL", "gpt-4.1-nano"),
        messages=[
            {"role": "user", "content": message}
        ],
    )
    return resp.choices[0].message.content

# 제품 벡터DB 의 데이터 중 정리필요한 항목들 ########################
def __get_water_saved(x:dict):
    used_l  = x.get('water_used_l',0)
    saved_l = x.get('water_saved_l',0)
    sum_l   = used_l + saved_l
    if sum_l != 0: return f"{saved_l * 100 / sum_l:.2f} %"
    else:          return f"{saved_l:.2f} L"
def __get_co2_saved(x:dict):
    used_l  = x.get('co2_used_kg',0)
    saved_l = x.get('co2_saved_kg',0)
    sum_l   = used_l + saved_l
    if sum_l != 0: return f"{saved_l * 100 / sum_l:.2f} %"
    else:          return f"{saved_l:.2f} L"
def __get_merged_material(meta:dict):
    materials = json.loads(meta.get("material"))
    materials = [f"{m['percent']}% {m['material']}" for m in materials]
    return ", ".join(materials)

# 제품 벡터DB 조회 ##################################################
def __get_products_from_vdb(query:str, top_k=3, filter=None):
    client = OpenAI()
    embed = client.embeddings.create(model=os.getenv("EMBED_MODEL"), input=query).data[0].embedding

    index   = os.getenv("PINECONE_INDEX_PRODUCT")
    api_key = os.getenv("PINECONE_API_KEY")
    pc      = Pinecone(api_key=api_key)
    index   = pc.Index(index)
    matches = index.query(vector=embed, top_k=top_k, filter=filter, include_metadata=True)["matches"]

    products = []
    for match in matches:
        meta = match["metadata"]
        products.append({
            "id"          : int(meta.get("id")),
            "name"        : meta.get("name"),
            "category"    : meta.get("category"),
            "price"       : f"{meta.get('currency')} {meta.get('price')}",
            "image"       : meta.get("image_url"),
            "color"       : meta.get("color"),
            "color_detail": meta.get("color_detail"),
            "material"    : __get_merged_material(meta),
            "url"         : meta.get("url"),
            "saved_water" : __get_water_saved(json.loads(meta.get("water_saved_l"))),
            "saved_co2"   : __get_co2_saved(json.loads(meta.get("co2_saved_kg"))),
            "url"         : meta.get("url"),
            "spec"        : meta.get("spec"),
        })
    return products

# 스타일링 벡터DB 조회
def __get_styles_from_vdb(query:str, top_k=10) -> list[dict]:
    client = OpenAI()
    embed = client.embeddings.create(model=os.getenv("EMBED_MODEL"), input=query).data[0].embedding

    index   = os.getenv("PINECONE_INDEX_NAME")
    api_key = os.getenv("PINECONE_API_KEY")
    pc      = Pinecone(api_key=api_key)
    index   = pc.Index(index)
    matches = index.query(vector=embed, top_k=top_k, namespace="transcripts-kr", include_metadata=True)["matches"]

    styles = []
    for match in matches:
        meta = match["metadata"]
        styles.append({
            "occation"    : meta.get("occasion"),
            "season"      : meta.get("season"),
            "section"     : meta.get("section"),
            "snippet"     : meta.get("snippet"),
        })
    return styles
def __ask_style_sheet(client, query, styles:list[dict]):
    msg = f"""
당신은 재활용소재로 패션제품을 만드는 회사에서 고문으로 일하고 있는 친환경 활동가이자 패션과 스타일링의 전문가입니다.
<<스타일링 팁>> 을 사용하여 <<사용자 질의>> 에 어울리는 5개의 스타일을 추천해야 합니다.
1. 답변에서 인사말/당신을 소개하는 문구/불필요한 미사여구는 제외합니다.

<<사용자 질의>>
{query}

<<스타일링 팁>>
{json.dumps(styles)}

아래와 같이 스타일링 표를 구성한 후,
```markdown
| 스타일                | score | 상의/하의 조합                                                                      | 특징                                                    |
|:----------------------|---|:------------------------------------------------------------------------------------|:--------------------------------------------------------|
| 미니멀 (깔끔, 심플)   |0.4987| 화이트 셔츠/블랙 슬랙스, 솔리드 티셔츠/그레이 와이드 팬츠, 슬림핏 니트/H라인 스커트 | 패턴 최소화·모노톤 중심, 실루엣 강조, 액세서리 최소     |
| 캐주얼 (편안, 데일리) |0.4811| 루즈핏 맨투맨/청바지, 스트라이프 티셔츠/치노 팬츠, 데님 셔츠/조거 팬츠              | 활동성 중심, 자연 소재(코튼·데님), 컬러 포인트로 경쾌함 |
```

이 내용을 그대로 아래처럼 json 으로 바꿔서 반환합니다.
1. 스타일에 상의/하의 조합이 여러개씩 있다면 상의/하의를 조합마다 json 객체로 만들어야 합니다.
2. color 와 category 는 영어 소문자 단어 하나만 허용합니다.
2.1. 예를 들어 color 의 경우, dark brown 은 두 단어이므로 brown 으로만 표시합니다.
2.2  예를 들어 category 의 경우, wide pants 는 두 단어이므로 pants 로만 표시합니다. 
3. look_style, score, top, bottom, color, category, desc 외에 다른 key 는 허용하지 않습니다.
""" + """
[
	// color 와 category 가 있는 경우
	{{"look_style" : "깔끔한 미니멀", "score":0.4987, "top" : {{"color":"white", "category":"shirt"}}, "bottom" : {{"color":"black", "category":"slacks"}}, "desc" : "패턴 최소화·모노톤 중심, 실루엣 강조, 액세서리 최소"}},
	// color 나 category 가 없는 경우
	{{"look_style" : "깔끔한 미니멀", "score":0.4987, "top" : {{"color":"white", "category":"shirt"}}, "bottom" : {{"color":"black", "category":"slacks"}}, "desc" : "패턴 최소화·모노톤 중심, 실루엣 강조, 액세서리 최소"}},
	{{"look_style" : "깔끔한 미니멀", "score":0.4987, "top" : {{"desc":"solid t-shirt"}}, "bottom" : {{"color":"grey", "category":"pants"}}, "desc" : "패턴 최소화·모노톤 중심, 실루엣 강조, 액세서리 최소"}},
	{{"look_style" : "데일리 캐주얼", "score":0.4811, "top" : {{"desc":"denim shirt"}}, "bottom" : {{"desc":"jogger pants"}}, "desc" : "활동성 중심, 자연 소재(코튼·데님), 컬러 포인트로 경쾌함"}},
]
"""
    resp = client.chat.completions.create(
        model=os.getenv("CHAT_MODEL", "gpt-4.1-nano"),
        messages=[
            {"role": "user", "content": msg}
        ],
    )
    return json.loads(resp.choices[0].message.content)


# 사용자 질의 목적에 따른 동작 수행 #################################
def __material_explain(client, query:str, **kwargs):
    msg_system = f"""
당신은 재활용소재로 패션제품을 만드는 회사에서 고문으로 일하고 있는 친환경 활동가입니다.
환경을 소중히 생각하기 때문에 친환경소재에 대해 빠삭하게 알고 있고
사람들에게 친환경 활동과 친환경소재의 특성/장단점 및 환경에 미치는 영향에 대해 친절하게 설명해야 합니다.  
친환경 소재가 아닌 일반 소재의 경우에도 친환경소재와 비교하여 설명함으로써 사람들에게 친환경 활동에 관심을 갖도록 유도해야 합니다.
1. 답변에서 인사말/당신을 소개하는 문구/불필요한 미사여구는 제외합니다.
2. 답변은 bootstrap5 가 적용된 html 로 반환합니다.
3. 답변은 재활용 소재에 대한 내용으로만 한정하여 답변합니다.
4. 답변은 공백과 html tag 및 속성을 제외했을 때 250자 내외로 작성합니다.
"""
    resp = client.chat.completions.create(
        model=os.getenv("CHAT_MODEL", "gpt-4.1-nano"),
        messages=[
            {"role": "system", "content": msg_system},
            {"role": "user"  , "content": query}
        ],
    )
    return resp.choices[0].message.content
def __product_find(client, query:str, **kwargs):
    products = __get_products_from_vdb(query)       # list[dict]

    msg_system = f"""
당신은 재활용소재로 패션제품을 만드는 회사에서 고문으로 일하고 있는 친환경 활동가입니다.
JSON 양식으로 구성된 <<제품정보>>의 내용만을 사용하여 사용자의 질의에 답변합니다. 
1. 답변에서 인사말/당신을 소개하는 문구/불필요한 미사여구는 제외합니다.
2. 답변은 bootstrap5 가 적용된 html 로 반환합니다.
3. <<제품정보>> 가 여러개일 경우 좌우로 스크롤할 수 있는 HTML 요소로 만들어서 답변에 추가해야 합니다.
4. 제품에 대한 내용은 카드형식으로 구성하고, 제품이미지/제품이름/제품가격/제품URL 등이 반드시 카드에 표시되어야 합니다.

<<제품정보>>
{json.dumps(products)}
"""
    resp = client.chat.completions.create(
        model=os.getenv("CHAT_MODEL", "gpt-4.1-nano"),
        messages=[
            {"role": "system", "content": msg_system},
            {"role": "user", "content": query}
        ],
    )
    return resp.choices[0].message.content
def __outfit_reco(client, query:str, user_id):
    # styles = __get_styles_from_vdb(query)  # list[dict]
    # styles = __ask_style_sheet(client, query, styles)
    # for style in styles:
    #     top_keys   = style["top"].keys()
    #     top_filter = None
    #     if "color" in top_keys and "category" in top_keys:
    #         top_filter = {"color":{"$eq":[style["top"]["color"]]}, "category":{"$in":[style["top"]["category"]]}}
    #     top = __get_products_from_vdb(query=f"{style['look_style']}, {style['desc']}, {style['top']['desc']}", top_k=1, filter=top_filter)
    #
    #     bot_keys = style["bottom"].keys()
    #     bot_filter = None
    #     if "color" in bot_keys and "category" in bot_keys:
    #         bot_filter = {"color": {"$eq": [style["bottom"]["color"]]}, "category": {"$in": [style["bottom"]["category"]]}}
    #     bottom = __get_products_from_vdb(query=f"{style['look_style']}, {style['desc']}, {style['bottom'].get('desc', '')}", top_k=1, filter=bot_filter)
    #
    #     # 검색결과가 DB 에 있으면 - DB 조회결과 사용
    #     #                  없으면 - DB 에 저장하고 이미지 합성
    #     with SessionLocal() as db:
    #         _query = (
    #             "SELECT search_id FROM search_history_product "
    #             "WHERE  product_id = :bottom AND search_id IN ( "
    #             "    SELECT search_id FROM search_history_product WHERE product_id = :top"
    #             ")"
    #         )
    #         result = db.execute(text(_query), {"top":top[0]["id"], "bottom":bottom[0]["id"]})
    #         row = result.fetchone()  # 리스트로 받기
    #         if row.count() > 0:
    #             search_id = row.search_id
    #         else:
    #             history = db.execute(text("INSERT INTO search_history(user_Id, look_style, look_desc) VALUES(:user_id, :look, :desc)"), {"user_id":user_id, "look":style["look_style"], "desc":style["desc"]})


        # 검색결과를 html 로 만들어서 반환.

    # products = __get_products_from_vdb(query)  # list[dict]
    #
    # msg_system = f"""
    # 당신은 재활용소재로 패션제품을 만드는 회사에서 고문으로 일하고 있는 친환경 활동가입니다.
    # JSON 양식으로 구성된 <<제품정보>>의 내용만을 사용하여 사용자의 질의에 답변합니다.
    # 1. 답변에서 인사말/당신을 소개하는 문구/불필요한 미사여구는 제외합니다.
    # 2. 답변은 bootstrap5 가 적용된 html 로 반환합니다.
    # 3. <<제품정보>> 가 여러개일 경우 좌우로 스크롤할 수 있는 HTML 요소로 만들어서 답변에 추가해야 합니다.
    # 4. 제품에 대한 내용은 카드형식으로 구성하고, 제품이미지/제품이름/제품가격/제품URL 등이 반드시 카드에 표시되어야 합니다.
    #
    # <<제품정보>>
    # {json.dumps(products)}
    # """
    # resp = client.chat.completions.create(
    #     model=os.getenv("CHAT_MODEL", "gpt-4.1-nano"),
    #     messages=[
    #         {"role": "system", "content": msg_system},
    #         {"role": "user", "content": query}
    #     ],
    # )
    # return resp.choices[0].message.content
    return "우선은 아직 작업중임."
def __cert_verify(client, query:str, **kwargs):
    msg_system = f"""
당신은 재활용소재로 패션제품을 만드는 회사에서 고문으로 일하고 있는 친환경 활동가입니다.
친환경 인증과 관련된 질문에만 답변합니다.
당신이 알고있는 수준으로만 답변하고, 일체의 거짓이나 꾸밈이 없어야 하며, 당신이 모르는 정보라면 모른다고 답변합니다.
1. 답변에서 인사말/당신을 소개하는 문구/불필요한 미사여구는 제외합니다.
2. 답변은 공백을 제외했을 때 250자 내외로 작성합니다.
"""
    resp = client.chat.completions.create(
        model=os.getenv("CHAT_MODEL", "gpt-4.1-nano"),
        messages=[
            {"role": "system", "content": msg_system},
            {"role": "user", "content": query}
        ],
    )
    return resp.choices[0].message.content
def __fallback(client, query:str, **kwargs):
    msg_system = f"""
당신은 재활용소재로 패션제품을 만드는 회사에서 고문으로 일하고 있는 친환경 활동가입니다.
재활용소재나 재활용소재로 만든 패션제품과 같은 친환경 질문에만 답변합니다.  
단, 사용자의 질문이 재활용소재나 재활용소재로 만든 패션제품과 같은 친환경 질문에 해당하지 않더라도
사용자의 기분이 상하지 않게 거절하면서 동시에 사용자가 재활용소재나 재활용소재로 만든 패션제품과 같은 친환경 정보를 질문할 수 있도록
부드럽게 유도하는 답변을 반환해야합니다.
1. 답변에서 인사말/당신을 소개하는 문구/불필요한 미사여구는 제외합니다.
2. 답변은 공백을 제외했을 때 250자 내외로 작성합니다.
"""
    resp = client.chat.completions.create(
        model=os.getenv("CHAT_MODEL", "gpt-4.1-nano"),
        messages=[
            {"role": "system", "content": msg_system},
            {"role": "user"  , "content": query}
        ],
    )
    return resp.choices[0].message.content

INTENTS = {
    "material_explain" : {
        "description" : "제품의 소재/그 소재의 장단점/그 소재가 환경에 미치는 영향과 같이, 소재나 환경정보를 묻는 질문인 경우.",
        "process"     : __material_explain
    },
    "product_find" : {
        "description" : "의류제품에 대한 질문인 경우.",
        "process"     : __product_find
    },
    "outfit_reco" : {
        "description" : "스타일링 추천을 요청하는 경우.",
        "process"     : __outfit_reco
    },
    "cert_verify" : {
        "description" : "GRS/RCS 등의 식별자를 기반으로 인증 사실에 대한 검증을 요청하는 경우.",
        "process"     : __cert_verify
    },
    "fallback" : {
        "description" : "material_explain, product_find, outfit_reco, cert_verify 로 분류되지 않는 경우.",
        "process"     : __fallback
    }
}
@router.post("/ask")
def api_ask(param:dict = BODY_EXAMPLE):
    query   = param["query"]
    user_id = param.get("user_id")
    ############################################# 1 질의를 사전에 정해놓은 분류대로 나누기
    client = OpenAI()
    intent = __distinguish(client, query)
    print(f"{intent=}")

    ############################################# 2 분류에 맞게 사용자 질의를 재정의하여 데이터 전달하기
    result = INTENTS[intent]["process"](client, query, user_id=user_id)
    print(result)

    ############################################# 3 각 흐름에 맞게 동작 후, 분류에 따라 결과 반환하기
    return {
        "intent": intent,
        "result": result
    }


