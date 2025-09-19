import json
import os
import threading
import uuid
import html as _html
from typing import Literal

from fastapi import APIRouter, Body
from fastapi.responses import HTMLResponse
from pinecone import Pinecone
from sqlalchemy import text

from S3.add_image_by_llm import build_prompt, generate_model_wearing_refs, S3Uploader
from app import models
from app.database import SessionLocal
from app.clients.runpod_api import RunpodTTSClient, summarize_p_text_via_llm

#################################################### FastAPI api Routing
router = APIRouter(prefix="/api", tags=["API"], responses={404: {"description": "Not found"}} )

#################################################### 전역 변수 선언
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from app.simple_llm import SimpleChatLLM

simple_user_llm = SimpleChatLLM()
embedding       = OpenAIEmbeddings(model=os.getenv("EMBED_MODEL"))
pinecone        = Pinecone(os.getenv("PINECONE_API_KEY"))
index_product   = pinecone.Index(os.getenv("PINECONE_INDEX_PRODUCT"))
index_style     = pinecone.Index(os.getenv("PINECONE_INDEX_NAME"))

tts_client = RunpodTTSClient()


#################################################### 기능별 프롬프트 선언
prompt_intent_routing    = PromptTemplate.from_template("""
<<사용자 질의>> 를 보고 아래 <<분류기준>>에 따라 어디에 속하는지 <<항목>> 만을 반환하세요.
<<분류기준>>
| 항목 | 설명 |
|------|------|
{context}

<<사용자 질의>>
{query}
""")    # 사용자가 뭘 원하는지 파악
prompt_style_compose     = PromptTemplate.from_template("""
당신은 재활용소재로 패션제품을 만드는 회사에서 고문으로 일하고 있는 친환경 활동가이자 패션과 스타일링의 전문가입니다.
<<스타일링 팁>> 만 사용하여 <<사용자 질의>> 에 어울리는 5개의 스타일을 추천해야 합니다.
1. 답변에서 인사말/당신을 소개하는 문구/불필요한 미사여구는 제외합니다.
2. <<스타일링 팁>> 을 <<스타일링 표>> 와 같이 구성한 뒤, <<JSON 양식 규칙>>에 맞게 변경합니다. 

<<사용자 질의>>
{query}

<<스타일링 팁>>
{style_tips}

<<스타일링 표>>
```markdown
| 스타일                | score  | 상의/하의 조합                                                                      | 특징                                                    |
|:----------------------|--------|:------------------------------------------------------------------------------------|:--------------------------------------------------------|
| 미니멀 (깔끔, 심플)   | 0.4987 | 화이트 셔츠/블랙 슬랙스, 솔리드 티셔츠/그레이 와이드 팬츠, 슬림핏 니트/H라인 스커트 | 패턴 최소화·모노톤 중심, 실루엣 강조, 액세서리 최소     |
| 캐주얼 (편안, 데일리) | 0.4811 | 루즈핏 맨투맨/청바지, 스트라이프 티셔츠/치노 팬츠, 데님 셔츠/조거 팬츠              | 활동성 중심, 자연 소재(코튼·데님), 컬러 포인트로 경쾌함 |
```

<<JSON 양식 규칙>>
1. 스타일에 상의/하의 조합이 여러개씩 있다면 상의/하의를 조합마다 json 객체로 만들어야 합니다.
1.1. 예를 들어, | 미니멀 (깔끔, 심플)   | 0.4987 | 화이트 셔츠/블랙 슬랙스, 솔리드 티셔츠/그레이 와이드 팬츠, 슬림핏 니트/H라인 스커트 | 패턴 최소화·모노톤 중심, 실루엣 강조, 액세서리 최소     | 는
{{"look_style" : "깔끔한 미니멀", "score":0.4987, "top" : {{"color":"white", "category":"shirt", "desc":"white shirt"}}, "bottom" : {{"color":"black", "category":"slacks", "desc":"black slacks"}}, "desc" : "패턴 최소화·모노톤 중심, 실루엣 강조, 액세서리 최소"}},
{{"look_style" : "깔끔한 미니멀", "score":0.4987, "top" : {{"desc":"solid t-shirt"}}, "bottom" : {{"color":"grey", "category":"pants", "desc":"grey wide pants"}}, "desc" : "패턴 최소화·모노톤 중심, 실루엣 강조, 액세서리 최소"}},
{{"look_style" : "깔끔한 미니멀", "score":0.4987, "top" : {{"desc":"slimfit knit"}}, "bottom" : {{"desc":"H-line skirt"}}, "desc" : "패턴 최소화·모노톤 중심, 실루엣 강조, 액세서리 최소"}}
와 같이, 모든 조합에 대하여 json 형식으로 추출해야 합니다.
2. color 와 category 는 영어 소문자 단어 하나만 허용합니다.
2.1. 예를 들어 color 의 경우, dark brown 은 두 단어이므로 brown 으로만 표시합니다.
2.2  예를 들어 category 의 경우, wide pants 는 두 단어이므로 pants 로만 표시합니다. 
3. look_style, score, top, bottom, color, category, desc 외에 다른 key 는 허용하지 않습니다.
[
	// color 와 category 가 있는 경우
	{{"look_style" : "깔끔한 미니멀", "score":0.4987, "top" : {{"color":"white", "category":"shirt", "desc":"white shirt"}}, "bottom" : {{"color":"black", "category":"slacks", "desc":"black slacks"}}, "desc" : "패턴 최소화·모노톤 중심, 실루엣 강조, 액세서리 최소"}},
	// color 나 category 가 없는 경우
    {{"look_style" : "깔끔한 미니멀", "score":0.4987, "top" : {{"desc":"solid t-shirt"}}, "bottom" : {{"color":"grey", "category":"pants", "desc":"grey wide pants"}}, "desc" : "패턴 최소화·모노톤 중심, 실루엣 강조, 액세서리 최소"}},
    {{"look_style" : "깔끔한 미니멀", "score":0.4987, "top" : {{"desc":"slimfit knit"}}, "bottom" : {{"desc":"H-line skirt"}}, "desc" : "패턴 최소화·모노톤 중심, 실루엣 강조, 액세서리 최소"}}
]
""")    # 스타일 조합을 짜달라고 요청
prompt_style_suggest     = PromptTemplate.from_template("""
당신은 재활용소재로 패션제품을 만드는 회사에서 고문으로 일하고 있는 친환경 활동가이자 패션과 스타일링의 전문가입니다.
이전에 당신은 <<사용자 질의>> 에 대한 답변으로 <<룩 정보>> 를 찾아냈고,
이제는 사용자에게 이 <<룩 정보>> 의 내용을 잘 정리하여 그대로 추천합니다.
<<룩 정보>>와 <<사용자 질의>> 에 어울리는 답변을 생성하여 반환합니다.

1. 답변에서 인사말/당신을 소개하는 문구는 제외합니다.
2. 반드시 <<html 양식>>을 지켜야 합니다. 다른 요소나 클래스를 임의로 넣지 마세요.

<<룩 정보>>
{styles_info}

<<html 양식>>
1. 반드시 아래 코드블럭의 양식을 지켜야 합니다.
1.1. 단 한 개의 div.product-container 가 여러개의 div.product-card를 감싼 형태여야만 합니다.
1.2. div.product-container 안에는 <<룩 정보>>의 룩 개수만큼의 div.product-card 가 들어갑니다.
```html
<div class="product-container">
    <div class="product-card" onclick="selectProduct(this)" data-id="(여기에 룩의 id 가 들어갑니다.)">
        <div class="product-image" style="background-image: url('(여기에 룩의 이미지 URL 이 들어갑니다.)');"></div>
            <div class="heart-icon unliked" onclick="toggleHeart(this)">🤍</div>
            <div class="product-info">
            <div class="product-title">(여기에 룩의 이름이 들어갑니다.)</div>
            <div class="product-description">(여기에 룩의 설명이 들어갑니다.)</div>
        </div>
    </div>
</div>
```

<<사용자 질의>>
{query}
""")    # 최종 추천 스타일들을 html 로 반환

prompt_material_explain  = PromptTemplate.from_template("""
당신은 재활용소재로 패션제품을 만드는 회사에서 고문으로 일하고 있는 친환경 활동가입니다.
환경을 소중히 생각하기 때문에 친환경소재에 대해 빠삭하게 알고 있고
사람들에게 친환경 활동과 친환경소재의 특성/장단점 및 환경에 미치는 영향에 대해 친절하게 설명해야 합니다.  
친환경 소재가 아닌 일반 소재의 경우에도 친환경소재와 비교하여 설명함으로써 사람들에게 친환경 활동에 관심을 갖도록 유도해야 합니다.
1. 답변에서 인사말/당신을 소개하는 문구/불필요한 미사여구는 제외합니다.
2. 답변은 bootstrap5 가 적용된 html 로 반환합니다.
3. 답변은 재활용 소재에 대한 내용으로만 한정하여 답변합니다.
4. 답변은 공백과 html tag 및 속성을 제외했을 때 250자 내외로 작성합니다.
""")    # 소재관련       질문에 대한 답변 생성
prompt_product_find      = PromptTemplate.from_template("""
당신은 재활용소재로 패션제품을 만드는 회사에서 고문으로 일하고 있는 친환경 활동가입니다.
JSON 양식으로 구성된 <<제품정보>>의 내용만을 사용하여 사용자의 질의에 답변합니다. 
1. 답변에서 인사말/당신을 소개하는 문구/불필요한 미사여구는 제외합니다.
2. 반드시 <<html 양식>>을 지켜야 합니다. 다른 요소나 클래스를 임의로 넣지 마세요.

<<html 양식>>
1. 반드시 아래 코드블럭의 양식을 지켜야 합니다.
1.1. 단 한 개의 div.product-container 가 여러개의 div.product-card를 감싼 형태여야만 합니다.
1.2. div.product-container 안에는 <<제품정보>>의 제품 개수만큼의 div.product-card 가 들어갑니다.
```html
<div class="product-container">
    <div class="product-card" onclick="selectProduct((제품이 상의면 'top', 하의면 'bottom'이 들어갑니다.))" data-id="(여기에 제품의 id 가 들어갑니다.)">
        <div class="product-image" style="background-image: url('(여기에 제품의 이미지 URL 이 들어갑니다.)');"></div>
            <div class="product-info">
            <div class="product-title">(여기에 제품의 이름이 들어갑니다.)</div>
            <div class="product-description">(여기에 제품의 설명이 들어갑니다.)</div>
        </div>
    </div>
</div>
```

<<제품정보>>
{products_info}
""")    # 의류제품관련   질문에 대한 답변 생성
prompt_cert_verify       = PromptTemplate.from_template("""
당신은 재활용소재로 패션제품을 만드는 회사에서 고문으로 일하고 있는 친환경 활동가입니다.
친환경 인증과 관련된 질문에만 답변합니다.
당신이 알고있는 수준으로만 답변하고, 일체의 거짓이나 꾸밈이 없어야 하며, 당신이 모르는 정보라면 모른다고 답변합니다.
1. 답변에서 인사말/당신을 소개하는 문구/불필요한 미사여구는 제외합니다.
2. 답변은 공백을 제외했을 때 250자 내외로 작성합니다.
""")    # 친환경인증관련 질문에 대한 답변 생성
prompt_fallback          = PromptTemplate.from_template("""
당신은 재활용소재로 패션제품을 만드는 회사에서 고문으로 일하고 있는 친환경 활동가입니다.
재활용소재나 재활용소재로 만든 패션제품과 같은 친환경 질문에만 답변합니다.  
단, 사용자의 질문이 재활용소재나 재활용소재로 만든 패션제품과 같은 친환경 질문에 해당하지 않더라도
사용자의 기분이 상하지 않게 거절하면서 동시에 사용자가 재활용소재나 재활용소재로 만든 패션제품과 같은 친환경 정보를 질문할 수 있도록
부드럽게 유도하는 답변을 반환해야합니다.
1. 답변에서 인사말/당신을 소개하는 문구/불필요한 미사여구는 제외합니다.
2. 답변은 공백을 제외했을 때 250자 내외로 작성합니다.
""")    # 의류제품관련   질문에 대한 답변 생성


#################################################### 제품 벡터DB 조회
def __parse_water_saved(x:dict):
    used_l  = x.get('water_used_l',0)
    saved_l = x.get('water_saved_l',0)
    sum_l   = used_l + saved_l
    if sum_l != 0: return f"{saved_l * 100 / sum_l:.2f} %"
    else:          return f"{saved_l:.2f} L"
def __parse_co2_saved(x:dict):
    used_l  = x.get('co2_used_kg',0)
    saved_l = x.get('co2_saved_kg',0)
    sum_l   = used_l + saved_l
    if sum_l != 0: return f"{saved_l * 100 / sum_l:.2f} %"
    else:          return f"{saved_l:.2f} L"
def __parse_merged_material(meta:dict):
    materials = json.loads(meta.get("material"))
    materials = [f"{m['percent']}% {m['material']}" for m in materials]
    return ", ".join(materials)
def __get_products_from_vdb(query:str, top_k=3, filter=None):
    vector  = embedding.embed_query(query)
    matches = index_product.query(vector=vector, top_k=top_k, filter=filter, include_metadata=True)["matches"]

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
            "material"    : __parse_merged_material(meta),
            "url"         : meta.get("url"),
            "saved_water" : __parse_water_saved(json.loads(meta.get("water_saved_l"))),
            "saved_co2"   : __parse_co2_saved(json.loads(meta.get("co2_saved_kg"))),
            "spec"        : meta.get("spec"),
        })
    return products


#################################################### 스타일링 벡터DB 조회
def __get_styles_from_vdb(query:str, top_k=10) -> list[dict]:
    vector  = embedding.embed_query(query)
    matches = index_style.query(vector=vector, top_k=top_k, namespace="transcripts-kr", include_metadata=True)["matches"]

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


#################################################### 스타일링 정보를 LLM 으로 가공
def __ask_style_sheet(query, styles:list[dict]):
    message = prompt_style_compose.format(query=query, style_tips=json.dumps(styles))
    result  = simple_user_llm(message)
    # print(f"{result=}")

    return json.loads(result)


#################################################### S3 설정 및 이미지 합성
AWS_S3_BUCKET  = os.getenv("AWS_S3_BUCKET_NAME")
AWS_REGION     = os.getenv("AWS_S3_REGION", "ap-northeast-2")
AWS_S3_PREFIX  = os.getenv("AWS_S3_PREFIX", "tryon")
def __ask_image_composition(look_style, image_urls):
    prompt = build_prompt(image_urls, look_style)
    _uuid  = uuid.uuid1()
    out_name = f"look_{_uuid}.png"
    try:
        uploader = S3Uploader(
            bucket=AWS_S3_BUCKET,
            region=AWS_REGION,
            prefix=AWS_S3_PREFIX,
            public_read=True,
            presign_expire=0
        )
        s3_key = uploader.build_key(out_name, str(_uuid))
        png_bytes = generate_model_wearing_refs(image_urls, prompt)
        res = uploader.put_bytes(png_bytes, s3_key, content_type="image/png")
        return res.get('url')
    except Exception as e:
        print(e)
        print(f"[ERROR] {look_style} 스타일 룩 이미지 생성 실패.\n{', '.join(image_urls)}")
        return ""
def __ask_look_list(query, looks):
    message = prompt_style_suggest.format(query=query, styles_info=json.dumps(looks))

    return simple_user_llm(message)


####################################################
def __get_product_info_for_top_bottom(style, key:Literal["top", "bottom"], look_style, look_desc):
    target = style[key]
    keys   = target.keys()
    filter = None
    if "color" in keys and "category" in keys:
        filter = {"color":{"$eq":target["color"]}, "category":{"$in":[target["category"]]}}
    return __get_products_from_vdb(query=f"{look_style}, {look_desc}, {target.get('desc','')}", top_k=1, filter=filter)
def __style_thread(user_id, style:dict, context:list):
    look_style = style["look_style"]
    look_desc  = style['desc']
    top    = __get_product_info_for_top_bottom(style, "top"   , look_style, look_desc)
    bottom = __get_product_info_for_top_bottom(style, "bottom", look_style, look_desc)

    # 검색결과가 DB 에 있으면 - DB 조회결과 사용
    #                  없으면 - DB 에 저장하고 이미지 합성
    if top and bottom:
        top    = top[0]
        bottom = bottom[0]
        with SessionLocal() as db:
            _query = (
                "SELECT search_id FROM search_history_product "
                "WHERE  product_id = :bottom AND search_id IN ( "
                "    SELECT search_id FROM search_history_product WHERE product_id = :top"
                ")"
            )
            result = db.execute(text(_query), {"top":top["id"], "bottom":bottom["id"]})
            row = result.fetchone()
            # print(f"{row.id if row else row}")
            if not row :
                #TODO 이미지 합성 실패 하면 빈 문자열만 나옴. 보완 필요.
                look_img_url = __ask_image_composition(look_style, [top["image"], bottom["image"]])

                row = models.SearchHistory(user_Id=user_id, look_style=look_style, look_desc=look_desc, look_img_url=look_img_url)
                db.add(row)
                db.commit()
                db.refresh(row)
                # print(f"{look_style} image = {look_img_url}\n{row.id=}\n========================")

        context.append({
            "look_id"    : row.id,
            "look_style" : look_style,
            "look_image" : row.look_img_url,
            "look_desc"  : look_desc
        })


#################################################### 사용자 질의 목적에 따른 동작 수행
def __material_explain(query:str, **kwargs):
    msg_system = prompt_material_explain.format()

    return simple_user_llm(query, [{"role": "system", "content": msg_system}])
def __product_find(query:str, **kwargs):
    products   = __get_products_from_vdb(query)       # list[dict]
    msg_system = prompt_product_find.format(products_info=json.dumps(products))

    return simple_user_llm(query, [{"role": "system", "content": msg_system}])
def __outfit_reco(query:str, user_id, **kwargs):
    styles = __get_styles_from_vdb(query, top_k=20)     # list[dict] :: 벡터DB 에서 스타일 정보 조회
    styles = __ask_style_sheet(query, styles)           # list[dict] :: LLM 으로 스타일양식 정리

    context = []
    threads = [threading.Thread(target=__style_thread, args=(user_id, style, context)) for style in styles]

    for thread in threads:  thread.start()
    for thread in threads:  thread.join()

    # 검색결과를 html 로 만들어서 반환.
    return __ask_look_list(query, context)
def __cert_verify(query:str, **kwargs):
    msg_system = prompt_cert_verify.format()
    result_txt = simple_user_llm(query, [{"role": "system", "content": msg_system}])
    tts_maker = RunpodTTSClient()
    tts = tts_maker.run_tts(text=result_txt, persona="2")
    return f"{result_txt}<br><audio src='{tts['s3_url']}'></audio>"
def __fallback(query:str, **kwargs):
    msg_system  = prompt_fallback.format()

    return simple_user_llm(query, [{"role": "system", "content": msg_system}])


#################################################### 사용자 질의의 목적 구분
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
def __distinguish(query:str):
    context = "\n".join([f"| {catg} | {info['description']} |" for catg,info in INTENTS.items()])
    message = prompt_intent_routing.format(context=context, query=query)

    return simple_user_llm(message)


#################################################### Intent Routing 파라미터 예시
BODY_EXAMPLE:dict = Body(None, examples=[{
        "explain" : "소재 관련 질문 예시( material_explain )",
        "query": "폴리에스터는 왜 재활용하는거야?",
        "user_id": 1
    },
    {
        "explain" : "의류 제품 관련 질문 예시( product_find )",
        "query": "재활용소재로 만든 옷들은 뭐가 있어?",
        "user_id": 1
    },
    {
        "explain" : "스타일링 관련 질문 예시( outfit_reco )",
        "query": "날씨가 슬슬 선선해지는데, 어떤 옷을 입을까?",
        "user_id": 1
    },
    {
        "explain" : "친환경인증 관련 질문 예시( cert_verify )",
        "query": "RCS 인증마크는 뭐하는 녀석이야?",
        "user_id": 1
    },
    {
        "explain" : "기타 fallback 예시( fallback )",
        "query": "날씨가 슬슬 선선해지는데, 감기에 좋은 음식이 뭐야?",
        "user_id": 1
    }]
)

####################################################
@router.post("/ask")
def api_ask(param:dict = Body(None, examples=[{
        "query": "폴리에스터는 왜 재활용하는거야?",
        "user_id": 1,
        "persona":"2"
    }])):
    query   = param["query"]
    user_id = param.get("user_id")
    persona = param.get("persona")

    intent  = __distinguish(query)                  # 질의를 사전에 정해놓은 분류대로 나누기
    process = INTENTS[intent]["process"]            # 수행할 함수 확인
    result  = process(query, user_id=user_id)       # 분류에 맞게 사용자 질의를 재정의하여 데이터 전달하기

    return HTMLResponse(content=result, status_code=200)
