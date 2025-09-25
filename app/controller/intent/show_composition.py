import json
import os
import random
import threading
import uuid

from langchain_core.prompts import PromptTemplate
from sqlalchemy import text, select
from starlette.responses import HTMLResponse

from S3.add_image_by_llm import S3Uploader, generate_model_wearing_refs
from .IntentBase import IntentBase
from ... import models
from ...database import SessionLocal
from ...models import SearchHistory, Like


class ShowComposition(IntentBase):

    def __init__(self):
        super().__init__()

        self._prompt_step1   = PromptTemplate.from_template("""
이전 대화내역들과 사용자의 질의를 비교하여, 사용자가 어떤 제품을 합성하고자 하는지 찾아야합니다. 
1. 답변은 반드시 <<출력양식>> 을 따라 JSON 으로만 반환해야 합니다.
2. 사용자의 질의내용이 이전 대화내역중에 있는 스타일이나 제품을 찾고있다고 판단되는 경우 related 타입으로 답변을 반환합니다.
3. 이전 대화내역이 html 로 이뤄져있다면, innerText 와 더불어 태그와 속성에서 url 과 id 값들을 추출해내야 합니다.
4. data-label 이 "합성결과" 인 대화내역은 모두 무시합니다.
5. data-label 이 "제품정보" 인 대화내역을 위주로 검색합니다.
6. 사용자가 스타일을 지칭하지는 않았으나 이전 대화내역에 제품정보가 있다면, 그 제품을 합성하겠다고 판단합니다.
6.1. 사용자가 스타일을 지칭하지는 않았으나 이전 대화내역에 제품정보가 여러개 있다면, 가장 최근 대화의 제품정보가 더 높은 우선순위를 갖습니다.

<<출력양식>>
- 사용자가 이전 대화내역에서 검색했던 스타일이나 제품을 찾고있다고 판단되는 경우
{{
    "type"   : "related",
    "query"   : "패턴을 최소화하고 모노톤 중심, 실루엣을 강조하고 액세서리는 최소화한 깔끔하고 심플한 미니멀 스타일. (화이트 셔츠 / 블랙 슬랙스) , (솔리드 티셔츠 / 그레이 와이드 팬츠), (슬림핏 니트 / H라인 스커트)"
    "target_chat" : "(여기엔 이전대화내역 중에서 사용자 질의와 가장 근접한 대화 내역을 하나 넣어주세요.)",
    "styles" : [
        {{
            "name"  : "깔끔하고 심플한 미니멀",
            "desc"  : "패턴 최소화·모노톤 중심, 실루엣 강조, 액세서리 최소",
            "combi" : [
                {{ "top" : {{ "id" : (여기엔 이 제품의 data-id 값을 넣어주세요.), "name" : "화이트 셔츠"  , "image":"(여기엔 제품의 이미지 URL 을 찾아 넣어주세요.)" }}, "bottom" : {{ "id" : (여기엔 이 제품의 data-id 값을 넣어주세요.), "name" : "블랙 슬랙스"       , "image":"(여기엔 제품의 이미지 URL 을 찾아 넣어주세요.)" }} }},
                {{ "top" : {{ "id" : (여기엔 이 제품의 data-id 값을 넣어주세요.), "name" : "솔리드 티셔츠", "image":"(여기엔 제품의 이미지 URL 을 찾아 넣어주세요.)" }}, "bottom" : {{ "id" : (여기엔 이 제품의 data-id 값을 넣어주세요.), "name" : "그레이 와이드 팬츠", "image":"(여기엔 제품의 이미지 URL 을 찾아 넣어주세요.)" }} }},
                {{ "top" : {{ "id" : (여기엔 이 제품의 data-id 값을 넣어주세요.), "name" : "슬림핏 니트"  , "image":"(여기엔 제품의 이미지 URL 을 찾아 넣어주세요.)" }}, "bottom" : {{ "id" : (여기엔 이 제품의 data-id 값을 넣어주세요.), "name" : "H라인 스커트"      , "image":"(여기엔 제품의 이미지 URL 을 찾아 넣어주세요.)" }} }},
            ]
        }}
    ]
}}
- 사용자가 이전 대화내역과 관련없는 제품을 찾고있다고 판단되는 경우
{{
    "type"  : "original",
    "query" : "(여기에 사용자 질의에 어울릴 제품을 검색할 수 있도록 안내하는 멘트를 생성하여 넣어주세요.)"
}}
""")
        self._prompt_compose = PromptTemplate.from_template("""
1. 포토리얼한 20대 모델 이미지를 생성하고 아래 <<참조 의류>>를 자연스럽게 착용·레이어링한 모습으로 표현해줘.
2. 의상이 여성의류면 여성 모델로, 남성의류면 남성 모델로 생성해줘.
2.1. 여성과 남성 어느쪽인지 불확실할 때는 여성으로 생성해줘.
3. 전반적 스타일은 {style_name} 무드에 맞추고, 실제 착장처럼 핏·주름·광택·그림자·겹침을 자연스럽게 만들고, 왜곡은 최소화해.
4. 배경은 심플한 스튜디오 스타일로.
5. <<참조 의류>> 이미지에 사람이 포함되어있으면 사람은 반드시 무시하고, 생성한 모델을 사용해야되.

<<참조 의류>>
상의 URL : {top_image}
하의 URL : {bottom_image}
""")
        self.s3_uploader   = S3Uploader(
            bucket=os.getenv("AWS_S3_BUCKET_NAME"),
            region=os.getenv("AWS_S3_REGION", "ap-northeast-2"),
            prefix=os.getenv("AWS_S3_PREFIX", "tryon"),
            public_read=True,
            presign_expire=0
        )

    def __call__(self, **kwargs):
        # Step1 :: 이전에 물어본 스타일에 대해서 묻는 질문인지, 아예 새로운 제품에 대한 질문인지 판단.
        query   = kwargs.get("query")
        user_id = kwargs.get("user_id")
        ai_id   = kwargs.get("ai_id")
        step1_result = self.ask_llm(query, prompt=self._prompt_step1.format(), model="gpt-5-mini-2025-08-07", user_id=user_id, ai_id=ai_id)
        step1_result = json.loads(step1_result)
        # print(f"{step1_result=}")

        type      = step1_result['type']
        # Step2 :: 질의에 해당하는 제품을 찾아 결과 반환.
        if type == "related":
            result     = self.search_product(step1_result["styles"], user_id)
            ment       = self.influencer(json.dumps(step1_result["styles"], ensure_ascii=False), ai_id)
            voice      = self.get_voice(ment, True, ai_id)
            result    += f"<audio controls src={voice}></audio>"
        else:
            # products   = self.get_products_from_vdb(new_query)
            # result     = "".join([self.__prod_to_html(prod) for prod in products])
            # result = "임의의 조합으로는 생성이 제한됩니다. 특정 스타일을 먼저 검색해주세요."
            result = step1_result["query"]

        return HTMLResponse(result, status_code=200)
        # return HTMLResponse("시도중", status_code=200)

    def __ask_image_composition(self, name, top_image, bottom_image):
        prompt   = self._prompt_compose.format(style_name=name, top_image=top_image, bottom_image=bottom_image)
        _uuid    = uuid.uuid1()
        out_name = f"look_{_uuid}.png"
        images   = [top_image, bottom_image]
        try:
            s3_key = self.s3_uploader.build_key(out_name, str(_uuid))
            png_bytes = generate_model_wearing_refs(images, prompt)
            res = self.s3_uploader.put_bytes(png_bytes, s3_key, content_type="image/png")
            return res.get('url')
        except Exception as e:
            print(e)
            print(f"[ERROR] {name} 스타일 룩 이미지 생성 실패.\n{images}")
            return ""
    def __save_image(self, user_id:int, name:str, desc:str, top:dict, bottom:dict):
        with SessionLocal() as db:
            top_id    = top.get("id")
            bottom_id = bottom.get("id")
            if top_id and bottom_id:
                _query = (
                    "SELECT search_id FROM search_history_product "
                    "WHERE  product_id = :bottom AND search_id IN ( "
                    "    SELECT search_id FROM search_history_product WHERE product_id = :top"
                    ") LIMIT 1"
                )
                result = db.execute(text(_query), {"top":top_id, "bottom":bottom_id})
                row    = result.fetchone()
                if not row:
                    # print("\nNo record for style.")
                    look_img_url = self.__ask_image_composition(name, top["image"], bottom["image"])
                    row = models.SearchHistory(user_id=user_id, look_style=name, look_desc=desc, look_img_url=look_img_url)
                    db.add(row)
                    db.flush()
                    db.refresh(row)
                    # print(f"look_img_url = {look_img_url}")

                    history_product = models.SearchHistoryProduct(product_id=top_id, search_id=row.id)
                    db.add(history_product)
                    # print(f"top product {top['name']} save.")
                    history_product = models.SearchHistoryProduct(product_id=bottom_id, search_id=row.id)
                    db.add(history_product)
                    # print(f"bottom product {bottom['name']} save.")
                else:
                    stmt = select(SearchHistory).where(SearchHistory.id == row[0])
                    row = db.execute(stmt).scalar_one_or_none()  # 없으면 None

                print(f"look_img_url = {row.look_img_url}")

                stmt = select(Like).where(Like.user_id == user_id, Like.search_id == row.id)
                like = db.execute(stmt).scalar_one_or_none()

                result = {
                    "id"   : row.id,
                    "name" : name,
                    "desc" : desc,
                    "image": row.look_img_url,
                    "url"  : f"/detail/{row.id}",
                    "like" : "unliked" if like is None else "liked"
                }

                db.commit()

        return result
    def __save_image_thread(self, store:list, user_id:int, name:str, desc:str, top:dict, bottom:dict):
        result_html = self.__save_image(user_id, name, desc, top, bottom)
        store.append(result_html)

    def _search_product_vdb(self, store, key, q, f):
        products = self.get_products_from_vdb(q, 3, f)
        pick = random.choice(products)
        pick = {
            "id"          : pick["id"],
            "name"        : pick["name"],
            # "category"    : pick[""],
            "price"       : pick["price"],
            "image"       : pick["image"],
            # "color"       : pick[""],
            # "color_detail": pick[""],
            "url": pick["url"],
            # "material"    : pick[""],
            # "saved_water" : pick[""],
            # "saved_co2"   : pick[""],
            # "spec"        : pick[""],
        }
        store[key] = pick
    def search_product(self, styles, user_id):
        threads   = []
        info_list = []
        for style in styles:
            for combi in style["combi"]:
                thread = threading.Thread(target=self.__save_image_thread, args=(info_list, user_id, style["name"], style["desc"], combi["top"], combi["bottom"]))
                thread.start()
                threads.append(thread)

        for thread in threads:  thread.join()
        styles_html = ''.join([self.__prod_to_html(info) for info in info_list])

        return f"""<div class="product-container">{styles_html}</div>"""

    def __prod_to_html(self, info:dict):
        return """
<div class='product-card' data-id='{id}' data-label="합성결과">
    <a href='{url}' target='_blank'>
        <div class='product-image' style='background-image: url({image});'></div>
        <div class='heart-icon {like}'>🤍</div>
        <div class='product-info'>
            <div class='product-title'>{name}</div>
            <div class='product-description'>{desc}</div>
        </div>
    </a>
</div>""".format(**info)