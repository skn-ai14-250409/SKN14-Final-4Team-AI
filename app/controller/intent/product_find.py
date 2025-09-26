import json
import os
import random
import threading

from langchain_core.prompts import PromptTemplate
from sqlalchemy import text
from starlette.responses import HTMLResponse

from S3.add_image_by_llm import S3Uploader
from .IntentBase import IntentBase
from ...database import SessionLocal


class ProductFind(IntentBase):

    def __init__(self):
        super().__init__()

        self._prompt_step1   = PromptTemplate.from_template("""
이전 대화내역들과 사용자의 질의를 비교하여, 사용자가 어떤 제품을 찾고자 하는지 판별하세요. 
1. 답변은 반드시 <<출력양식>> 을 따라 JSON 으로만 반환해야 합니다.
2. 사용자의 질의내용이 이전 대화내역중에 있는 스타일이나 제품을 찾고있다고 판단되는 경우 related 타입으로 답변을 반환합니다.
3. <<출력양식>> 에서 상의/하의의 color는 {colors} 중 하나로만 표현해야 합니다. 영어로만 넣어주세요.
3.1. 해당하는 항목이 없다면 가장 무난한 색상으로 넣어주세요.
4. <<출력양식>> 에서 상의/하의의 category 는 {categories} 중 하나로만 표현해야 합니다. 영어로만 넣어주세요.
5. 답변에 인사말/당신을 소개하는 문구/불필요한 미사여구는 제외합니다.

<<출력양식>>
- 사용자가 이전 대화내역에서 검색했던 스타일이나 제품을 찾고있다고 판단되는 경우
{{
    "type"   : "related",
    "query"   : "패턴을 최소화하고 모노톤 중심, 실루엣을 강조하고 액세서리는 최소화한 깔끔하고 심플한 미니멀 스타일. (화이트 셔츠 / 블랙 슬랙스) , (솔리드 티셔츠 / 그레이 와이드 팬츠), (슬림핏 니트 / H라인 스커트)"
    "styles" : [
        {{
            "name"  : "깔끔하고 심플한 미니멀",
            "desc"  : "패턴 최소화·모노톤 중심, 실루엣 강조, 액세서리 최소",
            "combi" : [
                {{ "top" : {{ "name" : "화이트 셔츠"  , "color":"(색상을 넣어주세요.)", "category" : "(category를 넣어주세요.)"}}  , "bottom" : {{ "name" : "블랙 슬랙스"     , "color":"(색상을 넣어주세요.)", "category" : "(category를 넣어주세요.)" }} }},
                {{ "top" : {{ "name" : "솔리드 티셔츠", "color":"(색상을 넣어주세요.)", "category" : "(category를 넣어주세요.)"}}, "bottom" : {{ "name" : "그레이 와이드 팬츠", "color":"(색상을 넣어주세요.)", "category" : "(category를 넣어주세요.)" }} }},
                {{ "top" : {{ "name" : "슬림핏 니트"  , "color":"(색상을 넣어주세요.)", "category" : "(category를 넣어주세요.)"}}  , "bottom" : {{ "name" : "H라인 스커트"    , "color":"(색상을 넣어주세요.)", "category" : "(category를 넣어주세요.)" }} }},
            ]
        }}
    ]
}}
- 사용자가 이전 대화내역과 관련없는 제품을 찾고있다고 판단되는 경우
{{
    "type"  : "original",
    "query" : "(여기에 사용자의 질의를 바꾸지말고 그대로 넣어주세요.)"
}}
""")    # 이전 스타일검색 결과의 제품을 찾는지, 그냥 제품을 찾는지 파악 요청

        self.all_colors     = self.get_all_product_data("color")
        self.all_categories = self.get_all_product_data("category")

        self.s3_uploader = S3Uploader(
            bucket=os.getenv("AWS_S3_BUCKET_NAME"),
            region=os.getenv("AWS_S3_REGION", "ap-northeast-2"),
            prefix=os.getenv("AWS_S3_PREFIX", "tryon"),
            public_read=True,
            presign_expire=0
        )

    def get_all_product_data(self, column):
        with SessionLocal() as db:
            _query = f"SELECT DISTINCT({column}) {column} FROM app_product"
            result = db.execute(text(_query))
            rows   = result.fetchall()
            return [row[0] for row in rows]

    def __call__(self, **kwargs):
        # Step1 :: 이전에 물어본 스타일에 대해서 묻는 질문인지, 아예 새로운 제품에 대한 질문인지 판단.
        query   = kwargs.get("query")
        user_id = kwargs.get("user_id")
        ai_id   = kwargs.get("ai_id")
        step1_result = self.ask_llm(query, prompt=self._prompt_step1.format(colors=self.all_colors, categories=self.all_categories), model="gpt-5-mini-2025-08-07", user_id=user_id, ai_id=ai_id)
        step1_result = json.loads(step1_result)

        type      = step1_result['type']
        new_query = step1_result['query']
        print(f"\ntype = {type}")
        # print("\nstep1_result = ")
        # pprint(step1_result)

        # Step2 :: 질의에 해당하는 제품을 찾아 결과 반환.
        if type == "related":
            result   = self.search_product(step1_result["styles"], user_id)
            ment     = self.influencer(json.dumps(step1_result["styles"], ensure_ascii=False), ai_id)
            voice    = self.get_voice(ment, True, ai_id)
            result  += f"<audio controls src={voice}></audio>"
        else:
            products = self.get_products_from_vdb(new_query)
            result   = "".join([self.__prod_to_html2(prod) for prod in products])
            result   = f"""<div class="product-container">{result}</div>"""

        return HTMLResponse(result, status_code=200)


    def _search_product_vdb(self, store, key, q, f):
        products = self.get_products_from_vdb(q, 3, f)
        if products:
            pick = random.choice(products)
            pick = {
                "id"          : pick["id"],
                "name"        : pick["name"],
                # "category"    : pick[""],
                "price"       : pick["price"],
                "image"       : pick["image"],
                # "color"       : pick[""],
                # "color_detail": pick[""],
                # "material"    : pick[""],
                "url"         : pick["url"],
                # "saved_water" : pick[""],
                # "saved_co2"   : pick[""],
                # "spec"        : pick[""],
            }
            store[key] = pick
        else:
            print("VDB에서 검색된 제품 없음", f"{q=}", f"{f=}", sep="\n")

    def search_product(self, styles, user_id):
        threads  = []
        for style in styles:
            query = f"{style['name']}, {style['desc']}"
            for combi in style["combi"]:
                for catg in ["top", "bottom"]:
                    dress  = combi[catg]            # 해당 스타일 조합의 top 또는 bottom
                    filter = {"color":{"$eq":dress["color"]}, "category":{"$in":[dress["category"]]}}
                    thread = threading.Thread(target=self._search_product_vdb, args=(dress, "info", query, filter))
                    thread.start()
                    threads.append(thread)

        for thread in threads:  thread.join()

        styles_html = []
        for style in styles:
            combies_html = []
            for combi in style["combi"]:
                cards_html = ""
                for catg in ["top", "bottom"]:
                    dress  = combi[catg]            # 해당 스타일 조합의 top 또는 bottom
                    cards_html += self.__prod_to_html(dress.get("info"))

                combies_html.append(f"""<div class="product-container">{cards_html}</div>""")
            styles_html.append(f"""<li><div class="text">{style['name']} :: {style['desc']}</div>{'<br/>'.join(combies_html)}</li>""")

        return "".join(styles_html)

    def __prod_to_html(self, info:dict):
        try:
            return """
<div class='product-card' data-id='{id}' data-label="제품정보">
    <a href='{url}' target='_blank'>
       <div class='product-image' style='background-image: url({image});'></div>
        <div class='product-info'>
            <div class='product-title'>{name}</div>
            <div class='product-description'>{price}</div>
        </div>
    </a>
</div>""".format(**info)
        except Exception as e:
            print(e, f"ProductFind.__prod_to_html() :: info = {info}")
            return ""
    def __prod_to_html2(self, info:dict):
        try:
            return """
<div class='product-card' data-id='{id}'>
    <a href='{url}' target='_blank'>
       <div class='product-image' style='background-image: url({image});'></div>
        <div class='product-info'>
            <div class='product-title'>{name}</div>
            <div class='product-description'>{price}</div>
        </div>
    </a>
</div>""".format(**info)
        except Exception as e:
            print(e, f"ProductFind.__prod_to_html() :: info = {info}")
            return ""