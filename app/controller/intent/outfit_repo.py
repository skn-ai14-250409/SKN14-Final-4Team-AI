import json

from langchain_core.prompts import PromptTemplate
from starlette.responses import HTMLResponse

from .IntentBase import IntentBase


class OutfitReco(IntentBase):

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

    def __call__(self, **kwargs):
        query       = kwargs['query']
        with_voice  = kwargs.get('with_voice', False)
        user_id     = kwargs.get('user_id', None)
        persona     = kwargs.get('ai_id', 1)

        styles  = self.get_styles_from_vdb(query, user_id=user_id) # list[dict] :: 벡터DB 에서 스타일 정보 조회
        result  = self.__ask_style_sheet(styles=styles, **kwargs)  # list[dict]
        # print("styles = ")
        # pprint(styles)
        # print("styles = ")
        # pprint(result)

        result_html = (
            f'<div class="text">{result["desc"]}</div>'
            f'{self.__style_to_html(result["styles"])}'
            # f'<br/><div class="text">원하시는 스타일을 말씀해주시면 그 스타일에 맞는 제품을 보여드릴게요.</div>'
        )
        if with_voice:
            voice = self.get_voice(result["desc"], with_voice, persona)
            result_html += f'<audio controls src="{voice}"></audio>'

        return HTMLResponse(result_html, status_code=200)

    def __ask_style_sheet(self, query, styles:list[dict], user_id=None, ai_id=None, **kwargs) -> dict:
        _prompt = PromptTemplate.from_template("""
당신은 재활용소재로 패션제품을 만드는 회사에서 고문으로 일하고 있는 친환경 활동가이자 패션과 스타일링의 전문가입니다.
<<스타일링 팁>> 만 사용하여 사용자가 원하는 스타일 3개를 추천하고, 각 스타일마다 상의/하의 조합을 3개씩 추천합니다.
1. 답변에 인사말/당신을 소개하는 문구/불필요한 미사여구는 제외합니다.
2. <<스타일링 팁>> 을 <<스타일링 표 예시>> 와 같이 구성한 뒤, <<JSON 출력 예시>>에 맞게 변경한 json 결과만 반환합니다.
3. 상의/하의 조합이 없다면 결과에서 제외합니다.

<<스타일링 팁>>
{style_tips}

<<스타일링 표 예시>>
```markdown
| 스타일                | score  | 상의/하의 조합(무조건 상의와 하의만 조합합니다.)                                    | 특징                                                    |
|:----------------------|--------|:------------------------------------------------------------------------------------|:--------------------------------------------------------|
| 미니멀 (깔끔, 심플)   | 0.4987 | 화이트 셔츠/블랙 슬랙스, 솔리드 티셔츠/그레이 와이드 팬츠, 슬림핏 니트/H라인 스커트 | 패턴 최소화·모노톤 중심, 실루엣 강조, 액세서리 최소     |
| 캐주얼 (편안, 데일리) | 0.4811 | 루즈핏 맨투맨/청바지, 스트라이프 티셔츠/치노 팬츠, 데님 셔츠/조거 팬츠              | 활동성 중심, 자연 소재(코튼·데님), 컬러 포인트로 경쾌함 |
```

<<JSON 출력 예시>>
{{
    "desc"   : "(여기에는 스타일들에 대한 종합 설명을 100자 내외로 넣어주세요.)"
    "styles" : [
        {{
            "name"  : "깔끔하고 심플한 미니멀",
            "desc"  : "패턴 최소화·모노톤 중심, 실루엣 강조, 액세서리 최소",
            "combi" : [
                {{ "top" : "화이트 셔츠"  , "bottom" : "블랙 슬랙스" }},
                {{ "top" : "솔리드 티셔츠", "bottom" : "그레이 와이드 팬츠" }},
                {{ "top" : "슬림핏 니트"  , "bottom" : "H라인 스커트" }},
            ]
        }},
        {{
            (다른 스타일들도 위 양식과 동일하게 추가해주세요.)
        }}
    ]
}}
""")
        message = _prompt.format(style_tips=json.dumps(styles, ensure_ascii=False))
        result  = self.ask_llm(query, user_id=user_id, ai_id=ai_id, prompt=message)
        # print(message, result, sep="\n")
        return json.loads(result)

    def __style_to_html(self, styles):
        result = []
        for style in styles:
            combi_list = [f"<li class='combi'>상의 : {dress['top']} / 하의 : {dress['bottom']}</li>" for dress in style['combi']]
            combi_list = "".join(combi_list)
            result.append(  "<ul class='style' data-label='스타일링'>"
                            f"  <li class='name'>{style['name']}</li>"
                            f"  <li class='desc'>{style['desc']}</li>"
                            f"  {combi_list}"
                            "</ul>")
        return "".join(result)