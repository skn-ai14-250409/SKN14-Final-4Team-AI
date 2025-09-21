from starlette.responses import HTMLResponse

from .IntentBase import IntentBase


class CertVerify(IntentBase):

    def __init__(self):
        prompt = """
당신은 재활용소재로 패션제품을 만드는 회사에서 고문으로 일하고 있는 친환경 활동가입니다.
친환경 인증과 관련된 질문에만 답변합니다.
당신이 알고있는 수준으로만 답변하고, 일체의 거짓이나 꾸밈이 없어야 하며, 당신이 모르는 정보라면 모른다고 답변합니다.
1. 답변에서 인사말/당신을 소개하는 문구/불필요한 미사여구는 제외합니다.
2. 답변은 공백을 제외했을 때 250자 내외로 작성합니다.
"""
        super().__init__(prompt)

    def __call__(self, **kwargs):
        with_voice = kwargs.get("with_voice", False)
        result = self.ask_llm(**kwargs)
        if with_voice:
            voice  = self.get_voice(result, with_voice=with_voice)
            result += f"<audio controls loop='false' src={voice}></audio>"
        return HTMLResponse(result, status_code=200)