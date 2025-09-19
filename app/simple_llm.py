import html
import json
import re

from openai import OpenAI
from sqlalchemy import text

from app.database import SessionLocal


class SimpleChatLLM:

    client: OpenAI  = None

    def __init__(self, model:str = "gpt-4.1-nano", client: OpenAI = None):
        if SimpleChatLLM.client is None:
            SimpleChatLLM.client = OpenAI()

        self._client = client if client else SimpleChatLLM.client
        self._model  = model


    def __call__(self, message:str, history:list[dict] = None, model=None, user_id=None, ai_id=None, *args, **kwargs):
        history = history or []
        history.append({"role": "user", "content": message})

        if user_id and ai_id:
            chat_log = self.load_chat_history(user_id, ai_id)
            history = chat_log + history

        resp = self.client.chat.completions.create(
            model=self._model if model is None else model,
            messages=history,
        )
        return resp.choices[0].message.content

    def __refine(self, text):
        text = re.sub(r"<(\w+) [^>]*>", r"<\1>", text)
        text = re.sub(r"[\n\r\t]", "", text)
        text = re.sub(r"  +", " ", text)
        text = html.escape(text, quote=True)
        return text
    def load_chat_history(self, user_id, ai_id, top_k=20):
        roles = {
            "ai"    : "assistant",
            "user"  : "user",
            "system": "system"
        }
        with SessionLocal() as db:
            _query = ( "SELECT talker_type, style_text "
                       "FROM   apiapp_chathistory "
                       "WHERE  user_id       = :user_id"
                       "   AND influencer_id = :ai_id "
                       "ORDER BY talked_at DESC "
                       "LIMIT :limit OFFSET 1" )
            result = db.execute(text(_query), {"user_id":user_id, "ai_id":ai_id, "limit":top_k})
            rows   = result.fetchall()
            history = [ {"role":roles[row[0]], "content":self.__refine(row[1])} for row in rows ]
            return history[::-1]
