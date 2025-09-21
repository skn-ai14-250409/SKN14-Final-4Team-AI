import html
import json
import os
import re

from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone

from app.clients.runpod_api import RunpodTTSClient
from app.simple_llm import SimpleChatLLM


class IntentBase:
    simple_user_llm = SimpleChatLLM()
    embedding       = OpenAIEmbeddings(model=os.getenv("EMBED_MODEL"))
    pinecone        = Pinecone(os.getenv("PINECONE_API_KEY"))
    index_product   = pinecone.Index(os.getenv("PINECONE_INDEX_PRODUCT"))
    index_style     = pinecone.Index(os.getenv("PINECONE_INDEX_STYLE"))
    index_style_ns  = os.getenv("PINECONE_INDEX_STYLE_NAMESPACE", "transcripts-kr")
    tts_maker       = RunpodTTSClient()

    def __init__(self, prompt:str=None, **kwargs):
        if prompt:
            self._prompt = PromptTemplate.from_template(prompt)

    def ask_llm(self, query:str, history:list=None, prompt:str=None, **kwargs):
        model      = kwargs.get("model")
        user_id    = kwargs.get("user_id")
        ai_id      = kwargs.get("ai_id")
        with_voice = kwargs.get("with_voice", False)
        persona    = kwargs.get("persona", 1)
        # print(f"{kwargs=}")

        msg      = prompt  or self._prompt.format()
        history  = history or []
        history += [{"role": "system", "content": msg}]

        result   = IntentBase.simple_user_llm(query, history, model, user_id, ai_id)

        if with_voice:
            voice  = self.get_voice(result, with_voice=with_voice, persona=persona)
            result += f"<audio controls loop='false' src={voice}></audio>"

        return result

    def get_voice(self, text, with_voice=False, persona=1):
        if with_voice:
            voice = IntentBase.tts_maker.run_tts(text=text, persona=str(persona))
            voice = voice["s3_url"]
        else:
            voice = None

        return voice


    def get_styles_from_vdb(self, query:str, top_k=10) -> list[dict]:
        vector  = IntentBase.embedding.embed_query(query)
        matches = IntentBase.index_style.query(vector=vector, top_k=top_k, namespace=IntentBase.index_style_ns, include_metadata=True)["matches"]

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

    def __refine(self, text):
        text = re.sub(r"<(\w+) [^>]*>", r"<\1>", text)
        text = re.sub(r"[\n\r\t]", "", text)
        text = re.sub(r"\s+", " ", text)
        text = html.escape(text, quote=True)
        return text
    def __parse_water_saved(self, x:dict):
        used_l  = x.get('water_used_l',0)
        saved_l = x.get('water_saved_l',0)
        sum_l   = used_l + saved_l
        if sum_l != 0: return f"{saved_l * 100 / sum_l:.2f} %"
        else:          return f"{saved_l:.2f} L"
    def __parse_co2_saved(self, x:dict):
        used_l  = x.get('co2_used_kg',0)
        saved_l = x.get('co2_saved_kg',0)
        sum_l   = used_l + saved_l
        if sum_l != 0: return f"{saved_l * 100 / sum_l:.2f} %"
        else:          return f"{saved_l:.2f} L"
    def __parse_merged_material(self, meta:dict):
        materials = json.loads(meta.get("material"))
        materials = [f"{m['percent']}% {m['material']}" for m in materials]
        return ", ".join(materials)
    def get_products_from_vdb(self, query:str, top_k=3, filter=None):
        vector  = IntentBase.embedding.embed_query(query)
        matches = IntentBase.index_product.query(vector=vector, top_k=top_k, filter=filter, include_metadata=True)["matches"]

        products = []
        for match in matches:
            meta = match["metadata"]
            products.append({
                "id"          : int(meta.get("id")),
                "name"        : self.__refine(meta.get("name")),
                "category"    : meta.get("category"),
                "price"       : f"{meta.get('currency')} {meta.get('price')}",
                "image"       : meta.get("image_url"),
                "color"       : meta.get("color"),
                "color_detail": meta.get("color_detail"),
                "material"    : self.__parse_merged_material(meta),
                "url"         : meta.get("url"),
                "saved_water" : self.__parse_water_saved(json.loads(meta.get("water_saved_l"))),
                "saved_co2"   : self.__parse_co2_saved(json.loads(meta.get("co2_saved_kg"))),
                "spec"        : self.__refine(meta.get("spec")),
            })
        return products