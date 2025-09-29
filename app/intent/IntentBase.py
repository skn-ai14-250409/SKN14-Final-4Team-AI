import html
import json
import os, re

from functools import lru_cache
from typing import List, Dict, Any, Optional 
from dotenv import load_dotenv, find_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
from openai import OpenAI

from app.clients.runpod_api import RunpodTTSClient
from app.simple_llm import SimpleChatLLM

# .env는 클래스 정의 전에 로드
_ = load_dotenv(find_dotenv(usecwd=True))

class IntentBase:
    simple_user_llm = SimpleChatLLM()
    embedding       = OpenAIEmbeddings(model=os.getenv("EMBED_MODEL"))
    index_host      = os.getenv("PINECONE_INDEX_HOST")
    index_style_ns  = os.getenv("PINECONE_INDEX_STYLE_NAMESPACE")
    tts_maker       = RunpodTTSClient()

    @staticmethod
    @lru_cache(maxsize=1)
    def get_clients():
        pc_key = os.getenv("PINECONE_API_KEY")
        host = os.getenv("PINECONE_INDEX_HOST")
        oa_key = os.getenv("OPENAI_API_KEY")
        
        if not pc_key: raise RuntimeError("PINECONE_API_KEY is not set")
        if not host:   raise RuntimeError("PINECONE_INDEX_HOST is not set")
        if not oa_key: raise RuntimeError("OPENAI_API_KEY is not set")

        pc = Pinecone(api_key=pc_key)
        index = pc.Index(host=host)  
        oc = OpenAI(api_key=oa_key)

        return index, oc

    @staticmethod
    @lru_cache(maxsize=1)
    def get_product_index():
        pc_key = os.getenv("PINECONE_API_KEY")
        host   = os.getenv("PINECONE_INDEX_HOST")
        if not pc_key: raise RuntimeError("PINECONE_API_KEY is not set")
        pc = Pinecone(api_key=pc_key)
        if host:
            return pc.Index(host=host)
        if not host:
            raise RuntimeError("PINECONE_INDEX_HOST or PINECONE_INDEX_PRODUCT is required")
        return pc.Index(host)

    @staticmethod
    @lru_cache(maxsize=1)
    def get_style_index():
        pc_key   = os.getenv("PINECONE_API_KEY")
        host_alt = os.getenv("PINECONE_INDEX_STYLE_HOST")
        if not pc_key: raise RuntimeError("PINECONE_API_KEY is not set")
        pc = Pinecone(api_key=pc_key)
        if host_alt:
            return pc.Index(host=host_alt)
        if not host_alt:
            raise RuntimeError("PINECONE_INDEX_STYLE_HOST or PINECONE_INDEX_STYLE is required")
        return pc.Index(host_alt)

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
            result += f"<audio controls src='{voice}'></audio>"
        return result

    def get_voice(self, text, with_voice=False, persona=1):
        if with_voice:
            voice = IntentBase.tts_maker.run_tts(text=text, persona=str(persona))
            voice = voice["s3_url"]
        else:
            voice = None
        return voice

    def get_styles_from_vdb(self, query:str, top_k=5) -> List[Dict[str, Any]]:
        vector  = IntentBase.embedding.embed_query(query)
        style_index = IntentBase.get_style_index()
        ns = IntentBase.index_style_ns
        
        res = style_index.query(
            vector=vector,
            top_k=top_k,
            namespace=ns,
            include_metadata=True,
        )
        
        matches = getattr(res, "matches", None) or (res.get("matches", []) if isinstance(res, dict) else [])

        styles: List[Dict[str, Any]] = []
        for m in matches:
            meta = m.get("metadata", {}) if isinstance(m, dict) else getattr(m, "metadata", {}) or {}
            styles.append({
                "index": meta.get("index"),
                "gender": meta.get("gender"),
                "occasion": meta.get("occasion"),
                "season":   meta.get("season")
            })
        return styles

    def get_products_from_vdb(self, query: str, top_k: int = 3, metadata_filter: Optional[Dict[str, Any]] = None):
        vector = IntentBase.embedding.embed_query(query)
        product_index = IntentBase.get_product_index()

        res = product_index.query(
            vector=vector,
            top_k=top_k,
            include_metadata=True,
            filter=metadata_filter,                # Pinecone arg 이름은 'filter'
        )
        matches = getattr(res, "matches", None) or (res.get("matches", []) if isinstance(res, dict) else [])

        products = []
        for m in matches:
            meta = m.get("metadata", {}) if isinstance(m, dict) else getattr(m, "metadata", {}) or {}
            try:
                saved_water = self.__parse_water_saved(json.loads(meta.get("water_saved_l")))
            except Exception:
                saved_water = None
            try:
                saved_co2 = self.__parse_co2_saved(json.loads(meta.get("co2_saved_kg")))
            except Exception:
                saved_co2 = None
            try:
                material_merged = self.__parse_merged_material(meta)
            except Exception:
                material_merged = meta.get("material")

            products.append({
                "id":           int(meta.get("id")) if meta.get("id") is not None else None,
                "brnad_id":     int(meta.get("brand_id")) if meta.get("brand_id") is not None else None,
                "name":         self.__refine(meta.get("name") or ""),
                "category":     meta.get("category"),
                "price":        f"{meta.get('currency')} {meta.get('price')}" if meta.get("currency") and meta.get("price") else None,
                "image":        meta.get("image_url"),
                "color":        meta.get("color"),
                "color_detail": meta.get("color_detail"),
                "material":     material_merged,
                "url":          meta.get("url"),
                "saved_water":  saved_water,
                "saved_co2":    saved_co2,
                "spec":         self.__refine(meta.get("spec") or ""),
            })
        return products
    
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