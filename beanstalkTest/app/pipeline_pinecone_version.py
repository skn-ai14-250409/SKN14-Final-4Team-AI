# from __future__ import annotations
# from typing import List, Dict, Any, Optional
# import os, json, sys
# from pathlib import Path
# from dotenv import load_dotenv
# from pinecone import Pinecone
# from openai import OpenAI
# import sqlalchemy as sa
# from sqlalchemy import text

# load_dotenv()

# # ================== 기본 설정 ==================
# client = OpenAI()
# DEFAULT_CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4.1-nano")
# EMBED_MODEL       = os.getenv("EMBED_MODEL")

# PINECONE_API_KEY   = os.getenv("PINECONE_API_KEY")
# PINECONE_INDEX_TEXT = os.getenv("PINECONE_INDEX_1")   
# PINECONE_INDEX_RDB  = os.getenv("PINECONE_INDEX_2")  

# APP_DB_URL         = os.getenv("APP_DB_URL")        

# ALLOWED_STYLES = [
#     "미니멀 (깔끔, 심플)",
#     "캐주얼 (편안, 데일리)",
#     "포멀 (정장, 단정)",
#     "페미닌 (부드럽고 여성스러움)",
#     "스트리트 (트렌디, 자유분방)",
# ]

# STYLE_MATCHING_TABLE = """
# << 스타일 매칭 예시 >>
# | 스타일                  | 상의                        | 하의                          | 특징                                     |
# | -------------------- | ------------------------- | --------------------------- | -------------------------------------- |
# | **미니멀 (깔끔, 심플)**     | 화이트 셔츠, 솔리드 티셔츠, 슬림핏 니트   | 블랙 슬랙스, 그레이 와이드 팬츠, H라인 스커트 | 패턴 최소화·모노톤 중심, 실루엣 강조, 액세서리 최소         |
# | **캐주얼 (편안, 데일리)**    | 루즈핏 맨투맨, 스트라이프 티셔츠, 데님 셔츠 | 청바지, 치노 팬츠, 조거 팬츠           | 활동성 중심, 자연 소재(코튼·데님), 컬러 포인트로 경쾌함      |
# | **포멀 (정장, 단정)**      | 블라우스, 테일러드 자켓, 셔츠         | 슬랙스, 펜슬 스커트, 세미와이드 팬츠       | 직장·공식석상 적합, 세트 매치 시 완성도 ↑, 뉴트럴 톤·고급 소재 |
# | **페미닌 (부드럽고 여성스러움)** | 리본 블라우스, 레이스 니트, 오프숄더 톱   | 플레어 스커트, 미디스커트, 슬림핏 슬랙스     | 곡선 실루엣, 파스텔/뉴드 컬러, 러플·레이스·주름 디테일       |
# | **스트리트 (트렌디, 자유분방)** | 오버핏 후드티, 그래픽 티셔츠, 크롭 톱    | 카고 팬츠, 와이드 데님, 조거 팬츠        | 과감한 실루엣·프린트, 스니커즈/모자 매치 중요, 서브컬처 감성    |
# """

# # =============== DB 유틸 (검색기록 저장 유지) ===============
# def _ensure_sqlite_url(path_or_url: str) -> str:
#     if not path_or_url:
#         raise ValueError("APP_DB_URL is required.")
#     if path_or_url.startswith("sqlite:///") or "://" in path_or_url:
#         return path_or_url
#     return f"sqlite:///{Path(path_or_url).as_posix()}"

# def _engine(db_url: str):
#     return sa.create_engine(_ensure_sqlite_url(db_url), pool_pre_ping=True, future=True)

# def _last_insert_id(conn, result):
#     try:
#         lrid = getattr(result, "lastrowid", None)
#         if lrid: return int(lrid)
#     except Exception:
#         pass
#     dname = conn.engine.dialect.name
#     if dname == "mysql":
#         return int(conn.execute(text("SELECT LAST_INSERT_ID()")).scalar_one())
#     if dname == "sqlite":
#         return int(conn.execute(text("SELECT last_insert_rowid()")).scalar_one())
#     raise RuntimeError(f"Unsupported dialect: {dname}")

# def save_search_history(db_url: str = APP_DB_URL) -> int:
#     eng = _engine(db_url)
#     with eng.begin() as conn:
#         res = conn.execute(text("INSERT INTO search_history (searched_at) VALUES (CURRENT_TIMESTAMP)"))
#         return _last_insert_id(conn, res)

# def update_search_history_look_style(search_id: int, look_style: str, db_url: str = APP_DB_URL):
#     eng = _engine(db_url)
#     with eng.begin() as conn:
#         conn.execute(text(
#             "UPDATE search_history SET look_style = :look_style WHERE id = :search_id"
#         ), {"look_style": look_style, "search_id": search_id})

# def save_search_history_product(search_id: int, product_id: int, db_url: str = APP_DB_URL) -> int:
#     eng = _engine(db_url)
#     with eng.begin() as conn:
#         res = conn.execute(text(
#             "INSERT INTO search_history_product (search_id, product_id) VALUES (:sid, :pid)"
#         ), {"sid": search_id, "pid": product_id})
#         return _last_insert_id(conn, res)

# def ensure_app_product_exists(product_id: int, db_url: str = APP_DB_URL) -> bool:
#     """FK 안전을 위해 app_product에 해당 id가 실제 존재하는지 확인 (케이스 A)."""
#     eng = _engine(db_url)
#     with eng.begin() as conn:
#         exists = conn.execute(
#             text("SELECT 1 FROM app_product WHERE id = :pid LIMIT 1"),
#             {"pid": product_id}
#         ).scalar()
#         return bool(exists)

# # ================== Pinecone 헬퍼 ==================
# def _pc() -> Pinecone:
#     return Pinecone(api_key=PINECONE_API_KEY)

# def _embed(text: str) -> List[float]:
#     return client.embeddings.create(model=EMBED_MODEL, input=text).data[0].embedding

# # ============ 1) Index_1: 스타일 후보 검색 ============
# def vedb_styles_from_index1(user_query: str, top_k: int = 3, namespace: str = "") -> List[Dict[str, Any]]:
#     pc = _pc()
#     index = pc.Index(PINECONE_INDEX_TEXT)
#     emb = _embed(user_query)
#     res = index.query(vector=emb, top_k=top_k, include_metadata=True, namespace=namespace) or {}
#     matches = res.get("matches", []) or []
#     out = []
#     for m in matches:
#         meta = (m.get("metadata") or {})
#         raw = meta.get("occasion") or meta.get("style") or meta.get("look") or "Unknown"
#         out.append({"raw_style": str(raw)})
#     return out

# # ============ 2) LLM: 스타일 정규화(참고만, 강제 X) ============
# STYLE_LIST_SCHEMA = {
#     "type": "json_schema",
#     "json_schema": {
#         "name": "style_list",
#         "schema": {
#             "type": "object",
#             "properties": {
#                 "results": {
#                     "type": "array",
#                     "items": {"type": "object", "properties": {
#                         "look_style": {"type": "string"}
#                     }, "required": ["look_style"]}
#                 }
#             },
#             "required": ["results"]
#         }
#     }
# }

# def normalize_styles_with_llm(raw_looks: List[Dict[str, Any]], model: str = DEFAULT_CHAT_MODEL) -> List[Dict[str, Any]]:
#     """
#     ALLOWED_STYLES를 '참고'해서 가장 가까운 한국어 스타일명을 제안하되,
#     애매하면 원본(raw_style)을 그대로 반환해도 됨(강제 X).
#     출력: {"results":[{"look_style": "..."}]}
#     """
#     sys_msg = (
#         "너는 스타일 태그를 한국어 스타일명으로 요약하는 조력자다. "
#         "아래 테이블을 참고하되, 정확히 매칭되지 않으면 원본을 유지해도 좋다. "
#         "허용 리스트는 '참고용'이며, 반드시 그중 하나로 제한하지는 않는다."
#     )
#     user_msg = (
#         f"참고용 허용 스타일: {ALLOWED_STYLES}\n"
#         f"스타일 매칭표:\n{STYLE_MATCHING_TABLE}\n\n"
#         f"raw_styles: {json.dumps(raw_looks, ensure_ascii=False)}\n"
#         "결과는 JSON만 출력: {\"results\":[{\"look_style\":\"...\"}, ...]}"
#     )

#     resp = client.chat.completions.create(
#         model=model,
#         messages=[
#             {"role": "system", "content": sys_msg},
#             {"role": "user", "content": user_msg},
#         ],
#         response_format=STYLE_LIST_SCHEMA,
#         temperature=0.2
#     )
#     try:
#         data = json.loads(resp.choices[0].message.content)
#         return data.get("results", [])
#     except Exception:
#         # 실패 시 raw_looks를 그대로 노출(없으면 기본값)
#         return [{"look_style": (raw_looks[0].get("raw_style") if raw_looks else "캐주얼 단정룩")}]

# # ============ 3) Index_2: 상품 후보 검색 =============
# def search_candidates_in_index2(
#     plan: List[Dict[str, Any]],
#     top_k: int = 10,
#     namespace: str = ""
# ) -> List[Dict[str, List[Dict[str, Any]]]]:
#     """
#     plan = [{"look_style": "..."}] 형태.
#     각 스타일에 대해 '상의', '하의' 쿼리를 만들어 INDEX_2에서 후보 조회.
#     """
#     pc = _pc()
#     index = pc.Index(PINECONE_INDEX_RDB)
#     out: List[Dict[str, List[Dict[str, Any]]]] = []

#     for item in plan:
#         look_style = str(item.get("look_style", "")).strip() or "캐주얼 단정룩"

#         emb_top = _embed(f"{look_style} 상의")
#         tops_res = index.query(vector=emb_top, top_k=top_k, include_metadata=True, namespace=namespace) or {}
#         tops = [m["metadata"] for m in (tops_res.get("matches") or []) if m.get("metadata")]

#         emb_bottom = _embed(f"{look_style} 하의")
#         bottoms_res = index.query(vector=emb_bottom, top_k=top_k, include_metadata=True, namespace=namespace) or {}
#         bottoms = [m["metadata"] for m in (bottoms_res.get("matches") or []) if m.get("metadata")]

#         out.append({"top": tops, "bottom": bottoms})
#     return out

# # ============ 4) LLM: 최종 pick + 이력 저장 (FK 보존) ============
# FINAL_RESPONSE_FORMAT_CODY = {
#     "type": "json_schema",
#     "json_schema": {
#         "name": "final_cody_recommendation",
#         "schema": {
#             "type": "object",
#             "properties": {
#                 "results": {
#                     "type": "array",
#                     "items": {
#                         "type": "object",
#                         "properties": {
#                             "look_style": {"type": "string"},
#                             "top":    {"type": ["object", "null"]},
#                             "bottom": {"type": ["object", "null"]},
#                             "reason_selected": {"type": "string"}
#                         },
#                         "required": ["look_style", "top", "bottom", "reason_selected"]
#                     }
#                 }
#             },
#             "required": ["results"]
#         }
#     }
# }

# def pick_best_with_llm_and_log(
#     plan: List[Dict[str, Any]],
#     candidates: List[Dict[str, List[Dict[str, Any]]]],
#     search_id: int,
#     model: str = DEFAULT_CHAT_MODEL
# ) -> List[Dict[str, Any]]:
#     """
#     candidates[i] = {"top":[metadata...], "bottom":[metadata...]}
#     metadata에는 반드시 app_product의 'id'가 포함되어 있다고 가정.
#     """
#     system = "당신은 상의/하의 후보 중 최선의 1개씩을 골라 코디를 만드는 전문가입니다."
#     dev = (
#         "규칙:\n"
#         "1) 결과 배열 길이는 plan과 동일.\n"
#         "2) 각 항목에서 상의 1개, 하의 1개를 반드시 candidates에서 선택(새로 만들지 말 것).\n"
#         "3) 후보가 완전히 비었을 때만 null 허용. 하나라도 있으면 반드시 선택.\n"
#         "4) reason_selected는 간결하게.\n"
#         "5) 선택한 상품 객체는 후보 metadata를 그대로 사용(필드 변형/환각 금지)."
#     )
#     user = (
#         f"스타일 매칭표(참고):\n{STYLE_MATCHING_TABLE}\n\n"
#         f"plan: {json.dumps(plan, ensure_ascii=False)}\n\n"
#         f"candidates: {json.dumps(candidates, ensure_ascii=False)[:6000]}"
#     )
#     resp = client.chat.completions.create(
#         model=model,
#         messages=[
#             {"role": "system", "content": system},
#             {"role": "system", "content": dev},
#             {"role": "user", "content": user},
#         ],
#         response_format=FINAL_RESPONSE_FORMAT_CODY,
#         temperature=0.1
#     )
#     data = json.loads(resp.choices[0].message.content)
#     results = data.get("results", [])

#     # 검색 기록 업데이트 + FK 보존
#     for res in results:
#         res["search_id"] = search_id
#         ls = res.get("look_style")
#         if ls:
#             try:
#                 update_search_history_look_style(search_id, ls)
#             except Exception as e:
#                 print(f"[WARN] look_style 저장 실패: {e}")

#         for side in ["top", "bottom"]:
#             prod = res.get(side)
#             if isinstance(prod, dict) and "id" in prod and prod["id"] is not None:
#                 pid = int(prod["id"])
#                 # 케이스 A: app_product에 항상 있어야 FK가 OK
#                 if ensure_app_product_exists(pid):
#                     try:
#                         hid = save_search_history_product(search_id, pid)
#                         prod["search_history_product_id"] = hid
#                     except Exception as e:
#                         print(f"[WARN] search_history_product 저장 실패(product_id={pid}): {e}")
#                 else:
#                     print(f"[WARN] app_product({pid}) 미존재: FK 저장 스킵")
#     return results

# # ================== 실행 플로우(Main) ==================
# if __name__ == "__main__":
#     query = " ".join(sys.argv[1:]).strip()

#     # 0) 검색 이력 생성
#     search_id = save_search_history()
#     print(f"[search_id] {search_id}")

#     # 1) Index_1에서 스타일 후보 추출
#     raw_styles = vedb_styles_from_index1(query, top_k=3)
#     print("[raw_styles]", raw_styles)

#     # 2) LLM으로 스타일 정규화(참고만, 강제 X)
#     plan = normalize_styles_with_llm(raw_styles)
#     print("[plan]", json.dumps(plan, ensure_ascii=False, indent=2))

#     # 3) Index_2에서 상/하의 후보 검색
#     candidates = search_candidates_in_index2(plan, top_k=1)
#     print("[candidates_found_per_style]", [{k: len(v) for k, v in c.items()} for c in candidates])

#     # 4) LLM 최종 pick + 기록 저장(FK)
#     final = pick_best_with_llm_and_log(plan, candidates, search_id)
#     print("\n=== FINAL CODY RECOMMENDATIONS ===")
#     print(json.dumps(final, ensure_ascii=False, indent=2))
