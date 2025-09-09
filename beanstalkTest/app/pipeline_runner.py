from __future__ import annotations
from typing import List, Dict, Any, Optional
from pathlib import Path
import os, json, sys
import pandas as pd
import sqlalchemy as sa
from sqlalchemy import text
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI

load_dotenv()

# --- 기본 설정 ---
client = OpenAI()
DEFAULT_CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4.1-nano")
EMBED_MODEL = os.getenv("EMBED_MODEL")
DEFAULT_DB_URL = os.getenv("APP_DB_URL")

# -------------------------------------------------------------------
# 허용 스타일 & 키워드 (프롬프트/폴백에 공통 사용)
# -------------------------------------------------------------------
ALLOWED_STYLES = [
    "캐주얼 단정룩",
    "페미닌 클래식룩",
    "시크·고급룩",
    "모던 미니멀룩",
    "포인트 활용룩",
]

TOP_KEYWORDS = [
    "top","shirt","blouse","knit","tee","t-shirt",
    "니트","셔츠","블라우스","탑","상의"
]
BOTTOM_KEYWORDS = [
    "pants","slacks","jeans","denim","skirt","shorts","trousers",
    "슬랙스","데님","청바지","스커트","치마","하의","코튼 팬츠"
]

# --- 스타일 매칭표 ---
STYLE_MATCHING_TABLE = """
<< 스타일 매칭 예시 >>
| 스타일 | 상의 | 하의 | 아우터 | 슈즈 | 가방/액세서리 | 특징 |
|------------------|---------------------------------------|-----------------------------------|-------------------------------------|--------------------------|-------------------------------|-------------------------------------------------|
| 캐주얼 단정룩 | 블라우스 / 블랙 이너탑 | 데님(연청·흑청) or 코튼 팬츠 | 밝은 톤 린넨 재킷, 트위드 반팔 재킷 | 로퍼 / 플랫슈즈 | 얇은 벨트, 토트백 | 평소 캐주얼 선호자도 부담 없이 단정하게 |
| 페미닌 클래식룩 | 블라우스(디테일 최소) / 여름 니트 | 롱스커트(H라인) or A라인 미니스커트 | 반팔 재킷 or 니트 | 앞코 얄상한 힐 / 샌들 | 이어링, 스카프, 토트백 | 가장 무난하면서도 우아한 정석 하객룩 |
| 시크·고급룩 | 넥라인 높은 니트·드레이프 탑 | 블랙 슬랙스(스트레이트핏) | 노카라 재킷 / 수트 셋업 | 얄상한 구두 or 블록힐 | 미니 토트백, 주얼리 | 군더더기 없는 절제된 세련미 |
| 모던 미니멀룩 | 단색 탑(화이트/블랙) | 화이트 슬랙스 / 크림 코튼 팬츠 | 베이지 재킷 | 로퍼 / 뮬 | 스카프, 가죽 토트 | 심플+깨끗한 인상, 사진발 잘 받음 |
| 포인트 활용룩 | 블랙 원피스 / 블라우스 | 원피스 or H라인 스커트 | (아우터 생략 가능) | 굽 있는 샌들 | 스카프 / 포인트 이어링 | 전체는 단정, 소품으로 화사함 강조 |
"""

# --- 유틸리티 함수 ---
def _ensure_sqlite_url(path_or_url: str) -> str:
    if not path_or_url:
        raise ValueError("db_url is required (set APP_DB_URL or pass db_url).")
    if path_or_url.startswith("sqlite:///") or "://" in path_or_url:
        return path_or_url
    return f"sqlite:///{Path(path_or_url).as_posix()}"

def _engine(db_url: str):
    return sa.create_engine(_ensure_sqlite_url(db_url), pool_pre_ping=True, future=True)

def _last_insert_id(conn, result):
    try:
        lrid = getattr(result, "lastrowid", None)
        if lrid:
            return int(lrid)
    except Exception:
        pass
    dname = conn.engine.dialect.name
    if dname == "mysql":
        return int(conn.execute(text("SELECT LAST_INSERT_ID()")).scalar_one())
    if dname == "sqlite":
        return int(conn.execute(text("SELECT last_insert_rowid()")).scalar_one())
    raise RuntimeError(f"Unsupported dialect: {dname}")

def validate_sql(sql: str) -> None:
    s = sql.strip().lower()
    if not s.startswith("select "): raise ValueError("SELECT 쿼리만 허용합니다.")
    if 'app_product' not in s: raise ValueError("허용된 테이블만 조회할 수 있습니다.")
    if any(k in s for k in [' update ', ' delete ', ' insert ', ' drop ']): raise ValueError("금지된 키워드가 포함된 쿼리입니다.")

# --- JSON 스키마 ---
RESPONSE_FORMAT_CODY_PLAN = {
    "type": "json_schema", "json_schema": {"name": "cody_sql_plan", "schema": {
        "type": "object", "properties": {"results": {"type": "array", "items": {
            "type": "object", "properties": {
                "cody_style": {"type": "string"},
                "top_query": {"type": "string"},
                "bottom_query": {"type": "string"}
            }, "required": ["cody_style", "top_query", "bottom_query"]
        }}}, "required": ["results"]
    }}
}

product_schema = {
    "type": ["object", "null"],
    "properties": {
        "name": {"type": "string"},
        "price": {"type": ["number", "null"]},
        "image_url": {"type": "string"},
        "color": {"type": "string"},
        "spec": {"type": "string"},
        "sustainable_detail": {"type": "string"},
        "url": {"type": "string"},
        "search_history_product_id": {"type": ["number", "null"]}
    },
    "required": ["name", "price", "image_url", "color", "spec", "sustainable_detail", "url", "search_history_product_id"]
}

FINAL_RESPONSE_FORMAT_CODY = {
    "type": "json_schema",
    "json_schema": {
        "name": "final_cody_recommendation",
        "schema": {
            "type": "object",
            "properties": {
                "results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "look_style": {"type": "string"},
                            "top": product_schema,
                            "bottom": product_schema,
                            "reason_selected": {"type": "string"}
                        },
                        "required": ["look_style", "top", "bottom", "reason_selected"]
                    }
                }
            },
            "required": ["results"]
        }
    }
}

# --- 프롬프트 규칙 ---
SYSTEM_PROMPT = (
    "너는 사용자의 요청과 벡터DB 스타일 정보를 바탕으로 상의+하의 코디 세트를 추천하는 AI 스타일리스트다. "
    "주어진 '스타일 매칭표'만을 기준으로 판단하고, 결과의 스타일명은 반드시 표의 5가지 한국어 스타일명 중 하나여야 한다."
)

DEV_RULES_CODY = (
    "아래 '스타일 매칭표'를 기준으로, 벡터DB에서 받은 원시 스타일 토큰(영문 work/minimal/chic 등)이 오더라도 "
    "반드시 표의 5가지 한국어 스타일 중 하나로 정규화하여(cody_style) 사용하라.\n"
    f"{STYLE_MATCHING_TABLE}\n\n"
    "- 테이블: app_product\n"
    "- SELECT 컬럼: id, name, price, image_url, color, url, category, details_intro, material, fit, sustainable_detail\n"
    "- cody_style은 다음 중 하나의 '정확한 문자열'만 허용: "
    f"{ALLOWED_STYLES}\n"
    "- 쿼리 작성 규칙:\n"
    "  * 모든 쿼리는 `SELECT [컬럼들] FROM app_product WHERE [조건들] LIMIT 10` 구조.\n"
    "  * 한국어/영어 동시 대응: 상의는 category/name/details_intro 중 최소 한 컬럼에 다음 키워드 집합 중 하나 이상 포함: "
    f"{TOP_KEYWORDS}\n"
    "  * 한국어/영어 동시 대응: 하의는 category/name/details_intro 중 최소 한 컬럼에 다음 키워드 집합 중 하나 이상 포함: "
    f"{BOTTOM_KEYWORDS}\n"
    "  * 후보가 1개 이상이면 반드시 상의/하의 각각 1개를 선택하라(후보가 완전히 비어 있을 때만 null 허용).\n"
    "- 출력 스키마: {\"results\": [{\"cody_style\": \"위 5개 중 하나\", \"top_query\": \"상의 SQL\", \"bottom_query\": \"하의 SQL\"}]}\n"
)

# --- DB 함수 ---
def save_search_history(db_url: str = DEFAULT_DB_URL) -> int:
    eng = _engine(db_url)
    with eng.begin() as conn:
        res = conn.execute(text("""
            INSERT INTO search_history (searched_at)
            VALUES (CURRENT_TIMESTAMP)
        """))
        sid = _last_insert_id(conn, res)
        if not sid:
            raise RuntimeError("Failed to obtain search_id.")
        return sid

def save_search_history_product(search_id: int, product_id: int, db_url: str = DEFAULT_DB_URL) -> int:
    eng = _engine(db_url)
    with eng.begin() as conn:
        result = conn.execute(text("""
            INSERT INTO search_history_product (search_id, product_id)
            VALUES (:search_id, :product_id)
        """), {"search_id": search_id, "product_id": product_id})
        return _last_insert_id(conn, result)

def update_search_history_look_style(search_id: int, look_style: str, db_url: str = DEFAULT_DB_URL):
    eng = _engine(db_url)
    with eng.begin() as conn:
        conn.execute(text("""
            UPDATE search_history
            SET look_style = :look_style
            WHERE id = :search_id
        """), {"look_style": look_style, "search_id": search_id})

# --- VDB 조회 ---
def vedb_list(user_query: str, namespace: str = "transcripts-kr", top_k: int = 2) -> List[Dict[str, Any]]:
    api_key, index_name = os.getenv("PINECONE_API_KEY"), os.getenv("PINECONE_INDEX_NAME")
    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)
    emb = client.embeddings.create(model=EMBED_MODEL, input=user_query).data[0].embedding

    res = index.query(vector=emb, top_k=top_k, namespace=namespace, include_metadata=True)
    if not res.get("matches"):
        print(f"❌ 네임스페이스 '{namespace}'에서 검색 결과 없음")
        return []

    looks = [{"look_style": m.get("metadata", {}).get("occasion", "Unknown Style")} for m in res["matches"]]
    print(f"✅ 벡터DB에서 {len(looks)}개 스타일 발견: {[l['look_style'] for l in looks]}")
    return looks

# --- 쿼리 플랜 생성 ---
def prompting_to_cody_query_plan(looks: List[Dict[str, Any]], model: str = DEFAULT_CHAT_MODEL) -> List[Dict[str, Any]]:
    user_message = (
        "아래 스타일 목록을 보고 각 항목마다 (1) 한국어 5종 중 하나의 cody_style, "
        "(2) 상의 SQL, (3) 하의 SQL 을 생성하라.\n"
        f"허용 스타일: {ALLOWED_STYLES}\n"
        f"스타일 목록(원시): {json.dumps(looks, ensure_ascii=False)}"
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": DEV_RULES_CODY},
            {"role": "user", "content": user_message}
        ],
        response_format=RESPONSE_FORMAT_CODY_PLAN
    )
    data = json.loads(resp.choices[0].message.content)
    return data.get("results", [])

# --- 헬퍼: LIKE OR 절 & 폴백 SQL ---
def _build_like_or(columns: List[str], keywords: List[str]) -> str:
    parts = []
    for col in columns:
        for kw in keywords:
            parts.append(f"LOWER({col}) LIKE '%{kw.lower()}%'")
    return " OR ".join(parts)

def _fallback_sql(kind: str) -> str:
    cols = ["category", "name", "details_intro"]
    where = _build_like_or(cols, TOP_KEYWORDS if kind == "top" else BOTTOM_KEYWORDS)
    return (
        "SELECT id,name,price,image_url,color,url,category,details_intro,material,fit,sustainable_detail "
        "FROM app_product "
        f"WHERE ({where}) "
        "LIMIT 10"
    )

# --- 후보 -> product 객체 변환 (폴백/보정용) ---
def _row_to_product(row: Dict[str, Any]) -> Dict[str, Any]:
    def _get(v, default=""):
        return "" if v is None else v
    spec_bits = []
    for k in ["details_intro", "material", "fit"]:
        if row.get(k):
            spec_bits.append(str(row[k]))
    spec = " / ".join(spec_bits) if spec_bits else ""
    return {
        "name": _get(row.get("name")),
        "price": row.get("price"),
        "image_url": _get(row.get("image_url")),
        "color": _get(row.get("color")),
        "spec": spec,
        "sustainable_detail": _get(row.get("sustainable_detail")),
        "url": _get(row.get("url")),
        "search_history_product_id": None,
    }

def _find_product_id_by_name(candidates: List[Dict[str, Any]], name: str) -> Optional[int]:
    name_lower = (name or "").strip().lower()
    for r in candidates:
        if str(r.get("name","")).strip().lower() == name_lower and r.get("id") is not None:
            return int(r["id"])
    return None

# --- 최종 선택 ---
def _llm_pick_best_cody_from_candidates(
    plan: List[Dict[str, Any]],
    candidate_rows: Dict[str, Dict[str, List[Dict]]],
    search_id: int,
    model: str = DEFAULT_CHAT_MODEL,
    db_url: str = DEFAULT_DB_URL
) -> List[Dict[str, Any]]:

    system = "당신은 상의/하의 후보 목록에서 가장 잘 어울리는 코디 조합을 만드는 최고의 코디네이터입니다."
    dev = (
        "규칙:\n"
        "1) 결과(results)의 길이는 plan과 동일.\n"
        "2) 각 코디마다 candidates에서 상의 1개, 하의 1개를 선택.\n"
        "3) 후보가 완전히 비었을 때만 null 허용. 1개 이상이면 반드시 선택.\n"
        "4) reason_selected에는 스타일 매칭표 근거와 조합 이유를 간결히 서술.\n"
        "5) look_style은 plan의 cody_style과 동일(이미 5종 한국어로 제한됨).\n"
        "6) 상품 spec에는 details_intro/material/fit 요약을 담되 과장 금지.\n"
        "7) search_history_product_id는 모델이 채우지 말고 null로 두라(코드가 저장 후 채움).\n"
        "8) 후보가 존재하면 반드시 top/bottom을 null이 아니게 선택하라."
    )
    user = (
        f"스타일 매칭표:\n{STYLE_MATCHING_TABLE}\n\n"
        f"plan: {json.dumps(plan, ensure_ascii=False)}\n\n"
        f"candidates(일부 생략 가능): {json.dumps(candidate_rows, ensure_ascii=False)[:6000]}"
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "system", "content": dev},
            {"role": "user", "content": user},
        ],
        response_format=FINAL_RESPONSE_FORMAT_CODY,
        temperature=0.1
    )
    data = json.loads(resp.choices[0].message.content)
    results = data.get("results", [])

    # 각 결과 아이템에 search_id를 포함
    for res in results:
        res["search_id"] = search_id

    # ----- 사후 보정 이하 원래 코드 유지 -----

    for idx, res in enumerate(results):
        if "cody_style" in res:
            res["look_style"] = res.pop("cody_style", None)

        look_style = res.get("look_style")
        if look_style:
            update_search_history_look_style(search_id, look_style, db_url)

        cands = candidate_rows[idx] if idx < len(candidate_rows) else {"top":[], "bottom":[]}

        if not res.get("top") and cands["top"]:
            res["top"] = _row_to_product(cands["top"][0])
        if not res.get("bottom") and cands["bottom"]:
            res["bottom"] = _row_to_product(cands["bottom"][0])

        for side in ["top", "bottom"]:
            prod = res.get(side)
            if not prod:
                continue
            product_id: Optional[int] = None
            if prod.get("name") and cands.get(side):
                product_id = _find_product_id_by_name(cands[side], prod["name"])
            if product_id is None and prod.get("name"):
                eng = _engine(db_url)
                with eng.connect() as conn:
                    row = conn.execute(
                        text("SELECT id FROM app_product WHERE name = :name LIMIT 1"),
                        {"name": prod["name"]}
                    ).fetchone()
                    if row:
                        product_id = int(row[0])
            if product_id is not None:
                hid = save_search_history_product(search_id, product_id, db_url)
                prod["search_history_product_id"] = hid

    return results


# --- 플랜 실행 + 후보 조회 ---
def json_search_with_cody_plan(
    plan: List[Dict[str, Any]],
    search_id: int,
    db_url: str = DEFAULT_DB_URL,
    model: str = DEFAULT_CHAT_MODEL
) -> Optional[List[Dict[str, Any]]]:
    eng = _engine(db_url)
    candidate_rows = []

    for item in plan:
        top_sql, bottom_sql = item["top_query"], item["bottom_query"]

        # 상의
        try:
            validate_sql(top_sql)
            tops = pd.read_sql(top_sql, eng).to_dict(orient="records")
        except Exception as e:
            print(f"[WARN] 상의 SQL 오류: {e}")
            tops = []
        if not tops:
            fb_top_sql = _fallback_sql("top")
            print(f"[FB] 상의 fallback 실행: {fb_top_sql}")
            tops = pd.read_sql(fb_top_sql, eng).to_dict(orient="records")

        # 하의
        try:
            validate_sql(bottom_sql)
            bottoms = pd.read_sql(bottom_sql, eng).to_dict(orient="records")
        except Exception as e:
            print(f"[WARN] 하의 SQL 오류: {e}")
            bottoms = []
        if not bottoms:
            fb_bottom_sql = _fallback_sql("bottom")
            print(f"[FB] 하의 fallback 실행: {fb_bottom_sql}")
            bottoms = pd.read_sql(fb_bottom_sql, eng).to_dict(orient="records")
        
        candidate_rows.append({"top":tops, "bottom":bottoms})

    return _llm_pick_best_cody_from_candidates(plan, candidate_rows, search_id, model=model, db_url=db_url)

# --- 메인 실행 (CLI 테스트용) ---
if __name__ == "__main__":
    q = " ".join(sys.argv[1:]).strip()

    print("\n=== 0단계: 검색 이력 생성 ===")
    search_id = save_search_history(DEFAULT_DB_URL)
    print(f"search_id = {search_id}")

    print("\n=== 1단계: 벡터DB 조회 ===")
    looks = vedb_list(q, top_k=2)
    if not looks:
        print("❌ 벡터DB에서 스타일을 찾지 못해 코디 추천을 진행할 수 없습니다.")
        sys.exit(1)

    print("\n=== 2단계: 코디 SQL 계획 생성 ===")
    cody_plan = prompting_to_cody_query_plan(looks)
    if not cody_plan:
        print("❌ AI가 코디 계획을 생성하지 못했습니다.")
        sys.exit(1)
    print(json.dumps(cody_plan, ensure_ascii=False, indent=2))

    print("\n=== 3단계: DB 조회 및 최종 코디 선택 ===")
    final_codies = json_search_with_cody_plan(cody_plan, search_id, db_url=DEFAULT_DB_URL)

    print("\n=== FINAL CODY RECOMMENDATIONS ===")
    print(json.dumps(final_codies or [], ensure_ascii=False, indent=2))
