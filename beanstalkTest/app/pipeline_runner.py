from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import os, json, re, sys
import pandas as pd
import sqlalchemy as sa
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI

load_dotenv()

# --- 기본 설정 ---
client = OpenAI()
DEFAULT_CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4.1-nano")
EMBED_MODEL = os.getenv("EMBED_MODEL")
DEFAULT_DB_URL = os.getenv("APP_DB_URL")

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
    if path_or_url.startswith("sqlite:///") or "://" in path_or_url: return path_or_url
    return f"sqlite:///{Path(path_or_url).as_posix()}"

def validate_sql(sql: str) -> None:
    s = sql.strip().lower()
    if not s.startswith("select "): raise ValueError("SELECT 쿼리만 허용합니다.")
    if 'app_product' not in s: raise ValueError("허용된 테이블만 조회할 수 있습니다.")
    if any(k in s for k in ['update', 'delete', 'insert', 'drop']): raise ValueError("금지된 키워드가 포함된 쿼리입니다.")

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

# 상의/하의 객체 스키마 정의 (search_history_product_id 추가)
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
SYSTEM_PROMPT = """
너는 사용자의 요청과 벡터DB에서 찾아온 스타일 정보를 바탕으로, '상의와 하의'로 구성된 패션 코디 세트를 추천하는 AI 스타일리스트다.
주어진 '스타일 매칭표'를 완벽히 이해하고 활용하여, 각 스타일에 맞는 '상의용 SQL'과 '하의용 SQL'을 한 쌍으로 생성해야 한다.
"""

DEV_RULES_CODY = (
    "아래 스타일 매칭표를 참고해서, 벡터DB에서 찾아온 각 스타일에 가장 적합한 '상의'와 '하의' 검색 키워드를 조합하여 SQL 쿼리 쌍을 생성해라.\n"
    f"{STYLE_MATCHING_TABLE}\n\n"
    "- 테이블: app_product\n"
    "- SELECT 컬럼: id, name, price, image_url, color, url, category, details_intro, material, fit, sustainable_detail\n"
    "- 규칙:\n"
    "  * 모든 쿼리는 `SELECT [컬럼들] FROM app_product WHERE [조건들] LIMIT 10` 구조를 따라야 한다.\n"
    "  * 쿼리 조건은 'OR'를 사용해 더 유연하게 만들어라. 예를 들어, 'work' 스타일의 상의를 찾는다면 `LOWER(category) LIKE '%top%' OR LOWER(details_intro) LIKE '%work%'` 와 같이 작성한다.\n"
    "  * 상의 쿼리는 `LOWER(category) LIKE '%top%'` 등 상의 관련 조건을, 하의 쿼리는 `LOWER(category) LIKE '%pants%'` 등 하의 관련 조건을 반드시 포함해야 한다.\n"
    "- 출력 스키마: `{\"results\": [{\"cody_style\": \"스타일명\", \"top_query\": \"상의용 SQL\", \"bottom_query\": \"하의용 SQL\"}]}`"
)

# --- 데이터베이스 관련 함수 ---

def save_search_history(user_query: str, db_url: str = DEFAULT_DB_URL) -> int:
    """search_history 테이블에 검색 이력 저장하고 ID 반환"""
    engine = sa.create_engine(_ensure_sqlite_url(db_url))
    with engine.connect() as conn:
        result = conn.execute(sa.text("""
            INSERT INTO search_history (user_id, searched_at) 
            VALUES (1, CURRENT_TIMESTAMP)
        """))
        conn.commit()
        # 마지막 삽입된 ID 가져오기
        search_id = conn.execute(sa.text("SELECT last_insert_rowid()")).scalar()
        print(f"✅ search_history에 저장됨 (ID: {search_id})")
        return search_id

def save_search_history_product(search_id: int, product_id: int, db_url: str = DEFAULT_DB_URL) -> int:
    """search_history_product 테이블에 상품 저장하고 ID 반환"""
    engine = sa.create_engine(_ensure_sqlite_url(db_url))
    with engine.connect() as conn:
        result = conn.execute(sa.text("""
            INSERT INTO search_history_product (search_id, product_id) 
            VALUES (:search_id, :product_id)
        """), {"search_id": search_id, "product_id": product_id})
        conn.commit()
        # 마지막 삽입된 ID 가져오기
        history_product_id = conn.execute(sa.text("SELECT last_insert_rowid()")).scalar()
        return history_product_id

def update_search_history_look_style(search_id: int, look_style: str, db_url: str = DEFAULT_DB_URL):
    """search_history 테이블의 look_style 컬럼 업데이트"""
    engine = sa.create_engine(_ensure_sqlite_url(db_url))
    with engine.connect() as conn:
        conn.execute(sa.text("""
            UPDATE search_history 
            SET look_style = :look_style 
            WHERE id = :search_id
        """), {"look_style": look_style, "search_id": search_id})
        conn.commit()
        print(f"✅ search_history (ID: {search_id})의 look_style을 '{look_style}'로 업데이트")

# --- 핵심 기능 함수 ---

def vedb_list(user_query: str, namespace: str = "transcripts", top_k: int = 2) -> List[Dict[str, Any]]:
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

def prompting_to_cody_query_plan(looks: List[Dict[str, Any]], model: str = DEFAULT_CHAT_MODEL) -> List[Dict[str, Any]]:
    user_message = (
        f"벡터DB에서 찾아온 아래 스타일 목록을 보고, 각 스타일에 맞는 '상의용 쿼리'와 '하의용 쿼리'를 한 쌍씩 생성해줘. 총 {len(looks)}개의 쿼리 쌍을 만들어야 해.\n\n"
        f"스타일 목록: {json.dumps(looks, ensure_ascii=False)}"
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "system",
            "content": SYSTEM_PROMPT},
            {
                "role": "system",
                "content": DEV_RULES_CODY
            },
            {
                "role": "user",
                "content": user_message
            }],
        response_format=RESPONSE_FORMAT_CODY_PLAN
    )
    data = json.loads(resp.choices[0].message.content)
    return data.get("results", [])

def _llm_pick_best_cody_from_candidates(plan: List[Dict[str, Any]], candidate_rows: Dict[str, Dict[str, List[Dict]]], search_id: int, model: str = DEFAULT_CHAT_MODEL, db_url: str = DEFAULT_DB_URL) -> List[Dict[str, Any]]:
    system = "당신은 상의와 하의 후보 목록을 보고 가장 잘 어울리는 코디 조합을 만드는 최고의 패션 코디네이터입니다. '스타일 매칭표'를 참고하여 선택 이유를 명확하게 설명해야 합니다."
    dev = (
        "규칙:\n"
        "1. 결과 배열(results)은 입력 plan의 길이와 정확히 같아야 합니다.\n"
        "2. 각 코디마다 `candidates`에서 가장 잘 어울리는 상의 1개와 하의 1개를 선택하여 조합하세요.\n"
        "3. 후보가 없으면 해당 필드(top 또는 bottom)를 null로 두고 `reason_selected`에 '적합한 상품을 찾지 못했습니다.'라고 설명하세요.\n"
        "4. `reason_selected`는 '스타일 매칭표'의 어떤 스타일에 기반했고, 왜 이 조합이 잘 어울리는지 구체적으로 설명해야 합니다.\n"
        "5. `look_style`은 plan의 `cody_style`을 그대로 사용하세요.\n"
        "6. 각 상품 객체 안의 `spec` 필드에는 `details_intro`, `material`, `fit` 정보를 요약해서 간결한 문장으로 채워주세요.\n"
        "7. `search_history_product_id` 필드는 null로 설정하세요 (이후 별도로 처리됩니다)."
    )
    user = (
        f"스타일 매칭표:\n{STYLE_MATCHING_TABLE}\n\n"
        f"코디 계획(plan): {json.dumps(plan, ensure_ascii=False)}\n\n"
        f"후보 상품들(candidates): {json.dumps(candidate_rows, ensure_ascii=False, indent=2)}"
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "system", "content": dev}, {"role": "user", "content": user}],
        response_format=FINAL_RESPONSE_FORMAT_CODY,
        temperature=0.1
    )
    data = json.loads(resp.choices[0].message.content)
    results = data.get("results", [])
    
    # cody_style을 look_style로 변환
    for res in results:
        if "cody_style" in res:
            res["look_style"] = res.pop("cody_style")
    
    # search_history_product에 저장하고 ID 추가
    for result in results:
        # look_style을 search_history에 업데이트
        if result.get("look_style"):
            update_search_history_look_style(search_id, result["look_style"], db_url)
        
        # 상의 상품 처리
        if result.get("top") and result["top"].get("name"):
            # 상품 ID 찾기 (name으로 검색)
            engine = sa.create_engine(_ensure_sqlite_url(db_url))
            with engine.connect() as conn:
                product_result = conn.execute(sa.text("""
                    SELECT id FROM app_product WHERE name = :name LIMIT 1
                """), {"name": result["top"]["name"]}).fetchone()
                
                if product_result:
                    product_id = product_result[0]
                    history_product_id = save_search_history_product(search_id, product_id, db_url)
                    result["top"]["search_history_product_id"] = history_product_id
        
        # 하의 상품 처리
        if result.get("bottom") and result["bottom"].get("name"):
            engine = sa.create_engine(_ensure_sqlite_url(db_url))
            with engine.connect() as conn:
                product_result = conn.execute(sa.text("""
                    SELECT id FROM app_product WHERE name = :name LIMIT 1
                """), {"name": result["bottom"]["name"]}).fetchone()
                
                if product_result:
                    product_id = product_result[0]
                    history_product_id = save_search_history_product(search_id, product_id, db_url)
                    result["bottom"]["search_history_product_id"] = history_product_id
    
    return results

def json_search_with_cody_plan(plan: List[Dict[str, Any]], search_id: int, db_url: str = DEFAULT_DB_URL, model: str = DEFAULT_CHAT_MODEL) -> Optional[List[Dict[str, Any]]]:
    engine = sa.create_engine(_ensure_sqlite_url(db_url))
    candidate_rows = {}

    for item in plan:
        cody_style = item["cody_style"]
        top_sql, bottom_sql = item["top_query"], item["bottom_query"]
        candidate_rows[cody_style] = {"top": [], "bottom": []}
        
        try:
            validate_sql(top_sql)
            candidate_rows[cody_style]["top"] = pd.read_sql(top_sql, engine).to_dict(orient="records")
            validate_sql(bottom_sql)
            candidate_rows[cody_style]["bottom"] = pd.read_sql(bottom_sql, engine).to_dict(orient="records")
        except Exception as e:
            print(f"'{cody_style}' 쿼리 실행 중 오류 발생: {e}")

    final_cody_json = _llm_pick_best_cody_from_candidates(plan, candidate_rows, search_id, model=model, db_url=db_url)
    return final_cody_json

# --- 메인 실행 블록 ---
if __name__ == "__main__":
    q = " ".join(sys.argv[1:]).strip()
    
    print("\n=== 0단계: 검색 이력 저장 ===")
    search_id = save_search_history(q)
    
    print("\n=== 1단계: 벡터DB 조회 ===")
    looks = vedb_list(q, top_k=2)
    
    if not looks:
        print("❌ 벡터DB에서 스타일을 찾지 못해 코디 추천을 진행할 수 없습니다.")
        sys.exit(1)
    
    print("\n=== 2단계: 코디 SQL 계획 생성 ===")
    cody_plan = prompting_to_cody_query_plan(looks)
    
    if not cody_plan:
        print("❌ AI가 코디 계획을 생성하지 못했습니다. 프로그램을 종료합니다.")
        sys.exit(1)
        
    print(json.dumps(cody_plan, ensure_ascii=False, indent=2))
    
    print("\n=== 3단계: DB 조회 및 최종 코디 선택 ===")
    final_codies = json_search_with_cody_plan(cody_plan, search_id)
    
    print("\n=== FINAL CODY RECOMMENDATIONS ===")
    print(json.dumps(final_codies or [], ensure_ascii=False, indent=2))