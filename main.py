import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import text

# 라우터
from app.controller import prompt, routing
from app.database import SessionLocal
from app.pipeline_runner import vedb_list, prompting_to_cody_query_plan, json_search_with_cody_plan

origins = [
    "http://localhost:8000",  # 개발 환경 프론트엔드
    "http://127.0.0.1:8000",  # 개발 환경 프론트엔드
    "http://www.looplabel.com",  # 실제 서비스 도메인
    "https://www.looplabel.com",  # 실제 서비스 도메인
]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# router 등록
# app.include_router(items.router)
# app.include_router(users.router)
# app.include_router(admins.router)
app.include_router(routing.router)
app.include_router(prompt.router)

class QueryBody(BaseModel):
    query: str
    top_k: int = 2
    model: str | None = None
    db_url: str | None = None

@app.post("/", tags=["query"])
def root_post(body: QueryBody):
    try:
        # 1) 검색 로그 저장
        with SessionLocal() as db:
            result = db.execute(
                text("INSERT INTO search_history (look_style, searched_at) VALUES (:look_style, NOW())"),
                {"look_style": body.query}  
            )
            db.commit()
            search_id = result.lastrowid  # 새 search_history.id

        # 2) 기존 파이프라인 실행
        looks = vedb_list(body.query, top_k=body.top_k)
        if not looks:
            raise HTTPException(status_code=400, detail="벡터DB에서 스타일을 찾지 못했습니다.")
        plan = prompting_to_cody_query_plan(looks)
        if not plan:
            raise HTTPException(status_code=400, detail="AI가 코디 계획을 생성하지 못했습니다.")
        print(f"{search_id=}")
        out = json_search_with_cody_plan(
        plan,
        search_id,  
        body.db_url or os.getenv("APP_DB_URL"),
        body.model or os.getenv("CHAT_MODEL"),
        )

        return {
            "query": body.query,
            "top_k": body.top_k,
            "results": out.get("results", [])
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")