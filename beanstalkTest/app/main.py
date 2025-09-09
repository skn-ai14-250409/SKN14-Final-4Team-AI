from fastapi import FastAPI, HTTPException
import os, inspect
from pydantic import BaseModel
from sqlalchemy import text
from database import SessionLocal
from pipeline_runner import vedb_list, prompting_to_cody_query_plan, json_search_with_cody_plan

# 라우터
from controller import items, users, admins, prompt

app = FastAPI()

# router 등록
app.include_router(items.router)
app.include_router(users.router)
app.include_router(admins.router)
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


# @app.get("/healthz")
# def healthz(db=Depends(get_db)):
#     db.execute(text("SELECT 1"))
#     return {"status": "ok"}

# 테스트용 포인트
@app.get("/db-test")
def db_test():
    try:
        db = SessionLocal()
        result = db.execute(text("SELECT 1")).scalar()
        return {"db_status": "ok", "result": result}
    except Exception as e:
        return {"db_status": "error", "detail": str(e)}
    


