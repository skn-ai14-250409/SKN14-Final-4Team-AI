from fastapi import FastAPI, HTTPException
from controller import items, users, admins, prompt
from pydantic import BaseModel
import os
from prompt_query.pipeline_runner import (vedb_list, prompting_to_cody_query_plan, json_search_with_cody_plan)

app = FastAPI()

# router 추가
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
        looks = vedb_list(body.query, top_k=body.top_k)
        if not looks:
            raise HTTPException(status_code=400, detail="벡터DB에서 스타일을 찾지 못했습니다.")
        plan = prompting_to_cody_query_plan(looks)
        if not plan:
            raise HTTPException(status_code=400, detail="AI가 코디 계획을 생성하지 못했습니다.")
        out = json_search_with_cody_plan(
            plan,
            db_url = body.db_url or os.getenv("APP_DB_URL"),
            model  = body.model  or os.getenv("CHAT_MODEL"),
        )
        return {"query": body.query, "top_k": body.top_k, "results": out.get("results", [])}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")