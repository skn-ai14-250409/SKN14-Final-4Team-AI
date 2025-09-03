# controller/prompt.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import os

# Import pipeline functions
try:
    from prompt_query.pipeline_runner import vedb_list, prompting_to_cody_query_plan, json_search_with_cody_plan
except Exception as e:
    # Fallback: add parent to path if running from subdir
    import sys, pathlib
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
    from prompt_query.pipeline_runner import vedb_list, prompting_to_cody_query_plan, json_search_with_cody_plan  # type: ignore

router = APIRouter(
    prefix="/prompt",
    tags=["prompt"],
    responses={404: {"description": "Not found"}}
)

class PromptRequest(BaseModel):
    query: str
    top_k: int = 2
    model: str | None = None
    db_url: str | None = None

@router.post("/query")
def run_prompt(req: PromptRequest):
    try:
        # 1) Vector DB lookup for looks
        looks = vedb_list(req.query, top_k=req.top_k)
        # 2) Ask LLM to craft SQL plan
        plan = prompting_to_cody_query_plan(looks)
        if not plan:
            raise HTTPException(status_code=400, detail="AI가 코디 계획을 생성하지 못했습니다.")
        # 3) Execute plan against DB and let LLM pick best cody
        db_url = req.db_url or os.getenv("APP_DB_URL")
        model = req.model or os.getenv("CHAT_MODEL")
        final_codies = json_search_with_cody_plan(plan, db_url=db_url, model=model)
        return {
            "query": req.query,
            "top_k": req.top_k,
            "results": final_codies or []
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")

@router.get("/health")
def health():
    return {"status": "ok"}
