from fastapi import FastAPI, Depends
from sqlalchemy import text
from .deps import get_db

app = FastAPI()

@app.get("/healthz")
def healthz(db = Depends(get_db)):
    # DB 연결/권한/네트워크까지 체크되는 헬스 엔드포인트
    db.execute(text("SELECT 1"))
    return {"status": "ok"}
