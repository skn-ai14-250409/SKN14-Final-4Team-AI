# controller/admins.py

# from fastapi import APIRouter
# from model import mysql_test

# router = APIRouter(
#     prefix='/admins',
#     tags=["admins"],
#     responses={404: {"description": "Not found"}}
# )


# @router.get("/list")
# def list_admin():
#     results = mysql_test.list_admin()
#     return results

# controller/admins.py  (SQLite / 단일 DB URL 사용)
from fastapi import APIRouter, HTTPException
from sqlalchemy import create_engine, text
import os

router = APIRouter(prefix="/admins", tags=["admins"])

DB_URL = os.getenv("APP_DB_URL", "sqlite:///./data/db.sqlite3").strip()
# SQLite는 쓰레드 세이프 옵션 필요
connect_args = {"check_same_thread": False} if DB_URL.startswith("sqlite") else {}
engine = create_engine(DB_URL, connect_args=connect_args, future=True)

@router.get("/ping")
def ping():
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"ok": True, "db_url": DB_URL}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB ping failed: {e}")

@router.get("/tables")
def list_tables():
    """SQLite 내부 테이블 이름을 확인할 때 사용."""
    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            ).all()
        return [r[0] for r in rows]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"List tables failed: {e}")

@router.get("/list")
def list_admin():
    try:
        with engine.connect() as conn:
            rows = conn.execute(text("SELECT * FROM app_product")).mappings().all()
        return list(rows)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB query failed: {e}")

@router.get("/list/{table}")
def list_any_table(table: str, limit: int = 100):
    """
    임의의 테이블을 조회할 수 있는 유틸(간이 버전).
    SQL 인젝션 방지를 위해 테이블명은 영숫자/언더스코어만 허용.
    """
    safe = "".join(ch for ch in table if ch.isalnum() or ch == "_")
    if safe != table:
        raise HTTPException(status_code=400, detail="Invalid table name.")
    try:
        with engine.connect() as conn:
            rows = conn.execute(text(f"SELECT * FROM {safe} LIMIT :n"), {"n": limit}).mappings().all()
        return list(rows)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB query failed: {e}")
