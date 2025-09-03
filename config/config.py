# config/config.py
from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ---- App ----
PORT: int = int(os.getenv("PORT", "8080"))

# 기본 SQLite 경로 (루트/data/db.sqlite3)
DEFAULT_SQLITE_URL = "sqlite:///./data/db.sqlite3"

def _resolve_db_url(url: str | None = None) -> str:
    """
    APP_DB_URL 우선, 없으면 기본 SQLite.
    공백/None 안전 처리.
    """
    val = (url or os.getenv("APP_DB_URL") or DEFAULT_SQLITE_URL).strip()
    return val

# 최종 사용될 DB URL
APP_DB_URL: str = _resolve_db_url()

def build_sqlalchemy_engine(url: str | None = None, *, echo: bool = False):
    """
    SQLAlchemy 엔진 생성 헬퍼.
    - SQLite면 check_same_thread=False 자동 적용
    - ./data 폴더 자동 생성 시도
    """
    from sqlalchemy import create_engine

    db_url = _resolve_db_url(url)
    connect_args = {}
    if db_url.startswith("sqlite"):
        try:
            Path("data").mkdir(exist_ok=True)  # 상대경로 사용 시 data 폴더 보장
        except Exception:
            pass
        connect_args = {"check_same_thread": False}
    return create_engine(
        db_url,
        connect_args=connect_args,
        pool_pre_ping=True,
        future=True,
        echo=echo,
    )

# ---- MySQL (배포 시 사용할 수 있게 옵션만 유지) ----
MYSQL_DB_CONFIG = {
    "host": os.getenv("MYSQL_HOST", "127.0.0.1"),
    "port": int(os.getenv("MYSQL_PORT", "3306")),
    "database": os.getenv("MYSQL_DB", "RDB"),   # 배포 시 .env에서 실제 DB명으로 지정
    "user": os.getenv("MYSQL_USER", "root"),
    "password": os.getenv("MYSQL_PASSWORD", "mysql2025"),
    "pool_name": os.getenv("MYSQL_POOL_NAME", "mysql_pool"),
    "pool_size": int(os.getenv("MYSQL_POOL_SIZE", "10")),
    "connection_timeout": int(os.getenv("MYSQL_CONN_TIMEOUT", "5")),
}

def mysql_connect():
    """mysql-connector-python 사용 연결 헬퍼 (필요 시)"""
    import mysql.connector
    return mysql.connector.connect(**MYSQL_DB_CONFIG)
