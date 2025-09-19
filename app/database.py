import os

from dotenv import load_dotenv
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.asyncio import AsyncSession

load_dotenv()
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

def _when_local_env():
    SQLITE_URL = os.getenv("SQLITE_URL")
    engine = create_engine(
        SQLITE_URL,
        connect_args={"check_same_thread": False},  # SQLite에서 멀티스레드 사용시 필요
        future=True,
    )
    if not SQLITE_URL:
        raise RuntimeError("SQLITE_URL not set")
    return engine
def _when_prod_env():
    MYSQL_URL = os.getenv("APP_DB_URL")
    engine = create_engine(
        MYSQL_URL,
        pool_pre_ping=True,
        pool_recycle=1800,
        pool_size=5,
        max_overflow=10,
        future=True,
    )
    if not MYSQL_URL:
        raise RuntimeError("MYSQL_URL not set")
    return engine

is_local  = os.getenv("WORKING_ENV", "prod") == "local"

if is_local:  engine = _when_local_env()
else:         engine = _when_prod_env()

SessionLocal      = sessionmaker(engine, autoflush=False, autocommit=False, future=True)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

async def get_async_db():
    db = AsyncSessionLocal()
    try:
        yield db
    finally:
        db.close()
