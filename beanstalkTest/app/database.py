import os
from dotenv import load_dotenv
load_dotenv()
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

MYSQL_URL = os.getenv("MYSQL_URL")
if not MYSQL_URL:
    raise RuntimeError("MYSQL_URL not set")

engine = create_engine(
    MYSQL_URL,
    pool_pre_ping=True,
    pool_recycle=1800,
    pool_size=5,
    max_overflow=10,
    future=True,
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
