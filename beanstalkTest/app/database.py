import os
from dotenv import load_dotenv
load_dotenv()
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

MYSQL_URL = os.getenv("MYSQL_URL")  # EB 환경변수/로컬 .env 둘 다 지원

# 운영 권장 옵션
engine = create_engine(
    MYSQL_URL,
    pool_pre_ping=True,           # 끊긴 커넥션 자동 감지
    pool_recycle=1800,            # 30분마다 재생성(프록시/방화벽 고려)
    pool_size=5,                  # 트래픽에 따라 조절
    max_overflow=10,              # 버스트 허용
    future=True
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
