import os
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse

# 라우터
from app.controller import routing


origins = [
    "http://localhost:8000",        # 개발 환경 프론트엔드
    "http://127.0.0.1:8000",        # 개발 환경 프론트엔드
    "http://www.looplabel.site",    # 실제 서비스 도메인
    "https://www.looplabel.site",   # 실제 서비스 도메인
]

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# router 등록
app.include_router(routing.router)

ROBOTS_TXT_PATH = Path("robots.txt")

@app.get("/robots.txt")
async def get_robots_txt():
    return FileResponse(ROBOTS_TXT_PATH, media_type="text/plain")

@app.get("/health", summary="AWS Health Check 용", response_description="항상 {status:'ok'} 를 200 으로 반환.")
def health():
    return {"status": "ok"}



if __name__ == "__main__":
    uvicorn.run("main:app", port=8100, reload=False)
