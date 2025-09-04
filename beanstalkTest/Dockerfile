# Python 3.11 slim 이미지 기반
FROM python:3.11-slim

# 컨테이너 내부 작업 디렉토리
WORKDIR /app

# 시스템 의존성 설치 (필요 없으면 생략 가능)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# 파이썬 의존성 설치(beanstalkTest의 req.. 사용 시 상대경로로 표시)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 앱 코드 복사 v (경로상의 문제 없음)
COPY app .

# 컨테이너가 노출할 포트 (EB는 80 포트를 default로 바라봄)
EXPOSE 8100

# FastAPI 실행 (uvicorn)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8100"]
