# Python 3.11 slim 이미지 기반
FROM python:3.11-slim

# 환경 변수 설정
# .pyc 파일 생성과 Python 출력 버퍼링 방지
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 컨테이너 내부 작업 디렉토리
WORKDIR /app

# 의존성 파일 복사 및 설치
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# 앱 코드 복사 v (경로상의 문제 없음)
COPY . .

# 컨테이너가 노출할 포트 (EB는 80 포트를 default로 바라봄)
EXPOSE 8100

# FastAPI 실행 (uvicorn)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8100"]
