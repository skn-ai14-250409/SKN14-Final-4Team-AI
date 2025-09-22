import os
import json
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pinecone import Pinecone
from openai import OpenAI
# from sqlalchemy import text
# from app.database import SessionLocal
from fastapi.responses import JSONResponse
from dotenv import load_dotenv, find_dotenv

# -------------------------
# .env 로드 (가장 먼저)
# -------------------------
_ = load_dotenv(find_dotenv(usecwd=True))

# -------------------------
# 환경변수
# -------------------------
PINECONE_API_KEY        = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_PRODUCT  = os.getenv("PINECONE_INDEX_PRODUCT")
PINECONE_INDEX_HOST = os.getenv("PINECONE_INDEX_HOST")
OPENAI_API_KEY          = os.getenv("OPENAI_API_KEY")
CHAT_MODEL              = os.getenv("CHAT_MODEL", "gpt-4o-mini")
LLM_ENABLE              = os.getenv("LLM_ENABLE", "true").lower() == "true"

if not (PINECONE_API_KEY and PINECONE_INDEX_PRODUCT and OPENAI_API_KEY):
    raise RuntimeError("환경변수 PINECONE_API_KEY, PINECONE_INDEX_PRODUCT, OPENAI_API_KEY 설정 필요")


# -------------------------
# 클라이언트 초기화
# -------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)
if PINECONE_INDEX_HOST:
    index = pc.Index(host=PINECONE_INDEX_HOST)
else:
    index = pc.Index(PINECONE_INDEX_PRODUCT)
index = pc.Index(PINECONE_INDEX_PRODUCT) 
oc = OpenAI(api_key=OPENAI_API_KEY)

# --- 공용 헬퍼: 존재여부 로깅(민감값은 마스킹) ---
def _mask(v): return "SET" if v else "MISSING"
print("[ENV] PINECONE_API_KEY=%s, INDEX_PRODUCT=%s, INDEX_HOST=%s, OPENAI_API_KEY=%s, CHAT_MODEL=%s, LLM_ENABLE=%s" %
      (_mask(PINECONE_API_KEY), _mask(PINECONE_INDEX_PRODUCT), _mask(PINECONE_INDEX_HOST),
       _mask(OPENAI_API_KEY), CHAT_MODEL, LLM_ENABLE))

# -------------------------
# FastAPI
# -------------------------
app = FastAPI(title="Products Certification API", version="1.0.0")

# -------------------------
# 모델
# -------------------------
class CertRequest(BaseModel):
    id: str  # Pinecone vector id (필수)

class CertResponse(BaseModel):
    id: Optional[str]
    name: Optional[str]
    summary: str                      # 3~5문장 요약 (친환경 소재/인증 중심)
    certifications: List[str]         # 예: ["GRS", "RCS"]
    source: Dict[str, Any]            # 메타데이터 일부(브랜드/URL 등)

# -------------------------
# 유틸
# -------------------------

# ESG 보고서 URL 조회 (brand_id → URL)하는 함수 추가



def llm_summarize(spec_text: str, name: Optional[str]= None, brand: Optional[str]= None, esg_url: Optional[str] = None) -> Dict[str, Any]:
    """
    목적:
      - spec에서 '친환경 소재/인증'만 3~5문장으로 요약
      - 요약문에는 '확인 불가/해당 정보 없음' 등 부정 문구 노출 금지
      - (A) certifications에 'GRS/RCS' 존재 또는 '리사이클/재활용' 언급 → ESG 보고서 CTA 추가
      - (B) 둘 다 없으면 → 인증 제품 추천 CTA 추가
      - certifications 배열은 요약문과 분리(요약문에 [NONE] 등 금지)

    반환:
      {"summary": <str>, "certifications": <List[str]>}
    """
    # --- 전제 검사 ---
    spec_trim = (spec_text or "").strip()
    if len(spec_trim) > 3800:  # 토큰 여유
        spec_trim = spec_trim[:3800] + " ..."

    CTA_CERT = "다만 GRS/RCS 같은 공식 인증 정보는 제공되지 않았어요. 원하시면, 재활용 인증(GRS/RCS)이 확인된 유사 제품으로 추천해 드릴까요?"
    # ESG CTA: URL 있으면 링크 포함
    CTA_ESG = (
        f"더 자세한 근거와 데이터는 브랜드 ESG 보고서에서 확인해 보세요: {esg_url}"
        if esg_url else
        "더 자세한 근거와 데이터는 해당 브랜드의 ESG 보고서에서 확인해 보세요."
    )
    ALLOWED_CERTS = ["GRS", "RCS", "OEKO-TEX", "GOTS", "FSC", "bluesign", "B Corp", "Fair Trade"]
    RECYCLE_KEYS = [
        # 한글
        "리사이클", "재활용", "재생섬유", "재생 폴리에스터", "재생 나일론",
        # 영문/약어
        "recycled", "recycle", "rpet", "r-pet", "recycled polyester", "recycled nylon", "pre-consumer", "post-consumer"
    ]
    
    # LLM 비활성화 시 폴백
    if not LLM_ENABLE or not oc:
        fb = (
            f"{name or '이 제품'}은(는) 제공된 스펙을 바탕으로 친환경 소재/인증 정보를 요약할 수 있습니다. "
            "현재 요약 기능이 비활성화되어 간단 안내만 제공합니다."
        )
        return {"summary": fb, "certifications": []}

    # --- 프롬프트(출력 JSON 강제) ---
    system = (
        "당신은 지속가능 패션 인증 전문가 겸 UX 라이터입니다. "
        "사실 기반으로 간결하고 친절한 한국어를 사용하고, 과장/광고 문구는 제외하세요."
    )
    user = f"""
다음 제품의 스펙을 바탕으로 '친환경 소재/인증'에 한정해 요약을 작성하고, 인증 키워드를 분리해 주세요.

[요약문 규칙]
- 3~5문장.
- 친환경 소재(예: 린넨/오가닉 코튼/리사이클 폴리 등)의 특성을 사용자 관점에서 이해하기 쉽게 1~2문장 설명(물 사용량·통기성·내구성·생분해성 등) — 사실 범위 내.
- 인증 정보가 있으면 스킴(예: GRS, RCS)·대상(소재/완제품)·재활용 함량(%)·원산지·트레이서빌리티·인증번호 등 구체적으로.
- 요약문에는 '확인 불가/해당 정보 없음' 등 부정 문구를 쓰지 마세요.
- 요약문에는 대괄호 리스트, 해시태그, [NONE] 표기 금지.

[CTA 규칙(요약문 마지막 문장)]
- 아래 두 조건 중 하나라도 참이면 ESG CTA를 마지막 문장으로 추가:
  ① 인증 키워드(GRS/RCS 등)가 존재하거나 ② 스펙에 '리사이클/재활용' 원단 언급이 있음.
  ESG CTA: "{CTA_ESG}"
- 위 두 조건이 모두 거짓(= 인증 키워드 없음 AND 리사이클 언급 없음)이면 인증추천 CTA를 마지막 문장으로 추가:
  인증추천 CTA: "{CTA_CERT}"

[인증 키워드 규칙]
- 아래 목록 중 실제로 명시된 것만 대문자 표기로 추출: {ALLOWED_CERTS}
- 전혀 없으면 빈 배열 [].

출력은 반드시 아래 JSON 하나의 객체로만:
{{
  "summary": "<요약문 (3~5문장), 규칙에 따라 마지막 문장에 적절한 CTA 포함)>",
  "certifications": ["GRS", "RCS"]  // 없으면 []
}}

제품명: {name or "미상"}
브랜드: {brand or "미상"}
spec:
\"\"\"{spec_trim}\"\"\"
""".strip()

    # --- LLM 호출 ---
    try:
        resp = oc.chat.completions.create(
            model=CHAT_MODEL,
            temperature=0.2,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
        )
        raw = (resp.choices[0].message.content or "").strip()
    except Exception:
        return {
            "summary": "요약 생성 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.",
            "certifications": []
        }

    def _contains_recycle(*texts: str) -> bool:
        blob = " ".join([t for t in texts if t]).lower()
        return any(key.lower() in blob for key in RECYCLE_KEYS)
    
    # JSON 파싱
    summary = ""
    certs: List[str] = []
    try:
        data = json.loads(raw)
        summary = (data.get("summary") or "").strip()
        certs_in = data.get("certifications") or []
        # 화이트리스트 필터링 + 표기 통일
        certs = sorted({c for c in ALLOWED_CERTS if any(c.lower() == x.lower() for x in certs_in)})
    except Exception:
        # 모델이 JSON을 안 지켰을 때의 폴백: 간단 키워드 탐지
        text_low = raw.lower()
        certs = sorted({c for c in ALLOWED_CERTS if c.lower() in text_low})
        summary = raw

    # ---- 후처리: 부정 문구 제거 ----
    NEG_PATTERNS = [
        "해당 정보 없음", "확인할 수 없습니다", "확인 불가",
        "제공되지 않았습니다", "정보가 없습니다", "[none]", "[해당 정보 없음]"
    ]
    for p in NEG_PATTERNS:
        summary = summary.replace(p, "")
    summary = " ".join(summary.split()).strip()

    # CTA 결정
    has_recycle = _contains_recycle(summary, spec_trim)
    if certs or has_recycle:
        # 인증이 있거나 리사이클 언급이 있으면 ESG CTA
        if not summary.endswith(("다.", "요.", "!", "…")) and summary:
            summary += " "
        if CTA_ESG not in summary:
            summary = (summary + CTA_ESG).strip()
        # 혹시 기존 인증 추천 CTA가 붙어 있으면 제거
        summary = summary.replace(CTA_CERT, "").strip()
    else:
        # 인증 키워드도 없고 리사이클 언급도 없으면 인증 제품 추천 CTA
        if not summary.endswith(("다.", "요.", "!", "…")) and summary:
            summary += " "
        if CTA_CERT not in summary:
            summary = (summary + CTA_CERT).strip()

    return {"summary": summary, "certifications": certs}

# -------------------------
# 메인 엔드포인트
# -------------------------
def _id_to_str(v):
    if v is None:
        return None
    # float로 들어온 정수값 736.0 → "736"
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)

@app.post("/certification-info", response_model=CertResponse)
def get_certification_info(req: CertRequest):
    # 요청 검증
    if not req.id:
        raise HTTPException(status_code=400, detail="id를 확인해주세요. (벡터 ID 필요)")

    # Pinecone fetch
    try:
        fetched = index.fetch(ids=[req.id])
    except Exception as e:
        # 서버 내부/외부 문제를 구분해서 리턴(디버깅 편의)
        raise HTTPException(status_code=502, detail=f"Pinecone fetch 실패: {e}")

    # 응답 방어적 파싱
    vectors = getattr(fetched, "vectors", None)
    if isinstance(vectors, dict):
        rec = vectors.get(req.id)
    elif hasattr(fetched, "get") and callable(getattr(fetched, "get")):
        # 혹시 dict로 리턴될 경우를 방어
        rec = fetched.get("vectors", {}).get(req.id)
    else:
        rec = None

    if not rec:
        # 벡터 ID가 맞는지 힌트 제공
        raise HTTPException(
            status_code=404,
            detail=f"id={req.id} 레코드를 찾을 수 없습니다. (벡터 ID인지 확인: 예시 UUID 형태)"
        )

    meta = (rec.get("metadata") or {}) if isinstance(rec, dict) else getattr(rec, "metadata", {}) or {}
    spec = meta.get("spec") or meta.get("description") or meta.get("text")
    if not spec:
        raise HTTPException(status_code=422, detail="spec(설명) 필드가 없어 요약할 수 없습니다.")

    name = meta.get("name")
    res = llm_summarize(spec, name)

    meta_id = meta.get("id")
    return CertResponse(
        id = _id_to_str(meta_id) or _id_to_str(req.id),  # ← 여기서 문자열로 고정
        name = name,
        summary = res["summary"],
        certifications = res["certifications"],
        source = {k: meta.get(k) for k in ["id","name","url","sku","model"] if k in meta}, # 추후 url -> esg_url로 변경
)

# --- 선택: 전역 예외 핸들러(500 원인 즉시 노출) ---
@app.exception_handler(Exception)
async def all_exception_handler(request, exc):
    # 프로덕션에선 상세 숨기고 로그로만 남기세요
    return JSONResponse(status_code=500, content={"error": str(exc), "type": exc.__class__.__name__})