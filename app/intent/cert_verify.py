import os, json
from typing import Optional, Dict, Any, List, Callable
from pydantic import BaseModel
from starlette.responses import HTMLResponse

# 프로젝트 내부 베이스
from app.intent.IntentBase import IntentBase

from app.database import SessionLocal
from app.models import Brand
from sqlalchemy import select

# 외부 라이브러리
from fastapi import HTTPException

# ── 공용 클라이언트
index, oc = IntentBase.get_clients()

# ── 구성 값 (환경변수는 IntentBase.get_clients() 안에서 이미 검증했다고 가정) ──
CHAT_MODEL = os.getenv("CHAT_MODEL")
LLM_ENABLE = os.getenv("LLM_ENABLE", "true").lower() == "true"

# -------------------------
# Pydantic 스키마
# -------------------------
class CertRequest(BaseModel):
    id: str  # Pinecone vector id (필수)

class CertResponse(BaseModel):
    id: Optional[str]
    name: Optional[str]
    summary: str                      # 3~5문장 요약 (친환경 소재/인증 중심)
    certifications: List[str]         # 예: ["GRS", "RCS"]
    source: Dict[str, Any]            # 메타데이터 일부(브랜드/URL 등)

def get_esg_url_by_brand(brand_id: Optional[int] = None, brand_name: Optional[str] = None) -> Optional[str]:
    if brand_id is None and not brand_name:
        return None

    # brand_id가 문자열/float일 수도 있으니 안전 변환
    try:
        bid = int(brand_id) if brand_id is not None else None
    except Exception:
        bid = None

    with SessionLocal() as db:
        if bid is not None:
            url = db.execute(select(Brand.esg_report_url).where(Brand.id == bid)).scalar_one_or_none()
            if url:
                return url
        if brand_name:
            url = db.execute(select(Brand.esg_report_url).where(Brand.brand_name == brand_name)).scalar_one_or_none()
            if url:
                return url
    return None

# -------------------------
# 서비스 레이어
# -------------------------
class CertService:
    ALLOWED_CERTS = ["GRS", "RCS", "OEKO-TEX", "GOTS", "FSC", "bluesign", "B Corp", "Fair Trade"]
    RECYCLE_KEYS  = [
        "리사이클","재활용","재생섬유","재생 폴리에스터","재생 나일론",
        "recycled","recycle","rpet","r-pet","recycled polyester","recycled nylon",
        "pre-consumer","post-consumer","ECONYL","Parley"
    ]

    def __init__(
        self,
        pinecone_index,
        openai_client,
        chat_model: str = CHAT_MODEL,
        llm_enable: bool = LLM_ENABLE,
        esg_url_lookup: Optional[Callable[[Optional[int], Optional[str]], Optional[str]]] = None,
    ):
        self.index = pinecone_index
        self.oc = openai_client
        self.chat_model = chat_model
        self.llm_enable = llm_enable
        self.esg_url_lookup = esg_url_lookup  # (brand_id, brand_name) -> esg_url

    @staticmethod
    def _id_to_str(v):
        if v is None: return None
        if isinstance(v, float) and v.is_integer(): return str(int(v))
        return str(v)

    @staticmethod
    def _contains(texts: List[str], keys: List[str]) -> bool:
        blob = " ".join(t for t in texts if t).lower()
        return any(k.lower() in blob for k in keys)

    def llm_summarize(self, spec_text: str, name: Optional[str]=None, brand: Optional[str]=None, esg_url: Optional[str]=None) -> Dict[str, Any]:
        spec_trim = (spec_text or "").strip()
        if len(spec_trim) > 3800:
            spec_trim = spec_trim[:3800] + " ..."

        CTA_CERT = "다만 GRS/RCS 같은 공식 인증 정보는 제공되지 않았어요. 원하시면, 재활용 인증(GRS/RCS)이 확인된 유사 제품으로 추천해 드릴까요?"
        CTA_ESG  = (f"더 자세한 근거와 데이터는 브랜드 ESG 보고서에서 확인해 보세요: {esg_url}"
                    if esg_url else
                    "더 자세한 근거와 데이터는 해당 브랜드의 ESG 보고서에서 확인해 보세요.")

        if not self.llm_enable or not self.oc:
            fb = (f"{name or '이 제품'}은(는) 제공된 스펙을 바탕으로 친환경 소재/인증 정보를 요약할 수 있습니다. "
                  "현재 요약 기능이 비활성화되어 간단 안내만 제공합니다.")
            return {"summary": fb, "certifications": []}

        system = (
            "당신은 지속가능 패션 인증 전문가 겸 UX 라이터입니다. "
            "사실 기반으로 간결하고 친절한 한국어로 작성하고, 과장/광고 문구는 제외하세요."
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

[인증 키워드 추출 규칙]
- 아래 목록 중 실제로 명시된 것만 대문자 표기로 추출(없으면 []): {self.ALLOWED_CERTS}

[출력: JSON 하나의 객체만]
{{
  "summary": "<요약문 (3~5문장), 규칙에 따라 마지막 문장에 적절한 CTA 포함>",
  "certifications": ["GRS", "RCS"]  // 없으면 []
}}

제품명: {name or "미상"}
브랜드: {brand or "미상"}
spec:
\"\"\"{spec_trim}\"\"\"
""".strip()

        try:
            resp = self.oc.chat.completions.create(
                model=self.chat_model,
                temperature=0.2,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
            )
            raw = (resp.choices[0].message.content or "").strip()
        except Exception:
            return {"summary": "요약 생성 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.", "certifications": []}

        # JSON 파싱
        try:
            data = json.loads(raw)
            summary = (data.get("summary") or "").strip()
            certs_in = data.get("certifications") or []
            certs = sorted({c for c in self.ALLOWED_CERTS if any(c.lower() == x.lower() for x in certs_in)})
        except Exception:
            low = raw.lower()
            certs = sorted({c for c in self.ALLOWED_CERTS if c.lower() in low})
            summary = raw

        # 부정 문구 제거
        for p in ["해당 정보 없음","확인할 수 없습니다","확인 불가","제공되지 않았습니다","정보가 없습니다","[none]","[해당 정보 없음]"]:
            summary = summary.replace(p, "")
        summary = " ".join(summary.split()).strip()

        # CTA 최종 보정
        has_recycle = self._contains([summary, spec_trim], self.RECYCLE_KEYS)
        if certs or has_recycle:
            if not summary.endswith(("다.","요.","!","…")) and summary:
                summary += " "
            if CTA_ESG not in summary:
                summary = (summary + CTA_ESG).strip()
            summary = summary.replace(CTA_CERT, "").strip()
        else:
            if not summary.endswith(("다.","요.","!","…")) and summary:
                summary += " "
            if CTA_CERT not in summary:
                summary = (summary + CTA_CERT).strip()

        return {"summary": summary, "certifications": certs}

    def fetch_and_summarize(self, vector_id: str) -> Dict[str, Any]:
        try:
            fetched = self.index.fetch(ids=[vector_id])
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Pinecone fetch 실패: {e}")

        vectors = getattr(fetched, "vectors", None)
        if isinstance(vectors, dict):
            rec = vectors.get(vector_id)
        elif hasattr(fetched, "get") and callable(getattr(fetched, "get")):
            rec = fetched.get("vectors", {}).get(vector_id)
        else:
            rec = None

        if not rec:
            raise HTTPException(status_code=404, detail=f"id={vector_id} 레코드를 찾을 수 없습니다. (벡터 ID 확인)")

        meta = (rec.get("metadata") or {}) if isinstance(rec, dict) else getattr(rec, "metadata", {}) or {}
        spec = meta.get("spec") or meta.get("description") or meta.get("text")
        if not spec:
            raise HTTPException(status_code=422, detail="spec(설명) 필드가 없어 요약할 수 없습니다.")

        name     = meta.get("name")
        brand    = meta.get("brand")
        brand_id = meta.get("brand_id") or meta.get("brandId")

        # ESG URL RDB 조회 함수 연결
        esg_url = None
        if self.esg_url_lookup:
            try:
                esg_url = self.esg_url_lookup(brand_id, brand)
            except Exception:
                esg_url = None

        res = self.llm_summarize(spec_text=spec, name=name, brand=brand, esg_url=esg_url)

        return {
            "id": self._id_to_str(meta.get("id")) or self._id_to_str(vector_id),
            "name": name,
            "summary": res["summary"],
            "certifications": res["certifications"],
            "source": {
                **{k: meta.get(k) for k in ["id","name","brand","brand_id","url","sku","model"] if k in meta},
                **({"esg_report_url": esg_url} if esg_url else {})
            }
        }

# ── 인텐트 클래스 ──
class CertVerify(IntentBase):
    def __init__(self, service: Optional[CertService] = None):
        prompt = ("""
당신은 재활용소재로 패션제품을 만드는 회사에서 고문으로 일하고 있는 친환경 활동가입니다.
친환경 인증과 관련된 질문에만 답변합니다.
당신이 알고있는 수준으로만 답변하고, 일체의 거짓이나 꾸밈이 없어야 하며, 당신이 모르는 정보라면 모른다고 답변합니다.
1. 답변에서 인사말/당신을 소개하는 문구/불필요한 미사여구는 제외합니다.
2. 답변은 공백을 제외했을 때 250자 내외로 작성합니다.
""")
        super().__init__(prompt)
        self.service = service or CertService(index, oc, CHAT_MODEL, LLM_ENABLE, esg_url_lookup=get_esg_url_by_brand)


    # ✅ 오케스트레이터 표준 엔트리포인트: 항상 dict 반환
    async def run(self, query: str, slots: Dict[str, Any] | None = None, profile: Dict[str, Any] | None = None, **kwargs):
        slots = slots or {}

        # 1) 제품 기준 인증 요약 (벡터 ID/제품 ID가 있으면 Pinecone에서 spec 요약)
        vector_id = (
            slots.get("vector_id") or slots.get("id") or slots.get("product_id")
            or kwargs.get("id") or kwargs.get("vector_id")
        )
        if vector_id:
            try:
                payload = self.service.fetch_and_summarize(str(vector_id))
                # payload: {"id","name","summary","certifications","source":{...}}
                text = payload.get("summary", "").strip()
                if not text:
                    text = "해당 제품의 인증 요약을 불러오지 못했습니다. 제품 스펙 정보가 부족할 수 있습니다."
                return {
                    "text": text,
                    "html": "",
                    "slots": slots,
                    "meta": {
                        "certifications": payload.get("certifications", []),
                        "source": payload.get("source", {}),
                        "mode": "by_product",
                    },
                }
            except Exception as e:
                # 제품 경로 실패 시 LLM 일반 설명으로 폴백
                fail_note = f"[제품기반 요약 실패] {e}"

        # 2) 인증 스킴 자체 설명 (GRS/RCS 등)
        scheme = (slots.get("인증") or slots.get("cert") or "").strip().upper()
        if scheme in {"GRS", "RCS"}:
            sys = (
                "당신은 지속가능 패션 인증 전문가입니다. "
                "과장 없이 사실 기반으로, 한국어로 간결하게 답하세요."
            )
            user = f"""
다음 인증 스킴({scheme})에 대해 4~6문장으로 핵심만 설명하세요.
- 정의/적용범위(소재/완제품), 체인오브커스터디(CoC) 여부
- 주요 검증 항목(재활용 함량, 추적성, 관리 요건 등)
- 소비자가 라벨에서 확인할 수 있는 점/오해하기 쉬운 점
- 마크다운/목록/해시태그 금지
질문: {query}
""".strip()
            text = self.ask_llm(query=query, prompt=f"{sys}\n\n{user}", **kwargs)
            if not text.strip():
                text = f"{scheme} 인증의 핵심 요건을 간단히 정리했습니다. 추가로 궁금한 제품이나 브랜드를 알려주시면 해당 사례로 설명을 보완해 드릴게요."
            return {
                "text": text,
                "html": "",
                "slots": slots,
                "meta": {"mode": "by_scheme", "scheme": scheme},
            }

        # 3) 일반 인증 질의(스킴 미지정) — 차이/유효성/조회 등
        sys = (
            "당신은 지속가능 패션 인증 전문가입니다. "
            "과장 없이 사실 기반으로, 한국어로 간결하게 답하세요."
        )
        user = f"""
사용자의 질문은 친환경 인증과 관련되어 있습니다. 4~6문장으로 핵심을 설명하고 안내하세요.
- 자주 묻는 주제(정의/차이/유효성/번호 조회/적용범위)를 문맥에 맞게 답변
- 제품 사례가 필요하면 '제품 링크나 이름을 알려달라'는 안내 1문장
- 마크다운/목록/해시태그 금지
질문: {query}
""".strip()
        text = self.ask_llm(query=query, prompt=f"{sys}\n\n{user}", **kwargs)
        if not text.strip():
            text = "인증의 정의·차이·조회 방법 등 어떤 점이 궁금하신지 조금만 더 구체적으로 알려주시면 정확히 안내드리겠습니다."

        return {
            "text": text,
            "html": "",
            "slots": slots,
            "meta": {"mode": "generic"},
        }
    
    def __call__(self, **kwargs):
        with_voice = kwargs.get("with_voice", False)
        result = self.ask_llm(**kwargs)
        if with_voice:
            voice  = self.get_voice(result, with_voice=with_voice)
            result += f"<audio controls loop='false' src={voice}></audio>"
        return HTMLResponse(result, status_code=200)