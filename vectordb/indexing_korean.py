# ===== 1) 필요한 라이브러리 import =====
# 필요한 패키지들이 이미 설치되어 있다고 가정

# ===== 2) 환경변수 설정 (.env 파일 로드) =====
import os, glob
from pathlib import Path

# .env 파일 로드
def load_env_file():
    env_file = Path(".env")
    if env_file.exists():
        with open(env_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

# .env 파일 로드 시도
try:
    load_env_file()
except Exception as e:
    print(f"⚠️  .env 파일 로드 실패: {e}")

# Colab userdata 시도 (Colab 환경에서만)
try:
    from google.colab import userdata
    os.environ["OPENAI_API_KEY"]   = userdata.get("FINAL_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY","")
    os.environ["PINECONE_API_KEY"] = userdata.get("PINECONE_API_KEY") or os.getenv("PINECONE_API_KEY","")
    # LangSmith (선택)
    for k in ["LANGSMITH_TRACING","LANGSMITH_ENDPOINT","LANGSMITH_API_KEY","LANGSMITH_PROJECT"]:
        v = userdata.get(k) or os.getenv(k)
        if v: os.environ[k] = v
except Exception:
    pass

assert os.environ.get("OPENAI_API_KEY"),   "❌ OPENAI_API_KEY가 없습니다."
assert os.environ.get("PINECONE_API_KEY"), "❌ PINECONE_API_KEY가 없습니다."

# ===== 3) 로컬 이름 충돌 방지 (./pinecone.py 등) =====
_conflicts = [p for p in glob.glob("./pinecone*") if Path(p).exists()]
if _conflicts:
    raise RuntimeError(f"❌ 작업 폴더 이름 충돌: {_conflicts}  (이름 변경 후 재실행)")

# ===== 4) 임포트 =====
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
print("OK: imports")

# ===== 설정 (한국어 최적화) =====
INDEX_NAME   = "transcripts-korean"
NAMESPACE    = "transcripts-kr"
CLOUD, REGION = "aws", "us-east-1"
METRIC       = "cosine"

# 한국어에 더 적합한 모델 설정
OPENAI_EMBEDDING_MODEL = "text-embedding-3-large"  # 한국어 성능이 더 좋음
OPENAI_LLM_MODEL       = "gpt-4o-mini"             # 한국어 지원

MODEL_DIMS = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072}
EMBED_DIM  = MODEL_DIMS[OPENAI_EMBEDDING_MODEL]

# ===== 텍스트 전처리 & 청크 =====
import re, uuid

def clean_text(t: str) -> str:
    # 한국어 텍스트 정리
    t = re.sub(r"[ \t]+\n", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    t = t.replace("–", "-").replace("—", "-")
    # 한국어 특수 문자 정리
    t = re.sub(r"\[음악\]", "", t)  # [음악] 제거
    t = re.sub(r"\[.*?\]", "", t)   # 대괄호 내용 제거
    t = re.sub(r"\s+", " ", t)      # 연속 공백 정리
    return t.strip()

def split_sections(t: str):
    # YouTube 자막 파일의 경우 연속된 텍스트이므로 General 섹션으로 처리
    # 패션 관련 키워드로 섹션을 자동 분류
    text_lower = t.lower()
    
    # 패션 관련 키워드로 섹션 분류
    if any(keyword in text_lower for keyword in ['출근', '직장', '오피스', '업무', '회사']):
        section = "출근룩"
    elif any(keyword in text_lower for keyword in ['데이트', '연애', '남친', '여친', '커플']):
        section = "데이트룩"
    elif any(keyword in text_lower for keyword in ['캐주얼', '일상', '데일리', '편한']):
        section = "캐주얼룩"
    elif any(keyword in text_lower for keyword in ['미니멀', '심플', '깔끔', '베이직']):
        section = "미니멀룩"
    elif any(keyword in text_lower for keyword in ['여름', 'summer', '시원한', '가벼운']):
        section = "여름룩"
    elif any(keyword in text_lower for keyword in ['겨울', 'winter', '따뜻한', '보온']):
        section = "겨울룩"
    else:
        section = "General"
    
    return [{"section": section, "text": t}]

def chunk_by_paragraph(t, max_chars=1600, overlap_chars=200):
    # YouTube 자막은 연속된 텍스트이므로 문장 단위로 분할
    # 문장 구분자: 마침표, 느낌표, 물음표
    sentences = re.split(r'[.!?]+', t)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    chunks, cur = [], ""
    for sentence in sentences:
        # 현재 청크에 문장을 추가했을 때의 길이
        potential_chunk = (cur + " " + sentence).strip() if cur else sentence
        
        if len(potential_chunk) <= max_chars:
            cur = potential_chunk
        else:
            # 현재 청크가 있으면 저장
            if cur:
                chunks.append(cur)
            # 새 청크 시작
            cur = sentence
    
    # 마지막 청크 추가
    if cur:
        chunks.append(cur)
    
    # 청크가 없으면 전체 텍스트를 하나의 청크로
    if not chunks:
        chunks = [t[:max_chars]] if len(t) > max_chars else [t]
    
    # 오버랩 처리
    if overlap_chars and len(chunks) > 1:
        for k in range(1, len(chunks)):
            prev = chunks[k-1]
            if len(prev) > overlap_chars:
                overlap = prev[-overlap_chars:]
                chunks[k] = overlap + " " + chunks[k]
    
    return chunks

# ===== occasion 자동 태깅 =====
def infer_occasion(text: str):
    t = text.lower()
    tags = set()
    if re.search(r"\b(office|work|commute|promotion|slacks|shirt|tweed|mary jane)\b", t):
        tags.add("work")
    if re.search(r"\b(festival|concert)\b", t):
        tags.add("festival")
    if re.search(r"\b(travel|airport)\b", t):
        tags.add("travel")
    if re.search(r"\b(casual|weekend|hobo bag)\b", t):
        tags.add("casual")
    return list(tags or {"general"})

    # ===== Pinecone 초기화 & 인덱스 생성/접속 =====
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
if INDEX_NAME not in {i.name for i in pc.list_indexes()}:
    pc.create_index(
        name=INDEX_NAME, dimension=EMBED_DIM, metric=METRIC,
        spec=ServerlessSpec(cloud=CLOUD, region=REGION),
    )
    print(f"✅ Created index: {INDEX_NAME}")
index = pc.Index(INDEX_NAME)

# ===== 임베딩 인스턴스 =====
emb = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)
print("OK: Pinecone & Embeddings ready")

from pathlib import Path

def ingest_files(file_paths, *, namespace=NAMESPACE, add_snippet=True):
    total_vectors, doc_map = 0, {}
    for f in file_paths:
        path = Path(f)
        assert path.exists(), f"❌ 파일 없음: {path.resolve()}"

        text = clean_text(path.read_text(encoding="utf-8"))
        docs = split_sections(text)
        
        print(f"📄 처리 중: {path.name} - 텍스트 길이: {len(text)}, 섹션 수: {len(docs)}")

        # 레코드 구성(스니펫/occasion 메타 포함)
        records = []
        for d in docs:
            chunks = chunk_by_paragraph(d["text"], max_chars=1600, overlap_chars=200)
            print(f"  섹션: {d['section']} - 청크 수: {len(chunks)}")
            for ch in chunks:
                md = {
                    "section": d["section"],
                    "season": "summer",
                    "exposure": "non_revealing",
                    "source": str(path),
                    "occasion": infer_occasion(ch)
                }
                if add_snippet:
                    md["snippet"] = ch[:220].replace("\n", " ")
                records.append({"text": f"Section: {d['section']}\n\n{ch}", "metadata": md})

        # 업서트
        if not records:
            print(f"  ⚠️  {path.name}: 레코드가 없어서 건너뜀")
            continue
        doc_id = str(uuid.uuid4())
        texts  = [r["text"] for r in records]
        metas  = [r["metadata"] for r in records]
        ids    = [f"{doc_id}-{i}" for i in range(len(records))]

        vectors = []
        BATCH = 64
        for s in range(0, len(texts), BATCH):
            embs = emb.embed_documents(texts[s:s+BATCH])
            for j, vec in enumerate(embs):
                k = s + j
                vectors.append({
                    "id": ids[k],
                    "values": vec,
                    "metadata": {**metas[k], "doc_id": doc_id, "chunk_id": k, "lang": "ko"},
                })

        index.upsert(vectors=vectors, namespace=namespace)
        total_vectors += len(vectors)
        doc_map[doc_id] = str(path.name)
        print(f"✅ Upserted {len(vectors)} vectors from '{path.name}' (ns='{namespace}')")
    print(f"🎯 Done. total_vectors={total_vectors}, files={len(file_paths)}, namespace='{namespace}'")
    return {"total_vectors": total_vectors, "doc_map": doc_map}

# transcripts_output 폴더의 모든 개별 텍스트 파일들을 동적으로 찾기
import glob
transcripts_dir = "./style_rules/transcripts_output/txt"
transcript_files_with_path = []

# 개별 자막 파일들만 찾기 (숫자로 시작하는 파일들)
pattern = f"{transcripts_dir}/*.txt"
all_files = glob.glob(pattern)

for file_path in all_files:
    filename = Path(file_path).name
    # 개별 자막 파일만 선택 (숫자로 시작하는 파일들)
    if filename.startswith(('01_', '02_', '03_', '04_', '05_', '06_', '07_', '08_', '09_', '10_', 
                           '11_', '12_', '13_', '14_', '15_', '16_', '17_', '18_', '19_', '20_',
                           '21_', '22_', '23_', '24_', '25_', '26_', '27_', '28_', '29_', '30_',
                           '31_', '32_', '33_', '34_', '35_', '36_', '37_', '38_', '39_', '40_')):
        transcript_files_with_path.append(file_path)

# 파일 정렬
transcript_files_with_path.sort()

print(f"📁 발견된 자막 파일 수: {len(transcript_files_with_path)}")
for i, file_path in enumerate(transcript_files_with_path[:5], 1):
    print(f"  {i}. {Path(file_path).name}")
if len(transcript_files_with_path) > 5:
    print(f"  ... 및 {len(transcript_files_with_path) - 5}개 더")

result = ingest_files(transcript_files_with_path)
result

def pinecone_search(query_text, k=5, season="summer", occasion=None, section=None, namespace=NAMESPACE):
    qvec = emb.embed_query(query_text)
    _filter = {}
    if season:   _filter["season"] = {"$eq": season}
    if occasion: _filter["occasion"] = {"$eq": occasion}
    if section:  _filter["section"] = {"$eq": section}

    res = index.query(
        vector=qvec, top_k=k, include_values=False, include_metadata=True,
        namespace=namespace, filter=_filter or None
    )
    matches = res.get("matches", []) or []
    for i, m in enumerate(matches, 1):
        md = m.get("metadata", {}) or {}
        print(f"\n[{i}] score={m.get('score'):.4f}  section={md.get('section')}  occasion={md.get('occasion')} source={md.get('source')}")
        if "snippet" in md:
            print("snippet:", md["snippet"])

def korean_search(query_text, k=5, namespace=NAMESPACE):
    """한국어 패션 검색 함수"""
    print(f"🔍 한국어 검색: '{query_text}'")
    qvec = emb.embed_query(query_text)
    
    res = index.query(
        vector=qvec, top_k=k, include_values=False, include_metadata=True,
        namespace=namespace
    )
    
    matches = res.get("matches", []) or []
    print(f"📊 검색 결과: {len(matches)}개")
    
    for i, m in enumerate(matches, 1):
        md = m.get("metadata", {}) or {}
        print(f"\n[{i}] 유사도: {m.get('score'):.4f}")
        print(f"    섹션: {md.get('section', 'N/A')}")
        print(f"    상황: {md.get('occasion', 'N/A')}")
        print(f"    출처: {md.get('source', 'N/A')}")
        if "snippet" in md:
            print(f"    내용: {md['snippet'][:100]}...")
    
    return matches

def build_context_from_matches(res, max_chunks=3, max_chars=800):
    chunks = []
    for m in (res.get("matches") or [])[:max_chunks]:
        md = m.get("metadata", {}) or {}
        body = md.get("snippet","")
        title = md.get("section","")
        chunks.append(f"[{title}] {body[:max_chars]}")
    return "\n\n".join(chunks)

# 예시: 한국어 패션 검색 테스트
print("\n=== 한국어 패션 검색 테스트 ===")
korean_search("여름 출근룩", k=3)

# 기존 검색도 테스트
print("\n=== 기존 검색 테스트 ===")
res = pinecone_search("30대 여자 여름 출근룩", k=5, occasion="work")
if res:
    docs_str = build_context_from_matches(res, max_chunks=3)
    print("\n--- context preview ---\n", docs_str)
else:
    print("검색 결과가 없습니다.")

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# docs_str 가 미리 정의되어 있어야 합니다. (검색 결과 컨텍스트 문자열)
if 'docs_str' not in globals() or not isinstance(docs_str, str) or len(docs_str) == 0:
    print("⚠️  docs_str가 비어있습니다. 검색 결과가 없거나 오류가 발생했습니다.")
    docs_str = "검색 결과가 없습니다."

prompt = PromptTemplate.from_template("""
[ROLE]
You are a fashion coordinator and product curator. You must follow the rules and output format below, and ground your answers in {context} whenever possible.

[INPUTS]
context: {context}
user query: {user_query}

[TASKS]
Select internally 10 product keywords from the context that best fit the user query, each as "keyword + 1–2 key attributes".
Combine the selected keywords to internally design 5 style matches (looks) that fit the query. Each look should have a coherent set of top/bottom/outer/shoes/bag–accessories and be season- and occasion-appropriate.
Pick the best 3 looks among the 5 and return ONLY a JSON array that follows the output format.

[REFERENCE — CONSULT (do not copy verbatim; adapt to the query)]
Example product keywords
- Linen jacket (light tone, short-sleeve/tweed options)
- Slacks / cotton pants (straight fit, white/beige tones)
- Skirt (long H-line / A-line mini)
- Blouse & summer knit (neckline/ruffle details)
- Dress (black or color-point, textured fabric)
- Shoes (flats, loafers, slim-toe heels/sandals)
- Belt (≈2 cm slim belt, accessory point)
- Tote bag (small structured handbag, light point color)
- Scarf (point item for black dress/blouse looks)
- Jewelry (earrings, neat pieces with some weight)

Example style matches (gist)
- Casual neat: blouse/black inner + denim/cotton pants + light linen/tweed jacket + loafers/flats + slim belt/tote
- Feminine classic: minimal-detail blouse/summer knit + long H-line or A-line mini + short-sleeve jacket/knit + slim-toe heels/sandals + earrings/scarf/tote
- Chic refined: high-neck knit/draped top + black straight slacks + collarless jacket/suit set + slim-toe shoes + mini tote/jewelry
- Modern minimal: solid top (white/black) + white slacks/cream cotton + beige jacket + loafers/mules + scarf/leather tote
- Point-focused: black dress/blouse + H-line skirt/dress + (outer optional) + heeled sandals + scarf/statement earrings

[SCORING & SELECTION]
- Fitness: alignment with season/occasion/exposure/tone in the user query.
- Practicality: substitutable items, styling difficulty, movement/photo situations.
- Harmony: color/material/silhouette balance; avoid excessive details.
- Diversity: best-3 should represent distinct concepts.
- Grounding: prefer combinations/rules mentioned in the context.

[CONSTRAINTS]
- No hallucination: prioritize keywords/looks grounded in the context. If context lacks specifics, backfill with safe, general choices; avoid exaggerated claims or brand mentions.
- Terminology normalization: colors (white/black/beige/light gray), materials (linen/cotton/tweed/knit/leather), fits (straight/wide/H/A) must be consistent.
- No repetition: avoid identical compositions/sentences across the best-3.
- Language: English only (all fields and reasons).
- Do NOT use Markdown, code fences, or comments. Output must be plain JSON only.

[OUTPUT FORMAT — JSON ONLY]
Return a JSON array using ONLY the keys below (exactly 3 objects). Keep key order; use double quotes for all keys/strings; no trailing commas.
[
  {{
    "look_name": "string",
    "catg_tops": "string",
    "top_color": "string",
    "top_material": "string",
    "catg_bottoms": "string",
    "bottom_color": "string",
    "bottom_material": "string",
    "reason": ["string", "string"]
  }},
  {{ ... second look ... }},
  {{ ... third look ... }}
]

[PROCESS NOTES]
- The 10 keywords and 5-look design steps are internal only and must NOT be printed.
- Ensure the best-3 have distinct concepts; include at least one context-based justification in each reason (e.g., "Light-tone linen jacket offers neatness and airflow for summer office/guest looks").
- If omitting outerwear is reasonable, state the condition in the reason (e.g., heat vs. indoor A/C).

[FINAL INSTRUCTION]
Follow the rules above and return ONLY the "JSON array (best-3)". Do not include Markdown, code blocks, pre/post text, or any extra content.
""")

llm = ChatOpenAI(model=OPENAI_LLM_MODEL, temperature=0)
output_parser = JsonOutputParser()
chain = prompt | llm | output_parser

question = "여름 오피스룩"
answer = chain.invoke({"user_query": question, "context": docs_str})
print(answer)


