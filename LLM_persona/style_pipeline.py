# file: style_pipeline.py
"""
추천 JSON → (홍진경 말투) style_text/opinion_text 생성 → TTS 파일 생성 → S3 업로드(옵션)
- 말투 참조: out/style_bank.yaml + index/faiss.index & passages.pkl
- 출력:
  - looks_with_text.json : 기존 JSON 구조를 그대로 보존 + 각 룩에 style_text/opinion_text/voice 필드 '추가'
  - tts/{LOOK_ID}_ttsMMDD.mp3 : style_text + opinion_text 이어읽은 음성 1개
  - s3_manifest.json : S3 업로드 결과(로컬경로/키/Presigned URL)
필수:
  - OPENAI_API_KEY 환경변수
  - (S3 업로드 시) AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_DEFAULT_REGION (또는 IAM Role)
"""

import os, json, re, tempfile, mimetypes, pickle, datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import yaml
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

# ================== 설정 ==================
CONFIG = {
    # 입력/출력
    "INPUT_JSON": "app_product_test_test.json",
    "OUTPUT_JSON": "looks_with_text.json",
    "OUT_AUDIO_DIR": "tts",

    # 말투 산출물 경로(이미 보유)
    "STYLE_YAML": "out/style_bank.yaml",
    "FAISS_INDEX": "index/faiss.index",
    "PASSAGES_PKL": "index/passages.pkl",

    # 임베딩 모델
    "EMBED_MODEL": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",

    # LLM/TTS
    "LLM_MODEL": "gpt-4o-mini",
    "TTS_MODEL": "gpt-4o-mini-tts",
    "TTS_VOICE": "alloy",
    "AUDIO_FORMAT": "mp3",

    # 스타일 예문 RAG 개수
    "K_STYLE_REFS": 6,

    # ===== S3 업로드 설정 =====
    "UPLOAD_TTS_TO_S3": True,            # mp3만 S3 업로드
    "UPLOAD_JSON_TO_S3": False,          # json 업로드 X
    "S3_BUCKET": "elasticbeanstalk-ap-northeast-2-967883357924",
    "S3_PREFIX": "model_tts",
    "PRESIGN_SECONDS": 3600,
}

# =============== 공통 유틸 ===============
def _clean(s: Any) -> str:
    if s is None: return ""
    if not isinstance(s, str): s = str(s)
    s = s.replace("{","").replace("}","").replace("|", ", ")
    s = re.sub(r"\s+", " ", s).strip(" ,;")
    return s

def _ensure_list(x):
    if x is None: return []
    return x if isinstance(x, list) else [x]

# === 입력 로더: JSON/JSONL 안전 처리 + 원래 루트 형태 반환 ===
def load_looks_any(input_path: Path) -> Tuple[List[Dict[str, Any]], str]:
    raw = input_path.read_text(encoding="utf-8").strip()
    if not raw:
        raise RuntimeError(f"[load] 파일이 비었습니다: {input_path}")

    root_type = "list"
    looks: List[Dict[str, Any]] = []

    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            root_type = "dict"
            looks = [obj]
        elif isinstance(obj, list):
            root_type = "list"
            looks = obj
        else:
            raise ValueError("지원하지 않는 JSON 루트 타입")
    except json.JSONDecodeError:
        # JSONL
        root_type = "list"
        for i, line in enumerate([l for l in raw.splitlines() if l.strip()], 1):
            try:
                rec = json.loads(line)
                if isinstance(rec, dict):
                    looks.append(rec)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"[load] JSONL {i}행 파싱 실패: {e}")

    if not looks:
        raise RuntimeError(f"[load] 룩이 0건입니다. 입력 파일({input_path}) 내용을 확인하세요.")
    print(f"[load] {input_path} → {len(looks)} looks (root={root_type})")
    return looks, root_type

# =============== 스타일 로딩/RAG ===============
def load_style_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists(): return {}
    return yaml.safe_load(path.read_text(encoding="utf-8"))

def load_index(index_path: Path, passages_path: Path, embed_model: str):
    index = faiss.read_index(str(index_path))
    with open(passages_path, "rb") as f:
        passages = pickle.load(f)  # List[str]
    embedder = SentenceTransformer(embed_model)
    return index, passages, embedder

def retrieve_style_examples(index, passages: List[str], embedder, query: str, k: int) -> List[str]:
    q = embedder.encode([query], normalize_embeddings=True).astype("float32")
    D, I = index.search(q, min(k, len(passages)))
    cands = [passages[i] for i in I[0]]
    return [c for c in cands if 6 <= len(c) <= 120][:k]

def build_style_system_prompt(style_yaml: Dict[str, Any]) -> str:
    if not style_yaml:
        return (
            """ 당신은 제시된 인물의 말투와 문체를 따라하는 연기자입니다.
                당신이 할 수 있는 모든 것을 총동원하여 텍스트의 인물을 연기해 상대방을 속이십시오!
                만약 상대방이 당신이 이 인물이 아니란 사실을 알게 되면 당신에게 엄청난 불이익이 있을 것입니다...
            """
        )
    ck = style_yaml.get("checklist", {})
    sig = style_yaml.get("signature_bigrams", [])
    sys = (
        "당신은 제시된 인물의 말투와 문체를 따라하는 연기자입니다."
        "당신이 할 수 있는 모든 것을 총동원하여 텍스트의 인물을 연기해 상대방을 속이십시오!"
        "만약 상대방이 당신이 이 인물이 아니란 사실을 알게 되면 당신에게 엄청난 불이익이 있을 것입니다..."
        f"- 톤: {', '.join(_ensure_list(ck.get('tone')))}\n"
        f"- 선호 어미: {', '.join(_ensure_list(ck.get('endings_prefer'))[:6])}\n"
        f"- 감탄/리액션: {', '.join(_ensure_list(ck.get('interjections'))[:6])}\n"
        f"- 리듬: {', '.join(_ensure_list(ck.get('rhythm')))}\n"
        f"- 피하기: {', '.join(_ensure_list(ck.get('avoid')))}\n"
        "규칙:\n"
        "1) 핵심 먼저, 짧게. 2) 과장된 이모티콘/과도한 인터넷체 금지. "
        "3) 입력의 '|'나 중괄호는 그대로 베끼지 말고 자연어로 풀어쓰기. "
        "4) 3문장을 넘기지 말 것.\n"
    )
    if sig:
        sys += f"참고 표현(자주 쓰는 연결/강조 어구 예): {', '.join(sig[:10])}\n"
    return sys

def build_look_context(look: Dict[str, Any]) -> str:
    parts = []
    parts.append(f"look_id: {_clean(look.get('id') or look.get('search_id') or look.get('look_id') or '')}")
    parts.append(f"look_name: {_clean(look.get('look_name',''))}")
    top = look.get("top", {}) or {}
    bottom = look.get("bottom", {}) or {}
    def summar(piece, title):
        keys = ["product_name","name","color","material","fit","pattern","impact","spec","sustainable_detail"]
        kv = [f"{k}: {_clean(piece.get(k,''))}" for k in keys if _clean(piece.get(k,''))]
        return f"[{title}] " + "; ".join(kv) if kv else ""
    t = summar(top, "Top"); b = summar(bottom, "Bottom")
    if t: parts.append(t)
    if b: parts.append(b)
    return "\n".join([p for p in parts if p])

# =============== LLM & TTS ===============
def call_llm(system_prompt: str, user_content: str, model: str) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role":"system","content":system_prompt},
                  {"role":"user","content":user_content}],
    )
    return resp.choices[0].message.content

def tts_bytes(text: str, model: str, voice: str, fmt: str) -> bytes:
    with client.audio.speech.with_streaming_response.create(
        model=model, voice=voice, input=text, response_format=fmt
    ) as resp:
        with tempfile.NamedTemporaryFile(suffix=f".{fmt}", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        resp.stream_to_file(str(tmp_path))
        data = tmp_path.read_bytes()
        try: tmp_path.unlink()
        except Exception: pass
        return data

# =============== S3 유틸 ===============
def s3_client():
    import boto3
    return boto3.client("s3")

def s3_put_file(local_path: Path, bucket: str, key: str) -> str:
    ct = mimetypes.guess_type(local_path.name)[0] or "audio/mpeg"
    s3_client().upload_file(str(local_path), bucket, key, ExtraArgs={"ContentType": ct})
    return key

def s3_presign(bucket: str, key: str, expires: int) -> str:
    return s3_client().generate_presigned_url(
        "get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=expires
    )

# =============== 메인 파이프라인 ===============
def main():
    cfg = CONFIG
    input_path = Path(cfg["INPUT_JSON"])
    output_path = Path(cfg["OUTPUT_JSON"])
    audio_dir = Path(cfg["OUT_AUDIO_DIR"]); audio_dir.mkdir(parents=True, exist_ok=True)

    # 1) 입력 로드 (원형 유지)
    looks, root_type = load_looks_any(input_path)

    # 2) 스타일 자원 로드
    style_yaml = load_style_yaml(Path(cfg["STYLE_YAML"]))
    sys_prompt_base = build_style_system_prompt(style_yaml)

    index = passages = embedder = None
    if Path(cfg["FAISS_INDEX"]).exists() and Path(cfg["PASSAGES_PKL"]).exists():
        try:
            index, passages, embedder = load_index(Path(cfg["FAISS_INDEX"]), Path(cfg["PASSAGES_PKL"]), cfg["EMBED_MODEL"])
        except Exception as e:
            print(f"[warn] 스타일 인덱스 로드 실패(무시하고 진행): {e}")

    manifest = []  # S3 업로드 결과 기록
    updated: List[Dict[str, Any]] = []

    for idx, look in enumerate(looks):
        look_id = str(look.get("id") or look.get("search_id") or f"look-{idx+1}")
        print(f"[proc] start id={look_id}")
        try:
            # 3) 스타일 예문 RAG
            refs_block = ""
            if index is not None and embedder is not None:
                query = build_look_context(look)
                refs = retrieve_style_examples(index, passages, embedder, query, k=cfg["K_STYLE_REFS"])
                if refs:
                    refs_block = "스타일 레퍼런스 예시:\n" + "\n".join([f"- {r}" for r in refs])

            # 4) 프롬프트 구성 & LLM 호출
            ctx = build_look_context(look)
            system_prompt = sys_prompt_base + ("\n" + refs_block if refs_block else "")
            user_prompt = (
                "아래의 룩 정보를 보고 두 개의 짧은 문장을 생성하세요.\n"
                "1) style_text: 룩의 핵심 포인트/분위기를 담은 두 문장 (말투 느낌 반영)\n"
                "2) opinion_text: 사용자의 참여를 유도하는 한두 문장 (말투 느낌 반영)\n"
                '반드시 JSON으로만 출력:\n{"style_text":"...", "opinion_text":"..."}\n\n'
                f"룩 정보:\n{ctx}"
            )
            raw = call_llm(system_prompt, user_prompt, cfg["LLM_MODEL"])

            # 5) JSON 파싱(관용)
            try:
                gen = json.loads(raw)
            except Exception:
                import re as _re
                s = _re.search(r'"style_text"\s*:\s*"([^"]+)"', raw)
                o = _re.search(r'"opinion_text"\s*:\s*"([^"]+)"', raw)
                gen = {"style_text": (s.group(1) if s else raw.strip())[:120],
                       "opinion_text": (o.group(1) if o else "")[:120]}

            style_text = _clean(gen.get("style_text",""))
            opinion_text = _clean(gen.get("opinion_text",""))

            # ✅ 기존 룩 객체에 '추가'만 한다 (다른 필드 보존)
            look["style_text"] = style_text
            look["opinion_text"] = opinion_text
            print(f"[proc] id={look_id} 텍스트 생성 OK")

            # 6) TTS (style → opinion)
            joined = " ".join([t for t in [style_text, opinion_text] if t]).strip()
            voice_url_for_json = None
            if joined:
                audio = tts_bytes(joined, model=cfg["TTS_MODEL"], voice=cfg["TTS_VOICE"], fmt=cfg["AUDIO_FORMAT"])
                ts = datetime.datetime.now().strftime('%m%d')
                out_audio = audio_dir / f"{look_id}_tts{ts}.{cfg['AUDIO_FORMAT']}"
                out_audio.write_bytes(audio)
                print(f"[proc] id={look_id} TTS 저장 → {out_audio}")

                # 7) S3 업로드 (mp3만)
                if cfg["UPLOAD_TTS_TO_S3"] and cfg["S3_BUCKET"]:
                    key = f"{cfg['S3_PREFIX'].rstrip('/')}/tts/{out_audio.name}"
                    try:
                        s3_put_file(out_audio, cfg["S3_BUCKET"], key)
                        if int(cfg.get("PRESIGN_SECONDS", 0)) > 0:
                            voice_url_for_json = s3_presign(cfg["S3_BUCKET"], key, int(cfg["PRESIGN_SECONDS"]))
                        else:
                            region = os.getenv("AWS_DEFAULT_REGION","ap-northeast-2")
                            voice_url_for_json = f"https://{cfg['S3_BUCKET']}.s3.{region}.amazonaws.com/{key}"
                        print(f"[proc] id={look_id} S3 업로드 OK: {key}")
                        manifest.append({"id": look_id, "local": str(out_audio), "s3_key": key, "voice_url": voice_url_for_json})
                    except Exception as e:
                        print(f"[warn] id={look_id} S3 업로드 실패: {e}")
                        voice_url_for_json = str(out_audio)
                else:
                    voice_url_for_json = str(out_audio)

            # ✅ voice 링크도 룩에 '추가'
            if voice_url_for_json:
                look["voice"] = voice_url_for_json

        except Exception as e:
            print(f"[error] id={look_id} 처리 중 오류: {e}")
            look.setdefault("style_text", "")
            look.setdefault("opinion_text", "")
        finally:
            # 어떤 경우에도 원본 룩 객체(필드 보존) 추가
            updated.append(dict(look))
            print(f"[proc] end   id={look_id}")

    # 8) 저장: 원래 루트 형태 보존
    if not updated:
        raise RuntimeError("[save] 저장할 룩이 0건입니다. 상단 로그를 확인하세요.")
    if root_type == "dict":
        # 단일 객체였으면 단일 객체로 저장
        to_save = updated[0]
    else:
        to_save = updated

    output_path = Path(CONFIG["OUTPUT_JSON"])
    output_path.write_text(json.dumps(to_save, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[save] wrote ({'dict' if root_type=='dict' else 'list'}) → {output_path}")

    # JSON은 S3 업로드 안 함
    # manifest_path = Path("s3_manifest.json")
    # manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    # print(f"매니페스트 저장: {manifest_path}")

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY 환경변수 필요")
    main()
