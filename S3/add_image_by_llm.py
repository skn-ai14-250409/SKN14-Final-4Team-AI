# file: tryon_looks_openai_s3.py
import os, io, json, base64, re, datetime
from pathlib import Path
from typing import List, Any, Dict, Optional
import requests
from PIL import Image
from openai import OpenAI
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

load_dotenv()

# =========================
# 환경변수 제어 (동적 설정)
# =========================
def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default

def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.lower() in ("1", "true", "yes", "y", "t")

# 출력 해상도 및 참조 이미지 전처리 파라미터
SIZE_W        = _env_int("TRYON_SIZE_W", 1024)        # 최종 폭
SIZE_H        = _env_int("TRYON_SIZE_H", 1024)        # 최종 높이
MAX_REF_SIDE  = _env_int("TRYON_REF_MAX_SIDE", 1024)  # 참조 이미지 축소 상한(긴 변)
MAX_REFS      = _env_int("TRYON_MAX_REFS", 12)        # 참조 이미지 최대 개수(과도한 payload 방지)
SIZE          = (SIZE_W, SIZE_H)

# ======== 경로/모델 설정 ========
LOOKS_JSON = Path(os.getenv("LOOKS_JSON_PATH", "beanstalkTest\\S3\\app_product_test.json"))
# OUT_DIR    = Path(os.getenv("TRYON_OUT_DIR", "out/tryon_openai")) #로컬에도 이미지 생성
MODEL_NAME = os.getenv("TRYON_MODEL_NAME", "gpt-image-1")

# ======== S3 설정 ========
AWS_S3_BUCKET  = os.getenv("AWS_S3_BUCKET_NAME")
AWS_REGION     = os.getenv("AWS_S3_REGION", "ap-northeast-2")
AWS_S3_PREFIX  = os.getenv("AWS_S3_PREFIX", "tryon")
AWS_S3_PUBLIC  = _env_bool("AWS_S3_PUBLIC_READ", True)
AWS_S3_PRESIGN = _env_int("AWS_S3_PRESIGN_EXPIRE", 0)  # 0이면 presign 안함

# ======== OpenAI 클라이언트 ========
client = OpenAI()

# ---------- 유틸 ----------
def ensure_list(x):
    if x is None: return []
    return x if isinstance(x, list) else [x]

def download_image_bytes(url: str) -> bytes:
    """원본 바이트 다운로드 (타임아웃/에러 전파)"""
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.content

def blank_base_png(size=(1024,1024)) -> bytes:
    """완전 투명 베이스 PNG"""
    img = Image.new("RGBA", size, (255,255,255,0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def full_white_mask_png(size=(1024,1024)) -> bytes:
    """전체 편집 허용 마스크(흰색)"""
    img = Image.new("L", size, 255)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def bytesio_named(data: bytes, name: str) -> io.BytesIO:
    """OpenAI API mimetype 추론을 위한 파일명 부여"""
    b = io.BytesIO(data)
    b.seek(0)
    b.name = name
    return b

def save_png_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)

def _guess_ext_from_url(url: str) -> str:
    """URL 기반 확장자 힌트 (알 수 없으면 .png)"""
    name = url.split("?")[0]
    ext = Path(name).suffix.lower()
    return ext if ext in (".png", ".jpg", ".jpeg", ".webp") else ".png"

def _ensure_reasonable_image(bytes_in: bytes, max_side: int = MAX_REF_SIDE) -> bytes:
    """
    참조 이미지를 안정적으로 만들기 위해
    - RGBA로 로드
    - 긴 변이 max_side를 넘으면 축소
    - PNG로 재인코딩 (일관성)
    """
    try:
        im = Image.open(io.BytesIO(bytes_in)).convert("RGBA")
        w, h = im.size
        longest = max(w, h)
        if longest > max_side:
            ratio = max_side / float(longest)
            im = im.resize((max(1,int(w*ratio)), max(1,int(h*ratio))), Image.LANCZOS)
        out = io.BytesIO()
        im.save(out, format="PNG")
        return out.getvalue()
    except Exception:
        # 이미지 파서 실패 시 원본 그대로 반환(최후의 보루)
        return bytes_in

# ---------- JSON → 룩 정규화 ----------
KNOWN_PART_KEYS = ["top", "bottom", "outer", "onepiece", "dress", "bag", "shoes", "acc", "accessory"]

def normalize_look(look: Dict) -> Dict:
    garment_urls: List[str] = []
    ref_ids: List[str] = []

    def add_part(part: Any):
        if not isinstance(part, dict): return
        url = part.get("image_url")
        pid = part.get("search_history_product_id")
        if url: garment_urls.append(url)
        if pid is not None: ref_ids.append(str(pid))

    for k in KNOWN_PART_KEYS:
        if k in look:
            v = look.get(k)
            if isinstance(v, list):
                for it in v: add_part(it)
            else:
                add_part(v)

    # 기타 파트에 image_url / search_history_product_id 있으면 수집
    for k, v in look.items():
        if k in KNOWN_PART_KEYS: continue
        if isinstance(v, dict) and (("image_url" in v) or ("search_history_product_id" in v)):
            add_part(v)

    raw_look_id = look.get("look_id") or look.get("id")
    if raw_look_id is None:
        raw_look_id = "_".join(ref_ids) if ref_ids else "look"

    return {
        "look_id": str(raw_look_id),
        "garment_urls": [u for u in garment_urls if u],
        "ref_ids": ref_ids,
        "meta": { "look_style": look.get("look_style") }
    }

def load_all_looks(path: Path) -> List[Dict]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise RuntimeError(f"룩 JSON 로드 실패: {e}")

    if isinstance(obj, dict) and isinstance(obj.get("results"), list):
        data = obj["results"]
    elif isinstance(obj, list):
        data = obj
    elif isinstance(obj, dict):
        data = [obj]
    else:
        raise RuntimeError("룩 JSON 형식을 인식하지 못했습니다. (list 또는 {results:[...]})")

    looks = [normalize_look(look) for look in data]
    looks = [lk for lk in looks if lk["garment_urls"]]
    if not looks:
        raise RuntimeError("룩 JSON에서 의류 이미지가 있는 항목을 찾지 못했습니다.")
    return looks

# ---------- 프롬프트 ----------
def build_prompt(garment_urls: List[str], look_style: Optional[str]) -> str:
    refs = ", ".join(garment_urls)
    style_hint = (f" 전반적 스타일은 '{look_style}' 무드에 맞추고," if look_style else "")
    return (
        "포토리얼한 모델 이미지를 생성하고 아래 참조 의류/액세서리를 자연스럽게 착용·레이어링한 모습으로 표현해줘. "
        f"{style_hint}실제 착장처럼 핏·주름·광택·그림자·겹침을 자연스럽게 만들고, 왜곡은 최소화해. "
        f"참조 URL들: {refs} "
        "배경은 심플한 스튜디오 스타일로."
    )

# ---------- OpenAI 호출 (그리드 없이: 참조 이미지만 전달) ----------
def generate_model_wearing_refs(garment_urls: List[str], prompt: str, size=(1024,1024)) -> bytes:
    if not garment_urls:
        raise RuntimeError("참조 의류 이미지 URL이 비어 있습니다.")

    # 1) 빈 베이스 + 전체 편집 마스크
    base_png = blank_base_png(size=size)
    mask_png = full_white_mask_png(size=size)
    base_buf = bytesio_named(base_png, "base.png")
    mask_buf = bytesio_named(mask_png, "mask.png")

    # 2) 참조 이미지를 '추가 이미지'로 전달 (결과 이미지에는 보이지 않음)
    refs_trimmed = garment_urls[:MAX_REFS]  # 과도한 payload 방지
    ref_bufs: List[io.BytesIO] = []
    for i, url in enumerate(refs_trimmed, start=1):
        try:
            ref_bytes = download_image_bytes(url)
            ref_bytes = _ensure_reasonable_image(ref_bytes, MAX_REF_SIDE)
            ext = _guess_ext_from_url(url)  # 확장자 힌트
            ref_bufs.append(bytesio_named(ref_bytes, f"ref{i}{ext}"))
        except Exception as e:
            # 개별 참조 실패 시 건너뛰고 진행 (로그만)
            print(f"[WARN] 참조 이미지 다운로드 실패: {url} ({e})")
            continue

    images_payload = [base_buf] + ref_bufs if ref_bufs else base_buf

    try:
        resp = client.images.edit(
            model=MODEL_NAME,
            image=images_payload,
            mask=mask_buf,
            prompt=prompt,
            size=f"{size[0]}x{size[1]}",
            n=1,
        )
    except Exception as e:
        # 에러 가시성 강화
        msg = getattr(e, "message", None) or str(e)
        raise RuntimeError(f"OpenAI images.edit 실패: {msg}")

    try:
        return base64.b64decode(resp.data[0].b64_json)
    except Exception as e:
        raise RuntimeError(f"OpenAI 응답 파싱 실패: {e}")

# ---------- S3 업로더 ----------
def _sanitize_key_part(s: str) -> str:
    s = s.strip().replace(" ", "_")
    return re.sub(r"[^A-Za-z0-9._/-]+", "-", s)

class S3Uploader:
    def __init__(self, bucket: str, region: str, public_read: bool = True, prefix: str = "tryon", presign_expire: int = 0):
        try:
            import boto3
        except Exception as e:
            raise RuntimeError(f"boto3 임포트 실패: {e}")
        self.bucket = bucket
        self.region = region
        self.public_read = public_read
        self.prefix = (prefix or "").strip("/")
        self.presign_expire = max(0, presign_expire)
        try:
            self.s3 = boto3.client("s3", region_name=region)
        except Exception as e:
            raise RuntimeError(f"S3 클라이언트 생성 실패: {e}")

    def put_bytes(self, data: bytes, key: str, content_type: str = "image/png") -> Dict[str, str]:
        extra_args = {"ContentType": content_type}
        if self.public_read:
            extra_args["ACL"] = "public-read"
        try:
            self.s3.put_object(Bucket=self.bucket, Key=key, Body=data, **extra_args)
        except Exception as e:
            raise RuntimeError(f"S3 put_object 실패: {e}")

        public_url = f"https://{self.bucket}.s3.{self.region}.amazonaws.com/{key}"
        res = {"bucket": self.bucket, "key": key, "url": public_url}

        if self.presign_expire > 0 and not self.public_read:
            try:
                url = self.s3.generate_presigned_url(
                    ClientMethod="get_object",
                    Params={"Bucket": self.bucket, "Key": key},
                    ExpiresIn=self.presign_expire,
                )
                res["presigned_url"] = url
            except Exception as e:
                print(f"[WARN] presigned URL 생성 실패: {e}")
        return res

    def build_key(self, filename: str, look_id: str) -> str:
        date = datetime.datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d")
        safe_look = _sanitize_key_part(look_id)
        safe_file = _sanitize_key_part(filename)
        # 디렉토리 구성 편집 시 여기 수정
        if self.prefix:
            return f"{self.prefix}/{date}/{safe_look}/{safe_file}"
        return f"{date}/{safe_look}/{safe_file}"

# ---------- 메인 ----------
def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("환경변수 OPENAI_API_KEY 필요")
    if not AWS_S3_BUCKET:
        raise RuntimeError("환경변수 AWS_S3_BUCKET_NAME 필요")

    # OUT_DIR.mkdir(parents=True, exist_ok=True) #로컬에도 이미지 생성

    looks = load_all_looks(LOOKS_JSON)
    print(f"[INFO] 총 {len(looks)}개의 룩을 생성합니다. (SIZE={SIZE[0]}x{SIZE[1]}, MAX_REFS={MAX_REFS}, MAX_REF_SIDE={MAX_REF_SIDE})")

    uploader = S3Uploader(
        bucket=AWS_S3_BUCKET,
        region=AWS_REGION,
        public_read=AWS_S3_PUBLIC,
        prefix=AWS_S3_PREFIX,
        presign_expire=AWS_S3_PRESIGN
    )

    for idx, look in enumerate(looks, start=1):
        garment_urls = look["garment_urls"]
        look_style   = look["meta"].get("look_style")
        look_id      = look["look_id"]

        prompt = build_prompt(garment_urls, look_style)

        try:
            png_bytes = generate_model_wearing_refs(garment_urls, prompt, size=SIZE)
        except Exception as e:
            print(f"[ERROR] look#{idx} ({look_id}) 이미지 생성 실패: {e}")
            continue

        out_name = f"look_{idx:03d}_{_sanitize_key_part(look_id)}.png"
        # out_path = OUT_DIR / out_name # 로컬에도 이미지 생성 
        # try:
        #     save_png_bytes(out_path, png_bytes)
        # except Exception as e:
        #     print(f"[WARN] 로컬 저장 실패: {out_path} ({e})")

        try:
            s3_key = uploader.build_key(out_name, look_id)
            res = uploader.put_bytes(png_bytes, s3_key, content_type="image/png")
            if "presigned_url" in res:
                print(f"[OK] look#{idx} ({look_id}) → S3: s3://{res['bucket']}/{res['key']}\n      URL: {res.get('presigned_url')}")
            else:
                print(f"[OK] look#{idx} ({look_id}) → S3: s3://{res['bucket']}/{res['key']}\n      URL: {res.get('url')}")
        except Exception as e:
            print(f"[WARN] look#{idx} ({look_id}) S3 업로드 실패(로컬 저장 완료): {e}")

if __name__ == "__main__":
    main()
