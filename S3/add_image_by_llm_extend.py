# file: tryon_looks_controlnet_s3.py
import os, io, json, re, datetime, asyncio, time, requests
from pathlib import Path
from typing import List, Any, Dict, Optional
from PIL import Image
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.utils import load_image
import cv2
import numpy as np

load_dotenv()

# =========================
# 환경변수
# =========================
LOOKS_JSON = Path(os.getenv("LOOKS_JSON_PATH", "S3/app_product_test.json"))
AWS_S3_BUCKET  = os.getenv("AWS_S3_BUCKET_NAME")
AWS_REGION     = os.getenv("AWS_S3_REGION", "ap-northeast-2")
AWS_S3_PREFIX  = os.getenv("AWS_S3_PREFIX", "tryon")

# ControlNet 모델 (Canny)
device = "cuda" if torch.cuda.is_available() else "cpu"
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16 if device=="cuda" else torch.float32
)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet,
    torch_dtype=torch.float16 if device=="cuda" else torch.float32
)
pipe = pipe.to(device)

# ---------- JSON → 룩 정규화 ----------
KNOWN_PART_KEYS = ["top","bottom","outer","onepiece","dress","bag","shoes","acc","accessory"]

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
            v = look[k]
            if isinstance(v, list):
                for it in v: add_part(it)
            else:
                add_part(v)
    raw_look_id = look.get("look_id") or look.get("id") or "_".join(ref_ids) or "look"
    return {
        "look_id": str(raw_look_id),
        "garment_urls": garment_urls,
        "ref_ids": ref_ids,
        "meta": {"look_style": look.get("look_style")}
    }

def load_all_looks(path: Path) -> List[Dict]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, dict) and isinstance(obj.get("results"), list):
        data = obj["results"]
    elif isinstance(obj, list):
        data = obj
    elif isinstance(obj, dict):
        data = [obj]
    else:
        raise RuntimeError("룩 JSON 형식 오류")
    return [normalize_look(look) for look in data if normalize_look(look)["garment_urls"]]

# ---------- 유틸 ----------
def download_image(url: str) -> Image.Image:
    resp = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert("RGB")

def _make_canny(img: Image.Image) -> Image.Image:
    arr = np.array(img)
    edges = cv2.Canny(arr, 100, 200)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(edges)

def _sanitize_key_part(s: str) -> str:
    s = s.strip().replace(" ", "_")
    return re.sub(r"[^A-Za-z0-9._/-]+", "-", s)

# ---------- 프롬프트 ----------
def build_prompt(garment_urls: List[str], look_style: Optional[str]) -> str:
    refs = ", ".join(garment_urls)
    style_hint = f" Style: {look_style}." if look_style else ""
    return (
        "A full-body studio photo of a fashion model wearing the following garments exactly as shown: "
        f"{refs}. Natural lighting, photorealistic textures, wrinkles, and shadows."
        f"{style_hint} Minimal clean background."
    )

# ---------- 이미지 생성 ----------
async def generate_image(garment_urls: List[str], prompt: str) -> Optional[bytes]:
    loop = asyncio.get_event_loop()
    def _run():
        # 여러 garment 이미지를 받아 Canny 합성
        control_imgs = []
        for url in garment_urls:
            try:
                garment = download_image(url)
                control_imgs.append(_make_canny(garment))
            except Exception as e:
                print(f"[WARN] 참조 이미지 실패: {url} ({e})")
        if not control_imgs:
            return None
        # 첫 번째 이미지만 ControlNet 입력 (멀티 컨트롤은 추후 확장)
        result = pipe(
            prompt=prompt,
            control_image=control_imgs[0],
            num_inference_steps=25,
            guidance_scale=7.5,
        ).images[0]
        buf = io.BytesIO()
        result.save(buf, format="PNG")
        buf.seek(0)
        return buf.getvalue()
    return await loop.run_in_executor(None, _run)

# ---------- S3 업로더 ----------
class S3Uploader:
    def __init__(self, bucket: str, region: str, prefix: str = "tryon"):
        import boto3
        self.bucket = bucket
        self.region = region
        self.prefix = (prefix or "").strip("/")
        self.s3 = boto3.client("s3", region_name=region)
    def put_bytes(self, data: bytes, key: str, content_type="image/png") -> Dict[str,str]:
        self.s3.put_object(
            Bucket=self.bucket, Key=key, Body=data,
            ContentType=content_type, ACL="public-read"
        )
        return {"url": f"https://{self.bucket}.s3.{self.region}.amazonaws.com/{key}"}
    def build_key(self, filename: str, look_id: str) -> str:
        date = datetime.datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d")
        safe_look = _sanitize_key_part(look_id)
        safe_file = _sanitize_key_part(filename)
        return f"{self.prefix}/{date}/{safe_look}/{safe_file}"

# ---------- 실행 ----------
async def process_look(look, idx, uploader):
    look_id = look["look_id"]
    garment_urls = look["garment_urls"]
    prompt = build_prompt(garment_urls, look["meta"].get("look_style"))
    print(f"[INFO] look#{idx} ({look_id}) 시작")
    t0 = time.time()
    img_bytes = await generate_image(garment_urls, prompt)
    if not img_bytes:
        print(f"[ERROR] look#{idx} ({look_id}) 생성 실패")
        return None
    out_name = f"look_{idx:03d}_{_sanitize_key_part(look_id)}.png"
    s3_key = uploader.build_key(out_name, look_id)
    res = uploader.put_bytes(img_bytes, s3_key)
    print(f"[OK] {look_id} → {res['url']} (⏱️ {time.time()-t0:.2f}s)")
    return res

async def main_async():
    looks = load_all_looks(LOOKS_JSON)[:5]
    uploader = S3Uploader(AWS_S3_BUCKET, AWS_REGION, AWS_S3_PREFIX)
    tasks = [process_look(look, idx, uploader) for idx, look in enumerate(looks, start=1)]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    start = time.time()
    asyncio.run(main_async())
    print(f"총 소요시간: {time.time()-start:.2f} 초")
