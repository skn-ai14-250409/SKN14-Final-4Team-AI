# app/api/dev_preview.py
from fastapi import APIRouter, Body
from fastapi.responses import HTMLResponse

dev = APIRouter(prefix="/dev", tags=["dev"])

HTML_SHELL = """<!doctype html>
<html lang="ko"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Preview</title>
<style>
  body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;background:#fafafa;margin:24px}
  .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:16px}
  .product-card{border:1px solid #eee;border-radius:12px;overflow:hidden;background:#fff}
  .product-link{display:block;text-decoration:none;color:inherit}
  .product-image{width:100%;height:220px;object-fit:cover;display:block}
  .product-info{padding:12px}
  .product-title{font-weight:600}
  .product-description{opacity:.7;font-size:14px;margin-top:4px}
  .button-box{display:flex;gap:8px;padding:12px}
</style>
</head><body>
  <div class="grid">{html}</div>
</body></html>"""

@dev.post("/preview-cards", response_class=HTMLResponse)
async def preview_cards(payload: dict = Body(...)):
    if "html" in payload:
        html = payload["html"]
    else:
        # query/profile이 온 경우 파이프라인 실행
        from app.services.orchestrator import run_pipeline
        res = await run_pipeline(payload.get("query",""), payload.get("profile") or {})
        html = res.get("html") or f"<div style='padding:20px'>{res.get('text','(no text)')}</div>"
    return HTMLResponse(HTML_SHELL.format(html=html))
