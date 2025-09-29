from fastapi import APIRouter, Body
from fastapi.responses import HTMLResponse
from app.services.orchestrator import run_pipeline

askpv = APIRouter(prefix="/dev", tags=["dev"])

HTML_SHELL = """<!doctype html><html lang="ko"><head>
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
</style></head><body>
  <div class="grid">{html}</div>
</body></html>"""

@askpv.post("/ask-preview", response_class=HTMLResponse,
            summary="Run pipeline & preview HTML",
            description="query/profile을 받아 파이프라인 실행 후 product_find라면 카드 HTML을 렌더합니다.")
async def ask_preview(payload: dict = Body(..., example={"query":"가을 원피스 추천해줘","profile":{"gender":"여성"}})):
    result = await run_pipeline(payload.get("query",""), payload.get("profile") or {})
    html = result.get("html") or f"<div style='padding:20px'>{result.get('text','(no text)')}</div>"
    return HTMLResponse(HTML_SHELL.format(html=html))
