import httpx, os, json
from typing import Any, Dict, List

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHAT_MODEL     = os.getenv("CHAT_MODEL", "gpt-5-mini-2025-08-07")
LLM_ENABLE     = os.getenv("LLM_ENABLE", "true").lower() == "true"

class LLM:
    _client = httpx.AsyncClient(timeout=20)

    @staticmethod
    async def call(messages: List[Dict[str,str]], *, json_mode=False, temperature=0.2) -> str:
        if not (LLM_ENABLE and OPENAI_API_KEY):
            return json.dumps({"tool":"FALLBACK","slots":{},"reason":"LLM disabled"}) if json_mode else "(LLM disabled)"
        payload: Dict[str, Any] = {"model": CHAT_MODEL, "messages": messages, "temperature": temperature}
        if json_mode:
            payload["response_format"] = {"type":"json_object"}
        r = await LLM._client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
            json=payload
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
