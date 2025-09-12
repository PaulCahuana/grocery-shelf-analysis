# app/llm_gpt.py
import os, base64, json, httpx, asyncio, random, io, time
from typing import Dict, Any
from PIL import Image

OPENAI_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4.1-mini")

# --- simple in-module throttle (serialize and space out calls) ---
_MIN_INTERVAL = float(os.getenv("LLM_MIN_INTERVAL", "2.0"))  # seconds between provider calls
_LLM_LOCK = asyncio.Lock()
_LAST_CALL_TS = 0.0

def _shrink_image(image_bytes: bytes, max_side: int = 720, quality: int = 70) -> bytes:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)))
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=quality, optimize=True, progressive=True)
    return out.getvalue()

def _get_api_key() -> str | None:
    key = os.getenv("OPENAI_API_KEY")
    if key:
        return key
    try:
        from dotenv import load_dotenv
        load_dotenv()
        return os.getenv("OPENAI_API_KEY")
    except Exception:
        return None

PROMPT = (
    "You are a retail shelf auditor. Analyze the image and return JSON with: "
    "fullness (0..1), empty (0..1), confidence (0..1), notes (string), and "
    "rows (array of {row_index, fullness, notes?}). Consider long horizontal "
    "empty spans as empty space. Ignore price labels/hooks as product."
)

def _build_data_url(b: bytes, mime: str = "image/jpeg") -> str:
    return f"data:{mime};base64,{base64.b64encode(b).decode('utf-8')}"

async def _post_with_retries(url: str, headers: dict, payload: dict, *, max_retries: int = 8) -> httpx.Response:
    """POST with exponential backoff on 429; surface other HTTP errors."""
    backoff = 0.75
    async with httpx.AsyncClient(timeout=60) as client:
        for attempt in range(max_retries):
            r = await client.post(url, headers=headers, json=payload)
            if r.status_code == 200:
                return r
            if r.status_code == 429 and attempt < max_retries - 1:
                retry_after = r.headers.get("retry-after")
                delay = float(retry_after) if retry_after else backoff + random.uniform(0, 0.3)
                await asyncio.sleep(delay)
                backoff = min(backoff * 1.8, 10.0)
                continue
            if r.status_code in (401, 403):
                raise RuntimeError(f"OpenAI auth/permission error ({r.status_code}). Check API key & model access.")
            raise httpx.HTTPStatusError(
                f"Upstream error {r.status_code}: {r.text[:300]}",
                request=r.request,
                response=r,
            )

def _clamp01(x, default=0.0):
    try:
        return max(0.0, min(1.0, float(x)))
    except Exception:
        return default

async def analyze_with_gpt(image_bytes: bytes, detail: str = "auto") -> Dict[str, Any]:
    """Analyze a grocery shelf image using the Responses API and return structured JSON."""
    key = _get_api_key()
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    # shrink + data URL
    image_bytes = _shrink_image(image_bytes, max_side=720, quality=70)
    data_url = _build_data_url(image_bytes, mime="image/jpeg")

    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    strict_prompt = (
        PROMPT
        + "\nReturn ONLY a compact JSON object with keys: "
        + 'fullness, empty, confidence, notes, rows. Example: '
        + '{"fullness":0.62,"empty":0.38,"confidence":0.82,'
        + '"notes":"short note","rows":[{"row_index":0,"fullness":0.7},'
        + '{"row_index":1,"fullness":0.5}]}\n'
        + "Do not include any extra text."
    )

    payload = {
        "model": OPENAI_MODEL,
        "input": [{
            "role": "user",
            "content": [
                {"type": "input_text", "text": strict_prompt},
                {"type": "input_image", "image_url": data_url, "detail": detail},  # "low" | "auto" | "high"
            ],
        }],
        "temperature": 0.2,
        "max_output_tokens": 300,
    }

    # serialize & space out outbound provider calls
    global _LAST_CALL_TS
    async with _LLM_LOCK:
        now = time.time()
        wait = _MIN_INTERVAL - (now - _LAST_CALL_TS)
        if wait > 0:
            await asyncio.sleep(wait)
        r = await _post_with_retries("https://api.openai.com/v1/responses", headers, payload, max_retries=8)
        _LAST_CALL_TS = time.time()

    data = r.json()

    # Parse JSON from output_text; fallback to parts if needed
    text = (data.get("output_text") or "").strip()
    if text.startswith("```"):
        first = text.find("{")
        last = text.rfind("}")
        if first != -1 and last != -1 and last > first:
            text = text[first:last + 1]

    try:
        parsed = json.loads(text)
    except Exception:
        parsed = None
        for o in data.get("output", []):
            if o.get("type") == "message":
                for p in o.get("content", []):
                    if p.get("type") == "output_text":
                        t = p.get("text", "").strip()
                        if t.startswith("```"):
                            f = t.find("{"); l = t.rfind("}")
                            if f != -1 and l != -1 and l > f:
                                t = t[f:l + 1]
                        try:
                            parsed = json.loads(t)
                            break
                        except Exception:
                            pass
            if parsed:
                break
        if not parsed:
            parsed = {"fullness": 0.0, "empty": 1.0, "confidence": 0.5, "notes": "unparsed", "rows": []}

    fullness   = _clamp01(parsed.get("fullness"), 0.0)
    empty      = _clamp01(parsed.get("empty"), 1.0 if fullness == 0 else 1.0 - fullness)
    confidence = _clamp01(parsed.get("confidence"), 0.6)
    notes      = parsed.get("notes") or ""
    rows       = parsed.get("rows") or []

    if abs((1.0 - fullness) - empty) > 0.15:
        empty = round(1.0 - fullness, 4)

    return {
        "fullness": round(fullness, 4),
        "empty": round(empty, 4),
        "confidence": round(confidence, 4),
        "notes": notes,
        "rows": rows,
    }
