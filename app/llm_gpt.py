# app/llm_gpt.py
import os
import io
import json
import time
import base64
import asyncio
from typing import Dict, Any
from PIL import Image
from openai import AsyncOpenAI

# Default model (override via .env: OPENAI_VISION_MODEL=gpt-4o-mini or gpt-4.1-mini)
OPENAI_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini")

# Simple in-module throttle: serialize and space out upstream calls
_MIN_INTERVAL = float(os.getenv("LLM_MIN_INTERVAL", "2.0"))  # seconds
_LLM_LOCK = asyncio.Lock()
_LAST_CALL_TS = 0.0

def _shrink_image(image_bytes: bytes, max_side: int = 720, quality: int = 70) -> bytes:
    """Resize and recompress to cut tokens/costs and reduce 429s."""
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

def _build_data_url(b: bytes, mime: str = "image/jpeg") -> str:
    return f"data:{mime};base64,{base64.b64encode(b).decode('utf-8')}"

def _clamp01(x, default=0.0):
    try:
        return max(0.0, min(1.0, float(x)))
    except Exception:
        return default

PROMPT = (
    "You are a retail shelf auditor. Analyze the image and return JSON with: "
    "fullness (0..1), empty (0..1), confidence (0..1), notes (string), and "
    "rows (array of {row_index, fullness, notes?}). Consider long horizontal "
    "empty spans as empty space. Ignore price labels/hooks as product."
)

async def analyze_with_gpt(image_bytes: bytes, detail: str = "auto") -> Dict[str, Any]:
    """
    Analyze a grocery shelf image using Chat Completions (vision) and return:
      {fullness, empty, confidence, notes, rows}
    """
    api_key = _get_api_key()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    # 1) Shrink + make data URL
    image_bytes = _shrink_image(image_bytes, max_side=720, quality=70)
    data_url = _build_data_url(image_bytes, mime="image/jpeg")

    client = AsyncOpenAI(api_key=api_key)

    # 2) Messages per Chat Completions vision format
    #    Use response_format={"type": "json_object"} to guarantee JSON content.
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
                {
                    "type": "image_url",
                    "image_url": {"url": data_url, "detail": detail},  # "low" | "auto" | "high"
                },
            ],
        }
    ]

    # 3) Throttle + call
    global _LAST_CALL_TS
    async with _LLM_LOCK:
        now = time.time()
        wait = _MIN_INTERVAL - (now - _LAST_CALL_TS)
        if wait > 0:
            await asyncio.sleep(wait)
        try:
            resp = await client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                response_format={"type": "json_object"},  # guarantees JSON
                temperature=0,
                max_tokens=300,
            )
        except Exception as e:
            # surface auth/perm and rate-limit upstream errors as RuntimeError;
            # FastAPI layer maps them to nice HTTP errors already
            raise e
        _LAST_CALL_TS = time.time()

    # 4) Parse guaranteed JSON
    try:
        content = resp.choices[0].message.content
        parsed = json.loads(content)
    except Exception:
        parsed = {"fullness": 0.0, "empty": 1.0, "confidence": 0.5, "notes": "unparsed", "rows": []}

    # 5) Clamp & tidy
    fullness   = _clamp01(parsed.get("fullness"), 0.0)
    empty      = _clamp01(parsed.get("empty"), 1.0 if fullness == 0 else 1.0 - fullness)
    confidence = _clamp01(parsed.get("confidence"), 0.6)
    notes      = parsed.get("notes") or ""
    rows_in    = parsed.get("rows") or []

    rows_out = []
    for idx, r in enumerate(rows_in):
        if not isinstance(r, dict):
            continue
        rows_out.append({
            "row_index": int(r.get("row_index", idx)),
            "fullness": _clamp01(r.get("fullness"), fullness),
            "notes": r.get("notes", ""),
        })

    # ensure empty matches fullness if far off
    if abs((1.0 - fullness) - empty) > 0.15:
        empty = round(1.0 - fullness, 4)

    return {
        "fullness": round(fullness, 4),
        "empty": round(empty, 4),
        "confidence": round(confidence, 4),
        "notes": notes,
        "rows": rows_out,
    }
