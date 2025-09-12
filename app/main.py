# app/main.py
from dotenv import load_dotenv
load_dotenv()  # make env vars available to imports below

from fastapi import FastAPI, Depends, File, UploadFile, HTTPException
from datetime import datetime
from httpx import HTTPStatusError
import numpy as np
import cv2
import time
import hashlib

from .auth import issue_token, require_jwt
from .schemas import LoginReq, LoginResp
from .llm_gpt import analyze_with_gpt

app = FastAPI(title="Grocery Shelf Analysis", version="0.1.0")

# -------------------------
# Simple in-memory cache (15 min TTL)
# -------------------------
_CACHE: dict[str, tuple[float, dict]] = {}
CACHE_TTL_SEC = 15 * 60

def _cache_get(key: str):
    hit = _CACHE.get(key)
    if not hit:
        return None
    exp, val = hit
    if exp < time.time():
        _CACHE.pop(key, None)
        return None
    return val

def _cache_put(key: str, val: dict, ttl: int = CACHE_TTL_SEC):
    _CACHE[key] = (time.time() + ttl, val)

# -------------------------
# Token-bucket rate limiter (per user)
# -------------------------
_BUCKETS: dict[str, tuple[float, float]] = {}  # user_id -> (tokens, last_ts)

def _rate_limit_token_bucket(user_id: str, capacity=3, refill_per_sec=0.5):
    """
    Allows short bursts up to 'capacity', refilling at 'refill_per_sec' tokens/sec.
    Raise 429 if bucket empty.
    """
    now = time.time()
    tokens, last = _BUCKETS.get(user_id, (capacity, now))
    tokens = min(capacity, tokens + (now - last) * refill_per_sec)
    if tokens < 1:
        raise HTTPException(status_code=429, detail="Local rate limit: retry shortly.")
    _BUCKETS[user_id] = (tokens - 1, now)

# -------------------------
# Routes
# -------------------------
@app.get("/health")
def health():
    return {"ok": True}

@app.post("/login", response_model=LoginResp)
def login(body: LoginReq):
    token = issue_token(body.user, body.role)
    return {"access_token": token}

@app.get("/protected")
def protected(user=Depends(require_jwt)):
    return {"hello": user.get("sub"), "role": user.get("role")}

@app.post("/analyze", tags=["analyze"])
async def analyze_endpoint(
    file: UploadFile = File(..., description="Shelf image (jpg/png)"),
    detail: str = "auto",                 # "low" | "auto" | "high"
    restock_threshold: float = 0.70,      # alert if fullness < threshold
    use_cache: bool = True,               # allow bypass with ?use_cache=false
    user=Depends(require_jwt),
):
    # Per-user rate limit
    _rate_limit_token_bucket(user.get("sub", "anon"))

    # 1) Read bytes first
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file")

    # 2) Quick sanity check the image decodes
    arr = np.frombuffer(data, np.uint8)
    if cv2.imdecode(arr, cv2.IMREAD_COLOR) is None:
        raise HTTPException(400, "Invalid image")

    # 2.5) Cache check (by image content)
    img_key = hashlib.sha256(data).hexdigest()
    if use_cache:
        cached = _cache_get(img_key)
        if cached:
            return cached

    # 3) Call the LLM with proper error handling
    try:
        gpt = await analyze_with_gpt(data, detail=detail)
    except RuntimeError as e:
        # config/auth problems (missing key, 401/403 from provider wrapped as RuntimeError)
        raise HTTPException(status_code=500, detail=str(e))
    except HTTPStatusError as e:
        if e.response is not None:
            if e.response.status_code == 429:
                ra = e.response.headers.get("retry-after", "2")
                raise HTTPException(
                    status_code=429,
                    detail={"message": "Upstream rate limit. Retry later.", "retry_after": ra},
                )
            # Echo a snippet of the upstream body for debugging
            raise HTTPException(
                status_code=502,
                detail=f"Upstream {e.response.status_code}: {e.response.text[:300]}",
            )
        raise HTTPException(status_code=502, detail="Upstream error")

    # 4) Compose response
    status = "ok"
    alert = None
    if gpt.get("fullness", 0.0) < restock_threshold:
        status = "low_stock"
        alert = {
            "type": "restock",
            "threshold": restock_threshold,
            "message": f"Fullness {gpt['fullness']:.2f} below {restock_threshold:.2f}",
        }

    resp = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "model": "gpt-vision",
        "params": {"detail": detail, "restock_threshold": restock_threshold},
        "result": {
            "fullness": gpt["fullness"],
            "empty": gpt["empty"],
            "confidence": gpt["confidence"],
            "notes": gpt["notes"],
            "rows": gpt["rows"],
        },
        "status": status,
        "alert": alert,
    }

    # 5) Save to cache and return
    if use_cache:
        _cache_put(img_key, resp)

    return resp
