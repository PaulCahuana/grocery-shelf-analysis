# app/auth.py
import os, jwt
from datetime import datetime, timedelta
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

JWT_SECRET = os.getenv("JWT_SECRET", "devsecret")
ALGO = "HS256"
bearer = HTTPBearer()

def issue_token(user: str, role: str = "uploader", hours: int = 8) -> str:
    exp = datetime.utcnow() + timedelta(hours=hours)
    payload = {"sub": user, "role": role, "exp": exp}
    return jwt.encode(payload, JWT_SECRET, algorithm=ALGO)

def require_jwt(creds: HTTPAuthorizationCredentials = Depends(bearer)) -> dict:
    token = creds.credentials
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[ALGO])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")
