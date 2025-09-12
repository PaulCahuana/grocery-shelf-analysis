# app/schemas.py
from pydantic import BaseModel

class LoginReq(BaseModel):
    user: str
    role: str = "uploader"

class LoginResp(BaseModel):
    access_token: str
