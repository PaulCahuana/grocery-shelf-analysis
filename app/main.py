# app/main.py
from fastapi import FastAPI, Depends
from .auth import issue_token, require_jwt
from .schemas import LoginReq, LoginResp
from dotenv import load_dotenv
load_dotenv()  # this reads .env and sets os.environ


app = FastAPI(title="Grocery Shelf Analysis", version="0.1.0")

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/login", response_model=LoginResp)
def login(body: LoginReq):
    token = issue_token(body.user, body.role)
    return {"access_token": token}

# Example protected endpoint (we'll later turn this into /analyze)
@app.get("/protected")
def protected(user=Depends(require_jwt)):
    # 'user' is the decoded JWT payload
    return {"hello": user.get("sub"), "role": user.get("role")}
