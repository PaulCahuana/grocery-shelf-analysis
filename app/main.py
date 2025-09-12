
from fastapi import FastAPI

app = FastAPI(title="Grocery Shelf Analysis", version="0.1.0")

@app.get("/health")
def health():
    return {"ok": True}
