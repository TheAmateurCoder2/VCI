from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RagRequest(BaseModel):
    query: str

@app.post("/rag")
async def rag(req: RagRequest):
    return {
        "answer": f"You asked: {req.query}",
        "sources": []
    }

@app.get("/")
def root():
    return {"status": "ok"}
