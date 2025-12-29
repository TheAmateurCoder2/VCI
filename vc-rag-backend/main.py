from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# 🔑 CORS (THIS IS CRITICAL)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # allow Firebase
    allow_credentials=True,
    allow_methods=["*"],        # <-- THIS allows OPTIONS
    allow_headers=["*"],
)


class Query(BaseModel):
    query: str

@app.post("/rag")
async def rag(data: Query):
    return {
        "answer": f"You asked: {data.query}",
        "sources": []
    }
