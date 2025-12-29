from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import JSONResponse

app = FastAPI()

# 🔥 THIS IS THE IMPORTANT PART
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],     # MUST be *
    allow_headers=["*"],     # MUST be *
)

class Query(BaseModel):
    query: str

@app.post("/rag")
async def rag(data: Query):
    return {
        "answer": f"You asked: {data.query}",
        "sources": []
    }

