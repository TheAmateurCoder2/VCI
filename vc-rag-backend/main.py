# DEPLOY TEST CORS ENABLED AGAIN

from fastapi.middleware.cors import CORSMiddleware


from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # allow Firebase hosting
    allow_credentials=True,
    allow_methods=["*"],   # allow POST, OPTIONS
    allow_headers=["*"],   # allow Content-Type
)


class Query(BaseModel):
    query: str

@app.post("/rag")
async def rag(q: Query):
    return {
        "answer": f"AI result for: {q.query}"
    }
