from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Query(BaseModel):
    query: str

@app.post("/rag")
def rag(q: Query):
    return {
        "answer": f"You asked: {q.query}"
    }

