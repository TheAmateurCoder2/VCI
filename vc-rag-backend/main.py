# # from fastapi import FastAPI
# # from fastapi.middleware.cors import CORSMiddleware
# # from pydantic import BaseModel
# # from fastapi.responses import JSONResponse
# #
# # app = FastAPI()
# #
# # # 🔥 THIS IS THE IMPORTANT PART
# # app.add_middleware(
# #     CORSMiddleware,
# #     allow_origins=["*"],
# #     allow_credentials=False,
# #     allow_methods=["*"],     # MUST be *
# #     allow_headers=["*"],     # MUST be *
# # )
# #
# # class Query(BaseModel):
# #     query: str
# #
# # @app.post("/rag")
# # async def rag(data: Query):
# #     return {
# #         "answer": f"You asked: {data.query}",
# #         "sources": []
# #     }
# #
#
#
#
#
# from dotenv import load_dotenv
# load_dotenv()
# import os
#
# PPLX_API_KEY = os.getenv("PPLX_API_KEY")
# PPLX_URL = "https://api.perplexity.ai/chat/completions"
#
#
# print("PPLX KEY LOADED:", PPLX_API_KEY[:6])
#
#
#
# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from sentence_transformers import SentenceTransformer
# import faiss
# import numpy as np
# import json
#
# import requests
#
# # ---------------- APP ----------------
# app = FastAPI()
#
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=False,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
#
# class Query(BaseModel):
#     query: str
#
# # ---------------- LOAD DATA ----------------
# with open("data.json", "r", encoding="utf-8") as f:
#     RAW_DATA = json.load(f)
#
# # ---------------- EMBEDDINGS ----------------
# embedder = SentenceTransformer("all-MiniLM-L6-v2")
# DIM = 384
#
# index = faiss.IndexFlatL2(DIM)
# documents = []
# metadata = []
#
# # ---------------- INGEST STORED DATA ----------------
# for item in RAW_DATA:
#     text = item["text"]
#     chunks = [text[i:i+400] for i in range(0, len(text), 400)]
#
#     for chunk in chunks:
#         emb = embedder.encode(chunk)
#         index.add(np.array([emb]).astype("float32"))
#         documents.append(chunk)
#         metadata.append({
#             "source": item["source"],
#             "url": item["url"],
#             "type": item["type"]
#         })
#
# print("TOTAL CHUNKS INGESTED:", len(documents))
#
# # ---------------- PERPLEXITY SETUP ----------------
#
# HEADERS = {
#     "Authorization": f"Bearer {PPLX_API_KEY}",
#     "Content-Type": "application/json"
# }
#
# # ---------------- QUERY ----------------
# @app.post("/rag")
# async def rag(q: Query):
#     q_emb = embedder.encode(q.question)
#     D, I = index.search(np.array([q_emb]).astype("float32"), k=5)
#
#     retrieved_chunks = [documents[i] for i in I[0]]
#     retrieved_meta = [metadata[i] for i in I[0]]
#
#     context = "\n\n".join(retrieved_chunks)
#
#     prompt = f"""
# You are a venture capital research analyst.
#
# Using ONLY the context below, write a clear, concise answer.
# Do NOT copy text verbatim.
# Do NOT include citation markers like [1], [2], or [3].
# Add clickable URLs when appropriate to text.
# Summarize and synthesize the information.
#
# Question:
# {q.question}
#
# Context:
# {context}
# """
#
#     payload = {
#         "model": "sonar-pro",
#         "messages": [
#             {
#                 "role": "system",
#                 "content": "You are a concise venture capital research analyst."
#             },
#             {
#                 "role": "user",
#                 "content": prompt
#             }
#         ],
#         "temperature": 0.2,
#         "max_tokens": 3000
#     }
#
#     r = requests.post(PPLX_URL, headers=HEADERS, json=payload)
#     r.raise_for_status()
#
#     answer = r.json()["choices"][0]["message"]["content"]
#
#     unique_sources = []
#     for m in retrieved_meta:
#         if m not in unique_sources:
#             unique_sources.append(m)
#
#     return {
#         "answer": answer,
#         "sources": unique_sources
#     }





from dotenv import load_dotenv
load_dotenv()
import os
import threading
import json
import numpy as np
import requests

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------- ENV ----------------
PPLX_API_KEY = os.getenv("PPLX_API_KEY")
PPLX_URL = "https://api.perplexity.ai/chat/completions"

# ---------------- APP ----------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    query: str

# ---------------- GLOBALS (lazy loaded) ----------------
embedder = None
index = None
documents = []
metadata = []
rag_ready = False

# ---------------- RAG LOADER ----------------
def load_rag():
    global embedder, index, documents, metadata, rag_ready

    from sentence_transformers import SentenceTransformer
    import faiss

    print("Loading RAG data...")

    with open("data.json", "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    dim = 384
    index = faiss.IndexFlatL2(dim)

    for item in raw_data:
        text = item["text"]
        chunks = [text[i:i+400] for i in range(0, len(text), 400)]

        for chunk in chunks:
            emb = embedder.encode(chunk)
            index.add(np.array([emb]).astype("float32"))
            documents.append(chunk)
            metadata.append({
                "source": item["source"],
                "url": item["url"],
                "type": item["type"]
            })

    rag_ready = True
    print("RAG READY. Total chunks:", len(documents))

# ---------------- STARTUP ----------------
@app.on_event("startup")
def startup_event():
    # Run RAG loading in background so Cloud Run can bind PORT=8080
    threading.Thread(target=load_rag, daemon=True).start()

# ---------------- PERPLEXITY ----------------
HEADERS = {
    "Authorization": f"Bearer {PPLX_API_KEY}",
    "Content-Type": "application/json"
}

# ---------------- QUERY ENDPOINT ----------------
@app.post("/rag")
async def rag(q: Query):
    if not rag_ready:
        return {
            "answer": "Server is warming up. Please retry in a few seconds.",
            "sources": []
        }

    q_emb = embedder.encode(q.query)
    _, I = index.search(np.array([q_emb]).astype("float32"), k=5)

    retrieved_chunks = [documents[i] for i in I[0]]
    retrieved_meta = [metadata[i] for i in I[0]]

    context = "\n\n".join(retrieved_chunks)

    prompt = f"""
You are a venture capital research analyst.

Using ONLY the context below, write a clear, concise answer.
Do NOT copy text verbatim.
Do NOT include citation markers like [1], [2], or [3].
Add clickable URLs when appropriate to text.
Summarize and synthesize the information.

Question:
{q.query}

Context:
{context}
"""

    payload = {
        "model": "sonar-pro",
        "messages": [
            {"role": "system", "content": "You are a concise venture capital research analyst."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 3000
    }

    # r = requests.post(PPLX_URL, headers=HEADERS, json=payload)
    # r.raise_for_status()
    #
    # answer = r.json()["choices"][0]["message"]["content"]

    r = requests.post(PPLX_URL, headers=HEADERS, json=payload)
    print("PPLX STATUS:", r.status_code)
    print("PPLX BODY:", r.text)

    if r.status_code != 200:
        return {
            "answer": f"Perplexity error {r.status_code}",
            "sources": []
        }

    answer = r.json()["choices"][0]["message"]["content"]

    unique_sources = []
    for m in retrieved_meta:
        if m not in unique_sources:
            unique_sources.append(m)

    return {
        "answer": answer,
        "sources": unique_sources
    }

