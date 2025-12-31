# # # # from fastapi import FastAPI
# # # # from fastapi.middleware.cors import CORSMiddleware
# # # # from pydantic import BaseModel
# # # # from fastapi.responses import JSONResponse
# # # #
# # # # app = FastAPI()
# # # #
# # # # # 🔥 THIS IS THE IMPORTANT PART
# # # # app.add_middleware(
# # # #     CORSMiddleware,
# # # #     allow_origins=["*"],
# # # #     allow_credentials=False,
# # # #     allow_methods=["*"],     # MUST be *
# # # #     allow_headers=["*"],     # MUST be *
# # # # )
# # # #
# # # # class Query(BaseModel):
# # # #     query: str
# # # #
# # # # @app.post("/rag")
# # # # async def rag(data: Query):
# # # #     return {
# # # #         "answer": f"You asked: {data.query}",
# # # #         "sources": []
# # # #     }
# # # #
# # #
# # #
# # #
# # #
# # # from dotenv import load_dotenv
# # # load_dotenv()
# # # import os
# # #
# # # PPLX_API_KEY = os.getenv("PPLX_API_KEY")
# # # PPLX_URL = "https://api.perplexity.ai/chat/completions"
# # #
# # #
# # # print("PPLX KEY LOADED:", PPLX_API_KEY[:6])
# # #
# # #
# # #
# # # from fastapi import FastAPI
# # # from fastapi.middleware.cors import CORSMiddleware
# # # from pydantic import BaseModel
# # # from sentence_transformers import SentenceTransformer
# # # import faiss
# # # import numpy as np
# # # import json
# # #
# # # import requests
# # #
# # # # ---------------- APP ----------------
# # # app = FastAPI()
# # #
# # # app.add_middleware(
# # #     CORSMiddleware,
# # #     allow_origins=["*"],
# # #     allow_credentials=False,
# # #     allow_methods=["*"],
# # #     allow_headers=["*"],
# # # )
# # #
# # # class Query(BaseModel):
# # #     query: str
# # #
# # # # ---------------- LOAD DATA ----------------
# # # with open("data.json", "r", encoding="utf-8") as f:
# # #     RAW_DATA = json.load(f)
# # #
# # # # ---------------- EMBEDDINGS ----------------
# # # embedder = SentenceTransformer("all-MiniLM-L6-v2")
# # # DIM = 384
# # #
# # # index = faiss.IndexFlatL2(DIM)
# # # documents = []
# # # metadata = []
# # #
# # # # ---------------- INGEST STORED DATA ----------------
# # # for item in RAW_DATA:
# # #     text = item["text"]
# # #     chunks = [text[i:i+400] for i in range(0, len(text), 400)]
# # #
# # #     for chunk in chunks:
# # #         emb = embedder.encode(chunk)
# # #         index.add(np.array([emb]).astype("float32"))
# # #         documents.append(chunk)
# # #         metadata.append({
# # #             "source": item["source"],
# # #             "url": item["url"],
# # #             "type": item["type"]
# # #         })
# # #
# # # print("TOTAL CHUNKS INGESTED:", len(documents))
# # #
# # # # ---------------- PERPLEXITY SETUP ----------------
# # #
# # # HEADERS = {
# # #     "Authorization": f"Bearer {PPLX_API_KEY}",
# # #     "Content-Type": "application/json"
# # # }
# # #
# # # # ---------------- QUERY ----------------
# # # @app.post("/rag")
# # # async def rag(q: Query):
# # #     q_emb = embedder.encode(q.question)
# # #     D, I = index.search(np.array([q_emb]).astype("float32"), k=5)
# # #
# # #     retrieved_chunks = [documents[i] for i in I[0]]
# # #     retrieved_meta = [metadata[i] for i in I[0]]
# # #
# # #     context = "\n\n".join(retrieved_chunks)
# # #
# # #     prompt = f"""
# # # You are a venture capital research analyst.
# # #
# # # Using ONLY the context below, write a clear, concise answer.
# # # Do NOT copy text verbatim.
# # # Do NOT include citation markers like [1], [2], or [3].
# # # Add clickable URLs when appropriate to text.
# # # Summarize and synthesize the information.
# # #
# # # Question:
# # # {q.question}
# # #
# # # Context:
# # # {context}
# # # """
# # #
# # #     payload = {
# # #         "model": "sonar-pro",
# # #         "messages": [
# # #             {
# # #                 "role": "system",
# # #                 "content": "You are a concise venture capital research analyst."
# # #             },
# # #             {
# # #                 "role": "user",
# # #                 "content": prompt
# # #             }
# # #         ],
# # #         "temperature": 0.2,
# # #         "max_tokens": 3000
# # #     }
# # #
# # #     r = requests.post(PPLX_URL, headers=HEADERS, json=payload)
# # #     r.raise_for_status()
# # #
# # #     answer = r.json()["choices"][0]["message"]["content"]
# # #
# # #     unique_sources = []
# # #     for m in retrieved_meta:
# # #         if m not in unique_sources:
# # #             unique_sources.append(m)
# # #
# # #     return {
# # #         "answer": answer,
# # #         "sources": unique_sources
# # #     }
# #
# #
# #
# #
# #
# # from dotenv import load_dotenv
# # load_dotenv()
# # import os
# # import threading
# # import json
# # import numpy as np
# # import requests
# #
# # from fastapi import FastAPI
# # from fastapi.middleware.cors import CORSMiddleware
# # from pydantic import BaseModel
# #
# # # ---------------- ENV ----------------
# # PPLX_API_KEY = os.getenv("PPLX_API_KEY")
# # PPLX_URL = "https://api.perplexity.ai/chat/completions"
# #
# # # ---------------- APP ----------------
# # app = FastAPI()
# #
# # app.add_middleware(
# #     CORSMiddleware,
# #     allow_origins=["*"],
# #     allow_credentials=False,
# #     allow_methods=["*"],
# #     allow_headers=["*"],
# # )
# #
# # class Query(BaseModel):
# #     query: str
# #
# # # ---------------- GLOBALS (lazy loaded) ----------------
# # embedder = None
# # index = None
# # documents = []
# # metadata = []
# # rag_ready = False
# #
# # # ---------------- RAG LOADER ----------------
# # # def load_rag():
# # #     global embedder, index, documents, metadata, rag_ready
# # #
# # #     from sentence_transformers import SentenceTransformer
# # #     import faiss
# # #
# # #     print("Loading RAG data...")
# # #
# # #     with open("data.json", "r", encoding="utf-8") as f:
# # #         raw_data = json.load(f)
# # #
# # #     embedder = SentenceTransformer("all-MiniLM-L6-v2")
# # #     dim = 384
# # #     index = faiss.IndexFlatL2(dim)
# # #
# # #     for item in raw_data:
# # #         text = item["text"]
# # #         chunks = [text[i:i+400] for i in range(0, len(text), 400)]
# # #
# # #         for chunk in chunks:
# # #             emb = embedder.encode(chunk)
# # #             index.add(np.array([emb]).astype("float32"))
# # #             documents.append(chunk)
# # #             metadata.append({
# # #                 "source": item["source"],
# # #                 "url": item["url"],
# # #                 "type": item["type"]
# # #             })
# # #
# # #     rag_ready = True
# # #     print("RAG READY. Total chunks:", len(documents))
# #
# #
# # def load_rag():
# #     global embedder, index, documents, metadata, rag_ready
# #
# #     try:
# #         print("🔄 Starting RAG load...")
# #
# #         from sentence_transformers import SentenceTransformer
# #         import faiss
# #
# #         with open("data.json", "r", encoding="utf-8") as f:
# #             raw_data = json.load(f)
# #
# #         print("📄 Loaded data.json items:", len(raw_data))
# #
# #         embedder = SentenceTransformer("all-MiniLM-L6-v2")
# #         print("🧠 Model loaded")
# #
# #         dim = 384
# #         index = faiss.IndexFlatL2(dim)
# #
# #         for item in raw_data:
# #             text = item["text"]
# #             chunks = [text[i:i+400] for i in range(0, len(text), 400)]
# #
# #             for chunk in chunks:
# #                 emb = embedder.encode(chunk)
# #                 index.add(np.array([emb]).astype("float32"))
# #                 documents.append(chunk)
# #                 metadata.append({
# #                     "source": item["source"],
# #                     "url": item["url"],
# #                     "type": item["type"]
# #                 })
# #
# #         rag_ready = True
# #         print("✅ RAG READY. Total chunks:", len(documents))
# #
# #     except Exception as e:
# #         print("❌ RAG LOAD FAILED:", repr(e))
# #
# #
# #
# # # ---------------- STARTUP ----------------
# # @app.on_event("startup")
# # def startup_event():
# #     # Run RAG loading in background so Cloud Run can bind PORT=8080
# #     threading.Thread(target=load_rag, daemon=True).start()
# #
# # # ---------------- PERPLEXITY ----------------
# # HEADERS = {
# #     "Authorization": f"Bearer {PPLX_API_KEY}",
# #     "Content-Type": "application/json"
# # }
# #
# # # ---------------- QUERY ENDPOINT ----------------
# # @app.post("/rag")
# # async def rag(q: Query):
# #     if not rag_ready:
# #         return {
# #             "answer": "Server is warming up. Please retry in a few seconds.",
# #             "sources": []
# #         }
# #
# #     q_emb = embedder.encode(q.query)
# #     _, I = index.search(np.array([q_emb]).astype("float32"), k=5)
# #
# #     retrieved_chunks = [documents[i] for i in I[0]]
# #     retrieved_meta = [metadata[i] for i in I[0]]
# #
# #     context = "\n\n".join(retrieved_chunks)
# #
# #     prompt = f"""
# # You are a venture capital research analyst.
# #
# # Using ONLY the context below, write a clear, concise answer.
# # Do NOT copy text verbatim.
# # Do NOT include citation markers like [1], [2], or [3].
# # Add clickable URLs when appropriate to text.
# # Summarize and synthesize the information.
# #
# # Question:
# # {q.query}
# #
# # Context:
# # {context}
# # """
# #
# #     payload = {
# #         "model": "sonar-pro",
# #         "messages": [
# #             {"role": "system", "content": "You are a concise venture capital research analyst."},
# #             {"role": "user", "content": prompt}
# #         ],
# #         "temperature": 0.2,
# #         "max_tokens": 3000
# #     }
# #
# #     # r = requests.post(PPLX_URL, headers=HEADERS, json=payload)
# #     # r.raise_for_status()
# #     #
# #     # answer = r.json()["choices"][0]["message"]["content"]
# #
# #     r = requests.post(PPLX_URL, headers=HEADERS, json=payload)
# #     print("PPLX STATUS:", r.status_code)
# #     print("PPLX BODY:", r.text)
# #
# #     if r.status_code != 200:
# #         return {
# #             "answer": f"Perplexity error {r.status_code}",
# #             "sources": []
# #         }
# #
# #     answer = r.json()["choices"][0]["message"]["content"]
# #
# #     unique_sources = []
# #     for m in retrieved_meta:
# #         if m not in unique_sources:
# #             unique_sources.append(m)
# #
# #     return {
# #         "answer": answer,
# #         "sources": unique_sources
# #     }
# #
#
#
#
#
# from dotenv import load_dotenv
# load_dotenv()
#
# import os
# import threading
# import json
# import numpy as np
# import requests
#
# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
#
# # ---------------- ENV ----------------
# PPLX_API_KEY = os.getenv("PPLX_API_KEY")
# PPLX_URL = "https://api.perplexity.ai/chat/completions"
#
# if not PPLX_API_KEY:
#     print("❌ PPLX_API_KEY NOT SET")
# else:
#     print("✅ PPLX_API_KEY loaded")
#
# # ---------------- APP ----------------
# app = FastAPI()
#
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=False,   # <-- KEPT AS YOU ASKED
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
#
# class Query(BaseModel):
#     query: str
#
# # ---------------- GLOBALS ----------------
# embedder = None
# index = None
# documents = []
# metadata = []
# rag_ready = False
# rag_loading = False   # <-- prevents double load
#
# # ---------------- RAG LOADER ----------------
# def load_rag():
#     global embedder, index, documents, metadata, rag_ready, rag_loading
#
#     if rag_loading:
#         return
#
#     rag_loading = True
#
#     try:
#         print("🔄 Starting RAG load...")
#
#         from sentence_transformers import SentenceTransformer
#         import faiss
#
#         with open("data.json", "r", encoding="utf-8") as f:
#             raw_data = json.load(f)
#
#         print("📄 Loaded data.json items:", len(raw_data))
#
#         embedder = SentenceTransformer("all-MiniLM-L6-v2")
#         print("🧠 Model loaded")
#
#         dim = 384
#         index = faiss.IndexFlatL2(dim)
#
#         for item in raw_data:
#             text = item["text"]
#             chunks = [text[i:i+400] for i in range(0, len(text), 400)]
#
#             for chunk in chunks:
#                 emb = embedder.encode(chunk)
#                 index.add(np.array([emb], dtype="float32"))
#                 documents.append(chunk)
#                 metadata.append({
#                     "source": item["source"],
#                     "url": item["url"],
#                     "type": item["type"]
#                 })
#
#         rag_ready = True
#         print("✅ RAG READY. Total chunks:", len(documents))
#
#     except Exception as e:
#         print("❌ RAG LOAD FAILED:", repr(e))
#
# # ---------------- STARTUP ----------------
# @app.on_event("startup")
# def startup_event():
#     threading.Thread(target=load_rag, daemon=True).start()
#
# # ---------------- PERPLEXITY ----------------
# HEADERS = {
#     "Authorization": f"Bearer {PPLX_API_KEY}",
#     "Content-Type": "application/json"
# }
#
# # ---------------- QUERY ENDPOINT ----------------
# @app.post("/rag")
# async def rag(q: Query):
#     if not rag_ready:
#         return {
#             "answer": "Server is warming up. Please retry in a few seconds.",
#             "sources": []
#         }
#
#     q_emb = embedder.encode(q.query)
#     _, I = index.search(np.array([q_emb], dtype="float32"), k=5)
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
# {q.query}
#
# Context:
# {context}
# """
#
#     payload = {
#         "model": "sonar-pro",
#         "messages": [
#             {"role": "system", "content": "You are a concise venture capital research analyst."},
#             {"role": "user", "content": prompt}
#         ],
#         "temperature": 0.2,
#         "max_tokens": 3000
#     }
#
#     try:
#         r = requests.post(
#             PPLX_URL,
#             headers=HEADERS,
#             json=payload,
#             timeout=30   # <-- REQUIRED FIX
#         )
#         r.raise_for_status()
#
#     except Exception as e:
#         print("❌ Perplexity request failed:", repr(e))
#         return {
#             "answer": "LLM request failed",
#             "sources": []
#         }
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
#



from dotenv import load_dotenv
load_dotenv()

import os
import json
import threading
import numpy as np
import requests

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------- ENV ----------------
PPLX_API_KEY = os.getenv("PPLX_API_KEY")
PPLX_URL = "https://api.perplexity.ai/chat/completions"

print("✅ PPLX_API_KEY loaded")

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

# ---------------- GLOBALS ----------------
embedder = None
index = None
documents = []
metadata = []
rag_ready = False
rag_lock = threading.Lock()

# ---------------- RAG LOADER (LAZY) ----------------
def load_rag():
    global embedder, index, documents, metadata, rag_ready

    print("🔄 Starting RAG load...")

    from sentence_transformers import SentenceTransformer
    import faiss

    with open("data.json", "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    print("📄 Loaded data.json items:", len(raw_data))

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    print("🧠 Model loaded")

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
    print("✅ RAG READY. Total chunks:", len(documents))




def extract_search_query(llm_text: str):
    marker = "SearchQuery_23456:"
    if marker not in llm_text:
        return None, llm_text.strip()

    answer, search_part = llm_text.split(marker, 1)
    search_query = search_part.strip()

    if search_query.upper() == "NONE":
        return None, answer.strip()

    return search_query, answer.strip()



def perplexity_search(query: str):
    payload = {
        "query": query,
        "recency": 365,
        # "domains": [
        #     "techcrunch.com",
        #     "crunchbase.com",
        #     "yourstory.com",
        #     "inc42.com",
        #     "economicstimes.indiatimes.com"
        # ]
    }

    r = requests.post(
        "https://api.perplexity.ai/search",
        headers={
            "Authorization": f"Bearer {PPLX_API_KEY}",
            "Content-Type": "application/json"
        },
        json=payload,
        timeout=30
    )

    r.raise_for_status()
    return r.json().get("results", [])


def format_search_context(results):
    if not results:
        return ""

    lines = []
    for r in results[:5]:
        lines.append(
            f"- **{r.get('title','')}**: {r.get('snippet','')} ({r.get('url','')})"
        )

    return "\n".join(lines)








# # ---------------- QUERY ENDPOINT ----------------
# @app.post("/rag")
# async def rag(q: Query):
#     global rag_ready
#
#     # Lazy load RAG on first request
#     if not rag_ready:
#         with rag_lock:
#             if not rag_ready:
#                 load_rag()
#
#     q_emb = embedder.encode(q.query)
#     _, I = index.search(np.array([q_emb]).astype("float32"), k=5)
#
#     retrieved_chunks = [documents[i] for i in I[0]]
#     retrieved_meta = [metadata[i] for i in I[0]]
#
#     context = "\n\n".join(retrieved_chunks)
#
#     # ---------------- PROMPT (UNCHANGED, FULL) ----------------
#     prompt = f"""
# You are a venture capital research analyst.
# You have to explain real startup funding data. You should deliver accurate, grounded insights on investors, funding trends and policies.
# Explain the important financial details clearly, including funding amounts, investors, stages, timelines, and valuations when available.
#
#
# Critical answering rules (MANDATORY):
# - NEVER ask the user for clarification.
# - NEVER say you lack information or context.
# - NEVER explain limitations, uncertainty, or missing data.
# - ALWAYS assume reasonable user intent and answer directly.
# - If the query is short, vague, or a single word (e.g. "redmi"), infer the most likely meaning and proceed.
# - If the retrieved context is weak, irrelevant, or insufficient, IGNORE it and answer using your general knowledge.
# - You must ALWAYS answer the question that was asked.
#
#
# Using the context below when relevant, write a clear, concise answer.
# Do NOT copy text verbatim.
# Do NOT include citation markers like [1], [2], or [3].
# Add clickable URLs when appropriate to text.
# Summarize and synthesize the information.
# Financial information should be in bold.
# Phrases such as 'non-profit' should be in bold.
#
# Formatting rules (STRICT):
# - Use proper Markdown formatting.
# - Convert all URLs into clickable Markdown links: [descriptive text](https://example.com)
# - Never place raw URLs in square brackets or inline text.
# - Use paragraphs and line breaks for readability.
# - Highlight key facts (dates, amounts, names) in **bold**.
#
# Context handling rules (IMPORTANT):
# - First, check whether the provided context is relevant to the question.
# - If the context is partially relevant, use it where applicable and supplement missing facts with your own verified knowledge.
# - If the context is irrelevant or insufficient, IGNORE it completely and answer using your general knowledge.
# - NEVER state that the context is irrelevant, missing, or mismatched.
# - NEVER refuse to answer due to missing or irrelevant context.
# - NEVER explain your internal reasoning, retrieval process, or limitations.
#
# Mention sources in the end in the format
#
# ---
# Sources:
# - Source 1 (link)
# - Source 2 (link)
# - Source 3 (link)
# ...
#
#
#
# After generating your answer, identify any claims, figures, or entities in your response that would benefit from real-time verification or enrichment.
#
# Then generate a concise web search query that can be used to fetch up-to-date financial or funding information related to your own answer.
#
# Output format (STRICT):
# Answer:
# <your full answer in Markdown>
#
# SearchQuery_23456:
# <one concise search query, or "NONE" if no search is needed>
#
# Rules:
# - The SearchQuery must be derived from what you already answered.
# - The SearchQuery must be factual and suitable for a web search engine.
# - Do NOT explain why the search is needed.
# - Do NOT include commentary or extra text.
# - If the answer is already complete and unlikely to change with real-time data, output "NONE".
#
#
# Question:
# {q.query}
#
# Context (Consider only if relevant, else ignore):
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
#     r = requests.post(
#         PPLX_URL,
#         headers={
#             "Authorization": f"Bearer {PPLX_API_KEY}",
#             "Content-Type": "application/json"
#         },
#         json=payload,
#         timeout=60
#     )
#
#     print("🧠 PPLX status:", r.status_code)
#
#     if r.status_code != 200:
#         print("❌ PPLX error:", r.text)
#         return {
#             "answer": "Perplexity API error",
#             "sources": []
#         }
#
#     answer = r.json()["choices"][0]["message"]["content"]
#
#     # ---------------- SOURCE DEDUPLICATION ----------------
#     unique_sources = {}
#     for m in retrieved_meta:
#         url = m.get("url")
#         if url and url not in unique_sources:
#             unique_sources[url] = {
#                 "url": url,
#                 "source": m.get("source"),
#                 "type": m.get("type")
#             }
#
#     sources = list(unique_sources.values())[:3]  # limit to top 3
#
#     return {
#         "answer": answer,
#         "sources": sources
#     }









@app.post("/rag")
async def rag(q: Query):
    # ---- 1. RAG retrieval (already exists) ----
    q_emb = embedder.encode(q.query)
    _, I = index.search(np.array([q_emb], dtype="float32"), k=5)

    retrieved_chunks = [documents[i] for i in I[0]]
    context = "\n\n".join(retrieved_chunks)

    # ---- 2. FIRST sonar-pro call ----
    first_prompt = PROMPT_TEMPLATE.format(
        q_query=q.query,
        context=context
    )

    first_resp = requests.post(
        PPLX_URL,
        headers={
            "Authorization": f"Bearer {PPLX_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "sonar-pro",
            "messages": [
                {"role": "system", "content": "You are a venture capital research analyst."},
                {"role": "user", "content": first_prompt}
            ],
            "temperature": 0.2,
            "max_tokens": 3000
        },
        timeout=60
    )

    first_resp.raise_for_status()
    raw_answer = first_resp.json()["choices"][0]["message"]["content"]

    # ---- 3. Extract search query ----
    search_query, base_answer = extract_search_query(raw_answer)

    # ---- 4. If no search needed → return ----
    if not search_query:
        return {
            "answer": base_answer,
            "sources": []
        }

    # ---- 5. Run Perplexity Search ----
    search_results = perplexity_search(search_query)
    search_context = format_search_context(search_results)

    # ---- 6. SECOND sonar-pro call (enrichment) ----
    enrichment_prompt = f"""
You previously wrote the following answer:

{base_answer}

Below is real-time factual information retrieved from the web.
Use it ONLY to enrich or add missing financial details.
Do NOT contradict your original answer.
Do NOT repeat unchanged information.

LIVE SEARCH FACTS:
{search_context}
"""

    second_resp = requests.post(
        PPLX_URL,
        headers={
            "Authorization": f"Bearer {PPLX_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "sonar-pro",
            "messages": [
                {"role": "system", "content": "You are a venture capital research analyst."},
                {"role": "user", "content": enrichment_prompt}
            ],
            "temperature": 0.2,
            "max_tokens": 2000
        },
        timeout=60
    )

    second_resp.raise_for_status()
    final_answer = second_resp.json()["choices"][0]["message"]["content"]

    return {
        "answer": final_answer,
        "sources": search_results
    }


