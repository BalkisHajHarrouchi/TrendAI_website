# backend/app.py
import os
import re
import glob
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from chromadb import PersistentClient
import httpx, feedparser

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Paths & env (make sure we open the SAME DB as ingest.py)
# ingest.py defaulted to: PERSIST_DIR = "vectorstore/db1" (relative to CWD)
# We resolve that RELATIVE TO THE PROJECT ROOT (one level above backend/)
_here = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_here, ".."))
_default_vs = os.path.abspath(os.path.join(_project_root, "vectorstore", "db1"))

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "Lajavaness/bilingual-embedding-large")
PERSIST_DIR = os.path.abspath(os.getenv("PERSIST_DIR", _default_vs))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "langchain")  # default used by Chroma wrappers

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-70b-8192")

print(f">>> TrendAI API booting")
print(f"    PERSIST_DIR = {PERSIST_DIR}")
print(f"    COLLECTION  = {COLLECTION_NAME}")
print(f"    EMBEDDINGS  = {EMBEDDING_MODEL}")
print(f"    GROQ_MODEL  = {GROQ_MODEL}")

# ─────────────────────────────────────────────────────────────────────────────
# FastAPI
app = FastAPI(title="TrendAI RAG API", version="0.1")

# CORS (tighten for prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
def _first_heading_or_sentence(text: str) -> str:
    m = re.search(r'^\s*#{2,6}\s*(.+)$', text, re.M)
    if m:
        return m.group(1).strip()
    s = re.split(r'(?<=[.!?])\s+', text.strip())[0]
    return (s[:80] + "…") if len(s) > 80 else s

def _preview(text: str, maxlen: int = 140) -> str:
    t = re.sub(r'\s+', ' ', text).strip()
    return (t[:maxlen] + "…") if len(t) > maxlen else t

def _pack_sources_as_chunks(docs, limit: int = 3, maxlen: int = 260):
    items = []
    for idx, d in enumerate((docs or [])[:limit], start=1):
        content = (getattr(d, "page_content", "") or "").strip()
        snippet = " ".join(content.split())
        if len(snippet) > maxlen:
            snippet = snippet[:maxlen] + "…"
        items.append({"title": f"Document {idx}", "kind": None, "preview": snippet})
    return items

# ─────────────────────────────────────────────────────────────────────────────
# Schemas
class AskRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=2000)

class SourceCard(BaseModel):
    title: str
    kind: Optional[str] = None
    preview: Optional[str] = None

class AskResponse(BaseModel):
    answer: str
    sources: List[SourceCard] = Field(default_factory=list)

# ─────────────────────────────────────────────────────────────────────────────
# RAG singletons
embedding = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"trust_remote_code": True},
)

# IMPORTANT: open the SAME collection name & persist dir as ingestion
vectorstore = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embedding,
    collection_name=COLLECTION_NAME,
)

# Start with similarity (easier to validate). Switch to MMR later if you want.
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)
# For MMR, once validated:
# retriever = vectorstore.as_retriever(
#     search_type="mmr",
#     search_kwargs={"k": 3, "fetch_k": 50, "lambda_mult": 0.5},
# )

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are a helpful assistant for TrendAI.\n"
        "Answer the user's question using ONLY the context provided.\n"
        "Reply in ENGLISH ONLY.\n"
        "If the context is insufficient to answer, reply exactly: \"I don't know.\"\n"
        "Do not invent or infer facts not present in the context.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n"
        "Answer:"
    ),
)

llm = ChatGroq(api_key=GROQ_API_KEY, model_name=GROQ_MODEL, temperature=0.2)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt},
    verbose=False,
)

# ─────────────────────────────────────────────────────────────────────────────
# Routes
@app.get("/health")
def health():
    # count may raise if underlying API changes; guard it
    try:
        raw_count = vectorstore._collection.count()
    except Exception:
        raw_count = None
    return {
        "ok": True,
        "persist_dir_abs": PERSIST_DIR,
        "collection": COLLECTION_NAME,
        "embedding_model": EMBEDDING_MODEL,
        "groq_model": GROQ_MODEL,
        "raw_doc_count": raw_count,
    }

@app.post("/api/ask", response_model=AskResponse)
def ask(payload: AskRequest):
    # Optional guard: if no docs retrieved, short-circuit cleanly
    docs = retriever.get_relevant_documents(payload.question)
    if not docs:
        return {"answer": "I don't know.", "sources": []}

    try:
        res = qa_chain({"query": payload.question})
    except Exception as e:
        # Surface a clean message instead of 500
        raise HTTPException(status_code=503, detail="LLM backend temporarily unavailable.") from e

    answer = (res.get("result") or "").strip()
    srcs = _pack_sources_as_chunks(res.get("source_documents", []), limit=3, maxlen=260)
    return {"answer": answer, "sources": srcs}

# ── Debug: verify you’re opening the right DB & collection ───────────────────
@app.get("/debug/info")
def debug_info():
    files = []
    if os.path.isdir(PERSIST_DIR):
        for p in glob.glob(os.path.join(PERSIST_DIR, "**"), recursive=True):
            if os.path.isfile(p):
                files.append(os.path.relpath(p, PERSIST_DIR))
    try:
        raw_count = vectorstore._collection.count()
    except Exception:
        raw_count = None
    return {
        "persist_dir_abs": PERSIST_DIR,
        "collection": COLLECTION_NAME,
        "dir_exists": os.path.isdir(PERSIST_DIR),
        "raw_doc_count": raw_count,
        "embedding_model_api": EMBEDDING_MODEL,
        "files_sample": files[:50],
    }

@app.get("/debug/chroma-collections")
def debug_chroma_collections():
    client = PersistentClient(path=PERSIST_DIR)
    cols = []
    for c in client.list_collections():
        try:
            cnt = c.count()
        except Exception:
            cnt = None
        cols.append({"name": c.name, "count": cnt})
    return {"persist_dir_abs": PERSIST_DIR, "collections": cols}

@app.get("/debug/retrieve")
def debug_retrieve(q: str = Query(..., min_length=2, max_length=2000)):
    docs = retriever.get_relevant_documents(q)
    return {"k": len(docs), "snippets": [(getattr(d, "page_content", "") or "")[:400] for d in docs]}

@app.get("/debug/sim")
def debug_sim(q: str = Query(..., min_length=2, max_length=2000)):
    docs = vectorstore.similarity_search(q, k=3)
    return {"k": len(docs), "snippets": [(getattr(d, "page_content", "") or "")[:400] for d in docs]}

def extract_first_img(html: str | None) -> str | None:
    if not html:
        return None
    m = re.search(r'<img[^>]+src="([^"]+)"', html)
    return m.group(1) if m else None

@app.get("/api/medium")
async def medium_feed(user: str):
    # Try both Medium URL styles
    feed_urls = [f"https://medium.com/feed/@{user}", f"https://{user}.medium.com/feed"]
    async with httpx.AsyncClient(timeout=12) as client:
        for url in feed_urls:
            r = await client.get(url, headers={"Accept": "application/rss+xml"})
            if r.status_code == 200 and r.text:
                parsed = feedparser.parse(r.text)
                items = []
                for e in parsed.entries[:3]:
                    # Try to find a thumbnail from html content/summary
                    html_parts = []
                    if "content" in e and isinstance(e.content, list) and e.content:
                        html_parts.append(e.content[0].value)
                    if "summary_detail" in e and getattr(e.summary_detail, "value", None):
                        html_parts.append(e.summary_detail.value)
                    if "summary" in e:
                        html_parts.append(e.summary)

                    img = None
                    for h in html_parts:
                        img = extract_first_img(h)
                        if img:
                            break

                    items.append({
                        "title": e.get("title", "Untitled"),
                        "link": e.get("link"),
                        "published": e.get("published", ""),
                        "image": img,
                    })
                return {"user": user, "items": items}
    raise HTTPException(status_code=404, detail="Medium feed not found")