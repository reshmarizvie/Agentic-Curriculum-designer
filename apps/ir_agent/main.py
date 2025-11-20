from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Tuple, Optional
import os, glob, time
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

app = FastAPI(title="IR Agent (NumPy Cosine, no FAISS)", version="1.1")

MODEL_NAME = os.getenv("IR_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CORPUS_DIR = os.getenv("IR_CORPUS_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "data", "corpus"))

CACHE_TTL_SEC = int(os.getenv("IR_CACHE_TTL", "120"))
_cache: Dict[Tuple[str, int], Tuple[float, List[Dict]]] = {}

# Globals
model: Optional[SentenceTransformer] = None
docs: List[Dict] = []
embeddings: Optional[np.ndarray] = None  # shape: (N, D), L2-normalized

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

def _read_corpus() -> List[Dict]:
    os.makedirs(CORPUS_DIR, exist_ok=True)
    items = []
    for path in sorted(glob.glob(os.path.join(CORPUS_DIR, "*.txt"))):
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read().strip()
        except Exception:
            text = ""
        items.append({
            "path": path,
            "title": os.path.basename(path),
            "text": text
        })
    return items

def _encode_texts(texts: List[str]) -> np.ndarray:
    vecs = model.encode(texts, batch_size=32, convert_to_numpy=True, normalize_embeddings=True)
    if vecs.dtype != np.float32:
        vecs = vecs.astype(np.float32)
    return vecs

def build_index() -> None:
    global model, docs, embeddings
    if model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer(MODEL_NAME, device=device)
    docs = _read_corpus()
    texts = [d["text"] if d["text"] else d["title"] for d in docs]
    if len(texts) == 0:
        embeddings = np.zeros((0, 384), dtype=np.float32)
        return
    embeddings = _encode_texts(texts)
    print(f"[IR] Indexed {len(docs)} documents from {CORPUS_DIR} with dim={embeddings.shape[1]}")

def _top_k_cosine(query: str, k: int) -> Tuple[List[int], np.ndarray]:
    if embeddings is None or len(docs) == 0:
        return [], np.zeros((0,), dtype=np.float32)
    q = _encode_texts([query])[0]  # normalized
    scores = embeddings @ q  # cosine == dot
    k = max(1, min(k, scores.shape[0]))
    idx = np.argpartition(-scores, kth=k-1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return idx.tolist(), scores

@app.on_event("startup")
def _startup():
    build_index()
    if embeddings is not None and len(docs) > 0:
        try:
            _ = _top_k_cosine("warmup", 3)
        except Exception:
            pass

@app.get("/health")
def health():
    return {"status": "ok", "num_docs": len(docs)}

@app.post("/reindex")
def reindex():
    build_index()
    _cache.clear()
    return {"status": "reindexed", "num_docs": len(docs)}

@app.post("/search")
def search(req: SearchRequest):
    now = time.time()
    key = (req.query.strip(), int(req.top_k))
    hit = _cache.get(key)
    if hit and (now - hit[0]) <= CACHE_TTL_SEC:
        return {"results": hit[1]}

    if embeddings is None or len(docs) == 0:
        results: List[Dict] = []
        _cache[key] = (now, results)
        return {"results": results}

    idxs, scores = _top_k_cosine(req.query, req.top_k)
    results = []
    for i in idxs:
        d = docs[i]
        text = d["text"] or ""
        snippet = (text[:280] + "…") if len(text) > 280 else text
        results.append({
            "title": d["title"],
            "path": d["path"],
            "score": float(scores[i]),
            "snippet": snippet
        })
    _cache[key] = (now, results)
    return {"results": results}
