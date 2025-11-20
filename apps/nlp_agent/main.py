from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Literal, Tuple
from transformers import pipeline
import re

app = FastAPI(title="NLP Agent", version="1.1")

# --------- Pipelines (load once at startup) ----------
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
ner = pipeline("token-classification", model="dslim/bert-base-NER", aggregation_strategy="simple")

# Warmup for faster first request
@app.on_event("startup")
def warmup():
    try:
        _ = summarizer("warmup text", max_length=16, min_length=5, do_sample=False)
        _ = ner("warmup")
    except Exception:
        pass

# --------- Schemas ----------
class SummarizeIn(BaseModel):
    text: str = Field(..., min_length=1)
    max_words: int = Field(200, gt=10, le=400)

class SummarizeOut(BaseModel):
    summary: str

RedactLabel = Literal["PERSON", "ORG", "LOC", "GPE", "FAC", "NORP", "DATE", "TIME", "EVENT", "LANGUAGE"]

class RedactIn(BaseModel):
    text: str = Field(..., min_length=1)
    redact_labels: List[RedactLabel] = Field(default_factory=lambda: ["PERSON", "ORG", "GPE", "LOC"])
    show_label: bool = False

class RedactOut(BaseModel):
    redacted: str
    entities: List[Tuple[str, str]]  # (entity text, label)

# --------- Helpers ----------
def _chunk_text(text: str, max_chars: int = 2000) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return [text]
    chunks = []
    i = 0
    while i < len(text):
        j = min(len(text), i + max_chars)
        k = text.rfind(". ", i, j)
        if k == -1:
            k = j
        else:
            k += 1
        chunks.append(text[i:k].strip())
        i = k
    return [c for c in chunks if c]

def _adaptive_lengths(text: str, max_words: int):
    n_words = max(1, len(text.split()))
    target_words = min(max_words, max(10, int(n_words * 0.6)))
    max_len = max(8, min(256, int(target_words * 1.3)))
    min_len = max(5, int(max_len * 0.4))
    return max_len, min_len

def _mask_spans(text: str, spans: List[Tuple[int, int, str]], show_label: bool) -> str:
    out = []
    last = 0
    for start, end, label in spans:
        start = max(0, start)
        end = min(len(text), end)
        if start < last:
            continue
        out.append(text[last:start])
        out.append(f"[{label}]" if show_label else "[REDACTED]")
        last = end
    out.append(text[last:])
    return "".join(out)

# --------- Entity label mapping (model -> standard) ----------
LABEL_MAP = {
    "PER": "PERSON",
    "ORG": "ORG",
    "LOC": "LOC",
    "GPE": "GPE",
    "FAC": "FAC",
    "NORP": "NORP",
    "DATE": "DATE",
    "TIME": "TIME",
    "EVENT": "EVENT",
    "LANG": "LANGUAGE"
}

# --------- Endpoints ----------
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/summarize", response_model=SummarizeOut)
def summarize(payload: SummarizeIn):
    try:
        chunks = _chunk_text(payload.text, max_chars=1800)
        parts = []
        max_len, min_len = _adaptive_lengths(payload.text, payload.max_words)
        for c in chunks:
            out = summarizer(
                c,
                max_length=max_len,
                min_length=min_len,
                do_sample=False,
                truncation=True,
            )
            parts.append(out[0]["summary_text"].strip())
        combined = " ".join(parts)
        return {"summary": combined}
    except Exception as e:
        raise HTTPException(500, f"summarization failed: {e}")

@app.post("/ner_redact", response_model=RedactOut)
def ner_redact(payload: RedactIn):
    try:
        ents = ner(payload.text)
        # filter and map labels
        keep = [
            e for e in ents
            if LABEL_MAP.get(e["entity_group"], None) in set(payload.redact_labels)
        ]
        keep.sort(key=lambda e: (e["start"], e["end"]))

        merged: List[Tuple[int, int, str]] = []
        for e in keep:
            s, e_, lab = int(e["start"]), int(e["end"]), LABEL_MAP.get(e["entity_group"], "UNKNOWN")
            if not merged or s > merged[-1][1]:
                merged.append((s, e_, lab))
            else:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e_), merged[-1][2])

        redacted = _mask_spans(payload.text, merged, show_label=payload.show_label)
        return {
            "redacted": redacted,
            "entities": [(payload.text[s:e], lab) for s, e, lab in merged],
        }
    except Exception as e:
        raise HTTPException(500, f"ner_redact failed: {e}")
