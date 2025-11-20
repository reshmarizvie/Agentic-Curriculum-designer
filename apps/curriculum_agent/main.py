from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError
from typing import List, Dict, Any, Optional, Tuple
import os, json, re, random
import requests

# -------------------------
# Environment / Defaults
# -------------------------
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "YOUR_TOKEN_HERE")
MODEL_NAME = os.getenv("CURRICULUM_MODEL", "google/gemma-2b-it")
DEFAULT_MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "380"))
FAST_MODE = os.getenv("FAST_MODE", "OFF").upper() == "ON"
DEBUG_RAW = os.getenv("DEBUG_RAW", "1") == "1"

# ======================================================
# JSON utilities and helpers
# ======================================================
def _strip_code_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json|JSON)?\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    t = re.sub(r"^json\s*", "", t, flags=re.IGNORECASE)
    return t.strip()

def _first_json_like_span(s: str) -> Optional[str]:
    start, stack = None, []
    for i, ch in enumerate(s):
        if ch in "{[":
            if not stack: start = i
            stack.append(ch)
        elif ch in "}]":
            if stack:
                stack.pop()
                if not stack and start is not None:
                    return s[start:i+1]
    return None

def _normalize_pythonish(s: str) -> str:
    s = re.sub(r"(?<!\\)'", '"', s)
    s = s.replace("True", "true").replace("False", "false").replace("None", "null")
    s = re.sub(r",\s*([}\]])", r"\1", s)
    return s

def parse_json_from_text(text: str) -> Any:
    t = _strip_code_fences(text)
    try:
        return json.loads(t)
    except Exception:
        pass
    span = _first_json_like_span(t)
    if span:
        try:
            return json.loads(span)
        except Exception:
            span2 = _normalize_pythonish(span)
            return json.loads(span2)
    t2 = _normalize_pythonish(t)
    return json.loads(t2)

# ======================================================
# Variation helpers
# ======================================================
VARIATION_VERBS = [
    ("Identify", ["Recognize", "List", "Distinguish", "Classify"]),
    ("Describe", ["Outline", "Summarize", "Characterize", "Depict"]),
    ("Explain", ["Clarify", "Elucidate", "Interpret", "Expound"]),
    ("Apply", ["Use", "Implement", "Execute", "Operate"]),
    ("Analyze", ["Examine", "Decompose", "Investigate", "Probe"]),
    ("Evaluate", ["Assess", "Appraise", "Judge", "Critique"]),
    ("Create", ["Design", "Develop", "Construct", "Synthesize"])
]

def vary_verb(sentence: str) -> str:
    for root, variants in VARIATION_VERBS:
        if sentence.startswith(root):
            return sentence.replace(root, random.choice(variants), 1)
    return sentence

def uniq_preserve(seq: List[str]) -> List[str]:
    seen, out = set(), []
    for x in seq:
        k = re.sub(r"\s+", " ", x.strip().lower())
        if k and k not in seen:
            seen.add(k)
            out.append(x.strip())
    return out

def ensure_unique_and_varied(lines: List[str], min_n: int, max_n: int) -> List[str]:
    lines = [re.sub(r"\s+", " ", x.strip()) for x in lines if x.strip()]
    lines = uniq_preserve(lines)
    varied, used = [], set()
    for l in lines:
        v = vary_verb(l)
        stem = re.sub(r"[^a-z]+", "", v.lower())[:24]
        if stem in used:
            v += " in practical contexts."
        used.add(stem)
        varied.append(v)
    while len(varied) < min_n:
        varied.append("Explore a key concept through a hands-on example.")
    return varied[:max_n]

BLOOM = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]

def pick_bloom_span(level: str) -> Tuple[int, int]:
    lv = (level or "").lower()
    if "beginner" in lv or "intro" in lv:
        return (0, 3)
    if "advanced" in lv or "expert" in lv:
        return (2, 5)
    return (1, 5)

def bloomify(outcome: str, span: Tuple[int, int]) -> str:
    lo = re.sub(r"^\s*[-*]\s*", "", outcome.strip())
    i = random.randint(span[0], span[1])
    return f"{BLOOM[i]}: {lo[0].upper() + lo[1:]}" if lo else "Apply: Complete a practical task."

# ======================================================
# HF Inference API wrappers
# ======================================================
def llm_generate(prompt: str, max_new_tokens: int = None) -> str:
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    temperature = 0.2 if not FAST_MODE else 0.01
    top_p = 0.85 if not FAST_MODE else 0.7
    max_tokens = max_new_tokens or (DEFAULT_MAX_NEW_TOKENS if not FAST_MODE else 200)

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "return_full_text": False
        }
    }
    response = requests.post(f"https://api-inference.huggingface.co/models/{MODEL_NAME}", headers=headers, json=payload)
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail=f"HF API error: {response.text}")
    out = response.json()
    return out[0]["generated_text"]

def llm_json(prompt: str, max_new_tokens: int = None) -> Any:
    text = llm_generate(prompt, max_new_tokens)
    return parse_json_from_text(text)

# ======================================================
# FastAPI setup
# ======================================================
app = FastAPI(title="Curriculum Agent (Gemma 2B API)", version="2.3")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# ======================================================
# Models
# ======================================================
class OutcomesIn(BaseModel):
    title: str
    level: str = "Beginner"
    grounding: Optional[str] = None

class ModulePlan(BaseModel):
    module_no: int
    title: str
    description: str
    topics: List[str]
    estimated_hours: int
    outcomes: List[str]
    assessments: List[Dict[str, Any]]

class DesignIn(BaseModel):
    title: str
    level: str = "Beginner"
    grounding: Optional[str] = None

class AssessmentsIn(BaseModel):
    title: str
    level: str = "Beginner"
    module_no: int
    module_outcomes: List[str]

class DesignOut(BaseModel):
    course_title: str
    level: str
    grounding_summary: str
    outcomes: List[str]
    modules: List[ModulePlan]
    modules_count: int
    total_estimated_hours: int
    citations: Optional[List[Dict[str, Any]]] = None

# ======================================================
# Helper: Normalize module outcomes
# ======================================================
def normalize_module_outcomes(module_outcomes: List[str], module_no: int) -> List[str]:
    normalized = []
    for i, o in enumerate(module_outcomes, 1):
        clean = re.sub(r"^\s*[-*\d\.]+\s*", "", o).strip()
        normalized.append(f"M{module_no}-O{i}: {clean}")
    return normalized

# ======================================================
# Builders
# ======================================================
def build_outcomes(title: str, level: str, grounding: Optional[str]) -> List[str]:
    prompt = f'Return JSON: {{"learningOutcomes":["..."]}}. Topic: {title}, Level: {level}, Context: {grounding or ""}'
    try:
        raw = llm_json(prompt, max_new_tokens=180)
        ol = raw.get("learningOutcomes") if isinstance(raw, dict) else raw
    except Exception:
        ol = []
    if not ol:
        ol = [
            f"Identify core principles of {title.lower()}.",
            f"Describe methods used in {title.lower()}.",
            f"Apply core techniques in {title.lower()}.",
            f"Analyze challenges in {title.lower()}.",
            f"Evaluate outcomes of {title.lower()} projects.",
            f"Create a solution applying {title.lower()} concepts."
        ]
    span = pick_bloom_span(level)
    ol = [bloomify(x, span) for x in ol]
    return ensure_unique_and_varied(ol, 5, 7)

def build_assessments(title: str, level: str, module_no: int, module_outcomes: List[str]) -> List[Dict[str, Any]]:
    prompt = f"Return JSON array of 10–12 assessments for {title} Module {module_no}. Each links to outcomes."
    try:
        a = llm_json(prompt, max_new_tokens=260)
    except Exception:
        a = []

    tasks = []

    if isinstance(a, list):
        for i, it in enumerate(a, 1):
            if isinstance(it, dict):
                t = {
                    "id": f"M{module_no}-A{i}",
                    "prompt": str(it.get("prompt", f"Task {i} for {title}")),
                    "type": str(it.get("type", "short-answer")),
                    "outcome_refs": it.get("outcome_refs", [f"M{module_no}-O1"])
                }
            elif isinstance(it, str):
                t = {
                    "id": f"M{module_no}-A{i}",
                    "prompt": it,
                    "type": "short-answer",
                    "outcome_refs": [f"M{module_no}-O1"]
                }
            else:
                # fallback for unexpected types
                t = {
                    "id": f"M{module_no}-A{i}",
                    "prompt": f"Task {i} for {title}",
                    "type": "short-answer",
                    "outcome_refs": [f"M{module_no}-O1"]
                }
            tasks.append(t)

    # Ensure at least 10 assessments
    while len(tasks) < 10:
        tasks.append({
            "id": f"M{module_no}-A{len(tasks)+1}",
            "prompt": f"Demonstrate {title} concept in scenario {len(tasks)+1}.",
            "type": "short-answer",
            "outcome_refs": [f"M{module_no}-O1"]
        })

    return tasks[:12]


def build_module(title: str, level: str, module_no: int, outcomes: List[str]) -> ModulePlan:
    mo = normalize_module_outcomes(outcomes[:6], module_no)
    prompt = f'Return JSON with "title" and "description" for Module {module_no} of {title}.'
    try:
        j = llm_json(prompt, max_new_tokens=100)
        mtitle = j.get("title", f"Module {module_no}")
        mdesc = j.get("description", f"Intro to {title}.")
    except Exception:
        mtitle, mdesc = f"Module {module_no}", f"Core concepts in {title}."

    topics = ["Principles", "Worked Example", "Hands-on Lab", "Common Issues", "Reflection"]
    assessments = build_assessments(title, level, module_no, mo)

    return ModulePlan(
        module_no=module_no,
        title=mtitle,
        description=mdesc,
        topics=topics,
        estimated_hours=3,
        outcomes=mo,
        assessments=assessments
    )

def build_course(title: str, level: str, grounding: Optional[str]) -> DesignOut:
    outs = build_outcomes(title, level, grounding)
    modules = [build_module(title, level, i, outs) for i in range(1, 6)]
    total_hours = sum(m.estimated_hours for m in modules)
    return DesignOut(
        course_title=title,
        level=level,
        grounding_summary=(grounding or "")[:500],
        outcomes=outs,
        modules=modules,
        modules_count=len(modules),
        total_estimated_hours=total_hours
    )

# ======================================================
# API Routes
# ======================================================
@app.get("/health")
def health():
    return {"ok": True, "model": MODEL_NAME, "device": "cpu", "fast_mode": FAST_MODE}

@app.post("/outcomes")
def outcomes_api(data: OutcomesIn):
    return {"learningOutcomes": build_outcomes(data.title, data.level, data.grounding)}

@app.post("/assessments")
def assessments_api(data: dict):
    try:
        # -----------------------------
        # Extract and sanitize input
        # -----------------------------
        title = str(data.get("title", "")).strip()
        level = str(data.get("level", "Beginner")).strip()
        try:
            module_no = int(data.get("module_no", 1))
        except (ValueError, TypeError):
            module_no = 1

        outcomes_raw = data.get("module_outcomes", [])
        module_outcomes = []

        if isinstance(outcomes_raw, str) and outcomes_raw.strip():
            module_outcomes = [outcomes_raw.strip()]
        elif isinstance(outcomes_raw, list):
            module_outcomes = [str(o).strip() for o in outcomes_raw if isinstance(o, str) and o.strip()]

        if not title or not module_outcomes:
            # fallback: generate dummy outcome if missing
            module_outcomes = ["Understand core concepts"]
            title = title or "Sample Module"

        # Normalize outcomes
        module_outcomes_normalized = normalize_module_outcomes(module_outcomes, module_no)

        # -----------------------------
        # Generate assessments
        # -----------------------------
        try:
            assessments = build_assessments(title, level, module_no, module_outcomes_normalized)
        except Exception as e:
            print(f"[ERROR] LLM failed for assessments: {e}")
            assessments = []

        # -----------------------------
        # Ensure at least 10 assessments
        # -----------------------------
        while len(assessments) < 10:
            assessments.append({
                "id": f"M{module_no}-A{len(assessments)+1}",
                "prompt": f"Demonstrate {title} concept in scenario {len(assessments)+1}.",
                "type": "short-answer",
                "outcome_refs": module_outcomes_normalized[:1] or [f"M{module_no}-O1"]
            })

        # Limit to 12 assessments max
        return assessments[:12]

    except Exception as e:
        # -----------------------------
        # Ultimate fallback (never crash)
        # -----------------------------
        print(f"[FATAL] /assessments fallback triggered: {e}")
        return [
            {
                "id": "M1-A1",
                "prompt": "Default assessment due to server error.",
                "type": "short-answer",
                "outcome_refs": ["M1-O1"]
            }
        ]

@app.post("/modules")
def modules_api(data: OutcomesIn):
    try:
        outcomes = build_outcomes(data.title, data.level, data.grounding)
        modules = [build_module(data.title, data.level, i, outcomes) for i in range(1, 6)]
        return {
            "modules": modules,
            "modules_count": len(modules),
            "total_estimated_hours": sum(m.estimated_hours for m in modules),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Modules error: {e}")

@app.post("/design_course", response_model=DesignOut)
async def design_api(data: DesignIn, request: Request):
    try:
        return build_course(data.title, data.level, data.grounding)
    except ValidationError as ve:
        raise HTTPException(status_code=422, detail=ve.errors())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Design error: {e}")

@app.get("/")
def root():
    return {"status": "ok"}
