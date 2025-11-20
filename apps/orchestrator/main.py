from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, Any, Optional, List
import os, time, asyncio
import httpx
from jose import jwt, JWTError
from datetime import datetime, timedelta

app = FastAPI(title="Orchestrator Agent", version="1.5")

# ---------- CORS ----------
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ORCH_CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Config ----------
JWT_SECRET = os.getenv("JWT_SECRET", "change_me")
JWT_ALG = os.getenv("JWT_ALG", "HS256")

CURR_BASE = os.getenv("ORCH_CURRICULUM_BASE", "http://localhost:8001")
IR_BASE   = os.getenv("ORCH_IR_BASE",         "http://localhost:8002")
NLP_BASE  = os.getenv("ORCH_NLP_BASE",        "http://localhost:8003")

STUB_MODE = os.getenv("STUB_MODE", "0") == "1"

USE_HTTP2        = os.getenv("ORCH_HTTP2", "0") == "1"
TIMEOUT_CONNECT  = float(os.getenv("ORCH_TIMEOUT_CONNECT", "10.0"))
TIMEOUT_READ     = float(os.getenv("ORCH_TIMEOUT_READ",    "600.0"))
TIMEOUT_WRITE    = float(os.getenv("ORCH_TIMEOUT_WRITE",   "60.0"))
TIMEOUT_POOL     = float(os.getenv("ORCH_TIMEOUT_POOL",    "10.0"))

RETRY_ATTEMPTS   = int(os.getenv("ORCH_RETRY_ATTEMPTS", "2"))
BACKOFF_BASE_S   = float(os.getenv("ORCH_BACKOFF_BASE_S", "0.6"))

# ---------- Auth ----------
bearer_scheme = HTTPBearer(auto_error=True)

class AuthUser(BaseModel):
    sub: str

def verify_jwt(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)) -> AuthUser:
    token = credentials.credentials
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
        sub = payload.get("sub")
        if not sub:
            raise HTTPException(401, "Invalid token: no sub")
        return AuthUser(sub=sub)
    except JWTError:
        raise HTTPException(401, "Invalid token")

# ---------- Rate limit ----------
BUCKETS: Dict[str, Dict[str, Any]] = {}
RATE = 60
REFILL = 60

def rate_limit(user: AuthUser):
    now = time.time()
    b = BUCKETS.get(user.sub)
    if not b:
        BUCKETS[user.sub] = {"tokens": RATE, "ts": now}
        return
    elapsed = now - b["ts"]
    b["tokens"] = min(RATE, b["tokens"] + (elapsed / 60.0) * REFILL)
    b["ts"] = now
    if b["tokens"] < 1:
        raise HTTPException(429, "Rate limit exceeded")
    b["tokens"] -= 1

# ---------- Models ----------
class DesignReq(BaseModel):
    course_title: str = Field(..., description="Course title")
    level: str = Field("Beginner", description="Audience level")
    top_k: int = Field(5, description="IR top-k (fallback if ir_top_k missing)")
    notes: Optional[str] = Field(None, description="Optional notes to guide grounding")
    ir_top_k: Optional[int] = None
    include_trace: Optional[bool] = True
    include_citations: Optional[bool] = True
    max_new_tokens: Optional[int] = None
    model_config = ConfigDict(extra="ignore")

class TokenReq(BaseModel):
    username: Optional[str] = None
    sub: Optional[str] = None

# ---------- Token endpoints ----------
@app.post("/token")
def generate_token(body: Optional[TokenReq] = None, username: Optional[str] = None):
    name = (body.username if body and body.username else
            body.sub if body and body.sub else
            username if username else
            "demo-user")
    expire = datetime.utcnow() + timedelta(days=30)
    claims = {"sub": name, "exp": expire}
    token = jwt.encode(claims, JWT_SECRET, algorithm=JWT_ALG)
    return {"access_token": token, "token_type": "bearer", "expires_at": expire.isoformat()}

@app.get("/whoami")
def whoami(user: AuthUser = Depends(verify_jwt)):
    return {"sub": user.sub}

# ---------- Health ----------
@app.get("/")
def root():
    return {"status": "ok", "curriculum": CURR_BASE, "ir": IR_BASE, "nlp": NLP_BASE}

@app.get("/health")
async def health():
    client = _client()

    async def ping_json(method: str, url: str, *, payload: dict | None = None):
        try:
            if method == "GET":
                r = await client.get(
                    url,
                    timeout=httpx.Timeout(connect=5.0, read=5.0, write=5.0, pool=5.0),
                )
            else:
                r = await client.post(
                    url,
                    json=(payload or {}),
                    timeout=httpx.Timeout(connect=5.0, read=5.0, write=5.0, pool=5.0),
                )
            ok = (200 <= r.status_code < 300)
            try:
                body = r.json()
            except Exception:
                body = (r.text or "")[:200]
            return {"ok": ok, "status": r.status_code, "url": url, "body": body}
        except Exception as e:
            return {"ok": False, "url": url, "error": str(e)}

    results = {
        "ir":         await ping_json("POST", f"{IR_BASE}/search",     payload={"query": "ping", "top_k": 1}),
        "nlp":        await ping_json("POST", f"{NLP_BASE}/summarize", payload={"text": "ping", "max_words": 12}),
        "curriculum": await ping_json("GET",  f"{CURR_BASE}/"),
    }
    return {"status": "ok", "services": results}

# ---------- HTTP client (reused) ----------
TIMEOUT = httpx.Timeout(
    connect=TIMEOUT_CONNECT,
    read=TIMEOUT_READ,
    write=TIMEOUT_WRITE,
    pool=TIMEOUT_POOL,
)
CLIENT: Optional[httpx.AsyncClient] = None

@app.on_event("startup")
async def _startup_client():
    global CLIENT
    CLIENT = httpx.AsyncClient(timeout=TIMEOUT, http2=USE_HTTP2)

@app.on_event("shutdown")
async def _shutdown_client():
    global CLIENT
    if CLIENT is not None:
        try:
            await CLIENT.aclose()
        finally:
            CLIENT = None

def _client() -> httpx.AsyncClient:
    global CLIENT
    if CLIENT is None:
        CLIENT = httpx.AsyncClient(timeout=TIMEOUT, http2=USE_HTTP2)
    return CLIENT

# ---------- Robust POST with retries ----------
async def _post_json(
    url: str,
    payload: Dict[str, Any],
    *,
    attempts: int = RETRY_ATTEMPTS,
    per_request_timeout: Optional[httpx.Timeout] = None,
) -> Dict[str, Any]:
    client = _client()
    last_exc: Optional[Exception] = None
    tries = 1 + max(0, attempts)
    for i in range(tries):
        try:
            r = await client.post(url, json=payload, timeout=per_request_timeout)
            r.raise_for_status()
            return r.json()
        except httpx.ReadTimeout as e:
            last_exc = e
            if i == tries - 1:
                raise HTTPException(504, f"Upstream timeout calling {url}") from e
        except httpx.HTTPStatusError as e:
            last_exc = e
            code = e.response.status_code
            if 500 <= code < 600:
                if i == tries - 1:
                    raise HTTPException(502, f"Upstream {url} error: HTTP {code}") from e
            else:
                detail = e.response.text or f"HTTP {code}"
                raise HTTPException(502, f"Upstream {url} error: {detail}") from e
        except ValueError as e:
            last_exc = e
            if i == tries - 1:
                raise HTTPException(502, f"Invalid JSON from {url}") from e
        await asyncio.sleep(BACKOFF_BASE_S * (2 ** i))
    raise HTTPException(502, f"Upstream {url} failed") from last_exc

# ---------- Helpers ----------
def _derive_top_k(req: DesignReq) -> int:
    if isinstance(req.ir_top_k, int) and req.ir_top_k > 0:
        return req.ir_top_k
    return max(1, int(req.top_k or 5))

def _extract_totals(modules_obj: Any) -> Dict[str, Optional[int]]:
    try:
        if isinstance(modules_obj, dict):
            keys = [k for k in modules_obj.keys() if isinstance(k, str) and k.lower().startswith("module")]
            modules_count = len(keys)
            total_hours = 0
            for k in keys:
                v = modules_obj.get(k) or {}
                hours = v.get("Estimated Hours")
                if isinstance(hours, (int, float)):
                    total_hours += int(hours)
            return {"modules_count": modules_count, "total_estimated_hours": total_hours}
        elif isinstance(modules_obj, list):
            modules_count = len(modules_obj)
            total_hours = sum(int(m.get("estimated_hours", 0)) for m in modules_obj if isinstance(m, dict))
            return {"modules_count": modules_count, "total_estimated_hours": total_hours}
    except Exception:
        pass
    return {"modules_count": None, "total_estimated_hours": None}

# ---------- Main orchestration ----------
@app.post("/design")
async def design(req: DesignReq, user: AuthUser = Depends(verify_jwt)):
    rate_limit(user)

    title = (req.course_title or "").strip()
    if not title:
        raise HTTPException(400, "course_title is required")
    if len(title) > 200:
        raise HTTPException(400, "course_title too long")

    if STUB_MODE:
        stub = {
            "course_title": title,
            "level": req.level,
            "grounding_summary": (req.notes or "stub redacted"),
            "citations": [{"text": "stub"}],
            "outcomes": ["Outcome 1", "Outcome 2"],
            "modules": {"Module 1": {"Title": "Stub", "Description": "Stub", "Topics": [], "Estimated Hours": 1}},
            "assessments": [
                "Explain a key concept with a concrete example relevant to the course domain.",
                "Compare two methods, highlighting trade-offs and appropriate use cases in practice.",
                "Analyze a brief scenario and identify the most relevant factors affecting outcomes.",
                "Design a simple workflow to evaluate a chosen method and describe expected results.",
                "Evaluate limitations of an approach and suggest strategies to mitigate common pitfalls."
            ],
        }
        stub.update(_extract_totals(stub["modules"]))
        return stub

    trace: Dict[str, Any] = {"user": user.sub, "title": title, "steps": []}

    # 1) IR
    ir_topk = _derive_top_k(req)
    ir = await _post_json(
        f"{IR_BASE}/search",
        {"query": title, "top_k": ir_topk},
    )
    if req.include_trace:
        trace["steps"].append({"ir_results": ir})

    # 2) NLP summarize
    joined_snippets = "\n\n".join([(x.get("snippet") or x.get("text") or "") for x in ir.get("results", [])])[:4000]
    context = (joined_snippets + ("\n\nNOTES:\n" + req.notes if req.notes else "")).strip() or title
    nlp_sum = await _post_json(
        f"{NLP_BASE}/summarize",
        {"text": context, "max_words": 200},
    )
    summary = nlp_sum.get("summary", "") or context
    if req.include_trace:
        trace["steps"].append({"summary": summary})

    # 3) NLP redact
    nlp_red = await _post_json(
        f"{NLP_BASE}/ner_redact",
        {"text": summary},
    )
    redacted = nlp_red.get("redacted", summary) or summary
    if req.include_trace:
        trace["steps"].append({"redacted_summary": redacted})

    # 4) Curriculum
    common = {"title": title, "level": req.level, "grounding": redacted}

    cur_out = await _post_json(f"{CURR_BASE}/outcomes", common)
    outcomes: List[str] = cur_out.get("outcomes") or []
    if req.include_trace:
        trace["steps"].append({"outcomes": outcomes})

    long_timeout = httpx.Timeout(
        connect=TIMEOUT_CONNECT, read=TIMEOUT_READ, write=TIMEOUT_WRITE, pool=TIMEOUT_POOL
    )

    cur_mod = await _post_json(
        f"{CURR_BASE}/modules",
        {**common, "learning_outcomes": outcomes},
        per_request_timeout=long_timeout,
    )
    modules = cur_mod.get("modules") or {}
    if req.include_trace:
        trace["steps"].append({"modules": modules})

    cur_ass = await _post_json(
    f"{CURR_BASE}/assessments",
    {**common, "modules": modules},
    )

    # Normalize assessments safely
    assessments = cur_ass.get("assessments") if isinstance(cur_ass, dict) else (cur_ass if isinstance(cur_ass, list) else [])

    if req.include_trace:
        trace["steps"].append({"assessments": assessments})

    totals = _extract_totals(modules)

    response = {
        "course_title": title,
        "level": req.level,
        "grounding_summary": redacted,
        "outcomes": outcomes,
        "modules": modules,
        "assessments": assessments,
        "modules_count": totals["modules_count"],
        "total_estimated_hours": totals["total_estimated_hours"],
    }
    if req.include_citations:
        response["citations"] = ir.get("results", [])
    if req.include_trace:
        response["trace"] = trace

    return response
