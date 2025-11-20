# streamlit_app.py
import os
import json
import time
import typing as t

import streamlit as st
import requests

APP_TITLE = "Agentic Curriculum Designer (ACD) — Streamlit UI"

# -------- Config --------
DEFAULT_ORCH_BASE = os.getenv("ORCH_BASE", "http://localhost:8000")
DEFAULT_LEVELS = ["Beginner", "Intermediate", "Advanced"]
DEFAULT_MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "400"))

# -------- Helpers --------
def get_headers(token: str | None) -> dict[str, str]:
    h = {"Content-Type": "application/json"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h

def api_get(base: str, path: str, token: str | None = None, timeout: int = 30) -> tuple[int, dict | str]:
    try:
        r = requests.get(f"{base}{path}", headers=get_headers(token), timeout=timeout)
        if "application/json" in r.headers.get("content-type", ""):
            return r.status_code, r.json()
        return r.status_code, r.text
    except Exception as e:
        return 0, f"{type(e).__name__}: {e}"

def api_post(base: str, path: str, payload: dict, token: str | None = None, timeout: int = 120) -> tuple[int, dict | str]:
    try:
        r = requests.post(f"{base}{path}", headers=get_headers(token), data=json.dumps(payload), timeout=timeout)
        if "application/json" in r.headers.get("content-type", ""):
            return r.status_code, r.json()
        return r.status_code, r.text
    except Exception as e:
        return 0, f"{type(e).__name__}: {e}"

def pretty_json(obj: t.Any) -> str:
    try:
        return json.dumps(obj, indent=2, ensure_ascii=False)
    except Exception:
        return str(obj)

def download_button_bytes(data: bytes, filename: str, label: str):
    import base64
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">{label}</a>'
    st.markdown(href, unsafe_allow_html=True)

# -------- UI --------
st.set_page_config(page_title="ACD — Streamlit", page_icon="🧠", layout="wide")

# Initialize defaults BEFORE widgets that use these keys
if "orch_base" not in st.session_state:
    st.session_state["orch_base"] = DEFAULT_ORCH_BASE
if "notes" not in st.session_state:
    st.session_state["notes"] = "Focus on practical labs where possible. Prefer Bloom's measurable verbs."

st.title(APP_TITLE)
st.caption("Front-end for your multi-agent FastAPI backend (IR, NLP, Curriculum, Orchestrator).")

with st.sidebar:
    st.header("🔧 Settings")
    st.text_input("Orchestrator Base URL", key="orch_base", value=st.session_state["orch_base"], help="e.g., http://localhost:8000 or http://orch:8000")
    use_dev_token = st.checkbox("Get Dev Token from /token", value=True, help="If disabled, paste a JWT manually below.")
    dev_user = st.text_input("Dev Username (sub)", value="demo_user", help="Used only if 'Get Dev Token' is enabled.")
    manual_token = st.text_area("Manual JWT (optional)", value="", help="Paste a Bearer token. Ignored if 'Get Dev Token' is enabled.")
    st.divider()
    st.write("**Backend Health Checks**")
    if st.button("Ping /health"):
        status, data = api_get(st.session_state["orch_base"], "/health", token=None)
        st.code(f"HTTP {status}\n{pretty_json(data)}")
    if st.button("Who Am I (/whoami) — requires token"):
        token = None
        if use_dev_token:
            # try minting a short-lived token
            s, tok = api_post(st.session_state["orch_base"], "/token", {"sub": dev_user})
            if s == 200 and isinstance(tok, dict) and "access_token" in tok:
                token = tok["access_token"]
            else:
                st.error(f"Failed to get token: HTTP {s} {tok}")
        else:
            token = manual_token.strip() or None
        s2, data2 = api_get(st.session_state["orch_base"], "/whoami", token=token)
        st.code(f"HTTP {s2}\n{pretty_json(data2)}")

# Use a local variable without mutating session_state after widget creation
ORCH_BASE = st.session_state["orch_base"]

st.subheader("🎯 Design a Course")
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    course_title = st.text_input("Course Title", value="Cyber Security")
with col2:
    level = st.selectbox("Level", DEFAULT_LEVELS, index=0)
with col3:
    max_new_tokens = st.number_input("LLM Max New Tokens", min_value=128, max_value=2048, value=DEFAULT_MAX_NEW_TOKENS, step=32)

st.text_area(
    "Optional Notes to Guide IR/NLP (will be appended to the prompt)",
    key="notes",
    value=st.session_state["notes"],
    help="This text is sent to the Orchestrator to influence the grounding/prompt."
)

advanced = st.expander("Advanced Options", expanded=False)
with advanced:
    top_k = st.slider("IR: Top-K Documents", 1, 10, 5)
    return_trace = st.checkbox("Include trace in response", value=True)
    return_citations = st.checkbox("Include citations in response", value=True)

submit = st.button("🚀 Generate Curriculum", type="primary", use_container_width=True)

# -------- Submit --------
if submit:
    # Obtain token first (dev or manual)
    token: str | None = None
    if use_dev_token:
        s, tok = api_post(ORCH_BASE, "/token", {"sub": dev_user})
        if s == 200 and isinstance(tok, dict) and "access_token" in tok:
            token = tok["access_token"]
        else:
            st.error(f"Failed to get token from /token: HTTP {s} {tok}")
    else:
        token = manual_token.strip() or None

    if not token:
        st.stop()

    payload = {
        "course_title": course_title.strip(),
        "level": level,
        "notes": st.session_state.get("notes", ""),
        "ir_top_k": top_k,
        "include_trace": return_trace,
        "include_citations": return_citations,
        "max_new_tokens": int(max_new_tokens),
    }

    with st.status("Calling Orchestrator → IR → NLP → Curriculum…", state="running"):
        t0 = time.time()
        status_code, data = api_post(ORCH_BASE, "/design", payload, token=token, timeout=600)
        elapsed = time.time() - t0

    if status_code != 200:
        st.error(f"Backend error (HTTP {status_code}). See details below:")
        st.code(pretty_json(data))
        st.stop()

    st.success(f"Generated in {elapsed:.1f}s")
    st.divider()

    # ---- Overview ----
    meta_cols = st.columns([2, 1, 1, 1])
    with meta_cols[0]:
        st.markdown(f"### 🧠 {data.get('course_title', course_title)}")
        st.caption(f"Level: **{data.get('level', level)}**")
    with meta_cols[1]:
        hours = data.get("total_estimated_hours")
        if hours is not None:
            st.metric("Estimated Hours", hours)
    with meta_cols[2]:
        mcount = data.get("modules_count") or (len(data.get("modules", [])) if isinstance(data.get("modules"), list) else "—")
        st.metric("Modules", mcount)
    with meta_cols[3]:
        st.metric("Assessments", len(data.get("assessments", [])) if isinstance(data.get("assessments"), list) else 0)

    # ---- Grounding Summary ----
    if data.get("grounding_summary"):
        st.markdown("#### 📚 Grounding Summary")
        st.write(data["grounding_summary"])

    # ---- Learning Outcomes (course-level, optional) ----
    if data.get("learning_outcomes"):
        st.markdown("#### 🎯 Course Learning Outcomes")
        for i, lo in enumerate(data["learning_outcomes"], 1):
            st.write(f"{i}. {lo}")

    # ---- Modules ----
    modules = data.get("modules", [])
    if isinstance(modules, dict):
        modules = modules.get("modules", modules.get("items", [])) or []

    st.markdown("#### 🧩 Modules")
    if isinstance(modules, list) and modules:
        for m in modules:
            with st.expander(f"Module {m.get('module_no', '?')}: {m.get('title', 'Untitled')}", expanded=False):
                st.markdown(f"**Description:** {m.get('description', '')}")
                if m.get("estimated_hours") is not None:
                    st.write(f"**Estimated Hours:** {m['estimated_hours']}")
                if m.get("topics"):
                    st.write("**Topics:**")
                    for tpc in m["topics"]:
                        st.write(f"- {tpc}")
                if m.get("outcomes"):
                    st.write("**Outcomes:**")
                    for o in m["outcomes"]:
                        st.write(f"- {o}")
                if m.get("assessments"):
                    st.write("**Assessments:**")
                    for a in m["assessments"]:
                        st.write(f"- {a}")
    else:
        st.info("No modules returned.")

    # ---- Course-level Assessments ----
    if data.get("assessments"):
        st.markdown("#### 📝 Course Assessments")
        for i, qa in enumerate(data["assessments"], 1):
            st.write(f"{i}. {qa}")

    # ---- Citations ----
    if data.get("citations"):
        st.markdown("#### 🔎 Citations")
        for c in data["citations"]:
            title = c.get("title") or "Source"
            snippet = c.get("snippet", "")
            st.write(f"- **{title}** — {snippet}")

    # ---- Trace ----
    if data.get("trace"):
        st.markdown("#### 🧪 Trace (debug)")
        st.code(pretty_json(data["trace"]))

    # ---- Raw JSON + Download ----
    st.markdown("#### ⬇️ Export")
    raw = pretty_json(data)
    st.code(raw, language="json")
    download_button_bytes(raw.encode("utf-8"), filename=f"{course_title.replace(' ', '_').lower()}_curriculum.json", label="Download JSON")

else:
    st.info("Fill the form and click **Generate Curriculum** to run the full pipeline.")
    st.caption("Tip: The Orchestrator should expose /token, /whoami, /health, and /design endpoints as described in your project.")
