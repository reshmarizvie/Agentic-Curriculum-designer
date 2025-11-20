import os, sys, time, subprocess, signal, json, pathlib
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
from dotenv import load_dotenv

ROOT = pathlib.Path(__file__).resolve().parent
VENV = ROOT / ".venv" / "Scripts"  # Windows venv path
PYTHON = str(VENV / "python.exe")
UVICORN = str(VENV / "uvicorn.exe")
STREAMLIT = str(VENV / "streamlit.exe")

# Load .env for all agents
load_dotenv(dotenv_path=ROOT / ".env")

# --- Service definitions ---
CURR = {"name": "curriculum",  "module": "apps.curriculum_agent.main:app", "port": 8001, "health": ("GET", "http://localhost:8001/")}
IR   = {"name": "ir",          "module": "apps.ir_agent.main:app",         "port": 8002, "health": ("POST", "http://localhost:8002/search", {"query": "ping", "top_k": 1})}
NLP  = {"name": "nlp",         "module": "apps.nlp_agent.main:app",        "port": 8003, "health": ("POST", "http://localhost:8003/summarize", {"text": "ping", "max_words": 12})}
ORCH = {"name": "orchestrator","module": "apps.orchestrator.main:app",     "port": 8000, "health": ("GET", "http://localhost:8000/health")}

STREAMLIT_UI = str(ROOT / "streamlit_app.py")

# --- Shared env for all services ---
ENV_COMMON = {
    "FAST_MODE": "1"  # override for fast test mode
}

def _env(extra=None):
    env = os.environ.copy()
    env.update(ENV_COMMON)
    if extra:
        env.update(extra)
    env["PYTHONPATH"] = str(ROOT)
    return env

def _spawn_uvicorn(name, module, port, reload=False):
    args = [UVICORN, module, "--host", "0.0.0.0", "--port", str(port)]
    if reload:
        args.append("--reload")
    print(f"[spawn] {name} on port {port}")
    return subprocess.Popen(args, env=_env(), cwd=str(ROOT))

def _http_check(method, url, payload=None, timeout=2.5):
    try:
        if method == "GET":
            req = Request(url, method="GET")
        else:
            data = json.dumps(payload or {}).encode()
            req = Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
        with urlopen(req, timeout=timeout) as resp:
            return True, resp.status
    except Exception as e:
        return False, str(e)

def _wait_healthy(svc, tries=30, delay=1.0):
    print(f"[wait] {svc['name']} ...")
    method, url, *rest = svc["health"]
    payload = rest[0] if rest else None
    for _ in range(tries):
        ok, _ = _http_check(method, url, payload)
        if ok:
            print(f"[ok] {svc['name']} ready on port {svc['port']}")
            return True
        time.sleep(delay)
    print(f"[fail] {svc['name']} not responding.")
    return False

def _shutdown(procs):
    print("\n[shutdown] stopping all...")
    for p in procs.values():
        if p and p.poll() is None:
            p.terminate()
    time.sleep(1)
    for p in procs.values():
        if p and p.poll() is None:
            p.kill()

def main():
    with_ui = "--ui" in sys.argv
    reload_mode = "--reload" in sys.argv
    print(f"\n[launcher] Starting all agents {'with UI' if with_ui else ''}\n")

    procs = {}
    try:
        procs["curriculum"] = _spawn_uvicorn("curriculum", CURR["module"], CURR["port"], reload_mode)
        _wait_healthy(CURR)

        procs["ir"] = _spawn_uvicorn("ir", IR["module"], IR["port"], reload_mode)
        _wait_healthy(IR)

        procs["nlp"] = _spawn_uvicorn("nlp", NLP["module"], NLP["port"], reload_mode)
        _wait_healthy(NLP)

        procs["orchestrator"] = _spawn_uvicorn("orchestrator", ORCH["module"], ORCH["port"], reload_mode)
        _wait_healthy(ORCH)

        if with_ui:
            print("[spawn] Streamlit UI")
            procs["streamlit"] = subprocess.Popen([STREAMLIT, "run", STREAMLIT_UI], env=_env(), cwd=str(ROOT))
            print("→ open: http://localhost:8501")

        print("\n✅ All services running. Press Ctrl+C to stop.\n")
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        pass
    finally:
        _shutdown(procs)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    main()
