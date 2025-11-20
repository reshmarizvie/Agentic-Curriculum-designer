@echo off
setlocal
REM ============================================================
REM  Agentic Curriculum Launcher (Gemma-2B Local Snapshot Mode)
REM ============================================================

cd /d "%~dp0"
set PY=%~dp0.venv\Scripts\python.exe

REM use your cached Gemma snapshot
set CURRICULUM_MODEL=C:\Users\shaki\.cache\huggingface\hub\models--google--gemma-2b-it\snapshots\96988410cbdaeb8d5093d1ebdc5a8fb563e02bad

set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
set SKIP_MODEL_LOAD=0
set FAST_MODE=1
set PYTHONPATH=%~dp0

echo =============================================
echo Launching Agentic Curriculum System (Local Gemma Snapshot)
echo Using cached model: %CURRICULUM_MODEL%
echo Virtual env: %PY%
echo =============================================

"%PY%" "%~dp0run_acd.py" --ui

pause
