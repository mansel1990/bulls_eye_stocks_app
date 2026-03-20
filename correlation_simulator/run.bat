@echo off
echo Starting Correlation Simulator...
cd /d "%~dp0"
uvicorn app:app --reload --port 5050 --host 0.0.0.0
pause
