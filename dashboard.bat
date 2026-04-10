@echo off
echo ========================================================
echo Starting High Precision Proton Therapy Dashboard...
echo ========================================================

:: 1. Start FastAPI Backend in a new window
echo Starting Python Backend (FastAPI)...
start "Proton AI - Backend" cmd /c ".\venv_projeto\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8000 --reload"

:: 2. Wait a couple of seconds for the backend to initialize
timeout /t 3 /nobreak > nul

:: 3. Start Vite Frontend in a new window
echo Starting React Frontend (Vite)...
cd frontend
start "Proton AI - Frontend" cmd /c "npm run dev -- --open"

echo.
echo All services launched!
echo The dashboard should automatically open in your web browser.
echo Close the two new terminal windows to stop the servers.
pause
