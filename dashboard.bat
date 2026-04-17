@echo off
setlocal

set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"
cd /d "%ROOT%"

set "PY=%ROOT%\venv_projeto\Scripts\python.exe"
set "BACKEND_LOG=%ROOT%\backend.log"

echo ========================================================
echo Starting High Precision Proton Therapy Dashboard...
echo ========================================================

if not exist "%PY%" goto missing_python
if not exist "%ROOT%\mc_engine.exe" goto build_engine
if not exist "%ROOT%\frontend\node_modules" goto install_frontend
goto launch

:build_engine
echo [INFO] mc_engine.exe nao encontrado. A compilar automaticamente...
call "%ROOT%build_vs.bat"
if errorlevel 1 goto build_failed
if not exist "%ROOT%\mc_engine.exe" goto build_failed
if not exist "%ROOT%\frontend\node_modules" goto install_frontend
goto launch

:install_frontend
echo [INFO] A instalar dependencias do frontend (npm install)...
pushd "%ROOT%\frontend"
call npm install
if errorlevel 1 (
  popd
  goto npm_failed
)
popd
goto launch

:launch
echo Starting Python Backend (FastAPI)...
start "Proton AI - Backend" /D "%ROOT%" cmd /k ""%PY%" -m uvicorn api:app --host 127.0.0.1 --port 8000 --reload"

timeout /t 4 /nobreak >nul
echo Starting React Frontend (Vite)...
start "Proton AI - Frontend" /D "%ROOT%\frontend" cmd /k "npm run dev -- --open"

echo.
echo All services launched successfully.
echo Close the two new terminal windows to stop the servers.
pause
exit /b 0

:missing_python
echo [ERROR] Python do venv nao encontrado:
echo         %PY%
pause
exit /b 1

:build_failed
echo [ERROR] Falha ao compilar o MC engine.
pause
exit /b 1

:npm_failed
echo [ERROR] Falha no npm install.
pause
exit /b 1
