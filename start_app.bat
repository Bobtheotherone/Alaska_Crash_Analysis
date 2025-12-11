@echo off
setlocal EnableDelayedExpansion

REM ============================================
REM  Move to the folder this .bat lives in
REM ============================================
cd /d "%~dp0"

set "BACKEND_URL=http://127.0.0.1:8000"
set "FRONTEND_URL=http://127.0.0.1:5173"
set "FRONTEND_DIR=%CD%\alaska_ui"
set "DOCKER_COMPOSE_CMD=docker compose"
set "BACKEND_STARTED=0"

echo.
echo [%date% %time%] ============================================
echo [%date% %time%] Starting Alaska Crash Data Analysis - Docker backend + local frontend
echo [%date% %time%] ============================================
echo.

REM Optional: force local backend instead of Docker
IF /I "%USE_LOCAL_BACKEND%"=="1" (
    echo [%date% %time%] USE_LOCAL_BACKEND=1 detected. Starting local Django runserver; antivirus is likely skipped. Recommended path is the Docker backend.
    GOTO local_backend
)

REM ============================================
REM  Ensure Docker is available and start backend (db + clamav + web)
REM ============================================
where docker >nul 2>&1
IF ERRORLEVEL 1 (
    echo [%date% %time%] [ERROR] Docker is not available in PATH. Start Docker Desktop or set USE_LOCAL_BACKEND=1 to fall back to a local runserver with no AV.
    GOTO frontend_only
)

set "DOCKER_COMPOSE_CMD=docker compose"
docker compose version >nul 2>&1
IF ERRORLEVEL 1 (
    docker-compose --version >nul 2>&1
    IF ERRORLEVEL 1 (
        echo [%date% %time%] [ERROR] Neither "docker compose" nor "docker-compose" is available. Install Docker Desktop or set USE_LOCAL_BACKEND=1 for a local backend without AV.
        GOTO frontend_only
    )
    set "DOCKER_COMPOSE_CMD=docker-compose"
)

echo [%date% %time%] Bringing up Docker services: db, clamav, web ...
%DOCKER_COMPOSE_CMD% up -d db clamav web
IF ERRORLEVEL 1 (
    echo [%date% %time%] [ERROR] Failed to start Docker services. Check Docker Desktop or run "%DOCKER_COMPOSE_CMD% logs web".
    GOTO frontend_only
)
set "BACKEND_STARTED=1"

REM ============================================
REM  Wait briefly for backend HTTP to respond
REM ============================================
set /a WEB_WAIT_MAX=20
for /l %%I in (1,1,%WEB_WAIT_MAX%) do (
    powershell -NoProfile -Command "try { Invoke-WebRequest -UseBasicParsing -Uri '%BACKEND_URL%/' -TimeoutSec 2 | Out-Null; exit 0 } catch { exit 1 }" >nul 2>&1
    IF !errorlevel! EQU 0 (
        echo [%date% %time%] Backend responded at %BACKEND_URL% - attempt %%I of %WEB_WAIT_MAX%.
        GOTO backend_ready
    )
    echo [%date% %time%] Waiting for backend at %BACKEND_URL% - attempt %%I of %WEB_WAIT_MAX% ...
    timeout /t 2 /nobreak >nul 2>&1
)
echo [%date% %time%] [WARN] Backend did not respond yet. Frontend will still start; check "%DOCKER_COMPOSE_CMD% logs web".

:backend_ready

REM ============================================
REM  FRONTEND DEV SERVER (Vite in alaska_ui on 5173)
REM ============================================
:frontend_only
IF NOT EXIST "%FRONTEND_DIR%\package.json" (
    echo [%date% %time%] [WARN] No package.json found in "%FRONTEND_DIR%". Skipping frontend dev server on port 5173.
    GOTO done
)

echo [%date% %time%] Frontend directory: "%FRONTEND_DIR%"
pushd "%FRONTEND_DIR%"

IF NOT EXIST "node_modules" (
    echo [%date% %time%] node_modules not found - installing frontend dependencies with npm install ...
    call npm install
)

REM Point Vite proxy/API to the Docker backend on localhost:8000
set "VITE_API_BASE_URL=%BACKEND_URL%"

echo [%date% %time%] Launching frontend dev server with "npm run dev" on port 5173, proxying to %BACKEND_URL%.
REM Env vars set in this script are inherited by the new cmd, so we just need npm run dev here.
start "frontend-dev" cmd /k "npm run dev"

popd

REM ============================================
REM  Auto-open frontend in default browser
REM ============================================
echo [%date% %time%] Waiting 6 seconds for frontend dev server to start...
timeout /t 6 /nobreak >nul 2>&1

echo [%date% %time%] Front-end URL: %FRONTEND_URL%
echo [%date% %time%] Back-end URL:  %BACKEND_URL%

echo [%date% %time%] Opening frontend in your default browser via PowerShell...
powershell -Command "Start-Process '%FRONTEND_URL%'"

echo.
echo [%date% %time%] Frontend logs: window titled "frontend-dev"
IF "%BACKEND_STARTED%"=="1" (
    echo [%date% %time%] Close that window to stop the frontend. The Docker backend stays running until you run "%DOCKER_COMPOSE_CMD% down".
) ELSE (
    echo [%date% %time%] Close that window to stop the frontend.
)
GOTO done

REM ============================================
REM  Optional fallback: local Django runserver (no Docker, AV likely skipped)
REM ============================================
:local_backend
echo [%date% %time%] Starting local Django runserver on 0.0.0.0:8000 without autoreload. Antivirus scanning may be skipped without clamd.
start "django-dev" cmd /k "cd /d \"%CD%\" ^& python manage.py runserver --noreload 0.0.0.0:8000"
echo [%date% %time%] Waiting 6 seconds for local backend to start...
timeout /t 6 /nobreak >nul 2>&1
GOTO frontend_only

:done
endlocal
exit /b
