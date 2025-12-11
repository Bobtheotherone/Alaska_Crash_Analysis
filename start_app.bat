@echo off
setlocal

REM ============================================
REM  Move to the folder this .bat lives in
REM ============================================
cd /d "%~dp0"

REM ============================================
REM  Activate conda environment (alaska-gpu)
REM ============================================
set "MINICONDA=C:\Users\dimen\miniconda3"

IF EXIST "%MINICONDA%\Scripts\activate.bat" (
    echo [%date% %time%] Activating conda env alaska-gpu from %MINICONDA%
    call "%MINICONDA%\Scripts\activate.bat" alaska-gpu
) ELSE (
    echo [%date% %time%] [WARN] Miniconda activate.bat not found at %MINICONDA% - using system Python.
)

REM ============================================
REM  DATABASE CREDENTIALS (match settings.py)
REM ============================================
set "POSTGRES_DB=alaska_crash_analysis"
set "POSTGRES_USER=postgres"
set "POSTGRES_PASSWORD=Bobtheother_101"
set "POSTGRES_HOST=localhost"
set "POSTGRES_PORT=5432"

REM Let manage.py decide the settings module
set "DJANGO_SETTINGS_MODULE="
set "PYTHONUNBUFFERED=1"

echo.
echo [%date% %time%] ============================================
echo [%date% %time%] Starting Alaska Crash Data Analysis
echo [%date% %time%] Project dir: %CD%
echo [%date% %time%] Python / Django env:
python -c "import sys, os; print('  exe:', sys.executable); print('  settings:', os.environ.get('DJANGO_SETTINGS_MODULE')); print('  DB_USER:', os.environ.get('POSTGRES_USER') or os.environ.get('DB_USER')); print('  DB_NAME:', os.environ.get('POSTGRES_DB') or os.environ.get('DB_NAME'))"
echo [%date% %time%] ============================================
echo.

REM ============================================
REM  Minimal OS + GPU info
REM ============================================
echo [%date% %time%] ===== OS INFO =====
ver
systeminfo | findstr /B /C:"OS Name" /C:"OS Version"

echo.
echo [%date% %time%] ===== GPU INFO (NVIDIA via nvidia-smi) =====
where nvidia-smi >nul 2>&1
IF ERRORLEVEL 1 (
    echo [%date% %time%] nvidia-smi not found in PATH. Install NVIDIA drivers or add nvidia-smi to PATH.
) ELSE (
    echo [%date% %time%] nvidia-smi found, querying GPU information:
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader,nounits
    nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu,utilization.memory --format=csv,noheader,nounits
)

echo.
echo [%date% %time%] ===== END OF SYSTEM SUMMARY =====
echo.

REM ============================================
REM  Wait for PostgreSQL to be ready
REM ============================================
set "DB_WAIT_MAX_ATTEMPTS=30"
set /a DB_WAIT_COUNT=0

:wait_for_db
set /a DB_WAIT_COUNT+=1
echo [%date% %time%] Checking database availability (attempt %DB_WAIT_COUNT%/%DB_WAIT_MAX_ATTEMPTS%)...

python manage.py migrate --check >nul 2>&1
IF %ERRORLEVEL% EQU 0 GOTO db_ready

echo [%date% %time%] Database not ready yet; waiting 2 seconds...
IF %DB_WAIT_COUNT% GEQ %DB_WAIT_MAX_ATTEMPTS% GOTO db_timeout

timeout /t 2 /nobreak >nul 2>&1
GOTO wait_for_db

:db_ready
echo [%date% %time%] Database appears ready (migrate --check succeeded).
GOTO after_db_wait

:db_timeout
echo [%date% %time%] [ERROR] Database did not become ready after %DB_WAIT_MAX_ATTEMPTS% attempts.
echo [%date% %time%] Proceeding to start Django anyway. You may still see database connection errors if PostgreSQL is not actually up.

:after_db_wait

REM ============================================
REM  Start Django development server (new window)
REM  NOTE: we use --noreload to avoid StatReloader / WinError 123 noise
REM ============================================
echo [%date% %time%] Starting Django development server on 0.0.0.0:8000 (no autoreload) in new window...
start "django-dev" cmd /k "cd /d \"%CD%\" & python manage.py runserver --noreload 0.0.0.0:8000"

REM Give Django a few seconds to boot up before we hit it from the browser
echo [%date% %time%] Giving Django a few seconds to start...
timeout /t 8 /nobreak >nul 2>&1

REM ============================================
REM  FRONTEND DEV SERVER (Vite in alaska_ui on 5173)
REM ============================================
set "FRONTEND_DIR=%CD%\alaska_ui"

IF NOT EXIST "%FRONTEND_DIR%\package.json" (
    echo [%date% %time%] [WARN] No package.json found in "%FRONTEND_DIR%" - skipping frontend dev server on 5173.
    GOTO after_frontend
)

echo [%date% %time%] Frontend directory: "%FRONTEND_DIR%"
pushd "%FRONTEND_DIR%"

IF NOT EXIST "node_modules" (
    echo [%date% %time%] node_modules not found - installing frontend dependencies with npm install
    call npm install
)

echo [%date% %time%] Launching frontend dev server with "npm run dev" on port 5173
start "frontend-dev" cmd /k "cd /d \"%FRONTEND_DIR%\" & npm run dev"
popd

:after_frontend

REM ============================================
REM  URLs
REM ============================================
set "BACKEND_URL=http://127.0.0.1:8000/"
set "FRONTEND_URL=http://127.0.0.1:5173/"

REM Small wait so Vite is up before opening browser
echo [%date% %time%] Waiting 6 seconds for frontend dev server to start...
timeout /t 6 /nobreak >nul 2>&1

echo [%date% %time%] Back-end on %BACKEND_URL%
echo [%date% %time%] Front-end on %FRONTEND_URL%

REM ============================================
REM  Auto-open frontend (and optionally backend) in default browser
REM  Using PowerShell Start-Process to avoid Windows 'filename' quirks.
REM ============================================
echo [%date% %time%] Opening frontend in your default browser via PowerShell...
powershell -Command "Start-Process '%FRONTEND_URL%'"

REM If you ALSO want the Django backend page to open automatically,
REM uncomment the two lines below:
REM echo [%date% %time%] Opening backend in your default browser via PowerShell...
REM powershell -Command "Start-Process '%BACKEND_URL%'"

echo.
echo [%date% %time%] All dev services started.
echo [%date% %time%] Django logs:   window titled "django-dev"
echo [%date% %time%] Frontend logs: window titled "frontend-dev"
echo [%date% %time%] Close those two windows to stop the dev environment.

endlocal
exit /b
