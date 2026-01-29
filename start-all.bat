@echo off
cls
echo ==============================================
echo    FIAP - Full Stack with LMStudio + ChromaDB
echo ==============================================
echo.
echo Checking LMStudio...
netstat -ano | findstr :1234 >nul 2>&1
if errorlevel 1 (
    echo WARNING: LMStudio not found on port 1234
    echo Please start LMStudio before running this script
    echo.
    pause
    exit /b 1
) else (
    echo [OK] LMStudio is running on port 1234
)
echo.
echo Starting services...
echo.
echo [1] Starting ChromaDB locally...
start "ChromaDB" cmd /k "cd /d C:\Source\FIAPTCC\fiap_chromadb && python run_server.py"
if errorlevel 1 (
    echo [ERROR] Failed to start ChromaDB
    pause
    exit /b 1
)
timeout /t 5 /nobreak
echo [OK] ChromaDB started
echo.

echo [2] Starting Python API...
start "API" cmd /k "cd /d C:\Source\FIAPTCC\fiap_api && python main.py"
if errorlevel 1 (
    echo [ERROR] Failed to start Python API
    pause
    exit /b 1
)
timeout /t 5 /nobreak
echo [OK] Python API started
echo.

echo [3] Starting React Interface...
start "UI" cmd /k "cd /d C:\Source\FIAPTCC\fiap_interface && npm start"
if errorlevel 1 (
    echo [ERROR] Failed to start React Interface
    pause
    exit /b 1
)
echo [OK] React Interface started
echo.

echo ==============================================
echo Services started successfully:
echo  API: http://localhost:8000
echo  UI: http://localhost:3000
echo  ChromaDB: http://localhost:8200
echo ==============================================
timeout /t 3 /nobreak
echo.
echo Done!
pause
