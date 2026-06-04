Vou alterar o `debug.bat` para executar o `docker-compose-local.yml` em vez de iniciar os serviços manualmente, e remover a parte do ChromaDB que agora está no compose.

```batch file
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
echo [1] Starting Docker Compose...
docker-compose -f docker-compose-local.yml up -d
if errorlevel 1 (
    echo [ERROR] Failed to start Docker Compose
    pause
    exit /b 1
)
timeout /t 10 /nobreak
echo [OK] Docker Compose started
echo.

echo [2] Starting Python API...
start "API" cmd /k "cd /d fiap_api && python main.py"
if errorlevel 1 (
    echo [ERROR] Failed to start Python API
    pause
    exit /b 1
)
timeout /t 5 /nobreak
echo [OK] Python API started
echo.

echo [3] Starting React Interface...
start "UI" cmd /k "cd /d fiap_interface && npm start"
if errorlevel 1 (
    echo [ERROR] Failed to start React Interface
    pause
    exit /b 1
)
echo [OK] React Interface started
echo.

echo ==============================================
echo Services started successfully:
echo  API: http://localhost:8001
echo  UI: http://localhost:3000
echo  ChromaDB: http://localhost:8210
echo ==============================================
timeout /t 3 /nobreak
echo.
echo Done!
pause
```