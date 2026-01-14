@echo off
cls
echo ==============================================
echo    FIAP - Full Stack with LMStudio + ChromaDB
echo ==============================================
echo.
echo Checking Docker...
docker --version >nul 2>&1
if errorlevel 1 (
    echo WARNING: Docker not found. ChromaDB will not start.
    echo Please install Docker Desktop to use ChromaDB.
    echo.
)
echo.
echo Starting services...
echo.
echo [1] Starting ChromaDB...
docker run -d --name lmstudio-chromadb -p 8200:8200 -v chromadb-data:/chroma/data chromadb/chroma:latest >nul 2>&1
timeout /t 5 /nobreak
echo.
echo [2] Starting Python API...
start cmd /k "cd /d C:\Source\FIAPTCC\fiap_api && python main.py"
timeout /t 15 /nobreak
echo.
echo [3] Starting React Interface...
start cmd /k "cd /d C:\Source\FIAPTCC\fiap_interface && npm start"
echo.
echo ==============================================
echo Services starting:
echo  API: http://localhost:8000
echo  UI: http://localhost:3000
echo  ChromaDB: http://localhost:8200
echo ==============================================
timeout /t 3 /nobreak
start http://localhost:3000
echo.
echo Done!
pause
