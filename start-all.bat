@echo off
echo ==============================================
echo    FIAP - Full Stack with LMStudio + ChromaDB
echo ==============================================
echo.
echo 1. Starting Python API (port 8000)...
start cmd /k "cd /d C:\Source\lmstudio\fiap_api && python main.py"

echo.
echo 2. Waiting 3 seconds...
timeout /t 3 /nobreak >nul

echo.
echo 3. Starting React Interface (port 3000)...
start cmd /k "cd /d C:\Source\lmstudio\fiap_interface && npm start"

echo.
echo ==============================================
echo  Services started successfully!
echo ==============================================
echo  API Python: http://localhost:8000
echo  Interface React: http://localhost:3000
echo.
echo  💡 For VectorDB tests: Tests Tab > VectorDB
echo  💡 For ChromaDB: start-chromadb.bat
echo  💡 Make sure LMStudio is running
echo  on port 1234 before using the chat.
echo ==============================================
pause
