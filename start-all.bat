@echo off
echo ==============================================
echo    FIAP - Full Stack com LMStudio + ChromaDB
echo ==============================================
echo.
echo 1. Iniciando API Python (porta 8000)...
start cmd /k "cd /d C:\Source\lmstudio\fiap_api && python main.py"

echo.
echo 2. Aguardando 3 segundos...
timeout /t 3 /nobreak >nul

echo.
echo 3. Iniciando Interface React (porta 3000)...
start cmd /k "cd /d C:\Source\lmstudio\fiap_interface && npm start"

echo.
echo ==============================================
echo  Serviços iniciados com sucesso!
echo ==============================================
echo  API Python: http://localhost:8000
echo  Interface React: http://localhost:3000
echo.
echo  💡 Para testes VectorDB: Aba Tests > VectorDB
echo  💡 Para ChromaDB: start-chromadb.bat
echo  💡 Certifique-se de que o LMStudio está rodando
echo  na porta 1234 antes de usar o chat.
echo ==============================================
pause
