@echo off
echo ==============================================
echo    Iniciando LMStudio Full Stack com Docker
echo ==============================================
echo.
echo Incluindo ChromaDB Vector Database
echo.
echo 1. Parando containers existentes...
docker-compose down

echo.
echo 2. Construindo imagens Docker...
docker-compose build

echo.
echo 3. Iniciando todos os serviços...
docker-compose up -d

echo.
echo 4. Aguardando serviços iniciarem...
timeout /t 10 /nobreak >nul

echo.
echo 5. Verificando status dos containers...
docker-compose ps

echo.
echo ==============================================
echo  Serviços Docker iniciados com sucesso!
echo ==============================================
echo  LMStudio Chat: http://localhost:8080
echo  API Python: http://localhost:8000  
echo  Interface React: http://localhost:3000
echo  ChromaDB: http://localhost:8200
echo.
echo  Para ver logs: docker-compose logs -f
echo  Para parar: docker-compose down
echo ==============================================
pause
