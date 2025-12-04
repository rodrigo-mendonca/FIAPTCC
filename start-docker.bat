@echo off
echo ==============================================
echo    Starting LMStudio Full Stack with Docker
echo ==============================================
echo.
echo Including ChromaDB Vector Database
echo.
echo 1. Stopping existing containers...
docker-compose down

echo.
echo 2. Building Docker images...
docker-compose build

echo.
echo 3. Starting all services...
docker-compose up -d

echo.
echo 4. Waiting for services to start...
timeout /t 10 /nobreak >nul

echo.
echo 5. Checking container status...
docker-compose ps

echo.
echo ==============================================
echo  Docker services started successfully!
echo ==============================================
echo  LMStudio Chat: http://localhost:8080
echo  API Python: http://localhost:8000  
echo  Interface React: http://localhost:3000
echo  ChromaDB: http://localhost:8200
echo.
echo  To view logs: docker-compose logs -f
echo  To stop: docker-compose down
echo ==============================================
pause
