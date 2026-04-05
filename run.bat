@echo off

echo 1. Stopping existing containers...
docker-compose down
docker system prune -af
docker builder prune -af

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
echo  API Python: http://localhost:8001  
echo  Interface React: http://localhost:3000
echo  ChromaDB: http://localhost:8200
echo.
echo  To view logs: docker-compose logs -f
echo  To stop: docker-compose down
echo ==============================================
start http://localhost:3000
