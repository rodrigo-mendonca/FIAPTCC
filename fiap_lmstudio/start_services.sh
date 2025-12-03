#!/bin/bash

# Otimizações de performance e forçar GPU
export CUDA_VISIBLE_DEVICES=0
export NVIDIA_VISIBLE_DEVICES=all
export OMP_NUM_THREADS=8
export CUDA_CACHE_DISABLE=0
export CUDA_LAUNCH_BLOCKING=0
export LM_STUDIO_FORCE_CUDA=1
export LLAMA_CUDA=1
export GGML_USE_CUDA=1

# Configurar display
export DISPLAY=:99

# Ajustar prioridades do sistema
echo "🔧 Aplicando otimizações de sistema..."
# Permitir uso máximo de memória
echo never > /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null || true
echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true

# Iniciar Xvfb com configurações otimizadas
Xvfb :99 -screen 0 1920x1080x24 -ac +extension GLX +render -noreset &

# Aguardar X11 inicializar
sleep 3

# Remover locks
rm -f /tmp/.X99-lock

# Configurar auto-start OTIMIZADO
echo "🔧 Configurando auto-start otimizado..."
/configure_autostart.sh

# Configuração HTTP otimizada
mkdir -p /root/.cache/lm-studio/.internal
cp /http-server-config.json /root/.cache/lm-studio/.internal/http-server-config.json

# Pré-aquecer GPU
echo "🔥 Pré-aquecendo GPU..."
nvidia-smi -pm 1 2>/dev/null || true
nvidia-smi -ac 877,1215 2>/dev/null || true

# Iniciar LM Studio (remover nice devido a problemas de permissão)
echo "🚀 Iniciando LM Studio com otimizações..."
cd /squashfs-root
./lm-studio --no-sandbox --disable-gpu-sandbox --enable-gpu-rasterization --use-gl=desktop &
LMS_PID=$!

# Aguardar inicialização mais agressiva
sleep 20

# Função otimizada para auto-start
autostart_server_optimized() {
    echo "⚡ Iniciando servidor otimizado..."
    
    # Aguardar LM Studio estar pronto
    local attempts=0
    while [ $attempts -lt 30 ]; do
        if pgrep -f lm-studio > /dev/null; then
            echo "✅ LM Studio detectado, continuando..."
            break
        fi
        sleep 2
        ((attempts++))
    done
    
    # Verificar link simbólico
    if [ ! -f "/usr/local/bin/lms" ]; then
        ln -sf /squashfs-root/resources/app/.webpack/lms /usr/local/bin/lms
        chmod +x /usr/local/bin/lms
    fi
    
    # Verificar modelos disponíveis primeiro
    echo "📋 Listando modelos disponíveis..."
    /usr/local/bin/lms ls
    
    # Força detecção da GPU e aguarda estabilizar
    echo "🔍 Forçando detecção da GPU..."
    export CUDA_VISIBLE_DEVICES=0
    export NVIDIA_VISIBLE_DEVICES=0
    export LLAMA_CUDA=1
    export GGML_USE_CUDA=1
    export CUDA_LAUNCH_BLOCKING=0
    
    # Verificar se a GPU está disponível
    nvidia-smi > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo "✅ GPU detectada com sucesso!"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    else
        echo "❌ Erro: GPU não detectada!"
    fi
    sleep 10
    
    # Carregar modelo específico forçando GPU com parâmetros explícitos
    echo "⚡ Carregando modelo gpt-oss-20b com GPU forçada..."
    
    # Primeiro tentar descarregar qualquer modelo
    /usr/local/bin/lms unload --all --yes 2>/dev/null || true
    sleep 5
    
    # Forçar uso do backend CUDA
    export CUDA_VISIBLE_DEVICES=0
    export NVIDIA_VISIBLE_DEVICES=0
    export LD_LIBRARY_PATH="/root/.cache/lm-studio/extensions/backends/vendor/linux-llama-cuda-vendor-v1:$LD_LIBRARY_PATH"
    
    # Carregar modelo com configurações explícitas
    /usr/local/bin/lms load gpt-oss-20b --gpu 0.9 --context-length ${CONTEXT_LENGTH:-32768} --yes --verbose
    
    # Aguardar carregamento do modelo (mais tempo para GPU)
    sleep 60
    
    # Iniciar servidor com configurações corretas
    echo "📡 Iniciando servidor HTTP..."
    /usr/local/bin/lms server start --cors --port 1234 &
    SERVER_PID=$!
    
    # Monitorar carregamento
    echo "⏳ Monitorando carregamento do modelo..."
    local load_attempts=0
    while [ $load_attempts -lt 30 ]; do
        if /usr/local/bin/lms ps | grep -q "gpt-oss-20b"; then
            echo "✅ Modelo carregado com sucesso!"
            break
        fi
        sleep 10
        ((load_attempts++))
        echo "⏳ Aguardando carregamento... ($((load_attempts * 10))s)"
    done
    
    # Status final
    echo "📊 Status do sistema:"
    /usr/local/bin/lms server status 2>/dev/null || echo "Servidor iniciando..."
    /usr/local/bin/lms ps 2>/dev/null || echo "Modelos carregando..."
    
    echo "🎉 Sistema otimizado operacional!"
}
# Executar automação otimizada em background
autostart_server_optimized &

# VNC otimizado
mkdir -p /root/.vnc
x11vnc -storepasswd lmstudio123 /root/.vnc/passwd
chmod 600 /root/.vnc/passwd

# Iniciar VNC com configurações otimizadas
x11vnc -display :99 -forever -rfbauth /root/.vnc/passwd -quiet -listen 0.0.0.0 -xkb -noxrecord -noxfixes -noxdamage -rfbport 5900 &

# Aguardar VNC
sleep 3

# WebSocket otimizado para noVNC
websockify --web=/opt/noVNC --heartbeat=30 6080 localhost:5900 &

echo "🎯 Todos os serviços iniciados com otimizações!"
echo "📊 Recursos disponíveis:"
echo "   - GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits)"
echo "   - VRAM: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits) MB"
echo "   - CPU Threads: $OMP_NUM_THREADS"
echo "   - Context Length: ${CONTEXT_LENGTH:-32768}"

# Manter container ativo
wait