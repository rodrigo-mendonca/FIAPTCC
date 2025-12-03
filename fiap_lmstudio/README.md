# 🚀 LMChat Docker Setup

## 📋 Pré-requisitos

### 1. WSL2 (Windows Subsystem for Linux)

O WSL2 é necessário para executar Docker Desktop e acessar GPUs NVIDIA no Windows.

```powershell
# 1. Habilitar WSL e Virtual Machine Platform
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart

# 2. Reiniciar o computador

# 3. Definir WSL 2 como versão padrão
wsl --set-default-version 2

# 4. Instalar Ubuntu (ou outra distribuição)
wsl --install -d Ubuntu

# 5. Verificar instalação
wsl --list --verbose
```

### 2. Docker Desktop

Baixe e instale o [Docker Desktop](https://www.docker.com/products/docker-desktop/) configurado para usar WSL2:

1. **Instale o Docker Desktop**
2. **Configurações importantes:**
   - ✅ Use WSL 2 based engine
   - ✅ Enable integration with my default WSL distro
   - ✅ Enable integration with additional distros: Ubuntu

### 3. NVIDIA Container Toolkit (Para GPUs NVIDIA)

⚠️ **Obrigatório para usar aceleração por GPU NVIDIA**

Execute no **Ubuntu WSL2**:

```bash
# 1. Instalar CUDA Toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt update && sudo apt install -y cuda-toolkit-12-6

# 2. Instalar NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://nvidia.github.io/libnvidia-container/stable/deb amd64/" | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update && sudo apt install -y nvidia-container-toolkit

# 3. Configurar Docker para usar NVIDIA
sudo nvidia-ctk runtime configure --runtime=docker
```

### 4. Verificar Configuração

```powershell
# Testar acesso à GPU
docker run --rm --gpus all ubuntu:20.04 nvidia-smi
```

Se o comando acima mostrar informações da sua GPU NVIDIA, a configuração está correta! 🎉

---

## 🚀 Executando o LMChat

### **Opção 1: Build e Start em Etapas**

```powershell
# 1. Construir a imagem Docker
docker compose build

# 2. Iniciar os serviços
docker compose up -d
```

### **Opção 2: Build e Start em Um Comando**

```powershell
# Build e start em um comando
docker compose up --build -d
```

### **Opção 3: Logs em Tempo Real**

```powershell
# Para acompanhar os logs em tempo real
docker compose up --build
```

## 🌐 Acessando os Serviços

Após executar `docker compose up -d`, os seguintes serviços estarão disponíveis:

- **🤖 LM Studio**: http://localhost:1234
- **🖥️ VNC (Acesso remoto à interface)**: http://localhost:5900
  - Senha VNC: `31p0ybfj`

## 📊 Comandos Úteis

```powershell
# Verificar status dos containers
docker compose ps

# Ver logs em tempo real
docker compose logs -f

# Verificar GPU dentro do container
docker exec lmchat nvidia-smi

# Parar os serviços
docker compose down

# Rebuildar do zero
docker compose down && docker compose build --no-cache && docker compose up -d
```

## 🛠️ Troubleshooting

### ❌ Erro: "CDI device injection failed: unresolvable CDI devices nvidia.com/gpu=all"

**Problema**: NVIDIA Container Toolkit não está instalado ou configurado.

**Solução**:
1. Instale o NVIDIA Container Toolkit no Ubuntu WSL2 (veja seção de pré-requisitos)
2. Verifique se a GPU está acessível: `docker run --rm --gpus all ubuntu:20.04 nvidia-smi`

### ❌ Erro: "docker: command not found" no WSL

**Problema**: Docker Desktop não está integrado com WSL.

**Solução**:
1. Abra Docker Desktop → Settings → Resources → WSL Integration
2. Marque "Enable integration with my default WSL distro"
3. Marque "Ubuntu" (ou sua distribuição WSL)
4. Apply & Restart

### ❌ Container para logo após iniciar

**Problema**: Pode ser um problema com o AppImage ou dependências.

**Solução**:
```powershell
# Verificar logs detalhados
docker compose logs lmchat

# Acessar o container para debug
docker exec -it lmchat bash
```

### ❌ Porta 1234 ou 5900 já em uso

**Problema**: Outra aplicação está usando as portas.

**Solução**:
1. Pare o LM Studio desktop se estiver rodando
2. Ou modifique as portas no `docker-compose.yml`:
```yaml
ports:
  - "1235:1234"  # Muda porta local para 1235
  - "5901:5900"  # Muda porta VNC para 5901
```

### 💡 Verificar se GPU está sendo usada

```powershell
# Monitorar uso da GPU em tempo real
nvidia-smi -l 1

# Ver processos usando GPU
nvidia-smi pmon
```

## 📁 Estrutura do Projeto

```
lmstudio/
├── 📄 docker-compose.yml    # Configuração principal dos serviços
├── 📄 Dockerfile            # Imagem Docker personalizada
├── 📄 start_services.sh     # Script de inicialização dos serviços
├── 📄 keyboard              # Configuração de teclado
├── 📄 http-server-config.json # Configuração do servidor HTTP
├── 📁 AppImage/             # LM Studio AppImage
│   └── LM-Studio-0.3.23-3-x64.AppImage
└── 📄 README.md             # Este arquivo
```

## ⚙️ Configurações Importantes

### GPU e Recursos

O `docker-compose.yml` está configurado para:
- **🎯 Acesso completo à GPU NVIDIA**
- **💾 8GB de memória compartilhada** (ajuste conforme necessário)
- **📂 Volumes mapeados** para modelos e dados do LM Studio

### Volumes Mapeados

```yaml
volumes:
  - C:\Users\rodri\.lmstudio\models:/root/.cache/lm-studio/models
  - C:\Users\rodri\.lmstudio:/root/docs
```

**⚠️ Importante**: Ajuste os caminhos dos volumes para corresponder ao seu usuário do Windows.

### Variáveis de Ambiente

```yaml
environment:
  - CONTEXT_LENGTH=32768  # Tamanho máximo do contexto
  - DISPLAY=:99          # Display virtual para interface gráfica
```

## 🔧 Personalização

### Alterar Quantidade de VRAM

Modifique no `docker-compose.yml`:
```yaml
shm_size: '16gb'  # Aumente para mais VRAM
```

### Usar CPU ao invés de GPU

Remova ou comente a seção `deploy` no `docker-compose.yml`:
```yaml
# deploy:
#   resources:
#     reservations:
#       devices:
#         - driver: nvidia
#           count: all
#           capabilities: [gpu]
```

## 📞 Suporte

Se encontrar problemas:

1. **📋 Verifique os pré-requisitos** estão todos instalados
2. **🔍 Analise os logs**: `docker compose logs lmchat`
3. **🧪 Teste a GPU**: `docker run --rm --gpus all ubuntu:20.04 nvidia-smi`
4. **🔄 Rebuild**: `docker compose down && docker compose build --no-cache`

---

**🎉 Agora você tem o LM Studio rodando com aceleração GPU completa no Docker!**