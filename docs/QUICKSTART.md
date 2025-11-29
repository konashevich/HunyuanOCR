# Quick Start Guide - HunyuanOCR Installation & MCP Server Setup

## System Check Results

✅ **Python:** 3.13.3 (Compatible - 3.12+ required)  
✅ **GPU:** NVIDIA GeForce RTX 3060 (6GB VRAM)  
⚠️ **VRAM:** 6GB (20GB recommended for vLLM, may need CPU fallback)  
✅ **CUDA:** 13.0  
✅ **WSL2:** Ubuntu installed (needed for vLLM on Windows)

## ⚠️ Important Notice

**GPU Memory Limitation:** Your RTX 3060 has 6GB VRAM, but HunyuanOCR with vLLM recommends 20GB. 

**Options:**
1. **Run in WSL2 with reduced memory utilization** (slower but works)
2. **Use CPU-only mode** (much slower)
3. **Use Transformers backend** instead of vLLM (no server mode, slower)

## Phase 1: Local Installation & Testing

### Step 1.1: Set Up WSL2 Environment

```powershell
# Start WSL2 Ubuntu
wsl

# Inside WSL2, navigate to project
cd /mnt/c/Users/akona/OneDrive/Dev/HunyuanOCR
```

### Step 1.2: Install Dependencies in WSL2

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.12 if not available
sudo apt install python3.12 python3.12-venv python3-pip -y

# Create virtual environment
python3.12 -m venv hunyuanocr-env
source hunyuanocr-env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install vLLM (nightly build)
pip install -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly

# Install project dependencies
pip install -r requirements.txt

# Install transformers (specific commit)
pip install git+https://github.com/huggingface/transformers@82a06db03535c49aa987719ed0746a76093b1ec4

# Install OpenAI client
pip install openai tqdm
```

### Step 1.3: Test with Transformers (CPU/GPU Fallback)

```bash
# Navigate to HF implementation
cd Hunyuan-OCR-master/Hunyuan-OCR-hf

# Run test script (downloads model on first run)
python run_hy_ocr.py
```

**Expected:** Model downloads (~6GB), then processes sample image and outputs markdown.

### Step 1.4: Try vLLM Server (May require memory optimization)

```bash
# Start vLLM with reduced GPU memory usage
vllm serve tencent/HunyuanOCR \
    --no-enable-prefix-caching \
    --mm-processor-cache-gb 0 \
    --gpu-memory-utilization 0.1 \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype half

# If OOM (Out of Memory) errors occur, try CPU mode:
# vllm serve tencent/HunyuanOCR \
#     --device cpu \
#     --host 0.0.0.0 \
#     --port 8000
```

**Note:** Keep this terminal open, vLLM server runs in foreground.

### Step 1.5: Test vLLM API

Open new WSL2 terminal:

```bash
cd /mnt/c/Users/akona/OneDrive/Dev/HunyuanOCR
source hunyuanocr-env/bin/activate
cd docs/mcp-server

# Install MCP server dependencies
pip install -r requirements.txt

# Test vLLM API
python test_vllm.py
```

---

## Phase 2: MCP Server Deployment

### Step 2.1: Start MCP Server

```bash
# In WSL2, with vLLM server running in another terminal
cd /mnt/c/Users/akona/OneDrive/Dev/HunyuanOCR/docs/mcp-server
source ../../hunyuanocr-env/bin/activate

# Start MCP server
python server.py
```

### Step 2.2: Configure Network Access

Find your Windows machine's local IP:

```powershell
# In Windows PowerShell
ipconfig | findstr IPv4
# Example output: 192.168.1.100
```

Allow firewall access:

```powershell
# In PowerShell as Administrator
New-NetFirewallRule -DisplayName "HunyuanOCR vLLM" -Direction Inbound -LocalPort 8000 -Protocol TCP -Action Allow
```

### Step 2.3: Test from Another Computer

On a different machine in your network:

```python
# test_remote_access.py
from openai import OpenAI
import base64

client = OpenAI(
    api_key="EMPTY",
    base_url="http://192.168.1.100:8000/v1",  # Replace with your IP
    timeout=3600
)

def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

response = client.chat.completions.create(
    model="tencent/HunyuanOCR",
    messages=[
        {"role": "system", "content": ""},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image('test.jpg')}"}},
                {"type": "text", "text": "Extract the text in the image."}
            ]
        }
    ],
    temperature=0.0
)

print(response.choices[0].message.content)
```

### Step 2.4: Configure MCP Client (Claude Desktop)

Edit `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "hunyuan-ocr": {
      "command": "wsl",
      "args": [
        "bash", "-c",
        "cd /mnt/c/Users/akona/OneDrive/Dev/HunyuanOCR/docs/mcp-server && source ../../hunyuanocr-env/bin/activate && python server.py"
      ],
      "env": {
        "VLLM_BASE_URL": "http://localhost:8000"
      }
    }
  }
}
```

---

## Troubleshooting

### Problem: "Out of Memory" with vLLM

**Solution 1:** Reduce GPU memory utilization
```bash
vllm serve tencent/HunyuanOCR --gpu-memory-utilization 0.05
```

**Solution 2:** Use CPU mode
```bash
vllm serve tencent/HunyuanOCR --device cpu
```

**Solution 3:** Use Transformers instead (no vLLM)
- Skip vLLM entirely
- Use direct Python scripts from `Hunyuan-OCR-master/Hunyuan-OCR-hf/`
- MCP server will need modifications to call model directly

### Problem: vLLM won't install on Windows

**Solution:** Must use WSL2 or Docker. vLLM requires Linux environment.

### Problem: Model download fails

**Solution 1:** Manual download
```bash
huggingface-cli download tencent/HunyuanOCR
```

**Solution 2:** Use mirror
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### Problem: Network clients can't connect

**Checklist:**
1. ✅ Firewall rules added?
2. ✅ vLLM server bound to 0.0.0.0 (not 127.0.0.1)?
3. ✅ Correct IP address used?
4. ✅ Test with: `telnet <ip> 8000`

---

## Performance Expectations (RTX 3060 6GB)

| Task | Expected Performance |
|------|---------------------|
| Text Extraction (small image) | 5-10s (GPU) / 30-60s (CPU) |
| Document Parsing | 15-30s (GPU) / 2-5min (CPU) |
| Batch Processing | Not recommended (memory limited) |

**Recommendation:** For production use with faster performance, consider:
- Upgrading to RTX 4090 (24GB) or similar
- Using cloud GPU (AWS/GCP/Azure)
- Running on dedicated server with high VRAM

---

## Quick Commands Reference

### Start Everything (WSL2)

```bash
# Terminal 1: vLLM Server
cd /mnt/c/Users/akona/OneDrive/Dev/HunyuanOCR
source hunyuanocr-env/bin/activate
vllm serve tencent/HunyuanOCR --host 0.0.0.0 --port 8000 --gpu-memory-utilization 0.1

# Terminal 2: Test
source hunyuanocr-env/bin/activate
cd docs/mcp-server
python test_vllm.py

# Terminal 3: MCP Server (optional)
source hunyuanocr-env/bin/activate
cd docs/mcp-server
python server.py
```

### Stop Everything

```bash
# Ctrl+C in each terminal
# Or:
pkill -f "vllm serve"
pkill -f "python server.py"
```

---

## Next Steps

1. ✅ Follow Phase 1 steps to install and test locally
2. ✅ Verify vLLM API works with test script
3. ✅ Start MCP server if API test succeeds
4. 📝 Document your specific performance metrics
5. 🔧 Optimize settings for your 6GB GPU
6. 🌐 Configure network access for remote clients
7. 🤖 Integrate with AI agents via MCP

---

## Alternative: Docker Approach

If WSL2 is problematic, use Docker:

```dockerfile
# Dockerfile
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3.12 python3-pip
WORKDIR /app
COPY requirements.txt .
RUN pip install -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly
RUN pip install -r requirements.txt
COPY . .

CMD ["vllm", "serve", "tencent/HunyuanOCR", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# Build and run
docker build -t hunyuan-ocr .
docker run --gpus all -p 8000:8000 hunyuan-ocr
```

---

**Document Status:** Ready for Implementation  
**Last Updated:** November 29, 2025  
**Your Hardware:** RTX 3060 6GB - Memory optimization required
