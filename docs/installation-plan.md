# HunyuanOCR Installation & MCP Server Deployment Plan

**Created:** November 29, 2025  
**Machine:** Windows with CUDA-capable GPU  
**Goal:** Local OCR deployment + Network-accessible MCP server for AI agents

---

## 📋 Executive Summary

HunyuanOCR is a state-of-the-art 1B parameter OCR model from Tencent that achieves SOTA performance across multiple benchmarks. This plan outlines a two-phase approach:

- **Phase 1:** Install and test the model locally to validate performance and capabilities
- **Phase 2:** Deploy as an MCP (Model Context Protocol) server accessible across the local network

### Key Findings from Codebase Analysis

✅ **No Built-in UI** - Command-line and API-based inference only  
✅ **vLLM Server Support** - OpenAI-compatible REST API (recommended approach)  
✅ **Transformers Support** - Direct Python inference (simpler but slower)  
❌ **Linux Requirement** - Official docs specify Linux OS (WSL2/Docker needed for Windows)  
✅ **MCP Server** - Not included; custom implementation required

---

## 🎯 Phase 1: Local Installation & Testing

### Objectives
1. Set up proper Python environment on Windows
2. Install HunyuanOCR model and dependencies
3. Test OCR capabilities on sample images
4. Benchmark GPU performance
5. Deploy vLLM server and verify API functionality

### Phase 1.1: System Requirements Check

**Required:**
- 🖥️ **OS:** Linux (or Windows with WSL2/Docker)
- 🐍 **Python:** 3.12+ 
- ⚡ **CUDA:** 12.9
- 🔥 **PyTorch:** 2.7.1
- 🎮 **GPU:** NVIDIA GPU with CUDA support
- 🧠 **GPU Memory:** 20GB (for vLLM)
- 💾 **Disk Space:** 6GB for model weights

**Action Items:**
```powershell
# Check Python version
python --version

# Check CUDA version
nvidia-smi

# Check GPU memory
nvidia-smi --query-gpu=memory.total --format=csv

# Check WSL2 (if on Windows)
wsl --list --verbose
```

**Decision Point:** 
- ✅ If running Linux or have WSL2 → Proceed with native installation
- ⚠️ If Windows without WSL2 → Set up WSL2 or use Docker
- ❌ If GPU < 20GB → Consider CPU-only mode (slower) or upgrade hardware

---

### Phase 1.2: Environment Setup

**Option A: Native Linux/WSL2**
```bash
# Create virtual environment
python3.12 -m venv hunyuanocr-env
source hunyuanocr-env/bin/activate  # Linux/WSL2

# Install vLLM (nightly build recommended)
pip install -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly

# Install project dependencies
cd /path/to/HunyuanOCR
pip install -r requirements.txt

# Install transformers (specific commit for compatibility)
pip install git+https://github.com/huggingface/transformers@82a06db03535c49aa987719ed0746a76093b1ec4

# Install additional dependencies for API client
pip install openai tqdm pillow
```

**Option B: Docker (Windows/Cross-platform)**
```dockerfile
# Create Dockerfile in project root
FROM nvidia/cuda:12.9.0-runtime-ubuntu24.04

RUN apt-get update && apt-get install -y python3.12 python3-pip
WORKDIR /app
COPY requirements.txt .
RUN pip install -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly
RUN pip install -r requirements.txt
COPY . .

CMD ["bash"]
```

**Option C: Windows PowerShell (Experimental)**
```powershell
# Create virtual environment
python -m venv hunyuanocr-env
.\hunyuanocr-env\Scripts\Activate.ps1

# Note: vLLM may require WSL2 backend
pip install -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly
pip install -r requirements.txt
```

---

### Phase 1.3: Model Download & Testing

**Download Model (automatic on first run):**
```python
# This will download ~6GB from HuggingFace
from transformers import AutoProcessor
processor = AutoProcessor.from_pretrained("tencent/HunyuanOCR", use_fast=False)
```

**Test 1: Basic Text Extraction (Transformers)**
```bash
cd Hunyuan-OCR-master/Hunyuan-OCR-hf
python run_hy_ocr.py
```

Expected output: Parsed markdown from test image

**Test 2: Text Spotting with Coordinates**
```python
# Create test_spotting.py
from transformers import AutoProcessor, HunYuanVLForConditionalGeneration
from PIL import Image
import torch

model_path = "tencent/HunyuanOCR"
processor = AutoProcessor.from_pretrained(model_path, use_fast=False)
model = HunYuanVLForConditionalGeneration.from_pretrained(
    model_path,
    attn_implementation="eager",
    dtype=torch.bfloat16,
    device_map="auto"
)

img_path = "../assets/vis_document_23.jpg"
image = Image.open(img_path)

messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": img_path},
        {"type": "text", "text": "Detect and recognize text in the image, and output the text coordinates in a formatted manner."}
    ]
}]

texts = [processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)]
inputs = processor(text=texts, images=image, padding=True, return_tensors="pt")

with torch.no_grad():
    inputs = inputs.to(model.device)
    output = model.generate(**inputs, max_new_tokens=16384, do_sample=False)
    
result = processor.batch_decode(output, skip_special_tokens=True)[0]
print(result)
```

**Test 3: Document Parsing**
```python
# Test on complex document
messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": "../assets/vis_parsing_table.png"},
        {"type": "text", "text": "Extract all information from the main body of the document image and represent it in markdown format, ignoring headers and footers. Tables should be expressed in HTML format, formulas in the document should be represented using LaTeX format, and the parsing should be organized according to the reading order."}
    ]
}]
```

**Test 4: Information Extraction**
```python
# Test on receipt/card
messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": "../assets/ie_parallel.jpg"},
        {"type": "text", "text": "Extract the content of the fields: ['单价', '上车时间', '发票号码', '省前缀', '总金额', '发票代码', '下车时间', '里程数'] from the image and return it in JSON format."}
    ]
}]
```

---

### Phase 1.4: Deploy vLLM Server (Recommended)

**Start vLLM Server:**
```bash
# Recommended: Start in background
vllm serve tencent/HunyuanOCR \
    --no-enable-prefix-caching \
    --mm-processor-cache-gb 0 \
    --gpu-memory-utilization 0.2 \
    --host 0.0.0.0 \
    --port 8000

# Server will be available at http://localhost:8000
```

**Test OpenAI-Compatible API:**
```python
# test_vllm_api.py
import base64
from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1",
    timeout=3600
)

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

# Test request
img_path = "../assets/spotting1_cropped.png"
response = client.chat.completions.create(
    model="tencent/HunyuanOCR",
    messages=[
        {"role": "system", "content": ""},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encode_image(img_path)}"}
                },
                {"type": "text", "text": "Extract the text in the image."}
            ]
        }
    ],
    temperature=0.0,
    top_p=0.95
)

print(response.choices[0].message.content)
```

**Performance Benchmarking:**
```python
# benchmark_ocr.py
import time
import json

test_images = [
    "../assets/vis_document_23.jpg",
    "../assets/vis_parsing_table.png",
    "../assets/ie_parallel.jpg"
]

results = []
for img in test_images:
    start = time.time()
    response = client.chat.completions.create(...)  # Same as above
    elapsed = time.time() - start
    
    results.append({
        "image": img,
        "time_seconds": elapsed,
        "output_length": len(response.choices[0].message.content)
    })

print(json.dumps(results, indent=2))
```

---

## 🌐 Phase 2: MCP Server Deployment

### Objectives
1. Create MCP server wrapper around vLLM API
2. Expose OCR capabilities via MCP protocol
3. Configure network accessibility
4. Test from remote clients on local network
5. Document usage patterns for AI agents

### Phase 2.1: MCP Server Architecture

**Design Overview:**
```
┌─────────────────┐         ┌──────────────────┐         ┌─────────────────┐
│  AI Agent on    │         │   MCP Server     │         │  vLLM Server    │
│  Client Machine │ ◄─────► │  (Port 3000)     │ ◄─────► │  (Port 8000)    │
│                 │   MCP   │  + OCR Tools     │   HTTP  │  HunyuanOCR     │
└─────────────────┘         └──────────────────┘         └─────────────────┘
      LAN                          Host Machine                GPU Backend
```

**MCP Tools to Implement:**
1. `ocr_extract_text` - Simple text extraction
2. `ocr_spot_text` - Text detection with coordinates
3. `ocr_parse_document` - Full document parsing (markdown/HTML/LaTeX)
4. `ocr_extract_fields` - Structured information extraction (JSON)
5. `ocr_translate_image` - Image translation
6. `ocr_extract_subtitles` - Video subtitle extraction

---

### Phase 2.2: MCP Server Implementation

**Project Structure:**
```
docs/mcp-server/
├── server.py              # Main MCP server
├── ocr_tools.py           # OCR tool implementations
├── vllm_client.py         # vLLM API client wrapper
├── config.py              # Configuration
├── requirements.txt       # MCP dependencies
└── README.md              # Usage guide
```

**Install MCP SDK:**
```bash
pip install mcp anthropic-mcp
```

**Core Implementation (server.py):**
```python
#!/usr/bin/env python3
"""
HunyuanOCR MCP Server
Exposes OCR capabilities via Model Context Protocol
"""
import asyncio
import logging
from mcp.server import Server
from mcp.server.stdio import stdio_server
from ocr_tools import (
    extract_text,
    spot_text,
    parse_document,
    extract_fields,
    translate_image,
    extract_subtitles
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create MCP server instance
app = Server("hunyuan-ocr-server")

@app.list_tools()
async def list_tools():
    return [
        {
            "name": "ocr_extract_text",
            "description": "Extract all text from an image",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "image_path": {"type": "string", "description": "Path to image file or base64 data URI"},
                },
                "required": ["image_path"]
            }
        },
        {
            "name": "ocr_spot_text",
            "description": "Detect and recognize text with bounding box coordinates",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "image_path": {"type": "string"},
                },
                "required": ["image_path"]
            }
        },
        {
            "name": "ocr_parse_document",
            "description": "Parse complex document with markdown, HTML tables, LaTeX formulas",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "image_path": {"type": "string"},
                    "language": {"type": "string", "enum": ["english", "chinese"], "default": "english"}
                },
                "required": ["image_path"]
            }
        },
        {
            "name": "ocr_extract_fields",
            "description": "Extract specific fields from cards, receipts, forms as JSON",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "image_path": {"type": "string"},
                    "fields": {"type": "array", "items": {"type": "string"}, "description": "List of field names to extract"}
                },
                "required": ["image_path", "fields"]
            }
        },
        {
            "name": "ocr_translate_image",
            "description": "Extract text from image and translate to target language",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "image_path": {"type": "string"},
                    "target_language": {"type": "string", "enum": ["english", "chinese"], "default": "english"}
                },
                "required": ["image_path"]
            }
        },
        {
            "name": "ocr_extract_subtitles",
            "description": "Extract subtitles from video frame or screenshot",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "image_path": {"type": "string"}
                },
                "required": ["image_path"]
            }
        }
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict):
    try:
        if name == "ocr_extract_text":
            result = await extract_text(arguments["image_path"])
        elif name == "ocr_spot_text":
            result = await spot_text(arguments["image_path"])
        elif name == "ocr_parse_document":
            result = await parse_document(arguments["image_path"], arguments.get("language", "english"))
        elif name == "ocr_extract_fields":
            result = await extract_fields(arguments["image_path"], arguments["fields"])
        elif name == "ocr_translate_image":
            result = await translate_image(arguments["image_path"], arguments.get("target_language", "english"))
        elif name == "ocr_extract_subtitles":
            result = await extract_subtitles(arguments["image_path"])
        else:
            return {"error": f"Unknown tool: {name}"}
        
        return {"content": [{"type": "text", "text": result}]}
    except Exception as e:
        logger.error(f"Error in {name}: {e}")
        return {"error": str(e)}

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())
```

**OCR Tools Implementation (ocr_tools.py):**
```python
"""
OCR tool implementations using vLLM backend
"""
import base64
from vllm_client import VLLMClient

client = VLLMClient(base_url="http://localhost:8000")

PROMPTS = {
    "extract_text": {
        "english": "Extract the text in the image.",
        "chinese": "提取图中的文字。"
    },
    "spot_text": {
        "english": "Detect and recognize text in the image, and output the text coordinates in a formatted manner.",
        "chinese": "检测并识别图片中的文字，将文本坐标格式化输出。"
    },
    "parse_document": {
        "english": "Extract all information from the main body of the document image and represent it in markdown format, ignoring headers and footers. Tables should be expressed in HTML format, formulas in the document should be represented using LaTeX format, and the parsing should be organized according to the reading order.",
        "chinese": "提取文档图片中正文的所有信息用 markdown 格式表示，其中页眉、页脚部分忽略，表格用 html 格式表达，文档中公式用 latex 格式表示，按照阅读顺序组织进行解析。"
    },
    "translate": {
        "english": "First extract the text, then translate the text content into English. If it is a document, ignore the header and footer. Formulas should be represented in LaTeX format, and tables should be represented in HTML format.",
        "chinese": "先提取文字，再将文字内容翻译为中文。若是文档，则其中页眉、页脚忽略。公式用latex格式表示，表格用html格式表示。"
    },
    "subtitles": {
        "english": "Extract the subtitles from the image.",
        "chinese": "提取图片中的字幕。"
    }
}

async def extract_text(image_path: str, language: str = "english") -> str:
    """Extract all text from image"""
    prompt = PROMPTS["extract_text"][language]
    return await client.process_image(image_path, prompt)

async def spot_text(image_path: str, language: str = "english") -> str:
    """Detect text with coordinates"""
    prompt = PROMPTS["spot_text"][language]
    return await client.process_image(image_path, prompt)

async def parse_document(image_path: str, language: str = "english") -> str:
    """Parse complex document"""
    prompt = PROMPTS["parse_document"][language]
    return await client.process_image(image_path, prompt)

async def extract_fields(image_path: str, fields: list, language: str = "english") -> str:
    """Extract specific fields as JSON"""
    fields_str = str(fields)
    if language == "chinese":
        prompt = f"提取图片中的: {fields_str} 的字段内容，并按照 JSON 格式返回。"
    else:
        prompt = f"Extract the content of the fields: {fields_str} from the image and return it in JSON format."
    return await client.process_image(image_path, prompt)

async def translate_image(image_path: str, target_language: str = "english") -> str:
    """Translate image text"""
    prompt = PROMPTS["translate"][target_language]
    return await client.process_image(image_path, prompt)

async def extract_subtitles(image_path: str, language: str = "english") -> str:
    """Extract video subtitles"""
    prompt = PROMPTS["subtitles"][language]
    return await client.process_image(image_path, prompt)
```

**vLLM Client Wrapper (vllm_client.py):**
```python
"""
Async client for vLLM API
"""
import base64
import aiohttp
from pathlib import Path

class VLLMClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self.api_endpoint = f"{self.base_url}/v1/chat/completions"
    
    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64"""
        if image_path.startswith("data:"):
            return image_path  # Already base64 data URI
        
        with open(image_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode('utf-8')
        
        # Detect image format
        suffix = Path(image_path).suffix.lower()
        mime_type = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".bmp": "image/bmp"
        }.get(suffix, "image/jpeg")
        
        return f"data:{mime_type};base64,{encoded}"
    
    async def process_image(self, image_path: str, prompt: str, temperature: float = 0.0) -> str:
        """Send OCR request to vLLM server"""
        image_data = self._encode_image(image_path)
        
        payload = {
            "model": "tencent/HunyuanOCR",
            "messages": [
                {"role": "system", "content": ""},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_data}},
                        {"type": "text", "text": prompt}
                    ]
                }
            ],
            "temperature": temperature,
            "top_p": 0.95,
            "stream": False,
            "extra_body": {
                "top_k": 1,
                "repetition_penalty": 1.0
            }
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.api_endpoint,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=3600)
            ) as response:
                result = await response.json()
                return result["choices"][0]["message"]["content"]
```

**Configuration (config.py):**
```python
"""
MCP Server Configuration
"""
import os

# vLLM Backend
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000")

# MCP Server
MCP_HOST = os.getenv("MCP_HOST", "0.0.0.0")  # Bind to all interfaces for network access
MCP_PORT = int(os.getenv("MCP_PORT", "3000"))

# Security (optional)
REQUIRE_API_KEY = os.getenv("REQUIRE_API_KEY", "false").lower() == "true"
API_KEY = os.getenv("API_KEY", "")

# Limits
MAX_IMAGE_SIZE_MB = int(os.getenv("MAX_IMAGE_SIZE_MB", "10"))
REQUEST_TIMEOUT_SECONDS = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "300"))
```

---

### Phase 2.3: Network Configuration

**Windows Firewall Rules (PowerShell as Admin):**
```powershell
# Allow vLLM server (port 8000)
New-NetFirewallRule -DisplayName "HunyuanOCR vLLM Server" -Direction Inbound -LocalPort 8000 -Protocol TCP -Action Allow

# Allow MCP server (port 3000)
New-NetFirewallRule -DisplayName "HunyuanOCR MCP Server" -Direction Inbound -LocalPort 3000 -Protocol TCP -Action Allow
```

**Linux Firewall (ufw):**
```bash
sudo ufw allow 8000/tcp
sudo ufw allow 3000/tcp
sudo ufw reload
```

**Find Local IP Address:**
```powershell
# Windows
ipconfig | findstr IPv4

# Linux
ip addr show | grep inet
```

---

### Phase 2.4: Testing from Remote Clients

**Test 1: Direct vLLM API Access**
```python
# From another machine on LAN
from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://192.168.1.100:8000/v1",  # Replace with host IP
    timeout=3600
)

# Test request
response = client.chat.completions.create(...)
```

**Test 2: MCP Client Connection**
```python
# mcp_client_test.py
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def test_mcp_ocr():
    server_params = StdioServerParameters(
        command="python",
        args=["/path/to/server.py"],
        env={"VLLM_BASE_URL": "http://192.168.1.100:8000"}
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # List available tools
            tools = await session.list_tools()
            print("Available tools:", [t.name for t in tools])
            
            # Call OCR tool
            result = await session.call_tool(
                "ocr_extract_text",
                {"image_path": "/path/to/test.jpg"}
            )
            print("OCR Result:", result)

asyncio.run(test_mcp_ocr())
```

---

## 📚 Usage Examples

### Example 1: AI Agent Using MCP for OCR

```python
# AI agent code (runs on any machine in LAN)
from mcp import ClientSession

async def analyze_receipt(receipt_image_path: str):
    """Extract fields from receipt using OCR MCP server"""
    result = await session.call_tool(
        "ocr_extract_fields",
        {
            "image_path": receipt_image_path,
            "fields": ["total_amount", "date", "merchant_name", "items"]
        }
    )
    return json.loads(result.content[0].text)
```

### Example 2: Document Processing Pipeline

```python
async def process_document(doc_image: str):
    """Full document processing workflow"""
    # Step 1: Parse document structure
    parsed = await session.call_tool(
        "ocr_parse_document",
        {"image_path": doc_image, "language": "english"}
    )
    
    # Step 2: Extract specific fields
    fields = await session.call_tool(
        "ocr_extract_fields",
        {"image_path": doc_image, "fields": ["title", "author", "date"]}
    )
    
    return {
        "full_content": parsed,
        "metadata": json.loads(fields)
    }
```

---

## 🔧 Troubleshooting

### Common Issues

**Issue 1: vLLM won't start on Windows**
- **Solution:** Use WSL2 or Docker. vLLM has limited Windows support.
- **Command:** `wsl --install` then run in WSL2 Ubuntu

**Issue 2: GPU memory exhausted (OOM)**
- **Solution:** Reduce `--gpu-memory-utilization` from 0.2 to 0.15
- **Alternative:** Use CPU-only mode (much slower)

**Issue 3: Model download fails**
- **Solution:** Check HuggingFace access, use mirror or manual download
- **Command:** `huggingface-cli download tencent/HunyuanOCR`

**Issue 4: Network clients can't connect**
- **Solution:** Check firewall rules, verify host IP, test with `telnet`
- **Command:** `telnet 192.168.1.100 8000`

**Issue 5: Slow inference on CPU**
- **Solution:** GPU required for reasonable performance. Consider cloud GPU.

---

## 📊 Performance Benchmarks

### Expected Performance (RTX 4090, 20GB VRAM)

| Task | Image Size | Avg Time | Tokens/sec |
|------|-----------|----------|------------|
| Text Extraction | 1024x768 | 2-3s | 150-200 |
| Document Parsing | 2048x1536 | 5-8s | 100-150 |
| Field Extraction | 800x600 | 1-2s | 200-250 |

### Scaling Considerations

- **Single GPU:** ~10-20 concurrent requests
- **Multi-GPU:** Use vLLM tensor parallelism
- **Load Balancing:** Deploy multiple instances behind nginx

---

## 🔐 Security Considerations

### Production Deployment Recommendations

1. **API Authentication:** Add API key validation in MCP server
2. **Rate Limiting:** Implement request throttling
3. **Input Validation:** Sanitize file paths, check image sizes
4. **Network Isolation:** Use VPN or SSH tunnel instead of direct LAN exposure
5. **HTTPS/TLS:** Use reverse proxy (nginx) with SSL certificates
6. **Monitoring:** Log all requests, set up alerts for errors

---

## 📈 Next Steps After Phase 2

### Optional Enhancements

1. **Web UI:** Add Gradio/Streamlit interface for manual testing
2. **Batch Processing:** Queue system for bulk OCR jobs
3. **Result Caching:** Redis cache for duplicate image requests
4. **Multi-Language:** Extend beyond English/Chinese
5. **Cloud Deployment:** AWS/Azure/GCP for remote access
6. **API Gateway:** Kong/Tyk for enterprise features
7. **Monitoring:** Prometheus + Grafana dashboards

---

## 📝 Appendix: Quick Reference

### Start Services

```bash
# Terminal 1: Start vLLM server
vllm serve tencent/HunyuanOCR \
    --no-enable-prefix-caching \
    --mm-processor-cache-gb 0 \
    --gpu-memory-utilization 0.2 \
    --host 0.0.0.0 \
    --port 8000

# Terminal 2: Start MCP server
cd docs/mcp-server
python server.py
```

### Test Commands

```bash
# Test vLLM health
curl http://localhost:8000/health

# Test vLLM models
curl http://localhost:8000/v1/models

# Test MCP server (requires MCP client)
python mcp_client_test.py
```

### Environment Variables

```bash
export VLLM_BASE_URL="http://localhost:8000"
export MCP_HOST="0.0.0.0"
export MCP_PORT="3000"
export CUDA_VISIBLE_DEVICES="0"  # Select GPU
```

---

## ✅ Checklist

### Phase 1 Completion Criteria
- [ ] Python 3.12+ installed
- [ ] CUDA 12.9 verified
- [ ] vLLM installed and working
- [ ] HunyuanOCR model downloaded
- [ ] Sample images tested successfully
- [ ] vLLM server responding to API requests
- [ ] Performance benchmarks recorded

### Phase 2 Completion Criteria
- [ ] MCP server code implemented
- [ ] All 6 OCR tools working
- [ ] Firewall rules configured
- [ ] Remote client can connect
- [ ] Documentation complete
- [ ] Test suite passing
- [ ] Production checklist reviewed

---

**Document Version:** 1.0  
**Last Updated:** November 29, 2025  
**Author:** AI Assistant + User  
**Status:** Ready for Implementation
