# HunyuanOCR MCP Server

Network-accessible OCR service powered by HunyuanOCR via Model Context Protocol (MCP).

## Overview

This MCP server wraps the HunyuanOCR vLLM API to provide OCR capabilities across your local network. AI agents on any computer can connect to this server to perform OCR tasks.

## Architecture

```
┌─────────────────┐         ┌──────────────────┐         ┌─────────────────┐
│  AI Agent on    │         │   MCP Server     │         │  vLLM Server    │
│  Client Machine │ ◄─────► │  (stdio)         │ ◄─────► │  (Port 8000)    │
│                 │   MCP   │  + OCR Tools     │   HTTP  │  HunyuanOCR     │
└─────────────────┘         └──────────────────┘         └─────────────────┘
      LAN                          Host Machine                GPU Backend
```

## Prerequisites

1. **vLLM Server Running:** Must have HunyuanOCR vLLM server running on port 8000
2. **Python 3.12+** with asyncio support
3. **Network Access:** Firewall configured for port access

## Installation

```bash
# 1. Install dependencies
cd docs/mcp-server
pip install -r requirements.txt

# 2. Configure environment (optional)
cp .env.example .env
# Edit .env with your settings

# 3. Verify vLLM server is running
python test_vllm.py
```

## Configuration

Create `.env` file in this directory:

```bash
# vLLM Backend URL
VLLM_BASE_URL=http://localhost:8000

# Security (optional)
REQUIRE_API_KEY=false
API_KEY=your-secret-key

# Limits
MAX_IMAGE_SIZE_MB=10
REQUEST_TIMEOUT_SECONDS=300
```

## Usage

### Starting the Server

```bash
python server.py
```

The server runs in stdio mode for MCP protocol communication.

### Available Tools

1. **ocr_extract_text** - Simple text extraction
   ```json
   {
     "image_path": "/path/to/image.jpg",
     "language": "english"
   }
   ```

2. **ocr_spot_text** - Text with bounding box coordinates
   ```json
   {
     "image_path": "/path/to/image.jpg",
     "language": "english"
   }
   ```

3. **ocr_parse_document** - Full document parsing (markdown + HTML tables + LaTeX)
   ```json
   {
     "image_path": "/path/to/document.jpg",
     "language": "english"
   }
   ```

4. **ocr_parse_table** - Table extraction to HTML
   ```json
   {
     "image_path": "/path/to/table.jpg",
     "language": "english"
   }
   ```

5. **ocr_parse_formula** - Mathematical formula to LaTeX
   ```json
   {
     "image_path": "/path/to/formula.jpg",
     "language": "english"
   }
   ```

6. **ocr_extract_fields** - Structured field extraction (JSON)
   ```json
   {
     "image_path": "/path/to/receipt.jpg",
     "fields": ["total_amount", "date", "merchant_name"],
     "language": "english"
   }
   ```

7. **ocr_translate_image** - Image text translation
   ```json
   {
     "image_path": "/path/to/foreign_doc.jpg",
     "target_language": "english"
   }
   ```

8. **ocr_extract_subtitles** - Video subtitle extraction
   ```json
   {
     "image_path": "/path/to/video_frame.jpg",
     "language": "english"
   }
   ```

9. **health_check** - Check vLLM server status
   ```json
   {}
   ```

## Client Configuration

### Claude Desktop

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "hunyuan-ocr": {
      "command": "python",
      "args": ["C:/path/to/HunyuanOCR/docs/mcp-server/server.py"],
      "env": {
        "VLLM_BASE_URL": "http://192.168.1.100:8000"
      }
    }
  }
}
```

Replace `192.168.1.100` with your server's IP address.

### Python MCP Client

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def use_ocr():
    server_params = StdioServerParameters(
        command="python",
        args=["path/to/server.py"],
        env={"VLLM_BASE_URL": "http://192.168.1.100:8000"}
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # Extract text from image
            result = await session.call_tool(
                "ocr_extract_text",
                {"image_path": "/path/to/image.jpg"}
            )
            print(result.content[0].text)
```

## Testing

### Test vLLM API
```bash
python test_vllm.py
```

### Test MCP Server
```bash
# Terminal 1: Start MCP server
python server.py

# Terminal 2: Run tests
python test_mcp_client.py
```

## Network Access

### Allow Firewall Access (Windows PowerShell as Admin)

```powershell
# Allow vLLM server
New-NetFirewallRule -DisplayName "HunyuanOCR vLLM" -Direction Inbound -LocalPort 8000 -Protocol TCP -Action Allow

# Find your local IP
ipconfig | findstr IPv4
```

### Allow Firewall Access (Linux)

```bash
sudo ufw allow 8000/tcp
sudo ufw reload

# Find your local IP
ip addr show | grep inet
```

## Troubleshooting

### vLLM Server Not Responding

```bash
# Check if vLLM server is running
curl http://localhost:8000/health

# Start vLLM server if needed
vllm serve tencent/HunyuanOCR --host 0.0.0.0 --port 8000
```

### Connection Timeout

- Check firewall rules
- Verify vLLM server is bound to 0.0.0.0 (not just 127.0.0.1)
- Test with: `telnet <server-ip> 8000`

### Image Not Found

- Use absolute paths for images
- Check file permissions
- Supported formats: JPG, PNG, GIF, BMP, WEBP

### Slow Performance

- GPU recommended (20GB VRAM)
- Reduce `--gpu-memory-utilization` if OOM errors
- Consider CPU-only mode (much slower): remove CUDA requirements

## Performance

### Expected Response Times (RTX 4090)

| Task | Image Size | Typical Time |
|------|-----------|--------------|
| Text Extraction | 1024x768 | 2-3s |
| Document Parsing | 2048x1536 | 5-8s |
| Field Extraction | 800x600 | 1-2s |
| Translation | 1024x768 | 3-5s |

## Security Notes

⚠️ **Important for Production:**

1. This server runs locally and trusts all requests
2. Add authentication for production use
3. Validate image paths to prevent directory traversal
4. Use HTTPS/TLS for remote access
5. Consider VPN instead of direct internet exposure
6. Monitor logs for suspicious activity

## File Structure

```
docs/mcp-server/
├── server.py              # Main MCP server
├── ocr_tools.py           # OCR tool implementations
├── vllm_client.py         # vLLM API client
├── config.py              # Configuration
├── requirements.txt       # Dependencies
├── test_vllm.py          # vLLM API tests
├── test_mcp_client.py    # MCP client tests
└── README.md             # This file
```

## License

This MCP server follows the same license as HunyuanOCR (Tencent Hunyuan Community License Agreement).

## Support

For issues:
1. Check logs: server.py outputs to console
2. Verify vLLM server is healthy: `python test_vllm.py`
3. Check main documentation: `../installation-plan.md`

## Next Steps

1. ✅ Install and test vLLM server (see main plan)
2. ✅ Start MCP server: `python server.py`
3. Configure Claude Desktop or other MCP clients
4. Test OCR tools from client applications
5. Deploy across local network
6. Add monitoring and logging as needed
