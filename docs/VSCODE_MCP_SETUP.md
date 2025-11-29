# VS Code MCP Integration Guide

## Overview

This guide shows how to integrate HunyuanOCR with VS Code using the Model Context Protocol (MCP). The MCP server runs locally and provides OCR capabilities to GitHub Copilot and other AI assistants in VS Code.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  VS Code                                                │
│  ┌─────────────────┐         ┌──────────────────┐     │
│  │  GitHub Copilot │ ◄─────► │   MCP Extension  │     │
│  │  or AI Assistant│   MCP   │                  │     │
│  └─────────────────┘         └──────────────────┘     │
└──────────────────────────────────│──────────────────────┘
                                   │ stdio
                    ┌──────────────▼─────────────┐
                    │  MCP Server (Python)       │
                    │  docs/mcp-server/server.py │
                    └──────────────┬─────────────┘
                                   │ HTTP
                    ┌──────────────▼─────────────┐
                    │  vLLM Server (WSL2)        │
                    │  localhost:8000            │
                    │  HunyuanOCR Model          │
                    └────────────────────────────┘
```

## Prerequisites

1. ✅ VS Code installed
2. ✅ vLLM server running (see setup scripts)
3. ✅ Python 3.12+ in WSL2 (for vLLM) or native Windows (for MCP server)

## Installation Steps

### Step 1: Install MCP Extension in VS Code

```bash
# Search for and install the MCP extension in VS Code
# Extension ID: modelcontextprotocol.mcp
# Or install from VS Code marketplace
```

### Step 2: Configure MCP Server for VS Code

VS Code MCP configuration is stored in your workspace or user settings.

**Option A: Workspace Configuration (Recommended)**

Create `.vscode/settings.json` in your project:

```json
{
  "mcp.servers": {
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

**Option B: User Settings (Global)**

Open VS Code settings (Ctrl+,) and add to `settings.json`:

```json
{
  "mcp.servers": {
    "hunyuan-ocr": {
      "command": "python",
      "args": ["C:/Users/akona/OneDrive/Dev/HunyuanOCR/docs/mcp-server/server.py"],
      "env": {
        "VLLM_BASE_URL": "http://localhost:8000"
      }
    }
  }
}
```

**Note:** For Option B to work, you need Python with MCP dependencies installed on Windows (not just WSL2).

### Step 3: Install MCP Dependencies (if using Windows Python)

If running MCP server on Windows Python (Option B above):

```powershell
cd C:\Users\akona\OneDrive\Dev\HunyuanOCR\docs\mcp-server
pip install -r requirements.txt
```

### Step 4: Start vLLM Server

Before using MCP tools, ensure vLLM is running:

```cmd
# In Windows terminal
scripts\start-vllm.bat
```

Keep this terminal open while using MCP.

### Step 5: Verify MCP Server in VS Code

1. Open VS Code Command Palette (Ctrl+Shift+P)
2. Type "MCP: Show MCP Servers"
3. You should see "hunyuan-ocr" listed
4. Check status - should show "Connected" when vLLM is running

## Available MCP Tools

Once connected, these tools are available to GitHub Copilot and AI assistants:

### 1. ocr_extract_text
Extract plain text from images.

```json
{
  "image_path": "C:/path/to/image.jpg",
  "language": "english"
}
```

### 2. ocr_spot_text
Get text with bounding box coordinates.

```json
{
  "image_path": "C:/path/to/image.jpg",
  "language": "english"
}
```

### 3. ocr_parse_document
Parse complex documents with structure.

```json
{
  "image_path": "C:/path/to/document.jpg",
  "language": "english"
}
```

### 4. ocr_parse_table
Extract tables as HTML.

```json
{
  "image_path": "C:/path/to/table.jpg",
  "language": "english"
}
```

### 5. ocr_parse_formula
Extract mathematical formulas as LaTeX.

```json
{
  "image_path": "C:/path/to/formula.jpg",
  "language": "english"
}
```

### 6. ocr_extract_fields
Extract specific fields from forms/receipts.

```json
{
  "image_path": "C:/path/to/receipt.jpg",
  "fields": ["total_amount", "date", "merchant_name"],
  "language": "english"
}
```

### 7. ocr_translate_image
Translate text in images.

```json
{
  "image_path": "C:/path/to/foreign_doc.jpg",
  "target_language": "english"
}
```

### 8. ocr_extract_subtitles
Extract subtitles from video frames.

```json
{
  "image_path": "C:/path/to/video_frame.jpg",
  "language": "english"
}
```

### 9. health_check
Check if vLLM server is running.

```json
{}
```

## Usage Examples in VS Code

### Example 1: Extract Text with GitHub Copilot

In VS Code chat:
```
@workspace Use the ocr_extract_text tool to extract text from C:/Users/akona/Documents/receipt.jpg
```

Copilot will call the MCP tool and return the extracted text.

### Example 2: Parse Document Structure

```
@workspace Parse the document at C:/Users/akona/Documents/paper.pdf page 1 screenshot and format it as markdown with HTML tables
```

### Example 3: Extract Receipt Fields

```
@workspace Extract the total amount, date, and merchant name from the receipt at C:/Users/akona/Documents/receipt.jpg
```

### Example 4: Automated Workflow

Create a VS Code task that processes multiple images:

```json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "OCR Batch Process",
      "type": "shell",
      "command": "python",
      "args": [
        "-c",
        "import asyncio; from mcp import ClientSession; # your batch processing code"
      ]
    }
  ]
}
```

## Path Formats

**Important:** Use appropriate path formats:

- **Windows paths in JSON:** `C:/path/to/file.jpg` (forward slashes)
- **WSL2 paths:** `/mnt/c/Users/akona/...`
- **Relative paths:** Relative to workspace root

## Troubleshooting

### MCP Server Shows "Disconnected"

**Check:**
1. Is vLLM server running? Run `scripts\test-vllm.bat`
2. Is the path in settings.json correct?
3. Are MCP dependencies installed?

**Fix:**
```cmd
# Test vLLM connection
scripts\test-vllm.bat

# Restart VS Code after configuration changes
```

### "Module not found" Error

**For WSL2 setup (Option A):**
```bash
wsl
cd /mnt/c/Users/akona/OneDrive/Dev/HunyuanOCR
source hunyuanocr-env/bin/activate
cd docs/mcp-server
python -c "import mcp; print('MCP OK')"
```

**For Windows setup (Option B):**
```powershell
cd C:\Users\akona\OneDrive\Dev\HunyuanOCR\docs\mcp-server
pip install -r requirements.txt
```

### Tool Calls Timeout

**vLLM might be slow on RTX 3060:**
- Increase timeout in MCP settings
- Check GPU utilization: `wsl nvidia-smi`
- Consider CPU mode if persistent issues

### Image Path Not Found

**Ensure:**
- Use absolute paths
- Use forward slashes in JSON: `C:/Users/...`
- File exists and is readable
- Supported formats: JPG, PNG, GIF, BMP, WEBP

## Performance Tips

### 1. Keep vLLM Running
Start vLLM once and leave it running:
```cmd
scripts\start-vllm.bat
```

### 2. Use Small Images
Resize large images before OCR:
```python
from PIL import Image
img = Image.open('large.jpg')
img.thumbnail((2048, 2048))
img.save('resized.jpg')
```

### 3. Batch Processing
For multiple images, process in sequence (RTX 3060 limited memory):
```python
for img in images:
    result = await ocr_extract_text(img)
    # process result
```

### 4. Monitor GPU Memory
```bash
wsl nvidia-smi
```

## Advanced Configuration

### Custom Environment Variables

Add to MCP settings:

```json
{
  "mcp.servers": {
    "hunyuan-ocr": {
      "command": "wsl",
      "args": ["bash", "-c", "..."],
      "env": {
        "VLLM_BASE_URL": "http://localhost:8000",
        "REQUEST_TIMEOUT_SECONDS": "600",
        "MAX_IMAGE_SIZE_MB": "20"
      }
    }
  }
}
```

### Remote vLLM Server

If vLLM runs on another machine:

```json
{
  "env": {
    "VLLM_BASE_URL": "http://192.168.1.100:8000"
  }
}
```

### Logging

Enable debug logging:

```json
{
  "env": {
    "LOG_LEVEL": "DEBUG"
  }
}
```

Check logs in VS Code Output panel (MCP Server output).

## Network Access for Remote Machines

If you want other machines to access your OCR server:

### Option 1: Each Machine Runs MCP Client

Each VS Code instance connects to your vLLM server:
- Configure firewall (see QUICKSTART.md)
- Use your machine's IP in `VLLM_BASE_URL`
- Each machine runs its own MCP server (lightweight)

### Option 2: Shared MCP Server (Advanced)

Run one MCP server accessible over network:
- Requires network protocol wrapper (not stdio)
- More complex setup
- Not recommended for local network

**Recommended:** Use Option 1 - let each machine run its own MCP server instance pointing to your shared vLLM server.

## Integration with GitHub Copilot

### Enable MCP Tools in Copilot

1. Ensure MCP extension is installed
2. Configure MCP server (above)
3. In Copilot chat, tools are automatically available

### Example Copilot Prompts

**Document Analysis:**
```
Analyze this document image and extract all key information
[attach image or provide path]
```

**Data Extraction:**
```
Extract invoice details from C:/invoices/inv_001.jpg as JSON
```

**Batch Processing:**
```
Process all receipt images in C:/receipts/ and create a CSV with totals
```

**Translation:**
```
Translate this Chinese document to English: C:/docs/chinese_doc.jpg
```

## Best Practices

### 1. Organize Images
Keep OCR images in a dedicated folder:
```
C:/Users/akona/OCR_Processing/
  ├── input/
  ├── output/
  └── archive/
```

### 2. Use Descriptive Names
```
receipt_2024-11-29_walmart.jpg
document_contract_page1.jpg
```

### 3. Create VS Code Snippets
Add to `.vscode/settings.json`:
```json
{
  "snippets": {
    "ocr-extract": {
      "prefix": "ocr-text",
      "body": [
        "Use ocr_extract_text tool on ${1:image_path}"
      ]
    }
  }
}
```

### 4. Error Handling
Always check if vLLM is running before OCR tasks:
```
First check health_check tool, then proceed with OCR
```

## Comparison: VS Code vs Claude Desktop

| Feature | VS Code MCP | Claude Desktop |
|---------|-------------|----------------|
| Integration | Native in editor | Separate app |
| Workflow | In-context | Copy/paste |
| Automation | VS Code tasks | Manual |
| Git integration | Yes | No |
| File handling | Direct access | Limited |
| Best for | Development | General use |

**Recommendation:** Use VS Code MCP for development workflows, file processing, and automation.

## Next Steps

1. ✅ Configure MCP in VS Code (Step 2 above)
2. ✅ Start vLLM server: `scripts\start-vllm.bat`
3. ✅ Test with simple OCR in Copilot chat
4. 📝 Create your own OCR workflows
5. 🚀 Automate repetitive OCR tasks

## Support

- **MCP Server Logs:** VS Code Output panel → "MCP Server"
- **vLLM Server:** Check terminal where `start-vllm.bat` is running
- **Test Tools:** Run `scripts\test-vllm.bat` for diagnostics

---

**Status:** Ready for VS Code integration  
**Last Updated:** November 29, 2025  
**Configuration:** Windows + WSL2 + VS Code MCP
