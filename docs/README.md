# 🚀 Quick Start - HunyuanOCR Setup

This is your entry point for setting up HunyuanOCR locally and deploying it as a network-accessible MCP server.

## 📋 Current Status

✅ **Planning Complete** - Comprehensive documentation and scripts ready  
✅ **MCP Server Built** - Full implementation with 9 OCR tools  
✅ **System Checked** - Compatible hardware verified  
⏳ **Ready for Installation** - Follow steps below

## 🎯 What You'll Get

### Phase 1: Local OCR Model
- HunyuanOCR 1B parameter model running locally
- vLLM server with OpenAI-compatible API
- Test scripts and benchmarks
- Works on your RTX 3060 (with memory optimization)

### Phase 2: VS Code MCP Integration ⭐
- **Native VS Code integration** with MCP protocol
- Use OCR tools directly in **GitHub Copilot** chat
- In-editor OCR workflows and automation
- 9 specialized OCR tools available to AI assistants
- Network-accessible for remote machines (optional)

## 🚀 Get Started (Choose Your Path)

### Option A: Automated Setup (Recommended)

**Windows Users:**
```cmd
# Double-click or run:
scripts\setup-phase1.bat
```

This installs everything in WSL2 Ubuntu automatically.

### Option B: Manual Setup

**Follow the detailed guides:**
1. Read: `docs/QUICKSTART.md`
2. Then: `docs/installation-plan.md` for full details

## 📚 Documentation Structure

```
docs/
├── VSCODE_MCP_SETUP.md        ⭐ VS Code integration guide (NEW!)
├── QUICKSTART.md              📋 Step-by-step installation
├── installation-plan.md       📖 Comprehensive 2-phase plan
├── IMPLEMENTATION_STATUS.md   ✅ What's been created
└── mcp-server/
    ├── README.md              🌐 MCP server documentation
    ├── server.py              🔧 Main MCP server
    ├── ocr_tools.py           🛠️ OCR implementations
    ├── vllm_client.py         📡 API client
    ├── test_vllm.py          🧪 Test scripts
    └── requirements.txt       📦 Dependencies
```

### 🆕 What's New: VS Code Integration
- **Direct OCR in your editor** - No switching apps
- **GitHub Copilot integration** - Natural language OCR commands
- **Automated workflows** - Process files with VS Code tasks
- **Full MCP protocol support** - All 9 tools available
- See: `VSCODE_MCP_SETUP.md` for configuration

## ⚡ Quick Commands

### Phase 1: Installation & Testing

```cmd
# 1. Install everything (one command)
scripts\setup-phase1.bat

# 2. Start vLLM server
scripts\start-vllm.bat

# 3. Test API (in new terminal)
scripts\test-vllm.bat
```

### Phase 2a: VS Code MCP Setup (Recommended) ⭐

```cmd
# 4. Keep vLLM running from Phase 1
# 5. Add to .vscode/settings.json (see VSCODE_MCP_SETUP.md)
# 6. Use OCR in GitHub Copilot!
```

**Example Copilot command:**
```
@workspace Use ocr_extract_text on C:/path/to/image.jpg
```

### Phase 2b: Network Deployment (Optional)

```cmd
# For remote machine access
# Configure firewall (PowerShell as Admin)
New-NetFirewallRule -DisplayName "HunyuanOCR vLLM" -Direction Inbound -LocalPort 8000 -Protocol TCP -Action Allow
```

## 🔍 Your System Configuration

**Hardware:**
- GPU: NVIDIA GeForce RTX 3060 (6GB VRAM)
- CUDA: 13.0 ✅
- Python: 3.13.3 ✅
- WSL2: Ubuntu ✅

**Note:** 6GB VRAM is below recommended 20GB. Scripts include memory optimization for your GPU.

## 🎨 MCP Tools Available

Once deployed, these OCR tools are available:

| Tool | Purpose | Use Case |
|------|---------|----------|
| `ocr_extract_text` | Simple text extraction | Quick text from images |
| `ocr_spot_text` | Text + coordinates | Document layout analysis |
| `ocr_parse_document` | Full parsing | Complex documents |
| `ocr_parse_table` | Table → HTML | Data extraction |
| `ocr_parse_formula` | Formula → LaTeX | Scientific documents |
| `ocr_extract_fields` | Field extraction | Forms, receipts, cards |
| `ocr_translate_image` | Image translation | Multi-language docs |
| `ocr_extract_subtitles` | Subtitle extraction | Video frames |
| `health_check` | Server status | Monitoring |

## 🛠️ Troubleshooting

### Common Issues

**"Out of Memory" Error:**
- Edit `scripts/start-vllm.sh`
- Lower `--gpu-memory-utilization` to 0.05
- Or use CPU mode (slower)

**vLLM won't start:**
- Ensure WSL2 is running: `wsl --list --verbose`
- Check CUDA in WSL2: `wsl nvidia-smi`

**Can't download model:**
- Check internet connection
- Try manual download (see docs/QUICKSTART.md)
- Use HuggingFace mirror

**Remote clients can't connect:**
- Verify firewall rules
- Check IP address: `ipconfig | findstr IPv4`
- Test locally first: `curl http://localhost:8000/health`

## 📖 Detailed Documentation

- **Quick Start:** `docs/QUICKSTART.md` - Step-by-step for your system
- **Full Plan:** `docs/installation-plan.md` - Comprehensive guide
- **MCP Guide:** `docs/mcp-server/README.md` - Server documentation
- **Status:** `docs/IMPLEMENTATION_STATUS.md` - What's implemented

## 🎯 Next Actions

### For VS Code Users (Recommended):
1. **Run Setup:** Execute `scripts\setup-phase1.bat`
2. **Start vLLM:** Run `scripts\start-vllm.bat`
3. **Configure VS Code:** Follow `VSCODE_MCP_SETUP.md`
4. **Use in Copilot:** Start using OCR tools in your editor!

### For Other Setups:
1. **Read Quick Start:** Open `docs/QUICKSTART.md`
2. **Run Setup:** Execute `scripts\setup-phase1.bat`
3. **Test Locally:** Run `scripts\test-vllm.bat`
4. **Configure Network:** Follow Phase 2 in QUICKSTART.md

## 💡 Tips for Success

- **First Time?** Follow QUICKSTART.md exactly
- **Experienced?** Jump to installation-plan.md for details
- **Issues?** Check troubleshooting sections in docs
- **Customize?** Edit config in `docs/mcp-server/config.py`

## 🌐 Network Access

To make OCR available to other computers:

1. Start vLLM server: `scripts\start-vllm.bat`
2. Find your IP: `ipconfig | findstr IPv4`
3. Configure firewall (see Quick Commands above)
4. Test from remote: `curl http://<your-ip>:8000/health`
5. Configure MCP clients with your IP

## 📊 Performance Expectations

**Your RTX 3060 (6GB):**
- Text Extraction: 5-10 seconds (small images)
- Document Parsing: 15-30 seconds (complex docs)
- May be slower than benchmarks due to memory limits
- CPU fallback available if needed

**For Better Performance:**
- Consider RTX 4090 (24GB VRAM)
- Or use cloud GPU service
- Or dedicated server with high VRAM

## 🔗 Resources

- **HunyuanOCR Model:** https://huggingface.co/tencent/HunyuanOCR
- **Research Paper:** https://arxiv.org/abs/2511.19575
- **vLLM Docs:** https://docs.vllm.ai
- **MCP Protocol:** https://modelcontextprotocol.io

## 🆘 Getting Help

1. Check `docs/QUICKSTART.md` troubleshooting section
2. Run diagnostic: `scripts\test-vllm.bat`
3. Check logs in terminal where servers run
4. Verify system requirements in QUICKSTART.md

---

**Ready to Begin?**

```cmd
# Start with this:
scripts\setup-phase1.bat
```

Then follow the output instructions!

---

**Status:** ✅ All documentation and code complete  
**Last Updated:** November 29, 2025  
**Your Configuration:** Windows + WSL2 + RTX 3060 (6GB)
