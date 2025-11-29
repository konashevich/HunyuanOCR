#!/usr/bin/env python3
"""
HunyuanOCR MCP Server
Exposes OCR capabilities via Model Context Protocol

This server wraps the HunyuanOCR vLLM API to provide network-accessible
OCR services for AI agents across the local network.
"""
import asyncio
import logging
from typing import Any
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

import ocr_tools
import config
from vllm_client import VLLMClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("hunyuan-ocr-mcp")

# Create MCP server instance
app = Server("hunyuan-ocr-server")

# Initialize vLLM client for health checks
vllm_client = VLLMClient()


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List all available OCR tools"""
    return [
        Tool(
            name="ocr_extract_text",
            description="Extract all text content from an image. Simple text extraction without coordinates or structure.",
            inputSchema={
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "Absolute path to image file or base64 data URI"
                    },
                    "language": {
                        "type": "string",
                        "enum": ["english", "chinese"],
                        "default": "english",
                        "description": "Language for prompts"
                    }
                },
                "required": ["image_path"]
            }
        ),
        Tool(
            name="ocr_spot_text",
            description="Detect and recognize text with bounding box coordinates. Returns text with position information in format: text(x1,y1),(x2,y2)",
            inputSchema={
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "Absolute path to image file"
                    },
                    "language": {
                        "type": "string",
                        "enum": ["english", "chinese"],
                        "default": "english"
                    }
                },
                "required": ["image_path"]
            }
        ),
        Tool(
            name="ocr_parse_document",
            description="Parse complex document with structured output. Returns markdown with HTML tables and LaTeX formulas, organized by reading order.",
            inputSchema={
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "Absolute path to document image"
                    },
                    "language": {
                        "type": "string",
                        "enum": ["english", "chinese"],
                        "default": "english"
                    }
                },
                "required": ["image_path"]
            }
        ),
        Tool(
            name="ocr_parse_table",
            description="Parse table from image into HTML format. Best for extracting tabular data.",
            inputSchema={
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "Absolute path to image containing table"
                    },
                    "language": {
                        "type": "string",
                        "enum": ["english", "chinese"],
                        "default": "english"
                    }
                },
                "required": ["image_path"]
            }
        ),
        Tool(
            name="ocr_parse_formula",
            description="Extract mathematical formula from image in LaTeX format. Best for scientific documents.",
            inputSchema={
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "Absolute path to image containing formula"
                    },
                    "language": {
                        "type": "string",
                        "enum": ["english", "chinese"],
                        "default": "english"
                    }
                },
                "required": ["image_path"]
            }
        ),
        Tool(
            name="ocr_extract_fields",
            description="Extract specific named fields from cards, receipts, invoices, or forms. Returns structured JSON with requested fields.",
            inputSchema={
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "Absolute path to image"
                    },
                    "fields": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of field names to extract (e.g., ['total_amount', 'date', 'merchant_name'])"
                    },
                    "language": {
                        "type": "string",
                        "enum": ["english", "chinese"],
                        "default": "english"
                    }
                },
                "required": ["image_path", "fields"]
            }
        ),
        Tool(
            name="ocr_translate_image",
            description="Extract text from image and translate to target language. Preserves document structure with LaTeX formulas and HTML tables.",
            inputSchema={
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "Absolute path to image"
                    },
                    "target_language": {
                        "type": "string",
                        "enum": ["english", "chinese"],
                        "default": "english",
                        "description": "Target language for translation"
                    }
                },
                "required": ["image_path"]
            }
        ),
        Tool(
            name="ocr_extract_subtitles",
            description="Extract subtitles from video frame or screenshot. Supports bilingual subtitles.",
            inputSchema={
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "Absolute path to video frame/screenshot"
                    },
                    "language": {
                        "type": "string",
                        "enum": ["english", "chinese"],
                        "default": "english"
                    }
                },
                "required": ["image_path"]
            }
        ),
        Tool(
            name="health_check",
            description="Check if the vLLM OCR backend server is healthy and responsive",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls and route to appropriate OCR function"""
    try:
        logger.info(f"Tool called: {name} with args: {arguments}")
        
        # Extract common parameters
        image_path = arguments.get("image_path")
        language = arguments.get("language", "english")
        
        # Route to appropriate tool
        if name == "ocr_extract_text":
            result = await ocr_tools.extract_text(image_path, language)
        
        elif name == "ocr_spot_text":
            result = await ocr_tools.spot_text(image_path, language)
        
        elif name == "ocr_parse_document":
            result = await ocr_tools.parse_document(image_path, language)
        
        elif name == "ocr_parse_table":
            result = await ocr_tools.parse_table(image_path, language)
        
        elif name == "ocr_parse_formula":
            result = await ocr_tools.parse_formula(image_path, language)
        
        elif name == "ocr_extract_fields":
            fields = arguments.get("fields", [])
            result = await ocr_tools.extract_fields(image_path, fields, language)
        
        elif name == "ocr_translate_image":
            target_language = arguments.get("target_language", "english")
            result = await ocr_tools.translate_image(image_path, target_language)
        
        elif name == "ocr_extract_subtitles":
            result = await ocr_tools.extract_subtitles(image_path, language)
        
        elif name == "health_check":
            is_healthy = await vllm_client.health_check()
            result = f"vLLM server is {'healthy' if is_healthy else 'not responding'}"
        
        else:
            logger.error(f"Unknown tool: {name}")
            return [TextContent(type="text", text=f"Error: Unknown tool '{name}'")]
        
        logger.info(f"Tool {name} completed successfully")
        return [TextContent(type="text", text=result)]
    
    except Exception as e:
        logger.error(f"Error in {name}: {str(e)}", exc_info=True)
        return [TextContent(type="text", text=f"Error: {str(e)}")]


async def main():
    """Main entry point for MCP server"""
    logger.info("Starting HunyuanOCR MCP Server")
    logger.info(f"vLLM Backend: {config.VLLM_BASE_URL}")
    logger.info(f"Model: {config.MODEL_NAME}")
    
    # Check vLLM server health
    is_healthy = await vllm_client.health_check()
    if is_healthy:
        logger.info("✓ vLLM server is healthy and ready")
    else:
        logger.warning("⚠ vLLM server is not responding. Please start it first:")
        logger.warning(f"   vllm serve {config.MODEL_NAME} --host 0.0.0.0 --port 8000")
    
    # Start MCP server
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
