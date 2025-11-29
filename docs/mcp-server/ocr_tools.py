"""
OCR tool implementations using vLLM backend
"""
import json
from vllm_client import VLLMClient

# Initialize client
client = VLLMClient()

# Prompt templates for different OCR tasks
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
    "parse_table": {
        "english": "Parse the table in the image into HTML.",
        "chinese": "把图中的表格解析为 HTML。"
    },
    "parse_formula": {
        "english": "Identify the formula in the image and represent it using LaTeX format.",
        "chinese": "识别图片中的公式，用 LaTeX 格式表示。"
    },
    "translate": {
        "to_english": "First extract the text, then translate the text content into English. If it is a document, ignore the header and footer. Formulas should be represented in LaTeX format, and tables should be represented in HTML format.",
        "to_chinese": "先提取文字，再将文字内容翻译为中文。若是文档，则其中页眉、页脚忽略。公式用latex格式表示，表格用html格式表示。"
    },
    "subtitles": {
        "english": "Extract the subtitles from the image.",
        "chinese": "提取图片中的字幕。"
    }
}


async def extract_text(image_path: str, language: str = "english") -> str:
    """
    Extract all text from image.
    
    Args:
        image_path: Path to image file
        language: Language for prompt ("english" or "chinese")
    
    Returns:
        Extracted text
    """
    prompt = PROMPTS["extract_text"][language]
    return await client.process_image(image_path, prompt)


async def spot_text(image_path: str, language: str = "english") -> str:
    """
    Detect text with bounding box coordinates.
    
    Args:
        image_path: Path to image file
        language: Language for prompt ("english" or "chinese")
    
    Returns:
        Text with coordinates in format: "text(x1,y1),(x2,y2)"
    """
    prompt = PROMPTS["spot_text"][language]
    return await client.process_image(image_path, prompt)


async def parse_document(image_path: str, language: str = "english") -> str:
    """
    Parse complex document with structured output.
    
    Args:
        image_path: Path to document image
        language: Language for prompt ("english" or "chinese")
    
    Returns:
        Markdown with HTML tables and LaTeX formulas
    """
    prompt = PROMPTS["parse_document"][language]
    return await client.process_image(image_path, prompt, max_tokens=16384)


async def parse_table(image_path: str, language: str = "english") -> str:
    """
    Parse table to HTML format.
    
    Args:
        image_path: Path to image containing table
        language: Language for prompt ("english" or "chinese")
    
    Returns:
        HTML table representation
    """
    prompt = PROMPTS["parse_table"][language]
    return await client.process_image(image_path, prompt)


async def parse_formula(image_path: str, language: str = "english") -> str:
    """
    Extract mathematical formula in LaTeX format.
    
    Args:
        image_path: Path to image containing formula
        language: Language for prompt ("english" or "chinese")
    
    Returns:
        LaTeX formula
    """
    prompt = PROMPTS["parse_formula"][language]
    return await client.process_image(image_path, prompt)


async def extract_fields(image_path: str, fields: list, language: str = "english") -> str:
    """
    Extract specific fields from cards, receipts, forms as JSON.
    
    Args:
        image_path: Path to image
        fields: List of field names to extract
        language: Language for prompt ("english" or "chinese")
    
    Returns:
        JSON string with extracted fields
    """
    fields_str = str(fields)
    if language == "chinese":
        prompt = f"提取图片中的: {fields_str} 的字段内容，并按照 JSON 格式返回。"
    else:
        prompt = f"Extract the content of the fields: {fields_str} from the image and return it in JSON format."
    
    return await client.process_image(image_path, prompt)


async def translate_image(image_path: str, target_language: str = "english") -> str:
    """
    Extract text from image and translate to target language.
    
    Args:
        image_path: Path to image
        target_language: Target language ("english" or "chinese")
    
    Returns:
        Translated text
    """
    prompt_key = "to_english" if target_language == "english" else "to_chinese"
    prompt = PROMPTS["translate"][prompt_key]
    return await client.process_image(image_path, prompt, max_tokens=16384)


async def extract_subtitles(image_path: str, language: str = "english") -> str:
    """
    Extract subtitles from video frame or screenshot.
    
    Args:
        image_path: Path to video frame/screenshot
        language: Language for prompt ("english" or "chinese")
    
    Returns:
        Extracted subtitle text
    """
    prompt = PROMPTS["subtitles"][language]
    return await client.process_image(image_path, prompt)
