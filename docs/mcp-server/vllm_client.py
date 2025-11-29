"""
Async client for vLLM API
"""
import base64
import aiohttp
from pathlib import Path
from typing import Optional
import config

class VLLMClient:
    def __init__(self, base_url: Optional[str] = None):
        self.base_url = (base_url or config.VLLM_BASE_URL).rstrip("/")
        self.api_endpoint = f"{self.base_url}/v1/chat/completions"
        self.model_name = config.MODEL_NAME
    
    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64 data URI"""
        if image_path.startswith("data:"):
            return image_path  # Already base64 data URI
        
        # Read and encode image
        with open(image_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode('utf-8')
        
        # Detect image format
        suffix = Path(image_path).suffix.lower()
        mime_type = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".bmp": "image/bmp",
            ".webp": "image/webp"
        }.get(suffix, "image/jpeg")
        
        return f"data:{mime_type};base64,{encoded}"
    
    async def process_image(
        self,
        image_path: str,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 16384
    ) -> str:
        """Send OCR request to vLLM server and return text response"""
        try:
            image_data = self._encode_image(image_path)
            
            payload = {
                "model": self.model_name,
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
                "max_tokens": max_tokens,
                "stream": False,
                "extra_body": {
                    "top_k": 1,
                    "repetition_penalty": 1.0
                }
            }
            
            timeout = aiohttp.ClientTimeout(total=config.REQUEST_TIMEOUT_SECONDS)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(self.api_endpoint, json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"vLLM API error: {response.status} - {error_text}")
                    
                    result = await response.json()
                    return result["choices"][0]["message"]["content"]
        
        except FileNotFoundError:
            raise Exception(f"Image file not found: {image_path}")
        except aiohttp.ClientError as e:
            raise Exception(f"Network error connecting to vLLM server: {str(e)}")
        except Exception as e:
            raise Exception(f"Error processing image: {str(e)}")
    
    async def health_check(self) -> bool:
        """Check if vLLM server is responsive"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/health", timeout=aiohttp.ClientTimeout(total=5)) as response:
                    return response.status == 200
        except:
            return False
