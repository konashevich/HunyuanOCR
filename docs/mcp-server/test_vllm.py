"""
Test script for vLLM API functionality
Run this after starting vLLM server to verify it's working
"""
import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from vllm_client import VLLMClient


async def test_health():
    """Test server health check"""
    print("=" * 60)
    print("Testing vLLM Server Health")
    print("=" * 60)
    
    client = VLLMClient()
    is_healthy = await client.health_check()
    
    if is_healthy:
        print("✓ vLLM server is healthy and responding")
    else:
        print("✗ vLLM server is not responding")
        print("\nPlease start the vLLM server first:")
        print("  vllm serve tencent/HunyuanOCR --host 0.0.0.0 --port 8000")
        return False
    
    return True


async def test_simple_ocr():
    """Test simple text extraction"""
    print("\n" + "=" * 60)
    print("Testing Simple Text Extraction")
    print("=" * 60)
    
    # Look for test image
    test_images = [
        "../../assets/spotting1_cropped.png",
        "../../assets/vis_document_23.jpg",
        "../assets/spotting1_cropped.png"
    ]
    
    test_image = None
    for img_path in test_images:
        full_path = Path(__file__).parent / img_path
        if full_path.exists():
            test_image = str(full_path)
            break
    
    if not test_image:
        print("⚠ No test images found. Please specify an image path.")
        print("Available test images in assets/:")
        assets_dir = Path(__file__).parent.parent.parent / "assets"
        if assets_dir.exists():
            for img in assets_dir.glob("*.jpg"):
                print(f"  - {img.name}")
            for img in assets_dir.glob("*.png"):
                print(f"  - {img.name}")
        return
    
    print(f"\nUsing test image: {test_image}")
    
    client = VLLMClient()
    prompt = "Extract the text in the image."
    
    print(f"Prompt: {prompt}")
    print("\nProcessing image (this may take 10-30 seconds)...")
    
    try:
        result = await client.process_image(test_image, prompt)
        print("\n" + "-" * 60)
        print("OCR Result:")
        print("-" * 60)
        print(result[:500])  # Show first 500 chars
        if len(result) > 500:
            print(f"\n... (truncated, total {len(result)} characters)")
        print("-" * 60)
        print("✓ Text extraction successful")
    except Exception as e:
        print(f"✗ Error: {str(e)}")


async def test_document_parsing():
    """Test document parsing with structure"""
    print("\n" + "=" * 60)
    print("Testing Document Parsing")
    print("=" * 60)
    
    # Look for document test image
    test_images = [
        "../../assets/vis_parsing_table.png",
        "../../assets/vis_document_23.jpg",
        "../assets/vis_parsing_table.png"
    ]
    
    test_image = None
    for img_path in test_images:
        full_path = Path(__file__).parent / img_path
        if full_path.exists():
            test_image = str(full_path)
            break
    
    if not test_image:
        print("⚠ Skipping document parsing test (no test image found)")
        return
    
    print(f"\nUsing test image: {test_image}")
    
    client = VLLMClient()
    prompt = ("Extract all information from the main body of the document image "
              "and represent it in markdown format, ignoring headers and footers. "
              "Tables should be expressed in HTML format, formulas in the document "
              "should be represented using LaTeX format, and the parsing should be "
              "organized according to the reading order.")
    
    print(f"Prompt: {prompt[:80]}...")
    print("\nProcessing document (this may take 20-60 seconds)...")
    
    try:
        result = await client.process_image(test_image, prompt, max_tokens=16384)
        print("\n" + "-" * 60)
        print("Parsed Document:")
        print("-" * 60)
        print(result[:800])  # Show first 800 chars
        if len(result) > 800:
            print(f"\n... (truncated, total {len(result)} characters)")
        print("-" * 60)
        print("✓ Document parsing successful")
    except Exception as e:
        print(f"✗ Error: {str(e)}")


async def main():
    """Run all tests"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 12 + "HunyuanOCR vLLM API Test Suite" + " " * 15 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    # Test 1: Health check
    if not await test_health():
        print("\n⚠ Tests aborted: vLLM server is not running")
        return
    
    # Test 2: Simple OCR
    await test_simple_ocr()
    
    # Test 3: Document parsing
    await test_document_parsing()
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. If tests passed, the vLLM server is working correctly")
    print("2. You can now start the MCP server: python server.py")
    print("3. Test MCP server with: python test_mcp_client.py")
    print()


if __name__ == "__main__":
    asyncio.run(main())
