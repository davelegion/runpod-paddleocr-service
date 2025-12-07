"""
RunPod Serverless Handler - PaddleOCR-VL Integration

This handler implements RunPod's expected serverless pattern while using our
existing PaddleOCR-VL inference logic.

RunPod calls: handler(event)
Returns: {"output": {...}, "error": ...}
"""

import os
import time
import json
import tempfile
from pathlib import Path
import runpod
from paddleocr import PaddleOCRVL


# ============================================================================
# Model Initialization (once on worker startup)
# ============================================================================

print("🚀 Initializing PaddleOCR-VL model...")

# Configure volume caching for models
VOLUME_PATH = Path("/runpod-volume")
MODEL_CACHE_DIR = VOLUME_PATH / "paddleocr"

# Check if volume is mounted
if VOLUME_PATH.exists():
    print(f"✅ RunPod volume detected at {VOLUME_PATH}")

    # Create cache directory if it doesn't exist
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Set PaddleOCR cache directory to volume
    os.environ["PADDLEOCR_CACHE_DIR"] = str(MODEL_CACHE_DIR)

    # Check if models already cached
    if (MODEL_CACHE_DIR / "paddleocr-vl").exists():
        print(f"✅ Using cached models from {MODEL_CACHE_DIR}")
    else:
        print(f"📥 First run: Downloading ~3GB models to {MODEL_CACHE_DIR}")
        print(f"   This will take 5-10 minutes (one-time only)")
else:
    print(f"⚠️  No volume mounted - models will download to container (slower)")

try:
    start = time.time()
    pipeline = PaddleOCRVL()
    elapsed = time.time() - start
    print(f"✅ PaddleOCR-VL model loaded in {elapsed:.1f}s")

    # Warmup to reduce cold start latency
    try:
        from PIL import Image
        dummy_img = Image.new('RGB', (100, 100), color='white')
        dummy_path = Path(tempfile.gettempdir()) / "warmup.jpg"
        dummy_img.save(dummy_path)
        pipeline.predict(str(dummy_path))
        print("✅ Model warmup complete")
    except Exception as e:
        print(f"⚠️  Warmup failed (non-critical): {e}")

except Exception as e:
    print(f"❌ Failed to initialize PaddleOCR-VL: {e}")
    pipeline = None


# ============================================================================
# Helper Functions
# ============================================================================


def download_document(url: str) -> Path:
    """Download document from URL to temporary file."""
    print(f"📥 Downloading document from: {url}")

    import requests as sync_requests

    response = sync_requests.get(url, timeout=30.0)
    response.raise_for_status()

    # Determine file extension from URL or content-type
    content_type = response.headers.get("content-type", "")
    if "pdf" in content_type or url.lower().endswith(".pdf"):
        ext = ".pdf"
    elif "png" in content_type or url.lower().endswith(".png"):
        ext = ".png"
    elif "jpg" in content_type or "jpeg" in content_type or url.lower().endswith((".jpg", ".jpeg")):
        ext = ".jpg"
    else:
        ext = ".pdf"  # Default to PDF

    # Save to temp file
    temp_file = Path(tempfile.gettempdir()) / f"document_{int(time.time())}{ext}"
    temp_file.write_bytes(response.content)

    print(f"✅ Document downloaded: {temp_file} ({len(response.content)} bytes)")
    return temp_file


def parse_ocr_output(result: dict) -> dict:
    """
    Parse PaddleOCR-VL output into structured format.

    PaddleOCR-VL returns comprehensive document understanding results.
    We extract key invoice/receipt fields using heuristics.
    """
    # PaddleOCR-VL returns complex nested structure
    # For now, extract text and basic structure
    # TODO: Enhance with proper field extraction logic

    extracted_fields = {}
    full_text = ""
    avg_confidence = 0.95

    # Extract text from results
    # PaddleOCR-VL format: results typically contain text blocks with positions and content
    if isinstance(result, list):
        text_blocks = []
        confidences = []

        for item in result:
            if isinstance(item, dict) and "text" in item:
                text_blocks.append(item["text"])
                confidences.append(item.get("confidence", 0.95))
            elif isinstance(item, (list, tuple)) and len(item) > 1:
                # Format: [bounding_box, (text, confidence)]
                if isinstance(item[1], (list, tuple)) and len(item[1]) > 0:
                    text_blocks.append(str(item[1][0]))
                    if len(item[1]) > 1:
                        confidences.append(float(item[1][1]))

        full_text = "\n".join(text_blocks)
        if confidences:
            avg_confidence = sum(confidences) / len(confidences)

    # Basic field extraction (heuristic-based)
    # In production, this would use more sophisticated NER or pattern matching
    lines = full_text.split("\n")

    for i, line in enumerate(lines):
        line_lower = line.lower()

        # Invoice/Receipt number
        if "invoice" in line_lower or "receipt" in line_lower or "no." in line_lower:
            # Next line might contain the number
            if i + 1 < len(lines):
                extracted_fields["invoice_number"] = lines[i + 1].strip()

        # Date
        if "date" in line_lower:
            if i + 1 < len(lines):
                extracted_fields["invoice_date"] = lines[i + 1].strip()

        # Total amount
        if "total" in line_lower or "amount" in line_lower:
            if i + 1 < len(lines):
                extracted_fields["total_amount"] = lines[i + 1].strip()

        # VAT
        if "vat" in line_lower or "tax" in line_lower:
            if i + 1 < len(lines):
                extracted_fields["vat_amount"] = lines[i + 1].strip()

        # Merchant name (usually first few lines)
        if i < 3 and len(line.strip()) > 3:
            if "merchant_name" not in extracted_fields:
                extracted_fields["merchant_name"] = line.strip()

    return {
        "text": full_text,
        "confidence": avg_confidence,
        "fields": extracted_fields,
        "raw_output": result if isinstance(result, (dict, list)) else str(result),
    }


# ============================================================================
# RunPod Handler
# ============================================================================


def inference(job):
    """
    Main inference handler called by RunPod.

    Args:
        job: RunPod job input containing model inference request

    Returns:
        dict: {"output": {...}, "error": None} on success
              {"output": None, "error": {...}} on failure
    """
    print(f"\n{'=' * 70}")
    print(f"📥 New inference request")
    print(f"{'=' * 70}")

    start_time = time.time()
    correlation_id = job.get("correlation_id", "unknown")

    try:
        # Check if model is loaded
        if pipeline is None:
            return {
                "output": None,
                "error": {
                    "code": "MODEL_NOT_LOADED",
                    "message": "PaddleOCR-VL model failed to initialize",
                    "retriable": False,
                },
            }

        # Extract inputs
        inputs = job.get("input", {})  # Fixed: RunPod sends "input" (singular), not "inputs"
        document_url = inputs.get("document_url")

        if not document_url:
            return {
                "output": None,
                "error": {
                    "code": "INVALID_INPUT",
                    "message": "Missing required input: document_url",
                    "retriable": False,
                },
            }

        print(f"📄 Document URL: {document_url}")
        print(f"🔑 Correlation ID: {correlation_id}")

        # Download document
        try:
            document_path = download_document(document_url)
        except Exception as e:
            # Handle both requests.HTTPError and other errors
            error_msg = str(e)
            if hasattr(e, 'response') and hasattr(e.response, 'status_code'):
                error_msg = f"HTTP {e.response.status_code}: {error_msg}"
            return {
                "output": None,
                "error": {
                    "code": "DOCUMENT_DOWNLOAD_FAILED",
                    "message": f"Failed to download document: {error_msg}",
                    "retriable": True,
                },
            }

        # Run OCR prediction
        print(f"🔮 Running PaddleOCR-VL prediction...")
        try:
            result = pipeline.predict(str(document_path))
            print(f"✅ OCR prediction complete")
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                return {
                    "output": None,
                    "error": {
                        "code": "CUDA_OOM",
                        "message": "GPU out of memory. Try with a smaller document.",
                        "retriable": True,
                    },
                }
            raise
        finally:
            # Clean up temp file
            try:
                document_path.unlink()
            except:
                pass

        # Parse results
        parsed_output = parse_ocr_output(result)

        # Calculate latency
        latency_ms = int((time.time() - start_time) * 1000)

        # Return successful response
        output = {
            "success": True,
            "model_name": "ocr-vl-0.9b",
            "model_version": "2025-01-01",
            "latency_ms": latency_ms,
            "prediction": parsed_output,
            "metadata": {
                "correlation_id": correlation_id,
                "gpu_node": os.getenv("RUNPOD_POD_ID", "unknown"),
            },
        }

        print(f"✅ Inference complete in {latency_ms}ms")
        print(f"   Confidence: {parsed_output['confidence']:.2f}")
        print(f"   Fields extracted: {list(parsed_output['fields'].keys())}")

        return {"output": output, "error": None}

    except Exception as e:
        import traceback

        error_trace = traceback.format_exc()
        print(f"❌ Inference failed: {e}")
        print(error_trace)

        return {
            "output": None,
            "error": {
                "code": "INFERENCE_ERROR",
                "message": str(e),
                "retriable": False,
                "traceback": error_trace,
            },
        }


# ============================================================================
# RunPod Serverless Startup
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("RunPod Model Service - PaddleOCR-VL")
    print("=" * 70)
    print(f"Model loaded: {'✅ Yes' if pipeline else '❌ No'}")
    print("Starting RunPod serverless handler...")
    print("=" * 70)

    # Start RunPod serverless handler
    runpod.serverless.start({"handler": inference})
