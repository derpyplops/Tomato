"""
FastAPI server exposing the Tomato encoder.
Handles one encoding request at a time to avoid RAM exhaustion.
"""

import asyncio
import numpy as np
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Step 1: Import logger and patch FIRST (before Encoder import)
from tomato.utils.early_patch_logger import EarlyPatchLogger

# Global state
encoding_lock = asyncio.Lock()
logger = None
Encoder = None  # Will be imported after patch

# Default configuration
DEFAULT_MODEL = "google/gemma-3-1b-it"
DEFAULT_TEMPERATURE = 1.3


# Pydantic Models
class EncodeRequest(BaseModel):
    plaintext: str = Field(..., description="Secret message to encode")
    prompt: str = Field(..., description="Cover story prompt for stegotext generation")
    cipher_len: int = Field(..., description="Length of the cipher in bytes", gt=0)
    max_len: int = Field(..., description="Maximum length of generated stegotext", gt=0)


class EncodeResponse(BaseModel):
    stegotext: list = Field(..., description="Token IDs of the stegotext")
    formatted_stegotext: str = Field(..., description="Formatted stegotext for display")


class ErrorResponse(BaseModel):
    detail: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic"""
    global logger, Encoder

    # Startup: Apply early patch
    print("🚀 Starting Tomato Encoder API Server...")
    logger = EarlyPatchLogger(log_dir="./logs")
    logger.patch_now()
    print("✓ Applied early patch to greedy_mec")

    # Import Encoder after patching
    from tomato import Encoder as EncoderClass
    Encoder = EncoderClass
    print("✓ Encoder class imported")

    yield

    # Shutdown: Clean up
    if logger:
        logger.unpatch()
        print("✓ Removed patch from greedy_mec")
    print("👋 Shutting down...")


app = FastAPI(
    title="Tomato Encoder API",
    description="API for encoding secret messages into natural-looking text using LLM steganography",
    version="1.0.0",
    lifespan=lifespan
)


def convert_numpy_types(obj):
    """Recursively convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


def create_encoder(prompt: str, cipher_len: int, max_len: int):
    """Create a new encoder instance with the given parameters"""
    print(f"🔄 Creating encoder: model={DEFAULT_MODEL}, cipher_len={cipher_len}, max_len={max_len}")

    enc = Encoder(
        model_name=DEFAULT_MODEL,
        prompt=prompt,
        cipher_len=cipher_len,
        max_len=max_len,
        temperature=DEFAULT_TEMPERATURE
    )

    print("✓ Encoder created successfully")
    return enc


@app.get("/health", summary="Health check")
async def health_check():
    """Check if the server is running"""
    return {
        "status": "healthy",
        "model": DEFAULT_MODEL,
        "temperature": DEFAULT_TEMPERATURE
    }


@app.post(
    "/encode",
    response_model=EncodeResponse,
    responses={
        503: {
            "model": ErrorResponse,
            "description": "Server is busy processing another request"
        }
    },
    summary="Encode plaintext into stegotext"
)
async def encode(request: EncodeRequest):
    """
    Encode a secret message into natural-looking text.

    Only one encoding operation is allowed at a time to prevent RAM exhaustion.
    Returns 503 if another request is being processed.
    """

    # Try to acquire lock (non-blocking)
    if encoding_lock.locked():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server is currently processing another request. Please try again."
        )

    async with encoding_lock:
        try:
            # Create new encoder for this request
            enc = create_encoder(
                prompt=request.prompt,
                cipher_len=request.cipher_len,
                max_len=request.max_len
            )

            print(f"🔐 Encoding: {len(request.plaintext)} chars")

            # Run encoding in thread pool (CPU-intensive operation)
            loop = asyncio.get_event_loop()
            formatted_stegotext, stegotext, probs = await loop.run_in_executor(
                None,
                enc.encode,
                request.plaintext,
                False  # debug=False
            )

            print(f"✓ Encoding complete: {len(stegotext)} tokens generated")

            # Convert stegotext to list if it's a numpy array
            if hasattr(stegotext, 'tolist'):
                stegotext_list = stegotext.tolist()
            else:
                stegotext_list = list(stegotext)

            # Build response (convert all numpy types to native Python types)
            response = {
                "stegotext": convert_numpy_types(stegotext_list),
                "formatted_stegotext": formatted_stegotext
            }

            return response

        except Exception as e:
            print(f"❌ Encoding failed: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Encoding failed: {str(e)}"
            )


if __name__ == "__main__":
    import uvicorn

    print("=" * 70)
    print("TOMATO ENCODER API SERVER")
    print("=" * 70)
    print(f"Model: {DEFAULT_MODEL}")
    print(f"Temperature: {DEFAULT_TEMPERATURE}")
    print("=" * 70)

    uvicorn.run(app, host="0.0.0.0", port=8000)
