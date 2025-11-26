"""
FastAPI server exposing the Tomato encoder.
Handles one encoding request at a time to avoid RAM exhaustion.
"""

import asyncio
import numpy as np
import secrets
from pathlib import Path
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
import json

# Step 1: Import logger and patch FIRST (before Encoder import)
from tomato.utils.early_patch_logger import EarlyPatchLogger

# Global state
encoding_lock = asyncio.Lock()
logger = None
Encoder = None  # Will be imported after patch
server_key = None  # Shared private key for all operations

# Default configuration
DEFAULT_MODEL = "google/gemma-3-1b-it"
DEFAULT_TEMPERATURE = 1.3
KEY_FILE = ".server_key"
KEY_LENGTH = 100  # 100 bytes for the shared key


def load_or_create_key() -> bytes:
    """Load existing key from file or create a new one"""
    key_path = Path(KEY_FILE)

    if key_path.exists():
        print(f"📂 Loading existing key from {KEY_FILE}")
        with open(key_path, 'rb') as f:
            key = f.read()
        if len(key) != KEY_LENGTH:
            raise ValueError(f"Invalid key length in {KEY_FILE}: expected {KEY_LENGTH}, got {len(key)}")
        print(f"✓ Loaded {len(key)}-byte key")
        return key
    else:
        print(f"🔑 Generating new {KEY_LENGTH}-byte key")
        key = secrets.token_bytes(KEY_LENGTH)
        with open(key_path, 'wb') as f:
            f.write(key)
        print(f"✓ Saved key to {KEY_FILE}")
        return key


# Pydantic Models
class EncodeRequest(BaseModel):
    plaintext: str = Field(..., description="Secret message to encode")
    prompt: str = Field(..., description="Cover story prompt for stegotext generation")
    cipher_len: int = Field(..., description="Length of the cipher in bytes", gt=0, le=100)
    max_len: int = Field(..., description="Maximum length of generated stegotext", gt=0)
    calculate_failure_probs: bool = Field(False, description="Calculate failure probabilities (requires extra decoding pass, ~2x slower)")


class EncodeResponse(BaseModel):
    stegotext: list = Field(..., description="Token IDs of the stegotext")
    formatted_stegotext: str = Field(..., description="Formatted stegotext for display")


class DecodeRequest(BaseModel):
    stegotext: list = Field(..., description="Token IDs of the stegotext to decode")
    prompt: str = Field(..., description="Same cover story prompt used during encoding")
    cipher_len: int = Field(..., description="Same cipher length used during encoding", gt=0, le=100)
    max_len: int = Field(..., description="Same max_len used during encoding", gt=0)


class DecodeResponse(BaseModel):
    plaintext: str = Field(..., description="Decoded secret message")


class StreamEncodeRequest(BaseModel):
    plaintext: str = Field(..., description="Secret message to encode")
    prompt: str = Field(..., description="Cover story prompt for stegotext generation")
    cipher_len: int = Field(..., description="Length of the cipher in bytes", gt=0, le=100)
    max_len: int = Field(..., description="Maximum length of generated stegotext", gt=0)
    chunk_size: int = Field(1, description="Number of tokens per chunk (1 = token-by-token streaming)", gt=0)
    calculate_failure_probs: bool = Field(False, description="Calculate failure probabilities")


class ErrorResponse(BaseModel):
    detail: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic"""
    global logger, Encoder, server_key

    # Startup: Apply early patch
    print("🚀 Starting Tomato Encoder API Server...")
    logger = EarlyPatchLogger(log_dir="./logs")
    logger.patch_now()
    print("✓ Applied early patch to greedy_mec")

    # Import Encoder after patching
    from tomato import Encoder as EncoderClass
    Encoder = EncoderClass
    print("✓ Encoder class imported")

    # Load or create server key
    server_key = load_or_create_key()
    print(f"✓ Server key ready (first 8 bytes: {server_key[:8].hex()}...)")

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


def create_encoder(prompt: str, cipher_len: int, max_len: int, shared_private_key: Optional[bytes] = None):
    """Create a new encoder instance with the given parameters"""
    print(f"🔄 Creating encoder: model={DEFAULT_MODEL}, cipher_len={cipher_len}, max_len={max_len}")

    enc = Encoder(
        model_name=DEFAULT_MODEL,
        prompt=prompt,
        cipher_len=cipher_len,
        max_len=max_len,
        temperature=DEFAULT_TEMPERATURE,
        shared_private_key=shared_private_key
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
            # Use the first cipher_len bytes of the server key
            key_slice = server_key[:request.cipher_len]

            # Create new encoder for this request
            enc = create_encoder(
                prompt=request.prompt,
                cipher_len=request.cipher_len,
                max_len=request.max_len,
                shared_private_key=key_slice
            )

            print(f"🔐 Encoding: {len(request.plaintext)} chars (failure_probs={request.calculate_failure_probs})")

            # Run encoding in thread pool (CPU-intensive operation)
            loop = asyncio.get_event_loop()
            formatted_stegotext, stegotext, probs = await loop.run_in_executor(
                None,
                lambda: enc.encode(
                    request.plaintext,
                    debug=False,
                    calculate_failure_probs=request.calculate_failure_probs
                )
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


@app.post(
    "/decode",
    response_model=DecodeResponse,
    responses={
        503: {
            "model": ErrorResponse,
            "description": "Server is busy processing another request"
        }
    },
    summary="Decode stegotext back to plaintext"
)
async def decode(request: DecodeRequest):
    """
    Decode stegotext back into the original secret message.

    Requires the same parameters used during encoding (prompt, cipher_len, max_len)
    and the shared_private_key that was returned from the encode endpoint.

    Only one decoding operation is allowed at a time to prevent RAM exhaustion.
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
            # Create encoder with the SAME parameters and key used during encoding
            enc = create_encoder(
                prompt=request.prompt,
                cipher_len=request.cipher_len,
                max_len=request.max_len,
                shared_private_key=server_key
            )

            print(f"🔓 Decoding: {len(request.stegotext)} tokens")

            # Pass stegotext as-is (list of ints)
            # The decoder will handle the conversion internally
            stegotext = request.stegotext

            # Run decoding in thread pool (CPU-intensive operation)
            loop = asyncio.get_event_loop()
            estimated_plaintext, estimated_bytetext = await loop.run_in_executor(
                None,
                enc.decode,
                stegotext,
                False  # debug=False
            )

            # Strip padding characters
            decoded_plaintext = estimated_plaintext.rstrip('A')

            print(f"✓ Decoding complete: {len(decoded_plaintext)} chars")

            response = {
                "plaintext": decoded_plaintext
            }

            return response

        except Exception as e:
            print(f"❌ Decoding failed: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Decoding failed: {str(e)}"
            )


@app.post(
    "/encode/stream",
    responses={
        503: {
            "model": ErrorResponse,
            "description": "Server is busy processing another request"
        }
    },
    summary="Encode plaintext into stegotext with streaming output"
)
async def encode_stream(request: StreamEncodeRequest):
    """
    Encode a secret message into natural-looking text with TRUE real-time streaming.

    Returns Server-Sent Events (SSE) stream with tokens as they're generated by FIMEC.
    Each event is a JSON object with 'type' field indicating the event type:
    - 'token': A token (or chunk of tokens) as it's generated in real-time
      - With chunk_size=1: Individual tokens streamed immediately
      - With chunk_size>1: Batched tokens for efficiency
    - 'complete': Final message with full formatted stegotext and metadata

    This is TRUE streaming - tokens are yielded as FIMEC generates them,
    not collected first then streamed.

    Only one encoding operation is allowed at a time to prevent RAM exhaustion.
    Returns 503 if another request is being processed.
    """

    # Try to acquire lock (non-blocking)
    if encoding_lock.locked():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server is currently processing another request. Please try again."
        )

    async def event_generator():
        async with encoding_lock:
            try:
                # Use the first cipher_len bytes of the server key
                key_slice = server_key[:request.cipher_len]

                # Create new encoder for this request
                enc = create_encoder(
                    prompt=request.prompt,
                    cipher_len=request.cipher_len,
                    max_len=request.max_len,
                    shared_private_key=key_slice
                )

                print(f"🔐 Streaming encode: {len(request.plaintext)} chars (chunk_size={request.chunk_size})")

                # Create the generator
                stream_gen = enc.encode_stream(
                    plaintext=request.plaintext,
                    chunk_size=request.chunk_size,
                    calculate_failure_probs=request.calculate_failure_probs
                )

                # Yield each chunk as SSE in real-time as they're generated
                for chunk in stream_gen:
                    # Convert to SSE format: "data: {json}\n\n"
                    yield f"data: {json.dumps(chunk)}\n\n"
                    # Allow other async tasks to run between tokens
                    await asyncio.sleep(0)

                print(f"✓ Streaming encode complete")

            except Exception as e:
                print(f"❌ Streaming encode failed: {str(e)}")
                error_event = {
                    'type': 'error',
                    'detail': str(e)
                }
                yield f"data: {json.dumps(error_event)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


if __name__ == "__main__":
    import uvicorn

    print("=" * 70)
    print("TOMATO ENCODER API SERVER")
    print("=" * 70)
    print(f"Model: {DEFAULT_MODEL}")
    print(f"Temperature: {DEFAULT_TEMPERATURE}")
    print("=" * 70)

    uvicorn.run(app, host="0.0.0.0", port=8001)
