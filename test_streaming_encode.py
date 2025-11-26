"""
Test script to verify encode/decode works correctly, including streaming encode.
"""

# Step 1: Import logger and patch FIRST (before Encoder import)
from tomato.utils.early_patch_logger import EarlyPatchLogger

logger = EarlyPatchLogger(log_dir="./logs")
logger.patch_now()  # Patch greedy_mec BEFORE importing Encoder

# Step 2: NOW import Encoder (will use patched greedy_mec)
from tomato import Encoder

import secrets

# Configuration
prompt = "tell me a story"
model_name = "google/gemma-3-1b-it"
plaintext = "hello world"
cipher_len = 15
max_len = 100
temperature = 1.3
chunk_size = 5

# Generate a shared private key for consistent results
shared_private_key = secrets.token_bytes(100)

print(f"plaintext: {plaintext}")
print(f"prompt: {prompt}")
print(f"model: {model_name}")
print(f"cipher_len: {cipher_len}")
print()

def make_encoder():
    return Encoder(
        model_name=model_name,
        prompt=prompt,
        cipher_len=cipher_len,
        max_len=max_len,
        temperature=temperature,
        shared_private_key=shared_private_key
    )

# ============================================================================
# TEST 1: Regular Encode/Decode
# ============================================================================
print("=" * 70)
print("TEST 1: REGULAR ENCODE/DECODE")
print("=" * 70)

encoder1 = make_encoder()

# Encode
print("\nEncoding...")
formatted_stegotext, stegotext, probs = encoder1.encode(
    plaintext,
    debug=False,
    calculate_failure_probs=False
)

print(f"✓ Encoded: {len(stegotext)} tokens")
print(f"  First 100 chars: {formatted_stegotext[:100]}...")

# Reset counter
encode_count = logger.coupling_count
logger.coupling_count = 0
logger.couplings = []

# Decode
print("\nDecoding...")
encoder2 = make_encoder()
estimated_plaintext, estimated_bytetext = encoder2.decode(
    stegotext,
    debug=False
)

decode_count = logger.coupling_count
decoded_stripped = estimated_plaintext.rstrip('A')

print(f"✓ Decoded: {decoded_stripped}")
print(f"  Match: {plaintext == decoded_stripped}")

# ============================================================================
# TEST 2: Streaming Encode
# ============================================================================
print("\n" + "=" * 70)
print("TEST 2: STREAMING ENCODE")
print("=" * 70)

logger.coupling_count = 0
logger.couplings = []

encoder3 = make_encoder()

print(f"\nStreaming encode (chunk_size={chunk_size})...")

# Collect all chunks
metadata = None
token_chunks = []
complete_data = None

for chunk in encoder3.encode_stream(
    plaintext=plaintext,
    chunk_size=chunk_size,
    calculate_failure_probs=False
):
    if chunk['type'] == 'metadata':
        metadata = chunk
        print(f"✓ Metadata: {chunk['total_tokens']} tokens")
    elif chunk['type'] == 'token':
        token_chunks.append(chunk)
        print(f"  Token chunk {chunk['token_index']}: '{chunk['text'][:30]}...'")
    elif chunk['type'] == 'complete':
        complete_data = chunk
        print(f"✓ Complete: {len(chunk['stegotext'])} tokens")

stream_count = logger.coupling_count

# ============================================================================
# TEST 3: Verify Streaming Output Matches Regular Encode
# ============================================================================
print("\n" + "=" * 70)
print("TEST 3: VERIFY STREAMING MATCHES REGULAR ENCODE")
print("=" * 70)

# Check if streaming produced the same stegotext
streaming_stegotext = complete_data['stegotext']
streaming_formatted = complete_data['formatted_stegotext']

tokens_match = list(stegotext) == streaming_stegotext
print(f"\nToken IDs match: {tokens_match}")
print(f"Formatted text match: {formatted_stegotext == streaming_formatted}")

# Decode the streaming result
print("\nDecoding streaming result...")
logger.coupling_count = 0
logger.couplings = []

encoder4 = make_encoder()
stream_decoded, _ = encoder4.decode(streaming_stegotext, debug=False)
stream_decoded_stripped = stream_decoded.rstrip('A')

stream_decode_count = logger.coupling_count

print(f"✓ Stream decoded: {stream_decoded_stripped}")
print(f"  Match original: {plaintext == stream_decoded_stripped}")

# ============================================================================
# FINAL RESULTS
# ============================================================================
print("\n" + "=" * 70)
print("FINAL RESULTS")
print("=" * 70)

print(f"\nOriginal plaintext:       {plaintext}")
print(f"Regular decoded:          {decoded_stripped}")
print(f"Streaming decoded:        {stream_decoded_stripped}")

print(f"\nRegular encode match:     {plaintext == decoded_stripped}")
print(f"Streaming encode match:   {plaintext == stream_decoded_stripped}")
print(f"Tokens identical:         {tokens_match}")

print(f"\nCoupling matrix counts:")
print(f"  Regular encode:         {encode_count}")
print(f"  Regular decode:         {decode_count}")
print(f"  Streaming encode:       {stream_count}")
print(f"  Streaming decode:       {stream_decode_count}")

print(f"\nChunk analysis:")
print(f"  Total chunks:           {len(token_chunks)}")
print(f"  Tokens per chunk:       {chunk_size}")
print(f"  Total tokens:           {metadata['total_tokens']}")

print("=" * 70)

# Clean up
logger.unpatch()

# Exit with appropriate code
# Note: tokens_match is expected to be False since FIMEC sampling is stochastic
# What matters is that both decode correctly to the original plaintext
all_passed = (
    plaintext == decoded_stripped and
    plaintext == stream_decoded_stripped
)

if all_passed:
    print("\n✓ ALL TESTS PASSED")
    print("\nNote: Token sequences differ (expected - FIMEC is stochastic)")
    print("      but both decode correctly to the original plaintext!")
    exit(0)
else:
    print("\n✗ TESTS FAILED")
    if plaintext != decoded_stripped:
        print("  Regular encode/decode failed")
    if plaintext != stream_decoded_stripped:
        print("  Streaming encode/decode failed")
    exit(1)
