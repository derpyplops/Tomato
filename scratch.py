"""
Example with EARLY patching to capture coupling matrices.
IMPORTANT: Patch BEFORE importing Encoder!
"""

# # Step 1: Import logger and patch FIRST (before Encoder import)
# from tomato.utils.early_patch_logger import EarlyPatchLogger

# logger = EarlyPatchLogger(log_dir="./logs")
# logger.patch_now()  # Patch greedy_mec BEFORE importing Encoder

# # Step 2: NOW import Encoder (will use patched greedy_mec)
from test_streaming_encode import formatted_stegotext
from tomato import Encoder

# Step 3: Set up encoder

import torch
import random
import numpy as np
import os

seed = 0
random.seed(seed)                  # Python
np.random.seed(seed)               # NumPy
torch.manual_seed(seed)            # Torch CPU
torch.cuda.manual_seed(seed)       # Current GPU
torch.cuda.manual_seed_all(seed)   # All GPUs (if you have more than one)
torch.use_deterministic_algorithms(True)

prompt = "tell me a story"
model_name = "google/gemma-3-1b-it"
plaintext = "potato"

print(f"plaintext: {plaintext}")
print(f"prompt: {prompt}")
print(f"model: {model_name}")
print()

import secrets

shared_private_key = secrets.token_bytes(100)

secrets.token_bytes(100)

def make_encoder():
    return Encoder(
        model_name=model_name,
        prompt=prompt,
        cipher_len=10,
        max_len=150,
        temperature=1.3,
        shared_private_key=shared_private_key
    )

encoder = make_encoder()

# Step 4: Encode with STREAMING (couplings will be logged!)
print("=" * 70)
print("STREAMING ENCODE - Watch tokens appear in real-time!")
print("=" * 70)
print()

stegotext, formatted_stegotext = encoder.encode(plaintext)

# stegotext = []
# formatted_stegotext = ""
# probs = {}

# for chunk in encoder.encode_stream(plaintext, chunk_size=1, calculate_failure_probs=False):
#     if chunk['type'] == 'token':
#         # Accumulate tokens and decode ALL tokens so far for proper text
#         stegotext.append(chunk['token_id'])
#         text = encoder._covertext_dist.decode(stegotext)
#         # Update the line in-place (carriage return clears line)
#         print(f"\r{text}", end='', flush=True)
#     elif chunk['type'] == 'complete':
#         formatted_stegotext = chunk['formatted_stegotext']
#         stegotext = chunk['stegotext']
#         probs = chunk['failure_probs']
#         print()  # Final newline after streaming completes

print(f"\n\nFormatted version:\n{formatted_stegotext}")
# print(f"Failure probs: {probs}")

# # Save coupling logs for encode
# logger.save_logs(
#     'couplings_encode.json',
#     metadata={
#         'operation': 'encode',
#         'plaintext': plaintext,
#         'cipher_len': 15,
#         'max_len': 200,
#         'k': 50,
#         'model': model_name,
#     }
# )

# # Step 5: Reset coupling counter for decode
# encode_count = logger.coupling_count
# logger.coupling_count = 0
# logger.couplings = []

# Step 6: Decode (couplings will be logged!)
print("\n" + "=" * 70)
print("DECODING")
print("=" * 70)

encoder = make_encoder()

estimated_plaintext, estimated_bytetext = encoder.decode(
    stegotext,
    debug=True,
    true_plaintext=plaintext
)

print(f"\nDecoded: {estimated_plaintext}")

# Save coupling logs for decode
logger.save_logs(
    'couplings_decode.json',
    metadata={
        'operation': 'decode',
        'true_plaintext': plaintext,
        'decoded_plaintext': estimated_plaintext,
        'cipher_len': 15,
        'max_len': 100,
    }
)

# Step 7: Show comparison
print(f"\n{'='*70}")
print("FINAL RESULTS")
print(f"{'='*70}")
print(f"Original:           {plaintext}")
print(f"Decoded:            {estimated_plaintext}")
print(f"Match:              {plaintext == estimated_plaintext.rstrip('A')}")
print(f"\nEncoding matrices:  {encode_count}")
print(f"Decoding matrices:  {logger.coupling_count}")
print(f"\nUsage (encode):     {(encode_count/100)*100:.1f}% ({encode_count}/100 tokens)")
print(f"Usage (decode):     {(logger.coupling_count/100)*100:.1f}% ({logger.coupling_count}/100 tokens)")
print(f"\nCheck ./logs/ directory for detailed coupling matrix logs!")
print(f"{'='*70}")

# Clean up
# logger.unpatch()
