"""
Example with EARLY patching to capture coupling matrices.
IMPORTANT: Patch BEFORE importing Encoder!
"""

# Step 1: Import logger and patch FIRST (before Encoder import)
from tomato.utils.early_patch_logger import EarlyPatchLogger

logger = EarlyPatchLogger(log_dir="./logs")
logger.patch_now()  # Patch greedy_mec BEFORE importing Encoder

# Step 2: NOW import Encoder (will use patched greedy_mec)
from tomato import Encoder

# Step 3: Set up encoder

prompt = "what were the main gods of the roman empire"
model_name = "google/gemma-3-1b-it"
plaintext = "soft flows the gentle wind"

print(f"plaintext: {plaintext}")
print(f"prompt: {prompt}")
print(f"model: {model_name}")
print()

encoder = Encoder(
    model_name=model_name,
    prompt=prompt,
    cipher_len=30,
    max_len=300,
    temperature=1.3
)


# Step 4: Encode (couplings will be logged!)
print("=" * 70)
print("ENCODING")
print("=" * 70)
formatted_stegotext, stegotext, probs = encoder.encode(plaintext, debug=True)

print(f"\nEncoded:\n{formatted_stegotext}")
print(f"probs: {probs}")

# Save coupling logs for encode
logger.save_logs(
    'couplings_encode.json',
    metadata={
        'operation': 'encode',
        'plaintext': plaintext,
        'cipher_len': 15,
        'max_len': 200,
        'k': 50,
        'model': model_name,
    }
)

# Step 5: Reset coupling counter for decode
encode_count = logger.coupling_count
logger.coupling_count = 0
logger.couplings = []

# Step 6: Decode (couplings will be logged!)
print("\n" + "=" * 70)
print("DECODING")
print("=" * 70)
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
logger.unpatch()
