from tomato import Encoder

prompt = "Q: How do you peel an apple?\nA: "

model_name = "google/gemma-3-1b-it"
encoder = Encoder(
    model_name=model_name,
    prompt=prompt,
    cipher_len=15
)

plaintext = "password"

print(f"plaintext: {plaintext}")
print(f"prompt: {prompt}")
print(f"model: {model_name}")
print()

# Encode with debug logging
formatted_stegotext, stegotext = encoder.encode(plaintext, debug=True)

print(f"\nencoded:\n{formatted_stegotext}")
print()

# Decode with debug logging and accuracy tracking
estimated_plaintext, estimated_bytetext = encoder.decode(
    stegotext,
    debug=True,
    true_plaintext=plaintext
)

print(f"\ndecoded:\n{estimated_plaintext}")

# Show comparison
print(f"\n{'='*70}")
print("COMPARISON")
print(f"{'='*70}")
print(f"Original:  {plaintext}")
print(f"Decoded:   {estimated_plaintext}")
print(f"Match:     {plaintext == estimated_plaintext.rstrip('A')}")
print(f"\nCheck ./logs/ directory for detailed debug logs!")
