import pytest
from tomato import Encoder


def test_encode_decode_roundtrip():
    """Test that encoding and then decoding returns the original plaintext."""
    prompt = """Q: How do you peel an apple?
A: """

    # Create encoder
    encoder = Encoder(model_name='gpt2', prompt=prompt, cipher_len=15)

    # Original plaintext
    plaintext = "affair"

    # Encode the message
    formatted_stegotext, stegotext = encoder.encode(plaintext)

    # Decode the message
    estimated_plaintext, estimated_bytetext = encoder.decode(stegotext)

    # Strip padding (the encoder pads with 'A' characters to reach cipher_len)
    estimated_plaintext_stripped = estimated_plaintext.rstrip('A')

    # Assert that decoded plaintext matches original
    assert estimated_plaintext_stripped == plaintext, (
        f"Decoded plaintext '{estimated_plaintext_stripped}' does not match "
        f"original plaintext '{plaintext}'"
    )

    # Optional: print results for debugging
    print('\nResults:')
    print('Stegotext (covertext):')
    print(formatted_stegotext)
    print('\n------')
    print(f'Original plaintext: {plaintext}')
    print(f'Decoded plaintext: {estimated_plaintext}')
