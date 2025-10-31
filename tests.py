import pytest
from tomato import Encoder
from dataclasses import dataclass

@dataclass
class TestCase:
    """Dataclass to hold information about encoder test cases."""
    model_name: str
    prompt: str
    cipher_len: int
    plaintext: str
    

testcases = [
    TestCase(
        model_name='gpt2',
        prompt="Q: How do you peel an apple?\nA: ",
        cipher_len=15,
        plaintext="affair"
    ),
    TestCase(
        model_name='gpt2',
        prompt="Q: How do you peel an apple?\nA: ",
        cipher_len=100,
        plaintext="affair"
    ),

    # TestCase(
    #     model_name='gpt2',
    #     prompt="Hello! What brings you here today?",
    #     cipher_len=20,
    #     plaintext="secret message"
    # ),
    # TestCase(
    #     model_name='gpt2',
    #     prompt="The quick brown fox jumps over the lazy dog.",
    #     cipher_len=10,
    #     plaintext="hidden"
    # ),
]

#     """Test that encoding and then decoding returns the original plaintext."""
#     prompt = """Q: How do you peel an apple?
# A: """

def test_encode_decode_roundtrip():


    for testcase in testcases:
        # Create encoder
        encoder = Encoder(
            model_name=testcase.model_name,
            prompt=testcase.prompt,
            cipher_len=testcase.cipher_len
        )

        # Original plaintext
        plaintext = testcase.plaintext

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
