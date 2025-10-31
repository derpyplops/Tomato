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
        cipher_len=40,
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

def _process_testcase(testcase):
    """Process a single test case in a separate process."""
    # Create encoder INSIDE the process (can't pickle/share models across processes)
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

    # Check if successful
    success = estimated_plaintext_stripped == plaintext

    # Return result
    return {
        'testcase': testcase,
        'success': success,
        'original': plaintext,
        'decoded': estimated_plaintext,
        'stegotext': formatted_stegotext,
        'error': None if success else f"Decoded '{estimated_plaintext_stripped}' != '{plaintext}'"
    }

def test_encode_decode_roundtrip():
    import multiprocessing as mp

    # Run testcases in parallel using multiprocessing
    # Note: 'spawn' method is more reliable on macOS
    with mp.get_context('spawn').Pool(processes=len(testcases)) as pool:
        results = pool.map(_process_testcase, testcases)

    # Check all results and print
    for result in results:
        print('\nResults:')
        print(f'cipher_len={result["testcase"].cipher_len}')
        print('Stegotext (covertext):')
        print(result['stegotext'])
        print('\n------')
        print(f'Original plaintext: {result["original"]}')
        print(f'Decoded plaintext: {result["decoded"]}')

        # Assert after all processing is done
        assert result['success'], result['error']
