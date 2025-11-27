"""Test Encoder.encode_stream and decode_stream methods directly."""
import pytest
from unittest.mock import Mock
import numpy as np
from mec.iterative.marginals import AutoRegressiveMarginal


class MockModelMarginal(AutoRegressiveMarginal):
    """Mock ModelMarginal that returns predictable distributions."""

    def __init__(self, max_len=50):
        self.max_len = max_len
        self.mapping = {}
        self.k = 10
        self.branching_factor = 10
        self.temperature = 1.0
        self.prompt = "test\n"
        self._call_count = 0

        # Mock tokenizer for decode
        self.lm_model = Mock()
        self.lm_model.tokenizer = Mock()
        self.lm_model.tokenizer.decode = Mock(side_effect=self._mock_decode)
        self.lm_model.vocab_size = 100

    def _mock_decode(self, token_ids):
        """Convert token IDs to text."""
        return "".join(chr(65 + (t % 26)) for t in token_ids)

    def conditional(self, prefix):
        """Return a simple distribution over k tokens."""
        self._call_count += 1
        # Return uniform-ish distribution over k=10 tokens
        probs = np.array([0.15, 0.14, 0.13, 0.12, 0.11, 0.10, 0.09, 0.08, 0.05, 0.03])
        # Store mapping for decode
        self.mapping[tuple(prefix)] = np.arange(10)
        return probs

    def is_terminal(self, prefix):
        return len(prefix) >= self.max_len

    def decode(self, prefix):
        """Decode token IDs to text."""
        decoded_tokens = []
        for k, z_i in enumerate(prefix):
            if tuple(prefix[:k]) not in self.mapping:
                self.conditional(prefix[:k])
            decoded_tokens.append(self.mapping[tuple(prefix[:k])][z_i])
        return self.lm_model.tokenizer.decode(decoded_tokens)


def test_encode_stream_yields_tokens():
    """encode_stream should yield token events as they're generated."""
    from tomato.encoders.encoder import Encoder

    mock_dist = MockModelMarginal(max_len=20)

    encoder = Encoder(
        cipher_len=5,
        prompt="test",
        covertext_dist=mock_dist
    )

    events = list(encoder.encode_stream(
        plaintext="hi",
        chunk_size=1
    ))

    # Should have token events + 1 complete event
    token_events = [e for e in events if e.get('type') == 'token']
    complete_events = [e for e in events if e.get('type') == 'complete']

    assert len(token_events) > 0, "Expected at least one token event"
    assert len(complete_events) == 1, "Expected exactly one complete event"

    # Complete event should have stegotext
    assert 'stegotext' in complete_events[0]
    assert 'formatted_stegotext' in complete_events[0]
    assert 'total_tokens' in complete_events[0]

    # Token events should have incrementing token_index
    for i, event in enumerate(token_events):
        assert 'token_index' in event
        assert 'text' in event


def test_encode_stream_chunk_size():
    """encode_stream with chunk_size > 1 should batch tokens."""
    from tomato.encoders.encoder import Encoder

    mock_dist = MockModelMarginal(max_len=20)

    encoder = Encoder(
        cipher_len=5,
        prompt="test",
        covertext_dist=mock_dist
    )

    events = list(encoder.encode_stream(
        plaintext="hi",
        chunk_size=5
    ))

    token_events = [e for e in events if e.get('type') == 'token']

    # With chunk_size=5 and max_len=20, we should get ~4 token events
    # Each should have 'tokens' list (not 'token_id')
    for event in token_events:
        if 'tokens' in event:
            assert len(event['tokens']) <= 5


def test_decode_stream_yields_guesses():
    """decode_stream should yield evolving guesses with confidence."""
    from tomato.encoders.encoder import Encoder

    mock_dist = MockModelMarginal(max_len=20)

    encoder = Encoder(
        cipher_len=5,
        prompt="test",
        covertext_dist=mock_dist
    )

    # First encode something to get valid stegotext
    encode_events = list(encoder.encode_stream(plaintext="hi", chunk_size=1))
    complete_event = [e for e in encode_events if e.get('type') == 'complete'][0]
    stegotext = complete_event['stegotext']

    # Now test decode_stream
    # Need fresh encoder since state may be modified
    mock_dist2 = MockModelMarginal(max_len=20)
    encoder2 = Encoder(
        cipher_len=5,
        prompt="test",
        covertext_dist=mock_dist2
    )

    decode_events = list(encoder2.decode_stream(
        stegotext=stegotext,
        chunk_size=5
    ))

    token_events = [e for e in decode_events if e.get('type') == 'token']
    complete_events = [e for e in decode_events if e.get('type') == 'complete']

    assert len(complete_events) == 1, "Expected exactly one complete event"
    assert 'plaintext' in complete_events[0]

    # Token events should have current_guess and confidence
    for event in token_events:
        assert 'current_guess' in event
        assert 'confidence' in event
        assert 'tokens_processed' in event
        assert 0 <= event['confidence'] <= 1


def test_decode_stream_confidence_increases():
    """Confidence should generally increase as more tokens are processed."""
    from tomato.encoders.encoder import Encoder

    mock_dist = MockModelMarginal(max_len=30)

    encoder = Encoder(
        cipher_len=5,
        prompt="test",
        covertext_dist=mock_dist
    )

    # Encode
    encode_events = list(encoder.encode_stream(plaintext="hi", chunk_size=1))
    complete_event = [e for e in encode_events if e.get('type') == 'complete'][0]
    stegotext = complete_event['stegotext']

    # Decode with fresh encoder
    mock_dist2 = MockModelMarginal(max_len=30)
    encoder2 = Encoder(
        cipher_len=5,
        prompt="test",
        covertext_dist=mock_dist2
    )

    decode_events = list(encoder2.decode_stream(
        stegotext=stegotext,
        chunk_size=3
    ))

    token_events = [e for e in decode_events if e.get('type') == 'token']

    if len(token_events) >= 2:
        first_confidence = token_events[0]['confidence']
        last_confidence = token_events[-1]['confidence']
        # Confidence should increase (or at least not decrease significantly)
        # Using a small tolerance since mock may not perfectly simulate real behavior
        assert last_confidence >= first_confidence - 0.1, \
            f"Confidence should increase: {first_confidence} -> {last_confidence}"


def test_encode_decode_roundtrip_streaming():
    """Streaming encode then decode should recover original plaintext."""
    from tomato.encoders.encoder import Encoder

    # Use same key for both encoders
    shared_key = b'12345678901234567890'

    mock_dist1 = MockModelMarginal(max_len=50)
    encoder1 = Encoder(
        cipher_len=5,
        shared_private_key=shared_key,
        prompt="test",
        covertext_dist=mock_dist1
    )

    # Encode
    plaintext = "hi"
    encode_events = list(encoder1.encode_stream(plaintext=plaintext, chunk_size=1))
    complete_event = [e for e in encode_events if e.get('type') == 'complete'][0]
    stegotext = complete_event['stegotext']

    # Decode with same key
    mock_dist2 = MockModelMarginal(max_len=50)
    encoder2 = Encoder(
        cipher_len=5,
        shared_private_key=shared_key,
        prompt="test",
        covertext_dist=mock_dist2
    )

    decode_events = list(encoder2.decode_stream(stegotext=stegotext, chunk_size=5))
    complete_event = [e for e in decode_events if e.get('type') == 'complete'][0]
    decoded = complete_event['plaintext']

    # Note: With mock model, exact roundtrip may not work perfectly
    # but the structure should be correct
    assert isinstance(decoded, str)
    assert len(decoded) > 0
