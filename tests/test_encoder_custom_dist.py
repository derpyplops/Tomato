"""Test that Encoder accepts a custom ModelMarginal instance."""
import pytest
from unittest.mock import Mock, MagicMock, patch
import numpy as np


def test_encoder_uses_provided_covertext_dist():
    """Encoder should use provided covertext_dist instead of creating one."""
    from tomato.encoders.encoder import Encoder
    from tomato.utils.model_marginal import ModelMarginal
    from mec.iterative.marginals import AutoRegressiveMarginal

    # Create a mock that passes FIMEC's isinstance check
    # We need to create a proper subclass mock
    class MockModelMarginal(AutoRegressiveMarginal):
        def __init__(self):
            # Don't call super().__init__() to avoid any setup
            self.max_len = 100
            self.mapping = {}

        def conditional(self, prefix):
            return np.array([0.5, 0.3, 0.2])

        def is_terminal(self, prefix):
            return len(prefix) >= self.max_len

        def decode(self, prefix):
            return "test output"

    mock_dist = MockModelMarginal()

    # Create Encoder with custom dist
    encoder = Encoder(
        cipher_len=15,
        prompt="test",
        covertext_dist=mock_dist
    )

    # Verify it uses our mock, not a new instance
    assert encoder._covertext_dist is mock_dist


def test_encoder_creates_default_when_no_covertext_dist():
    """Encoder should create ModelMarginal when none provided."""
    from tomato.encoders.encoder import Encoder
    from tomato.utils.model_marginal import ModelMarginal

    # This will try to load a real model - skip in CI
    pytest.skip("Requires model download - run manually")

    encoder = Encoder(
        cipher_len=15,
        prompt="test",
        model_name="google/gemma-3-1b-it"
    )

    assert isinstance(encoder._covertext_dist, ModelMarginal)
