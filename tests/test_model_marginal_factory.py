"""Test ModelMarginal.with_custom_model factory method."""
import pytest
from unittest.mock import Mock
import numpy as np


def test_with_custom_model_creates_instance():
    """Factory should create a properly configured ModelMarginal."""
    from tomato.utils.model_marginal import ModelMarginal

    # Create a mock model with required interface
    mock_model = Mock()
    mock_model.vocab_size = 32000
    mock_model.tokenizer = Mock()
    mock_model.top_k_conditional = Mock(return_value=np.array([0.5, 0.3, 0.2]))

    # Create instance via factory
    dist = ModelMarginal.with_custom_model(
        model=mock_model,
        prompt="Test prompt",
        max_len=100,
        temperature=1.3,
        k=50
    )

    # Verify attributes are set correctly
    assert dist.lm_model is mock_model
    assert dist.prompt == "Test prompt\n"  # Note: newline added
    assert dist.max_len == 100
    assert dist.temperature == 1.3
    assert dist.k == 50
    assert dist.branching_factor == 50
    assert dist.mapping == {}


def test_with_custom_model_vs_init_parity():
    """Factory instance should behave the same as __init__ instance."""
    from tomato.utils.model_marginal import ModelMarginal

    # Create mock model
    mock_model = Mock()
    mock_model.vocab_size = 32000
    mock_model.tokenizer = Mock()

    # Create via factory
    factory_dist = ModelMarginal.with_custom_model(
        model=mock_model,
        prompt="Test",
        max_len=100,
        temperature=1.0,
        k=50
    )

    # Verify all expected attributes exist
    assert hasattr(factory_dist, 'conditional')
    assert hasattr(factory_dist, 'is_terminal')
    assert hasattr(factory_dist, 'decode')
    assert hasattr(factory_dist, 'sample')
