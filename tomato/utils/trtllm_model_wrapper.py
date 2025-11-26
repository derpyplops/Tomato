"""
TensorRT-LLM Model Wrapper

Drop-in replacement for ModelWrapper that uses TRT-LLM instead of HuggingFace.
"""

from typing import List, Optional
import numpy as np
import torch
import torch.nn as nn


class TRTLLMModelWrapper:
    """
    A wrapper class for TRT-LLM that provides the same interface as ModelWrapper.

    This allows the existing ModelMarginal and Encoder classes to work with
    TRT-LLM without modification.
    """

    def __init__(self, llm, tokenizer) -> None:
        """
        Initialize with a pre-built TRT-LLM engine and tokenizer.

        Args:
            llm: TensorRT-LLM LLM instance
            tokenizer: HuggingFace tokenizer
        """
        self.llm = llm
        self.tokenizer = tokenizer
        self.vocab_size = len(tokenizer)
        self.device = "cuda"

    def conditional(self, text: str, temperature: float) -> torch.Tensor:
        """
        Computes the conditional distribution given the input text.

        Uses context_logits to get the probability distribution for the NEXT token
        given the input text prefix. This is deterministic and suitable for
        steganography encode/decode.

        Args:
            text: The input text to condition on.
            temperature: Temperature for sampling.

        Returns:
            torch.Tensor: The conditional distribution over the vocabulary.
        """
        from tensorrt_llm import SamplingParams

        sampling_params = SamplingParams(
            temperature=1.0,
            max_tokens=1,
            return_context_logits=True,  # Get logits BEFORE generation
        )

        # Generate to trigger context processing and get context_logits
        outputs = self.llm.generate(text, sampling_params)

        # Handle both single RequestOutput and list of outputs
        if outputs is not None:
            # If it's a list, get the first element
            if isinstance(outputs, list):
                output = outputs[0] if len(outputs) > 0 else None
            else:
                output = outputs

            if output is not None:
                # Access context_logits - the logits at the end of context (for next token)
                if hasattr(output, 'context_logits') and output.context_logits is not None:
                    # context_logits shape: [context_len, vocab_size]
                    # We want the LAST row (logits for next token prediction)
                    logits = output.context_logits[-1].float()
                    if not isinstance(logits, torch.Tensor):
                        logits = torch.tensor(logits, device='cuda').float()
                    elif logits.device.type != 'cuda':
                        logits = logits.to('cuda')
                    return nn.Softmax(dim=-1)(logits / temperature)

        # Fallback: uniform distribution
        print("Warning: Could not get logits, using uniform distribution")
        return torch.ones(self.vocab_size, device='cuda') / self.vocab_size

    def top_k_conditional(self, text: str, temperature: float, k: int) -> np.ndarray:
        """
        Computes the top-k conditional distribution given the input text.

        Args:
            text: The input text to condition on.
            temperature: Temperature for sampling.
            k: The number of top elements to consider.

        Returns:
            np.ndarray: The top-k conditional distribution.
        """
        conditional = self.conditional(text, temperature)
        kth_value = torch.topk(conditional, k).values.flatten()[-1]
        conditional[conditional < kth_value] = 0
        conditional /= conditional.sum()
        return conditional.cpu().numpy().reshape(-1)

    def reduced_ids(self, prompt: str, text: str, k: int) -> Optional[List[int]]:
        """
        Computes token IDs using indexing post top-k reduction.

        Note: This is primarily used for encoding existing text, which is
        less common in steganography (we usually generate new text).
        """
        from tensorrt_llm import SamplingParams

        prompt_tokens = self.tokenizer(prompt)["input_ids"]
        text_tokens = self.tokenizer(text)["input_ids"]
        reduced_tokens: List[int] = []

        sampling_params = SamplingParams(
            temperature=1.0,
            max_tokens=1,
            return_context_logits=True,
        )

        for i, token in enumerate(text_tokens):
            prefix_text = self.tokenizer.decode(prompt_tokens + text_tokens[:i])

            # Get logits for this prefix
            outputs = self.llm.generate(prefix_text, sampling_params)

            # Handle both single RequestOutput and list of outputs
            if outputs is None:
                return None
            if isinstance(outputs, list):
                output = outputs[0] if len(outputs) > 0 else None
            else:
                output = outputs

            if output is None or not hasattr(output, 'context_logits') or output.context_logits is None:
                return None

            # Get last context logit (for next token prediction)
            logits = output.context_logits[-1]
            if not isinstance(logits, torch.Tensor):
                logits = torch.tensor(logits, device='cuda')
            top_k_vals, top_k_indices = torch.topk(logits, k)

            if token in top_k_indices:
                correct_idx = (top_k_indices == token).nonzero(as_tuple=True)[0]
                if correct_idx.numel() == 1:
                    reduced_tokens.append(correct_idx.item())
                else:
                    return None
            else:
                return None

        return reduced_tokens
