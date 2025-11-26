"""
MEC Steganography with TensorRT-LLM on Modal

Minimal implementation that reuses existing Encoder/ModelMarginal code,
only swapping the model wrapper for TRT-LLM.

Usage:
    modal run modal_stego_app.py
"""

import secrets
import time
from pathlib import Path

import modal

# Key file configuration (matching server.py)
KEY_FILE = ".server_key"
KEY_LENGTH = 100

# Modal Image with TRT-LLM
tensorrt_image = modal.Image.from_registry(
    "nvidia/cuda:12.8.1-devel-ubuntu22.04",
    add_python="3.12",
).entrypoint([])

tensorrt_image = tensorrt_image.apt_install(
    "openmpi-bin", "libopenmpi-dev", "git", "git-lfs", "wget"
).pip_install(
    "tensorrt-llm==0.18.0",
    "pynvml<12",
    "flashinfer-python==0.2.5",
    "cuda-python==12.9.1",
    "onnx==1.19.1",
    "mec @ git+https://github.com/user1342/mec",
    "transformers",
    "huggingface_hub==0.36.0",
    pre=True,
    extra_index_url="https://pypi.nvidia.com",
)

# Volume for caching
volume = modal.Volume.from_name("stego-trtllm-volume", create_if_missing=True)
VOLUME_PATH = Path("/vol")
MODELS_PATH = VOLUME_PATH / "models"

MODEL_ID = "NousResearch/Meta-Llama-3-8B-Instruct"

tensorrt_image = tensorrt_image.env({
    "HF_HOME": str(MODELS_PATH),
})

# Add local tomato package
tensorrt_image = tensorrt_image.add_local_python_source("tomato")

app = modal.App("stego-trtllm")


def load_or_create_key() -> bytes:
    """Load existing key from file or create a new one (matching server.py)"""
    key_path = Path(KEY_FILE)

    if key_path.exists():
        print(f"Loading existing key from {KEY_FILE}")
        with open(key_path, 'rb') as f:
            key = f.read()
        if len(key) != KEY_LENGTH:
            raise ValueError(f"Invalid key length in {KEY_FILE}: expected {KEY_LENGTH}, got {len(key)}")
        print(f"Loaded {len(key)}-byte key")
        return key
    else:
        print(f"Generating new {KEY_LENGTH}-byte key")
        key = secrets.token_bytes(KEY_LENGTH)
        with open(key_path, 'wb') as f:
            f.write(key)
        print(f"Saved key to {KEY_FILE}")
        return key


def get_build_config():
    from tensorrt_llm import BuildConfig
    from tensorrt_llm.plugin.plugin import PluginConfig
    return BuildConfig(
        plugin_config=PluginConfig.from_dict({
            "multiple_profiles": True,
            "paged_kv_cache": True,
        }),
        max_input_len=2048,
        max_num_tokens=4096,
        max_batch_size=1,
        gather_context_logits=True,  # Get logits BEFORE generation
    )


@app.cls(
    image=tensorrt_image,
    gpu="H100",
    timeout=15 * 60,
    volumes={VOLUME_PATH: volume},
)
class StegoModel:
    """Steganography encoder using TRT-LLM with existing MEC code."""

    @modal.enter()
    def setup(self):
        import os
        import shutil
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer
        from tensorrt_llm import LLM

        self.model_path = MODELS_PATH / MODEL_ID

        print(f"Downloading {MODEL_ID}...")
        snapshot_download(
            MODEL_ID,
            local_dir=self.model_path,
            ignore_patterns=["*.pt", "*.bin"],
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        engine_kwargs = {
            "build_config": get_build_config(),
            "tensor_parallel_size": torch.cuda.device_count(),
        }

        engine_path = self.model_path / "trtllm_engine_v3"  # New path for context logits engine
        if not os.path.exists(engine_path):
            print(f"Building engine at {engine_path}...")
            self.llm = LLM(model=str(self.model_path), **engine_kwargs)
            self.llm.save(str(engine_path))
        else:
            print(f"Loading engine from {engine_path}...")
            self.llm = LLM(model=str(engine_path), **engine_kwargs)

        print("Setup complete")

    @modal.method()
    def encode_decode_test(
        self,
        plaintext: str = "hello",
        prompt: str = "Write a short story:",
        cipher_len: int = 15,
        max_len: int = 200,  # Match server.py
        temperature: float = 1.3,
        k: int = 50,
        shared_private_key: bytes = None,
    ) -> dict:
        """
        Test encode/decode roundtrip using existing MEC code with TRT-LLM backend.
        """
        import numpy as np
        from mec import FIMEC

        # Import and patch ModelMarginal to use TRT-LLM
        from tomato.utils.model_marginal import ModelMarginal
        from tomato.utils.trtllm_model_wrapper import TRTLLMModelWrapper

        # Create TRT-LLM wrapper
        trtllm_wrapper = TRTLLMModelWrapper(self.llm, self.tokenizer)

        # Create ModelMarginal but replace the lm_model
        covertext_dist = ModelMarginal.__new__(ModelMarginal)
        covertext_dist.max_len = max_len
        covertext_dist.temperature = temperature
        covertext_dist.k = k
        covertext_dist.branching_factor = k
        covertext_dist.lm_model = trtllm_wrapper  # Inject TRT-LLM wrapper
        covertext_dist.prompt = f"{prompt}\n"
        covertext_dist.mapping = {}

        # Ciphertext distribution (uniform over bytes)
        from tomato.utils.random_string import RandomString
        ciphertext_dist = RandomString(num_chars=256, string_len=cipher_len)

        # Create FIMEC
        imec = FIMEC(ciphertext_dist, covertext_dist)

        # Use shared key (truncated to cipher_len for XOR)
        if shared_private_key is None:
            shared_private_key = secrets.token_bytes(KEY_LENGTH)
        shared_key = shared_private_key[:cipher_len]

        # Prepare plaintext
        bytetext = plaintext.encode("utf-8")
        if len(bytetext) < cipher_len:
            bytetext += b'A' * (cipher_len - len(bytetext))
        bytetext = bytetext[:cipher_len]

        # XOR encrypt
        ciphertext = [a ^ b for a, b in zip(bytetext, shared_key)]

        # Encode
        print(f"Encoding '{plaintext}'...")
        start = time.perf_counter()
        stegotext_tokens, log_prob = imec.sample_y_given_x(ciphertext)
        encode_time = time.perf_counter() - start

        stegotext = covertext_dist.decode(list(stegotext_tokens))
        print(f"Stegotext: {stegotext[:100]}...")

        # Decode
        print("Decoding...")
        start = time.perf_counter()
        estimated_ciphertext, _ = imec.estimate_x_given_y(list(stegotext_tokens))
        decode_time = time.perf_counter() - start

        # XOR decrypt
        estimated_bytetext = bytes([a ^ b for a, b in zip(estimated_ciphertext, shared_key)])
        decoded_plaintext = estimated_bytetext.decode("utf-8", errors="replace")

        # Check match
        decoded_stripped = decoded_plaintext.rstrip("A")
        match = decoded_stripped == plaintext

        return {
            "plaintext": plaintext,
            "decoded": decoded_plaintext,
            "decoded_stripped": decoded_stripped,
            "match": match,
            "stegotext": stegotext,
            "encode_time_ms": encode_time * 1000,
            "decode_time_ms": decode_time * 1000,
        }

    @modal.exit()
    def shutdown(self):
        if hasattr(self, 'llm'):
            self.llm.shutdown()


@app.local_entrypoint()
def main(
    plaintext: str = "hello",
    prompt: str = "Write a short story about a magical forest:",
):
    """Test steganography encode/decode."""
    # Load or create shared key (matching server.py)
    shared_key = load_or_create_key()
    print(f"Using shared key (first 8 bytes: {shared_key[:8].hex()}...)")

    print(f"Testing with plaintext='{plaintext}'")
    print(f"Prompt: {prompt}")
    print("-" * 50)

    model = StegoModel()
    result = model.encode_decode_test.remote(
        plaintext=plaintext,
        prompt=prompt,
        shared_private_key=shared_key,
    )

    print(f"\nResults:")
    print(f"  Original:  '{result['plaintext']}'")
    print(f"  Decoded:   '{result['decoded_stripped']}'")
    print(f"  Match:     {result['match']}")
    print(f"  Encode:    {result['encode_time_ms']:.1f}ms")
    print(f"  Decode:    {result['decode_time_ms']:.1f}ms")
    print(f"\nStegotext preview:")
    print(f"  {result['stegotext'][:200]}...")
