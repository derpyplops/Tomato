"""
MEC Steganography Server with TensorRT-LLM on Modal

FastAPI server exposing encode/decode endpoints, deployed on Modal with H100 GPU.

Usage:
    # Deploy the server
    modal deploy modal_server.py

    # Test locally (creates a temporary endpoint)
    modal serve modal_server.py
"""

import secrets
import time
from pathlib import Path

import modal

# Configuration
KEY_LENGTH = 100
CIPHER_LEN = 15
MAX_LEN = 200
DEFAULT_TEMPERATURE = 1.3
DEFAULT_K = 50

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
    "fastapi[standard]",
    pre=True,
    extra_index_url="https://pypi.nvidia.com",
)

# Volume for model caching and key storage
volume = modal.Volume.from_name("stego-trtllm-volume", create_if_missing=True)
VOLUME_PATH = Path("/vol")
MODELS_PATH = VOLUME_PATH / "models"
KEY_PATH = VOLUME_PATH / "server_key"

MODEL_ID = "NousResearch/Meta-Llama-3-8B-Instruct"

tensorrt_image = tensorrt_image.env({
    "HF_HOME": str(MODELS_PATH),
})

# Add local tomato package
tensorrt_image = tensorrt_image.add_local_python_source("tomato")

app = modal.App("stego-server")


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
        gather_context_logits=True,
    )


@app.cls(
    image=tensorrt_image,
    gpu="H100",
    timeout=5 * 60,  # 5 minute request timeout
    volumes={VOLUME_PATH: volume},
    container_idle_timeout=5 * 60,  # Keep warm for 5 minutes
)
class StegoServer:
    """Steganography server using TRT-LLM."""

    @modal.enter()
    def setup(self):
        import os
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer
        from tensorrt_llm import LLM

        # Load or create server key (persisted in volume)
        if KEY_PATH.exists():
            print(f"Loading existing key from volume")
            with open(KEY_PATH, 'rb') as f:
                self.server_key = f.read()
        else:
            print(f"Generating new {KEY_LENGTH}-byte key")
            self.server_key = secrets.token_bytes(KEY_LENGTH)
            KEY_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(KEY_PATH, 'wb') as f:
                f.write(self.server_key)
            volume.commit()  # Persist the key

        print(f"Server key ready (first 8 bytes: {self.server_key[:8].hex()}...)")

        # Download model
        self.model_path = MODELS_PATH / MODEL_ID
        print(f"Downloading {MODEL_ID}...")
        snapshot_download(
            MODEL_ID,
            local_dir=self.model_path,
            ignore_patterns=["*.pt", "*.bin"],
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        # Load or build TRT-LLM engine
        engine_kwargs = {
            "build_config": get_build_config(),
            "tensor_parallel_size": torch.cuda.device_count(),
        }

        engine_path = self.model_path / "trtllm_engine_v3"
        if not os.path.exists(engine_path):
            print(f"Building engine at {engine_path}...")
            self.llm = LLM(model=str(self.model_path), **engine_kwargs)
            self.llm.save(str(engine_path))
            volume.commit()  # Persist the engine
        else:
            print(f"Loading engine from {engine_path}...")
            self.llm = LLM(model=str(engine_path), **engine_kwargs)

        # Create reusable wrapper
        from tomato.utils.trtllm_model_wrapper import TRTLLMModelWrapper
        self.trtllm_wrapper = TRTLLMModelWrapper(self.llm, self.tokenizer)

        print("Server ready!")

    def _create_encoder(self, prompt: str):
        """Create an Encoder instance for a request with the given prompt."""
        from tomato.encoders.encoder import Encoder
        from tomato.utils.model_marginal import ModelMarginal

        # Create a fresh ModelMarginal with the request's prompt using factory
        covertext_dist = ModelMarginal.with_custom_model(
            model=self.trtllm_wrapper,
            prompt=prompt,
            max_len=MAX_LEN,
            temperature=DEFAULT_TEMPERATURE,
            k=DEFAULT_K
        )

        # Create Encoder with our TRT-LLM-backed ModelMarginal
        return Encoder(
            cipher_len=CIPHER_LEN,
            shared_private_key=self.server_key,
            prompt=prompt,
            max_len=MAX_LEN,
            temperature=DEFAULT_TEMPERATURE,
            k=DEFAULT_K,
            covertext_dist=covertext_dist
        )

    @modal.web_endpoint(method="GET", docs=True)
    def health(self) -> dict:
        """Health check endpoint."""
        return {
            "status": "healthy",
            "model": MODEL_ID,
            "gpu": "H100"
        }

    @modal.web_endpoint(method="POST", docs=True)
    def encode(self, request: dict) -> dict:
        """
        Encode a secret message into natural-looking text.

        Request body: {"plaintext": "secret message", "prompt": "Write a story:"}

        The plaintext is XOR-encrypted with the server's shared key,
        then encoded into stegotext using minimum entropy coupling.
        """
        plaintext = request["plaintext"]
        prompt = request["prompt"]

        print(f"Encoding {len(plaintext)} chars...")
        start = time.perf_counter()

        encoder = self._create_encoder(prompt)
        formatted_stegotext, stegotext, _ = encoder.encode(
            plaintext,
            debug=False,
            calculate_failure_probs=False
        )

        elapsed = time.perf_counter() - start
        print(f"Encoded in {elapsed:.1f}s: {formatted_stegotext[:50]}...")

        return {
            "stegotext": [int(t) for t in stegotext],
            "formatted_stegotext": formatted_stegotext
        }

    @modal.web_endpoint(method="POST", docs=True)
    def decode(self, request: dict) -> dict:
        """
        Decode stegotext back to the original secret message.

        Request body: {"stegotext": [1, 2, 3, ...], "prompt": "Write a story:"}

        Requires the same prompt used during encoding.
        """
        stegotext = request["stegotext"]
        prompt = request["prompt"]

        print(f"Decoding {len(stegotext)} tokens...")
        start = time.perf_counter()

        encoder = self._create_encoder(prompt)
        estimated_plaintext, _ = encoder.decode(stegotext, debug=False)

        # Strip padding
        decoded_stripped = estimated_plaintext.rstrip("A")

        elapsed = time.perf_counter() - start
        print(f"Decoded in {elapsed:.1f}s: '{decoded_stripped}'")

        return {"plaintext": decoded_stripped}

    @modal.exit()
    def shutdown(self):
        if hasattr(self, 'llm'):
            self.llm.shutdown()


# For local testing
@app.local_entrypoint()
def test():
    """Test the server endpoints locally."""
    server = StegoServer()

    # Test health
    health = server.health.remote()
    print(f"Health: {health}")

    # Test encode
    encode_resp = server.encode.remote(request={
        "plaintext": "hello",
        "prompt": "Write a short story about a magical forest:"
    })
    print(f"\nEncoded:")
    print(f"  Stegotext: {encode_resp['formatted_stegotext'][:100]}...")
    print(f"  Tokens: {len(encode_resp['stegotext'])}")

    # Test decode
    decode_resp = server.decode.remote(request={
        "stegotext": encode_resp['stegotext'],
        "prompt": "Write a short story about a magical forest:"
    })
    print(f"\nDecoded: '{decode_resp['plaintext']}'")
    print(f"Match: {decode_resp['plaintext'] == 'hello'}")
