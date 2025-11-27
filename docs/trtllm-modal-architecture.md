# MEC Steganography with TRT-LLM on Modal - Implementation Plan

## Overview

Integrate the existing FIMEC-based steganography system with TensorRT-LLM for high-performance inference, deployed on Modal.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Modal Container (H100 GPU)                                  │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  TRT-LLM Engine (LLaMA 3 8B FP8)                       │ │
│  │  - Quantized model with FP8                            │ │
│  │  - Paged KV cache                                      │ │
│  └───────────────────────┬────────────────────────────────┘ │
│                          │ logits (via LogitsProcessor)     │
│  ┌───────────────────────▼────────────────────────────────┐ │
│  │  MECLogitsProcessor                                    │ │
│  │  - Receives full logits tensor                         │ │
│  │  - Applies top-k filtering                             │ │
│  │  - Uses greedy_mec coupling matrix                     │ │
│  │  - Sets logits to force token selection                │ │
│  └───────────────────────┬────────────────────────────────┘ │
│                          │ selected token                   │
│  ┌───────────────────────▼────────────────────────────────┐ │
│  │  StegoEncoder (Modal endpoint)                         │ │
│  │  - encode(plaintext) -> stegotext                      │ │
│  │  - decode(stegotext) -> plaintext                      │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Key Design Decisions

### 1. LogitsProcessor for MEC Integration

TRT-LLM's `LogitsProcessor` callback receives:
- `req_ids`: Request ID
- `logits`: Full logits tensor `[batch, beam, vocab_size]`
- `ids`: Tokens generated so far
- `stream_ptr`: CUDA stream pointer
- `client_id`: Optional client ID

We use this to:
1. Apply temperature scaling
2. Get top-k tokens and their probabilities
3. Run MEC coupling algorithm
4. Set logits to `-inf` except for selected token

### 2. Stateful Generation

The MEC algorithm maintains state (posterior distributions) across tokens. We handle this via:
- A per-request state dictionary keyed by `req_ids`
- State includes: ciphertext, current posterior, token count

### 3. Simplified Model (No Speculative Decoding)

MEC requires token-by-token control. Speculative decoding generates multiple tokens at once, which conflicts with MEC's coupling approach. We disable it for steganography mode.

## Implementation Components

### A. `trtllm_model_wrapper.py`
- `TRTLLMModelWrapper`: Replaces HuggingFace wrapper
- Provides `top_k_conditional()` using TRT-LLM logits
- Handles vocab mapping between reduced indices and full vocab

### B. `mec_logits_processor.py`
- `MECLogitsProcessor(LogitsProcessor)`: Core integration
- Maintains MEC state per request
- Implements the coupling algorithm in the callback

### C. `modal_stego_app.py`
- Modal app definition with GPU container
- `StegoModel` class with encode/decode endpoints
- Engine building and caching

### D. `test_modal_stego.py`
- Local test script
- Roundtrip encode/decode verification

## Limitations & Trade-offs

1. **No speculative decoding**: MEC needs per-token control, so we lose the 4x speedup from lookahead decoding
2. **Single batch**: MEC state is per-request, batching would require careful synchronization
3. **Latency vs throughput**: Optimizing for correctness over raw speed

## Files to Create

1. `/Users/jon/projects/Tomato/tomato/utils/trtllm_wrapper.py` - TRT-LLM model wrapper
2. `/Users/jon/projects/Tomato/modal_stego_app.py` - Modal deployment
3. `/Users/jon/projects/Tomato/test_modal_stego.py` - Test script

## Testing Procedure

### Prerequisites

1. **Modal account**: Sign up at https://modal.com
2. **Modal CLI**: `pip install modal && modal token new`
3. **HuggingFace token**: For gated models (set `HF_TOKEN` env var if needed)

### Step 1: Local Syntax Check

```bash
cd /Users/jon/projects/Tomato
python -c "import ast; ast.parse(open('modal_stego_app.py').read()); print('Syntax OK')"
```

### Step 2: Deploy to Modal

```bash
modal deploy modal_stego_app.py
```

This will:
- Build a Docker image with CUDA 12.8 + TensorRT-LLM 0.18.0
- Download LLaMA 3.2 1B model
- Compile TRT-LLM engine (first run takes ~5-10 minutes)
- Cache everything in a Modal Volume

Expected output:
```
✓ Created objects.
├── 🔨 Created mount /Users/jon/projects/Tomato/modal_stego_app.py
├── 🔨 Created volume stego-trtllm-volume
└── 🔨 Created function StegoModel.
✓ App deployed! 🎉
```

### Step 3: Run Integration Test

```bash
python test_modal_stego.py
```

For a quick test:
```bash
python test_modal_stego.py -m "hi" -p "Once upon a time:"
```

For the full test suite:
```bash
python test_modal_stego.py --full
```

### Step 4: Expected Results

**Successful roundtrip:**
```
Testing roundtrip for: 'hello'
  Prompt: Tell me a story:...
  Encode time: 2500.0ms
  Stegotext preview: Once upon a time, there was a small village...
  Decode time: 2300.0ms
  Decoded: 'helloAAAAAAAAAA'
  PASS: Perfect roundtrip
```

**Performance expectations:**
- First encode: ~3-5 seconds (includes logits capture per token)
- Subsequent encodes: ~2-3 seconds
- Match rate: >80% for messages < 10 characters

### Step 5: Troubleshooting

**If deployment fails:**
```bash
# Check Modal status
modal app list

# View logs
modal app logs stego-trtllm
```

**If tests fail with "Model not deployed":**
```bash
modal deploy modal_stego_app.py
```

**If GPU not available:**
- Modal H100 availability varies; the deploy will queue until resources are available

### Step 6: Cleanup

```bash
# Stop running containers
modal app stop stego-trtllm

# Delete the app entirely
modal app delete stego-trtllm

# Delete cached models (optional)
modal volume delete stego-trtllm-volume
```
