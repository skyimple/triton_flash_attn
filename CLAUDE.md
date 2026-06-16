# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Verify correctness against PyTorch reference
python verify_flash_attn.py

# Run benchmark (placeholder — implement first)
python benchmark.py
```

## Architecture

Minimal implementation of FlashAttention-2 forward pass using Triton.

### Structure

- `kernels/flash_attn_kernels.py` — Triton JIT kernel (`@triton.jit`) implementing the flash attention forward pass. Uses 3D grid (batch × head × Q-blocks), online safe softmax with running maximum (`m_i`) and denominator (`l_i`), and Tensor Core matmul via `tl.dot`. Tiles along the K/V sequence dimension.

- `ops/flash_attn.py` — `flash_attention_v2(q, k, v)` Python wrapper. Forces contiguous inputs, sets block sizes (BLOCK_M=32, BLOCK_N=32), launches the Triton kernel on a 3D grid.

- `verify_flash_attn.py` — Correctness test: compares Triton output against `torch.nn.functional.scaled_dot_product_attention` with FP16 inputs. Passes if max error < 5e-3.

- `benchmark.py` — Performance benchmark (currently a placeholder file).

### Key design points

- **Online softmax**: The kernel maintains a running max `m_i` and sum `l_i` across K/V tiles, rescaling accumulated output with `alpha = exp(m_i - m_next)` and `beta = exp(m_ij - m_next)`.
- **Tensor Cores**: `tl.dot(q, k.T)` and `tl.dot(softmax_weights, v)` use FP16 inputs per Triton's `tl.dot` constraints.
- **No causal masking**: Current implementation does full attention; no `is_causal` support.
- **Block size globals**: BLOCK_M=32, BLOCK_N=32, HEAD_DIM=64 are the tested configuration. HEAD_DIM is a `tl.constexpr`.

### Dependencies

- `torch` (with CUDA)
- `triton` (nightly or latest stable)
