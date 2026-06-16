# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Verify correctness against PyTorch reference (FP16, B=4 H=8 S=1024 D=64)
python verify_flash_attn.py

# Run performance benchmark vs PyTorch SDPA (FP16, B=4 H=16 S=2048 D=64, 100 iters)
python benchmark.py
```

## Architecture

Minimal FlashAttention-2 forward pass in Triton — a single kernel with a Python wrapper, no causal mask, pure FP16.

### Structure

```
triton_flash_attn/
├── kernels/
│   ├── __init__.py                    # empty
│   ├── flash_attn_kernels.py          # Active Triton JIT kernel — uses tl.multiple_of(…, 16) for 128-bit aligned vectorized loads/stores
│   └── flash_attn_kernels_v0.py       # Older variant — same logic, no alignment hints
├── ops/
│   ├── __init__.py                    # empty
│   └── flash_attn.py                 # Python launcher: contiguous(), sets BLOCK_M/BLOCK_N, launches 3D grid
├── verify_flash_attn.py              # Correctness: max error < 5e-3 vs F.scaled_dot_product_attention
├── benchmark.py                      # Performance: average ms over 100 iterations vs PyTorch SDPA
├── README.md
└── CLAUDE.md
```

### How calls flow

1. `verify_flash_attn.py` or user code calls `flash_attention_v2(q, k, v)` from `ops/flash_attn.py`
2. The wrapper forces contiguous layout, computes strides, and launches the Triton kernel on a `(ceil(N_CTX/BLOCK_M), H, Z)` 3D grid
3. Each program processes one Q-row-block × one head × one batch item
4. Inside the kernel: loop over K/V tiles (tiled along N_CTX), online safe softmax via running max `m_i` and denominator `l_i`, accumulate with `tl.dot` on Tensor Cores

### Key design decisions

- **Online safe softmax**: Kernel maintains `m_i` (running max) and `l_i` (running sum of exp) across K/V tiles. On each tile: compute `m_ij = max(s)`, rescale old accumulator via `alpha = exp(m_i - m_next)`, add new tile's contribution via `beta = exp(m_ij - m_next)`, finally `acc = acc / l_i`.
- **128-bit aligned loads**: `tl.multiple_of(Q_ptrs, 16)` hints Triton to emit vectorized 128-bit (16-byte) loads. Works when all leading strides are multiples of 16 bytes / `HEAD_DIM * dtype_size` — safe for contiguous FP16 with HEAD_DIM=64 (128 bytes per row).
- **No causal masking**: Full attention only. Adding causal masking would mask upper-triangular entries in `s` before the softmax max/sum reduction.
- **Accumulator dtype**: `acc` is `tl.float32` regardless of input dtype — Ampere Tensor Cores accumulate in FP32 even with FP16 inputs.
- **BLOCK_M=BLOCK_N=32, HEAD_DIM=64**: Hardcoded in the Python wrapper. HEAD_DIM is a `tl.constexpr`; changing it doesn't recompile other configurations.

### Known limitations

- No `is_causal` / causal mask support
- No variable sequence length (all sequences in batch assumed same length `N_CTX`)
- No dropout, no bias, no `sm_scale` override
- `__init__.py` files are empty — importing `kernels` or `ops` alone won't re-export submodules; must use full `from kernels.flash_attn_kernels import ...`
- The kernel is tested only with FP16 on HEAD_DIM=64

### Dependencies

- `torch` (with CUDA, tested on Ampere/RTX 3080 Ti)
- `triton` (nightly or latest stable)
