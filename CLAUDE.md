# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Verify correctness vs PyTorch reference (FP16, B=4 H=8 S=1024 D=64, is_causal=True)
python verify_flash_attn.py

# Run performance benchmark vs PyTorch SDPA (FP16, B=4 H=16 S=2048 D=64, is_causal=True, 100 iters)
python benchmark.py
```

Both scripts now test **causal** attention by default. The threshold for correctness is max error < 5e-3.

## Architecture

Minimal FlashAttention-2 forward pass in Triton — a single fused kernel with a Python launcher, pure FP16, targeting Ampere Tensor Cores.

### Structure

```
triton_flash_attn/
├── kernels/
│   ├── __init__.py                    # empty
│   ├── flash_attn_kernels_v0.py       # v0 (original) — no alignment hints, no causal, no num_warps
│   ├── flash_attn_kernels_v1.py       # v1 — tl.multiple_of(..., (16,16)) alignment hints, no causal; Chinese comments
│   └── flash_attn_kernels.py          # v2 (ACTIVE) — alignment hints + IS_CAUSAL tl.constexpr, scalar multiple_of(..., 16)
├── ops/
│   ├── __init__.py                    # empty
│   └── flash_attn.py                 # Python launcher: contiguous(), BLOCK_M=BLOCK_N=64, num_warps=4, is_causal passthrough
├── verify_flash_attn.py              # Correctness: max error < 5e-3 vs F.scaled_dot_product_attention
├── benchmark.py                      # Performance: average ms over 100 iterations vs PyTorch SDPA
├── README.md
└── CLAUDE.md
```

### Active kernel version lineage

| File | multiple_of | IS_CAUSAL | BLOCK_M/N | num_warps | comments |
|------|------------|-----------|-----------|-----------|----------|
| `flash_attn_kernels_v0.py` | none | no | 32 | default | English, academic-style |
| `flash_attn_kernels_v1.py` | `(16, 16)` tuple | no | 64 (from wrapper) | 4 | Chinese, "3080 Ti" theme |
| `flash_attn_kernels.py` | scalar `16` | **yes** | 64 (from wrapper) | 4 | Chinese + English, production-style |

The active kernel (`kernels/flash_attn_kernels.py`) is the only one wired to the launcher. The v0 and v1 files are historical snapshots kept for reference.

### How calls flow

1. `verify_flash_attn.py` or user code calls `flash_attention_v2(q, k, v, is_causal=True/False)` from `ops/flash_attn.py`
2. The wrapper forces contiguous layout, and launches the Triton kernel on a `(ceil(N_CTX/BLOCK_M), H, Z)` 3D grid with `num_warps=4`
3. Each program processes one Q-row-block × one head × one batch item
4. Inside the kernel: loop over K/V tiles (tiled along N_CTX), online safe softmax via running max `m_i` and denominator `l_i`, accumulate with `tl.dot` on Tensor Cores
5. When `IS_CAUSAL=1`, each program's loop is bounded to `min(N_CTX, (pid_m+1)*BLOCK_M)` and a per-tile `tl.where` masks the upper triangle in `s`

### Key design decisions

- **Online safe softmax**: Kernel maintains `m_i` (running max) and `l_i` (running sum of exp) across K/V tiles. On each tile: compute `m_ij = max(s)`, rescale old accumulator via `alpha = exp(m_i - m_next)`, add new tile's contribution via `beta = exp(m_ij - m_next)`, finally `acc = acc / l_i`.
- **Causal mask via loop bound + tile-level masking**: Two-level optimization. (1) Compile-time `IS_CAUSAL` constexpr limits each Q-block's K/V loop to `min(N_CTX, (pid_m+1)*BLOCK_M)`, skipping all-zero upper-triangular tiles entirely. (2) The single tile that straddles the diagonal applies `tl.where(offs_m[:, None] >= offs_n[None, :], s, -inf)` for per-element masking.
- **128-bit aligned loads**: `tl.multiple_of(Q_ptrs, 16)` hints Triton to emit vectorized 128-bit (16-byte) loads. Works when all leading strides are multiples of 16 bytes / `HEAD_DIM * dtype_size` — safe for contiguous FP16 with HEAD_DIM=64 (128 bytes per row).
- **Accumulator dtype**: `acc` is `tl.float32` regardless of input dtype — Ampere Tensor Cores accumulate in FP32 even with FP16 inputs.
- **Block size 64**: `BLOCK_M=BLOCK_N=64` with `num_warps=4` targets the RTX 3080 Ti's 8 Tensor Core per SM geometry. Larger tiles reduce global memory round-trips.
- **`beta` is computed but unused in the active kernel**: The `beta` variable in the online softmax rescaling was present in v0 but is never used — the kernel computes `tl.math.exp(s - m_next[:, None])` directly instead of `beta * exp(s - m_ij)`.

### Known limitations

- No variable sequence length (all sequences in batch assumed same length `N_CTX`)
- No dropout, no bias, no `sm_scale` override
- `__init__.py` files are empty — importing `kernels` or `ops` alone won't re-export submodules; must use full `from ops.flash_attn import flash_attention_v2`
- The kernel is tested only with FP16 on HEAD_DIM=64
- Only tested on single-GPU (RTX 3080 Ti, Ampere)

### Dependencies

- `torch` (with CUDA, tested on Ampere/RTX 3080 Ti)
- `triton` (nightly or latest stable)
