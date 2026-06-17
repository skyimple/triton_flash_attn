# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Verify correctness vs PyTorch reference (FP16, B=4 H=8 S=1024 D=64, is_causal=True)
python verify_flash_attn.py        # exits 0 on pass, 1 on fail (max error < 5e-3)

# Run performance benchmark vs PyTorch SDPA (FP16, B=4 H=16 S=2048 D=64, is_causal=True, 100 iters)
python benchmark.py                # prints avg ms and speedup ratio
```

Both scripts test **causal** attention by default. The correctness threshold is `max(|triton_out - torch_out|) < 5e-3`, which is the expected limit for FP16 accumulation.

## Architecture

Minimal FlashAttention-2 forward pass in Triton — a single fused kernel with a Python launcher, pure FP16, targeting Ampere Tensor Cores.

### Structure

```
triton_flash_attn/
├── kernels/
│   ├── __init__.py                    # empty
│   ├── flash_attn_kernels_v0.py       # v0 — no alignment hints, no causal, BLOCK_M=32
│   ├── flash_attn_kernels_v1.py       # v1 — (16,16) multiple_of hints, still no causal
│   └── flash_attn_kernels.py          # v2 (ACTIVE) — scalar multiple_of(..., 16), IS_CAUSAL constexpr
├── ops/
│   ├── __init__.py                    # empty
│   └── flash_attn.py                 # Python launcher — contains 3 historical versions commented out, active at bottom
├── verify_flash_attn.py              # correctness: max error vs F.scaled_dot_product_attention
├── benchmark.py                      # performance: avg ms over 100 iterations vs PyTorch SDPA
├── README.md
└── CLAUDE.md
```

### Kernel version lineage

All three kernel files define a function called `flash_attn_fwd_kernel` with the same signature. Only the active one is imported by the launcher.

| File | `multiple_of` | `IS_CAUSAL` | `BLOCK_M`/`N` | `num_warps` | Comment language |
|------|--------------|-------------|----------------|-------------|------------------|
| `kernels_v0.py` | none | no | 32 (from kernel) | default | Chinese, academic style |
| `kernels_v1.py` | `(16, 16)` tuple | no | 64 (from launcher) | 4 | Chinese, "3080 Ti" branded |
| `kernels.py` (active) | scalar `16` | **yes** | 64 (from launcher) | 4 | Chinese + English, punchy/production style |

### Launcher evolution (`ops/flash_attn.py`)

The launcher file is a palimpsest of three iterations:
1. **Commented V1** — `BLOCK_M=32`, no `num_warps`, no `is_causal` parameter.
2. **Commented V2** — `BLOCK_M=64`, `num_warps=4`, still no `is_causal`.
3. **Active code at the bottom** — same block sizes as V2, adds `is_causal: bool = False` passthrough as `IS_CAUSAL`.

Only the active code at the bottom of the file is reachable. The commented blocks are historical notes.

### How calls flow

1. `verify_flash_attn.py` or user code calls `flash_attention_v2(q, k, v, is_causal=True/False)` from `ops/flash_attn.py`
2. The wrapper calls `.contiguous()` on all inputs, then launches the Triton kernel on a `(ceil(N_CTX/BLOCK_M), H, Z)` 3D grid with `num_warps=4`
3. Each program processes one Q-row-block × one head × one batch item
4. Inside the kernel: loop over K/V tiles (tiled along N_CTX), online safe softmax via running max `m_i` and denominator `l_i`, accumulate with `tl.dot` on Tensor Cores
5. When `IS_CAUSAL=1`, each program's loop is bounded to `min(N_CTX, (pid_m+1)*BLOCK_M)` and a per-tile `tl.where` masks the upper triangle

### Key design decisions

- **Online safe softmax**: Kernel maintains `m_i` (running max) and `l_i` (running sum of exp) across K/V tiles. On each tile: compute `m_ij = max(s)`, rescale old accumulator via `alpha = exp(m_i - m_next)`, add new tile's contribution, finally `acc = acc / l_i`.
- **Causal mask via loop bound + tile-level masking**: (1) Compile-time `IS_CAUSAL` constexpr limits each Q-block's K/V loop to `min(N_CTX, (pid_m+1)*BLOCK_M)`, skipping all-zero upper-triangular tiles entirely. (2) The single tile straddling the diagonal applies `tl.where(offs_m[:, None] >= offs_n[None, :], s, -inf)` for per-element masking.
- **128-bit aligned loads**: `tl.multiple_of(Q_ptrs, 16)` hints Triton to emit vectorized 128-bit loads. Safe for contiguous FP16 with HEAD_DIM=64 (128 bytes per row).
- **Accumulator dtype**: `acc` is `tl.float32` regardless of input dtype — Ampere Tensor Cores accumulate in FP32 even with FP16 inputs.
- **Block size 64**: `BLOCK_M=BLOCK_N=64` with `num_warps=4` targets the RTX 3080 Ti's 8 Tensor Cores per SM. Larger tiles reduce global memory round-trips.
- **`sm_scale` is hardcoded**: Always `1/sqrt(HEAD_DIM)`. No user override — unlike the standard FlashAttention API.
- **`beta` is dead code in v0 and v1**: The `beta = tl.math.exp(m_ij - m_next)` variable is computed but unused. The kernel computes `p = tl.math.exp(s - m_next[:, None])` directly instead of `beta * exp(s - m_ij)`. (Removed in the active v2 kernel.)

### Known limitations

- No variable sequence length (all sequences in batch assumed same length `N_CTX`)
- No dropout, no bias, no `sm_scale` override
- `__init__.py` files are empty — importing `kernels` or `ops` alone won't re-export submodules; must use full `from ops.flash_attn import flash_attention_v2`
- The kernel is tested only with FP16 on HEAD_DIM=64
- Only tested on single-GPU (RTX 3080 Ti, Ampere)
- No backward pass — forward only

### Dependencies

- `torch` (with CUDA, tested on Ampere/RTX 3080 Ti)
- `triton` (nightly or latest stable)
