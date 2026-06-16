import torch
import triton
from kernels.flash_attn_kernels import flash_attn_fwd_kernel

def flash_attention_v2(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    # 强制连续性，防止非连续张量导致底层 Stride 乘法发生物理位移
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    
    Z, H, N_CTX, HEAD_DIM = q.shape
    o = torch.zeros_like(q)

    # 大厂优化实战推荐的分块大小
    BLOCK_M = 32
    BLOCK_N = 32

    # 配置 3D 网格：X轴处理 Q 的分块数，Y轴处理 Head，Z轴处理 Batch 句子
    grid = (triton.cdiv(N_CTX, BLOCK_M), H, Z)

    # 轰入 Triton 动态 JIT 编译器
    flash_attn_fwd_kernel[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        Z, H, N_CTX,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=HEAD_DIM
    )
    return o