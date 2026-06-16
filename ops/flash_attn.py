# V1
#  import torch
# import triton
# from kernels.flash_attn_kernels import flash_attn_fwd_kernel

# def flash_attention_v2(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
#     # 强制连续性，防止非连续张量导致底层 Stride 乘法发生物理位移
#     q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    
#     Z, H, N_CTX, HEAD_DIM = q.shape
#     o = torch.zeros_like(q)

#     # 大厂优化实战推荐的分块大小
#     BLOCK_M = 32
#     BLOCK_N = 32

#     # 配置 3D 网格：X轴处理 Q 的分块数，Y轴处理 Head，Z轴处理 Batch 句子
#     grid = (triton.cdiv(N_CTX, BLOCK_M), H, Z)

#     # 轰入 Triton 动态 JIT 编译器
#     flash_attn_fwd_kernel[grid](
#         q, k, v, o,
#         q.stride(0), q.stride(1), q.stride(2), q.stride(3),
#         k.stride(0), k.stride(1), k.stride(2), k.stride(3),
#         v.stride(0), v.stride(1), v.stride(2), v.stride(3),
#         o.stride(0), o.stride(1), o.stride(2), o.stride(3),
#         Z, H, N_CTX,
#         BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=HEAD_DIM
#     )
#     return o

# V2 放大BLOCK_M = BLOCK_N = 64
# import torch
# import triton
# from kernels.flash_attn_kernels import flash_attn_fwd_kernel

# def flash_attention_v2(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
#     q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    
#     Z, H, N_CTX, HEAD_DIM = q.shape
#     o = torch.zeros_like(q)

#     # ⚡ 3080 Ti 工业级性能最优硬件配比：放大 Tiling 颗粒度，消灭显存读取频次
#     BLOCK_M = 64
#     BLOCK_N = 64

#     # 3D Grid 网格
#     grid = (triton.cdiv(N_CTX, BLOCK_M), H, Z)

#     # 轰入 JIT 编译器，并显式指定使用 4 个 Warp (128个线程) 来并发吞吐这个方块
#     flash_attn_fwd_kernel[grid](
#         q, k, v, o,
#         q.stride(0), q.stride(1), q.stride(2), q.stride(3),
#         k.stride(0), k.stride(1), k.stride(2), k.stride(3),
#         v.stride(0), v.stride(1), v.stride(2), v.stride(3),
#         o.stride(0), o.stride(1), o.stride(2), o.stride(3),
#         Z, H, N_CTX,
#         BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=HEAD_DIM,
#         num_warps=4
#     )
#     return o


import torch
import triton
from kernels.flash_attn_kernels import flash_attn_fwd_kernel

def flash_attention_v2(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = False) -> torch.Tensor:
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    
    Z, H, N_CTX, HEAD_DIM = q.shape
    o = torch.zeros_like(q)

    BLOCK_M = 64
    BLOCK_N = 64

    grid = (triton.cdiv(N_CTX, BLOCK_M), H, Z)

    flash_attn_fwd_kernel[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        Z, H, N_CTX,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=HEAD_DIM,
        IS_CAUSAL=is_causal, # 👈 注入工业级因果开关
        num_warps=4
    )
    return o