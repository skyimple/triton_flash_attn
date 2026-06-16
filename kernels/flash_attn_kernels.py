import torch
import triton
import triton.language as tl

@triton.jit
def flash_attn_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
    IS_CAUSAL: tl.constexpr # 👈 编译期常量优化
):
    pid_m = tl.program_id(0)   
    pid_h = tl.program_id(1)   
    pid_z = tl.program_id(2)   

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)

    q_offset = pid_z * stride_qz + pid_h * stride_qh + offs_m[:, None] * stride_qm
    Q_ptrs = Q_ptr + q_offset + offs_d[None, :] * stride_qk
    q = tl.load(tl.multiple_of(Q_ptrs, (16, 16)), mask=offs_m[:, None] < N_CTX, other=0.0)

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    qk_scale = 1.0 / (HEAD_DIM ** 0.5)

    # 🚨 中级主力绝杀：如果是因果模式，循环上限直接卡死在当前 Q 块的末尾！
    # 如果当前 Q 块在最前面，大循环跑 1 次就直接退出！彻底省掉后面所有全 0 块的读写与计算！
    loop_end = min(N_CTX, (pid_m + 1) * BLOCK_M) if IS_CAUSAL else N_CTX

    for start_n in range(0, loop_end, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)

        K_ptrs = K_ptr + pid_z * stride_kz + pid_h * stride_kh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
        V_ptrs = V_ptr + pid_z * stride_vz + pid_h * stride_vh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
        
        k = tl.load(tl.multiple_of(K_ptrs, (16, 16)), mask=offs_n[:, None] < N_CTX, other=0.0)
        v = tl.load(tl.multiple_of(V_ptrs, (16, 16)), mask=offs_n[:, None] < N_CTX, other=0.0)

        s = tl.dot(q, tl.trans(k)) * qk_scale

        # 🚨 细粒度控制：对于正好压在对角线上的那块“边界混合块”，我们执行局部的局部掩码保护
        if IS_CAUSAL:
            s = tl.where(offs_m[:, None] >= offs_n[None, :], s, float("-inf"))

        m_ij = tl.max(s, axis=1)
        m_next = tl.maximum(m_i, m_ij)
        
        alpha = tl.math.exp(m_i - m_next)

        l_i_next = l_i * alpha + tl.sum(tl.math.exp(s - m_next[:, None]), axis=1)
        acc = acc * alpha[:, None]

        p = tl.math.exp(s - m_next[:, None])
        acc += tl.dot(p.to(q.dtype), v)

        m_i = m_next
        l_i = l_i_next

    acc = acc / l_i[:, None]

    O_ptrs = O_ptr + pid_z * stride_oz + pid_h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(tl.multiple_of(O_ptrs, (16, 16)), acc.to(q.dtype), mask=offs_m[:, None] < N_CTX)