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
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr
):
    pid_m = tl.program_id(0)   
    pid_h = tl.program_id(1)   
    pid_z = tl.program_id(2)   

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)

    # 定位 Q 指针并施加 16 字节（128位）向量化对齐暗示
    q_offset = pid_z * stride_qz + pid_h * stride_qh + offs_m[:, None] * stride_qm
    Q_ptrs = Q_ptr + q_offset + offs_d[None, :] * stride_qk
    
    # ⚡ 激活 3080 Ti 强大的向量化加载
    q = tl.load(tl.multiple_of(Q_ptrs, (16, 16)), mask=offs_m[:, None] < N_CTX, other=0.0)

    # 账本初始化
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    
    # ⚡ 3080 Ti 算力恐怖，我们直接用 fp32 作为 acc 累加器，提供比 2080 Ti 更变态的数值稳定性，
    # 且 Ampere 架构对 fp32 寄存器的时钟周期进行了大幅优化
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    qk_scale = 1.0 / (HEAD_DIM ** 0.5)

    for start_n in range(0, N_CTX, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)

        K_ptrs = K_ptr + pid_z * stride_kz + pid_h * stride_kh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
        V_ptrs = V_ptr + pid_z * stride_vz + pid_h * stride_vh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
        
        # ⚡ 强制触发 128-bit 向量化合并访问
        k = tl.load(tl.multiple_of(K_ptrs, (16, 16)), mask=offs_n[:, None] < N_CTX, other=0.0)
        v = tl.load(tl.multiple_of(V_ptrs, (16, 16)), mask=offs_n[:, None] < N_CTX, other=0.0)

        # ⚡ Ampere Tensor Core 轰击：计算 S = Q * K^T
        s = tl.dot(q, tl.trans(k)) * qk_scale

        # 在线 Softmax 规约
        m_ij = tl.max(s, axis=1)
        m_next = tl.maximum(m_i, m_ij)
        
        alpha = tl.math.exp(m_i - m_next)
        beta = tl.math.exp(m_ij - m_next)

        l_i_next = l_i * alpha + tl.sum(tl.math.exp(s - m_next[:, None]), axis=1)

        # 🧬 动态调整之前 acc 的尺度
        acc = acc * alpha[:, None]

        # 累加当前分块
        p = tl.math.exp(s - m_next[:, None])
        acc += tl.dot(p.to(q.dtype), v)

        m_i = m_next
        l_i = l_i_next

    # 最终归一化
    acc = acc / l_i[:, None]

    # 写回全局显存
    O_ptrs = O_ptr + pid_z * stride_oz + pid_h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(tl.multiple_of(O_ptrs, (16, 16)), acc.to(q.dtype), mask=offs_m[:, None] < N_CTX)