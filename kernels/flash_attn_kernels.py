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
    # 1. 硬件多维网格切片映射
    pid_m = tl.program_id(0)   # 负责当前 Q 的第几个 Row Block
    pid_h = tl.program_id(1)   # 负责当前是第几个 Head
    pid_z = tl.program_id(2)   # 负责当前是第几个 Batch 句子

    # 2. 定位当前 Program 专属的 Q 块起始地址
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)

    # 构造二维指针矩阵网格并装载 
    Q_ptrs = Q_ptr + pid_z * stride_qz + pid_h * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    # 🛡️ 生产级掩码：拦截 Q 块行数越界
    q = tl.load(Q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)

    # 3. 初始化在线 Softmax 片上临时账本
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    qk_scale = 1.0 / (HEAD_DIM ** 0.5)

    # 4. 沿着 K/V 的长度轴横向进行大循环分块迭代
    for start_n in range(0, N_CTX, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)

        # 物理指针精确制导：拉取 K 块与 V 块
        K_ptrs = K_ptr + pid_z * stride_kz + pid_h * stride_kh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
        V_ptrs = V_ptr + pid_z * stride_vz + pid_h * stride_vh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
        
        # 🛡️ 生产级掩码：加载 K、V 时严密拦截动态非整除长度越界，越界部分用 0.0 垫底
        k = tl.load(K_ptrs, mask=offs_n[:, None] < N_CTX, other=0.0)
        v = tl.load(V_ptrs, mask=offs_n[:, None] < N_CTX, other=0.0)

        # ⚡ 硬件硬核调用：Tensor Core 矩阵乘法 S = Q * K^T
        s = tl.dot(q, tl.trans(k)) * qk_scale

        # 🚨 核心战役：片上最大值与单位重缩放
        m_ij = tl.max(s, axis=1)
        m_next = tl.maximum(m_i, m_ij)
        
        alpha = tl.math.exp(m_i - m_next)
        beta = tl.math.exp(m_ij - m_next)

        # 动态重平衡老分母与新分母
        l_i_next = l_i * alpha + tl.sum(tl.math.exp(s - m_next[:, None]), axis=1)

        # 🧬 物理级拯救：对齐老账本的数学单位
        acc = acc * alpha[:, None]

        # 计算当前块归一化前的权重，并用 Tensor Core 乘以 V 累加进输出
        p = tl.math.exp(s - m_next[:, None])
        acc += tl.dot(p.to(tl.float16), v)

        # 账本状态翻页
        m_i = m_next
        l_i = l_i_next

    # 5. 全局循环结束，执行最终的归一化分母收尾
    acc = acc / l_i[:, None]

    # 6. 安全、连续地写回全局显存 (O)
    O_ptrs = O_ptr + pid_z * stride_oz + pid_h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    # 🛡️ 生产级掩码：防止输出写回时踩踏未知内存
    tl.store(O_ptrs, acc, mask=offs_m[:, None] < N_CTX)