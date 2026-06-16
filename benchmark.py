import torch
import torch.nn.functional as F
import time
from ops.flash_attn import flash_attention_v2

# 性能测试配置
BATCH = 4
HEADS = 16
SEQ_LEN = 2048
HEAD_DIM = 64

q = torch.randn(BATCH, HEADS, SEQ_LEN, HEAD_DIM, device="cuda", dtype=torch.float16)
k = torch.randn(BATCH, HEADS, SEQ_LEN, HEAD_DIM, device="cuda", dtype=torch.float16)
v = torch.randn(BATCH, HEADS, SEQ_LEN, HEAD_DIM, device="cuda", dtype=torch.float16)

# 预热 GPU (Warmup) 防止动态编译和冷启动干扰计时
for _ in range(10):
    _ = flash_attention_v2(q, k, v)
    _ = F.scaled_dot_product_attention(q, k, v)
torch.cuda.synchronize()

# 1. 测试你的 Triton 版 FlashAttention 耗时
start_time = time.time()
for _ in range(100):
    out_triton = flash_attention_v2(q, k, v)
torch.cuda.synchronize()
triton_time = (time.time() - start_time) / 100

# 2. 测试 PyTorch 官方原生标杆耗时
start_time = time.time()
for _ in range(100):
    out_torch = F.scaled_dot_product_attention(q, k, v)
torch.cuda.synchronize()
torch_time = (time.time() - start_time) / 100

print("\n⚡ 【项目二阶段三：性能暴风雨报告】 ⚡")
print(f"   Triton FlashAttn 平均耗时: {triton_time * 1000:.4f} ms")
print(f"   PyTorch 原生标杆 平均耗时:  {torch_time * 1000:.4f} ms")

# 计算加速比
speedup = torch_time / triton_time
print(f"🚀 加速比 (Speedup): {speedup:.2f}x")