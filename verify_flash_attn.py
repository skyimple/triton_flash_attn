import torch
import torch.nn.functional as F
from ops.flash_attn import flash_attention_v2

print("🚀 【全新仓库：项目二阶段二测试】：正在启动 FlashAttention-2 Triton 核心真值审查...")

# 1. 构造标准 4D 大模型输入 [Batch, Head, SeqLen, Dim]
batch_size = 4
num_heads = 8
seq_len = 1024
head_dim = 64

# 必须使用 FP16 或 FP32。由于 Triton 的 tl.dot 硬件上强制要求输入是 16位 (FP16/BF16)
# 我们这里全部采用 FP16 运行，完美对齐工业界主流生产模式
q = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
k = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
v = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda", dtype=torch.float16)

# 2. 调用你手撕的 Triton FlashAttention 引擎
print("🛠️ 首次启动：触发 Triton JIT 编译器在后台狂轰乱炸生成机器汇编...")
triton_out = flash_attention_v2(q, k, v)

# 3. 调用 PyTorch 官方高度优化、公认绝对正确的原生标杆算子
torch_out = F.scaled_dot_product_attention(q, k, v, is_causal=False)

# 4. 分子级绝对精度对齐判决
max_diff = torch.max(torch.abs(triton_out - torch_out)).item()
print(f"\n📊 仓库独立真值对齐报告：")
print(f"   最大绝对误差 (Max Absolute Error): {max_diff}")

if max_diff < 5e-3: # FP16 下 1e-3 ~ 5e-3 属于机器精度极值限制的完全对齐
    print("\n🎉 乾坤大定！全新项目二阶段二通关！你重写的 FlashAttention 在数学上完成了绝对闭环！")
else:
    print("\n❌ 精度崩塌！立刻回看片上在线 Softmax 的 alpha 和 beta 的累加顺序！")
    exit(1)