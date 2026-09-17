# zero-to-hero

从零开始，用 PyTorch 手写实现一个小型 GPT 语言模型并训练。项目参考 Andrej Karpathy 的 zero-to-hero 系列，全部代码在 [video.ipynb](video.ipynb) 中。

## 模型结构

所有组件均从零实现，不依赖现成的 Transformer 库：

| 模块 | 说明 |
| --- | --- |
| `GPTConfig` | 模型配置（block_size=512, n_layer=2, n_head=12, n_embd=768） |
| `SingleHeadAttention` | 单头注意力 |
| `MultiHeadAttention` | 多头注意力 |
| `FeedForward` | 前馈网络 |
| `Block` | Transformer Block（注意力 + 前馈 + 残差） |
| `GPT` | 完整模型，支持 `model(x, targets=y)` 直接返回 loss |

## 数据

- 使用中文 JSONL 语料（`mobvoi_seq_monkey_general_open_corpus.jsonl`），取前 1000 条
- 使用 `tiktoken` 的 GPT-2 分词器（`gpt2` 编码）
- 按 90% / 10% 随机划分训练集和验证集

## 训练

- 优化器：AdamW（lr=3e-4）+ CosineAnnealing 学习率调度
- 自动检测 GPU，有 CUDA 用 `cuda`，否则用 `cpu`
- 训练循环每 100 个 batch 打印一次 loss

## 运行

```bash
pip install torch tiktoken
```

1. 在 notebook 的 cell 8 中把数据路径改成你自己的 JSONL 文件路径
2. 按顺序运行所有 cell 即可开始训练
