# ExplicitLM

基于 Qwen3-4B 的显式记忆增强语言模型。通过 Product Key Memory (PKM) 机制实现知识的显式存储和检索。
当前实现将记忆融合插入在每层 Attention 之后、FFN 之前。

## 环境配置

```bash
# 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 安装依赖
git clone https://github.com/pioneerLu/ExplicitLM.git
cd ExplicitLM
uv sync
```

## 项目结构

```
ExplicitLM/
├── models/                     # 模型定义
│   ├── core/                   # ExplicitLM, Qwen3ExplicitLMBlock
│   ├── layers/                 # RMSNorm
│   └── memory_bank/            # MemoryGate, GatedMemoryFusion
├── config/                     # 配置模块
├── utils/                      # 训练工具
├── util_py/                    # 数据预处理工具
├── scripts/
│   ├── train/                  # 训练脚本
│   ├── convert/                # 模型转换脚本
│   └── inference/              # 推理脚本
├── data/                       # 数据目录
├── checkpoints/                # 训练检查点
└── pyproject.toml
```

---

## 功能模块

### 1. 训练

#### SFT 训练（这个dataset还没完全定下来，还在pretrain部分）

```bash
# 使用脚本
bash scripts/train/run_sft.sh

# 或直接运行
export CUDA_VISIBLE_DEVICES=0,1,2
uv run accelerate launch --config_file accelerate_config.yaml scripts/train/train_sft.py \
    model.qwen3_model_path=Qwen_hg/Qwen3-4b \
    model.cache_path=data/cache/memory_bank.pt \
    model.keys_path=data/keys.pt \
    model.knowledge_num=1048576 \
    model.knowledge_length=32 \
    dataset.sft_dataset_path=data/parquet_data/256 \
    training.learning_rate=5e-5 \
    training.batch_size=2 \
    training.accumulation_steps=128 \
    training.epochs=3
```

#### Pretrain 训练

```bash
# 使用脚本
bash scripts/train/run_fusion_pretrain.sh

# 或直接运行
export CUDA_VISIBLE_DEVICES=0,1,2
uv run accelerate launch --config_file accelerate_config.yaml scripts/train/train_pretrain.py \
    --qwen3_model_path Qwen_hg/Qwen3-4b \
    --cache_path data/cache/memory_bank.pt \
    --keys_path data/keys.pt \
    --dataset_path data/parquet_data/256 \
    --knowledge_num 1048576 \
    --knowledge_length 32 \
    --batch_size 2 \
    --lr 1e-4 \
    --epochs 3 \
    --enable_memory_update
```

---

### 2. 模型转换

将训练 checkpoint 转换为 HuggingFace 格式。

#### Memory Bank 独立存储（推荐）

```bash
uv run python scripts/convert/pt2hg_apart.py \
    --checkpoint_path checkpoints/fusion_pretrain/checkpoint_step_14500 \
    --qwen3_path Qwen_hg/Qwen3-4b \
    --output_path hf_models/explicitlm_apart \
    --memory_bank_path data/cache/memory_bank.pt
```

#### Memory Bank 同步存储

```bash
uv run python scripts/convert/pt2hg_sync.py \
    --checkpoint_path checkpoints/fusion_pretrain/checkpoint_step_14500 \
    --qwen3_path Qwen_hg/Qwen3-4b \
    --output_path hf_models/explicitlm_sync
```

---

### 3. 推理

#### 使用转换后的 HuggingFace 模型

```bash
# 转换脚本会自动生成 inference_example.py
uv run python hf_models/explicitlm_apart/inference_example.py
```

#### 使用原始 checkpoint

```bash
uv run python scripts/inference/chat_example.py
```

---

### 4. 数据预处理

#### 从 Parquet 提取 Memory Bank

```bash
uv run python util_py/extract_parquet_to_memory_bank.py \
    --parquet_dir data/parquet_data \
    --output_path data/cache/memory_bank.pt \
    --qwen_model_path Qwen_hg/Qwen3-4b \
    --knowledge_num 1048576 \
    --knowledge_length 32
```

#### 从 Memory Bank 生成 Keys

```bash
uv run python util_py/generate_keys_from_memory_bank.py \
    --memory_bank_path data/cache/memory_bank.pt \
    --output_path data/keys.pt \
    --qwen_model_path Qwen_hg/Qwen3-4b \
    --pkm_k 1024
```

#### 转换对话数据为 SFT 格式

```bash
uv run python util_py/convert_extract_data_to_sft.py \
    --input data/train_data.json \
    --output_dir sft_data \
    --qwen_model_path Qwen_hg/Qwen3-4b
```

---

## 重要参数

### 模型参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `model.qwen3_model_path` | Qwen3 模型路径 | - |
| `model.cache_path` | Memory Bank 文件路径 | - |
| `model.keys_path` | PKM Keys 文件路径 | - |
| `model.knowledge_num` | Memory Bank 条目数（必须是完全平方数） | 1048576 |
| `model.knowledge_length` | 每个条目的 token 数 | 32 |
| `model.num_candidates` | 检索候选数 | 16 |
| `model.gate_rank` | MemoryGate LoRA rank | 128 |
| `model.fusion_rank` | Fusion LoRA rank | 128 |

### 训练参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `training.learning_rate` | 学习率 | 5e-5 |
| `training.batch_size` | 批次大小 | 2 |
| `training.accumulation_steps` | 梯度累积步数 | 128 |
| `training.epochs` | 训练轮数 | 3 |
| `training.zero_stage` | DeepSpeed ZeRO 阶段 | 2 |
| `training.similarity_loss_coef` | 相似度损失系数 | 1.0 |

### Memory 更新参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--enable_memory_update` | 启用动态更新 | False |
| `--memory_update_frequency` | 更新频率（步数） | 500 |
| `--memory_update_strategy` | 更新策略 (fifo/lru/random) | lru |

---

## 数据格式

### 训练数据 (JSONL)

```jsonl
{"conversations": [{"role": "user", "content": "问题"}, {"role": "assistant", "content": "回答"}]}
```

### 训练数据 (Parquet)

包含 `text` 字段的 Parquet 文件，支持目录下多文件。

### Memory Bank (.pt)

```python
{
    "memory_bank": torch.Tensor,  # [knowledge_num, knowledge_length] token IDs
    "valid_mask": torch.Tensor    # [knowledge_num] bool
}
```

### Keys (.pt)

```python
{
    "row_keys": torch.Tensor,  # [K, hidden_size]
    "col_keys": torch.Tensor   # [K, hidden_size]
}
# K = sqrt(knowledge_num), 例如 knowledge_num=1048576 时 K=1024
```

---

## 常用命令

```bash
# 查看训练日志
tail -f train.log

# 查看 GPU 状态
nvidia-smi

# 停止训练
pkill -f train_sft.py

# 检查 Memory Bank 信息
uv run python -c "import torch; d=torch.load('data/cache/memory_bank.pt'); print(f'Shape: {d[\"memory_bank\"].shape}, Valid: {d[\"valid_mask\"].sum()}')"
```
