# FineWeb Pipeline：预训练数据清洗三代方法论对比

> 在同一份 Common Crawl 数据上，实现并对比 Heuristic Filtering、Model-based Filtering、Hybrid Pipeline 三代方法论，量化"数据质量 vs 数据量"的 trade-off。

---

## 第一次来？从这里开始

根据你的目标，按以下路径进入：

| 我想… | 先看这里 |
|---|---|
| 理解整个项目是做什么的 | 👇 下面的"三代方法论"和"项目地图" |
| 直接跑起来看结果 | 👇 [环境安装](#一环境安装) |
| 看理论和分析 | 📒 打开 `notebooks/00_methodology_overview.ipynb` |
| 看三代对比的最终结论 | 📒 打开 `notebooks/06_cross_generation_comparison.ipynb` |
| 看代码实现细节 | 📂 进入 `src/gen1/`、`src/gen2/`、`src/gen3/` |
| 有疑问 | 📖 查看 `docs/FAQ.md` |

---

## 三代方法论是什么

本项目实现了预训练数据清洗领域三个发展阶段的方法，在**相同的原始数据**上运行，用统一的指标量化对比：

| | 第一代（Gen1） | 第二代（Gen2） | 第三代（Gen3） |
|---|---|---|---|
| 方法 | 人工规则过滤 | fastText 质量分类器 | 分类器集成 + 智能绕过 + LLM 改写 |
| 代表论文 | FineWeb / C4 / Gopher | DCLM（NeurIPS 2024） | Nemotron-CC（NVIDIA 2024） |
| 数据保留率 | ~35% | ~10% | ~40% |
| 核心问题 | 规则无法判断语义质量 | 90% 数据被丢弃 | — |
| MMLU 提升（7B 模型） | 基线 | +9% | +14% |

**核心问题**：第二代质量最高，但丢掉了 90% 数据。第三代的突破是：在质量不降的前提下，把保留率从 10% 提升到 40%。

```
原始 CC 数据（噪声 ~80%）
    │
    ▼
第一代：规则过滤（URL / 长度 / 重复率 / 语言）→ 保留 ~35%
    │
    ▼
第二代：fastText 分类器（top 10%）→ 保留 ~10%
    │
    ▼
第三代：集成分类 + Bypass + LLM 改写 → 保留 ~40%（同等质量）
    │
    ▼
去重（精确 MD5 + MinHash LSH）
    │
    ▼
Proxy Model 验证（GPT-2 125M 对比训练）
```

---

## 项目地图：文件在哪、怎么用

### 📒 Notebooks — 理论学习和实验分析

**如何打开**：在终端运行 `jupyter lab notebooks/`，然后在浏览器中点击文件。
或者用 VS Code 直接打开 `.ipynb` 文件（需安装 Jupyter 扩展）。

每个 Notebook 独立可读，不需要按顺序全部运行。**建议路径**：

**路径 A：只想看最终结论（30 分钟）**
```
06_cross_generation_comparison  ← 三代对比 Dashboard，直接看结果
07_ablation_study               ← 哪个组件贡献最大
```

**路径 B：系统学习方法论（2-3 小时）**
```
00_methodology_overview         ← 术语表 + 演进全貌（必读）
01_data_exploration             ← 原始数据长什么样
02_gen1_heuristic_filtering     ← 第一代：7 个规则过滤器逐一拆解
03_gen2_model_based_filtering   ← 第二代：fastText 训练 + 阈值实验
04_gen3_hybrid_pipeline         ← 第三代：集成 + Bypass + 改写
05_deduplication_analysis       ← 精确去重 + MinHash 原理
06_cross_generation_comparison  ← 核心对比（必读）
07_ablation_study               ← 消融：去掉哪个组件损失最大
```

**路径 C：扩展方向**
```
08_chinese_extension            ← 中文数据的特殊处理方式
09_proxy_model_validation       ← 用 GPT-2 125M 做端到端效果验证
```

### ⚙️ Scripts — 直接运行 Pipeline

**如何运行**：激活虚拟环境后，在终端用 `python scripts/xxx.py` 运行。

```
scripts/
├── download_sample.sh          # 第一步：下载样本数据
├── run_gen1.py                 # 运行第一代 Pipeline
├── run_gen2.py                 # 运行第二代 Pipeline
├── run_gen3.py                 # 运行第三代 Pipeline
├── generate_comparison_report.py  # 生成三代对比报告（Markdown + 图表）
└── run_proxy_training.py       # 进阶：训练 GPT-2 125M 做效果验证
```

### 🔧 Configs — 调整参数

**如何修改**：用任意文本编辑器（VS Code / 记事本）打开 `.yaml` 文件修改。

```
configs/
├── run_config.yaml     # ★ 最重要：切换 smoke_test/full_run 模式
├── gen1_config.yaml    # 第一代过滤器的阈值参数
├── gen2_config.yaml    # 分类器训练参数
├── gen3_config.yaml    # 集成策略和改写参数
├── api_config.yaml     # LLM API 配置（第三代改写用）
└── eval_config.yaml    # 评估体系参数
```

**最常用的修改**：打开 `configs/run_config.yaml`，把第一行改成：
```yaml
run_mode: "smoke_test"   # 1000 文档，10-15 分钟，用于验证环境
run_mode: "full_run"     # 50000 文档，2-4 小时，用于正式实验
```

### 📂 Source Code — 核心实现

```
src/
├── gen1/filters/       # URL过滤、语言检测、质量规则、重复率、PII、毒性
├── gen1/pipeline.py    # 第一代：串联所有过滤器
├── gen2/               # fastText 分类器训练、阈值调参、Pipeline
├── gen3/               # 分类器集成、条件Bypass、LLM改写、Pipeline
├── gen1_zh/            # 中文专用 Pipeline（字符数规则、垃圾检测）
├── dedup/              # 精确去重（xxhash）+ MinHash LSH 模糊去重
├── evaluation/         # 独立评估体系（不参与 Pipeline，防止循环偏差）
│   ├── quality_classifier.py   # 独立质量评分
│   ├── stage_tracker.py        # 记录每一步的指标变化
│   └── filter_auditor.py       # 采样已过滤文档，估算误杀率
└── proxy_model/        # 加载训练好的 GPT-2 模型做效果评估
```

### 📊 Results — 查看输出结果

**如何查看**：Pipeline 运行完成后，结果保存在 `results/` 目录：

```
results/
├── smoke_test/         # smoke_test 模式的 Pipeline 输出（JSONL 文件）
├── full_run/           # full_run 模式的输出
├── reports/
│   └── comparison_report.md    # 三代对比 Markdown 报告（用浏览器/VS Code 打开）
├── figures/
│   └── comparison_dashboard.png  # 四格可视化图表（直接双击打开）
├── quality_scores/     # 训练好的分类器模型文件
└── proxy_models/       # GPT-2 训练结果和 report.md
```

---

## 一、环境安装

```bash
# 需要 Python 3.10+（macOS 自带的 3.9 不够）
brew install python@3.11

# 一键安装所有依赖（约 5-10 分钟）
bash setup.sh

# 激活虚拟环境（每次新开终端都需要运行这一行）
source .venv/bin/activate
```

---

## 二、运行流程

### 步骤 1：下载样本数据
```bash
bash scripts/download_sample.sh
# 下载约 10-20 分钟，保存到 data/raw/
```

### 步骤 2：验证环境（smoke_test，1000 文档，约 15 分钟）
```bash
# 确认 configs/run_config.yaml 第一行是：run_mode: "smoke_test"
python scripts/run_gen1.py
python scripts/run_gen2.py
python scripts/run_gen3.py --no-rephrase   # --no-rephrase 跳过 LLM 改写，无需 API Key
```

### 步骤 3：生成对比报告
```bash
python scripts/generate_comparison_report.py
# 输出：results/reports/comparison_report.md
#       results/figures/comparison_dashboard.png
```

### 步骤 4：用 Notebook 深度分析
```bash
jupyter lab notebooks/
# 浏览器自动打开，点击任意 notebook 查看
```

### 步骤 5（可选）：正式实验
```bash
# 修改 configs/run_config.yaml → run_mode: "full_run"
caffeinate -i python scripts/run_gen1.py
caffeinate -i python scripts/run_gen2.py
caffeinate -i python scripts/run_gen3.py --no-rephrase
# caffeinate -i 防止 Mac 休眠中断任务
```

---

## 三、两种模式对比

| | smoke_test | full_run |
|---|---|---|
| 文档数量 | 1,000 | 50,000 |
| 运行时间 | 10-15 分钟 | 2-4 小时 |
| 用途 | 验证环境不报错 | 产出正式实验结果 |
| 切换方式 | `configs/run_config.yaml` 第一行 | 同左 |

---

## 四、第三代 LLM 改写配置（可选）

第三代 Pipeline 可以用 LLM 改写低质量文档（需要 API Key）。不配置也能运行，加 `--no-rephrase` 跳过即可。

```yaml
# configs/api_config.yaml
provider: "anthropic"
api_key: "YOUR_API_KEY_HERE"    # 填入你的 Anthropic API Key
                                 # 或设置环境变量：export FINEWEB_API_KEY=sk-ant-...
```

---

## 五、参考论文与资料

| 资源 | 说明 |
|---|---|
| [FineWeb 数据集](https://huggingface.co/datasets/HuggingFaceFW/fineweb) | 本项目数据来源，HuggingFace 开源 |
| [DCLM 论文（NeurIPS 2024）](https://arxiv.org/abs/2406.11794) | 第二代方法论的主要参考 |
| [Nemotron-CC 论文](https://arxiv.org/abs/2412.02595) | 第三代方法论的主要参考 |
| [datatrove](https://github.com/huggingface/datatrove) | HuggingFace 数据清洗框架，架构参考 |
| [NeMo Curator](https://github.com/NVIDIA/NeMo-Curator) | NVIDIA 数据清洗框架，Gen3 参考 |

---

> 📖 遇到问题请查看 [docs/FAQ.md](docs/FAQ.md)
> 方法论对标业界 2024 年最新 Pre-train 数据工程实践。
