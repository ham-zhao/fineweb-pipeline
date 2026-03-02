# FAQ — 常见问题解答

## 环境与安装

**Q: setup.sh 运行失败，提示 Python 版本不对怎么办？**

A: 项目需要 Python 3.10+。macOS 自带的 Python 是 3.9，需要单独安装：
```bash
brew install python@3.11
# 然后重新运行
bash setup.sh
```

**Q: `pip install` 安装 fasttext 失败（编译报错）怎么办？**

A: 直接安装预编译的 wheel：
```bash
pip install fasttext-wheel
```
如果仍然失败，确认已安装 Xcode Command Line Tools：
```bash
xcode-select --install
```

**Q: 安装 torch 很慢，有没有快速方法？**

A: M4 Max 建议直接安装最新版（自带 MPS 支持）：
```bash
pip install torch torchvision torchaudio
```
不需要指定 CUDA 版本。

---

## 运行 Pipeline

**Q: 第一次运行应该从哪个脚本开始？**

A: 按顺序：
```bash
bash scripts/download_sample.sh   # 下载样本数据
python scripts/run_gen1.py        # 第一代
python scripts/run_gen2.py        # 第二代
python scripts/run_gen3.py        # 第三代
python scripts/generate_comparison_report.py  # 生成对比报告
```

**Q: smoke_test 和 full_run 有什么区别，我应该用哪个？**

A: 在 `configs/run_config.yaml` 中切换：

| 模式 | 文档数量 | 运行时间 | 用途 |
|---|---|---|---|
| `smoke_test` | 1,000 | 10-15 分钟 | 验证环境和代码不报错 |
| `full_run` | 50,000 | 2-4 小时 | 产出正式实验结果 |

**建议**：先跑 smoke_test 确认没问题，再切换 full_run 跑完整实验。

**Q: 运行 run_gen3.py 提示没有 API Key 怎么办？**

A: 加 `--no-rephrase` 参数跳过 LLM 改写步骤（不需要 API Key）：
```bash
python scripts/run_gen3.py --no-rephrase
```
这样第三代 Pipeline 仍会运行集成分类和 Bypass，只跳过改写部分。

**Q: 电脑会进入休眠导致任务中断怎么办？**

A: 在命令前加 `caffeinate -i`：
```bash
caffeinate -i python scripts/run_gen1.py
```

**Q: 运行中途失败了，能从断点继续吗？**

A: 目前不支持断点续跑。但各阶段输出会保存为 JSONL，下游阶段可以直接读取已有输出：
- Gen2 会自动读取 `results/*/gen1_output/gen1_output.jsonl`
- Gen3 会自动读取 `results/*/gen1_output/gen1_output.jsonl`

只需重新运行失败的那一代即可。

---

## 数据与结果

**Q: 数据下载不了（download_sample.sh 报错）怎么办？**

A: `download_sample.sh` 支持多个数据源，如果 HuggingFace 连接超时，可以：
1. 配置镜像：`export HF_ENDPOINT=https://hf-mirror.com`
2. 或手动下载 FineWeb 样本放到 `data/raw/`

**Q: gen1/gen2/gen3 的输出文件在哪里？**

A: 默认在 `results/smoke_test/` 或 `results/full_run/` 下：
```
results/
├── smoke_test/
│   ├── gen1_output/gen1_output.jsonl
│   ├── gen2_output/gen2_output.jsonl
│   └── gen3_output/gen3_output.jsonl
└── full_run/  （切换 run_mode 后）
```

**Q: 对比报告在哪里，怎么看？**

A: 运行 `generate_comparison_report.py` 后生成：
- `results/reports/comparison_report.md` — Markdown 表格（用任何 Markdown 阅读器打开）
- `results/figures/comparison_dashboard.png` — 4 格可视化图表

**Q: quality_score 是怎么计算的，越高越好吗？**

A: 是的，越高越好（范围 0~1）。由一个**独立训练的** fastText 分类器打分（正样本为 Wikipedia，负样本为原始 CC 数据）。这个评估分类器与 Gen2 Pipeline 中的分类器**完全独立**（不同超参、不同训练数据），避免循环偏差。

---

## 第三代 Pipeline

**Q: 什么是 Classifier Ensemble（分类器集成）？**

A: Gen3 同时用两个分类器打分，取"两者之一认为高质量即保留"（Union 策略）：
1. Gen2 的 fastText 分类器
2. TF-IDF + Logistic Regression（Wikipedia 正样本训练）

Union 策略相比单个分类器，数据召回率更高，能保留更多 unique token。

**Q: 什么是 Conditional Bypass（条件性 Heuristic 绕过）？**

A: 对高质量文档（质量分 ≥ 0.7）跳过 Heuristic 过滤，直接保留。原因：Nemotron-CC 发现 18.1% 的高质量文档会被 Heuristic 规则误删（假阳性）。Bypass 通过保护这些文档，提高数据召回率。

**Q: LLM 改写（Synthetic Rephrasing）的 API 费用大概是多少？**

A: 取决于改写数量（由 `rewrite_count` 参数控制）：
- smoke_test：50 条，使用 claude-haiku，费用 < ¥0.1
- full_run：300 条，费用约 ¥0.5-1

改写只针对"中等质量"文档（分数 0.1~0.3），占比通常较少。

---

## 去重

**Q: 精确去重和 MinHash 去重有什么区别？**

A:
| | 精确去重 | MinHash 去重 |
|---|---|---|
| 检测对象 | 完全相同的文档 | 高度相似（≥80% Jaccard）的文档 |
| 算法 | MD5/xxhash | MinHash LSH |
| 速度 | O(n)，极快 | O(n)，较快 |
| 适用场景 | 爬虫重复抓取 | 模板页面、轻微改写的重复内容 |

**建议**：先跑精确去重，再跑 MinHash 去重（本项目 Gen1 Pipeline 已按此顺序组合）。

---

## Proxy Model 训练

**Q: Proxy Model 训练需要多长时间？**

A: M4 Max 上：
- smoke_test（1000 文档）：约 5-10 分钟
- full_run（50K 文档）：约 2-4 小时

加 `caffeinate -i` 防止休眠。

**Q: Proxy Model 训练需要 GPU 吗？**

A: 不强制要求，但 MPS 加速效果明显。脚本会自动检测并使用 MPS（M 系列芯片专用 GPU backend）。如果 MPS 不可用，自动回退到 CPU。

**Q: 什么是 Chinchilla Scaling Law？**

A: 2022 年 DeepMind 提出的最优训练配比规律：**最优 token 数 ≈ 模型参数量 × 20**。例如 GPT-2 125M 需要约 2.5B token 才能达到最优效果。smoke_test 的数据量远小于此，所以训练结果仅供对比参考，不代表真实预训练场景。

---

## Notebook 使用

**Q: 如何打开 Notebook？**

A:
```bash
source .venv/bin/activate
jupyter lab notebooks/
```
或用 VS Code 打开 `.ipynb` 文件（需安装 Jupyter 扩展）。

**Q: 建议先看哪个 Notebook？**

A: 按目的推荐：
- **理解方法论**：从 `00_methodology_overview` 开始
- **看实验结果**：直接看 `06_cross_generation_comparison`（核心对比）
- **深度研究**：按 01 → 02 → 03 → 04 → 05 → 06 → 07 顺序

**Q: Notebook 里的图表没有显示怎么办？**

A: 确保 `results/` 目录下已有 pipeline 输出。Notebook 内置了 Mock 数据 fallback，如果没有真实数据会显示示例图表结构。

---

## 中文扩展

**Q: 中文 Pipeline 怎么用？**

A:
```python
from src.gen1_zh.pipeline import ChineseGen1Pipeline

pipeline = ChineseGen1Pipeline(run_config=cfg, pipeline_config=pipe_cfg)
filtered_docs = pipeline.run(docs)
pipeline.save(filtered_docs, Path("results/gen1_zh_output"))
```

**Q: 中文 Pipeline 支持繁体中文吗？**

A: 支持。繁体和简体文档均会保留，并在 `doc["zh_script"]` 字段标注 `simplified`/`traditional`/`mixed`，下游可按需过滤。

**Q: 中文文档的长度规则和英文不一样，为什么？**

A: 中文没有空格分词，不能用 word count 衡量长度。改用 CJK 字符数：
- 最短：100 个汉字（≈ 1-2 个段落）
- 最长：50,000 个汉字（≈ 100 页）
这与 WuDao、ERNIE 等中文预训练数据集的清洗规则一致。
