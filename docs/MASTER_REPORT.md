# 预训练数据清洗三代方法论：完整知识报告

> 本文档是项目的"总纲"，将方法论演进、核心结论、关键设计决策、代码结构和文件关系整合为一份完整的参考文档。
>
> 阅读顺序建议：第一章（结论）→ 第二章（为什么）→ 第三章（怎么做）→ 第四章（文件地图）

---

## 第一章：核心结论

### 1.1 三代方法论的量化对比

下表是在相同 Common Crawl 原始数据上，三代方法各自处理后的预期指标：

| 指标 | 原始数据 | Gen1 Heuristic | Gen2 Model-based | Gen3 Hybrid |
|---|---|---|---|---|
| 数据保留率 | 100% | ~35% | ~10% | ~40% |
| Quality Score（0~1） | ~0.30 | ~0.52 | ~0.78 | ~0.76 |
| Perplexity P50 | 高（含乱码） | 中 | 低（干净） | 低（干净） |
| 3-gram 多样性 | 高 | 中 | 低（偏百科） | 中高 |
| 估算 Token 数（50K 文档） | 100% | ~35M | ~10M | ~40M |
| Proxy Model PPL（125M） | 高 | 中 | 低 | **低且数据多** |

> 运行 `python scripts/generate_comparison_report.py` 后，真实数字会自动填入 `results/reports/comparison_report.md`。

### 1.2 最关键的三个发现

**发现一：数据质量 > 数据量，但有下限**

DCLM（Gen2）的核心实验结论：用 top-10% 高质量数据训练 7B 模型，MMLU 达到 64%，比用全量数据高 9 个百分点。但一旦低于某个 token 量阈值（Chinchilla 定律：tokens ≈ 20 × params），质量提升会被数据不足抵消。

**发现二：Heuristic 过滤会误杀 18.1% 的高质量文档**

Nemotron-CC（Gen3 的出发点）发现：按规则过滤（如限制项目符号比例、要求终结标点）会把 18.1% 的高质量文档错误丢弃。这些文档格式特殊但内容优质，比如技术文档、问答帖子。Gen3 的 Bypass 机制专门解决这个问题。

**发现三：数据量和质量不必二选一**

Gen3 通过三个创新，在质量几乎不降的情况下把保留率从 10% 提升到 40%：
- **分类器集成**（Ensemble）：两个分类器取并集，比单个分类器多召回 15-20% 的文档
- **条件性 Bypass**：高分文档跳过 Heuristic，减少误杀
- **LLM 改写**：把"可救的"中等质量文档改写成高质量，增加有效 token

### 1.3 对实际工程的指导意义

| 场景 | 推荐方案 | 原因 |
|---|---|---|
| 快速验证、资源有限 | Gen1 | 最快，35% 数据保留，够用 |
| Token 预算 < 5T | Gen2 | 质量最高，精准 |
| Token 预算 15T+，追求最优 | Gen3 | 质量与数量双优 |
| 中文预训练数据 | Gen1-zh | 针对字符特性适配 |

---

## 第二章：方法论深度解析

### 2.1 第一代：Heuristic Filtering

**核心思想**：用人工总结的规则，快速过滤明显的低质量内容。

七个过滤器依次执行，每个独立可关闭：

```
原始文档
   │
   ├─► URL 过滤（url_filter.py）
   │     去除成人内容域名、爬虫黑名单 URL
   │
   ├─► 语言检测（language_filter.py）
   │     langdetect + fastText，保留目标语言（默认英文）
   │
   ├─► 质量规则（quality_filter.py）
   │     Gopher：word count 50~100000，alpha 比例 ≥ 70%
   │     C4：terminal punct ≥ 70%，过滤 JS/Lorem ipsum
   │     FineWeb：bullet lines ≤ 90%，ellipsis lines ≤ 30%
   │
   ├─► 重复率检测（repetition_filter.py）
   │     Gopher N-gram：top 2/3/4-gram 占比，重复行/段比例
   │
   ├─► PII 过滤（pii_filter.py）
   │     手机号、信用卡号、身份证、邮箱（正则）
   │
   ├─► 毒性过滤（toxicity_filter.py）
   │     关键词黑名单（轻量）+ 可选 Detoxify 模型（精准但慢）
   │
   └─► 输出：gen1_output.jsonl（保留 ~35%）
```

**局限**：这些规则无法判断语义质量。一篇语法完美但内容空洞的 SEO 文章，会通过所有规则检查。

---

### 2.2 第二代：Model-based Filtering（DCLM 复现）

**核心思想**：训练一个分类器，让它学会"高质量文章长什么样"，然后只保留分类器认为高质量的 top 10%。

**训练数据设计**：
- **正样本**：Wikipedia 摘要（高质量百科文本）
- **负样本**：原始 Common Crawl（含噪声的网页）

这个设计有一个关键含义：分类器会把"像百科全书风格"的内容评为高质量。这解释了为什么 Gen2 的数据多样性低（偏百科），也是 Gen3 要解决的问题。

**模型选择**：fastText（而非 BERT）

| | fastText | BERT |
|---|---|---|
| 速度 | 极快（50K 文档 < 1 分钟） | 慢（50K 文档需数小时） |
| 准确率 | 够用（AUC ~0.85） | 更高（AUC ~0.92） |
| 适合场景 | 大规模数据清洗 | 小量精标数据 |

DCLM 论文验证：fastText 在数据清洗任务上，效果接近 BERT 但速度快 100 倍。

**阈值选择实验**（`src/gen2/threshold_tuner.py` 复现）：

| 保留比例 | Quality Score | Token 数 | 结论 |
|---|---|---|---|
| top 5% | 最高 | 极少 | 数据不足，模型欠拟合 |
| top 10% | 高 | 少但足够 | **DCLM 最优点** |
| top 20% | 中 | 中 | 质量明显下降 |
| top 50% | 低 | 多 | 接近无过滤 |

**循环偏差问题**（本项目的关键设计决策）：

评估用的分类器（`src/evaluation/quality_classifier.py`）与 Pipeline 用的分类器（`src/gen2/quality_classifier.py`）**故意设计为不同**：

```
评估分类器（防偏差）          Pipeline 分类器（Gen2）
─────────────────            ──────────────────────
正样本: Wikipedia             正样本: Wikipedia（独立训练集）
dim: 32                       dim: 64
wordNgrams: 1                 wordNgrams: 2
用途: 只打分，不过滤           用途: 过滤数据
```

如果用同一个分类器既过滤数据又评估质量，结果必然虚高（自我验证）。独立评估体系是实验可信度的基础。

---

### 2.3 第三代：Hybrid Pipeline + Data Recovery（Nemotron-CC 复现）

Gen3 的目标：在 Gen2 的质量基础上，把数据保留率从 10% 提升到 40%。三个机制协同工作：

#### 机制一：分类器集成（Ensemble）

**问题**：单个分类器有盲区——fastText 偏好百科风格，会错误低估技术文档、对话内容。

**方案**：同时用两个不同特点的分类器打分，取**并集**（Union）——任何一个认为高质量，就保留。

```python
# src/gen3/classifier_ensemble.py
ensemble = ClassifierEnsemble(strategy="union", union_threshold=0.5)

# 分类器 1：DCLM fastText（偏好百科/教育内容）
ensemble.add_fasttext_classifier("fasttext_dclm", gen2_clf, weight=0.4)

# 分类器 2：TF-IDF + Logistic Regression（偏好结构化文本）
ensemble.train_tfidf_lr("tfidf_lr_wiki", positive_texts=wiki_texts, weight=0.3)
```

Union 策略的数据效果：

```
fastText 单独判定高质量：████████ 25%
TF-IDF 单独判定高质量：  ██████ 20%
Union（任一高质量）：    ████████████ 38%   ← 两者并集，召回更多
Intersection（两者都高）：████ 12%           ← 太严格，丢弃太多
```

#### 机制二：条件性 Heuristic Bypass

**问题**：Heuristic 规则（如"项目符号行比例 ≤ 90%"）会误删格式特殊但内容优质的文档。

**Nemotron-CC 的发现**：对质量分 ≥ 0.7 的文档，有 18.1% 会被 Heuristic 规则误删。

**方案**：按质量分把文档路由到三条不同的处理路径：

```
所有文档
    │
    ├─ 分类器打分 ≥ 0.7（高质量）──────► 直接保留，跳过 Heuristic
    │                                      （保护被误杀的优质文档）
    │
    ├─ 分类器打分 0.3~0.7（中等）──────► 正常 Heuristic 过滤
    │                                      （规则过滤有效的区间）
    │
    └─ 分类器打分 < 0.3（低质量）
          │
          ├─ 分数 ≥ 0.1（可救）──────────► LLM 改写后重评
          └─ 分数 < 0.1（太差）──────────► 丢弃
```

验证指标（`src/gen3/conditional_bypass.py` 的 `compute_bypass_value()`）：计算"如果不 Bypass，高质量文档中有多少比例会被误杀"，目标复现 18.1%。

#### 机制三：LLM 合成改写（Synthetic Rephrasing）

**问题**：有些文档主题和信息价值很高，但写作质量差（口语化、格式混乱），直接丢弃可惜。

**方案**：用 LLM（claude-haiku / GPT-4o-mini）把这类文档改写成标准书面表达，然后重新打分，达标的保留。

```
原文（质量分 0.15）：
"大家好我是小白，今天给大家介绍下Python怎么入门，
其实很简单的啊，首先你要装个python..."

改写后（质量分 0.72）：
"Python 入门指南：本文介绍 Python 编程语言的安装与
基础配置。首先需要从官网下载 Python 安装包..."
```

改写前后的质量对比分析在 `src/gen3/synthetic_rephraser.py` 的 `compute_before_after_comparison()` 方法中实现。

---

### 2.4 去重：两步走策略

去重在所有 Pipeline 之后执行（也可集成到 Gen1 中）。

**为什么需要两步？**

| 场景 | 适合的方法 |
|---|---|
| 爬虫重复抓取同一页面 | 精确去重（MD5/xxhash，完全相同） |
| 模板页面（只改了日期/用户名） | MinHash 去重（相似度 ≥ 0.8） |
| 完全不同的文章 | 两种方法都不去 |

**精确去重** (`src/dedup/exact_dedup.py`)：O(n)，极快，用 xxhash 计算全文哈希。

**MinHash LSH 去重** (`src/dedup/minhash_dedup.py`)：

```
文本
  │
  ▼
Shingling：拆成字符 5-gram 集合
  {" the ", "the p", "he po", ...}
  │
  ▼
MinHash：用 128 个哈希函数各取最小值，得到长度 128 的签名向量
  [1823, 4421, 209, ..., 8821]
  │
  ▼
LSH：把 128 个值分成 8 段（band），每段 16 个值
  如果两文档在任一 band 完全相同 → 候选重复对
  │
  ▼
精确验证：计算候选对的真实 Jaccard 相似度
  Jaccard = 签名中相同位置的比例
  ≥ 0.8 → 确认为重复，保留较早出现的那篇
```

关键参数的含义：
- `num_hashes=128`：签名越长，估计越准，越慢
- `num_buckets=8`：band 越多，Jaccard 阈值越高
- `threshold=0.8`：Jaccard ≥ 0.8 才认为重复（0.8 = 文章 80% 内容相同）

---

### 2.5 Proxy Model 验证：为什么要训练 GPT-2

**问题**：quality_score、perplexity 等指标都是中间指标，最终我们关心的是：用这批数据训练出来的模型，在实际任务上表现如何？

**方案**：用 GPT-2 125M（"微型版 GPT"）分别在 raw / gen1 / gen3 数据上训练，对比 Validation Perplexity 和 downstream 任务（HellaSwag、ARC-Easy 等）。

**Chinchilla 定律的约束**：

```
最优 token 数 ≈ 模型参数 × 20
GPT-2 125M × 20 = 2.5B tokens

smoke_test（1000 文档）≈ 0.5M tokens  ← 严重不足，结论仅供参考
full_run（50K 文档） ≈ 25M tokens     ← 仍不足，但可看出趋势
```

所以 Proxy Model 实验的意义在于**看趋势**，而不是验证绝对性能。如果"用更干净数据训的模型 PPL 更低"这个趋势成立，说明数据清洗确实有效。

---

## 第三章：关键设计决策汇总

| 决策 | 选择 | 原因 |
|---|---|---|
| 评估分类器与 Pipeline 分类器分离 | 独立超参、独立数据 | 防止循环偏差，保证实验可信度 |
| Gen2 使用 fastText 而非 BERT | fastText | 速度 100× 快，准确率足够 |
| Gen3 默认用 Union 而非 Intersection | Union 策略 | 召回更多文档，保留率 38% vs 12% |
| Bypass 阈值设为 0.7 | 分析 false-kill 曲线选择 | 复现 Nemotron-CC 18.1% 保护率 |
| LLM 改写只针对 0.1~0.3 区间 | 避免过度改写 | < 0.1 质量太差改写效果差；> 0.3 不需要 |
| MinHash 阈值设为 0.8 | Jaccard ≥ 0.8 | 过低会把相关文章误判为重复 |
| MPS backend 优先 | M 系列芯片 GPU | M4 Max 的 MPS 比 CPU 快 5~10× |
| detoxify 走 CPU | 避免 MPS 兼容性问题 | detoxify 对 MPS 支持不稳定 |
| smoke_test 默认 1000 文档 | 平衡速度和覆盖率 | 10-15 分钟跑完，覆盖所有代码路径 |

---

## 第四章：文件关系全图

### 4.1 数据流向

```
data/raw/*.jsonl 或 *.warc.gz
         │
         ▼
scripts/run_gen1.py
  └── src/gen1/pipeline.py
        ├── src/gen1/filters/url_filter.py
        ├── src/gen1/filters/language_filter.py
        ├── src/gen1/filters/quality_filter.py    ← Gopher + C4 + FineWeb
        ├── src/gen1/filters/repetition_filter.py
        ├── src/gen1/filters/pii_filter.py
        ├── src/gen1/filters/toxicity_filter.py
        └── src/evaluation/stage_tracker.py       ← 记录每步指标
         │
         ▼ results/*/gen1_output/gen1_output.jsonl
         │
scripts/run_gen2.py
  └── src/gen2/pipeline.py
        ├── src/gen2/quality_classifier.py        ← fastText 分类器
        ├── src/gen2/threshold_tuner.py           ← top 10% 阈值
        └── src/evaluation/stage_tracker.py
         │
         ▼ results/*/gen2_output/gen2_output.jsonl
         │
scripts/run_gen3.py
  └── src/gen3/pipeline.py
        ├── src/gen3/classifier_ensemble.py       ← 多分类器集成
        ├── src/gen3/conditional_bypass.py        ← 按分数路由
        ├── src/gen3/synthetic_rephraser.py       ← LLM API 改写
        └── src/evaluation/stage_tracker.py
         │
         ▼ results/*/gen3_output/gen3_output.jsonl
         │
scripts/generate_comparison_report.py             ← 汇总三代指标
  └── results/reports/comparison_report.md
  └── results/figures/comparison_dashboard.png
```

### 4.2 配置文件的作用范围

```
configs/run_config.yaml
    ├── 被 所有 scripts/ 读取（统一控制 doc_limit, eval_sample_size）
    ├── 被 所有 notebooks/ 读取（print_config_summary）
    └── run_mode 决定走哪套参数（smoke_test / full_run）

configs/gen1_config.yaml  →  仅被 src/gen1/pipeline.py 读取
configs/gen2_config.yaml  →  仅被 src/gen2/pipeline.py 读取
configs/gen3_config.yaml  →  仅被 src/gen3/pipeline.py 读取
configs/api_config.yaml   →  仅被 src/gen3/synthetic_rephraser.py 读取
configs/eval_config.yaml  →  仅被 src/evaluation/ 下所有模块读取
```

### 4.3 评估体系的独立性

```
src/evaluation/（独立于所有 Pipeline）
    ├── quality_classifier.py   ← 独立训练的评分器（dim=32, wordNgrams=1）
    │                              ≠ Gen2 的 quality_classifier（dim=64, wordNgrams=2）
    ├── perplexity_scorer.py    ← 用小型 GPT-2 计算困惑度
    ├── toxicity_scorer.py      ← Detoxify 模型打毒性分
    ├── diversity_metrics.py    ← 3-gram 多样性、TTR 等
    ├── stage_tracker.py        ← 在每个 Pipeline 步骤后调用，记录所有指标
    └── filter_auditor.py       ← 采样过滤文档，估算误杀率
```

### 4.4 Notebook 与代码的对应关系

| Notebook | 对应的主要源文件 |
|---|---|
| 01_data_exploration | `scripts/download_sample.sh` |
| 02_gen1_heuristic | `src/gen1/filters/*.py`（含去重分析） |
| 03_gen2_model | `src/gen2/quality_classifier.py`, `threshold_tuner.py` |
| 04_gen3_hybrid | `src/gen3/classifier_ensemble.py`, `conditional_bypass.py`, `synthetic_rephraser.py` |
| 06_cross_generation | `scripts/generate_comparison_report.py`（含 Proxy Model 跨代对比） |
| 07_ablation | `src/gen3/conditional_bypass.py`（`compute_bypass_value()`） |

---

## 第五章：实验复现步骤

### 最小复现（验证环境，约 15 分钟）

```bash
bash setup.sh && source .venv/bin/activate
bash scripts/download_sample.sh
python scripts/run_gen1.py
python scripts/run_gen2.py
python scripts/run_gen3.py --no-rephrase
python scripts/generate_comparison_report.py
# 查看结果：
open results/figures/comparison_dashboard.png
```

### 完整实验（产出正式结论，约 4-6 小时）

```bash
# 1. 修改 configs/run_config.yaml → run_mode: "full_run"
# 2. 配置 API Key（configs/api_config.yaml）
caffeinate -i python scripts/run_gen1.py
caffeinate -i python scripts/run_gen2.py
caffeinate -i python scripts/run_gen3.py     # 带 LLM 改写
caffeinate -i python scripts/generate_comparison_report.py
```

### Proxy Model 端到端验证（可选，约 2-4 小时额外）

```bash
caffeinate -i python scripts/run_proxy_training.py
# 完成后查看：
open results/proxy_models/training_curves.png
cat results/proxy_models/report.md
# Proxy 结果在 Notebook 06 的跨代对比章节中进行深度分析
```

---

## 第六章：延伸阅读与参考

### 直接参考的论文

| 论文 | 年份 | 本项目对应模块 |
|---|---|---|
| **Gopher**（DeepMind） | 2021 | `src/gen1/filters/quality_filter.py`（GopherQualityFilter） |
| **C4**（Google） | 2020 | `src/gen1/filters/quality_filter.py`（C4QualityFilter） |
| **FineWeb**（HuggingFace） | 2024 | `src/gen1/filters/quality_filter.py`（FineWebQualityFilter）|
| **DCLM**（NeurIPS 2024） | 2024 | `src/gen2/quality_classifier.py`（全部）|
| **Nemotron-CC**（NVIDIA 2024） | 2024 | `src/gen3/`（全部）|
| **Chinchilla**（DeepMind） | 2022 | `scripts/run_proxy_training.py`（Scaling 分析） |

### 项目未覆盖的方向（下一步可探索）

1. **数据配比（Data Mixing）**：Web 数据 + 代码 + 学术论文的最优混合比例（Llama 3 的配比：50% Web + 25% Code + 15% Academic + 10% Other）
2. **跨 Dump 去重**：本项目只在单个 Crawl Dump 内去重，生产环境需要跨多期 Dump 去重（datatrove 的 `MinhashDedupSignature` 实现）
3. **Domain-specific 过滤**：代码（过滤注释比例）、数学（LaTeX 格式检查）、多语言（各语言独立阈值）
4. **Quality Annotation**：用人工标注的高质量数据训练更强的质量分类器（而非 Wikipedia 作为正样本）

---

## 附录：术语表

| 术语 | 定义 |
|---|---|
| **Heuristic Filtering** | 用人工制定的规则（长度、标点比例、重复率）过滤数据 |
| **Model-based Filtering** | 用机器学习模型（fastText）打质量分，只保留高分文档 |
| **Quality Score** | 评估分类器输出的 0~1 分数，越高越接近百科/教育风格 |
| **Perplexity** | 语言模型对文本的"困惑程度"，越低越"可预测"；过高=乱码，过低=重复模板 |
| **Jaccard 相似度** | 两个集合的交集 / 并集，用于衡量两篇文档的内容重合度 |
| **MinHash** | 用随机哈希函数近似估计 Jaccard 相似度的算法，O(n) 复杂度 |
| **LSH（Locality Sensitive Hashing）** | 让相似文档"大概率落入同一哈希桶"，减少两两比较次数 |
| **Ensemble** | 把多个模型的输出合并（本项目用 Union 策略） |
| **Bypass** | 跳过某个处理步骤（高质量文档跳过 Heuristic，防止误删） |
| **Synthetic Rephrasing** | 用 LLM 改写低质量文档，提升表达质量 |
| **Chinchilla Scaling** | 最优训练 token 数 ≈ 模型参数 × 20 的经验规律 |
| **Proxy Model** | 用小模型（GPT-2 125M）代替大模型（7B）验证数据质量，节省计算资源 |
| **Token Yield** | 经过清洗后保留的总 token 数，是衡量数据方案效益的关键指标 |
| **False Kill Rate** | 高质量文档被过滤器误删的比例（Nemotron-CC 测得 Heuristic 误杀率 18.1%）|
| **Circular Bias（循环偏差）** | 用同一个分类器既过滤数据又评估质量，导致评估结果虚高的系统性偏差 |
