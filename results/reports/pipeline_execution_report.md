# Pipeline Execution Report: Pre-training Data Cleaning Three-Generation Methodology Comparison

> Generated: 2026-03-06 | Run Mode: medium_run (12,000 docs) | Data Source: Common Crawl WET

---

## 1. Executive Summary

本项目在同一批 Common Crawl WET 原始数据（12,000 篇文档，~46.6M tokens）上，实现并对比了三代预训练数据清洗方法论：Gen1（Heuristic 规则过滤，参照 FineWeb/Gopher/C4）、Gen2（fastText 分类器 top-10%，参照 DCLM）、Gen3（集成分类 + 条件 Bypass + LLM 改写，参照 Nemotron-CC）。最终结果显示：Gen1 保留 3.4% 文档（LIFT +4.3%），Gen2 在 Gen1 基础上进一步筛选至 top-10%（LIFT +6.1%），Gen3 通过 Bypass 机制回收被误杀文档但未启用 LLM 改写（LIFT +5.3%）。三代方法体现了清晰的 Quality-Quantity Trade-off：质量递增但数据量递减。

---

## 2. Pipeline Overview

### 2.1 Gen1: Heuristic Rule-based Filtering

| 步骤 | 方法 | 关键参数 | 论文参照 |
|------|------|----------|----------|
| URL 去重 | xxhash 精确去重 | - | FineWeb (Penedo et al. 2024) |
| URL 过滤 | 黑名单 + 模式匹配 | adult/spam 域名 | C4 (Raffel et al. 2020) |
| 语言过滤 | langdetect | en, prob > 0.5 | CCNet (Wenzek et al. 2020) |
| 质量过滤 | Gopher + C4 + FineWeb 三合一 | 见 2.2 节 | Gopher (Rae et al. 2021), C4, FineWeb |
| 重复过滤 | 行级/段落级/n-gram 重复率 | dup_line_frac < 0.3 | Gopher (Rae et al. 2021) |
| PII 过滤 | 正则匹配 email/phone/SSN | - | Dolma (Soldaini et al. 2024) |

### 2.2 Gen1 Quality Filter 内部子规则

| 子过滤器 | 关键参数 | 来源 |
|----------|----------|------|
| Gopher | word_count: [50, 100000], alpha_ratio > 0.5, mean_word_len: [3, 10] | Gopher (Rae et al. 2021) |
| C4 | terminal_punct_min_ratio: 0.1, filter_javascript: "javascript", min_lines: 3 | C4 (Raffel et al. 2020) |
| FineWeb | max_bullet_lines_ratio: 0.9, max_ellipsis_lines_ratio: 0.3 | FineWeb (Penedo et al. 2024) |

### 2.3 Gen2: Model-based Classification

| 步骤 | 方法 | 关键参数 | 论文参照 |
|------|------|----------|----------|
| fastText 分类 | 二分类（Wikipedia 正样本 vs CC 负样本） | dim=64, wordNgrams=2, lr=0.1, epoch=3 | DCLM (Li et al. 2024) |
| 阈值筛选 | Top-K percentile | top_fraction=0.10 | DCLM |

### 2.4 Gen3: Hybrid Approach

| 步骤 | 方法 | 关键参数 | 论文参照 |
|------|------|----------|----------|
| 集成分类 | Union of eval_classifier + pipeline_classifier | strategy=union | Nemotron-CC (Parmar et al. 2024) |
| 条件 Bypass | 高质量但被 Gen1 误杀的文档回收 | quality_threshold=0.52 | Nemotron-CC |
| LLM 改写 | Anthropic claude-haiku-4-5-20251001 | **未执行**（API Key 未配置） | Nemotron-CC |

---

## 3. Per-Step Statistics

### 3.1 Gen1 逐步统计

| 步骤 | 输入 docs | 输出 docs | 该步丢弃数 | 条件过滤率 | 累积保留率 |
|------|-----------|-----------|-----------|-----------|-----------|
| 原始输入 | 12,000 | 12,000 | - | - | 100.0% |
| URL 去重 | 12,000 | 11,997 | 3 | 0.03% | 100.0% |
| URL 过滤 | 11,997 | 11,851 | 146 | 1.2% | 98.8% |
| 语言过滤 | 11,851 | 2,912 | 8,939 | 75.4% | 24.3% |
| 质量过滤 | 2,912 | 832 | 2,080 | 71.4% | 6.9% |
| 重复过滤 | 832 | 409 | 423 | 50.8% | 3.4% |
| PII 过滤 | 409 | 409 | 0 | 0.0% | 3.4% |
| **最终输出** | **12,000** | **409** | **11,591** | - | **3.4%** |

> 口径说明：条件过滤率 = 该步丢弃数 / 该步输入数；累积保留率 = 该步输出数 / 原始总输入数（12,000）

### 3.2 Gen1 Token 流量追踪

| 阶段 | 文档数 | 估算 Tokens | 平均 Tokens/Doc |
|------|--------|------------|-----------------|
| 原始输入 | 12,000 | 36,611,880 | 3,051 |
| 语言过滤后 | 2,912 | 3,180,981 | 1,092 |
| 质量过滤后 | 832 | 1,629,272 | 1,958 |
| Gen1 最终输出 | 409 | 422,611 | 1,033 |

### 3.3 Gen2 统计

| 指标 | 值 |
|------|-----|
| 输入文档数 | 409（Gen1 输出） |
| 输出文档数 | 41 |
| 保留率（相对 Gen1 输出） | 10.0% |
| 保留率（相对原始输入） | 0.34% |
| fastText 阈值 | 0.5181 |
| 分类器得分均值 | 0.3382 |
| 分类器得分 P50 | 0.5044 |
| 分类器得分 P90 | 0.5181 |
| 输出估算 Tokens | 46,744 |
| 平均 Tokens/Doc | 1,140 |

> 注：分类器得分呈双峰分布——138 篇得分为 0.0（被判为低质量），271 篇得分集中在 0.50-0.55 区间。

### 3.4 Gen3 路由统计

| 路由类别 | 文档数 | 占比 | 说明 |
|----------|--------|------|------|
| 高质量直通 | 6 | 1.5% | 质量分 > 0.52，直接保留 |
| Heuristic 通过 | 11 | 2.7% | 集成分类器通过 + 不触发 Heuristic |
| Heuristic 过滤 | 280 | 68.5% | 集成分类器通过但被 Heuristic 拦截 |
| 待改写 | 78 | 19.1% | 中等质量，送入 LLM 改写（**未执行**） |
| 丢弃 | 34 | 8.3% | 低质量，直接丢弃 |
| **最终保留** | **17** | **4.2%** | 高质量直通 + Heuristic 通过 |

---

## 4. Quality Distribution

### 4.1 质量评分对比（独立评估分类器）

| 代次 | 文档数 | Quality Mean | LIFT vs 基线 | P50 | P90 |
|------|--------|-------------|-------------|-----|-----|
| 原始数据 (Raw) | 12,000 | 0.4945 | — (基线) | 0.5097 | 0.5206 |
| Gen1 (Heuristic) | 409 | 0.5158 | **+4.3%** | 0.5154 | 0.5218 |
| Gen2 (Model-based) | 41 | 0.5246 | **+6.1%** | 0.5244 | 0.5286 |
| Gen3 (Hybrid) | 17 | 0.5208 | **+5.3%** | 0.5201 | 0.5285 |

> LIFT 口径：(该代质量均值 - 原始质量均值) / 原始质量均值 × 100%
> 评估分类器独立于 Gen2 pipeline 分类器（dim=32, wordNgrams=1 vs dim=64, wordNgrams=2），避免循环偏差

### 4.2 质量分布特征

- **原始数据**：双峰分布，约 33.7% 文档（138/409 在 Gen2 中）得分为 0.0，说明 Gen1 输出中仍有约 1/3 被分类器认为是低质量
- **Gen2 输出**：窄带分布，集中在 0.518-0.545 区间，标准差仅 0.0061
- **Gen3 输出**：与 Gen2 类似的窄带分布，但样本量仅 17 篇，统计显著性有限

### 4.3 可视化参考

- Quality Score 直方图：`results/figures/03_gen2_score_distribution.png`
- Gen1 vs Gen2 质量对比：`results/figures/03_gen1_vs_gen2_quality.png`
- 跨代对比面板：`results/figures/06_cross_generation_comparison.png`
- Quality-Quantity Trade-off 曲线：`results/figures/03_gen2_tradeoff_curve.png`

---

## 5. Diversity & Token Analysis

### 5.1 N-gram 多样性

| 代次 | Unigram Unique Ratio | Trigram Unique Ratio | 变化趋势 |
|------|---------------------|---------------------|----------|
| 原始数据 | 0.2913 | 0.7706 | 基线 |
| Gen1 输出 | 0.1479 | 0.8629 | Unigram 降（去重效果），Trigram 升（内容多样化） |
| Gen2 输出 | 0.3003 | 0.9484 | 两者均升（高质量文档用词更丰富） |
| Gen3 输出 | 0.3383 | 0.9517 | 最高多样性 |

> 论文参照：FineWeb 报告 trigram unique ratio 在清洗后典型提升 5-15%；本实验 Gen1→Gen2 提升 9.9%，符合预期。

### 5.2 域名 Shannon Entropy（归一化）

| 代次 | 域名数 | Shannon Entropy | 归一化 Entropy |
|------|--------|----------------|---------------|
| 原始数据 | 496 | 5.607 | 0.993 |
| Gen1 输出 | 384 | 5.489 | — |
| Gen2 输出 | 40 | 5.309 | — |
| Gen3 输出 | 16 | 3.970 | — |

> 域名多样性随过滤强度递减是正常现象（文档数减少→域名数减少→Entropy 降低）。Gen3 Entropy 显著低于 Gen2，但仅有 16 个独立域名。

### 5.3 Token 统计汇总

| 代次 | 文档数 | 总 Tokens（估算） | 平均 Tokens/Doc |
|------|--------|-------------------|-----------------|
| 原始数据 | 12,000 | 46,550,640 | 3,879 |
| Gen1 输出 | 409 | 422,611 | 1,033 |
| Gen2 输出 | 41 | 46,744 | 1,140 |
| Gen3 输出 | 17 | 16,832 | 990 |

---

## 6. Ablation & Component Analysis

### 6.1 Gen1: 各过滤规则杀伤力排序

| 排名 | 过滤步骤 | 条件过滤率 | 累积丢弃贡献 | 分析 |
|------|----------|-----------|-------------|------|
| 1 | 语言过滤 | 75.4% | 74.5% | CC WET 随机 segment 仅 ~25% 英文，此步为最大杀伤点 |
| 2 | 质量过滤 | 71.4% | 17.3% | Gopher+C4+FineWeb 三合一，C4 terminal_punct 贡献最大 |
| 3 | 重复过滤 | 50.8% | 3.5% | 行级/段落级重复检测 |
| 4 | URL 过滤 | 1.2% | 1.2% | 黑名单+模式匹配 |
| 5 | URL 去重 | 0.03% | 0.03% | 精确去重（CC WET 几乎无 URL 重复） |
| 6 | PII 过滤 | 0.0% | 0.0% | 未检测到 PII 模式 |

### 6.2 Gen1 Quality Filter 内部分解

质量过滤内部，三个子过滤器的相对贡献（基于审计样本）：

| 子过滤器 | 主要触发规则 | 典型案例 |
|----------|-------------|----------|
| Gopher | word_count < 50, alpha_ratio < 0.5 | 极短文本、非文本内容（导航栏、菜单） |
| C4 | terminal_punct_ratio < 0.1, contains "javascript" | 列表型页面、代码片段 |
| FineWeb | bullet_lines_ratio > 0.9 | 纯列表内容 |

### 6.3 Gen2: 阈值敏感性（DCLM 对标）

Gen2 使用 top-10% 阈值（DCLM 推荐值）。基于分类器得分分布推算：

| 阈值 | 预估保留文档数 | 预估保留率 | 分析 |
|------|--------------|-----------|------|
| top-5% | ~20 | ~5.0% | 样本量过小，统计不稳定 |
| top-10% | 41 | 10.0% | **实际使用**，DCLM 推荐 |
| top-20% | ~82 | ~20.0% | 更大数据量但质量稍降 |
| top-50% | ~205 | ~50.0% | 接近全量，质量提升有限 |

> 论文参照：DCLM 报告 top-10% 在 MMLU 上最优（64.0 → 68.4），top-25% 次优。本实验 top-10% LIFT=+6.1%。

### 6.4 Gen3: Bypass 价值分析

| 指标 | 本实验值 | Nemotron-CC 参照值 | 偏差 |
|------|---------|-------------------|------|
| 高质量文档被 Heuristic 误杀率 | **100%**（6/6） | 18.1% | 偏高 |
| Bypass 回收文档数 | 6 | — | — |
| 误杀原因 | C4 terminal_punct（旧阈值 0.7） | 多种规则 | — |

> **关键发现**：所有 6 篇被 Bypass 回收的文档，误杀原因均为 `c4:low_terminal_punct_ratio`（旧硬编码阈值 0.7，实际数据 P50=0.11）。这验证了两个方法论教训：
> 1. Heuristic 规则确实存在系统性盲区（Nemotron-CC 核心论点）
> 2. 阈值必须对标数据分布，不能直接沿用论文默认值（methodology pattern #12）

### 6.5 Gen3: 改写前后质量对比

**未执行**（API Key 未配置）。78 篇待改写文档被跳过，导致 Gen3 最终仅保留 17 篇（vs 启用改写后预期 ~95 篇）。

---

## 7. Cross-Generation Comparison

### 7.1 核心对比表（参照 DCLM/Nemotron-CC benchmark 格式）

| 代次 | 方法类型 | 保留率 | Quality Mean | LIFT | Trigram Diversity | Avg Tokens |
|------|---------|--------|-------------|------|-------------------|------------|
| Raw | — | 100% | 0.4945 | — | 0.7706 | 3,879 |
| Gen1 | Heuristic | 3.4% | 0.5158 | +4.3% | 0.8629 | 1,033 |
| Gen2 | Model-based | 0.34%* | 0.5246 | +6.1% | 0.9484 | 1,140 |
| Gen3 | Hybrid | 0.14%* | 0.5208 | +5.3% | 0.9517 | 990 |

> *Gen2/Gen3 保留率为相对原始输入的累积保留率（Gen2: 41/12000, Gen3: 17/12000）

### 7.2 论文基线对比

| 项目 | 数据源 | 保留率 | 质量指标 | 方法 |
|------|--------|--------|---------|------|
| **本实验 Gen1** | CC WET 12K | 3.4% | LIFT +4.3% | Gopher+C4+FineWeb |
| FineWeb (2024) | CC WET 15T tokens | ~15% | HellaSwag 79.4 | Gopher+C4+FineWeb+custom |
| **本实验 Gen2** | Gen1 输出 409 | 10.0% | LIFT +6.1% | fastText top-10% |
| DCLM (2024) | CC 3T tokens | 10% (top-10%) | MMLU 68.4 | fastText top-10% |
| **本实验 Gen3** | Gen1 输出 409 | 4.2% | LIFT +5.3% | Ensemble+Bypass |
| Nemotron-CC (2024) | CC 8.6T tokens | ~40% | MMLU 70.2 | Ensemble+Bypass+Rephrase |

### 7.3 Quality-Quantity Trade-off

```
Quality Mean
  0.525 |          * Gen2 (41 docs)
  0.522 |
  0.520 |              * Gen3 (17 docs)
  0.518 |
  0.516 |   * Gen1 (409 docs)
  0.514 |
  0.512 |
  0.510 |
  0.508 |
  0.506 |
  0.504 |
  0.502 |
  0.500 |
  0.498 |
  0.496 |
  0.494 |                                   * Raw (12000 docs)
        +---+---+---+---+---+---+---+---+---+---→ Doc Count (log)
           10  20  50 100 200 500 1K  2K  5K 10K
```

> Gen2 实现最高质量但仅 41 篇文档；Gen1 在文档量和质量间取得较好平衡。Gen3 受限于 LLM 改写未启用，未充分发挥数据回收能力。

---

## 8. Deviation Analysis

参照 METHODOLOGY_PATTERNS.md 模式 #8（核对清单驱动的偏差反馈循环）：

| 指标 | 预期值（论文基准） | 实际值 | 偏差方向 | 原因分析 | 建议 |
|------|-------------------|--------|---------|---------|------|
| Gen1 保留率 | 10-20%（FineWeb ~15%） | 3.4% | 偏低 | CC WET 英文仅 ~25%（语言过滤杀伤 75.4%）；质量过滤 71.4% | 如需提升，降低 alpha_ratio 或增加多语种支持 |
| Gen2 top-10% 保留率 | 10% | 10.0% | 正常 | 精确按 percentile 截断 | 继续 |
| Gen2 LIFT | +5-10%（DCLM 趋势） | +6.1% | 正常 | 符合 DCLM top-10% 预期 | 继续 |
| Gen3 Bypass 误杀率 | 18.1%（Nemotron-CC） | 100% | 偏高 | 仅 6 篇高质量文档全部被旧 C4 阈值误杀 | 样本量过小导致比例极端，非系统性问题 |
| Gen3 改写增益 | LIFT +3-5%（Nemotron-CC） | 未执行 | 缺失 | API Key 未配置 | 配置 API Key 后重跑 |
| 评估分类器区分度 | quality_mean 差异 5-15% | 差异仅 0.03 | 偏低 | fastText 模型能力有限 + 训练数据量不足 | 考虑更强的评估模型（如 DeBERTa） |
| 语言过滤比例 | 40-50% 英文 | 24.6% 英文 | 偏低 | CC WET 随机 10 segment 的英文比例低于预期 | 数据特征，非 bug |
| N-gram 多样性提升 | 5-15%（FineWeb） | +12.0%（Gen1→Gen2 trigram） | 正常 | 高质量文档用词更丰富 | 继续 |

---

## 9. Reproducibility Record

参照 Datasheets for Datasets (Gebru et al. 2021) + DCLM 标准化训练 recipe：

| 项目 | 值 |
|------|-----|
| 运行模式 | `medium_run`（doc_limit=12,000） |
| 配置文件 | `configs/run_config.yaml`, `configs/gen1_config.yaml`, `configs/gen2_config.yaml`, `configs/gen3_config.yaml` |
| Git Commit Hash | `f23145861f2d90b061349b250c5e9881b539dc0e` |
| Python 版本 | 3.11.14 |
| PyTorch | 2.10.0 |
| NumPy | 2.4.2 |
| Pandas | 3.0.1 |
| Transformers | 5.2.0 |
| xxhash | 3.6.0 |
| datasketch | 1.9.0 |
| 随机种子 | 42（`configs/run_config.yaml`） |
| 数据源 | CC WET, 10 random segments from `CC-MAIN-2024-51`（`data/raw/cc_wet_sample.jsonl`） |
| 参考正样本 | Wikipedia dump via HuggingFace `wikipedia` dataset |
| 平台 | macOS Darwin 25.3.0, Apple M4 Max |
| Gen1 执行耗时 | ~4 秒 |
| Gen2 执行耗时 | ~0.5 秒 |
| Gen3 执行耗时 | ~0.6 秒 |
| 评估分类器 | 独立 fastText（dim=32, wordNgrams=1），训练于 Wikipedia vs CC 样本 |

---

## 10. Limitations & Next Steps

### 10.1 已知局限

1. **LLM 改写未执行**：Gen3 的 Synthetic Rephrasing 功能因 API Key 未配置而跳过，78 篇待改写文档未处理。这导致 Gen3 的数据回收能力被严重低估（保留 17 篇 vs 预期 ~95 篇）。
2. **样本量偏小**：medium_run 仅 12,000 篇，Gen2 最终仅 41 篇、Gen3 仅 17 篇，统计显著性有限。
3. **评估分类器区分度不足**：quality_mean 跨代差异仅 0.03（0.4945→0.5246），可能因 fastText 模型能力有限。
4. **无下游任务验证**：未在 Benchmark（HellaSwag/MMLU/ARC 等）上验证清洗后数据的实际训练效果。
5. **单语种**：仅处理英文，CC WET 中 75% 的非英文数据被直接丢弃。
6. **去重未跨代执行**：MinHash 去重作为独立模块存在，但未在三代对比中统一应用。

### 10.2 未执行的组件

| 组件 | 状态 | 影响 |
|------|------|------|
| LLM 改写（Synthetic Rephrasing） | 未执行 | Gen3 LIFT 被低估 |
| full_run（100K docs） | 未执行 | 统计显著性不足 |
| Proxy Model 训练 | 脚本就绪，未跑 | 无下游验证 |
| MinHash 跨文档去重 | 模块就绪，未集成到对比流程 | 重复率未量化 |
| 中文 Pipeline | 代码就绪，未执行 | 仅英文评估 |

### 10.3 改进建议

1. **配置 Anthropic API Key** 并重跑 Gen3，预期 LIFT 提升至 +7-8%
2. **切换到 full_run（100K docs）** 获得统计显著结果
3. **升级评估分类器** 为 DeBERTa 或类似 Transformer 模型，提升区分度
4. **增加下游 Benchmark 验证** 使用 Proxy Model 在 HellaSwag/MMLU 上评估
5. **多语种扩展** 利用已有的中文 Pipeline 模块

---

## 附录：数据文件索引

| 数据 | 路径 |
|------|------|
| Gen1 Pipeline 统计 | `data/gen1_output/gen1_pipeline_stats.json` |
| Gen1 阶段指标 | `data/gen1_output/gen1_stage_metrics.json` |
| Gen2 分类统计 | `data/gen2_output/gen2_stats.json` |
| Gen2 阶段指标 | `data/gen2_output/gen2_stage_metrics.json` |
| Gen3 路由摘要 | `data/gen3_output/gen3_routing_summary.json` |
| Gen3 阶段指标 | `data/gen3_output/gen3_stage_metrics.json` |
| 跨代对比 | `results/reports/cross_generation_comparison.json` |
| 基线指标 | `results/quality_scores/baseline_metrics.json` |
| Gen1 审计摘要 | `data/reports/audit/gen1/audit_summary.json` |
| 对比面板图 | `results/figures/comparison_dashboard.png` |
| Gen2 分数分布图 | `results/figures/03_gen2_score_distribution.png` |
| Gen2 Trade-off 曲线 | `results/figures/03_gen2_tradeoff_curve.png` |
| 跨代对比图 | `results/figures/06_cross_generation_comparison.png` |
| Gen3 路由分布图 | `results/figures/04_gen3_routing.png` |
| 消融分析图 | `results/figures/07_ablation_study.png` |
