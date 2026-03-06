请为当前项目的数据 pipeline 执行结果生成一份结构化报告。

报告基于业界标准格式（FineWeb NeurIPS 2024、DCLM NeurIPS 2024、Nemotron-CC、Dolma ACL 2024、Gopher DeepMind 2021）中的共性报告结构。

---

## 报告结构（8 个必选章节）

### 1. Executive Summary（一页摘要）

用 3-5 句话概括：
- 数据源和规模
- Pipeline 核心方法
- 最终结果（保留率、质量提升、关键发现）

### 2. Pipeline Overview（流程总览）

用表格列出完整处理链：

```markdown
| 步骤 | 方法 | 输入 | 输出 | 条件过滤率 | 论文参照 |
```

如有多代 pipeline 并行对比，分别列出每代的处理链。

### 3. Per-Step Statistics（逐步统计）

**必须包含**（FineWeb/Dolma 标准格式）：
- 每步的输入文档数/token 数
- 每步的输出文档数/token 数
- 每步的条件过滤率（分子=该步丢弃数，分母=该步输入数）
- 累积保留率（相对于原始输入）

```markdown
| 步骤 | 输入 docs | 输出 docs | 条件过滤率 | 累积保留率 | 输出 tokens |
```

### 4. Quality Distribution（质量分布）

引用或描述以下可视化：
- Quality Score 直方图（各代叠加对比）
- Quality Score 均值/P50/P90 表格
- LIFT 计算：(该代质量均值 - 基线质量均值) / 基线质量均值 * 100%

```markdown
| 代次 | 文档数 | Quality Mean | LIFT vs 基线 | P50 | P90 |
```

### 5. Diversity & Token Analysis（多样性与 Token 分析）

- N-gram unique ratio（unigram/trigram）
- 域名 Shannon Entropy（归一化）
- 平均 tokens/文档
- 总 token 估算

### 6. Ablation & Component Analysis（消融与组件分析）

每代 pipeline 的核心组件贡献度分析：
- Gen1: 哪个过滤规则杀伤力最大？
- Gen2: 阈值敏感性（top-5% vs top-10% vs top-20%）
- Gen3: Bypass 价值分析、改写前后质量对比

### 7. Cross-Generation Comparison（跨代对比，核心结论）

**这是报告的核心**。参照 DCLM/Nemotron-CC 的 benchmark 对比格式：

```markdown
| 代次 | 方法类型 | 保留率 | Quality Mean | LIFT | 3-gram Diversity | Avg Tokens |
```

附带 Quality-Quantity Trade-off 散点图描述。

### 8. Limitations & Next Steps（局限与改进方向）

- 当前方案的已知局限
- 未执行的组件（如 LLM 改写、full_run）
- 改进建议

---

## 数据获取指引

报告所需数据来源：

| 数据 | 路径 |
|------|------|
| Gen1 统计 | `data/gen1_output/gen1_pipeline_stats.json` |
| Gen2 统计 | `data/gen2_output/gen2_stats.json` |
| Gen3 路由 | `data/gen3_output/gen3_routing_summary.json` |
| 跨代对比 | `results/reports/cross_generation_comparison.json` |
| 基线指标 | `results/quality_scores/baseline_metrics.json` |
| 阶段追踪 | `data/gen{1,2,3}_output/gen*_stage_metrics.json` |
| 审计记录 | `data/reports/audit/gen1/audit_summary.json` |
| 图表 | `results/figures/*.png` |

---

## 执行要求

1. **先读数据再写报告**：从上述路径加载实际数据，不要编造数字
2. **LIFT 口径统一**：LIFT = (该代质量均值 - 原始质量均值) / 原始质量均值 * 100%
3. **条件过滤率口径**：分子=该步丢弃数，分母=该步输入数
4. **累积保留率口径**：分子=该步输出数，分母=原始总输入数
5. **论文参照**：每个关键发现旁标注对应的论文（如"对比 Nemotron-CC 的 18.1% bypass 误杀率"）
6. **输出格式**：Markdown 文件，保存到 `results/reports/pipeline_execution_report.md`
