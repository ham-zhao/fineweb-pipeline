# 交互记录

> 自动维护，记录项目开发过程中的关键决策、报错修复和经验总结。

---

## 阶段一：项目初始化 + 环境搭建 + 评估体系

### 交互 1: 阶段一启动
- **类型**：指令确认
- **用户输入**：启动阶段一，创建完整的预训练数据清洗三代方法论对比项目
- **处理要点**：
  1. 创建完整目录结构（src/gen1-3, evaluation, dedup, utils, notebooks, scripts, results）
  2. 创建 6 个配置文件（run_config, api_config, gen1/2/3_config, eval_config）
  3. 构建评估体系（quality_classifier, perplexity_scorer, toxicity_scorer, diversity_metrics, stage_tracker, filter_auditor）
  4. 创建 README.md（含方法论演进 Mermaid 图和数据链条说明）
- **关键发现**：
  - 评估分类器独立性是核心约束，通过差异化超参（dim=32 vs 64, wordNgrams=1 vs 2）和不同正样本来源保证
  - run_config.yaml 的 smoke_test/full_run 双模式是贯穿所有 pipeline 的核心控制机制

---

---

## 阶段二：三代 Pipeline 实现 + 对比评估 + Notebooks

### 交互 2: 阶段一确认间隙 Q&A（阶段二启动前）
- **类型**：澄清问题
- **用户提问要点**：
  1. "Hyper model" 是否指 claude-haiku-4-5-20251001 → 确认：此模型用于 Gen3 LLM 改写（configs/api_config.yaml），适合小批量改写场景；Claude Code 本身使用 claude-sonnet-4-6 足够。
  2. 内容审查重点 → 推荐优先审查 Notebook 06（核心对比）、03（Gen2 threshold 实验）、07（消融分析）；核心检验点：循环偏差、Bypass 假阳性率、改写质量提升量化。
  3. "VQ 文档" → 解释为项目内置的自动化质量验证/报告体系（StageTracker + FilterAuditor + comparison_report.py）。
  4. 自检发现缺失：`results/reports/audit/` 目录未创建 → 已修复（mkdir -p）。
- **教训**：setup.sh 的 mkdir -p 列表需包含所有 results 子目录；应在阶段一完成后执行一次 tree 检查。

### 交互 3: 阶段二启动确认
- **类型**：确认指令
- **用户输入**：可以启动阶段二
- **执行内容**：
  - **src/gen1/filters/**：url_filter.py, language_filter.py, quality_filter.py（Gopher+C4+FineWeb 三合一）, repetition_filter.py, pii_filter.py, toxicity_filter.py
  - **src/gen1/pipeline.py**：完整 Pipeline 协调器，含 WARC/JSONL/parquet 多格式输入
  - **src/gen2/**：quality_classifier.py（DCLM fastText，dim=64, wordNgrams=2）, threshold_tuner.py, pipeline.py
  - **src/gen3/**：classifier_ensemble.py（union/intersection/weighted_avg）, conditional_bypass.py, synthetic_rephraser.py（Anthropic/OpenAI/DeepSeek API）, pipeline.py
  - **src/dedup/**：exact_dedup.py（xxhash）, minhash_dedup.py（num_hashes=128, num_buckets=8, Jaccard=0.8）
  - **scripts/**：run_gen1.py, run_gen2.py, run_gen3.py, generate_comparison_report.py, **run_proxy_training.py**
  - **notebooks/**：00~08 全部 9 个 Jupyter Notebook（00 方法论概览、01 数据探索、02 Gen1、03 Gen2、04 Gen3、05 去重、06 跨代对比、07 消融分析、08 中文扩展）

### 关键设计决策记录（阶段二）
1. **质量过滤器分层**：Gopher（粗粒度 word count/alpha ratio）→ C4（标点/JS）→ FineWeb（bullet/ellipsis），每层独立可关闭
2. **Gen2 阈值实验**：ThresholdTuner 测试 [5%, 10%, 15%, 20%, 30%, 50%]，默认 top-10% 复现 DCLM 结论
3. **Gen3 Bypass 验证**：conditional_bypass.py 计算 `compute_bypass_value()` 量化 Heuristic 假杀率，目标复现 Nemotron-CC 18.1%
4. **MinHash 参数**：num_hashes=128 保证签名估计精度；num_buckets=8（rows_per_band=16）对应约 J≈0.8 的概率曲线
5. **Notebooks 编码**：遇到中文字符 JSON 序列化问题，使用 `json.dump(..., ensure_ascii=True)` 解决

### 错误修复（阶段二）
- Notebook 04/05 初版中文字符未正确 Unicode 转义导致 JSONDecodeError，重新用 ensure_ascii=True 生成修复

---

*后续阶段的交互记录将在此文件末尾追加。*
