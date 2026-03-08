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

---

## 阶段三：中文扩展 + Git 初始化 + CI

### 交互 4: 阶段三启动（用户指令：启动方案 3）
- **类型**：确认指令
- **用户输入**：启动方案 3
- **执行内容**：
  - **src/gen1_zh/**：`chinese_text_utils.py`（CJK 字符计数、简繁体检测、垃圾评分、token 估算）、`chinese_quality_filter.py`（`ChineseQualityFilter`，7 重过滤）、`pipeline.py`（`ChineseGen1Pipeline`，语言检测→标准化→质量过滤→简繁统计）
  - **.github/workflows/ci.yml**：4 个 Job（lint/import-check/smoke-test/test-chinese）
  - **git init + commit**：65 个文件，11327 行，commit hash e119911

### 交互 5: 阶段四补全（用户指令：你替我继续完成吧）
- **类型**：继续指令
- **执行内容**：
  - **src/proxy_model/evaluator.py**：`ProxyModelEvaluator`（加载 model.pt、compute_perplexity、generate、completion_accuracy、load_all）
  - **notebooks/09_proxy_model_validation.ipynb**：10 个 cell，涵盖训练曲线、Val PPL 对比、Chinchilla Scaling 分析、文本生成样本、训练效率、结论汇总
  - 第二次 commit：将 Notebook 09 + proxy_model 模块一起提交

### 关键设计决策记录（阶段三）
1. **中文过滤单位**：字符数（CJK chars）而非 word count，因中文无天然空格分词边界
2. **简繁体不过滤**：两种字体均保留，只做 script_type 元数据标注，下游可按需过滤
3. **垃圾检测纯规则**：`compute_spam_score()` 用 9 个正则模式加权，避免额外模型依赖
4. **CI 使用 CPU 版 torch**：GitHub Actions 无 GPU/MPS，通过 `--index-url .../cpu` 安装 CPU 版
5. **ProxyModelEvaluator 内联模型定义**：evaluator.py 内部重建模型架构，不依赖 run_proxy_training.py 中的类，避免循环导入

### 最终项目结构（全量）
```
fineweb-pipeline/
├── .github/workflows/ci.yml       # CI: lint + import-check + smoke-test + zh-unit-test
├── configs/                       # 6 个配置文件（run/api/gen1/2/3/eval）
├── docs/interaction_log.md        # 本文件
├── notebooks/                     # 00-09 共 10 个 Jupyter Notebook
├── requirements.txt
├── scripts/
│   ├── run_gen1/2/3.py
│   ├── generate_comparison_report.py
│   ├── run_proxy_training.py      # 阶段四独立脚本
│   └── download_sample.sh
├── setup.sh
└── src/
    ├── dedup/                     # exact_dedup + minhash_dedup
    ├── evaluation/                # 6 个评估模块（独立于 pipeline）
    ├── gen1/filters/ + pipeline   # 6 个 Heuristic 过滤器
    ├── gen1_zh/                   # 中文专用 Pipeline（阶段三新增）
    ├── gen2/                      # DCLM fastText + 阈值调参 + pipeline
    ├── gen3/                      # 集成分类 + bypass + LLM 改写 + pipeline
    ├── proxy_model/               # 评估器（阶段四新增）
    └── utils/                     # config_loader / downloader / tokenizer_utils
```

---

*项目全部阶段完成（阶段一～三已 commit，阶段四脚本就绪，Notebook 09 已生成）。*

---

## 阶段六：数据源替换 + CC WET 全量重跑

### 交互 7: CLAUDE.md 规则精炼 + Phase A-E 执行（2026-03-06）

- **类型**：规则优化 + 全量执行
- **用户输入**：精炼 CLAUDE.md 规则 + 按 PLAN_data_overhaul.md 执行 Phase A-E
- **关键决策**：
  1. CLAUDE.md 从粗略规则升级为 253 行精炼版（含 before/after 示例、验证命令、来源标注）
  2. CC WET 数据从 10 个随机 segment 采样 12K docs，确保域名多样性（10,334 unique domains）
  3. C4 terminal_punct_min_ratio 从 0.7 降至 0.1（CC WET p50=0.11）
  4. JS 检测精简为仅 "javascript"（C4 论文原始规则）
  5. Gen1 Pipeline 修复：config 阈值传参（之前全部使用硬编码默认值）

- **执行结果**：
  - Gen1: 12K -> 409 (3.4%), LIFT +4.3%
  - Gen2: 409 -> 41 (10.0%), LIFT +6.1%
  - Gen3: 409 -> 17 (4.2%), LIFT +5.3%（无 LLM 改写）
  - 三代质量差异明显：quality_mean 0.4945 -> 0.5158 -> 0.5246
  - 5 个核心 Notebook (01/02/03/04/06) 重新执行完毕

- **遗留问题**：
  1. LLM 改写未执行（API Key 未配置），Gen3 数据回收能力未完全展示
  2. 语言过滤 75.4%（CC WET 英文仅 ~25%），如需多语种需调整
  3. 评估分类器区分度有限（quality_mean 差异仅 0.03）

---

### 交互 7: NB02-04 五维度量 + 子过滤器详细分析 + 路径修复
- **类型**：功能增强 + Bug 修复
- **用户需求**：
  1. NB02-04 全部增加五维数据质量 Profile（规模/质量/语言/多样性/毒性）
  2. NB02 需要每个大类和子类过滤器的详细分析：分子/分母、预期值、3-5 个样例
  3. 子过滤器级别的贡献度表（Gopher/C4/FineWeb 分别多少）

- **关键变更**：
  1. **新增评估模块**：`baseline_profiler.py`（五维 Profile）、`kenlm_scorer.py`、`language_detector.py`
  2. **新增分析脚本**：`gen1_filter_analysis.py`（逐过滤器分析 + 样例抽取）
  3. **Pipeline 增强**：`pipeline.py` 新增 `reason_breakdown` / `detail_breakdown` 统计
  4. **NB02 全面重写**：23 cells，双模式对比，预期值 vs 实际值，子过滤器贡献表
  5. **NB03/NB04 路径修复**：Cell A 从硬编码 `../data/gen2_output/gen2_stats.json` 改为 `get_output_path()` 动态路径
  6. **NB03/NB04 五维 Profile**：新增 Cell Group E/D，Gen1→Gen2/Gen3 五维演进对比

- **Bug 修复**：
  - NB03/NB04 `FileNotFoundError`：硬编码路径不含 run_mode 子目录
  - `_gen_nb02.py` 中文引号导致 SyntaxError：`"平庸内容"` → 改用单引号包裹
  - audit CSV 含无效 UTF-8 字节：改用 `gen1_filter_analysis.py` + `clean_text_for_json()`

- **执行验证**：
  - NB02: 23 cells 全部有 output，五维 Profile 输入 PPL 2150 → 输出 PPL 887
  - NB03: 14 cells 全部有 output，Gen2 PPL 644 vs Gen1 1042（better），tail 53.7%→17.1%
  - NB04: 12 cells 全部有 output，路由漏斗 + 五维对比完整
  - 已 commit + push：`f2294e9`

---

### 交互 8: 方法论深化 — 截断策略、负样本选择、ML 检视指标体系
- **类型**：方法论改进 + Bug 修复 + 文档更新
- **时间**：2026-03-08 13:51 ~ 进行中

- **关键讨论与决策**：
  1. **正样本不截断原则**：分类器训练时只截断较长方（通常是负样本），正样本（参考文本）保留完整。理论依据：正样本是"标准答案"，截断会丢失质量信号
  2. **Gen3 负样本修正**：从 Gen1 输出改为原始 CC WET。Gen1 输出太干净（分离度仅 0.14-0.48），改用 CC WET 后分离度提升至 0.89-0.94
  3. **ML 检视指标体系**：pipeline 输出指标（保留率/MMLU）之外，新增三维检视：分类器健康度、数据分布一致性、端到端验证（Golden Set + bypass 误杀率）
  4. **级联架构 vs 统一输入**：用户质疑 Gen2/Gen3 以 Gen1 输出为输入导致保留率口径不一致。讨论方案 A（统一输入）vs 方案 B（统一端到端指标）

- **关键变更**：
  1. `src/gen2/quality_classifier.py`：单侧截断（只截断较长方）
  2. `src/evaluation/quality_classifier.py`：同上
  3. `scripts/run_gen3.py`：Gen3 负样本从 Gen1 输出改为原始 CC WET
  4. NB00 Cell 3/4/6：修复 Gen3 负样本描述 + 新增数据源统计特征表 + ML 检视指标表
  5. NB04 Cell 13：新增 bypass 误杀率 + 改写成功率输出
  6. `docs/METHODOLOGY_PATTERNS.md`：新增 Pattern #20-23（截断、Golden Set、健康度、ML 检视）
  7. Golden Set 基础设施：`golden_samples.jsonl` + `golden_validator.py` + `generate_golden_samples.py`
  8. NB03/NB04：新增 Cell Group F-H/E（健康度检查 + Error Analysis + Distribution Shift）

- **实验数据**（smoke_test 12K docs）：
  - Bypass 误杀率：90.7%（远高于 Nemotron-CC 18.1%，因 Gen1 输出已清洗）
  - Gen3 集成分离度：DCLM 0.94, EDU 0.89（修正后）
  - NB00-NB09 全部执行成功，无 error cells

- **后续落地**（用户确认方向后立即执行）：
  1. **统一输入架构（方案 A）**：用户选择方案 A，Gen2/Gen3 始终从 CC WET 开始
     - `scripts/run_gen2.py`：内部先跑 Gen1，报告端到端保留率 (Gen2输出/CC WET输入)
     - `scripts/run_gen3.py`：同上
     - NB00 级联图改为三叉并行结构，保留率口径统一
  2. **NB07 代理消融**：添加 eval score vs top-fraction 分析
     - 8 个阈值点 (5%~100%)，eval score 从 0.8379 单调递减至 0.6870
     - 确认 quality-quantity trade-off 存在
  3. **Q1-Q4 方法论备注**：Pattern #24 FAQ 加入 METHODOLOGY_PATTERNS.md
  4. **Pipeline 重跑**（smoke_test 12K，统一输入后）：
     - Gen2: 12,000 → 409 (Gen1) → 41 (e2e 0.3%)
     - Gen3: 12,000 → 409 (Gen1) → 56 (e2e 0.5%)
  5. NB00~NB09 全部重新执行，验证通过
