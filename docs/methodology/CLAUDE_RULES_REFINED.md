# 全局规则（提炼版）

> 基于 fineweb-pipeline / post-train-pipeline / safety-dataset 三个 ML 项目的实际教训总结。
> 每条规则标注了来源场景，方便追溯和裁剪。

---

## 一、执行纪律

### 1.1 增量执行：写一步、跑一步、验一步

创建任何可执行产物后，必须**立即执行并验证**，验收通过后才能开始下一步。

| 产物类型 | 执行方式 | 验收标准 |
|---------|---------|---------|
| Python 脚本 | `python3 script.py` | 退出码 0 + 预期输出文件存在且非空 |
| Notebook | `source .venv/bin/activate && jupyter nbconvert --to notebook --execute --inplace nb.ipynb` | 所有 cell 有 output，无 traceback |
| Config (YAML) | `python3 -c "import yaml; yaml.safe_load(open('config.yaml'))"` | 可加载 + 关键字段有值 |
| 数据下载 | `wc -l file` + `ls -lh file` | 行数和文件大小在预期范围内 |

**阻断**：步骤 N 未通过验收，禁止开始步骤 N+1。报错必须修复或给出降级方案，不能静默跳过。

> 来源：fineweb Phase 1-5 标记完成但脚本从未执行——"代码存在 ≠ 任务完成"。

### 1.2 报告完成时附带证据

向用户报告"已完成"时，必须附带：
- **日志摘要**：关键输出行（不是全部日志）
- **产出物清单**：`ls -lh` 输出
- **核心指标**：数量/比率/得分等关键数值

不合格：`"已创建 run_gen1.py、run_gen2.py、run_gen3.py"`（创建 ≠ 执行成功）
合格：`"Gen1 完成：输入 3000 条 → 输出 1247 条（存活率 41.6%），文件 data/gen1_output/gen1_output.jsonl (2.3MB)"`

### 1.3 阶段验收

每个阶段结束前，列出该阶段**全部预期产出物**，逐一确认文件存在且内容非空。格式：

```
阶段 X 验收：
  [x] file_a.jsonl (1,247 行, 2.3MB)
  [x] file_b.json (非空, 含 stats 字段)
  [ ] file_c.csv — 缺失，原因：...
```

---

## 二、架构契约

### 2.1 三层职责分离

```
src/          → 函数库（可被 scripts/ 和 notebooks/ import）
scripts/      → CLI 入口，执行 pipeline/训练，产出数据到 data/ 或 results/
notebooks/    → 读取已产出的数据文件，做分析和可视化
```

**核心约束**：Notebook 不导入也不执行 pipeline/训练逻辑。

Import 分类：
| 类别 | 示例 | Notebook 中是否允许 |
|------|------|-------------------|
| 配置/工具 | `config_loader`, `read_jsonl`, `save_jsonl` | 允许 |
| 独立评估器 | `EvalQualityClassifier`（参数与 pipeline 不同的独立模型） | 允许 |
| Pipeline 执行 | `Gen1Pipeline`, `Gen2Pipeline`, `QualityFilter`, `ConditionalBypass` | **禁止** |
| 训练模块 | `Trainer`, `train_classifier`, `SFTTrainer` | **禁止** |

验证命令：
```bash
# 检查 notebooks 中是否存在 pipeline/训练模块的 import
grep -rn "Pipeline\|Trainer\|Filter\|Bypass\|Ensemble" notebooks/ --include="*.ipynb" | grep -v "output"
# 期望：无匹配或仅出现在 markdown cell / 注释中
```

> 来源：fineweb NB02 导入 `src.gen1.filters.*` 逐 cell 运行 6 个过滤步骤；NB03 导入 `Gen2QualityClassifier` 现场训练分类器。重构后改为读取 `gen1_pipeline_stats.json` 和 `gen2_stats.json`。

### 2.2 数据契约：脚本声明产出，Notebook 声明依赖

每个 pipeline 脚本在文件头部注释中列出其产出文件：
```python
"""
产出文件：
  data/gen1_output/gen1_output.jsonl        - 过滤后文档
  data/gen1_output/gen1_pipeline_stats.json  - 各步过滤统计
"""
```

Notebook 开头声明并校验依赖：
```python
REQUIRED = ["data/gen1_output/gen1_pipeline_stats.json"]
for f in REQUIRED:
    assert Path(f).exists(), f"缺少: {f}，请先运行 scripts/run_gen1.py"
```

### 2.3 Config 驱动规模，代码不硬编码

所有数量参数（doc_limit、sample_size、epoch、batch_size 等）集中在 `configs/run_config.yaml`。多档运行通过切换 `run_mode` 实现：

```yaml
run_mode: "smoke_test"   # 切换到 "full_run" 即可放大规模
smoke_test:
  doc_limit: 12000
full_run:
  doc_limit: 100000
```

代码中用 `config["doc_limit"]` 读取，不写死数字。

---

## 三、文档规范

### 3.1 Overview 前置（前 50 行三问）

技术文档开头必须回答：
1. **问题**：现状与痛点（1-2 句）
2. **方案**：核心做法（1-2 句）
3. **预期**：关键指标的预期值（量化）

然后放术语表，再展开正文。

> 来源：PLAN v1 第 1 行直接是术语表，读者不知道文档在解决什么问题。v2 前 20 行就回答了"问题→方案→预期"。

### 3.2 Diataxis 分层：每节一个主类型

写每一节前，先判定它的主类型，然后**只写该类型的内容**：

| 主类型 | 核心动词 | 允许内容 | 禁止混入 |
|--------|---------|---------|---------|
| Explanation（为什么） | 解释、对比、论证 | 背景、选型理由、论文对比 | 操作步骤、核对清单 |
| Reference（是什么） | 定义、列表、查阅 | 指标口径、术语表、参数表 | 推导过程、操作步骤 |
| How-to（怎么做） | 执行、操作、验证 | 步骤列表、验收标准、核对清单 | 大段方法论解释 |

检查方法：如果一节中同时出现"为什么选 X"（Explanation）和"第一步做 Y"（How-to），需要拆分。

> 来源：PLAN v1 在执行计划（How-to）中嵌入 3 段 blockquote 解释错误处理策略（Explanation）。v2 将错误处理策略移到设计章节。

### 3.3 指标定义一次，全文引用（Single Source of Truth）

所有指标的口径（分子/分母/公式）在文档的 Reference 章节统一定义。后续核对清单、验收表、正文讨论中**只引用不重复定义**。

不合格：Gen1 存活率在"指标体系""核对清单""整体验收"三处各定义一遍
合格：§2.1 定义口径，核对清单写"Gen1 整体存活率（口径见 §2.1 #5）"

> 来源：PLAN v1 同一个"Gen1 存活率"在三处重复定义，修改时遗漏导致数值不一致。

### 3.4 百分比必须定义口径

文档中出现任何百分比或比率时，必须同时说明**分子是什么、分母是什么**。

| 不合格 | 合格 |
|--------|------|
| "过滤率 20%" | "条件过滤率 20%（分子=该步丢弃数，分母=该步输入数，非原始总量）" |
| "质量提升 +10%" | "质量 LIFT +0.10（绝对差，eval_score 均值从 0.35 升至 0.45）" |
| "保留 top-10%" | "保留率 10%（分子=Gen2 输出数，分母=Gen1 输出数）" |

特别注意：区分**绝对差**（+10 个百分点）和**相对变化率**（增长了 10%），两者含义完全不同。

### 3.5 长步骤分组 + Blockquote 控制

- 超过 10 步的执行计划 → 分子组，每组有独立核对清单
- 正文 blockquote 不超过 10 个 → 超过则收归附录

---

## 四、数据与实验设计

### 4.1 数据源必须匹配任务噪音需求

选择输入数据前先问：**"这个数据的噪音水平/难度适不适合当前任务？"**

| 任务类型 | 需要的数据特征 | 反模式 |
|---------|--------------|--------|
| 数据清洗 pipeline | 脏数据（垃圾率 >30%） | 用已清洗数据当输入（效果为零） |
| 模型训练 | 干净、有标签的数据 | 用未清洗的原始数据直接训练 |
| 评估/对比 | 有 ground truth 或独立评估器 | 用 pipeline 自身打分评估自身 |

> 来源：fineweb 用 FineWeb（已清洗，垃圾率 ~5%）当清洗 pipeline 输入，质量提升仅 0.2%，去重率 0%。换成 CC WET（垃圾率 ~50-70%）后效果正常。

### 4.2 样本量倒推，不拍脑袋

确定样本量前，从 pipeline 最末端往前推算：

```
最终需要 X 条有效输出
  ← 最后一步保留率 P_n → 输入需 X / P_n
    ← 前一步保留率 P_(n-1) → 需 X / (P_n × P_(n-1))
      ← ... 推到原始输入量
```

每一档的 doc_limit 必须让**每个中间步骤**都有足够输出（至少 50 条，否则统计无意义）。

> 来源：fineweb 初始用 1000 条，经 Gen1(40%) → Gen2(10%) 后仅剩 40 条，无统计意义。

### 4.3 执行流程先画依赖图

规划多步执行前：
1. 画出步骤间的依赖关系（哪些步骤需要前序输出）
2. 无依赖的步骤标记为可并行
3. 合并冗余步骤（"先下 3K 再追加到 12K" → "直接下 12K，取前 3K 做 smoke_test"）

---

## 五、Notebook 质量门禁

编辑或创建 Notebook 后逐项检查：

| # | 检查项 | 检查方法 | 失败后果 |
|---|--------|---------|---------|
| 1 | 中文字体配置 | 首个绘图 cell 含 `rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']` | 中文标签显示为方框 |
| 2 | 图表中文渲染 | 执行后肉眼检查标题/标签/图例 | 字体存在但未生效 |
| 3 | surrogate 字符 | Web 数据打印时用 `errors='replace'` | UnicodeEncodeError 崩溃 |
| 4 | 依赖文件校验 | 开头 assert 所有 REQUIRED_FILES 存在 | 空 DataFrame → 图表异常 |
| 5 | Notebook 生成方式 | 含中文的 notebook 用 Python 脚本 + `json.dump()` 生成 | 手写 JSON 中文引号转义失败 |

---

## 六、变更保护

### 6.1 重构前备份
文档或代码进行**结构性重构**前，备份原版：`filename_v1_YYYYMMDD.ext`

### 6.2 需求-实现同步
修改设计文档后，必须同步检查执行计划：
- 每个设计模块 → 对应实现步骤？
- 新增配置参数 → config 修改步骤中列出？
- 核对清单 → 覆盖所有新增模块的验收？

### 6.3 记忆描述准确
MEMORY.md 必须反映**当前真实状态**。"待落地" vs "已落地"含义完全不同。用户纠正后立即更新。

> 来源：fineweb 写了"阶段六重构时落地"，实际当场就重构了。用户纠正后改为"已重构落地"。

### 6.4 .gitignore 语法
`.gitignore` 不支持行内注释。`*.bin # model files` 中 `#` 是文件名的一部分。

> 来源：fineweb 差点因此提交 491MB 的模型文件。

---

## 七、规则管理

- 用户提出新规则 → 确认作用域：通用→本文件，仅本项目→`memory/MEMORY.md`
- 规则有深度（需示例/模板）→ 同步更新方法论文档（如 `docs/METHODOLOGY_PATTERNS.md`）
- 会话结束前，主动提醒是否有新规则需记录

## 八、环境

- 优先 `python3`（macOS 上 `python` 可能不存在）
- jupyter 命令需先 `source .venv/bin/activate`（注意 `.venv` 不是 `venv`）
- 耗时脚本加 `caffeinate -i` 前缀（macOS 防休眠）
