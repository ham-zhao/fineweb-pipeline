# 外部方法论调研汇总

> 从麦肯锡方法论、ML 工程标准流程、学术报告规范中筛选出的方法论。分为两类：A 类（与你的场景高度契合），B 类（有参考价值但不一定完全适用）。

---

## A 类：高度契合你的场景的方法论

---

### A1：MECE 原则（McKinsey，1960s）

```
来源：Barbara Minto，McKinsey & Company
全称：Mutually Exclusive, Collectively Exhaustive（相互独立，完全穷尽）

核心思想：
  把复杂问题拆分成子问题时，确保：
    ME（相互独立）：每个子问题不重叠
    CE（完全穷尽）：所有子问题加起来覆盖完整问题

与你的场景的契合点：
  ① Pipeline 设计：每个 Notebook 职责不重叠、整体不遗漏
  ② 分类体系：14 个风险类别必须 MECE（每条内容只属于一个类别、14 类覆盖所有风险）
  ③ 消融实验：每个消融组互不干扰、所有关键组件都测到
  ④ 面试回答：结构化回答时确保论点不重叠、覆盖完整

检查方法：
  "有没有两个部分做同一件事？"（ME 检查）
  "有没有关键环节没人负责？"（CE 检查）

参考资料：
  Barbara Minto《金字塔原理》
  McKinsey 内部培训材料
```

---

### A2：金字塔原理（Pyramid Principle, Barbara Minto, 1985）

```
来源：Barbara Minto, McKinsey & Company
著作：《The Pyramid Principle: Logic in Writing and Thinking》, 1985

核心思想：
  结论先行 → 支撑论点（不超过 3 个）→ 每个论点的数据/证据
  像金字塔：顶端是结论，底部是数据

  传统方式（底层向上）：
    "我们分析了数据...做了实验...发现...所以结论是 X"
    → 读者/面试官要等到最后才知道结论

  金字塔方式（顶层向下）：
    "结论是 X。理由有三个：A、B、C。支撑 A 的数据是..."
    → 读者 3 秒钟知道核心信息

四部分叙事结构：
  Situation（情境）→ Complication（矛盾）→ Question（问题）→ Answer（答案）

与你的场景的契合点：
  ① Notebook 结构：开头写结论，然后展开
  ② 面试回答：先说结论再给论据
  ③ 项目汇报：先说三代方法的核心结论，再展开每代的细节

"三的法则"（Rule of Three）：
  支撑论点不超过 3 个 → 人类工作记忆只能处理 3-4 项
  超过 3 个说明你没有充分归纳
```

---

### A3：SCR 叙事框架（McKinsey）

```
来源：McKinsey 内部方法论
全称：Situation → Complication → Resolution

核心思想：
  S（情境）：建立当前状态的共识
  C（矛盾/挑战）：引入打破现状的问题或机会
  R（解决方案）：给出你的方案

与你的场景的契合点：
  ① 每个 Notebook 的开头用 SCR 说明"为什么做这个实验"
  ② 面试中讲项目用 SCR 结构
  ③ 管理面讲 STAR 故事时的变体

示例：
  S："TikTok 每天 10 亿条内容需要安全审核"
  C："全用大模型 $36 亿/年不可承受，纯规则又漏判率高"
  R："三层级联：DistilBERT 初筛 + CLIP 中筛 + 大模型精筛，成本降 80 倍"
```

---

### A4：CRISP-ML(Q) 六阶段流程（Mercedes-Benz AG + TU Berlin, 2021）

```
来源：Studer et al., 2021, 基于 CRISP-DM 扩展
全称：Cross-Industry Standard Process for ML with Quality Assurance

六个阶段：
  Phase 1：业务理解 + 数据理解（合并，因为强相关）
  Phase 2：数据准备（清洗、特征工程、增强）
  Phase 3：建模（算法选择、训练、调参）
  Phase 4：评估（性能、鲁棒性、可解释性）
  Phase 5：部署
  Phase 6：监控与维护

相比 CRISP-DM 的改进：
  ① 业务+数据合并（实际中总是同时做的）
  ② 每个阶段增加 QA（质量保障）
  ③ 增加 Phase 6 监控维护（ML 特有需求）
  ④ 强调可复现性（记录元信息）

与你的场景的契合点：
  ① 项目框架直接按 6 阶段组织
  ② 每阶段的 QA 检查点 = 你说的"验收标准"
  ③ 可复现性要求 = 记录 seed、超参、环境

每阶段的 QA 结构：
  ① 定义该阶段的需求和约束
  ② 列出可能的风险（数据质量差？过拟合？不可复现？）
  ③ 选择 QA 方法（交叉验证？消融？可复现性检查？）
  ④ 记录决策和理由

可复现性元信息记录：
  算法、训练/验证/测试数据集、超参数、运行环境描述
  不同 random seed 下结果是否稳定
  推荐使用 Model Cards 框架记录模型信息

参考资料：
  论文：Studer et al., "Towards CRISP-ML(Q)", 2021
  网站：ml-ops.org/content/crisp-ml
```

---

### A5：ML 实验报告标准结构（UT Austin CS 391L）

```
来源：UT Austin CS 391L Machine Learning 课程报告标准

标准 ML 实验论文/报告结构：

  1. Introduction（引言）
     问题是什么？为什么重要？基本方法是什么？
     简要预览主要结论

  2. Problem Definition & Algorithm（问题定义与算法）
     2.1 精确定义输入和输出
     2.2 算法描述（伪代码 + 具体示例 trace）

  3. Experimental Evaluation（实验评估）
     3.1 Methodology（方法论）
         - 评估标准是什么？验证什么假设？
         - 自变量和因变量是什么？
         - 训练/测试数据是什么？为什么合理？
         - 收集了什么性能数据？怎么分析？
         - 和竞争方法的对比
     3.2 Results（结果）
         - 数据表格和图表
         - 具体数字
     3.3 Discussion（讨论）
         - 结果说明了什么？为什么？
         - 意外发现是什么？

  4. Related Work（相关工作）
     和已有方法的对比

  5. Conclusion & Future Work（结论与未来方向）
     主要发现 + 局限性 + 下一步

与你的场景的契合点：
  Notebook 的 Markdown Cell 按这个结构组织
  面试时打开 Notebook 直接讲 = 一份完整报告
  消融实验部分 = 3.1 + 3.2 + 3.3 的完美实践
```

---

### A6：ML 可复现性清单（Joelle Pineau, 2019）

```
来源：Joelle Pineau (McGill / Meta FAIR), "The Machine Learning Reproducibility Checklist"

核心清单：

  模型相关：
    ☐ 模型架构完整描述（层数、参数量、激活函数）
    ☐ 所有超参数的具体值
    ☐ 超参数搜索方法（网格搜索？随机搜索？范围是什么？）

  数据相关：
    ☐ 数据集完整描述（来源、大小、划分方式）
    ☐ 数据预处理步骤
    ☐ 训练/验证/测试划分的具体方式和比例

  实验相关：
    ☐ 评估指标的精确定义
    ☐ 随机种子（固定且报告）
    ☐ 运行次数和方差报告
    ☐ 计算资源描述（GPU 型号、训练时长）
    ☐ 代码是否公开可用

与你的场景的契合点：
  每个 Notebook 的"实验记录"部分按这个清单填写
  GitHub README 中列出环境和复现步骤
```

---

## B 类：有参考价值但不一定完全适用的方法论

---

### B1：CRISP-DM（原始版, 1999）

```
来源：IBM/SPSS/NCR 等联合发布, 1999
全称：Cross-Industry Standard Process for Data Mining

六个阶段：
  1. Business Understanding → 2. Data Understanding
  3. Data Preparation → 4. Modeling
  5. Evaluation → 6. Deployment

与 CRISP-ML(Q) 的区别：
  没有 QA（质量保障）→ CRISP-ML(Q) 补上了
  没有监控维护阶段 → 不适合需要持续运行的 ML 系统
  业务和数据理解分开 → CRISP-ML(Q) 合并了

为什么是 B 类而不是 A 类：
  CRISP-DM 是 1999 年的框架，面向传统数据挖掘
  CRISP-ML(Q) 是它的 ML 特化升级版 → 直接用 A4 更好
  但 CRISP-DM 的核心思想（业务目标驱动、迭代循环）仍然有效

参考价值：
  "先理解业务再做技术"的原则永远对
  "最后才开始写代码的团队反而表现最好"（研究结论）
```

---

### B2：SEMMA（SAS Institute）

```
来源：SAS Institute
全称：Sample → Explore → Modify → Model → Assess

五步流程（比 CRISP-DM 更简洁）：
  Sample：采样代表性数据
  Explore：探索初始模式
  Modify：数据清洗和变换
  Model：应用算法
  Assess：评估性能

为什么是 B 类：
  过于简洁，缺少业务理解和部署阶段
  但"先探索再建模"的顺序对你的 Notebook 组织有参考价值
  Notebook 01（数据探索）→ Notebook 02（数据修改）→ Notebook 03（建模）
  → 这个顺序就是 SEMMA 的 E→M→M
```

---

### B3：Agile ML 框架（Doximity, 2024）

```
来源：Doximity 数据团队，Built In 发表
核心思想：把 ML 项目组织成迭代周期，每个周期产出一份"报告"

和传统 Agile/Scrum 的区别：
  Scrum 的固定 Sprint 不适合 ML（有些任务需要 1 天，有些需要 2 周）
  Agile ML 用"功能迭代"替代"时间 Sprint"
  每个迭代 = 一个实验 → 产出一份报告 → 决定下一步

核心观点：
  "被放弃的项目不是失败，只是学到了不值得继续投资"
  "数据科学家应该解释结果，这是最有价值的环节"

为什么是 B 类：
  适合团队协作场景
  你是个人做面试项目 → 不需要 Sprint/迭代管理
  但"每个实验产出一份报告"的思想很好
  → 每个 Notebook = 一个实验报告 → 有明确的假设、实验、结论

参考价值：
  把你的每个 Notebook 看作一次"实验迭代"
  Notebook 开头写假设 → 中间做实验 → 结尾写结论和下一步建议
```

---

### B4：Model Cards（Google, 2019）

```
来源：Mitchell et al., Google, 2019
核心思想：每个发布的模型都附带一张"模型名片"

Model Card 包含：
  模型详情：架构、参数量、训练数据
  预期用途：适合什么场景，不适合什么场景
  性能指标：在不同子群体上的表现
  限制和偏差：已知的问题
  伦理考量：可能的负面影响

为什么是 B 类：
  Model Cards 主要用于"发布模型给别人用"的场景
  你的面试项目不涉及公开发布
  但 Model Cards 的"明确说明局限性"的思想很好
  → 每个 Notebook 结尾写"本实验的局限性"
  → 面试时主动说出局限性 = 展示成熟度
```

---

### B5：Google ML Test Score（Google, 2017）

```
来源：Breck et al., Google, 2017
核心思想：用清单给 ML 系统的"生产就绪度"打分

四类检查（共 28 个测试项）：
  测试数据：特征分布监控、训练/线上数据一致性
  测试模型：模型公平性、鲁棒性、可解释性
  测试基础设施：Pipeline 可复现、训练/推理一致
  测试监控：性能退化检测、数据漂移报警

为什么是 B 类：
  面向大规模生产系统 → 你的面试项目不需要全部满足
  但其中"训练/推理一致性"和"Pipeline 可复现"的要求
  对你的项目质量有直接帮助

参考价值：
  项目做完后用这个清单自检一遍
  能回答"你怎么保证模型质量"这类面试问题
```

---

### B6：数据分析 SCQA 框架（Barbara Minto 变体）

```
全称：Situation → Complication → Question → Answer

和 SCR 的区别：多了一步 Question

  S："银行数据取数需求每天 50+ 条"
  C："数据团队人力有限，平均响应时间 3 天"
  Q："能不能让业务人员自己取数？"
  A："建设 Data Agent，自然语言生成 SQL"

为什么是 B 类：
  SCR（A3）在面试场景更常用（更简洁直接）
  但 SCQA 在写文档和 Notebook 开头时更完整

参考价值：
  Notebook 00（方法论 Notebook）的开头用 SCQA 讲清楚"为什么做这个项目"
  面试开场用 SCR（更短）
```
