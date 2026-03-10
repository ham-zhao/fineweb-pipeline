# 方法论复用指南：如何在新项目中启用

> 将 fineweb-pipeline 项目提炼的标准方法论，复用到 post-train-pipeline、safety-dataset 或任何新 ML 项目中。

---

## 一、需要复制的文件

只需复制 **2 个文件**到目标项目的 `docs/` 目录下：

```bash
# 替换 <TARGET_PROJECT> 为目标项目路径
cp docs/STANDARD_METHODOLOGY.md <TARGET_PROJECT>/docs/
cp docs/METHODOLOGY_COMPACT.md  <TARGET_PROJECT>/docs/
```

**不需要**复制全局 `~/.claude/CLAUDE.md`——它已经自动对所有项目生效。

---

## 二、给 Claude Code 的初始化命令

### 情况 A：目标项目没有 CLAUDE.md（如 post-train-pipeline）

在目标项目的 Claude Code 窗口中粘贴：

```
请完成方法论初始化：

1. 读取 docs/STANDARD_METHODOLOGY.md（完整方法论）和 docs/METHODOLOGY_COMPACT.md（精简版）
2. 读取全局规则 ~/.claude/CLAUDE.md，确认理解
3. 在项目根目录创建 CLAUDE.md，内容为：
   - 首先粘贴 METHODOLOGY_COMPACT.md 的全部内容
   - 然后追加项目概况段落（从项目已有文件中了解）
   - 最后加一行：详细方法论参考 docs/STANDARD_METHODOLOGY.md
4. 在 memory 中记录：方法论已导入，完整版在 docs/STANDARD_METHODOLOGY.md
5. 输出初始化核销表，逐项确认完成
```

### 情况 B：目标项目已有 CLAUDE.md（如 safety-dataset）

在目标项目的 Claude Code 窗口中粘贴：

```
请完成方法论初始化：

1. 读取 docs/STANDARD_METHODOLOGY.md（完整方法论）和 docs/METHODOLOGY_COMPACT.md（精简版）
2. 读取全局规则 ~/.claude/CLAUDE.md，确认理解
3. 读取现有 CLAUDE.md，在其现有内容【之前】插入 METHODOLOGY_COMPACT.md 的全部内容，保留所有现有项目规则不删除。在末尾加一行：详细方法论参考 docs/STANDARD_METHODOLOGY.md
4. 在 memory 中记录：方法论已导入，完整版在 docs/STANDARD_METHODOLOGY.md
5. 输出初始化核销表，逐项确认完成
```

---

## 三、是否需要 /compact

**不需要。** 新项目窗口是新对话，上下文是干净的。`/compact` 只在长对话接近上下文上限时使用。

---

## 四、生效机制

```
三层规则体系：

  ~/.claude/CLAUDE.md              → 所有项目自动继承（执行纪律、架构契约等 9 大类）
  项目/CLAUDE.md                   → 方法论精简版 + 项目特有规则（上面命令创建/更新）
  docs/STANDARD_METHODOLOGY.md     → Agent 需要时查阅完整案例和参考文献
```

---

## 五、未来新规则的流入

```
新规则纳入流程（Agent 自动执行）：

  通用教训 → Agent 写入 ~/.claude/CLAUDE.md → 所有项目自动继承
  项目特有 → Agent 写入项目 MEMORY.md → 不影响其他项目
  方法论性质 → 写入 docs/STANDARD_METHODOLOGY.md → 需手动同步到其他项目

跨项目同步方案：
  短期：手动 cp 更新后的文件到其他项目
  长期：可考虑符号链接或共享目录，让多个项目引用同一份方法论文件
```

---

## 六、文件清单

| 文件 | 用途 | 大小 |
|------|------|------|
| `docs/STANDARD_METHODOLOGY.md` | 完整方法论（29 条原则 + 案例 + 参考文献） | ~22KB |
| `docs/METHODOLOGY_COMPACT.md` | 精简版（22 条，<500 字），可直接作为 prompt 前缀 | ~2KB |
| `~/.claude/CLAUDE.md` | 全局 Agent 行为规则（已存在，无需复制） | ~10KB |
