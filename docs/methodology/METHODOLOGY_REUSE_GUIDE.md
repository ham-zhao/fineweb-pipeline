# 方法论复用指南：如何在新项目中启用

> 将 fineweb-pipeline 项目提炼的标准方法论，复用到 post-train-pipeline、safety-dataset 或任何新 ML 项目中。

---

## 一、需要复制的文件

只需复制 **2 个文件**到目标项目的 `docs/` 目录下：

```bash
# 源项目路径
SRC="/Users/pengjuzhao/Desktop/claude code/tiktok-ml-projects/fineweb-pipeline"

# 替换 <TARGET_PROJECT> 为目标项目路径
cp "$SRC/docs/STANDARD_METHODOLOGY.md" <TARGET_PROJECT>/docs/
cp "$SRC/docs/METHODOLOGY_COMPACT.md"  <TARGET_PROJECT>/docs/
```

**不需要**复制全局 `~/.claude/CLAUDE.md`——它已经自动对所有项目生效。

**已完成的复制：**
- [x] post-train-pipeline/docs/ （2026-03-10）
- [x] safety-dataset/docs/ （2026-03-10）

---

## 二、给 Claude Code 的初始化命令

在目标项目的 Claude Code 窗口中粘贴以下**统一命令**（无需区分项目是否已有 CLAUDE.md，Agent 会自己判断并处理）：

```
请完成方法论初始化：

1. 读取 docs/STANDARD_METHODOLOGY.md（完整方法论）和 docs/METHODOLOGY_COMPACT.md（精简版），理解全部内容
2. 读取全局规则 ~/.claude/CLAUDE.md，确认理解
3. 用 METHODOLOGY_COMPACT.md 的全部内容覆盖项目 CLAUDE.md（忽略原有内容，从零开始），末尾加一行：完整方法论参考 docs/STANDARD_METHODOLOGY.md
4. 在 memory 中记录：方法论已导入，完整版位置，导入日期
5. 未来执行中发现新的方法论规则时（如踩坑、反模式、用户纠正），按以下流程处理：
   - 识别：主动向用户提出"发现一条潜在方法论规则：[描述]，是否纳入？"
   - 确认后分流：
     - 全局通用规则 → 写入 ~/.claude/CLAUDE.md
     - 项目特有规则 → 写入 memory
     - 方法论性质的规则 → 写入 docs/METHODOLOGY_DELTA.md（增量文件），不要直接修改 STANDARD_METHODOLOGY.md
   - METHODOLOGY_DELTA.md 格式：每条规则包含标题、发现日期、触发场景、规则内容、验证状态（已验证/待验证）
   - 提醒合并：当 DELTA 累积 5 条以上已验证规则时，主动提醒用户执行合并
6. 输出初始化核销表，逐项确认完成
```

---

## 三、是否需要 /compact

**建议先执行 `/compact`。** 如果项目窗口已有之前的对话历史（如已跑过一轮数据），上下文中会残留旧的规则和习惯。`/compact` 会压缩旧对话但保留关键信息，让 Agent 以更干净的状态接收新方法论，避免新旧规则冲突。

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

### 规则分流（Agent 自动执行）

```
每个项目执行中发现新规则时：

  全局规则 → 直接写 ~/.claude/CLAUDE.md → 所有项目自动继承，无需同步
  项目特有 → 直接写 MEMORY.md → 不需要同步
  方法论性质 → 写入 docs/METHODOLOGY_DELTA.md → 增量文件，不动主文档
```

### 增量文件格式（docs/METHODOLOGY_DELTA.md）

每个项目独立维护一份"新规则收集箱"，不直接修改 `STANDARD_METHODOLOGY.md`：

```markdown
# 方法论增量规则

## 来自项目 X

### 新规则 1：[标题]
- 发现日期：YYYY-MM-DD
- 触发场景：什么情况下暴露了这个问题
- 规则内容：具体的方法论规则
- 验证状态：已验证 / 待验证

### 新规则 2：...
```

### 定期合并流程（项目结束或阶段性回顾时）

在 fineweb-pipeline 窗口告诉 Agent：

```
请合并项目 2 和项目 3 的方法论增量：
1. 读取所有项目的 docs/METHODOLOGY_DELTA.md
2. 筛选"已验证"的规则
3. 合并到 docs/STANDARD_METHODOLOGY.md
4. 重新复制到其他项目
5. 清空已合并的 DELTA 条目
```

### 为什么不直接改主文档

- **审计追溯**：每条新规则记录了发现日期、来源项目、触发场景
- **检查门禁**：只有"已验证"的规则才会合并到主文档
- **主文档稳定**：STANDARD_METHODOLOGY.md 始终是经过验证的稳定版本

---

## 六、文件清单

| 文件 | 用途 | 大小 |
|------|------|------|
| `docs/STANDARD_METHODOLOGY.md` | 完整方法论（29 条原则 + 案例 + 参考文献） | ~22KB |
| `docs/METHODOLOGY_COMPACT.md` | 精简版（22 条，<500 字），可直接作为 prompt 前缀 | ~2KB |
| `~/.claude/CLAUDE.md` | 全局 Agent 行为规则（已存在，无需复制） | ~10KB |
