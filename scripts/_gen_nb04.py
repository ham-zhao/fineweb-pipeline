#!/usr/bin/env python3
"""Generate notebook 04_gen3_hybrid_pipeline.ipynb from pre-computed results."""

import json
import uuid
from pathlib import Path

NB_PATH = Path(__file__).resolve().parent.parent / "notebooks" / "04_gen3_hybrid_pipeline.ipynb"


def _id():
    return uuid.uuid4().hex[:8]


def _to_source(lines):
    """Convert a list of lines into ipynb source format.

    Each line except the last must end with '\\n'.
    If input is a plain string, split it into lines first.
    """
    if isinstance(lines, str):
        lines = lines.split("\n")
    # Ensure every line except the last ends with \n
    result = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            result.append(line if line.endswith("\n") else line + "\n")
        else:
            result.append(line)
    return result


def md(source):
    """Create a markdown cell."""
    return {
        "cell_type": "markdown",
        "id": _id(),
        "metadata": {},
        "source": _to_source(source),
    }


def code(source):
    """Create a code cell."""
    return {
        "cell_type": "code",
        "id": _id(),
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _to_source(source),
    }


cells = []

# ── Markdown 1: Title + methodology overview ──
cells.append(md([
    "# 04 \u2014 \u7b2c\u4e09\u4ee3\uff1aHybrid Pipeline + Data Recovery\n",
    "\n",
    "**\u65b9\u6cd5\u8bba\u5b9a\u4f4d\uff08\u7b2c\u4e09\u4ee3\uff0cNemotron-CC 2024\uff09**\n",
    "\n",
    "\u4e09\u4e2a\u6838\u5fc3\u521b\u65b0\uff1a\n",
    "1. **\u5206\u7c7b\u5668\u96c6\u6210\uff08Classifier Ensembling\uff09**\uff1a\u591a\u4e2a\u5206\u7c7b\u5668\u53d6\u5e76\u96c6\uff0c\u6269\u5927\u9ad8\u8d28\u91cf\u8986\u76d6\u9762\n",
    "2. **\u6761\u4ef6\u6027 Heuristic Bypass**\uff1a\u5bf9\u9ad8\u8d28\u91cf\u6587\u6863\u8df3\u8fc7 heuristic\uff0c\u51cf\u5c11\u8bef\u6740\uff08Nemotron-CC \u53d1\u73b0\u8bef\u6740\u7387 18.1%\uff09\n",
    "3. **\u5408\u6210\u6570\u636e\u6539\u5199\uff08Synthetic Rephrasing\uff09**\uff1a\u4f4e\u8d28\u91cf\u6570\u636e\u7528 LLM API \u6539\u5199\u540e\u56de\u6536\n",
    "\n",
    "**\u4e0e\u7b2c\u4e8c\u4ee3\u7684\u6838\u5fc3\u533a\u522b**\uff1a\n",
    "- \u7b2c\u4e8c\u4ee3\u4e22\u5f03 90% \u6570\u636e\uff08\u6fc0\u8fdb\u8fc7\u6ee4\uff09\n",
    "- \u7b2c\u4e09\u4ee3\u5728\u8d28\u91cf\u4e0d\u964d\u7684\u524d\u63d0\u4e0b\u4fdd\u7559\u7ea6 38% \u6570\u636e\uff084\u500d\u4e8e\u7b2c\u4e8c\u4ee3\uff09\n",
    "- Nemotron-CC\uff1a8B \u6a21\u578b 15T token \u8bad\u7ec3\u540e\u8d85\u8fc7 Llama 3.1 8B\uff1aMMLU +5, ARC-Challenge +3.1\n",
    "\n",
    "> **\u672c notebook \u8bfb\u53d6\u9884\u8ba1\u7b97\u7684 pipeline \u7ed3\u679c\u8fdb\u884c\u53ef\u89c6\u5316\u5206\u6790\uff0c\u4e0d\u6267\u884c pipeline \u672c\u8eab\u3002**"
]))

# ── Markdown 2: Classifier Ensembling explanation ──
cells.append(md([
    "## Cell Group A: \u5206\u7c7b\u5668\u96c6\u6210\uff08Classifier Ensembling\uff09\n",
    "\n",
    "> **\u4e3a\u4ec0\u4e48\u9700\u8981\u96c6\u6210\uff1f\u5355\u4e00\u5206\u7c7b\u5668\u7684\u76f2\u533a\u95ee\u9898**\n",
    ">\n",
    "> \u5355\u4e00\u5206\u7c7b\u5668\u90fd\u4f1a\u6709\u8986\u76d6\u76f2\u533a\u2014\u2014\u67d0\u4e9b\u9ad8\u8d28\u91cf\u5185\u5bb9\u88ab\u6b63\u6837\u672c\u5206\u5e03\u6240\u9057\u6f0f\u3002\n",
    "> \u4f8b\u5982\uff1a\u6280\u672f\u535a\u5ba2\u53ef\u80fd\u88ab\u201c\u767e\u79d1\u98ce\u683c\u201d\u5206\u7c7b\u5668\u4f4e\u4f30\uff0c\n",
    "> \u5374\u88ab\u201c\u6559\u80b2\u7c7b\u201d\u5206\u7c7b\u5668\u9ad8\u4f30\u3002\n",
    ">\n",
    "> **Union \u7b56\u7565**\uff1a\u4efb\u4e00\u5206\u7c7b\u5668\u8ba4\u4e3a\u9ad8\u8d28\u91cf \u2192 \u5224\u4e3a\u9ad8\u8d28\u91cf\n",
    "> - \u4f18\u70b9\uff1a\u6269\u5927\u8986\u76d6\u9762\uff0c\u51cf\u5c11\u6f0f\u7f51\u4e4b\u9c7c\n",
    "> - \u7f3a\u70b9\uff1a\u53ef\u80fd\u5f15\u5165\u66f4\u591a\u566a\u58f0\uff08\u5bf9\u6bd4 Intersection \u7b56\u7565\uff09\n",
    ">\n",
    "> **Intersection \u7b56\u7565**\uff1a\u6240\u6709\u5206\u7c7b\u5668\u90fd\u8ba4\u4e3a\u9ad8\u8d28\u91cf\n",
    "> - \u7c7b\u4f3c\u7b2c\u4e8c\u4ee3\uff0c\u66f4\u4fdd\u5b88\uff0c\u8d28\u91cf\u66f4\u9ad8\u4f46\u91cf\u66f4\u5c11\n",
    ">\n",
    "> Nemotron-CC \u4f7f\u7528 Union \u7b56\u7565\uff0c\u5b9e\u73b0\u4e86 +28% unique token \u8986\u76d6\u3002"
]))

# ── Code Cell A: Load config + routing summary + data ──
cell_a = """\
# === 加载配置和预计算结果 ===
# 读取 run_config、gen3 路由汇总、gen3 输出文档、gen1 输出文档。
# 本 notebook 不执行 pipeline，只读取已有结果进行可视化分析。

import sys, json
sys.path.insert(0, '..')
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
from pathlib import Path
from src.utils.config_loader import load_run_config, get_output_path, print_config_summary

run_cfg = load_run_config()
print_config_summary(run_cfg)
mode = run_cfg.get('run_mode', 'smoke_test')

# --- 路径（根据 run_mode 自动定位） ---
gen1_dir = get_output_path(1, run_cfg)
gen3_dir = get_output_path(3, run_cfg)

# --- 加载路由汇总 ---
summary_path = gen3_dir / 'gen3_routing_summary.json'
with open(summary_path) as f:
    summary = json.load(f)
print(f'\\n已加载路由汇总 [{mode}]: {summary_path}')

# --- 加载 Gen3 输出文档 ---
gen3_path = gen3_dir / 'gen3_output.jsonl'
gen3_docs = []
with open(gen3_path) as f:
    for line in f:
        gen3_docs.append(json.loads(line))
print(f'已加载 Gen3 输出: {len(gen3_docs):,} 条')

# --- 加载 Gen1 输出文档（Gen3 的输入） ---
gen1_path = gen1_dir / 'gen1_output.jsonl'
gen1_docs = []
with open(gen1_path) as f:
    for line in f:
        gen1_docs.append(json.loads(line))
print(f'已加载 Gen1 输出 (作为 Gen3 输入): {len(gen1_docs):,} 条')

# --- 提取关键数据 ---
routing = summary['routing']
bypass = summary['bypass_analysis']
rephrase = summary['rephrasing']
total_input = routing['total']

ensemble_scores = np.array([d.get('_ensemble_score', 0) for d in gen3_docs])
synthetic_count = sum(1 for d in gen3_docs if d.get('_synthetic'))

print(f'\\n路由汇总:')
print(f'  输入文档: {total_input:,}')
print(f'  最终保留: {routing["total_kept"]:,}')
print(f'  合成文档: {synthetic_count:,}')"""
cells.append(code(cell_a))

# ── Code Cell B: Ensemble coverage visualization ──
cell_b = """\
# === 集成覆盖率可视化 ===
# 左图：Gen3 输出文档的集成分数分布
# 右图：各路由捕获的文档数量对比（柱状图）

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# -- 左图：集成分数分布 --
axes[0].hist(ensemble_scores, bins=30, color='#3498db', alpha=0.8, edgecolor='white')
axes[0].axvline(x=0.7, color='#e74c3c', linestyle='--', linewidth=1.5, label='高质量阈值 (0.7)')
axes[0].axvline(x=0.3, color='#f39c12', linestyle='--', linewidth=1.5, label='低质量阈值 (0.3)')
axes[0].set_xlabel('集成分数 (Ensemble Score)')
axes[0].set_ylabel('文档数')
axes[0].set_title('Gen3 输出文档的集成分数分布')
axes[0].legend(fontsize=9)

# -- 右图：各路由捕获的文档数量 --
route_names = ['高质量\\n(bypass)', '中等质量\\n(直接通过)', '待改写', '丢弃']
route_counts = [
    routing['high_quality']['count'],
    routing['medium_quality']['count'],
    routing['to_rephrase']['count'],
    routing['discarded']['count'],
]
route_colors = ['#28a745', '#4CAF50', '#17a2b8', '#6c757d']

bars = axes[1].bar(route_names, route_counts, color=route_colors, alpha=0.85,
                    edgecolor='white', linewidth=1.5)
for bar, count in zip(bars, route_counts):
    pct = count / total_input * 100
    axes[1].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + max(route_counts) * 0.02,
                 f'{count:,}\\n({pct:.1f}%)',
                 ha='center', va='bottom', fontsize=9)
axes[1].set_ylabel('文档数')
axes[1].set_title('各路由捕获的文档数量')

plt.suptitle('第三代：分类器集成分析', fontweight='bold', fontsize=13)
plt.tight_layout()
plt.savefig('../results/figures/04_gen3_ensemble_coverage.png', dpi=150, bbox_inches='tight')
plt.show()
print('图表已保存: results/figures/04_gen3_ensemble_coverage.png')"""
cells.append(code(cell_b))

# ── Markdown 3: Conditional Bypass explanation ──
cells.append(md([
    "## Cell Group B: \u6761\u4ef6\u6027 Heuristic Bypass\n",
    "\n",
    "> **\u6838\u5fc3\u95ee\u9898\uff1aHeuristic \u4f1a\u8bef\u6740\u591a\u5c11\u9ad8\u8d28\u91cf\u6587\u6863\uff1f**\n",
    ">\n",
    "> Nemotron-CC \u7684\u5173\u952e\u53d1\u73b0\uff1a\u5bf9 fastText \u5224\u5b9a\u4e3a\u9ad8\u8d28\u91cf\u7684\u6587\u6863\uff0c\n",
    "> \u5982\u679c\u518d\u5e94\u7528 heuristic filter\uff0c\u4f1a\u8bef\u6740 **18.1% \u7684\u9ad8\u8d28\u91cf token**\u3002\n",
    ">\n",
    "> **\u8bef\u6740\u7684\u539f\u56e0\u4e3e\u4f8b**\uff1a\n",
    "> - \u4ee3\u7801\u6587\u6863\uff1a\u542b\u5927\u91cf\u7279\u6b8a\u5b57\u7b26 \u2192 \u88ab Gopher \u7684\u201calpha ratio\u201d\u89c4\u5219\u8fc7\u6ee4\n",
    "> - \u6280\u672f\u6559\u7a0b\uff1a\u542b\u4ee3\u7801\u7247\u6bb5\uff08\u77ed\u884c\uff09\u2192 \u88ab C4 \u7684\u884c\u89c4\u5219\u8fc7\u6ee4\n",
    "> - \u95ee\u7b54\u683c\u5f0f\u6587\u672c\uff1a\u5e73\u5747\u53e5\u5b50\u77ed \u2192 \u88ab Gopher \u7684 avg_sentence_length \u8fc7\u6ee4\n",
    ">\n",
    "> **\u89e3\u51b3\u65b9\u6848\uff08Bypass \u8def\u7531\uff09**\uff1a\n",
    "> - score >= 0.7\uff1a\u76f4\u63a5\u4fdd\u7559\uff08\u8df3\u8fc7 heuristic\uff09\n",
    "> - 0.3 <= score < 0.7\uff1a\u76f4\u63a5\u4fdd\u7559\uff08\u5df2\u5728 Gen1 \u901a\u8fc7 heuristic\uff09\n",
    "> - score < 0.3\uff1a\u9001\u53bb LLM \u6539\u5199\u6216\u4e22\u5f03"
]))

# ── Code Cell C: Conditional bypass routing funnel ──
cell_c = """\
# === 条件性 Bypass 路由漏斗图 ===
# 可视化路由流程：输入 -> 高质量 bypass -> heuristic -> 改写 -> 丢弃
# 突出 bypass_save_rate（对标 Nemotron-CC 18.1% 发现）

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# -- 左图：路由漏斗（横向条形图） --
hq = routing['high_quality']['count']
mq = routing['medium_quality']['count']
tr = routing['to_rephrase']['count']
dc = routing['discarded']['count']

funnel_labels = [
    f'输入\\n({total_input:,})',
    f'高质量 bypass\\n({hq:,})',
    f'中等质量 直接通过\\n({mq:,})',
    f'待改写\\n({tr:,})',
    f'丢弃\\n({dc:,})',
]
funnel_values = [total_input, hq, mq, tr, dc]
funnel_colors = ['#2c3e50', '#28a745', '#4CAF50', '#17a2b8', '#6c757d']

y_pos = range(len(funnel_labels))
hbars = axes[0].barh(y_pos, funnel_values, color=funnel_colors, alpha=0.85,
                      edgecolor='white', linewidth=1.5)
axes[0].set_yticks(list(y_pos))
axes[0].set_yticklabels(funnel_labels, fontsize=9)
axes[0].invert_yaxis()
axes[0].set_xlabel('文档数')
axes[0].set_title('路由漏斗：文档流向', fontweight='bold')
for bar, val in zip(hbars, funnel_values):
    if val > 0:
        axes[0].text(bar.get_width() + max(funnel_values) * 0.02,
                     bar.get_y() + bar.get_height() / 2,
                     f'{val:,} ({val/total_input:.1%})',
                     va='center', fontsize=9)

# -- 右图：Bypass 价值分析 --
hq_count = bypass['high_quality_count']
filtered_count = bypass['would_be_filtered_count']
filtered_rate = bypass['would_be_filtered_rate']
saved_count = hq_count - filtered_count

if hq_count > 0:
    if filtered_count > 0:
        wedge_sizes = [saved_count, filtered_count]
        wedge_labels = [
            f'Heuristic 也会通过\\n({saved_count})',
            f'Heuristic 会误杀\\n({filtered_count})',
        ]
        wedge_colors = ['#28a745', '#e74c3c']
    else:
        wedge_sizes = [hq_count]
        wedge_labels = [f'Heuristic 也会通过\\n({hq_count})']
        wedge_colors = ['#28a745']
    axes[1].pie(wedge_sizes, labels=wedge_labels, colors=wedge_colors,
                autopct='%1.1f%%', startangle=90, textprops={'fontsize': 10})
else:
    axes[1].text(0.5, 0.5, '无高质量 bypass 文档',
                 ha='center', va='center', fontsize=12, transform=axes[1].transAxes)

title_line2 = f'本实验误杀率: {filtered_rate:.1%} | Nemotron-CC: 18.1%'
axes[1].set_title('Bypass 价值分析\\n' + title_line2,
                   fontweight='bold', fontsize=11)

plt.suptitle('第三代：条件性路由结果（Bypass 分析）', fontweight='bold', fontsize=13)
plt.tight_layout()
plt.savefig('../results/figures/04_gen3_routing.png', dpi=150, bbox_inches='tight')
plt.show()
print('图表已保存: results/figures/04_gen3_routing.png')"""
cells.append(code(cell_c))

# ── Markdown 4: Synthetic Rephrasing explanation ──
cells.append(md([
    "## Cell Group C: \u5408\u6210\u6570\u636e\u6539\u5199\uff08Synthetic Rephrasing\uff09\n",
    "\n",
    "> **\u6838\u5fc3\u7406\u5ff5\uff1a\u4f4e\u8d28\u91cf\u6570\u636e\u4e0d\u662f\u5783\u573e\uff0c\u800c\u662f\u5f85\u6539\u5199\u7684\u539f\u6750\u6599**\n",
    ">\n",
    "> Nemotron-CC \u7684\u6d1e\u89c1\uff1a\u901a\u8fc7 LLM API \u5c06\u4f4e\u8d28\u91cf\u6587\u672c\u6539\u5199\u4e3a\u9ad8\u8d28\u91cf\u6587\u672c\uff0c\u5b9e\u73b0\u6570\u636e\u56de\u6536\u3002\n",
    ">\n",
    "> **\u6210\u672c\u6548\u76ca\u5206\u6790**\uff1a\n",
    "> - Anthropic Haiku\uff1a\u7ea6 $0.0003/\u5343 token\uff08\u8f93\u5165\uff09\n",
    "> - \u6539\u5199 300 \u6761\u6587\u6863\uff08full_run \u914d\u7f6e\uff09\u2248 300 x 500 tokens = 150K tokens \u2248 **$0.05**\n",
    "> - \u8fd9 300 \u6761\u53d8\u6210\u9ad8\u8d28\u91cf\u6570\u636e\u540e\uff0c\u4ef7\u503c >> \u6210\u672c\n",
    ">\n",
    "> **\u6539\u5199\u540e\u7684\u8d28\u91cf\u9a8c\u8bc1**\uff1a\n",
    "> \u6539\u5199\u540e\u7684\u6587\u6863\u9700\u8981\u901a\u8fc7\u8bc4\u4f30\u5206\u7c7b\u5668\u9a8c\u8bc1\uff08\u5206\u6570 >= 0.4 \u624d\u4fdd\u7559\uff09\uff0c\n",
    "> \u907f\u514d\u201c\u6539\u5199\u5931\u8d25\u201d\uff08LLM \u751f\u6210\u4e86\u66f4\u5dee\u7684\u6587\u672c\uff09\u7684\u60c5\u51b5\u3002"
]))

# ── Code Cell D: Rephrasing stats display ──
cell_d = """\
# === 改写统计 ===
# 显示 LLM 改写的尝试/成功/失败统计

print('=' * 50)
print('  合成数据改写 (Synthetic Rephrasing) 统计')
print('=' * 50)

if rephrase.get('skipped', False):
    print('  状态: 已跳过 (未配置 API Key 或无待改写文档)')
    rephrase_candidates = routing['to_rephrase']['count']
    print(f'  待改写候选: {rephrase_candidates:,} 条')
    print()
    print('  提示: 配置 configs/api_config.yaml 中的 api_key 可启用 LLM 改写')
else:
    attempted = rephrase.get('attempted', 0)
    succeeded = rephrase.get('succeeded', 0)
    failed = rephrase.get('failed', 0)
    print(f'  尝试改写: {attempted:,} 条')
    print(f'  成功: {succeeded:,} 条')
    print(f'  失败: {failed:,} 条')
    if attempted > 0:
        print(f'  成功率: {succeeded/attempted:.1%}')
    print(f'  合成文档占最终输出: {synthetic_count:,} / {len(gen3_docs):,}')

print()
elapsed = summary['elapsed_seconds']
print(f'  Pipeline 执行时间: {elapsed:.2f} 秒')"""
cells.append(code(cell_d))

# ── Markdown 5: Five-dimension profiling ──
cells.append(md([
    "## Cell Group D: \u4e94\u7ef4\u6570\u636e\u8d28\u91cf\u6f14\u8fdb\uff08Gen1 \u8f93\u51fa vs Gen3 \u8f93\u51fa\uff09\n",
    "\n",
    "\u5bf9 Gen3 \u8fc7\u6ee4\u524d\u540e\u7684\u6570\u636e\u8ba1\u7b97\u4e94\u7ef4\u8d28\u91cf profile\uff0c\u91cf\u5316 Hybrid Pipeline \u7684\u7efc\u5408\u6548\u679c\u3002\n",
    "\n",
    "| \u7ef4\u5ea6 | \u9884\u671f\u53d8\u5316 |\n",
    "|------|------|\n",
    "| \u89c4\u6a21 | \u4fdd\u7559\u7ea6 38%\uff084\u500d\u4e8e Gen2\uff09 |\n",
    "| \u8d28\u91cf | KenLM PPL \u964d\u4f4e |\n",
    "| \u8bed\u8a00 | \u82f1\u6587 ~100% |\n",
    "| \u591a\u6837\u6027 | \u5e94\u4fdd\u6301\uff08Ensemble \u6269\u5927\u8986\u76d6\uff09 |\n",
    "| \u6bd2\u6027 | \u5e94\u964d\u4f4e\u6216\u6301\u5e73 |",
]))

# ── Code Cell D2: Five-dimension profiling ──
cell_d2 = """\
# === 五维质量 Profile（Gen1 输出 vs Gen3 输出） ===
from src.evaluation.baseline_profiler import compute_profile, print_profile_summary

sample_size = min(500, len(gen1_docs), len(gen3_docs))

print("正在计算 Gen1 输出的五维 Profile（Gen3 输入）...")
gen1_texts = [d.get('text', '') for d in gen1_docs]
gen1_urls = [d.get('url', '') for d in gen1_docs]
gen1_profile = compute_profile(
    gen1_texts, urls=gen1_urls,
    sample_size=sample_size,
    model_dir='../data/models',
)

print("\\n正在计算 Gen3 输出的五维 Profile...")
gen3_texts = [d.get('text', '') for d in gen3_docs]
gen3_urls = [d.get('url', '') for d in gen3_docs]
gen3_profile = compute_profile(
    gen3_texts, urls=gen3_urls,
    sample_size=min(sample_size, len(gen3_docs)),
    model_dir='../data/models',
)

print_profile_summary(gen1_profile, label="Gen1 输出（Gen3 输入）")
print_profile_summary(gen3_profile, label="Gen3 输出（Hybrid Pipeline）")"""
cells.append(code(cell_d2))

cell_d3 = """\
# === 五维演进对比表 ===
def safe_get(profile, *keys, default='N/A'):
    obj = profile
    for k in keys:
        if isinstance(obj, dict):
            obj = obj.get(k, default)
        else:
            return default
    return obj

print("=" * 80)
print("  五维质量演进对比: Gen1 输出 vs Gen3 输出")
print("=" * 80)
print(f"  {'指标':<35} {'Gen1输出':>18} {'Gen3输出':>18} {'变化':>10}")
print(f"  {'-'*80}")

n_in = safe_get(gen1_profile, 'scale', 'n_docs', default=0)
n_out = safe_get(gen3_profile, 'scale', 'n_docs', default=0)
if n_in:
    print(f"  {'文档数':<35} {n_in:>18,} {n_out:>18,} {n_out/n_in:.1%}")

w_in = safe_get(gen1_profile, 'scale', 'avg_words', default=0)
w_out = safe_get(gen3_profile, 'scale', 'avg_words', default=0)
if isinstance(w_in, (int, float)) and isinstance(w_out, (int, float)):
    print(f"  {'平均词数/文档':<35} {w_in:>18,.0f} {w_out:>18,.0f} {'+' if w_out>w_in else ''}{w_out-w_in:,.0f}")

q_in = safe_get(gen1_profile, 'quality', 'stats', 'median', default=None)
q_out = safe_get(gen3_profile, 'quality', 'stats', 'median', default=None)
if q_in and q_out and isinstance(q_in, (int, float)):
    print(f"  {'KenLM PPL 中位数 (越低越好)':<35} {q_in:>18,.0f} {q_out:>18,.0f} {'better' if q_out<q_in else 'worse'}")
    for bname, label in [('head', 'PPL head(<300)'), ('middle', 'PPL middle'), ('tail', 'PPL tail(>=1000)')]:
        b_in = safe_get(gen1_profile, 'quality', 'buckets', bname, 'ratio', default=0)
        b_out = safe_get(gen3_profile, 'quality', 'buckets', bname, 'ratio', default=0)
        if isinstance(b_in, (int, float)):
            print(f"  {label:<35} {b_in:>18.1%} {b_out:>18.1%}")

en_in = safe_get(gen1_profile, 'language', 'english_ratio', default=0)
en_out = safe_get(gen3_profile, 'language', 'english_ratio', default=0)
if isinstance(en_in, (int, float)):
    print(f"  {'英文占比':<35} {en_in:>18.1%} {en_out:>18.1%}")

for ng, label in [('unigram', 'Unigram unique ratio'), ('bigram', 'Bigram unique ratio')]:
    d_in = safe_get(gen1_profile, 'diversity', 'ngram_diversity', ng, 'unique_ratio', default=None)
    d_out = safe_get(gen3_profile, 'diversity', 'ngram_diversity', ng, 'unique_ratio', default=None)
    if d_in and isinstance(d_in, (int, float)):
        print(f"  {label:<35} {d_in:>18.4f} {d_out:>18.4f}")

de_in = safe_get(gen1_profile, 'diversity', 'domain_entropy', 'normalized_entropy', default=None)
de_out = safe_get(gen3_profile, 'diversity', 'domain_entropy', 'normalized_entropy', default=None)
if de_in and isinstance(de_in, (int, float)):
    print(f"  {'域名 Entropy (归一化)':<35} {de_in:>18.4f} {de_out:>18.4f}")

t_in = safe_get(gen1_profile, 'toxicity', 'toxicity', 'mean', default=None)
t_out = safe_get(gen3_profile, 'toxicity', 'toxicity', 'mean', default=None)
if t_in and isinstance(t_in, (int, float)):
    print(f"  {'毒性均值':<35} {t_in:>18.4f} {t_out:>18.4f}")
    tr_in = safe_get(gen1_profile, 'toxicity', 'toxicity', 'toxic_rate_50', default=0)
    tr_out = safe_get(gen3_profile, 'toxicity', 'toxicity', 'toxic_rate_50', default=0)
    if isinstance(tr_in, (int, float)):
        print(f"  {'毒性>0.5 占比':<35} {tr_in:>18.2%} {tr_out:>18.2%}")

print(f"  {'='*80}")

# 保存
import os
os.makedirs('../results', exist_ok=True)
profiles = {'gen1_output': gen1_profile, 'gen3_output': gen3_profile}
with open('../results/gen3_5dim_profile.json', 'w', encoding='utf-8') as f:
    json.dump(profiles, f, ensure_ascii=False, indent=2, default=str)
print(f"五维 Profile 已保存: results/gen3_5dim_profile.json")"""
cells.append(code(cell_d3))

# ── Code Cell E: Summary table + gen2 comparison ──
cell_e = """\
# === 第三代最终汇总 + 与第二代对比 ===

total_kept = routing['total_kept']
retention_rate = total_kept / total_input
gen2_approx = int(total_input * 0.10)  # 第二代约保留 top-10%

print('=' * 60)
print('  第三代 Hybrid Pipeline -- 最终结论')
print('=' * 60)
print(f'  输入文档数: {total_input:,}')
print(f'  最终输出: {total_kept:,} 条')
hq_c = routing['high_quality']['count']
mq_c = routing['medium_quality']['count']
print(f'  |-- 高质量(bypass): {hq_c:,} 条')
print(f'  |-- 中等质量(直接通过): {mq_c:,} 条')
print(f'  +-- 合成数据(改写): {synthetic_count:,} 条')
print(f'  总保留率: {retention_rate:.1%}')
print()
print('  对比第二代（top-10%）:')
print(f'  第二代输出约: {gen2_approx:,} 条')
print(f'  第三代输出约: {total_kept:,} 条')
if gen2_approx > 0:
    print(f'  数据量倍数: {total_kept/gen2_approx:.1f}x')
print()

# 汇总表格
header_route = '路由'
header_count = '文档数'
header_pct = '占比'
print('路由明细表：')
print(f'{header_route:<20} {header_count:>8} {header_pct:>8}')
print('-' * 38)
for name, key in [('高质量 bypass', 'high_quality'),
                   ('中等质量 直接通过', 'medium_quality'),
                   ('待改写', 'to_rephrase'),
                   ('丢弃', 'discarded')]:
    c = routing[key]['count']
    r = routing[key]['rate']
    print(f'{name:<20} {c:>8,} {r:>8.1%}')
print('-' * 38)
footer_label = '最终保留'
print(f'{footer_label:<20} {total_kept:>8,} {retention_rate:>8.1%}')
print()
print('  下一步 -> Notebook 05：去重分析')"""
cells.append(code(cell_e))

# ── Assemble notebook ──
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11.0",
        },
    },
    "cells": cells,
}

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print(f"Notebook written to {NB_PATH}")
print(f"  {len(cells)} cells: {sum(1 for c in cells if c['cell_type']=='markdown')} markdown, "
      f"{sum(1 for c in cells if c['cell_type']=='code')} code")
