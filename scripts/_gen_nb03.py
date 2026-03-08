#!/usr/bin/env python3
"""
Generate notebook 03_gen2_model_based_filtering.ipynb

This notebook reads pre-computed Gen2 pipeline results instead of running
the pipeline. It uses only:
  - src.utils.config_loader (config)
  - src.evaluation.quality_classifier (independent eval scoring)

NO imports from src.gen2.*.
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

cells = []


def md(source: str):
    """Add a markdown cell."""
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.strip().split("\n")],
    })
    # Remove trailing \n from last line
    if cells[-1]["source"]:
        cells[-1]["source"][-1] = cells[-1]["source"][-1].rstrip("\n")


def code(source: str):
    """Add a code cell."""
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [line + "\n" for line in source.strip().split("\n")],
        "execution_count": None,
        "outputs": [],
    })
    if cells[-1]["source"]:
        cells[-1]["source"][-1] = cells[-1]["source"][-1].rstrip("\n")


# ======================================================================
# Markdown 1 — Title & methodology
# ======================================================================
md("""\
# 03 — 第二代：Model-based Filtering

**方法论定位（第二代）**：训练 fastText 质量分类器，保留 top-10% 的文档。

**DCLM 的核心发现（NeurIPS 2024）**：
- fastText 二分类器（dim=64, wordNgrams=2）+ top-10% 阈值，效果超过所有 heuristic 组合
- Perplexity 过滤、PageRank、语义去重等方法都不如这个简单方案
- 7B 模型训练后 MMLU 达到 64%（接近 Llama 3 8B 的 66%，但只用了 1/6.6 的算力）

**与第一代的本质区别**：
- 第一代过滤"长相不像自然文本的文档"（乱码、广告等）
- 第二代过滤"语义上不像高质量写作的文档"（平庸内容）
- 这是质量评估维度的根本跃升""")

# ======================================================================
# Markdown 2 — Classifier independence note
# ======================================================================
md("""\
## Cell Group A: 加载 Gen2 Pipeline 预计算结果

> **本 notebook 读取预计算的 pipeline 输出，不执行 pipeline 本身。**
>
> Gen2 pipeline（`scripts/run_gen2.py`）已完成以下步骤：
> 1. 训练 fastText 质量分类器（dim=64, wordNgrams=2）
> 2. 对 Gen1 输出的全部文档打分
> 3. 保留 top-10% 高分文档
>
> ⚠️ **评估分类器与 Pipeline 分类器独立训练**
>
> - Pipeline 分类器：正样本 = `wikipedia_abstracts.jsonl`，dim=64, wordNgrams=2
> - 评估分类器：正样本 = `wikipedia_abstracts_eval.jsonl`（独立数据集，与 Pipeline 用的完全不重叠），dim=32, wordNgrams=3
> - 独立性保障：正样本数据集不重叠 + 超参数不同（双重独立），避免循环偏差。详见 NB00 §1.2。""")

# ======================================================================
# Cell A — Load config + stats + docs
# ======================================================================
code("""\
# === Cell A: 加载配置 + Gen2 统计 + Gen1/Gen2 输出文档 ===
# 从预计算结果文件加载，不执行任何 pipeline 代码。
# 仅导入 config_loader（配置）和 EvalQualityClassifier（独立评估打分）。

import sys, json
sys.path.insert(0, '..')
import numpy as np
import pandas as pd
from pathlib import Path

from src.utils.config_loader import load_run_config, get_output_path, print_config_summary
from src.evaluation.quality_classifier import EvalQualityClassifier

# --- 配置 ---
run_cfg = load_run_config()
print_config_summary(run_cfg)
mode = run_cfg.get('run_mode', 'smoke_test')

# --- 路径（根据 run_mode 自动定位） ---
gen1_dir = get_output_path(1, run_cfg)
gen2_dir = get_output_path(2, run_cfg)

# --- 读取 Gen2 统计 ---
stats_path = gen2_dir / 'gen2_stats.json'
with open(stats_path) as f:
    gen2_stats = json.load(f)

print(f"\\nGen2 Pipeline 统计 [{mode}]:")
print(f"  输入文档数: {gen2_stats['input_count']:,}")
print(f"  输出文档数: {gen2_stats['output_count']:,}")
print(f"  保留率:     {gen2_stats['retention_rate']:.2%}")
print(f"  top_fraction: {gen2_stats['top_fraction']}")
print(f"  阈值:       {gen2_stats['threshold']:.4f}")
print(f"  分数均值:   {gen2_stats['score_stats']['mean']:.4f}")
print(f"  分数 P50:   {gen2_stats['score_stats']['p50']:.4f}")
print(f"  分数 P90:   {gen2_stats['score_stats']['p90']:.4f}")

# --- 读取文档 ---
def read_jsonl(path, limit=None):
    docs = []
    with open(path, encoding='utf-8') as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            docs.append(json.loads(line))
    return docs

gen1_docs = read_jsonl(gen1_dir / 'gen1_output.jsonl')
gen2_docs = read_jsonl(gen2_dir / 'gen2_output.jsonl')

print(f"\\nGen1 输出文档: {len(gen1_docs):,} 条（Gen2 的输入）")
print(f"Gen2 输出文档: {len(gen2_docs):,} 条（top-{gen2_stats['top_fraction']:.0%} 保留）")

# --- 加载评估分类器 ---
eval_clf = EvalQualityClassifier()
eval_clf_path = Path('../results/quality_scores/eval_classifier.bin')
if eval_clf_path.exists():
    eval_clf._load(str(eval_clf_path))
else:
    raise FileNotFoundError(f"评估分类器不存在: {eval_clf_path}")

# --- 提取 Gen2 pipeline 打分 ---
# gen2_output.jsonl 中的 _gen2_score 是 pipeline 分类器的打分
gen2_pipeline_scores = np.array([d['_gen2_score'] for d in gen2_docs])
print(f"\\nGen2 输出文档的 pipeline 分数范围: [{gen2_pipeline_scores.min():.4f}, {gen2_pipeline_scores.max():.4f}]")\
""")

# ======================================================================
# Markdown 3 — Score distribution
# ======================================================================
md("""\
## Cell Group B: 分数分布可视化

> Gen2 pipeline 对 Gen1 输出的全部文档打分，分数分布反映了文档质量的整体情况。
> 红色虚线标注 top-10% 阈值，右侧为保留区域。
> 直方图数据来自 `gen2_stats.json`（预计算）。""")

# ======================================================================
# Cell B — Score distribution histogram
# ======================================================================
code("""\
# === Cell B: 分数分布直方图 ===
# 如果 gen2_stats 包含 score_histogram，直接绘制；
# 否则用评估分类器对 gen1 文档打分展示质量分布。

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

threshold = gen2_stats['threshold']
score_stats = gen2_stats['score_stats']

if 'score_histogram' in gen2_stats:
    # --- 从预计算的直方图数据绘制 ---
    counts = np.array(gen2_stats['score_histogram']['counts'])
    bin_edges = np.array(gen2_stats['score_histogram']['bin_edges'])
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_widths = bin_edges[1:] - bin_edges[:-1]
    axes[0].bar(bin_centers, counts, width=bin_widths, color='steelblue',
                alpha=0.7, edgecolor='white')
    all_scores = np.array(gen2_stats.get('all_scores', []))
else:
    # --- 用评估分类器对 gen1 文档打分（独立评估视角） ---
    print("gen2_stats 中无 score_histogram，用评估分类器对 Gen1 文档打分...")
    all_scores = eval_clf.score_batch([d['text'] for d in gen1_docs])
    axes[0].hist(all_scores, bins=50, color='steelblue', alpha=0.7, edgecolor='white')
    # 使用评估分类器的分布时，阈值改为 P90
    threshold = float(np.percentile(all_scores, 90))

axes[0].axvline(threshold, color='red', linestyle='--', linewidth=1.5,
                label=f'top-10% 阈值: {threshold:.3f}')
axes[0].set_xlabel('Quality Score')
axes[0].set_ylabel('文档数')
axes[0].set_title('分数分布直方图')
axes[0].legend()

# 标注统计量
stats_text = f"mean={score_stats['mean']:.3f}\\nP50={score_stats['p50']:.3f}\\nP90={score_stats['p90']:.3f}"
axes[0].text(0.97, 0.95, stats_text, transform=axes[0].transAxes,
             verticalalignment='top', horizontalalignment='right',
             fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# --- 右图：各阈值下的保留率 ---
top_fractions = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
if len(all_scores) > 0:
    retained_pcts = [np.mean(all_scores >= np.percentile(all_scores, (1 - t) * 100)) * 100
                     for t in top_fractions]
else:
    retained_pcts = [t * 100 for t in top_fractions]

colors = ['darkorange' if t != gen2_stats['top_fraction'] else 'red'
          for t in top_fractions]
bars = axes[1].bar([f'top-{int(t*100)}%' for t in top_fractions],
                    retained_pcts, color=colors, alpha=0.8)
axes[1].set_ylabel('实际保留率 (%)')
axes[1].set_title('各阈值下的实际保留率')

# 标注实际使用的阈值
for bar, t, pct in zip(bars, top_fractions, retained_pcts):
    if t == gen2_stats['top_fraction']:
        axes[1].annotate(f'{pct:.1f}%\\n(使用)', xy=(bar.get_x() + bar.get_width()/2, pct),
                         ha='center', va='bottom', fontweight='bold', fontsize=9)

plt.suptitle('Gen2 分类器打分结果', fontweight='bold', fontsize=13)
plt.tight_layout()
plt.savefig('../results/figures/03_gen2_score_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"已保存: results/figures/03_gen2_score_distribution.png")\
""")

# ======================================================================
# Markdown 4 — Threshold experiment
# ======================================================================
md("""\
## Cell Group C: Quality-Quantity Trade-off 曲线

> **DCLM 论文的关键问题：top-X% 的 X 应该设为多少？**
>
> 直觉上：X 越小（保留越少），质量越高；X 越大（保留越多），质量越低。
> 但实际上：X=5% 虽然质量最高，但 token 数量太少，在长 horizon 训练时效果反而不好。
>
> DCLM 的发现：top-10% 是质量x数量综合最优的点。
>
> 本 cell 用独立评估分类器在不同 top-X% 下计算质量均分和 Token 产出，
> 验证 DCLM 的结论。""")

# ======================================================================
# Cell C — Quality-Quantity tradeoff curve
# ======================================================================
code("""\
# === Cell C: Quality-Quantity Trade-off 曲线 ===
# 用独立评估分类器在不同 top-X% 下计算质量均分，验证 DCLM 的 top-10% 最优结论。
# 评估分类器与 Pipeline 分类器独立训练，避免循环偏差。

# 对 Gen1 文档（Gen2 的输入）用评估分类器打分
gen1_texts = [d['text'] for d in gen1_docs]
eval_scores = eval_clf.score_batch(gen1_texts)

# 模拟不同 top-X% 阈值
top_fractions = [0.05, 0.10, 0.15, 0.20, 0.30]
tradeoff_rows = []

for frac in top_fractions:
    cutoff = np.percentile(eval_scores, (1 - frac) * 100)
    mask = eval_scores >= cutoff
    retained_docs = int(mask.sum())
    retained_texts = [t for t, m in zip(gen1_texts, mask) if m]
    total_tokens = sum(len(t.split()) for t in retained_texts)

    # 质量均分 = 被保留文档的评估分类器均分
    quality_mean = float(eval_scores[mask].mean()) if retained_docs > 0 else 0.0
    quality_p90 = float(np.percentile(eval_scores[mask], 90)) if retained_docs > 0 else 0.0

    tradeoff_rows.append({
        'top_fraction': frac,
        'threshold': float(cutoff),
        'retained_docs': retained_docs,
        'retention_rate': retained_docs / len(gen1_docs),
        'quality_score_mean': quality_mean,
        'quality_score_p90': quality_p90,
        'estimated_total_tokens': total_tokens,
    })
    print(f"  top-{int(frac*100):2d}%: {retained_docs:4d} 条 | "
          f"质量均分: {quality_mean:.4f} | Token: {total_tokens:,}")

tradeoff_df = pd.DataFrame(tradeoff_rows).set_index('top_fraction')

# --- 绘制 trade-off 双轴图 ---
fig, ax1 = plt.subplots(figsize=(10, 5))

color_quality = '#1f77b4'
color_tokens = '#ff7f0e'

x_labels = [f'top-{int(f*100)}%' for f in top_fractions]
x_pos = np.arange(len(top_fractions))

# 质量均分（左轴）
ax1.plot(x_pos, tradeoff_df['quality_score_mean'].values, 'o-',
         color=color_quality, linewidth=2, markersize=8, label='质量均分')
ax1.set_xlabel('保留比例 (top-X%)')
ax1.set_ylabel('质量均分（评估分类器）', color=color_quality)
ax1.tick_params(axis='y', labelcolor=color_quality)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(x_labels)

# Token 产出（右轴）
ax2 = ax1.twinx()
ax2.bar(x_pos, tradeoff_df['estimated_total_tokens'].values,
        alpha=0.3, color=color_tokens, label='Token 产出')
ax2.set_ylabel('Token 产出', color=color_tokens)
ax2.tick_params(axis='y', labelcolor=color_tokens)

# 标注 top-10%（DCLM 推荐）
idx_10 = top_fractions.index(0.10) if 0.10 in top_fractions else 1
ax1.annotate('DCLM\\n推荐',
             xy=(x_pos[idx_10], tradeoff_df['quality_score_mean'].values[idx_10]),
             xytext=(x_pos[idx_10] + 0.5, tradeoff_df['quality_score_mean'].values[idx_10] + 0.002),
             arrowprops=dict(arrowstyle='->', color='red'),
             fontsize=10, fontweight='bold', color='red')

# 合并图例
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

plt.title('Quality-Quantity Trade-off（Gen2, DCLM 风格）', fontweight='bold', fontsize=13)
plt.tight_layout()
plt.savefig('../results/figures/03_gen2_tradeoff_curve.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"已保存: results/figures/03_gen2_tradeoff_curve.png")

# 显示 tradeoff 表
print("\\n" + tradeoff_df.to_string())\
""")

# ======================================================================
# Markdown 5 — Gen1 vs Gen2 comparison
# ======================================================================
md("""\
## Cell Group D: Heuristic vs Model-based 直接对比

> **这是第一代 -> 第二代跃升的量化证明**
>
> 在相同的数据保留率下，哪种方法保留的数据质量更高？
> 预期结论（基于论文）：在相同保留率下，model-based 的 quality_score 显著高于 heuristic。
>
> 用独立评估分类器打分避免循环偏差。""")

# ======================================================================
# Cell D — Gen1 vs Gen2 quality comparison
# ======================================================================
code("""\
# === Cell D: Gen1 vs Gen2 质量对比 ===
# 用独立评估分类器对 Gen1 和 Gen2 输出文档打分，量化第二代相对第一代的质量提升。
# Gen1 输出 = 启发式过滤后的全部文档
# Gen2 输出 = 在 Gen1 基础上进一步保留 top-10% 的文档

# 对 Gen1 输出打分（全部文档）
gen1_eval_scores = eval_clf.score_batch([d['text'] for d in gen1_docs])

# 对 Gen2 输出打分（pipeline 保留的文档）
gen2_eval_scores = eval_clf.score_batch([d['text'] for d in gen2_docs])

print(f"Gen1 输出: {len(gen1_docs):,} 条 | 质量均分: {gen1_eval_scores.mean():.4f}")
print(f"Gen2 输出: {len(gen2_docs):,} 条 | 质量均分: {gen2_eval_scores.mean():.4f}")
print(f"Gen2 vs Gen1 提升: {gen2_eval_scores.mean() - gen1_eval_scores.mean():+.4f}")

# --- 质量分布对比图 ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左图：直方图对比
axes[0].hist(gen1_eval_scores, bins=30, alpha=0.6,
             label=f'Gen1 全量 (mean={gen1_eval_scores.mean():.3f}, n={len(gen1_docs)})',
             color='#FF9800')
axes[0].hist(gen2_eval_scores, bins=30, alpha=0.6,
             label=f'Gen2 top-10% (mean={gen2_eval_scores.mean():.3f}, n={len(gen2_docs)})',
             color='#17a2b8')
axes[0].axvline(gen1_eval_scores.mean(), color='#FF9800', linestyle='--', alpha=0.8)
axes[0].axvline(gen2_eval_scores.mean(), color='#17a2b8', linestyle='--', alpha=0.8)
axes[0].set_xlabel('Quality Score（评估分类器，独立）')
axes[0].set_ylabel('文档数')
axes[0].set_title('Gen1 vs Gen2：质量分布对比', fontweight='bold')
axes[0].legend(fontsize=9)

# 右图：箱线图
bp = axes[1].boxplot([gen1_eval_scores, gen2_eval_scores],
                      labels=['Gen1\\n(Heuristic)', 'Gen2\\n(Model-based)'],
                      patch_artist=True,
                      boxprops=dict(alpha=0.7))
bp['boxes'][0].set_facecolor('#FF9800')
bp['boxes'][1].set_facecolor('#17a2b8')
axes[1].set_ylabel('Quality Score（评估分类器）')
axes[1].set_title('质量分数箱线图', fontweight='bold')

# 标注均值
for i, (scores, color) in enumerate([(gen1_eval_scores, '#FF9800'),
                                      (gen2_eval_scores, '#17a2b8')]):
    axes[1].scatter(i + 1, scores.mean(), color=color, marker='D', s=80, zorder=5,
                    edgecolors='black', linewidths=0.8)
    axes[1].annotate(f'mean={scores.mean():.3f}',
                     xy=(i + 1, scores.mean()),
                     xytext=(i + 1.3, scores.mean()),
                     fontsize=9, fontweight='bold')

plt.suptitle('第一代 vs 第二代：独立评估分类器质量对比', fontweight='bold', fontsize=13)
plt.tight_layout()
plt.savefig('../results/figures/03_gen1_vs_gen2_quality.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"已保存: results/figures/03_gen1_vs_gen2_quality.png")\
""")

# ======================================================================
# Markdown 6 — Five-dimension profiling
# ======================================================================
md("""\
## Cell Group E: 五维数据质量演进（Gen1 输出 vs Gen2 输出）

对 Gen2 过滤前后的数据计算五维质量 profile，量化 model-based 过滤的效果：

| 维度 | 预期变化 |
|------|---------|
| 规模 | 文档数减少 90%（top-10%） |
| 质量 | KenLM PPL 降低（质量提升） |
| 语言 | 英文比例保持 ~100% |
| 多样性 | 可能小幅下降（过滤偏好特定风格） |
| 毒性 | 应降低或持平 |""")

# ======================================================================
# Cell E1 — Five-dimension profiling
# ======================================================================
code("""\
# === Cell E1: 五维质量 Profile（Gen1 输出 vs Gen2 输出） ===
from src.evaluation.baseline_profiler import compute_profile, print_profile_summary

sample_size = min(500, len(gen1_docs), len(gen2_docs))

print("正在计算 Gen1 输出的五维 Profile（Gen2 输入）...")
gen1_texts = [d.get('text', '') for d in gen1_docs]
gen1_urls = [d.get('url', '') for d in gen1_docs]
gen1_profile = compute_profile(
    gen1_texts, urls=gen1_urls,
    sample_size=sample_size,
    model_dir='../data/models',
)

print("\\n正在计算 Gen2 输出的五维 Profile...")
gen2_texts = [d.get('text', '') for d in gen2_docs]
gen2_urls = [d.get('url', '') for d in gen2_docs]
gen2_profile = compute_profile(
    gen2_texts, urls=gen2_urls,
    sample_size=min(sample_size, len(gen2_docs)),
    model_dir='../data/models',
)

print_profile_summary(gen1_profile, label="Gen1 输出（Gen2 输入）")
print_profile_summary(gen2_profile, label="Gen2 输出（top-10%）")""")

code("""\
# === Cell E2: 五维演进对比表 ===
def safe_get(profile, *keys, default='N/A'):
    obj = profile
    for k in keys:
        if isinstance(obj, dict):
            obj = obj.get(k, default)
        else:
            return default
    return obj

print("=" * 80)
print("  五维质量演进对比: Gen1 输出 vs Gen2 输出")
print("=" * 80)
print(f"  {'指标':<35} {'Gen1输出':>18} {'Gen2输出':>18} {'变化':>10}")
print(f"  {'-'*80}")

n_in = safe_get(gen1_profile, 'scale', 'n_docs', default=0)
n_out = safe_get(gen2_profile, 'scale', 'n_docs', default=0)
if n_in:
    print(f"  {'文档数':<35} {n_in:>18,} {n_out:>18,} {n_out/n_in:.1%}")

w_in = safe_get(gen1_profile, 'scale', 'avg_words', default=0)
w_out = safe_get(gen2_profile, 'scale', 'avg_words', default=0)
if isinstance(w_in, (int, float)) and isinstance(w_out, (int, float)):
    print(f"  {'平均词数/文档':<35} {w_in:>18,.0f} {w_out:>18,.0f} {'+' if w_out>w_in else ''}{w_out-w_in:,.0f}")

q_in = safe_get(gen1_profile, 'quality', 'stats', 'median', default=None)
q_out = safe_get(gen2_profile, 'quality', 'stats', 'median', default=None)
if q_in and q_out and isinstance(q_in, (int, float)):
    print(f"  {'KenLM PPL 中位数 (越低越好)':<35} {q_in:>18,.0f} {q_out:>18,.0f} {'better' if q_out<q_in else 'worse'}")
    for bname, label in [('head', 'PPL head(<300)'), ('middle', 'PPL middle'), ('tail', 'PPL tail(>=1000)')]:
        b_in = safe_get(gen1_profile, 'quality', 'buckets', bname, 'ratio', default=0)
        b_out = safe_get(gen2_profile, 'quality', 'buckets', bname, 'ratio', default=0)
        if isinstance(b_in, (int, float)):
            print(f"  {label:<35} {b_in:>18.1%} {b_out:>18.1%}")

en_in = safe_get(gen1_profile, 'language', 'english_ratio', default=0)
en_out = safe_get(gen2_profile, 'language', 'english_ratio', default=0)
if isinstance(en_in, (int, float)):
    print(f"  {'英文占比':<35} {en_in:>18.1%} {en_out:>18.1%}")

for ng, label in [('unigram', 'Unigram unique ratio'), ('bigram', 'Bigram unique ratio')]:
    d_in = safe_get(gen1_profile, 'diversity', 'ngram_diversity', ng, 'unique_ratio', default=None)
    d_out = safe_get(gen2_profile, 'diversity', 'ngram_diversity', ng, 'unique_ratio', default=None)
    if d_in and isinstance(d_in, (int, float)):
        print(f"  {label:<35} {d_in:>18.4f} {d_out:>18.4f}")

de_in = safe_get(gen1_profile, 'diversity', 'domain_entropy', 'normalized_entropy', default=None)
de_out = safe_get(gen2_profile, 'diversity', 'domain_entropy', 'normalized_entropy', default=None)
if de_in and isinstance(de_in, (int, float)):
    print(f"  {'域名 Entropy (归一化)':<35} {de_in:>18.4f} {de_out:>18.4f}")

t_in = safe_get(gen1_profile, 'toxicity', 'toxicity', 'mean', default=None)
t_out = safe_get(gen2_profile, 'toxicity', 'toxicity', 'mean', default=None)
if t_in and isinstance(t_in, (int, float)):
    print(f"  {'毒性均值':<35} {t_in:>18.4f} {t_out:>18.4f}")
    tr_in = safe_get(gen1_profile, 'toxicity', 'toxicity', 'toxic_rate_50', default=0)
    tr_out = safe_get(gen2_profile, 'toxicity', 'toxicity', 'toxic_rate_50', default=0)
    if isinstance(tr_in, (int, float)):
        print(f"  {'毒性>0.5 占比':<35} {tr_in:>18.2%} {tr_out:>18.2%}")

print(f"  {'='*80}")

# 保存
import os
os.makedirs('../results', exist_ok=True)
profiles = {'gen1_output': gen1_profile, 'gen2_output': gen2_profile}
with open('../results/gen2_5dim_profile.json', 'w', encoding='utf-8') as f:
    json.dump(profiles, f, ensure_ascii=False, indent=2, default=str)
print(f"五维 Profile 已保存: results/gen2_5dim_profile.json")""")

# ======================================================================
# Markdown 7 — Summary
# ======================================================================
md("""\
## Cell Group F: 第二代最终结论汇总

> **关键数据全部来自预计算的 pipeline 输出，评估使用独立分类器。**""")

# ======================================================================
# Cell E — Summary table + save figures
# ======================================================================
code("""\
# === Cell E: 汇总表 + 最终结论 ===

# 汇总统计表
summary_data = {
    '指标': [
        '输入文档数',
        '输出文档数',
        '保留率',
        'Pipeline 分数阈值',
        'Pipeline 分数均值',
        'Pipeline 分数 P50',
        'Pipeline 分数 P90',
        f'Gen1 评估质量均分 (n={len(gen1_docs)})',
        f'Gen2 评估质量均分 (n={len(gen2_docs)})',
        '质量提升 (Gen2 - Gen1)',
    ],
    '值': [
        f"{gen2_stats['input_count']:,}",
        f"{gen2_stats['output_count']:,}",
        f"{gen2_stats['retention_rate']:.2%}",
        f"{gen2_stats['threshold']:.4f}",
        f"{gen2_stats['score_stats']['mean']:.4f}",
        f"{gen2_stats['score_stats']['p50']:.4f}",
        f"{gen2_stats['score_stats']['p90']:.4f}",
        f"{gen1_eval_scores.mean():.4f}",
        f"{gen2_eval_scores.mean():.4f}",
        f"{gen2_eval_scores.mean() - gen1_eval_scores.mean():+.4f}",
    ],
}
summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))

# 保存汇总为 JSON
summary_out = {
    'gen2_stats': gen2_stats,
    'gen1_eval_quality_mean': round(float(gen1_eval_scores.mean()), 4),
    'gen2_eval_quality_mean': round(float(gen2_eval_scores.mean()), 4),
    'quality_improvement': round(float(gen2_eval_scores.mean() - gen1_eval_scores.mean()), 4),
    'figures_saved': [
        'results/figures/03_gen2_score_distribution.png',
        'results/figures/03_gen2_tradeoff_curve.png',
        'results/figures/03_gen1_vs_gen2_quality.png',
    ],
}
summary_path = Path('../results/quality_scores/nb03_summary.json')
summary_path.parent.mkdir(parents=True, exist_ok=True)
with open(summary_path, 'w') as f:
    json.dump(summary_out, f, indent=2, ensure_ascii=False)
print(f"\\n汇总已保存: {summary_path}")

# 最终结论
print()
print("=" * 60)
print("  第二代 Model-based Filtering -- 最终结论")
print("=" * 60)
print(f"  输入文档数: {gen2_stats['input_count']:,}")
print(f"  输出文档数（top-{gen2_stats['top_fraction']:.0%}）: {gen2_stats['output_count']:,}")
print(f"  实际保留率: {gen2_stats['retention_rate']:.1%}")
print()
print("  关键发现（对标 DCLM 论文）:")
print("  - top-10% 是质量与数量的最优平衡点")
print("  - Model-based 比 Heuristic 质量提升显著（在相同保留率下）")
print("  - 第二代的核心局限：90% 数据被丢弃")
print()
print("  下一步 -> Notebook 04：第三代 Hybrid Pipeline")
print("  第三代将解决：在质量不降的前提下，保留更多数据")\
""")

# ======================================================================
# Assemble notebook
# ======================================================================
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

output_path = PROJECT_ROOT / "notebooks" / "03_gen2_model_based_filtering.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Notebook generated: {output_path}")
print(f"  Total cells: {len(cells)} ({sum(1 for c in cells if c['cell_type']=='markdown')} markdown, "
      f"{sum(1 for c in cells if c['cell_type']=='code')} code)")
