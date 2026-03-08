"""
src/utils/profile_tables.py
五维 Profile 标准化对比表格工具

用途：在 Notebook 中生成格式化的 pandas DataFrame，
     支持双模式（smoke_test + full_run）并列、口径说明、论文参考值。

产出文件：无（仅生成 DataFrame 供 Notebook 显示）
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import pandas as pd
import numpy as np


# ═══════════════════════════════════════════════════════════════════
#  NB00 §1.1 论文参考值（Single Source of Truth）
# ═══════════════════════════════════════════════════════════════════
PAPER_REFERENCE = {
    # Gen1 整体
    'gen1_retention':       '30-40%',
    'gen1_retention_note':  '口径: Gen1输出/原始输入; 来源: FineWeb §3',
    # Gen2 整体
    'gen2_retention':       '~3-4%',
    'gen2_retention_note':  '口径: Gen2输出/原始输入 = Gen1×top-10%; 来源: DCLM Table 3',
    # Gen3 整体
    'gen3_retention':       '~38%',
    'gen3_retention_note':  '口径: Gen3输出/原始输入; 来源: Nemotron-CC §4',
    # 语言
    'english_ratio_post_lang': '≥95%',
    'language_filter_rate':    '~60%',
    # 质量相关
    'quality_filter_rate':  '~20-30%',
    'repetition_filter_rate': '~10-15%',
    # Gen2 分类器
    'gen2_classifier_auc':  '~0.85',
    'gen2_mmlu_7b':         '~64%',
    # Gen3 特有
    'bypass_false_kill':    '~18.1%',
    'rewrite_success':      '~70-80%',
    'ensemble_coverage':    '+28%',
}


# ═══════════════════════════════════════════════════════════════════
#  五维 Profile 提取定义
# ═══════════════════════════════════════════════════════════════════

def _safe_get(data: dict, *keys, default='—'):
    """安全地从嵌套 dict 取值。"""
    cur = data
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k, None)
        if cur is None:
            return default
    return cur


# 每行定义: (维度, 指标名, 提取路径, 格式化函数, 口径说明)
_METRIC_DEFS = [
    # ── 规模 ──
    ('规模', '文档数',
     lambda p: _safe_get(p, 'scale', 'n_docs'),
     lambda v: f'{v:,}' if isinstance(v, (int, float)) else v,
     '该阶段输出文档总数'),
    ('规模', '总词数',
     lambda p: _safe_get(p, 'scale', 'total_words'),
     lambda v: f'{v:,.0f}' if isinstance(v, (int, float)) else v,
     '所有文档词数之和（空格分词）'),
    ('规模', '平均词数/文档',
     lambda p: _safe_get(p, 'scale', 'avg_words'),
     lambda v: f'{v:,.0f}' if isinstance(v, (int, float)) else v,
     'total_words / n_docs'),
    ('规模', '中位数词数',
     lambda p: _safe_get(p, 'scale', 'median_words'),
     lambda v: f'{v:,.0f}' if isinstance(v, (int, float)) else v,
     'P50 文档长度（词数）'),

    # ── 质量 ──
    ('质量', 'KenLM PPL 中位数',
     lambda p: _safe_get(p, 'quality', 'stats', 'median'),
     lambda v: f'{v:,.0f}' if isinstance(v, (int, float)) else v,
     'Wikipedia 5-gram LM perplexity; 越低≈越接近 Wikipedia 风格'),
    ('质量', 'KenLM PPL 均值',
     lambda p: _safe_get(p, 'quality', 'stats', 'mean'),
     lambda v: f'{v:,.0f}' if isinstance(v, (int, float)) else v,
     '算术平均（受极端值影响大）'),
    ('质量', 'PPL head(<300)',
     lambda p: _safe_get(p, 'quality', 'buckets', 'head', 'ratio'),
     lambda v: f'{v:.1%}' if isinstance(v, (int, float)) else v,
     '分子=PPL<300 的文档数, 分母=采样总数'),
    ('质量', 'PPL middle(300-1K)',
     lambda p: _safe_get(p, 'quality', 'buckets', 'middle', 'ratio'),
     lambda v: f'{v:.1%}' if isinstance(v, (int, float)) else v,
     '分子=300≤PPL<1000 的文档数'),
    ('质量', 'PPL tail(≥1K)',
     lambda p: _safe_get(p, 'quality', 'buckets', 'tail', 'ratio'),
     lambda v: f'{v:.1%}' if isinstance(v, (int, float)) else v,
     '分子=PPL≥1000 的文档数'),

    # ── 语言 ──
    ('语言', '英文占比',
     lambda p: _safe_get(p, 'language', 'english_ratio'),
     lambda v: f'{v:.1%}' if isinstance(v, (int, float)) else v,
     '分子=fastText lid 判定 en 的文档数, 分母=采样总数'),
    ('语言', '检测语言数',
     lambda p: _safe_get(p, 'language', 'n_languages'),
     lambda v: f'{v}' if isinstance(v, (int, float)) else v,
     '采样中检测到的不同语言种类'),
    ('语言', '平均置信度',
     lambda p: _safe_get(p, 'language', 'avg_confidence'),
     lambda v: f'{v:.3f}' if isinstance(v, (int, float)) else v,
     'fastText lid 输出概率的采样均值'),

    # ── 多样性 ──
    ('多样性', 'Unigram unique ratio',
     lambda p: _safe_get(p, 'diversity', 'ngram_diversity', 'unigram', 'unique_ratio'),
     lambda v: f'{v:.4f}' if isinstance(v, (int, float)) else v,
     'unique_unigrams / total_unigrams; 越高=词汇越丰富'),
    ('多样性', 'Bigram unique ratio',
     lambda p: _safe_get(p, 'diversity', 'ngram_diversity', 'bigram', 'unique_ratio'),
     lambda v: f'{v:.4f}' if isinstance(v, (int, float)) else v,
     'unique_bigrams / total_bigrams'),
    ('多样性', 'Trigram unique ratio',
     lambda p: _safe_get(p, 'diversity', 'ngram_diversity', 'trigram', 'unique_ratio'),
     lambda v: f'{v:.4f}' if isinstance(v, (int, float)) else v,
     'unique_trigrams / total_trigrams'),
    ('多样性', '域名 Entropy(归一化)',
     lambda p: _safe_get(p, 'diversity', 'domain_entropy', 'normalized_entropy'),
     lambda v: f'{v:.4f}' if isinstance(v, (int, float)) else v,
     'Shannon entropy / log₂(n_domains); 1.0=完全均匀分布'),

    # ── 毒性 ──
    ('毒性', 'Toxicity 均值',
     lambda p: _safe_get(p, 'toxicity', 'toxicity', 'mean'),
     lambda v: f'{v:.4f}' if isinstance(v, (int, float)) else v,
     'Detoxify toxicity score 采样均值; 越低越好'),
    ('毒性', 'Toxicity >0.5 占比',
     lambda p: _safe_get(p, 'toxicity', 'toxicity', 'toxic_rate_50'),
     lambda v: f'{v:.2%}' if isinstance(v, (int, float)) else v,
     '分子=toxicity>0.5 的文档数, 分母=采样总数'),
    ('毒性', 'Toxicity >0.8 占比',
     lambda p: _safe_get(p, 'toxicity', 'toxicity', 'toxic_rate_80'),
     lambda v: f'{v:.2%}' if isinstance(v, (int, float)) else v,
     '分子=toxicity>0.8 的文档数, 分母=采样总数'),
]


def build_5dim_table(
    profiles: Dict[str, dict],
    show_caliber: bool = True,
    paper_ref_col: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    从五维 profile 字典构建标准化对比 DataFrame。

    Args:
        profiles: 有序字典，key=列名（如 "FR Gen1(100K)"），value=profile dict
        show_caliber: 是否显示口径说明列
        paper_ref_col: 指标名→论文参考值 的映射（可选）

    Returns:
        pandas DataFrame，可直接 display() 或 .to_string()

    使用示例:
        >>> profiles = {
        ...     'ST Gen1(12K)': st_gen1_profile,
        ...     'FR Gen1(100K)': fr_gen1_profile,
        ...     'FR Gen2(100K)': fr_gen2_profile,
        ... }
        >>> df = build_5dim_table(profiles)
        >>> display(df)
    """
    rows = []
    for dim, name, extract_fn, fmt_fn, caliber_text in _METRIC_DEFS:
        row = {'维度': dim, '指标': name}
        for col_name, profile in profiles.items():
            raw_val = extract_fn(profile)
            row[col_name] = fmt_fn(raw_val) if raw_val != '—' else '—'
        if paper_ref_col and name in paper_ref_col:
            row['论文参考值'] = paper_ref_col[name]
        elif paper_ref_col is not None:
            row['论文参考值'] = ''
        if show_caliber:
            row['口径说明'] = caliber_text
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def build_comparison_summary(
    profiles: Dict[str, dict],
    base_col: str,
    compare_col: str,
) -> pd.DataFrame:
    """
    构建两个 profile 的变化对比表（加 Δ 列）。

    Args:
        profiles: 同 build_5dim_table
        base_col: 基准列名
        compare_col: 对比列名

    Returns:
        DataFrame with 维度, 指标, base, compare, 变化, 口径说明
    """
    rows = []
    for dim, name, extract_fn, fmt_fn, caliber_text in _METRIC_DEFS:
        base_val = extract_fn(profiles[base_col])
        comp_val = extract_fn(profiles[compare_col])

        row = {
            '维度': dim,
            '指标': name,
            base_col: fmt_fn(base_val) if base_val != '—' else '—',
            compare_col: fmt_fn(comp_val) if comp_val != '—' else '—',
        }

        # 计算变化
        if isinstance(base_val, (int, float)) and isinstance(comp_val, (int, float)) and base_val != 0:
            if name == '文档数':
                row['变化'] = f'{comp_val/base_val:.1%}'
            elif 'PPL' in name and 'head' not in name.lower() and 'middle' not in name.lower() and 'tail' not in name.lower():
                # PPL 越低越好
                if comp_val < base_val:
                    row['变化'] = f'↓{base_val - comp_val:,.0f} (better)'
                else:
                    row['变化'] = f'↑{comp_val - base_val:,.0f}'
            elif '占比' in name or 'ratio' in name.lower() or '英文' in name:
                diff = comp_val - base_val
                if abs(diff) < 0.001:
                    row['变化'] = '≈'
                else:
                    sign = '+' if diff > 0 else ''
                    row['变化'] = f'{sign}{diff:.1%}'
            elif '毒性' in name and '均值' in name:
                diff = comp_val - base_val
                sign = '+' if diff > 0 else ''
                row['变化'] = f'{sign}{diff:.4f}'
            else:
                diff = comp_val - base_val
                sign = '+' if diff > 0 else ''
                row['变化'] = f'{sign}{diff:,.0f}'
        else:
            row['变化'] = '—'

        row['口径说明'] = caliber_text
        rows.append(row)

    return pd.DataFrame(rows)


def load_5dim_profile(path: str) -> Optional[dict]:
    """安全加载 5 维 profile JSON 文件。"""
    p = Path(path)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


def load_dual_mode_profiles(
    profile_path_template: str,
    modes: List[str] = None,
) -> Dict[str, Optional[dict]]:
    """
    加载双模式 profile 文件。

    Args:
        profile_path_template: 路径模板，{mode} 会被替换
                              如 "results/gen2_5dim_profile_{mode}.json"
        modes: 模式列表，默认 ['smoke_test', 'full_run']

    Returns:
        {mode: profile_dict_or_None}
    """
    if modes is None:
        modes = ['smoke_test', 'full_run']
    result = {}
    for mode in modes:
        path = profile_path_template.replace('{mode}', mode)
        result[mode] = load_5dim_profile(path)
    return result


# ═══════════════════════════════════════════════════════════════════
#  分类器健康度标准化表格
# ═══════════════════════════════════════════════════════════════════

def build_classifier_health_table(
    stats_dict: Dict[str, dict],
    labels: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    构建分类器健康度对比表。

    Args:
        stats_dict: {mode_key: gen2_stats_dict}
                   如 {'smoke_test': st_stats, 'full_run': fr_stats}
        labels: mode_key → 显示名

    Returns:
        DataFrame with 指标, mode columns, 参考范围, 口径
    """
    if labels is None:
        labels = {'smoke_test': 'ST(12K)', 'full_run': 'FR(100K)'}

    def _health(val, good, warn, higher_better=True):
        if higher_better:
            if val >= good:   return '✅'
            if val >= warn:   return '⚠️'
            return '❌'
        else:
            if val <= good:   return '✅'
            if val <= warn:   return '⚠️'
            return '❌'

    metric_defs = [
        ('分数均值', 'mean', lambda s: s['score_stats']['mean'], '.4f', '', ''),
        ('P50', 'p50', lambda s: s['score_stats']['p50'], '.4f', '', ''),
        ('P90', 'p90', lambda s: s['score_stats']['p90'], '.4f', '>0.7', '高分端应覆盖真实高质量文档'),
        ('Top-10% 阈值', 'threshold', lambda s: s['threshold'], '.4f', '', '分数从高到低排序，取前 10% 的分界线'),
        ('P90−P50 展开度', 'spread', lambda s: s['score_stats']['p90'] - s['score_stats']['p50'],
         '.4f', '>0.2', '衡量分类器区分能力; >0.2 健康, <0.05 异常'),
        ('输入文档数', 'input', lambda s: s['input_count'], ',d', '', 'Gen1 输出（Gen2 输入）'),
        ('输出文档数', 'output', lambda s: s['output_count'], ',d', '', 'Top-10% 保留'),
        ('保留率', 'retention', lambda s: s['retention_rate'], '.2%', '~10%', 'Gen2输出/Gen1输出'),
    ]

    rows = []
    for display_name, _, extract_fn, fmt, ref, caliber in metric_defs:
        row = {'指标': display_name}
        for mode_key, stats in stats_dict.items():
            label = labels.get(mode_key, mode_key)
            try:
                val = extract_fn(stats)
                row[label] = format(val, fmt)
            except (KeyError, TypeError):
                row[label] = '—'
        row['参考范围'] = ref
        row['口径说明'] = caliber
        rows.append(row)

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════
#  通用 Pipeline 步骤对比表
# ═══════════════════════════════════════════════════════════════════

def build_step_comparison_table(
    steps_data: Dict[str, List[dict]],
    labels: Optional[Dict[str, str]] = None,
    paper_ref: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    构建 Gen1 各步骤双模式对比表。

    Args:
        steps_data: {mode: [step_dict, ...]} 每个 step_dict 含 name, input, output, rate
        labels: mode → 显示名
        paper_ref: step_name → 论文参考过滤率

    Returns:
        DataFrame
    """
    if labels is None:
        labels = {'smoke_test': 'ST(12K)', 'full_run': 'FR(100K)'}

    # 合并所有 mode 的 step names
    all_steps = []
    for mode, steps in steps_data.items():
        for s in steps:
            if s['name'] not in all_steps:
                all_steps.append(s['name'])

    rows = []
    for step_name in all_steps:
        row = {'过滤步骤': step_name}
        for mode, steps in steps_data.items():
            label = labels.get(mode, mode)
            step = next((s for s in steps if s['name'] == step_name), None)
            if step:
                row[f'{label} 输入'] = f"{step.get('input', 0):,}"
                row[f'{label} 输出'] = f"{step.get('output', 0):,}"
                rate = step.get('rate', 0)
                row[f'{label} 条件过滤率'] = f"{rate:.1%}"
            else:
                row[f'{label} 输入'] = '—'
                row[f'{label} 输出'] = '—'
                row[f'{label} 条件过滤率'] = '—'
        if paper_ref and step_name in paper_ref:
            row['论文参考值'] = paper_ref[step_name]
        elif paper_ref is not None:
            row['论文参考值'] = ''
        row['口径'] = '条件过滤率 = 该步丢弃数/该步输入数'
        rows.append(row)

    return pd.DataFrame(rows)


def format_conclusion(
    title: str,
    findings: List[str],
    expectations: Optional[List[Tuple[str, str, str]]] = None,
) -> str:
    """
    生成标准化的结论文本。

    Args:
        title: 结论标题
        findings: 核心发现列表
        expectations: [(指标, 实际值, 预期值)] 与预期的对比

    Returns:
        格式化的结论字符串
    """
    lines = [f"\n{'='*70}", f"  {title}", f"{'='*70}"]
    lines.append("\n  核心发现:")
    for i, f in enumerate(findings, 1):
        lines.append(f"    {i}. {f}")

    if expectations:
        lines.append("\n  预期对比:")
        lines.append(f"    {'指标':<25} {'实际值':<15} {'论文预期':<15} {'判定'}")
        lines.append(f"    {'-'*65}")
        for metric, actual, expected in expectations:
            lines.append(f"    {metric:<25} {actual:<15} {expected:<15}")

    lines.append(f"{'='*70}")
    return '\n'.join(lines)
