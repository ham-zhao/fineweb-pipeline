"""
src/evaluation/diversity_metrics.py
多样性指标计算器

⚠️  Distribution Shift（分布偏移）的核心认知：
────────────────────────────────────────────────────
过滤不仅改变数据量，还改变数据分布。

激进质量过滤的典型副作用：
  百科/学术内容 → 过度富集（Wikipedia、学术论文的质量分高）
  对话、新闻、创意写作 → 被不成比例地过滤

后果：
  MMLU（百科问答）分高 ← 训练数据偏向百科
  对话、创作任务反而变差 ← 这类数据被过滤掉了

因此多样性指标和质量指标同等重要，是评估过滤策略的两个维度。
单看质量，容易忽视分布偏移；单看多样性，容易忽视质量提升。
────────────────────────────────────────────────────
"""

import re
import numpy as np
from collections import Counter
from typing import List, Dict, Optional, Tuple
from urllib.parse import urlparse
from tqdm import tqdm


def compute_ngram_diversity(
    texts: List[str],
    n: int = 3,
    sample_size: Optional[int] = None,
) -> Dict:
    """
    计算 N-gram 多样性指标（unique N-gram ratio）。

    Unique N-gram Ratio = 唯一 N-gram 数 / 总 N-gram 数
    越高 = 内容越多样，越低 = 内容越重复

    Args:
        texts: 文本列表
        n: N-gram 大小（1=unigram, 2=bigram, 3=trigram）
        sample_size: 采样数量（加速计算）

    Returns:
        dict，包含 unique_ratio, total_ngrams, unique_ngrams, top10_ngrams
    """
    if sample_size and len(texts) > sample_size:
        import random
        texts = random.sample(texts, sample_size)

    all_ngrams = []
    for text in texts:
        tokens = text.lower().split()
        ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
        all_ngrams.extend(ngrams)

    if not all_ngrams:
        return {"unique_ratio": 0.0, "total_ngrams": 0, "unique_ngrams": 0}

    counter = Counter(all_ngrams)
    total = len(all_ngrams)
    unique = len(counter)

    return {
        "unique_ratio": round(unique / total, 4),
        "total_ngrams": total,
        "unique_ngrams": unique,
        "top10_ngrams": [
            {" ".join(ng): count}
            for ng, count in counter.most_common(10)
        ],
    }


def compute_all_ngram_diversities(
    texts: List[str],
    ngram_sizes: List[int] = [1, 2, 3],
    sample_size: Optional[int] = None,
) -> Dict:
    """
    计算多个 N 值的多样性指标。

    Returns:
        dict，key 为 "unigram" / "bigram" / "trigram"
    """
    results = {}
    name_map = {1: "unigram", 2: "bigram", 3: "trigram", 4: "4gram"}

    for n in ngram_sizes:
        name = name_map.get(n, f"{n}gram")
        results[name] = compute_ngram_diversity(texts, n=n, sample_size=sample_size)

    return results


def compute_domain_entropy(
    urls: List[str],
    top_k: int = 50,
) -> Dict:
    """
    计算域名分布的 Shannon Entropy（领域多样性代理指标）。

    Shannon Entropy = -sum(p_i * log2(p_i))
    最大熵 = log2(n_domains)（所有域名等频率）
    熵越高 = 域名分布越均匀 = 数据来源越多样

    Args:
        urls: URL 列表
        top_k: 只统计 top-k 域名（避免长尾噪声）

    Returns:
        dict，包含 entropy, max_entropy, normalized_entropy, top_domains
    """
    domains = []
    for url in urls:
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            # 去掉 www. 前缀
            domain = re.sub(r"^www\.", "", domain)
            if domain:
                domains.append(domain)
        except Exception:
            pass

    if not domains:
        return {"entropy": 0.0, "n_domains": 0}

    counter = Counter(domains)
    top_domains = counter.most_common(top_k)
    total = sum(count for _, count in top_domains)

    # Shannon Entropy
    entropy = 0.0
    for _, count in top_domains:
        p = count / total
        if p > 0:
            entropy -= p * np.log2(p)

    n_domains = len(counter)
    max_entropy = np.log2(min(n_domains, top_k))  # 理论最大熵

    return {
        "entropy": round(float(entropy), 4),
        "max_entropy": round(float(max_entropy), 4),
        "normalized_entropy": round(float(entropy / max_entropy) if max_entropy > 0 else 0, 4),
        "n_domains": n_domains,
        "top_10_domains": [{"domain": d, "count": c} for d, c in top_domains[:10]],
    }


def compute_length_distribution(texts: List[str]) -> Dict:
    """
    计算文档长度分布（按词数）。

    Returns:
        dict，包含各分位数统计
    """
    lengths = [len(text.split()) for text in texts]
    arr = np.array(lengths)

    return {
        "mean": round(float(np.mean(arr)), 1),
        "median": round(float(np.median(arr)), 1),
        "p10": round(float(np.percentile(arr, 10)), 1),
        "p25": round(float(np.percentile(arr, 25)), 1),
        "p75": round(float(np.percentile(arr, 75)), 1),
        "p90": round(float(np.percentile(arr, 90)), 1),
        "p99": round(float(np.percentile(arr, 99)), 1),
        "min": int(np.min(arr)),
        "max": int(np.max(arr)),
        "total_words": int(np.sum(arr)),
    }


def compute_diversity_report(
    texts: List[str],
    urls: Optional[List[str]] = None,
    sample_size: Optional[int] = None,
    ngram_sizes: List[int] = [1, 2, 3],
) -> Dict:
    """
    综合多样性报告（Notebook 中一键调用）。

    Returns:
        完整的多样性指标字典
    """
    print(f"  📐 计算多样性指标（共 {len(texts):,} 条文档）...")

    report = {}

    # N-gram 多样性
    print("     N-gram 多样性...")
    report["ngram_diversity"] = compute_all_ngram_diversities(texts, ngram_sizes, sample_size)

    # 长度分布
    print("     长度分布...")
    report["length_distribution"] = compute_length_distribution(texts)

    # 域名熵（需要 URL）
    if urls:
        print("     域名分布熵...")
        report["domain_entropy"] = compute_domain_entropy(urls)

    return report


def compare_diversity(
    baseline: Dict,
    filtered: Dict,
    label_baseline: str = "原始数据",
    label_filtered: str = "过滤后",
) -> None:
    """
    对比两个多样性报告并打印差异（Notebook 中用于直观对比）。
    """
    print(f"\n  📊 多样性对比：{label_baseline} vs {label_filtered}")
    print(f"  {'指标':<25} {label_baseline:>12} {label_filtered:>12} {'变化':>10}")
    print(f"  {'─' * 60}")

    # N-gram 多样性对比
    for ng_name in ["unigram", "bigram", "trigram"]:
        b_val = baseline.get("ngram_diversity", {}).get(ng_name, {}).get("unique_ratio", 0)
        f_val = filtered.get("ngram_diversity", {}).get(ng_name, {}).get("unique_ratio", 0)
        change = f_val - b_val
        arrow = "↑" if change > 0 else "↓" if change < 0 else "→"
        print(f"  {ng_name + ' unique ratio':<25} {b_val:>12.4f} {f_val:>12.4f} {arrow} {abs(change):.4f}")

    # 长度分布对比
    b_med = baseline.get("length_distribution", {}).get("median", 0)
    f_med = filtered.get("length_distribution", {}).get("median", 0)
    change = f_med - b_med
    arrow = "↑" if change > 0 else "↓"
    print(f"  {'doc length (median words)':<25} {b_med:>12.1f} {f_med:>12.1f} {arrow} {abs(change):.1f}")
