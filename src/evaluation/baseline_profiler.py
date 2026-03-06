"""
src/evaluation/baseline_profiler.py
五维数据质量 Profiler

对任意一批文档计算完整的 5 维 profile：
  1. 规模：文档数、总 token、平均长度
  2. 质量：KenLM perplexity 分布 + head/middle/tail 分桶
  3. 语言：fastText langid 语言分布
  4. 多样性：n-gram unique ratio + 域名 Shannon entropy
  5. 毒性：detoxify 多维度毒性统计

每代 pipeline 都调用同一个 profiler，形成可比的"数据质量演进表"。
"""

import numpy as np
from typing import List, Dict, Optional
from pathlib import Path


def compute_profile(
    texts: List[str],
    urls: Optional[List[str]] = None,
    sample_size: int = 500,
    run_kenlm: bool = True,
    run_toxicity: bool = True,
    run_language: bool = True,
    run_diversity: bool = True,
    model_dir: str = "data/models",
) -> Dict:
    """
    计算一批文档的 5 维质量 profile。

    Args:
        texts: 文档文本列表
        urls: 对应的 URL 列表（多样性维度需要）
        sample_size: 采样条数（KenLM/毒性较慢，采样计算）
        run_kenlm: 是否计算 KenLM perplexity
        run_toxicity: 是否计算毒性
        run_language: 是否计算语言分布
        run_diversity: 是否计算多样性
        model_dir: 模型文件目录

    Returns:
        dict，包含 5 个维度的完整统计
    """
    import random
    random.seed(42)

    n_docs = len(texts)
    sample_idx = random.sample(range(n_docs), min(sample_size, n_docs))
    sample_texts = [texts[i] for i in sample_idx]
    sample_urls = [urls[i] for i in sample_idx] if urls else None

    profile = {
        "n_docs": n_docs,
        "sample_size": len(sample_texts),
    }

    # --- 1. 规模 ---
    word_counts = [len(t.split()) for t in texts]
    profile["scale"] = {
        "n_docs": n_docs,
        "total_words": sum(word_counts),
        "avg_words": float(np.mean(word_counts)),
        "median_words": float(np.median(word_counts)),
    }

    # --- 2. 质量（KenLM perplexity）---
    if run_kenlm:
        try:
            from src.evaluation.kenlm_scorer import KenLMScorer
            scorer = KenLMScorer(model_dir=model_dir)
            ppl_scores = scorer.score_batch(sample_texts, show_progress=True)
            profile["quality"] = {
                "method": "KenLM Wikipedia perplexity",
                "stats": scorer.compute_statistics(ppl_scores),
                "buckets": scorer.bucket_analysis(ppl_scores),
            }
        except Exception as e:
            profile["quality"] = {"error": str(e)}
    else:
        profile["quality"] = {"skipped": True}

    # --- 3. 语言 ---
    if run_language:
        try:
            from src.evaluation.language_detector import LanguageDetector
            detector = LanguageDetector(model_path=f"{model_dir}/lid.176.bin")
            lang_results = detector.detect_batch(sample_texts, show_progress=True)
            profile["language"] = detector.compute_statistics(lang_results)
        except Exception as e:
            profile["language"] = {"error": str(e)}
    else:
        profile["language"] = {"skipped": True}

    # --- 4. 多样性 ---
    if run_diversity:
        try:
            from src.evaluation.diversity_metrics import compute_diversity_report
            profile["diversity"] = compute_diversity_report(
                texts=sample_texts,
                urls=sample_urls or [],
                sample_size=len(sample_texts),
            )
        except Exception as e:
            profile["diversity"] = {"error": str(e)}
    else:
        profile["diversity"] = {"skipped": True}

    # --- 5. 毒性 ---
    if run_toxicity:
        try:
            from src.evaluation.toxicity_scorer import ToxicityScorer
            tox_scorer = ToxicityScorer(device="cpu")
            tox_scores = tox_scorer.score_batch(sample_texts, show_progress=True)
            profile["toxicity"] = tox_scorer.compute_statistics(tox_scores)
        except Exception as e:
            profile["toxicity"] = {"error": str(e)}
    else:
        profile["toxicity"] = {"skipped": True}

    return profile


def print_profile_summary(profile: Dict, label: str = ""):
    """打印 5 维 profile 汇总表。"""
    if label:
        print(f"\n{'=' * 70}")
        print(f"  {label}")
        print(f"{'=' * 70}")

    # 规模
    s = profile.get("scale", {})
    print(f"\n  [规模]")
    print(f"    文档数: {s.get('n_docs', 'N/A'):,}")
    print(f"    总词数: {s.get('total_words', 'N/A'):,}")
    print(f"    平均词数/文档: {s.get('avg_words', 0):.0f}")
    print(f"    中位数词数: {s.get('median_words', 0):.0f}")

    # 质量
    q = profile.get("quality", {})
    if "stats" in q:
        qs = q["stats"]
        print(f"\n  [质量] KenLM Wikipedia Perplexity（采样 {qs.get('n_valid', 0)} 条）")
        print(f"    中位数: {qs.get('median', 0):.0f}  |  均值: {qs.get('mean', 0):.0f}")
        print(f"    P10: {qs.get('p10', 0):.0f}  |  P25: {qs.get('p25', 0):.0f}  |  P75: {qs.get('p75', 0):.0f}  |  P90: {qs.get('p90', 0):.0f}")
        buckets = q.get("buckets", {})
        if buckets:
            for name in ["head", "middle", "tail"]:
                b = buckets.get(name, {})
                print(f"    {name:>6}: {b.get('count', 0):>5,} ({b.get('ratio', 0):.1%}) — {b.get('description', '')}")
    elif "error" in q:
        print(f"\n  [质量] 错误: {q['error']}")

    # 语言
    lang = profile.get("language", {})
    if "english_ratio" in lang:
        print(f"\n  [语言] fastText lid（采样 {lang.get('total_docs', 0)} 条）")
        print(f"    英文占比: {lang['english_ratio']:.1%} ({lang.get('english_count', 0):,} / {lang.get('total_docs', 0):,})")
        print(f"    检测语言数: {lang.get('n_languages', 0)}")
        print(f"    平均置信度: {lang.get('avg_confidence', 0):.3f}")
        top = lang.get("top_languages", [])[:5]
        if top:
            print(f"    Top 5: ", end="")
            print(" | ".join(f"{t['lang']}:{t['ratio']:.1%}" for t in top))

    # 多样性
    div = profile.get("diversity", {})
    if "ngram_diversity" in div:
        print(f"\n  [多样性]")
        for ng_name, ng_stats in div["ngram_diversity"].items():
            print(f"    {ng_name} unique ratio: {ng_stats.get('unique_ratio', 0):.4f}")
        de = div.get("domain_entropy", {})
        if de:
            print(f"    域名 Shannon Entropy: {de.get('entropy', 0):.4f} (归一化: {de.get('normalized_entropy', 0):.4f})")

    # 毒性
    tox = profile.get("toxicity", {})
    if "toxicity" in tox:
        t = tox["toxicity"]
        print(f"\n  [毒性] detoxify（采样 {profile.get('sample_size', 0)} 条）")
        print(f"    toxicity 均值: {t.get('mean', 0):.4f}  |  >0.5 占比: {t.get('toxic_rate_50', 0):.2%}  |  >0.8 占比: {t.get('toxic_rate_80', 0):.2%}")
        for dim in ["severe_toxicity", "insult", "identity_attack"]:
            d = tox.get(dim, {})
            if d:
                print(f"    {dim}: 均值 {d.get('mean', 0):.4f}, >0.5 = {d.get('toxic_rate_50', 0):.2%}")
