"""
src/evaluation/stage_tracker.py
逐阶段追踪器：Pipeline 每个阶段结束后自动采样并计算全部评估指标，
输出 DataFrame + 质量曲线图。

使用方式：
    tracker = StageTracker(eval_cfg, run_cfg)
    tracker.record("raw",     texts_raw,    urls=urls_raw)
    tracker.record("gen1_url_filter", texts_after_url, urls=urls_after_url)
    tracker.record("gen1_final",      texts_gen1_out,  urls=urls_gen1_out)
    tracker.record("gen2_final",      texts_gen2_out,  urls=urls_gen2_out)
    tracker.record("gen3_final",      texts_gen3_out,  urls=urls_gen3_out)

    df = tracker.to_dataframe()
    tracker.plot_quality_curve(save_path="results/figures/quality_curve.png")
"""

import json
import time
import random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime


class StageTracker:
    """
    三代方法论对比实验的统一指标追踪器。
    每次调用 record() 时，对数据采样并计算全部评估指标。
    """

    def __init__(
        self,
        eval_config: Dict,
        run_config: Dict,
        quality_classifier=None,
        perplexity_scorer=None,
        toxicity_scorer=None,
    ):
        """
        Args:
            eval_config: load_eval_config() 的返回值
            run_config: load_run_config() 的返回值
            quality_classifier: EvalQualityClassifier 实例（可选，不传则跳过质量打分）
            perplexity_scorer: PerplexityScorer 实例（可选）
            toxicity_scorer: ToxicityScorer 实例（可选）
        """
        self.eval_config = eval_config
        self.run_config = run_config
        self.quality_clf = quality_classifier
        self.ppl_scorer = perplexity_scorer
        self.tox_scorer = toxicity_scorer

        self.eval_sample_size = run_config.get("eval_sample_size", 200)
        self.random_seed = run_config.get("random_seed", 42)

        self.stages: List[Dict] = []  # 按顺序记录每个阶段的指标
        self.total_doc_count_baseline: Optional[int] = None  # 原始文档数（用于计算保留率）

    def record(
        self,
        stage_name: str,
        texts: List[str],
        urls: Optional[List[str]] = None,
        quality_scores: Optional[np.ndarray] = None,
        extra_metrics: Optional[Dict] = None,
    ) -> Dict:
        """
        记录一个阶段的指标。

        Args:
            stage_name: 阶段名称（如 "raw", "gen1_url_filter", "gen1_final"）
            texts: 该阶段的全量文本列表
            urls: 对应的 URL 列表（可选，用于域名多样性计算）
            quality_scores: 已计算的质量分数（避免重复计算）
            extra_metrics: 额外指标字典（直接追加到记录中）

        Returns:
            该阶段的指标字典
        """
        from src.evaluation.diversity_metrics import compute_all_ngram_diversities, compute_domain_entropy
        from src.utils.tokenizer_utils import count_tokens_batch, get_tokenizer

        print(f"\n  📊 阶段追踪: [{stage_name}] | 文档数: {len(texts):,}")

        # 记录基线文档数（第一次调用时）
        if self.total_doc_count_baseline is None:
            self.total_doc_count_baseline = len(texts)

        # 采样（避免每阶段都对全量数据打分）
        sample_size = min(self.eval_sample_size, len(texts))
        random.seed(self.random_seed)
        sample_indices = random.sample(range(len(texts)), sample_size)
        sample_texts = [texts[i] for i in sample_indices]
        sample_urls = [urls[i] for i in sample_indices] if urls else None

        metrics = {
            "stage": stage_name,
            "timestamp": datetime.now().isoformat(),
            "doc_count": len(texts),
            "retention_rate": round(len(texts) / self.total_doc_count_baseline, 4) if self.total_doc_count_baseline else 1.0,
            "sample_size": sample_size,
        }

        # ── 质量分数 ─────────────────────────────────────────────
        if quality_scores is not None:
            q_scores = quality_scores
        elif self.quality_clf is not None:
            print(f"     质量打分（样本 {sample_size} 条）...")
            q_scores = self.quality_clf.score_batch(sample_texts, batch_size=256)
        else:
            q_scores = None

        if q_scores is not None:
            metrics.update({
                "quality_score_mean": round(float(np.mean(q_scores)), 4),
                "quality_score_p50": round(float(np.percentile(q_scores, 50)), 4),
                "quality_score_p90": round(float(np.percentile(q_scores, 90)), 4),
                "quality_score_std": round(float(np.std(q_scores)), 4),
            })

        # ── Perplexity ────────────────────────────────────────────
        if self.ppl_scorer is not None:
            print(f"     Perplexity 打分...")
            ppl_sample = sample_texts[:min(100, sample_size)]  # PPL 较慢，最多 100 条
            ppl_scores = self.ppl_scorer.score_batch(ppl_sample, show_progress=False)
            valid_ppl = ppl_scores[np.isfinite(ppl_scores)]
            if len(valid_ppl) > 0:
                metrics.update({
                    "perplexity_p50": round(float(np.percentile(valid_ppl, 50)), 2),
                    "perplexity_p90": round(float(np.percentile(valid_ppl, 90)), 2),
                    "perplexity_mean": round(float(np.mean(valid_ppl)), 2),
                })

        # ── 毒性 ──────────────────────────────────────────────────
        if self.tox_scorer is not None:
            print(f"     毒性打分...")
            tox_scores = self.tox_scorer.score_batch(sample_texts, show_progress=False)
            tox_arr = tox_scores.get("toxicity", np.zeros(sample_size))
            metrics.update({
                "toxicity_p90": round(float(np.percentile(tox_arr, 90)), 4),
                "toxicity_mean": round(float(np.mean(tox_arr)), 4),
                "toxicity_rate_50": round(float(np.mean(tox_arr > 0.5)), 4),
            })

        # ── N-gram 多样性 ─────────────────────────────────────────
        print(f"     N-gram 多样性...")
        ngram_results = compute_all_ngram_diversities(sample_texts, [1, 2, 3])
        for ng_name, ng_stats in ngram_results.items():
            metrics[f"{ng_name}_unique_ratio"] = ng_stats.get("unique_ratio", 0)

        # ── Token 数量 ────────────────────────────────────────────
        print(f"     Token 数量估算...")
        tok = get_tokenizer()
        sample_token_counts = count_tokens_batch(sample_texts[:100], tok)
        avg_tokens_per_doc = np.mean(sample_token_counts)
        estimated_total_tokens = int(avg_tokens_per_doc * len(texts))
        metrics["estimated_total_tokens"] = estimated_total_tokens
        metrics["avg_tokens_per_doc"] = round(float(avg_tokens_per_doc), 1)

        # ── 域名多样性 ────────────────────────────────────────────
        if sample_urls:
            domain_stats = compute_domain_entropy(sample_urls)
            metrics["domain_entropy"] = domain_stats.get("entropy", 0)
            metrics["n_unique_domains"] = domain_stats.get("n_domains", 0)

        # 合并额外指标
        if extra_metrics:
            metrics.update(extra_metrics)

        self.stages.append(metrics)

        # 打印摘要
        print(f"     ✅ 文档数: {metrics['doc_count']:,} | 保留率: {metrics['retention_rate']:.1%}")
        if "quality_score_mean" in metrics:
            print(f"        质量均分: {metrics['quality_score_mean']:.4f} | P90: {metrics['quality_score_p90']:.4f}")
        if "trigram_unique_ratio" in metrics:
            print(f"        3-gram 多样性: {metrics['trigram_unique_ratio']:.4f}")
        if "estimated_total_tokens" in metrics:
            print(f"        Token 数量估算: {metrics['estimated_total_tokens']:,}")

        return metrics

    def to_dataframe(self) -> pd.DataFrame:
        """将所有阶段指标转为 DataFrame（方便 Notebook 展示）。"""
        if not self.stages:
            return pd.DataFrame()
        return pd.DataFrame(self.stages).set_index("stage")

    def save(self, output_path: str) -> None:
        """保存所有阶段指标为 JSON 文件。"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.stages, f, ensure_ascii=False, indent=2)
        print(f"  ✅ 阶段指标已保存: {output_path}")

    def plot_quality_curve(
        self,
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> None:
        """
        绘制逐阶段质量曲线图（quality_score + retention_rate 双轴）。
        """
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker

        df = self.to_dataframe()
        if df.empty:
            print("  ⚠️  没有数据，无法绘图")
            return

        stages = df.index.tolist()
        x = range(len(stages))

        fig, ax1 = plt.subplots(figsize=(12, 5))
        ax2 = ax1.twinx()

        # 质量分数（左轴）
        if "quality_score_mean" in df.columns:
            ax1.plot(x, df["quality_score_mean"], "b-o", label="Quality Score (mean)", linewidth=2)
            ax1.fill_between(x,
                             df.get("quality_score_p50", df["quality_score_mean"]) * 0.9,
                             df["quality_score_p90"],
                             alpha=0.15, color="blue", label="P50-P90 band")
            ax1.set_ylabel("Quality Score", color="blue", fontsize=12)
            ax1.tick_params(axis="y", labelcolor="blue")

        # 保留率（右轴）
        if "retention_rate" in df.columns:
            ax2.bar(x, df["retention_rate"], alpha=0.25, color="green", label="Retention Rate")
            ax2.set_ylabel("Retention Rate", color="green", fontsize=12)
            ax2.tick_params(axis="y", labelcolor="green")
            ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

        ax1.set_xticks(x)
        ax1.set_xticklabels(stages, rotation=30, ha="right", fontsize=9)
        ax1.set_title("Pipeline 逐阶段质量与保留率", fontsize=14, fontweight="bold")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left")

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  ✅ 质量曲线图已保存: {save_path}")
        if show:
            plt.show()
        plt.close()
