"""
src/gen2/threshold_tuner.py
阈值调优器：在不同保留比例下评估 Quality-Quantity Trade-off

DCLM 核心发现：
  - top-10% 是 MMLU 最优的阈值
  - 保留更多数据（top-20%, top-30%）会降低质量
  - 激进过滤（top-5%）token 数量太少，长 token 训练时效果变差
  - 这就是第三代要解决的问题：在质量不降的前提下保留更多数据

本模块复现 DCLM 的阈值实验，产出 quality-quantity trade-off 曲线。
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional


class ThresholdTuner:
    """
    在不同阈值下运行分类器，记录 quality-quantity 指标，
    产出 trade-off 曲线数据（供 Notebook 可视化）。
    """

    def __init__(
        self,
        classifier,                # Gen2QualityClassifier 实例
        eval_classifier,           # EvalQualityClassifier 实例（独立评估）
        thresholds: Optional[List[float]] = None,
    ):
        """
        Args:
            classifier: pipeline 用分类器（打分过滤）
            eval_classifier: 评估专用分类器（衡量过滤效果）
            thresholds: 要测试的保留比例列表（默认 [0.05, 0.10, ..., 0.50]）
        """
        self.classifier = classifier
        self.eval_clf = eval_classifier
        self.thresholds = thresholds or [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]

    def run_experiments(
        self,
        texts: List[str],
        pipeline_scores: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        在所有阈值下运行实验，记录指标。

        Args:
            texts: 文档文本列表
            pipeline_scores: 已计算的 pipeline 分类器分数（避免重复计算）

        Returns:
            DataFrame，index 为阈值，columns 为各指标
        """
        from src.utils.tokenizer_utils import count_tokens_batch, get_tokenizer

        print(f"  🔬 阈值实验：{self.thresholds}")
        print(f"     文档总数: {len(texts):,}")

        # 计算 pipeline 分类器分数
        if pipeline_scores is None:
            print("     计算 Pipeline 分类器分数...")
            pipeline_scores = self.classifier.score_batch(texts)

        tokenizer = get_tokenizer()

        results = []
        for top_fraction in self.thresholds:
            threshold = np.percentile(pipeline_scores, (1 - top_fraction) * 100)
            mask = pipeline_scores >= threshold
            retained_texts = [t for t, m in zip(texts, mask) if m]

            if not retained_texts:
                continue

            # 评估分类器打分（独立评估）
            eval_scores = self.eval_clf.score_batch(retained_texts[:500])

            # Token 数量
            token_counts = count_tokens_batch(retained_texts[:200], tokenizer)
            total_tokens = int(np.mean(token_counts) * len(retained_texts))

            results.append({
                "top_fraction": top_fraction,
                "threshold": round(float(threshold), 4),
                "retained_docs": len(retained_texts),
                "retention_rate": round(float(np.mean(mask)), 4),
                "quality_score_mean": round(float(eval_scores.mean()), 4),
                "quality_score_p90": round(float(np.percentile(eval_scores, 90)), 4),
                "estimated_total_tokens": total_tokens,
                "quality_per_token_yield": round(
                    float(eval_scores.mean()) / (total_tokens / 1e6) if total_tokens > 0 else 0, 6
                ),  # 每百万 token 的质量分（质量效率指标）
            })

            print(f"     top-{top_fraction:.0%}: {len(retained_texts):,} 条 | "
                  f"质量均分: {eval_scores.mean():.4f} | "
                  f"Token: {total_tokens:,}")

        df = pd.DataFrame(results).set_index("top_fraction")
        return df

    def find_optimal_threshold(self, df: pd.DataFrame, metric: str = "quality_score_mean") -> float:
        """
        找到指定指标下的最优阈值。

        Args:
            df: run_experiments() 的返回值
            metric: 优化目标（"quality_score_mean" | "quality_per_token_yield"）

        Returns:
            最优保留比例（top_fraction）
        """
        optimal_idx = df[metric].idxmax()
        optimal_quality = df.loc[optimal_idx, metric]
        optimal_tokens = df.loc[optimal_idx, "estimated_total_tokens"]

        print(f"\n  🏆 最优阈值（按 {metric}）:")
        print(f"     top_fraction  : {optimal_idx:.0%}")
        print(f"     {metric}: {optimal_quality:.4f}")
        print(f"     estimated tokens: {optimal_tokens:,}")

        return float(optimal_idx)

    def plot_tradeoff_curve(
        self,
        df: pd.DataFrame,
        save_path: Optional[str] = None,
        generation_label: str = "Gen2",
        show: bool = True,
    ) -> None:
        """
        绘制 quality-quantity trade-off 曲线。
        X 轴：retention_rate，Y 轴：quality_score_mean
        """
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # 左图：quality vs retention_rate
        ax1.plot(df["retention_rate"], df["quality_score_mean"], "bo-", linewidth=2, markersize=8)
        for idx, row in df.iterrows():
            ax1.annotate(
                f"top-{idx:.0%}",
                (row["retention_rate"], row["quality_score_mean"]),
                textcoords="offset points", xytext=(5, 5), fontsize=8
            )
        ax1.set_xlabel("数据保留率", fontsize=12)
        ax1.set_ylabel("Quality Score（评估分类器）", fontsize=12)
        ax1.set_title(f"{generation_label}：Quality-Quantity Trade-off", fontsize=13, fontweight="bold")
        ax1.grid(True, alpha=0.3)
        # 标注最优点
        optimal_idx = df["quality_score_mean"].idxmax()
        optimal_row = df.loc[optimal_idx]
        ax1.axvline(x=optimal_row["retention_rate"], color="red", linestyle="--", alpha=0.5, label=f"最优: top-{optimal_idx:.0%}")
        ax1.legend()

        # 右图：quality vs token 数量（token yield 视角）
        ax2.plot(df["estimated_total_tokens"] / 1e6, df["quality_score_mean"], "go-", linewidth=2, markersize=8)
        ax2.set_xlabel("Token 数量（百万）", fontsize=12)
        ax2.set_ylabel("Quality Score（评估分类器）", fontsize=12)
        ax2.set_title(f"{generation_label}：Quality vs Token Yield", fontsize=13, fontweight="bold")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            from pathlib import Path
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  ✅ Trade-off 曲线已保存: {save_path}")
        if show:
            plt.show()
        plt.close()
