"""
src/gen2/pipeline.py
第二代 Pipeline：Model-based Filtering

核心流程：
  1. （可选）先用第一代 Heuristic 做粗过滤（DCLM 的做法）
  2. 用 fastText 分类器对所有文档打分
  3. 只保留 top-10% 的文档（score >= threshold）

方法论意义（相比第一代的核心突破）：
  - 第一代无法区分"平庸内容"和"高质量内容"（都能通过规则）
  - 第二代通过语义特征（N-gram 分布）捕捉"像高质量文本的文档"
  - DCLM 论文验证：这个简单的分类器效果超过所有 heuristic 组合

第二代的核心局限：
  - 90% 数据被丢弃（激进过滤）
  - 不适合长 token horizon 训练（15T tokens 需要更多数据）
  → 这是第三代要解决的问题
"""

import json
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional

from src.utils.config_loader import load_run_config, load_pipeline_config


class Gen2Pipeline:
    """
    第二代 Model-based Filtering Pipeline。
    """

    def __init__(
        self,
        run_config: Dict,
        pipeline_config: Dict,
        classifier,          # Gen2QualityClassifier 实例
        stage_tracker=None,
        gen1_pipeline=None,  # 可选：先用第一代做粗过滤
    ):
        self.run_cfg = run_config
        self.pipe_cfg = pipeline_config
        self.classifier = classifier
        self.tracker = stage_tracker
        self.gen1 = gen1_pipeline

        # 生产阈值（top-10% 对应的分数阈值，在 run() 中动态计算）
        self.production_threshold: Optional[float] = None
        self.all_scores: Optional[np.ndarray] = None

    def run(
        self,
        docs: List[Dict],
        top_fraction: float = 0.10,
        use_heuristic_preprocessing: bool = True,
    ) -> Dict:
        """
        执行第二代 Pipeline。

        Args:
            docs: 输入文档（来自 WARC 或 Gen1 输出）
            top_fraction: 保留比例（0.10 = top 10%，DCLM 最优值）
            use_heuristic_preprocessing: 是否先跑第一代 heuristic

        Returns:
            dict，包含 filtered_docs, all_scores, threshold, stats
        """
        print(f"\n{'='*60}")
        print(f"  第二代 Pipeline 启动 | 输入: {len(docs):,} 条")
        print(f"  top_fraction: {top_fraction:.0%} (DCLM 最优)")
        print(f"{'='*60}")
        start = time.time()

        if self.tracker:
            self.tracker.record("gen2_input", [d["text"] for d in docs],
                                urls=[d.get("url", "") for d in docs])

        # ── Step 1: Heuristic 预处理（可选）────────────────────
        if use_heuristic_preprocessing and self.gen1:
            print(f"\n  Step 1: 第一代 Heuristic 预处理...")
            docs = self.gen1.run(docs)
            print(f"  Heuristic 后剩余: {len(docs):,} 条")
        else:
            print(f"\n  Step 1: 跳过 Heuristic 预处理（直接 model-based）")

        # ── Step 2: 批量打分 ─────────────────────────────────
        print(f"\n  Step 2: fastText 分类器打分...")
        texts = [d["text"] for d in docs]
        scores = self.classifier.score_batch(texts)
        self.all_scores = scores

        # 打分统计
        score_stats = {
            "mean": float(scores.mean()),
            "p50": float(np.percentile(scores, 50)),
            "p90": float(np.percentile(scores, 90)),
        }
        print(f"     分数分布: mean={score_stats['mean']:.4f} | "
              f"P50={score_stats['p50']:.4f} | P90={score_stats['p90']:.4f}")

        # ── Step 3: 阈值过滤（保留 top-X%）─────────────────────
        print(f"\n  Step 3: 阈值过滤（保留 top-{top_fraction:.0%}）...")
        threshold = float(np.percentile(scores, (1 - top_fraction) * 100))
        self.production_threshold = threshold
        print(f"     计算阈值: {threshold:.4f}")

        filtered_docs = []
        for doc, score in zip(docs, scores):
            doc["_gen2_score"] = round(float(score), 4)
            if score >= threshold:
                filtered_docs.append(doc)

        # ── 完成 ──────────────────────────────────────────────
        elapsed = time.time() - start
        retained_rate = len(filtered_docs) / len(docs) if docs else 0

        print(f"\n{'='*60}")
        print(f"  ✅ 第二代 Pipeline 完成！")
        print(f"  输入: {len(docs):,} 条 → 输出: {len(filtered_docs):,} 条")
        print(f"  实际保留率: {retained_rate:.1%} | 耗时: {elapsed:.1f}s")
        print(f"{'='*60}")

        if self.tracker:
            self.tracker.record(
                "gen2_output", [d["text"] for d in filtered_docs],
                urls=[d.get("url", "") for d in filtered_docs],
                quality_scores=scores[scores >= threshold],
            )

        return {
            "filtered_docs": filtered_docs,
            "all_scores": scores,
            "threshold": threshold,
            "stats": {
                "input_count": len(docs),
                "output_count": len(filtered_docs),
                "retention_rate": retained_rate,
                "top_fraction": top_fraction,
                "threshold": threshold,
                "score_stats": score_stats,
                "elapsed_seconds": elapsed,
            },
        }

    def save_with_scores(self, result: Dict, output_path: Path) -> None:
        """保存过滤后的文档（含分数信息）。"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for doc in result["filtered_docs"]:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
        print(f"  💾 Gen2 输出: {len(result['filtered_docs']):,} 条 → {output_path}")
