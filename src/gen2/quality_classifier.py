"""
src/gen2/quality_classifier.py
第二代 Pipeline 专用 fastText 质量分类器

⚠️  与评估分类器（src/evaluation/quality_classifier.py）的区别：
─────────────────────────────────────────────────────────────────
此分类器用于 Pipeline 过滤（决定哪些文档保留）。
评估分类器用于评估各代 Pipeline 的输出质量（只打分，不过滤）。
两者必须独立训练，否则产生循环偏差（Circular Bias）。

差异对照表：
  项目          Pipeline 分类器（本文件）     评估分类器
  正样本来源    OpenHermes / ELI5            Wikipedia 摘要
  负样本来源    原始 Common Crawl            原始 Common Crawl
  dim           64                            32
  wordNgrams    2（unigram + bigram）         1（只用 unigram）
  lr            0.1                           0.05

DCLM 核心发现（NeurIPS 2024）：
  fastText 二分类器（dim=64, wordNgrams=2）+ top-10% 阈值
  在 MMLU 上超过了所有 heuristic 组合，是第二代的核心突破。
─────────────────────────────────────────────────────────────────
"""

import os
import json
import tempfile
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple


class Gen2QualityClassifier:
    """
    第二代 Pipeline 专用 fastText 质量分类器。
    正样本：OpenHermes / ELI5 Stack Exchange（更接近人类写作精华）
    负样本：原始 Common Crawl
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.model_path = model_path
        if model_path and Path(model_path).exists():
            self._load(model_path)

    def _load(self, model_path: str) -> None:
        import fasttext
        self.model = fasttext.load_model(model_path)
        print(f"  ✅ Gen2 分类器已加载: {model_path}")

    def train(
        self,
        positive_texts: List[str],
        negative_texts: List[str],
        output_path: str,
        # DCLM 推荐超参
        dim: int = 64,
        wordNgrams: int = 2,
        lr: float = 0.1,
        epoch: int = 5,
        minCount: int = 5,
    ) -> Dict:
        """
        训练 Pipeline 专用分类器。

        Args:
            positive_texts: 高质量文本（OpenHermes / ELI5 高赞回答）
            negative_texts: 低质量文本（原始 CC 采样）
            output_path: 模型保存路径

        Returns:
            训练统计字典
        """
        import fasttext

        print(f"  🏋️  训练 Gen2 Pipeline 分类器（DCLM 风格）...")
        print(f"     正样本: {len(positive_texts):,} | 负样本: {len(negative_texts):,}")
        print(f"     超参: dim={dim}, wordNgrams={wordNgrams}, lr={lr}")

        # 写训练文件（fastText 格式）
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as tf:
            train_path = tf.name
            for text in positive_texts:
                clean = text.replace("\n", " ").strip()
                if clean:
                    tf.write(f"__label__high {clean}\n")
            for text in negative_texts:
                clean = text.replace("\n", " ").strip()
                if clean:
                    tf.write(f"__label__low {clean}\n")

        try:
            self.model = fasttext.train_supervised(
                input=train_path,
                dim=dim,
                wordNgrams=wordNgrams,
                lr=lr,
                epoch=epoch,
                minCount=minCount,
                loss="softmax",
                verbose=0,
            )
        finally:
            os.unlink(train_path)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(output_path)
        self.model_path = output_path
        print(f"  ✅ 分类器已保存: {output_path}")

        return {
            "n_positive": len(positive_texts),
            "n_negative": len(negative_texts),
            "dim": dim,
            "wordNgrams": wordNgrams,
            "model_path": output_path,
        }

    def score(self, text: str) -> float:
        """返回文档属于高质量的概率（0-1）。"""
        if self.model is None:
            raise RuntimeError("分类器未加载")
        clean = text.replace("\n", " ").strip() or " "
        labels, probs = self.model.predict(clean)
        for label, prob in zip(labels, probs):
            if label == "__label__high":
                return float(prob)
        return 0.0

    def score_batch(self, texts: List[str], batch_size: int = 512) -> np.ndarray:
        """批量打分。"""
        if self.model is None:
            raise RuntimeError("分类器未加载")
        scores = np.zeros(len(texts))
        for i in range(0, len(texts), batch_size):
            batch = [t.replace("\n", " ").strip() or " " for t in texts[i:i+batch_size]]
            results = self.model.predict(batch)
            for j, (labels, probs) in enumerate(zip(results[0], results[1])):
                for label, prob in zip(labels, probs):
                    if label == "__label__high":
                        scores[i + j] = float(prob)
                        break
        return scores

    def get_threshold(self, scores: np.ndarray, top_fraction: float = 0.10) -> float:
        """
        计算 top-X% 对应的分数阈值。

        Args:
            scores: 全量数据的分数数组
            top_fraction: 保留比例（0.10 = top 10%）

        Returns:
            float，分数阈值（高于此值的文档被保留）
        """
        threshold = np.percentile(scores, (1 - top_fraction) * 100)
        print(f"  Top-{top_fraction:.0%} 阈值: {threshold:.4f}")
        return float(threshold)

    def evaluate_performance(
        self,
        positive_texts: List[str],
        negative_texts: List[str],
    ) -> Dict:
        """评估分类器在正负样本上的性能。"""
        from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

        pos_scores = self.score_batch(positive_texts)
        neg_scores = self.score_batch(negative_texts)
        y_true = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
        y_scores = np.concatenate([pos_scores, neg_scores])

        roc_auc = roc_auc_score(y_true, y_scores)
        prec, rec, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(rec, prec)

        # Precision @ top-10%
        thresh_10 = np.percentile(y_scores, 90)
        pred_high = y_scores >= thresh_10
        precision_10 = np.sum(y_true[pred_high]) / np.sum(pred_high) if np.sum(pred_high) > 0 else 0

        results = {
            "roc_auc": round(float(roc_auc), 4),
            "pr_auc": round(float(pr_auc), 4),
            "precision_at_top10pct": round(float(precision_10), 4),
            "pos_mean_score": round(float(pos_scores.mean()), 4),
            "neg_mean_score": round(float(neg_scores.mean()), 4),
        }
        print(f"  📊 Gen2 分类器性能: ROC-AUC={results['roc_auc']:.4f} | "
              f"PR-AUC={results['pr_auc']:.4f} | "
              f"Prec@top10%={results['precision_at_top10pct']:.4f}")
        return results
