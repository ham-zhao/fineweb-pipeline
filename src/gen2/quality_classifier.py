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
  正样本来源    Wikipedia 摘要               Wikipedia 摘要（独立训练集）
  负样本来源    原始 Common Crawl            原始 Common Crawl
  dim           64                            32
  wordNgrams    2（unigram + bigram）         3（trigram，保留短语级特征）
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
    正样本：Wikipedia 摘要（高质量百科文本）
    负样本：原始 Common Crawl
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.model_path = model_path
        self._max_words = None  # 训练时自动计算，打分时复用
        self._truncate_side = ""  # "positive" 或 "negative"，记录哪边被截断
        if model_path and Path(model_path).exists():
            self._load(model_path)

    def _load(self, model_path: str) -> None:
        import fasttext
        self.model = fasttext.load_model(model_path)
        print(f"  ✅ Gen2 分类器已加载: {model_path}")

    @staticmethod
    def _truncate(text: str, max_words: int = 200) -> str:
        """截断到前 max_words 个词，确保正/负样本长度同分布。"""
        words = text.split()
        if len(words) > max_words:
            return " ".join(words[:max_words])
        return text

    @staticmethod
    def _compute_max_words(positive_texts: List[str], negative_texts: List[str]) -> Tuple[Optional[int], str]:
        """
        自适应计算截断长度：只在正/负样本长度差异大时截断较长方。

        原则：
          - 正样本（参考文本）不截断——完整保留高质量内容
          - 只截断较长方（通常是负样本），消除长度作为伪特征
          - 推理时统一截断（保持与训练一致的 n-gram 特征规模）

        规则：
          - 长度比 > 3x → 截断到较短方的 p90
          - 长度比 < 2x → 不截断（已接近同分布）
          - 中间 → 截断到较短方的 p95

        Returns:
            (max_words, longer_side) 其中 longer_side 为 "positive" 或 "negative"
        """
        pos_lens = np.array([len(t.split()) for t in positive_texts])
        neg_lens = np.array([len(t.split()) for t in negative_texts])
        pos_mean, neg_mean = pos_lens.mean(), neg_lens.mean()
        ratio = max(pos_mean, neg_mean) / max(min(pos_mean, neg_mean), 1)
        shorter_lens = pos_lens if pos_mean < neg_mean else neg_lens
        longer_side = "negative" if neg_mean > pos_mean else "positive"

        if ratio > 3:
            max_words = int(np.percentile(shorter_lens, 90))
            max_words = max(max_words, 100)  # 至少 100 词
            print(f"     长度比 {ratio:.1f}x > 3x → 只截断{longer_side}到 {max_words} 词（较短方 p90）")
            return max_words, longer_side
        elif ratio > 2:
            max_words = int(np.percentile(shorter_lens, 95))
            max_words = max(max_words, 100)
            print(f"     长度比 {ratio:.1f}x → 只截断{longer_side}到 {max_words} 词（较短方 p95）")
            return max_words, longer_side
        else:
            print(f"     长度比 {ratio:.1f}x < 2x → 不截断（已近似同分布）")
            return None, ""

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
            positive_texts: 高质量文本（Wikipedia 摘要）
            negative_texts: 低质量文本（原始 CC 采样）
            output_path: 模型保存路径

        Returns:
            训练统计字典
        """
        import fasttext

        print(f"  🏋️  训练 Gen2 Pipeline 分类器（DCLM 风格）...")
        print(f"     正样本: {len(positive_texts):,} | 负样本: {len(negative_texts):,}")
        print(f"     超参: dim={dim}, wordNgrams={wordNgrams}, lr={lr}")

        # 自适应计算截断长度（只截断较长方）
        self._max_words, self._truncate_side = self._compute_max_words(positive_texts, negative_texts)

        # 写训练文件（fastText 格式）
        # 训练时：只截断较长方，正样本（参考文本）保留完整内容
        truncate_pos = self._max_words and self._truncate_side == "positive"
        truncate_neg = self._max_words and self._truncate_side == "negative"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as tf:
            train_path = tf.name
            for text in positive_texts:
                clean = text.replace("\n", " ").strip()
                if truncate_pos:
                    clean = self._truncate(clean, self._max_words)
                if clean:
                    tf.write(f"__label__high {clean}\n")
            for text in negative_texts:
                clean = text.replace("\n", " ").strip()
                if truncate_neg:
                    clean = self._truncate(clean, self._max_words)
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

        # 训练后 sanity check：验证正/负样本分离度
        self._sanity_check(positive_texts[:200], negative_texts[:200])

        return {
            "n_positive": len(positive_texts),
            "n_negative": len(negative_texts),
            "dim": dim,
            "wordNgrams": wordNgrams,
            "model_path": output_path,
        }

    def _sanity_check(self, positive_texts: List[str], negative_texts: List[str]) -> None:
        """训练后立即检验分离度，确保分类器学到了有意义的信号。"""
        pos_scores = self.score_batch(positive_texts)
        neg_scores = self.score_batch(negative_texts)
        separation = float(pos_scores.mean() - neg_scores.mean())
        print(f"  📊 Sanity check: pos_mean={pos_scores.mean():.4f}, neg_mean={neg_scores.mean():.4f}, separation={separation:.4f}")
        if separation < 0.1:
            print(f"  ⚠️  WARNING: 分离度 {separation:.4f} < 0.1，分类器可能无法有效区分质量！")
            print(f"     请检查：正/负样本是否来自不同质量分布？训练数据是否正确？")

    def _prepare_text(self, text: str) -> str:
        """清洗 + 自适应截断（与训练保持一致）。"""
        clean = text.replace("\n", " ").strip() or " "
        if self._max_words:
            clean = self._truncate(clean, self._max_words)
        return clean

    def score(self, text: str) -> float:
        """返回文档属于高质量的概率（0-1）。"""
        if self.model is None:
            raise RuntimeError("分类器未加载")
        clean = self._prepare_text(text)
        labels, probs = self.model.predict(clean, k=2)
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
            batch = [self._prepare_text(t) for t in texts[i:i+batch_size]]
            results = self.model.predict(batch, k=2)
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
