"""
src/evaluation/quality_classifier.py
评估专用质量分类器（与 pipeline 分类器完全独立！）

⚠️  重要：循环偏差（Circular Bias）防范
─────────────────────────────────────────────
此分类器仅用于评估（衡量每一代 pipeline 产出的数据质量），
绝对不参与任何 pipeline 的过滤决策。

为什么必须独立？
  如果用同一个模型既过滤又评估，等于"自己出题自己考试"。
  第二代用 fastText 过滤后，再用同一个模型打分，
  分数必然很高——但这只说明分类器自洽，不说明数据真的好。

独立性措施（与 pipeline 分类器的差异）：
  1. 正样本数据集独立：使用 wikipedia_abstracts_eval.jsonl
     （pipeline 分类器使用 wikipedia_abstracts.jsonl，两者不重叠）
  2. fastText 超参不同：dim=32（pipeline 用 64），wordNgrams=3（pipeline 用 2）
  3. 独立训练（由 scripts/run_gen2.py 中的 train_eval_classifier() 完成），
     不参与任何过滤决策
─────────────────────────────────────────────
"""

import json
import os
import tempfile
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from tqdm import tqdm


class EvalQualityClassifier:
    """
    评估专用 fastText 二分类器。

    正样本：Wikipedia 摘要（高质量知识密度文本）
    负样本：原始 Common Crawl（未经过滤）
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Args:
            model_path: 预训练模型路径（.bin）。若为 None，需要先调用 train()。
        """
        self.model = None
        self.model_path = model_path
        self._max_words = None  # 训练时自动计算
        self._truncate_side = ""  # "positive" 或 "negative"
        if model_path and Path(model_path).exists():
            self._load(model_path)

    def _load(self, model_path: str) -> None:
        """加载已训练的 fastText 模型。"""
        import fasttext
        self.model = fasttext.load_model(model_path)
        print(f"  ✅ 评估分类器已加载: {model_path}")

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
        自适应计算截断长度：只截断较长方，保留正样本完整内容。

        规则：
          - 长度比 > 3x → 截断到较短方的 p90
          - 长度比 < 2x → 不截断
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
            max_words = max(max_words, 100)
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
        # fastText 超参（与 pipeline 分类器刻意不同）
        dim: int = 32,
        wordNgrams: int = 3,
        lr: float = 0.05,
        epoch: int = 3,
        minCount: int = 5,
    ) -> Dict:
        """
        训练评估专用分类器。

        Args:
            positive_texts: 高质量文本列表（Wikipedia 摘要）
            negative_texts: 低质量文本列表（原始 CC）
            output_path: 模型保存路径（.bin）
            dim, wordNgrams, lr, epoch, minCount: fastText 参数

        Returns:
            训练统计信息（样本数、AUC 等）
        """
        import fasttext

        print(f"  🏋️  训练评估分类器...")
        print(f"     正样本: {len(positive_texts):,} 条")
        print(f"     负样本: {len(negative_texts):,} 条")
        print(f"     超参: dim={dim}, wordNgrams={wordNgrams}, lr={lr}, epoch={epoch}")

        # 自适应计算截断长度（只截断较长方）
        self._max_words, self._truncate_side = self._compute_max_words(positive_texts, negative_texts)

        # 写入临时训练文件（fastText 格式）
        # 训练时：只截断较长方，正样本（参考文本）保留完整内容
        truncate_pos = self._max_words and self._truncate_side == "positive"
        truncate_neg = self._max_words and self._truncate_side == "negative"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as tf:
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

        # 保存模型
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(output_path)
        self.model_path = output_path
        print(f"  ✅ 评估分类器已保存: {output_path}")

        # 训练后 sanity check
        self._sanity_check(positive_texts[:200], negative_texts[:200])

        n_pos = len(positive_texts)
        n_neg = len(negative_texts)
        return {
            "n_positive": n_pos,
            "n_negative": n_neg,
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

    def _prepare_text(self, text: str) -> str:
        """清洗 + 自适应截断（与训练保持一致）。"""
        clean = text.replace("\n", " ").strip() or " "
        if self._max_words:
            clean = self._truncate(clean, self._max_words)
        return clean

    def score(self, text: str) -> float:
        """
        给单条文本打质量分（0-1，越高越好）。

        Returns:
            float，表示文本属于"高质量"的概率
        """
        if self.model is None:
            raise RuntimeError("分类器未加载，请先调用 train() 或传入 model_path")

        clean = self._prepare_text(text)
        if not clean.strip():
            return 0.0

        labels, probs = self.model.predict(clean, k=2)
        for label, prob in zip(labels, probs):
            if label == "__label__high":
                return float(prob)
        return 0.0

    def score_batch(self, texts: List[str], batch_size: int = 512) -> np.ndarray:
        """
        批量打分（比逐条快 5-10 倍）。

        Returns:
            np.ndarray，shape=(len(texts),)，每个值为高质量概率
        """
        if self.model is None:
            raise RuntimeError("分类器未加载")

        scores = np.zeros(len(texts))
        for i in range(0, len(texts), batch_size):
            batch = [self._prepare_text(t) for t in texts[i:i + batch_size]]
            results = self.model.predict(batch, k=2)
            for j, (labels, probs) in enumerate(zip(results[0], results[1])):
                for label, prob in zip(labels, probs):
                    if label == "__label__high":
                        scores[i + j] = float(prob)
                        break

        return scores

    def evaluate(
        self,
        positive_texts: List[str],
        negative_texts: List[str],
    ) -> Dict:
        """
        评估分类器性能：计算 AUC-ROC、Precision@Threshold。

        Returns:
            dict，包含 auc, precision_at_50, precision_at_80 等指标
        """
        from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

        pos_scores = self.score_batch(positive_texts)
        neg_scores = self.score_batch(negative_texts)

        y_true = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
        y_scores = np.concatenate([pos_scores, neg_scores])

        roc_auc = roc_auc_score(y_true, y_scores)
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall, precision)

        # Precision @ top 10%（模拟 DCLM 的 top-10% 过滤）
        threshold_10pct = np.percentile(y_scores, 90)
        predicted_high = y_scores >= threshold_10pct
        precision_at_10pct = np.sum(y_true[predicted_high]) / np.sum(predicted_high) if np.sum(predicted_high) > 0 else 0

        results = {
            "roc_auc": round(float(roc_auc), 4),
            "pr_auc": round(float(pr_auc), 4),
            "precision_at_top10pct": round(float(precision_at_10pct), 4),
            "mean_positive_score": round(float(pos_scores.mean()), 4),
            "mean_negative_score": round(float(neg_scores.mean()), 4),
            "score_separation": round(float(pos_scores.mean() - neg_scores.mean()), 4),
        }

        print(f"  📊 评估分类器性能:")
        print(f"     ROC-AUC           : {results['roc_auc']:.4f}")
        print(f"     PR-AUC            : {results['pr_auc']:.4f}")
        print(f"     Precision@top10%  : {results['precision_at_top10pct']:.4f}")
        print(f"     正样本平均分       : {results['mean_positive_score']:.4f}")
        print(f"     负样本平均分       : {results['mean_negative_score']:.4f}")
        print(f"     分数分离度         : {results['score_separation']:.4f}")

        return results
