"""
src/gen3/classifier_ensemble.py
第三代核心创新 1：分类器集成（Classifier Ensembling）

Nemotron-CC 的核心发现：
  单一 fastText 分类器都会有覆盖盲区——某些高质量内容被正样本分布所遗漏。
  例如：技术博客可能被"百科风格"分类器低估，
  却被"教育类文本风格"分类器高估。

解决方案：集成多个分类器，取并集（Union）而非交集（Intersection）：
  Union：任一分类器认为高质量 → 判为高质量（扩大覆盖）
  Intersection：所有分类器都认为高质量（更保守，类似第二代）
  Weighted Avg：加权平均分数，更平滑的过渡

效果（Nemotron-CC）：
  集成后高质量数据覆盖面 vs 单一分类器 +28% unique tokens
  同时 MMLU 不降。
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from pathlib import Path


class ClassifierEnsemble:
    """
    多分类器集成：结合 fastText 分类器（多个）+ TF-IDF+LR 分类器。
    三种集成策略：union / intersection / weighted_avg。
    """

    def __init__(
        self,
        strategy: str = "union",
        union_threshold: float = 0.5,
        weighted_avg_threshold: float = 0.5,
    ):
        """
        Args:
            strategy: "union" | "intersection" | "weighted_avg"
            union_threshold: union 策略中每个分类器的单独阈值
            weighted_avg_threshold: weighted_avg 策略的最终阈值
        """
        assert strategy in ("union", "intersection", "weighted_avg")
        self.strategy = strategy
        self.union_threshold = union_threshold
        self.weighted_avg_threshold = weighted_avg_threshold

        # 注册的分类器列表
        self._classifiers: List[Dict] = []  # {"name", "clf", "weight", "threshold"}

    def add_fasttext_classifier(
        self,
        name: str,
        classifier,    # Gen2QualityClassifier 实例
        weight: float = 1.0,
        threshold: Optional[float] = None,
    ) -> None:
        """注册一个 fastText 分类器。"""
        self._classifiers.append({
            "name": name,
            "type": "fasttext",
            "clf": classifier,
            "weight": weight,
            "threshold": threshold or self.union_threshold,
        })
        print(f"  ✅ 注册分类器: {name} (weight={weight})")

    def train_tfidf_lr(
        self,
        name: str,
        positive_texts: List[str],
        negative_texts: List[str],
        model_path: Optional[str] = None,
        weight: float = 1.0,
        max_features: int = 50000,
    ) -> None:
        """
        训练 TF-IDF + LogisticRegression 分类器并注册。
        作为 fastText 的补充（不同特征视角）。
        """
        print(f"  🏋️  训练 TF-IDF+LR 分类器: {name}...")
        X_pos = positive_texts
        X_neg = negative_texts
        X = X_pos + X_neg
        y = [1] * len(X_pos) + [0] * len(X_neg)

        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=3,
        )
        X_vec = vectorizer.fit_transform(X)

        lr = LogisticRegression(C=1.0, max_iter=200, solver="saga", n_jobs=-1)
        lr.fit(X_vec, y)

        print(f"  ✅ TF-IDF+LR 训练完成: {name}")

        if model_path:
            Path(model_path).parent.mkdir(parents=True, exist_ok=True)
            joblib.dump({"vectorizer": vectorizer, "lr": lr}, model_path)

        self._classifiers.append({
            "name": name,
            "type": "sklearn",
            "clf": {"vectorizer": vectorizer, "lr": lr},
            "weight": weight,
            "threshold": self.union_threshold,
        })

    def _score_single(self, clf_entry: Dict, texts: List[str]) -> np.ndarray:
        """用单个分类器给所有文本打分。"""
        if clf_entry["type"] == "fasttext":
            return clf_entry["clf"].score_batch(texts)
        elif clf_entry["type"] == "sklearn":
            vec = clf_entry["clf"]["vectorizer"]
            lr = clf_entry["clf"]["lr"]
            X = vec.transform(texts)
            proba = lr.predict_proba(X)
            return proba[:, 1]  # 正类（高质量）的概率
        else:
            raise ValueError(f"未知分类器类型: {clf_entry['type']}")

    def score_batch(
        self,
        texts: List[str],
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        用所有分类器打分，返回集成后的分数和各分类器的单独分数。

        Returns:
            (ensemble_scores, individual_scores_dict)
            - ensemble_scores: 集成后的最终分数（0-1）
            - individual_scores_dict: 每个分类器的分数
        """
        if not self._classifiers:
            raise RuntimeError("没有注册任何分类器，请先 add_fasttext_classifier() 或 train_tfidf_lr()")

        individual_scores = {}
        weights = []

        for entry in self._classifiers:
            print(f"     [{entry['name']}] 打分中...")
            scores = self._score_single(entry, texts)
            individual_scores[entry["name"]] = scores
            weights.append(entry["weight"])

        all_scores = np.stack(list(individual_scores.values()), axis=1)  # shape: (n_texts, n_classifiers)
        weight_arr = np.array(weights)

        if self.strategy == "union":
            # 每个分类器的阈值判断，取 OR
            thresholds = np.array([e["threshold"] for e in self._classifiers])
            is_high = all_scores >= thresholds  # shape: (n_texts, n_classifiers)
            ensemble_scores = is_high.any(axis=1).astype(float)
            # 也保留连续分数（用最高的那个分类器的分数）
            ensemble_scores_continuous = all_scores.max(axis=1)

        elif self.strategy == "intersection":
            thresholds = np.array([e["threshold"] for e in self._classifiers])
            is_high = all_scores >= thresholds
            ensemble_scores = is_high.all(axis=1).astype(float)
            ensemble_scores_continuous = all_scores.min(axis=1)

        else:  # weighted_avg
            total_weight = weight_arr.sum()
            ensemble_scores_continuous = (all_scores * weight_arr).sum(axis=1) / total_weight
            ensemble_scores = ensemble_scores_continuous

        return ensemble_scores_continuous, individual_scores

    def compare_coverage(
        self,
        texts: List[str],
        threshold: float = 0.5,
    ) -> Dict:
        """
        比较各分类器和集成策略的覆盖率差异（核心分析，用于 Notebook）。

        Returns:
            dict，包含各分类器和集成的"高质量文档集合"的交叉统计
        """
        ensemble_scores, individual_scores = self.score_batch(texts)
        n = len(texts)

        results = {
            "total_docs": n,
            "strategy": self.strategy,
            "individual_coverage": {},
            "ensemble_coverage": 0,
            "unique_by_ensemble": 0,
        }

        individual_high_sets = {}
        for name, scores in individual_scores.items():
            high_mask = scores >= threshold
            individual_high_sets[name] = set(np.where(high_mask)[0])
            results["individual_coverage"][name] = {
                "count": int(high_mask.sum()),
                "rate": round(float(high_mask.mean()), 4),
            }

        # 集成的覆盖
        ensemble_high = set(np.where(ensemble_scores >= threshold)[0])
        results["ensemble_coverage"] = {
            "count": len(ensemble_high),
            "rate": round(len(ensemble_high) / n, 4),
        }

        # 集成额外覆盖（union 策略相比任一单分类器多覆盖的）
        union_all_individual = set().union(*individual_high_sets.values())
        results["union_vs_best_single"] = {
            "union_count": len(union_all_individual),
            "best_single_count": max(len(s) for s in individual_high_sets.values()),
            "gain": len(union_all_individual) - max(len(s) for s in individual_high_sets.values()),
        }

        return results
