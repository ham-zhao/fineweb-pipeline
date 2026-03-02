"""
src/evaluation/toxicity_scorer.py
毒性打分器（基于 detoxify）

内容安全分类的核心认知：
──────────────────────────────────────────────────────────
1. 内容安全是多级体系，不是"有毒/无毒"二分类
   维度：toxicity（总体）/ severe_toxicity / obscene / threat / insult / identity_attack
   严重程度：轻微 / 中度 / 严重
   不同类型的处理策略不同：完全过滤 vs 标记保留 vs 改写脱敏

2. 安全过滤的 Precision-Recall 困境
   阈值太严 → 大量正常内容误杀（如医学文本被当"色情"）→ 数据多样性受损
   阈值太松 → 有害内容泄漏到训练集 → 模型生成不安全内容
   没有完美阈值，只有根据业务场景的 trade-off

3. 与 TikTok 内容安全审核体系的对应关系
   预训练数据过滤 ≈ 机器初筛（高阈值，宁可多杀不能漏）
   实际产品内容审核 = 多级人机协同（机器粗筛 → 人工复核 → 申诉机制）
──────────────────────────────────────────────────────────
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm


class ToxicityScorer:
    """
    基于 detoxify 的多维度毒性打分器。
    支持批量打分和多维度分析。
    """

    DIMENSIONS = [
        "toxicity",
        "severe_toxicity",
        "obscene",
        "threat",
        "insult",
        "identity_attack",
    ]

    def __init__(self, model_name: str = "original", device: str = "cpu"):
        """
        Args:
            model_name: detoxify 模型名（"original" | "unbiased" | "multilingual"）
            device: "cpu" | "cuda" | "mps"（detoxify 目前 mps 支持有限，建议 cpu）
        """
        self.model_name = model_name
        self.device = device
        self._model = None

    def _load_model(self):
        """懒加载 detoxify 模型。"""
        if self._model is not None:
            return
        from detoxify import Detoxify
        print(f"  📦 加载 Detoxify 模型（{self.model_name}）...")
        self._model = Detoxify(self.model_name, device=self.device)
        print(f"  ✅ Detoxify 已加载")

    def score(self, text: str) -> Dict[str, float]:
        """
        对单条文本做多维度毒性打分。

        Returns:
            dict，key 为毒性维度，value 为 0-1 分数（越高越毒）
        """
        self._load_model()
        results = self._model.predict(text)
        return {k: float(v) for k, v in results.items() if k in self.DIMENSIONS}

    def score_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        批量毒性打分。

        Returns:
            dict，key 为毒性维度，value 为 np.ndarray（shape=len(texts)）
        """
        self._load_model()

        all_scores = {dim: [] for dim in self.DIMENSIONS}

        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="  ☣️  毒性打分", unit="batch")

        for i in iterator:
            batch = texts[i:i + batch_size]
            # 过滤空文本
            batch = [t if t.strip() else " " for t in batch]
            results = self._model.predict(batch)
            for dim in self.DIMENSIONS:
                vals = results.get(dim, [0.0] * len(batch))
                if hasattr(vals, "tolist"):
                    vals = vals.tolist()
                elif not isinstance(vals, list):
                    vals = [float(vals)] * len(batch)
                all_scores[dim].extend(vals)

        return {dim: np.array(scores) for dim, scores in all_scores.items()}

    def compute_statistics(self, scores: Dict[str, np.ndarray]) -> Dict:
        """
        计算各维度毒性统计量。

        Returns:
            嵌套 dict，key 为维度，value 为统计指标
        """
        stats = {}
        for dim, arr in scores.items():
            stats[dim] = {
                "mean": float(np.mean(arr)),
                "p50": float(np.percentile(arr, 50)),
                "p90": float(np.percentile(arr, 90)),
                "p95": float(np.percentile(arr, 95)),
                "p99": float(np.percentile(arr, 99)),
                "toxic_rate_50": float(np.mean(arr > 0.5)),   # >0.5 为高毒性
                "toxic_rate_80": float(np.mean(arr > 0.8)),   # >0.8 为严重毒性
            }
        return stats

    def filter_toxic(
        self,
        texts: List[str],
        scores: Optional[Dict[str, np.ndarray]] = None,
        toxicity_threshold: float = 0.85,
        severe_toxicity_threshold: float = 0.5,
    ) -> Dict:
        """
        根据阈值识别高毒性文档（用于评估，不直接过滤数据）。

        Args:
            texts: 文本列表
            scores: 已有的打分结果（避免重复计算）
            toxicity_threshold: 总体毒性阈值
            severe_toxicity_threshold: 严重毒性阈值

        Returns:
            dict，包含：
              - toxic_mask: bool 数组（True = 有毒）
              - toxic_count: 有毒文档数
              - toxic_rate: 有毒比例
              - sample_toxic_texts: 有毒文本样本（脱敏后展示）
        """
        if scores is None:
            scores = self.score_batch(texts)

        toxic_mask = (
            (scores.get("toxicity", np.zeros(len(texts))) > toxicity_threshold)
            | (scores.get("severe_toxicity", np.zeros(len(texts))) > severe_toxicity_threshold)
        )

        toxic_indices = np.where(toxic_mask)[0]
        # 脱敏展示：截断文本，隐藏部分字符
        sample_toxic = []
        for idx in toxic_indices[:5]:
            text_preview = texts[idx][:100].replace("\n", " ")
            sample_toxic.append({
                "text_preview": text_preview[:50] + "..." if len(text_preview) > 50 else text_preview,
                "toxicity": float(scores["toxicity"][idx]),
                "severe_toxicity": float(scores.get("severe_toxicity", np.zeros(len(texts)))[idx]),
            })

        return {
            "toxic_mask": toxic_mask,
            "toxic_count": int(np.sum(toxic_mask)),
            "toxic_rate": float(np.mean(toxic_mask)),
            "sample_toxic_texts": sample_toxic,
            "thresholds_used": {
                "toxicity": toxicity_threshold,
                "severe_toxicity": severe_toxicity_threshold,
            },
        }

    def threshold_analysis(
        self,
        scores: Dict[str, np.ndarray],
        thresholds: Optional[List[float]] = None,
    ) -> Dict:
        """
        不同阈值下的过滤率分析（用于 Notebook 中可视化 Precision-Recall 困境）。

        Returns:
            dict，key 为阈值，value 为过滤率
        """
        if thresholds is None:
            thresholds = [0.3, 0.5, 0.7, 0.85, 0.9, 0.95]

        toxicity_scores = scores.get("toxicity", np.array([]))
        results = {}
        for threshold in thresholds:
            filter_rate = float(np.mean(toxicity_scores > threshold))
            results[threshold] = {
                "filter_rate": filter_rate,
                "retained_rate": 1 - filter_rate,
                "toxic_count": int(np.sum(toxicity_scores > threshold)),
            }
        return results
