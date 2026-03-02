"""
src/gen1/filters/toxicity_filter.py
内容安全过滤器（第一代 Pipeline 新增）

方法论定位：
  毒性过滤不是"质量过滤"，而是"内容安全过滤"。
  即使语言流畅、信息量高的文档，也可能含有有害内容。

  与 TikTok 内容安全体系的对应：
  - 预训练数据过滤 = 机器初筛（高阈值，减少有害内容进入训练集）
  - 目标是降低模型学习到有害内容的概率，而不是完全消除
  - 完全消除有害内容会过度削减数据量，且无法通过统计过滤实现 100%

  阈值选择的 Precision-Recall 困境：
  - 低阈值（如 0.5）：过滤更多，但误杀率高（医学文本/讨论类内容被误杀）
  - 高阈值（如 0.9）：漏杀率高，有害内容可能进入训练集
  - FineWeb 使用 0.85 作为默认阈值（经验值）

注意：此模块在 pipeline 中用于过滤；
      评估时的毒性打分使用 src/evaluation/toxicity_scorer.py（独立模块）。
"""

from typing import List, Tuple, Dict, Optional
import numpy as np


class ToxicityFilter:
    """
    基于 detoxify 的内容安全过滤器。
    在第一代 Pipeline 中作为最后一道关卡，过滤高毒性文档。
    """

    def __init__(
        self,
        toxicity_threshold: float = 0.85,
        severe_toxicity_threshold: float = 0.50,
        action: str = "filter",     # "filter" | "flag"
        model_name: str = "original",
        device: str = "cpu",
        batch_size: int = 32,
    ):
        """
        Args:
            toxicity_threshold: 总体毒性分数阈值（超过即过滤）
            severe_toxicity_threshold: 严重毒性阈值（更低，因为严重毒性容忍度更低）
            action: "filter" 直接过滤，"flag" 仅标记不过滤（用于研究/统计）
            model_name: detoxify 模型（"original" | "unbiased" | "multilingual"）
            device: 计算设备（MPS 对 detoxify 支持有限，建议 cpu）
            batch_size: 批处理大小
        """
        self.toxicity_threshold = toxicity_threshold
        self.severe_toxicity_threshold = severe_toxicity_threshold
        self.action = action
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return
        from detoxify import Detoxify
        print(f"  📦 加载 Detoxify({self.model_name})...")
        self._model = Detoxify(self.model_name, device=self.device)

    def _is_toxic(self, toxicity_score: float, severe_score: float) -> bool:
        return (toxicity_score > self.toxicity_threshold or
                severe_score > self.severe_toxicity_threshold)

    def score_batch_raw(self, texts: List[str]) -> Dict[str, List[float]]:
        """返回原始的多维度毒性分数。"""
        self._load_model()
        clean = [t if t.strip() else " " for t in texts]
        results = self._model.predict(clean)
        return {k: list(v) if hasattr(v, 'tolist') else [float(v)] * len(clean)
                for k, v in results.items()}

    def filter(self, text: str) -> Tuple[bool, float, str]:
        """
        判断单条文档是否应被过滤。

        Returns:
            (should_filter, toxicity_score, reason)
        """
        self._load_model()
        clean = text if text.strip() else " "
        result = self._model.predict(clean)

        tox = float(result.get("toxicity", 0))
        severe = float(result.get("severe_toxicity", 0))

        if self._is_toxic(tox, severe):
            if self.action == "filter":
                return True, tox, f"toxicity:{tox:.3f}_severe:{severe:.3f}"
            else:  # flag
                return False, tox, f"flagged:toxicity:{tox:.3f}"

        return False, tox, ""

    def filter_batch(
        self, texts: List[str]
    ) -> Tuple[List[bool], List[float], List[str]]:
        """
        批量过滤。

        Returns:
            (filter_masks, toxicity_scores, reasons)
        """
        self._load_model()
        masks, scores, reasons = [], [], []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            clean_batch = [t if t.strip() else " " for t in batch]
            result = self._model.predict(clean_batch)

            tox_scores = result.get("toxicity", [0.0] * len(batch))
            severe_scores = result.get("severe_toxicity", [0.0] * len(batch))

            for tox, severe in zip(tox_scores, severe_scores):
                tox, severe = float(tox), float(severe)
                if self._is_toxic(tox, severe):
                    masks.append(self.action == "filter")
                    reasons.append(f"toxicity:{tox:.3f}")
                else:
                    masks.append(False)
                    reasons.append("")
                scores.append(tox)

        return masks, scores, reasons

    def get_stats(self, texts: List[str]) -> Dict:
        """毒性过滤统计（用于 Notebook 报告）。"""
        masks, scores, reasons = self.filter_batch(texts)
        scores_arr = np.array(scores)
        return {
            "total": len(texts),
            "filtered": sum(masks),
            "retained": len(texts) - sum(masks),
            "filter_rate": sum(masks) / len(texts) if texts else 0,
            "toxicity_threshold": self.toxicity_threshold,
            "severe_toxicity_threshold": self.severe_toxicity_threshold,
            "toxicity_score_p50": float(np.percentile(scores_arr, 50)) if scores_arr.size else 0,
            "toxicity_score_p90": float(np.percentile(scores_arr, 90)) if scores_arr.size else 0,
            "toxicity_score_p99": float(np.percentile(scores_arr, 99)) if scores_arr.size else 0,
        }
