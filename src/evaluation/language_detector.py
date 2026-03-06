"""
src/evaluation/language_detector.py
语言检测器（基于 fastText lid.176.bin）

Meta 发布的 fastText 语言检测模型，支持 176 种语言。
工业界标准：FineWeb/Dolma/RedPajama/CCNet 均使用此模型做语言过滤。

模型文件：data/models/lid.176.bin（~126MB）
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter
from tqdm import tqdm


class LanguageDetector:
    """
    基于 fastText 的语言检测器。

    用法：
        detector = LanguageDetector(model_path="data/models/lid.176.bin")
        results = detector.detect_batch(texts)
        stats = detector.compute_statistics(results)
    """

    def __init__(self, model_path: str = "data/models/lid.176.bin"):
        self.model_path = Path(model_path)
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        import fasttext
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"fastText lid 模型未找到: {self.model_path}\n"
                f"请从 https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin 下载"
            )
        # suppress fasttext warning about deprecated load_model
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model = fasttext.load_model(str(self.model_path))
        print(f"  加载 fastText lid: {self.model_path.name}")

    def detect(self, text: str) -> Tuple[str, float]:
        """
        检测单条文本的语言。

        Returns:
            (language_code, confidence)，如 ("en", 0.95)
        """
        self._load()
        # fastText 需要单行输入
        clean = text.replace("\n", " ").strip()[:5000]
        if not clean:
            return ("unknown", 0.0)

        labels, probs = self._model.predict(clean, k=1)
        lang = labels[0].replace("__label__", "")
        return (lang, float(probs[0]))

    def detect_batch(
        self,
        texts: List[str],
        show_progress: bool = True,
    ) -> List[Tuple[str, float]]:
        """
        批量语言检测。

        Returns:
            List of (language_code, confidence)
        """
        self._load()
        results = []
        iterator = enumerate(texts)
        if show_progress:
            iterator = tqdm(iterator, total=len(texts), desc="  语言检测", unit="doc")

        for _, text in iterator:
            results.append(self.detect(text))

        return results

    def compute_statistics(self, results: List[Tuple[str, float]]) -> Dict:
        """
        计算语言分布统计。

        Returns:
            dict，包含：
              - language_counts: 各语言的文档数
              - top_languages: Top 10 语言
              - english_ratio: 英文占比（分子=en 文档数，分母=总文档数）
              - avg_confidence: 平均检测置信度
        """
        langs = [r[0] for r in results]
        confs = [r[1] for r in results]
        counter = Counter(langs)
        total = len(results)

        en_count = counter.get("en", 0)

        return {
            "total_docs": total,
            "english_count": en_count,
            "english_ratio": en_count / total if total > 0 else 0,
            "avg_confidence": float(np.mean(confs)) if confs else 0,
            "n_languages": len(counter),
            "top_languages": [
                {"lang": lang, "count": count, "ratio": count / total}
                for lang, count in counter.most_common(10)
            ],
            "language_counts": dict(counter),
        }
