"""
src/gen1/filters/language_filter.py
语言识别过滤器

方法论定位（第一代 Heuristic）：
  使用 fastText 的语言识别模型（lid.176.bin）判断文档语言。
  Common Crawl 中非目标语言文档约占 50-60%（对于英文 pipeline）。
  语言过滤是保证训练数据纯净度最基础的步骤。

技术选型：
  fastText langid vs langdetect vs cld3
  fastText langid（lid.176.bin）是业界标准：
  - 支持 176 种语言
  - 速度极快（微秒级/文档）
  - 在短文本上精度优于 langdetect
  FineWeb 和 DCLM 都用 fastText langid。
"""

import re
from pathlib import Path
from typing import List, Optional, Tuple
import urllib.request


# fastText 语言识别模型下载 URL（Meta 官方）
FASTTEXT_LANGID_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
FASTTEXT_LANGID_PATH = Path("data/models/lid.176.bin")


def download_langid_model(dest_path: Optional[Path] = None) -> Path:
    """下载 fastText 语言识别模型（只需下载一次，约 131MB）。"""
    dest = dest_path or FASTTEXT_LANGID_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        return dest

    print(f"  ⬇️  下载 fastText 语言识别模型（131MB）...")
    urllib.request.urlretrieve(FASTTEXT_LANGID_URL, dest)
    print(f"  ✅ 语言模型已保存: {dest}")
    return dest


class LanguageFilter:
    """
    基于 fastText 的语言识别过滤器。
    保留指定目标语言的文档，过滤其他语言。
    """

    def __init__(
        self,
        target_language: str = "en",
        min_confidence: float = 0.65,
        model_path: Optional[str] = None,
        fallback_to_langdetect: bool = True,
    ):
        """
        Args:
            target_language: 目标语言代码（ISO 639-1），如 "en", "zh", "de"
            min_confidence: 最低置信度阈值（低于此值视为"不确定"→过滤）
            model_path: lid.176.bin 路径（None 则自动下载）
            fallback_to_langdetect: fastText 不可用时是否回退到 langdetect
        """
        self.target_language = target_language
        self.min_confidence = min_confidence
        self._model = None
        self._model_path = model_path or str(FASTTEXT_LANGID_PATH)
        self._fallback = fallback_to_langdetect
        self._use_fasttext = False

    def _load_model(self) -> None:
        """懒加载语言识别模型。"""
        if self._model is not None:
            return

        try:
            import fasttext
            model_path = Path(self._model_path)
            if not model_path.exists():
                download_langid_model(model_path)
            self._model = fasttext.load_model(str(model_path))
            self._use_fasttext = True
            print(f"  ✅ fastText langid 模型已加载")
        except ImportError:
            if self._fallback:
                print(f"  ⚠️  fasttext 未安装，回退到 langdetect")
                self._use_fasttext = False
            else:
                raise

    def detect_language(self, text: str) -> Tuple[str, float]:
        """
        检测文本语言。

        Returns:
            (language_code, confidence)，如 ("en", 0.98)
        """
        self._load_model()

        # 清理文本（fastText 对换行符敏感）
        clean_text = text.replace("\n", " ").strip()
        if not clean_text or len(clean_text) < 10:
            return "unknown", 0.0

        if self._use_fasttext and self._model:
            # fastText 返回 (__label__en, 0.98)
            labels, probs = self._model.predict(clean_text[:1000], k=1)
            lang = labels[0].replace("__label__", "")
            prob = float(probs[0])
            return lang, prob
        else:
            # fallback: langdetect
            try:
                from langdetect import detect_langs
                results = detect_langs(clean_text[:1000])
                if results:
                    return results[0].lang, results[0].prob
            except Exception:
                pass
            return "unknown", 0.0

    def should_filter(self, text: str) -> Tuple[bool, str, float]:
        """
        判断文档是否应被过滤（非目标语言）。

        Returns:
            (should_filter, detected_language, confidence)
        """
        lang, confidence = self.detect_language(text)

        if lang == self.target_language and confidence >= self.min_confidence:
            return False, lang, confidence

        if lang != self.target_language:
            return True, lang, confidence

        # 目标语言但置信度低
        return True, lang, confidence

    def filter_batch(
        self, texts: List[str]
    ) -> Tuple[List[bool], List[str], List[float]]:
        """
        批量语言过滤。

        Returns:
            (filter_masks, detected_languages, confidences)
        """
        masks, langs, confs = [], [], []
        for text in texts:
            m, l, c = self.should_filter(text)
            masks.append(m)
            langs.append(l)
            confs.append(c)
        return masks, langs, confs

    def get_language_distribution(self, texts: List[str], sample_size: int = 500) -> dict:
        """
        统计文档集的语言分布（用于 Notebook 可视化）。
        """
        import random
        from collections import Counter

        sample = random.sample(texts, min(sample_size, len(texts)))
        lang_counts = Counter()
        for text in sample:
            lang, conf = self.detect_language(text)
            if conf >= 0.5:
                lang_counts[lang] += 1
            else:
                lang_counts["uncertain"] += 1

        total = sum(lang_counts.values())
        return {
            "language_counts": dict(lang_counts.most_common(20)),
            "target_language": self.target_language,
            "target_rate": lang_counts.get(self.target_language, 0) / total if total > 0 else 0,
            "sample_size": len(sample),
        }
