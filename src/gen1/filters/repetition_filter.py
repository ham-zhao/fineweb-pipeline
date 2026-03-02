"""
src/gen1/filters/repetition_filter.py
Gopher 重复过滤器：行级 + N-gram 级重复检测

方法论定位（第一代 Heuristic）：
  重复内容是 Web 数据中最常见的噪声之一，来源包括：
  1. 导航栏/页脚模板（同一网站大量页面复用相同的导航文本）
  2. 爬虫陷阱（无限重复的内容）
  3. 低质量 SEO 内容（关键词重复堆砌）
  4. 机器翻译/生成的重复句式

  Gopher 重复过滤的两个层次：
  - 行级重复：相同的行在文档中出现多次（导航模板）
  - N-gram 重复：Top N-gram 占总词数比例过高（关键词堆砌）

  重要提示：这里的"去重"是单文档内部的重复检测，
  不同于 MinHash 的跨文档去重（Notebook 05 会详细对比）。
"""

from collections import Counter
from typing import List, Tuple, Dict, Optional


def _get_ngrams(words: List[str], n: int) -> List[Tuple]:
    """生成词序列的 N-gram 列表。"""
    return [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]


def _duplicate_line_fraction(text: str) -> float:
    """计算重复行（完全相同的非空行）占总非空行数的比例。"""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return 0.0
    counts = Counter(lines)
    duplicate_lines = sum(count - 1 for count in counts.values() if count > 1)
    return duplicate_lines / len(lines)


def _duplicate_paragraph_fraction(text: str) -> float:
    """计算重复段落（以空行分隔）占总段落数的比例。"""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return 0.0
    counts = Counter(paragraphs)
    duplicate_paras = sum(count - 1 for count in counts.values() if count > 1)
    return duplicate_paras / len(paragraphs)


def _top_ngram_fraction(words: List[str], n: int) -> float:
    """
    计算最高频 N-gram 的总出现次数占总 N-gram 数的比例。
    Gopher 规则：top N-gram 占比过高 → 关键词堆砌。
    """
    if len(words) < n:
        return 0.0
    ngrams = _get_ngrams(words, n)
    if not ngrams:
        return 0.0
    counter = Counter(ngrams)
    # 只看最高频的那一个 N-gram
    top_count = counter.most_common(1)[0][1]
    return (top_count * n) / len(words)


def _duplicate_ngram_fraction(words: List[str], n: int) -> float:
    """
    计算重复 N-gram（出现超过 1 次的）覆盖的总词数比例。
    这捕获的是"多个不同的重复 N-gram"的情况，比 top_ngram 更全面。
    """
    if len(words) < n:
        return 0.0
    ngrams = _get_ngrams(words, n)
    if not ngrams:
        return 0.0

    counter = Counter(ngrams)
    duplicate_ngram_chars = sum(
        count * n
        for ngram, count in counter.items()
        if count > 1
    )
    return duplicate_ngram_chars / len(words)


class GopherRepetitionFilter:
    """
    Gopher 重复过滤器。
    检测文档内部的重复内容（行级 + N-gram 级）。

    注意：这是单文档内部检测，与跨文档的 MinHash 去重不同。
    """

    def __init__(
        self,
        # 行级重复阈值
        max_duplicate_line_fraction: float = 0.30,
        max_duplicate_paragraph_fraction: float = 0.30,
        # Top N-gram 占比阈值（防止关键词堆砌）
        max_top_2gram_fraction: float = 0.20,
        max_top_3gram_fraction: float = 0.18,
        max_top_4gram_fraction: float = 0.16,
        # 重复 N-gram 总覆盖率阈值（防止句式重复）
        max_dup_5gram_fraction: float = 0.15,
        max_dup_6gram_fraction: float = 0.14,
        max_dup_7gram_fraction: float = 0.13,
        max_dup_8gram_fraction: float = 0.12,
        max_dup_9gram_fraction: float = 0.11,
        max_dup_10gram_fraction: float = 0.10,
    ):
        self.max_dup_line = max_duplicate_line_fraction
        self.max_dup_para = max_duplicate_paragraph_fraction
        self.max_top_ngram = {
            2: max_top_2gram_fraction,
            3: max_top_3gram_fraction,
            4: max_top_4gram_fraction,
        }
        self.max_dup_ngram = {
            5: max_dup_5gram_fraction,
            6: max_dup_6gram_fraction,
            7: max_dup_7gram_fraction,
            8: max_dup_8gram_fraction,
            9: max_dup_9gram_fraction,
            10: max_dup_10gram_fraction,
        }

    def check(self, text: str) -> Tuple[bool, str]:
        """
        Returns:
            (passes: bool, fail_reason: str)
            passes=True 表示通过（保留文档）
        """
        # 行级重复
        dup_line = _duplicate_line_fraction(text)
        if dup_line > self.max_dup_line:
            return False, f"dup_line_fraction:{dup_line:.3f}>{self.max_dup_line}"

        dup_para = _duplicate_paragraph_fraction(text)
        if dup_para > self.max_dup_para:
            return False, f"dup_para_fraction:{dup_para:.3f}>{self.max_dup_para}"

        # 词级分析（小写化后统计）
        words = text.lower().split()
        if not words:
            return False, "empty_text"

        # Top N-gram 占比
        for n, threshold in self.max_top_ngram.items():
            frac = _top_ngram_fraction(words, n)
            if frac > threshold:
                return False, f"top_{n}gram_fraction:{frac:.3f}>{threshold}"

        # 重复 N-gram 覆盖率
        for n, threshold in self.max_dup_ngram.items():
            frac = _duplicate_ngram_fraction(words, n)
            if frac > threshold:
                return False, f"dup_{n}gram_fraction:{frac:.3f}>{threshold}"

        return True, ""

    def filter_batch(self, texts: List[str]) -> Tuple[List[bool], List[str]]:
        """
        批量过滤。filter_masks[i]=True 表示应被过滤。
        """
        results = [self.check(text) for text in texts]
        masks = [not r[0] for r in results]
        reasons = [r[1] for r in results]
        return masks, reasons

    def compute_scores(self, text: str) -> Dict[str, float]:
        """
        计算所有重复指标的分数（用于 Notebook 可视化和调参）。
        """
        words = text.lower().split()
        scores = {
            "dup_line_fraction": _duplicate_line_fraction(text),
            "dup_para_fraction": _duplicate_paragraph_fraction(text),
        }
        for n in [2, 3, 4]:
            scores[f"top_{n}gram_fraction"] = _top_ngram_fraction(words, n)
        for n in [5, 6, 7, 8, 9, 10]:
            scores[f"dup_{n}gram_fraction"] = _duplicate_ngram_fraction(words, n)
        return scores

    def get_stats(self, texts: List[str]) -> Dict:
        """统计过滤结果。"""
        from collections import Counter
        masks, reasons = self.filter_batch(texts)
        reason_cats = Counter(r.split(":")[0] for r, m in zip(reasons, masks) if m)
        return {
            "total": len(texts),
            "filtered": sum(masks),
            "retained": len(texts) - sum(masks),
            "filter_rate": sum(masks) / len(texts) if texts else 0,
            "reason_breakdown": dict(reason_cats),
        }
