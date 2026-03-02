"""
src/gen1/filters/quality_filter.py
质量过滤器：整合 Gopher、C4 和 FineWeb 三套规则

方法论定位（第一代 Heuristic）：
  三套规则体系的互补关系：
  - Gopher（DeepMind 2021）：文档整体统计特征（长度、字母比、停用词等）
  - C4（Google 2020）：行级规则（标点、JS内容、Lorem ipsum 等）
  - FineWeb（HuggingFace 2024）：精炼补充规则（子弹点、省略号等）

  Heuristic 的核心局限（第一代 vs 第二代的关键区别）：
  这些规则能识别"明显的垃圾"（乱码、广告、模板），
  但无法区分"平庸内容"和"高质量内容"——两者都能通过所有规则。
  这就是为什么需要第二代的 fastText 分类器。
"""

import re
import unicodedata
from typing import List, Tuple, Optional, Dict, Set


# Gopher 规则用到的英文停用词（高频功能词）
ENGLISH_STOP_WORDS: Set[str] = frozenset([
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "it",
    "for", "not", "on", "with", "he", "as", "you", "do", "at", "this",
    "but", "his", "by", "from", "they", "we", "say", "her", "she", "or",
    "an", "will", "my", "one", "all", "would", "there", "their", "what",
    "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
    "when", "make", "can", "like", "time", "no", "just", "him", "know",
    "take", "people", "into", "year", "your", "good", "some", "could",
    "them", "see", "other", "than", "then", "now", "look", "only", "come",
    "its", "over", "think", "also", "back", "after", "use", "two", "how",
    "our", "work", "first", "well", "way", "even", "new", "want", "because",
    "any", "these", "give", "day", "most", "us",
])

# C4 规则：认可的句末标点
TERMINAL_PUNCTUATION = frozenset([".", "?", "!", '"', "'", "»", "…"])


def _word_count(text: str) -> int:
    """按空格分词计数。"""
    return len(text.split())


def _line_count(text: str) -> int:
    return len(text.splitlines())


def _alpha_ratio(text: str) -> float:
    """字母字符占所有字符的比例。"""
    if not text:
        return 0.0
    alpha_count = sum(1 for c in text if c.isalpha())
    return alpha_count / len(text)


def _stop_word_count(text: str) -> int:
    """文档中包含的停用词数量（英文）。"""
    words = text.lower().split()
    return sum(1 for w in words if w.strip(".,!?;:'\"") in ENGLISH_STOP_WORDS)


def _avg_sentence_length(text: str) -> float:
    """平均句子长度（词数）。用句末标点分句。"""
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return 0.0
    lengths = [len(s.split()) for s in sentences]
    return sum(lengths) / len(lengths)


# ─────────────────────────────────────────────────────────────
# Gopher 质量过滤器
# ─────────────────────────────────────────────────────────────

class GopherQualityFilter:
    """
    DeepMind Gopher 论文的质量过滤规则。
    基于文档整体统计特征，过滤明显低质量文档。

    规则来源：Rae et al. 2021, "Scaling Language Models: Methods,
    Analysis & Insights from Training Gopher"
    """

    def __init__(
        self,
        min_words: int = 50,
        max_words: int = 100000,
        min_avg_sentence_length: float = 3.0,
        max_avg_sentence_length: float = 1000.0,
        min_alpha_ratio: float = 0.7,
        min_stop_words: int = 2,
        max_ellipsis_lines_ratio: float = 0.3,
        max_non_alpha_words_ratio: float = 0.2,
    ):
        self.min_words = min_words
        self.max_words = max_words
        self.min_avg_sentence_length = min_avg_sentence_length
        self.max_avg_sentence_length = max_avg_sentence_length
        self.min_alpha_ratio = min_alpha_ratio
        self.min_stop_words = min_stop_words
        self.max_ellipsis_lines_ratio = max_ellipsis_lines_ratio
        self.max_non_alpha_words_ratio = max_non_alpha_words_ratio

    def check(self, text: str) -> Tuple[bool, str]:
        """
        检查文档是否通过 Gopher 质量过滤。

        Returns:
            (passes: bool, fail_reason: str)
            passes=True 表示通过（保留），fail_reason 为空
        """
        words = text.split()
        n_words = len(words)

        # 规则 1: 文档长度（词数）
        if n_words < self.min_words:
            return False, f"too_short:{n_words}<{self.min_words}"
        if n_words > self.max_words:
            return False, f"too_long:{n_words}>{self.max_words}"

        # 规则 2: 平均句子长度
        avg_sent = _avg_sentence_length(text)
        if avg_sent < self.min_avg_sentence_length:
            return False, f"avg_sentence_too_short:{avg_sent:.1f}"
        if avg_sent > self.max_avg_sentence_length:
            return False, f"avg_sentence_too_long:{avg_sent:.1f}"

        # 规则 3: 字母字符占比
        alpha = _alpha_ratio(text)
        if alpha < self.min_alpha_ratio:
            return False, f"low_alpha_ratio:{alpha:.2f}<{self.min_alpha_ratio}"

        # 规则 4: 停用词数量（至少包含 N 个英文停用词）
        stop_count = _stop_word_count(text)
        if stop_count < self.min_stop_words:
            return False, f"too_few_stop_words:{stop_count}<{self.min_stop_words}"

        # 规则 5: 省略号行比例上限
        lines = text.splitlines()
        if lines:
            ellipsis_lines = sum(1 for l in lines if l.rstrip().endswith("..."))
            if ellipsis_lines / len(lines) > self.max_ellipsis_lines_ratio:
                return False, f"too_many_ellipsis_lines:{ellipsis_lines/len(lines):.2f}"

        # 规则 6: 非字母词比例（过多符号/数字开头的词 = 可能是代码或乱码）
        non_alpha_words = sum(1 for w in words if w and not w[0].isalpha())
        if n_words > 0 and non_alpha_words / n_words > self.max_non_alpha_words_ratio:
            return False, f"too_many_non_alpha_words:{non_alpha_words/n_words:.2f}"

        return True, ""


# ─────────────────────────────────────────────────────────────
# C4 质量过滤器
# ─────────────────────────────────────────────────────────────

class C4QualityFilter:
    """
    Google C4 数据集的质量过滤规则。
    基于行级特征，与 Gopher 规则互补。

    规则来源：Raffel et al. 2020, "Exploring the Limits of Transfer
    Learning with a Unified Text-to-Text Transformer"

    C4 vs Gopher 的互补性：
    - C4 聚焦于行级特征（单行不符合就过滤整个文档）
    - Gopher 聚焦于文档整体统计（全局比例）
    - 联合使用能捕获更多类型的低质量内容
    """

    def __init__(
        self,
        min_lines: int = 3,
        min_words_per_line: int = 3,
        filter_javascript: bool = True,
        filter_lorem_ipsum: bool = True,
        require_terminal_punctuation: bool = True,
        terminal_punct_min_ratio: float = 0.7,  # 至少 70% 的行以句末标点结尾
    ):
        self.min_lines = min_lines
        self.min_words_per_line = min_words_per_line
        self.filter_javascript = filter_javascript
        self.filter_lorem_ipsum = filter_lorem_ipsum
        self.require_terminal_punctuation = require_terminal_punctuation
        self.terminal_punct_min_ratio = terminal_punct_min_ratio

    def check(self, text: str) -> Tuple[bool, str]:
        """
        Returns:
            (passes: bool, fail_reason: str)
        """
        lines = [l.strip() for l in text.splitlines() if l.strip()]

        # 规则 1: 最少行数
        if len(lines) < self.min_lines:
            return False, f"too_few_lines:{len(lines)}<{self.min_lines}"

        # 规则 2: JavaScript 内容
        if self.filter_javascript:
            js_indicators = ["javascript", "function()", "var ", "document.", "window."]
            for indicator in js_indicators:
                if indicator in text.lower():
                    return False, f"contains_javascript:{indicator}"

        # 规则 3: Lorem ipsum（模板占位文本）
        if self.filter_lorem_ipsum and "lorem ipsum" in text.lower():
            return False, "contains_lorem_ipsum"

        # 规则 4: 句末标点比例（非模板化文本的特征）
        if self.require_terminal_punctuation and lines:
            terminal_lines = sum(
                1 for l in lines
                if l and l[-1] in TERMINAL_PUNCTUATION
            )
            ratio = terminal_lines / len(lines)
            if ratio < self.terminal_punct_min_ratio:
                return False, f"low_terminal_punct_ratio:{ratio:.2f}<{self.terminal_punct_min_ratio}"

        # 规则 5: 每行最少词数（过滤只有标题/导航栏的页面）
        short_lines = sum(1 for l in lines if len(l.split()) < self.min_words_per_line)
        if lines and short_lines / len(lines) > 0.5:
            return False, f"too_many_short_lines:{short_lines}/{len(lines)}"

        return True, ""


# ─────────────────────────────────────────────────────────────
# FineWeb 自定义质量过滤器
# ─────────────────────────────────────────────────────────────

class FineWebQualityFilter:
    """
    HuggingFace FineWeb 的自定义质量规则（2024）。
    在 Gopher 和 C4 基础上的进一步精炼，针对 Common Crawl 的特有噪声。

    FineWeb 的核心发现：
    Gopher + C4 联合使用后，仍有一类噪声——"子弹点列表堆砌"。
    这类内容（大量 "• item\n• item\n• item"）通过了 Gopher 的停用词检查
    和 C4 的标点检查，但实际上信息密度极低。
    FineWeb 专门增加了子弹点比例的检查。
    """

    def __init__(
        self,
        max_bullet_lines_ratio: float = 0.9,
        max_ellipsis_lines_ratio: float = 0.3,
        min_alpha_words_ratio: float = 0.6,
    ):
        self.max_bullet_lines_ratio = max_bullet_lines_ratio
        self.max_ellipsis_lines_ratio = max_ellipsis_lines_ratio
        self.min_alpha_words_ratio = min_alpha_words_ratio

        # 子弹点标记（Markdown 和 HTML 常见）
        self._bullet_re = re.compile(r"^[\s]*[•\-\*\+►▸▶◆◇○●]\s")

    def check(self, text: str) -> Tuple[bool, str]:
        """
        Returns:
            (passes: bool, fail_reason: str)
        """
        lines = [l for l in text.splitlines() if l.strip()]
        if not lines:
            return False, "empty_text"

        # 规则 1: 子弹点行比例（避免 SEO 列表堆砌）
        bullet_lines = sum(1 for l in lines if self._bullet_re.match(l))
        bullet_ratio = bullet_lines / len(lines)
        if bullet_ratio > self.max_bullet_lines_ratio:
            return False, f"too_many_bullet_lines:{bullet_ratio:.2f}"

        # 规则 2: 省略号结尾行比例（跨页截断内容）
        ellipsis_lines = sum(1 for l in lines if l.rstrip().endswith("..."))
        ellipsis_ratio = ellipsis_lines / len(lines)
        if ellipsis_ratio > self.max_ellipsis_lines_ratio:
            return False, f"too_many_ellipsis_endings:{ellipsis_ratio:.2f}"

        # 规则 3: 含字母词的比例（过滤纯数字/符号内容）
        words = text.split()
        if words:
            alpha_words = sum(1 for w in words if any(c.isalpha() for c in w))
            alpha_ratio = alpha_words / len(words)
            if alpha_ratio < self.min_alpha_words_ratio:
                return False, f"low_alpha_word_ratio:{alpha_ratio:.2f}"

        return True, ""


# ─────────────────────────────────────────────────────────────
# 联合质量过滤器（对外暴露的统一接口）
# ─────────────────────────────────────────────────────────────

class QualityFilter:
    """
    联合质量过滤器：串联 Gopher + C4 + FineWeb 三套规则。
    第一代 Pipeline 的核心组件。
    """

    def __init__(
        self,
        use_gopher: bool = True,
        use_c4: bool = True,
        use_fineweb: bool = True,
        gopher_kwargs: Optional[Dict] = None,
        c4_kwargs: Optional[Dict] = None,
        fineweb_kwargs: Optional[Dict] = None,
    ):
        self.gopher = GopherQualityFilter(**(gopher_kwargs or {})) if use_gopher else None
        self.c4 = C4QualityFilter(**(c4_kwargs or {})) if use_c4 else None
        self.fineweb = FineWebQualityFilter(**(fineweb_kwargs or {})) if use_fineweb else None

    def check(self, text: str) -> Tuple[bool, str]:
        """
        按顺序应用三套规则，任一不通过即返回 False。
        """
        if self.gopher:
            passes, reason = self.gopher.check(text)
            if not passes:
                return False, f"gopher:{reason}"

        if self.c4:
            passes, reason = self.c4.check(text)
            if not passes:
                return False, f"c4:{reason}"

        if self.fineweb:
            passes, reason = self.fineweb.check(text)
            if not passes:
                return False, f"fineweb:{reason}"

        return True, ""

    def filter_batch(self, texts: List[str]) -> Tuple[List[bool], List[str]]:
        """
        批量过滤，返回 (filter_masks, fail_reasons)。
        filter_masks[i]=True 表示应被过滤（不通过）。
        """
        results = [self.check(text) for text in texts]
        masks = [not r[0] for r in results]   # check 返回 passes，取反得 filter_mask
        reasons = [r[1] for r in results]
        return masks, reasons

    def get_stats(self, texts: List[str]) -> Dict:
        """统计各规则的过滤贡献（用于 Sankey 图）。"""
        from collections import Counter
        masks, reasons = self.filter_batch(texts)
        reason_counter = Counter(
            r.split(":")[0] for r, m in zip(reasons, masks) if m
        )
        return {
            "total": len(texts),
            "filtered": sum(masks),
            "retained": len(texts) - sum(masks),
            "filter_rate": sum(masks) / len(texts) if texts else 0,
            "reason_breakdown": dict(reason_counter),
        }
