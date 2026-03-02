"""
src/gen1_zh/chinese_quality_filter.py
中文专用质量过滤器

设计参考：
  - 微软 WuDao 数据集清洗规则（字符数阈值）
  - 百度 ERNIE 预训练数据清洗（标点比率、重复检测）
  - FineWeb 英文规则的中文适配版

与英文 QualityFilter 的主要差异：
  ┌─────────────────┬──────────────────┬──────────────────────┐
  │ 指标            │ 英文（Gopher）   │ 中文（本模块）       │
  ├─────────────────┼──────────────────┼──────────────────────┤
  │ 长度单位        │ word count       │ 字符数（CJK chars）  │
  │ 最短文档        │ 50 words         │ 100 字               │
  │ 最长文档        │ 100,000 words    │ 50,000 字            │
  │ 字母比例        │ alpha_ratio≥0.7  │ zh_char_ratio≥0.2    │
  │ 标点终止        │ terminal punct   │ 句末句号/！/？       │
  │ 重复检测        │ 2/3/4-gram       │ 字符 bi-gram/tri-gram│
  └─────────────────┴──────────────────┴──────────────────────┘

使用方式：
    from src.gen1_zh.chinese_quality_filter import ChineseQualityFilter
    f = ChineseQualityFilter()
    passes, reason = f.check(doc["text"])
"""

import re
from typing import Tuple, Dict, Optional
from collections import Counter

from src.gen1_zh.chinese_text_utils import (
    count_chinese_chars,
    char_type_ratio,
    compute_zh_ngrams,
    compute_spam_score,
    estimate_zh_tokens,
    detect_script,
)


# ── 中文质量规则 ──────────────────────────────────────────────────

class ChineseQualityFilter:
    """
    中文 Heuristic 质量过滤器。

    过滤逻辑（按顺序）：
      1. 字符数检查（过短/过长）
      2. 中文字符占比（排除英文/日文为主的文档）
      3. 句末标点率（低于阈值 → 碎片/列表型文档）
      4. 行级重复率（大量重复行 → 模板/爬虫内容）
      5. 字符 n-gram 重复率（bi-gram 重复 → 循环句/灌水）
      6. 垃圾内容评分（广告/SEO/引流话术）
      7. 段落数量检查（过少段落 → 碎片内容）
    """

    def __init__(
        self,
        # 字符数阈值（汉字数）
        min_chinese_chars: int = 100,
        max_chinese_chars: int = 50000,
        # 中文字符占比
        min_zh_char_ratio: float = 0.20,
        # 句末标点率（含 。！？…）
        min_terminal_punct_ratio: float = 0.20,
        # 行级重复
        max_duplicate_line_ratio: float = 0.50,
        # 字符 bi-gram 重复率（top-1 bi-gram 出现次数 / 总 bi-gram 数）
        max_bigram_top_ratio: float = 0.30,
        # 垃圾评分阈值
        max_spam_score: float = 0.35,
        # 段落数
        min_paragraphs: int = 1,
    ):
        self.min_chinese_chars = min_chinese_chars
        self.max_chinese_chars = max_chinese_chars
        self.min_zh_char_ratio = min_zh_char_ratio
        self.min_terminal_punct_ratio = min_terminal_punct_ratio
        self.max_duplicate_line_ratio = max_duplicate_line_ratio
        self.max_bigram_top_ratio = max_bigram_top_ratio
        self.max_spam_score = max_spam_score
        self.min_paragraphs = min_paragraphs

        # 句末标点集
        self._terminal_puncts = set("。！？…")
        # 中文句子分割符（用于行分析）
        self._sent_split = re.compile(r"[。！？\n]+")

    # ── 各规则实现 ──────────────────────────────────────────────

    def _check_length(self, text: str, zh_count: int) -> Tuple[bool, str]:
        if zh_count < self.min_chinese_chars:
            return False, f"zh_too_short:{zh_count}<{self.min_chinese_chars}"
        if zh_count > self.max_chinese_chars:
            return False, f"zh_too_long:{zh_count}>{self.max_chinese_chars}"
        return True, ""

    def _check_zh_ratio(self, ratios: Dict[str, float]) -> Tuple[bool, str]:
        zh_ratio = ratios["chinese"]
        if zh_ratio < self.min_zh_char_ratio:
            return False, f"low_zh_ratio:{zh_ratio:.3f}<{self.min_zh_char_ratio}"
        return True, ""

    def _check_terminal_punct(self, text: str) -> Tuple[bool, str]:
        """检查行/句的终结标点覆盖率。"""
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        if not lines:
            return False, "no_lines"
        terminal_count = sum(
            1 for l in lines if l and l[-1] in self._terminal_puncts
        )
        ratio = terminal_count / len(lines)
        if ratio < self.min_terminal_punct_ratio:
            return False, f"low_terminal_punct:{ratio:.3f}<{self.min_terminal_punct_ratio}"
        return True, ""

    def _check_duplicate_lines(self, text: str) -> Tuple[bool, str]:
        lines = [l.strip() for l in text.split("\n") if l.strip() and len(l.strip()) > 3]
        if not lines:
            return True, ""
        counter = Counter(lines)
        dup_count = sum(v - 1 for v in counter.values() if v > 1)
        dup_ratio = dup_count / len(lines)
        if dup_ratio > self.max_duplicate_line_ratio:
            return False, f"high_dup_lines:{dup_ratio:.3f}>{self.max_duplicate_line_ratio}"
        return True, ""

    def _check_bigram_repetition(self, text: str, zh_count: int) -> Tuple[bool, str]:
        """检查字符 bi-gram 重复（循环句检测）。"""
        if zh_count < 50:
            return True, ""
        bigrams = compute_zh_ngrams(text, n=2)
        if not bigrams:
            return True, ""
        total = sum(bigrams.values())
        top_count = bigrams.most_common(1)[0][1]
        top_ratio = top_count / total
        if top_ratio > self.max_bigram_top_ratio:
            return False, f"high_bigram_rep:{top_ratio:.3f}>{self.max_bigram_top_ratio}"
        return True, ""

    def _check_spam(self, text: str) -> Tuple[bool, str]:
        score = compute_spam_score(text)
        if score > self.max_spam_score:
            return False, f"spam_score:{score:.3f}>{self.max_spam_score}"
        return True, ""

    def _check_paragraphs(self, text: str) -> Tuple[bool, str]:
        paras = [p.strip() for p in text.split("\n\n") if p.strip()]
        if len(paras) < self.min_paragraphs:
            return False, f"too_few_paragraphs:{len(paras)}<{self.min_paragraphs}"
        return True, ""

    # ── 主接口 ─────────────────────────────────────────────────

    def check(self, text: str) -> Tuple[bool, str]:
        """
        对单个文档执行所有质量检查。

        Returns:
            (passes: bool, fail_reason: str)
            passes=True 时 fail_reason=""
        """
        if not text or not text.strip():
            return False, "empty_text"

        zh_count = count_chinese_chars(text)
        ratios = char_type_ratio(text)

        checks = [
            self._check_length(text, zh_count),
            self._check_zh_ratio(ratios),
            self._check_terminal_punct(text),
            self._check_duplicate_lines(text),
            self._check_bigram_repetition(text, zh_count),
            self._check_spam(text),
            self._check_paragraphs(text),
        ]

        for ok, reason in checks:
            if not ok:
                return False, reason

        return True, ""

    def compute_scores(self, text: str) -> Dict:
        """
        返回所有指标的数值（用于 Notebook 可视化和调参）。
        不做过滤决策，只计算数值。
        """
        zh_count = count_chinese_chars(text)
        ratios = char_type_ratio(text)

        lines = [l.strip() for l in text.split("\n") if l.strip()]
        terminal_count = sum(1 for l in lines if l and l[-1] in self._terminal_puncts)
        terminal_ratio = terminal_count / len(lines) if lines else 0

        dup_lines = [l for l in lines if len(l) > 3]
        dup_counter = Counter(dup_lines)
        dup_count = sum(v - 1 for v in dup_counter.values() if v > 1)
        dup_line_ratio = dup_count / len(dup_lines) if dup_lines else 0

        bigrams = compute_zh_ngrams(text, n=2)
        if bigrams:
            total_bg = sum(bigrams.values())
            top_bg_ratio = bigrams.most_common(1)[0][1] / total_bg
        else:
            top_bg_ratio = 0

        paras = [p.strip() for p in text.split("\n\n") if p.strip()]

        return {
            "zh_char_count": zh_count,
            "total_chars": len(text),
            "zh_char_ratio": ratios["chinese"],
            "latin_ratio": ratios["latin"],
            "digit_ratio": ratios["digit"],
            "punct_ratio": ratios["punctuation"],
            "terminal_punct_ratio": round(terminal_ratio, 4),
            "duplicate_line_ratio": round(dup_line_ratio, 4),
            "top_bigram_ratio": round(top_bg_ratio, 4),
            "spam_score": round(compute_spam_score(text), 4),
            "paragraph_count": len(paras),
            "line_count": len(lines),
            "estimated_tokens": estimate_zh_tokens(text),
            "script_type": detect_script(text),
        }

    def filter_batch(self, docs: list, text_field: str = "text") -> Tuple[list, Dict]:
        """
        对文档列表批量过滤。

        Returns:
            (passed_docs, stats_dict)
        """
        passed = []
        fail_reasons: Counter = Counter()

        for doc in docs:
            text = doc.get(text_field, "")
            ok, reason = self.check(text)
            if ok:
                passed.append(doc)
            else:
                fail_reasons[reason.split(":")[0]] += 1

        total = len(docs)
        stats = {
            "input_count": total,
            "passed_count": len(passed),
            "filtered_count": total - len(passed),
            "retention_rate": round(len(passed) / total, 4) if total else 0,
            "fail_reasons": dict(fail_reasons.most_common()),
        }
        return passed, stats
