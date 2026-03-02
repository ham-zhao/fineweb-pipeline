"""
src/gen1/filters/pii_filter.py
PII（个人身份信息）脱敏过滤器

方法论定位（第一代 Heuristic）：
  PII 过滤是数据合规的必要步骤，不是质量过滤。
  即使高质量文档，也可能含有用户的个人信息（邮件、电话、IP、信用卡等）。

  脱敏 vs 过滤：
  - 工业实践通常选择"脱敏"（mask）而非直接过滤整条文档
  - 因为完整文档往往有价值，只需替换敏感字段
  - 例外：整篇文档就是电话本/邮件列表 → 直接过滤整条

  FineWeb 的做法：
  使用 PIIFormatter 将 email/phone 替换为 [EMAIL]/[PHONE] 占位符，
  保留文档其余内容。
"""

import re
from typing import List, Tuple, Dict, Optional


# ─────────────────────────────────────────────────────────────
# 正则模式（按精确度排序，高精度在前）
# ─────────────────────────────────────────────────────────────

PII_PATTERNS = {
    # 电子邮件（RFC 5321 简化版）
    "email": re.compile(
        r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
    ),

    # 美国电话号码（多种格式）
    "phone_us": re.compile(
        r"\b(?:\+?1[-.\s]?)?"
        r"(?:\(?\d{3}\)?[-.\s]?)?"
        r"\d{3}[-.\s]?\d{4}\b"
    ),

    # IPv4 地址
    "ip_v4": re.compile(
        r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}"
        r"(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
    ),

    # 信用卡号（Luhn 算法验证略，仅格式）
    "credit_card": re.compile(
        r"\b(?:4[0-9]{12}(?:[0-9]{3})?|"   # Visa
        r"5[1-5][0-9]{14}|"                  # Mastercard
        r"3[47][0-9]{13}|"                   # Amex
        r"6(?:011|5[0-9]{2})[0-9]{12})\b"   # Discover
    ),

    # 美国社会安全号（SSN）
    "ssn": re.compile(
        r"\b(?!000|666|9\d{2})\d{3}"
        r"[-\s]?"
        r"(?!00)\d{2}"
        r"[-\s]?"
        r"(?!0{4})\d{4}\b"
    ),
}

# 脱敏占位符
PII_PLACEHOLDERS = {
    "email": "[EMAIL]",
    "phone_us": "[PHONE]",
    "ip_v4": "[IP_ADDRESS]",
    "credit_card": "[CREDIT_CARD]",
    "ssn": "[SSN]",
}


class PIIFilter:
    """
    PII 脱敏器：将文档中的个人信息替换为占位符。
    支持两种模式：mask（脱敏保留）和 filter（直接过滤整条）。
    """

    def __init__(
        self,
        mode: str = "mask",   # "mask" | "filter"
        pii_types: Optional[List[str]] = None,
        # filter 模式下：PII 密度超过此比例则过滤整条
        max_pii_density: float = 0.3,
    ):
        """
        Args:
            mode: "mask" 替换占位符，"filter" 检测到就过滤整条文档
            pii_types: 要处理的 PII 类型（None = 全部）
            max_pii_density: mask 模式下，PII 占文档词数比例超过此值则过滤整条
                            （说明整篇就是 PII 列表）
        """
        assert mode in ("mask", "filter"), "mode 必须是 'mask' 或 'filter'"
        self.mode = mode
        self.max_pii_density = max_pii_density

        active_types = pii_types or list(PII_PATTERNS.keys())
        self.patterns = {k: v for k, v in PII_PATTERNS.items() if k in active_types}
        self.placeholders = {k: v for k, v in PII_PLACEHOLDERS.items() if k in active_types}

    def detect(self, text: str) -> Dict[str, List[str]]:
        """
        检测文本中的所有 PII。

        Returns:
            dict，key 为 PII 类型，value 为匹配到的字符串列表
        """
        findings = {}
        for pii_type, pattern in self.patterns.items():
            matches = pattern.findall(text)
            if matches:
                findings[pii_type] = matches
        return findings

    def mask(self, text: str) -> Tuple[str, Dict[str, int]]:
        """
        将文本中的 PII 替换为占位符。

        Returns:
            (masked_text, replacement_counts)
        """
        masked = text
        counts = {}
        for pii_type, pattern in self.patterns.items():
            placeholder = self.placeholders.get(pii_type, f"[{pii_type.upper()}]")
            masked, n = pattern.subn(placeholder, masked)
            if n > 0:
                counts[pii_type] = n
        return masked, counts

    def process(self, text: str) -> Tuple[Optional[str], str, Dict]:
        """
        处理单条文档。

        Returns:
            (processed_text, action, stats)
            - processed_text: 处理后的文本（filter 模式下被过滤则为 None）
            - action: "kept" | "masked" | "filtered"
            - stats: 检测到的 PII 统计
        """
        findings = self.detect(text)

        if not findings:
            return text, "kept", {}

        if self.mode == "filter":
            return None, "filtered", {k: len(v) for k, v in findings.items()}

        # mask 模式
        masked_text, counts = self.mask(text)

        # 检查 PII 密度（如果整篇都是 PII，直接过滤）
        total_pii_matches = sum(counts.values())
        word_count = len(text.split())
        if word_count > 0 and total_pii_matches / word_count > self.max_pii_density:
            return None, "filtered_high_density", counts

        return masked_text, "masked", counts

    def process_batch(
        self, texts: List[str]
    ) -> Tuple[List[Optional[str]], List[str], List[Dict]]:
        """
        批量处理。

        Returns:
            (processed_texts, actions, stats_list)
        """
        results = [self.process(text) for text in texts]
        return (
            [r[0] for r in results],
            [r[1] for r in results],
            [r[2] for r in results],
        )

    def get_stats(self, texts: List[str]) -> Dict:
        """统计 PII 处理结果（用于 Notebook 报告）。"""
        from collections import Counter

        all_pii_types = Counter()
        action_counts = Counter()

        for text in texts:
            _, action, stats = self.process(text)
            action_counts[action] += 1
            for pii_type, count in stats.items():
                all_pii_types[pii_type] += count

        return {
            "total": len(texts),
            "action_counts": dict(action_counts),
            "filtered_count": action_counts.get("filtered", 0)
                              + action_counts.get("filtered_high_density", 0),
            "masked_count": action_counts.get("masked", 0),
            "pii_type_counts": dict(all_pii_types),
        }
