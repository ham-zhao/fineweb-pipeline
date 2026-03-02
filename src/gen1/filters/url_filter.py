"""
src/gen1/filters/url_filter.py
URL 过滤器：第一代 Pipeline 的第一道关卡

方法论定位（第一代 Heuristic）：
  URL 过滤是最轻量的预处理步骤，在文本提取之前就剔除明显的垃圾域名。
  成本极低（只检查 URL 字符串），可以节省后续文本提取的计算量。

FineWeb 的实践：
  维护一个黑名单域名列表（adult content, spam, malware 等），
  以及 URL 中的关键词黑名单（"porn", "xxx", "casino" 等）。
"""

import re
from urllib.parse import urlparse
from typing import List, Set, Optional


# 常见垃圾/成人内容域名后缀和关键词
DEFAULT_BLACKLIST_KEYWORDS = frozenset([
    "porn", "xxx", "adult", "sex", "nude", "naked",
    "casino", "slots", "gambling", "poker",
    "spam", "phishing", "malware",
    "payday-loan", "paydayloan",
    "viagra", "cialis", "pharmacy",
])

DEFAULT_BLACKLIST_TLDS = frozenset([
    # 高垃圾率的 TLD（根据 CC 统计）
    ".tk", ".ml", ".ga", ".cf", ".gq",
])


class URLFilter:
    """
    基于 URL 的文档过滤器。
    检查域名黑名单、URL 关键词、以及 URL 结构异常。
    """

    def __init__(
        self,
        blacklist_domains: Optional[List[str]] = None,
        blacklist_keywords: Optional[List[str]] = None,
        blacklist_tlds: Optional[List[str]] = None,
    ):
        """
        Args:
            blacklist_domains: 完整域名黑名单（如 ["spam.com", "xxx.org"]）
            blacklist_keywords: URL 中的关键词黑名单
            blacklist_tlds: 黑名单顶级域名（如 [".tk", ".ml"]）
        """
        self.blacklist_domains: Set[str] = set(blacklist_domains or [])
        self.blacklist_keywords: Set[str] = set(blacklist_keywords or []) | DEFAULT_BLACKLIST_KEYWORDS
        self.blacklist_tlds: Set[str] = set(blacklist_tlds or []) | DEFAULT_BLACKLIST_TLDS

        # 编译关键词正则（一次性，比循环 in 快）
        if self.blacklist_keywords:
            pattern = "|".join(re.escape(kw) for kw in self.blacklist_keywords)
            self._keyword_re = re.compile(pattern, re.IGNORECASE)
        else:
            self._keyword_re = None

    def should_filter(self, url: str) -> tuple[bool, str]:
        """
        判断 URL 是否应被过滤。

        Args:
            url: 文档的 URL

        Returns:
            (should_filter: bool, reason: str)
        """
        if not url or not url.strip():
            return True, "empty_url"

        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            path = parsed.path.lower()
            full_url_lower = url.lower()
        except Exception:
            return True, "invalid_url"

        # 1. 域名黑名单检查
        # 去掉 www. 前缀后比较
        bare_domain = re.sub(r"^www\.", "", domain)
        if bare_domain in self.blacklist_domains or domain in self.blacklist_domains:
            return True, f"blacklist_domain:{bare_domain}"

        # 2. TLD 黑名单
        for tld in self.blacklist_tlds:
            if domain.endswith(tld):
                return True, f"blacklist_tld:{tld}"

        # 3. URL 关键词检查（域名 + 路径）
        if self._keyword_re and self._keyword_re.search(full_url_lower):
            matched = self._keyword_re.search(full_url_lower).group()
            return True, f"blacklist_keyword:{matched}"

        # 4. 异常 URL 结构（IP 地址直接访问，可能是爬虫蜜罐）
        if re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", domain):
            return True, "ip_address_url"

        # 5. 无域名（相对 URL 或 file:// 等）
        if not domain:
            return True, "no_domain"

        return False, ""

    def filter_batch(self, urls: List[str]) -> tuple[List[bool], List[str]]:
        """
        批量过滤。

        Returns:
            (filter_mask: List[bool], reasons: List[str])
            filter_mask[i]=True 表示应被过滤
        """
        results = [self.should_filter(url) for url in urls]
        masks = [r[0] for r in results]
        reasons = [r[1] for r in results]
        return masks, reasons

    def get_stats(self, urls: List[str]) -> dict:
        """
        统计过滤结果（用于 Notebook 可视化）。
        """
        masks, reasons = self.filter_batch(urls)
        from collections import Counter
        reason_counts = Counter(r for r, m in zip(reasons, masks) if m)

        return {
            "total": len(urls),
            "filtered": sum(masks),
            "retained": len(urls) - sum(masks),
            "filter_rate": sum(masks) / len(urls) if urls else 0,
            "reasons": dict(reason_counts),
        }
