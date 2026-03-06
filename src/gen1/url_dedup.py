"""
src/gen1/url_dedup.py
URL 标准化与去重模块

产出：去重后的文档列表 + 去重统计
"""

import hashlib
from urllib.parse import urlparse, urlencode, parse_qs
from typing import List, Dict, Tuple


def url_normalize(url: str) -> str:
    """
    标准化 URL：
    1. http/https 统一为 https
    2. 移除 www. 前缀
    3. 移除尾部斜杠
    4. query 参数按 key 排序
    5. 移除 fragment (#...)
    6. 小写化 scheme 和 host
    """
    if not url:
        return ""
    try:
        parsed = urlparse(url.strip())
        scheme = "https"
        host = (parsed.hostname or "").lower()
        if host.startswith("www."):
            host = host[4:]
        path = parsed.path.rstrip("/") or "/"
        query_params = parse_qs(parsed.query, keep_blank_values=True)
        sorted_query = urlencode(sorted(query_params.items()), doseq=True)
        normalized = f"{scheme}://{host}{path}"
        if sorted_query:
            normalized += f"?{sorted_query}"
        return normalized
    except Exception:
        return url.strip().lower()


def url_dedup(docs: List[Dict]) -> Tuple[List[Dict], Dict]:
    """
    基于标准化 URL 去重。

    Args:
        docs: 输入文档列表，每条需含 "url" 字段

    Returns:
        (去重后文档列表, 统计字典)
    """
    seen_hashes = set()
    kept = []
    removed = 0

    for doc in docs:
        raw_url = doc.get("url", "")
        norm = url_normalize(raw_url)
        url_hash = hashlib.md5(norm.encode()).hexdigest()

        if url_hash in seen_hashes:
            removed += 1
        else:
            seen_hashes.add(url_hash)
            kept.append(doc)

    stats = {
        "input_count": len(docs),
        "output_count": len(kept),
        "removed_count": removed,
        "dedup_rate": removed / len(docs) if docs else 0,
        "unique_urls": len(seen_hashes),
    }
    return kept, stats
