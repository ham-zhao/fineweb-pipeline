"""
src/dedup/exact_dedup.py
精确去重（Exact Deduplication）

方法论定位：
  精确去重 vs 模糊去重是互补的两步，在生产级 pipeline 中缺一不可。

  两步去重的标准流程：
  Step 1: 精确去重（本文件）—— O(n) 哈希，极快
    → 先去掉 15-25% 的完全重复文档
    → 大幅减少后续 MinHash 的计算量
  Step 2: 模糊去重（minhash_dedup.py）—— O(n * B * K)，较慢
    → 去掉"高度相似"但非完全相同的文档

  FineWeb 和 Nemotron-CC 都采用此两步流程。

  精确重复的来源：
  1. 同一 URL 被多次爬取（CC 每年多个 dump）
  2. 网站 A/B 测试导致完全相同内容在不同 URL 下
  3. 聚合站点（内容复制自其他网站，一字不差）
  4. 爬虫蜜罐（无限重复的页面）

使用 xxhash（非密码学哈希，速度极快）进行内容指纹。
"""

import xxhash
import json
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple
from collections import Counter
from tqdm import tqdm


def compute_doc_hash(text: str, normalize: bool = True) -> str:
    """
    计算文档的哈希值（内容指纹）。

    Args:
        text: 文档文本
        normalize: 是否先归一化（去除多余空白，转小写）
                   normalize=True：捕获格式不同但内容相同的文档
                   normalize=False：严格精确匹配

    Returns:
        32 字符的十六进制哈希字符串
    """
    if normalize:
        # 归一化：合并多余空白，转小写，去首尾空格
        text = " ".join(text.lower().split())
    return xxhash.xxh64(text.encode("utf-8")).hexdigest()


def exact_dedup(
    docs: List[Dict],
    text_field: str = "text",
    normalize: bool = True,
    keep: str = "first",   # "first" | "last"
) -> Tuple[List[Dict], Dict]:
    """
    对文档列表进行精确去重。

    Args:
        docs: 文档列表
        text_field: 文本字段名
        normalize: 是否归一化后再哈希（见 compute_doc_hash）
        keep: "first" 保留第一次出现，"last" 保留最后一次出现

    Returns:
        (deduplicated_docs, stats)
    """
    seen_hashes: Set[str] = set()
    kept_docs = []
    duplicate_groups: Dict[str, int] = Counter()  # hash → 出现次数

    # 第一遍：计算所有哈希
    all_hashes = []
    for doc in docs:
        text = doc.get(text_field, "")
        h = compute_doc_hash(text, normalize)
        all_hashes.append(h)
        duplicate_groups[h] += 1

    # 第二遍：过滤
    if keep == "last":
        # 反向遍历，然后翻转
        seen_hashes.clear()
        for doc, h in zip(reversed(docs), reversed(all_hashes)):
            if h not in seen_hashes:
                seen_hashes.add(h)
                kept_docs.append(doc)
        kept_docs.reverse()
    else:
        for doc, h in zip(docs, all_hashes):
            if h not in seen_hashes:
                seen_hashes.add(h)
                doc["_content_hash"] = h
                kept_docs.append(doc)

    # 统计
    n_total = len(docs)
    n_kept = len(kept_docs)
    n_removed = n_total - n_kept
    n_unique_hashes = len(duplicate_groups)
    n_duplicate_groups = sum(1 for c in duplicate_groups.values() if c > 1)

    stats = {
        "total_input": n_total,
        "unique_kept": n_kept,
        "duplicates_removed": n_removed,
        "dedup_rate": round(n_removed / n_total, 4) if n_total > 0 else 0,
        "unique_hash_count": n_unique_hashes,
        "duplicate_group_count": n_duplicate_groups,
        "most_duplicated": [
            {"hash": h, "count": c, "sample_text": ""}
            for h, c in Counter(all_hashes).most_common(5)
        ],
    }

    # 填充最常重复文档的文本预览
    hash_to_text = {}
    for doc, h in zip(docs, all_hashes):
        if h not in hash_to_text:
            hash_to_text[h] = doc.get(text_field, "")[:100]
    for item in stats["most_duplicated"]:
        item["sample_text"] = hash_to_text.get(item["hash"], "")

    print(f"  🔄 精确去重: {n_total:,} → {n_kept:,} 条 | 去除 {n_removed:,} 条 ({stats['dedup_rate']:.1%})")
    return kept_docs, stats


def compute_url_hash(url: str) -> str:
    """计算 URL 的哈希（用于 URL 级别去重，比文本哈希更快）。"""
    return xxhash.xxh64(url.encode("utf-8")).hexdigest()


def url_dedup(docs: List[Dict], url_field: str = "url") -> Tuple[List[Dict], Dict]:
    """
    基于 URL 的精确去重（比文本哈希更快，但不能捕获不同 URL 的相同内容）。
    通常作为文本哈希去重的前置步骤。
    """
    seen_urls: Set[str] = set()
    kept_docs = []
    n_url_dup = 0

    for doc in docs:
        url = doc.get(url_field, "")
        url_hash = compute_url_hash(url) if url else compute_doc_hash(doc.get("text", ""))
        if url_hash not in seen_urls:
            seen_urls.add(url_hash)
            kept_docs.append(doc)
        else:
            n_url_dup += 1

    stats = {
        "total_input": len(docs),
        "unique_kept": len(kept_docs),
        "url_duplicates_removed": n_url_dup,
        "dedup_rate": round(n_url_dup / len(docs), 4) if docs else 0,
    }
    print(f"  🔄 URL 去重: {len(docs):,} → {len(kept_docs):,} 条 | 去除 {n_url_dup:,} 条 URL 重复")
    return kept_docs, stats


def analyze_duplicate_sources(
    docs: List[Dict],
    text_field: str = "text",
    url_field: str = "url",
    normalize: bool = True,
) -> Dict:
    """
    分析重复文档的来源分布（同一网站 vs 跨网站）。
    用于 Notebook 05 的深度分析。

    Returns:
        dict，包含：
          - same_domain_dups: 同域名重复数
          - cross_domain_dups: 跨域名重复数
          - top_dup_domains: 重复最多的域名
    """
    from urllib.parse import urlparse
    import re

    hash_to_urls: Dict[str, List[str]] = {}
    for doc in docs:
        text = doc.get(text_field, "")
        h = compute_doc_hash(text, normalize)
        url = doc.get(url_field, "")
        if h not in hash_to_urls:
            hash_to_urls[h] = []
        hash_to_urls[h].append(url)

    # 只看重复组
    dup_groups = {h: urls for h, urls in hash_to_urls.items() if len(urls) > 1}

    def get_domain(url: str) -> str:
        try:
            domain = urlparse(url).netloc
            return re.sub(r"^www\.", "", domain.lower())
        except Exception:
            return ""

    same_domain_count = 0
    cross_domain_count = 0
    domain_dup_counter: Counter = Counter()

    for h, urls in dup_groups.items():
        domains = [get_domain(u) for u in urls if u]
        unique_domains = set(d for d in domains if d)
        for d in domains:
            domain_dup_counter[d] += 1
        if len(unique_domains) == 1:
            same_domain_count += 1
        elif len(unique_domains) > 1:
            cross_domain_count += 1

    return {
        "total_duplicate_groups": len(dup_groups),
        "same_domain_duplicates": same_domain_count,
        "cross_domain_duplicates": cross_domain_count,
        "top_10_dup_domains": [
            {"domain": d, "count": c}
            for d, c in domain_dup_counter.most_common(10)
        ],
    }
