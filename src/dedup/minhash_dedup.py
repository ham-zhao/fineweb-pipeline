"""
src/dedup/minhash_dedup.py
MinHash LSH 模糊去重

核心原理（Notebook 05 详细讲解，这里是实现）：
  MinHash 用随机哈希函数近似估计两个集合的 Jaccard 相似度：
    J(A, B) = |A ∩ B| / |A ∪ B|

  三步流程（datatrove 的实现架构）：
  Step 1: Shingling —— 将文档拆成 n-gram 集合（Shingles）
  Step 2: MinHash  —— 用 num_hashes 个哈希函数，每个取最小值，构成签名
  Step 3: LSH      —— 将签名分成 num_buckets 段，相似文档大概率落入同一桶

  关键参数的 Trade-off：
  - num_hashes（签名长度）：越大估计越准，越慢
  - num_buckets（桶数）：越多，误判越少，但内存越大
  - threshold（Jaccard 阈值）：决定"多相似算重复"

  去重粒度选择：
  - 文档级（本文件）：整篇文档的 MinHash，速度快，FineWeb 使用
  - 句子级（C4 风格）：3-sentence sliding window，更彻底但贵

  ⚠️  MinHash 是概率性的：
  相似度恰好在阈值附近的文档对，有时被判重复有时不被判，
  取决于哈希函数的随机性。这不是 bug，是 LSH 的本质特性。
"""

import hashlib
import struct
import numpy as np
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict
from tqdm import tqdm


# ── MinHash 实现 ──────────────────────────────────────────────

def _shingle(text: str, n: int = 5) -> Set[str]:
    """
    将文本拆成 n-gram shingles（字符级）。
    字符级比词级更对噪声鲁棒（大小写、标点变化不影响相似度）。
    """
    text_norm = " ".join(text.lower().split())
    return {text_norm[i:i+n] for i in range(max(1, len(text_norm) - n + 1))}


def _hash_shingle(shingle: str, seed: int) -> int:
    """用 seed 参数化的哈希函数哈希一个 shingle（MurmurHash 风格）。"""
    # 使用 MD5 的前 8 字节作为哈希值（简单可复现实现）
    h = hashlib.md5(f"{seed}:{shingle}".encode()).digest()
    return struct.unpack("<Q", h[:8])[0]


def compute_minhash(text: str, num_hashes: int = 128, shingle_n: int = 5) -> np.ndarray:
    """
    计算文档的 MinHash 签名。

    Args:
        text: 文档文本
        num_hashes: 哈希函数数量（签名长度）
        shingle_n: n-gram 大小（字符级）

    Returns:
        np.ndarray，shape=(num_hashes,)，每个元素为最小哈希值
    """
    shingles = _shingle(text, shingle_n)
    if not shingles:
        return np.full(num_hashes, fill_value=2**64 - 1, dtype=np.uint64)

    signature = np.full(num_hashes, fill_value=2**64 - 1, dtype=np.uint64)
    for shingle in shingles:
        for i in range(num_hashes):
            h = _hash_shingle(shingle, seed=i) % (2**32)
            if h < signature[i]:
                signature[i] = h

    return signature


def estimate_jaccard(sig1: np.ndarray, sig2: np.ndarray) -> float:
    """
    用 MinHash 签名估计两个文档的 Jaccard 相似度。
    Jaccard ≈ (签名中相同位置值相等的比例)
    """
    if len(sig1) != len(sig2):
        raise ValueError("签名长度必须相同")
    return float(np.mean(sig1 == sig2))


# ── LSH 实现 ──────────────────────────────────────────────────

class MinHashLSH:
    """
    MinHash + LSH 文档去重器。

    LSH 原理：
    将 num_hashes 个哈希值分成 num_buckets 个 band，每个 band 有 rows_per_band 行。
    如果两个文档在任一 band 的所有行都完全相同，它们就被放入同一个桶（候选对）。
    只有同桶的文档才进行精确相似度比较，大幅减少比较次数。

    参数关系（近似）：
    threshold ≈ (1/num_buckets)^(1/rows_per_band)
    """

    def __init__(
        self,
        num_hashes: int = 128,
        num_buckets: int = 8,
        threshold: float = 0.8,
        shingle_n: int = 5,
    ):
        """
        Args:
            num_hashes: MinHash 签名长度（越大越精确，越慢）
            num_buckets: LSH 分桶数（越多，阈值越高）
            threshold: Jaccard 相似度阈值（高于此值视为重复）
            shingle_n: Shingle 大小（字符级 n-gram）
        """
        assert num_hashes % num_buckets == 0, "num_hashes 必须是 num_buckets 的整数倍"
        self.num_hashes = num_hashes
        self.num_buckets = num_buckets
        self.rows_per_band = num_hashes // num_buckets
        self.threshold = threshold
        self.shingle_n = shingle_n

        # 桶表：band_id → {band_hash: [doc_indices]}
        self._buckets: Dict[int, Dict[int, List[int]]] = defaultdict(lambda: defaultdict(list))
        self._signatures: List[np.ndarray] = []

    def _get_band_hash(self, signature: np.ndarray, band_id: int) -> int:
        """提取签名的第 band_id 段，并计算该段的哈希。"""
        start = band_id * self.rows_per_band
        end = start + self.rows_per_band
        band = signature[start:end]
        return int(hashlib.md5(band.tobytes()).hexdigest(), 16) % (2**32)

    def add_documents(self, texts: List[str], show_progress: bool = True) -> None:
        """
        计算所有文档的 MinHash 签名，并建立 LSH 索引。
        """
        iterator = enumerate(texts)
        if show_progress:
            iterator = tqdm(iterator, total=len(texts), desc="  MinHash 签名计算")

        self._signatures = []
        for doc_id, text in iterator:
            sig = compute_minhash(text, self.num_hashes, self.shingle_n)
            self._signatures.append(sig)

            # 将文档加入所有 band 的桶
            for band_id in range(self.num_buckets):
                band_hash = self._get_band_hash(sig, band_id)
                self._buckets[band_id][band_hash].append(doc_id)

    def find_duplicates(self) -> List[Tuple[int, int, float]]:
        """
        找出所有候选重复对（同桶文档），并验证 Jaccard 相似度。

        Returns:
            list of (doc_id_1, doc_id_2, jaccard_similarity)
        """
        candidate_pairs: Set[Tuple[int, int]] = set()

        # 找出同桶的文档对
        for band_id, buckets in self._buckets.items():
            for band_hash, doc_ids in buckets.items():
                if len(doc_ids) > 1:
                    for i in range(len(doc_ids)):
                        for j in range(i + 1, len(doc_ids)):
                            pair = (min(doc_ids[i], doc_ids[j]), max(doc_ids[i], doc_ids[j]))
                            candidate_pairs.add(pair)

        # 验证候选对的真实 Jaccard 相似度
        confirmed_pairs = []
        for doc_id_1, doc_id_2 in candidate_pairs:
            jaccard = estimate_jaccard(self._signatures[doc_id_1], self._signatures[doc_id_2])
            if jaccard >= self.threshold:
                confirmed_pairs.append((doc_id_1, doc_id_2, jaccard))

        return confirmed_pairs

    def dedup(self, docs: List[Dict], text_field: str = "text") -> Tuple[List[Dict], Dict]:
        """
        执行完整的 MinHash 去重流程。

        Returns:
            (deduplicated_docs, stats)
        """
        print(f"  🔄 MinHash 去重: {len(docs):,} 条文档")
        print(f"     num_hashes={self.num_hashes}, num_buckets={self.num_buckets}, "
              f"threshold={self.threshold}")

        texts = [d.get(text_field, "") for d in docs]

        # 建立索引
        print("  建立 MinHash LSH 索引...")
        self.add_documents(texts)

        # 找重复对
        print("  查找候选重复对...")
        duplicate_pairs = self.find_duplicates()
        print(f"  找到 {len(duplicate_pairs)} 对相似文档 (Jaccard >= {self.threshold})")

        # Union-Find 聚类：将重复文档合并为组
        parent = list(range(len(docs)))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[rx] = ry

        for doc_id_1, doc_id_2, _ in duplicate_pairs:
            union(doc_id_1, doc_id_2)

        # 每组保留第一个文档
        seen_roots: Set[int] = set()
        kept_docs = []
        for i, doc in enumerate(docs):
            root = find(i)
            if root not in seen_roots:
                seen_roots.add(root)
                kept_docs.append(doc)

        n_removed = len(docs) - len(kept_docs)
        stats = {
            "total_input": len(docs),
            "unique_kept": len(kept_docs),
            "near_duplicates_removed": n_removed,
            "dedup_rate": round(n_removed / len(docs), 4) if docs else 0,
            "candidate_pairs": len(duplicate_pairs),
            "jaccard_threshold": self.threshold,
            "parameters": {
                "num_hashes": self.num_hashes,
                "num_buckets": self.num_buckets,
                "rows_per_band": self.rows_per_band,
                "shingle_n": self.shingle_n,
            },
        }

        print(f"  ✅ MinHash 去重: {len(docs):,} → {len(kept_docs):,} 条 | "
              f"去除 {n_removed:,} 条近似重复 ({stats['dedup_rate']:.1%})")
        return kept_docs, stats
