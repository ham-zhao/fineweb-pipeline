"""
src/gen1_zh/chinese_text_utils.py
中文文本基础工具函数

设计原则：
  - 纯 Python 实现，不依赖 jieba/pkuseg 等分词库（轻量化）
  - 中文分析基于 Unicode 字符属性判断，无需网络或模型文件
  - 简繁体均支持，不做强制转换（保留原始字符）

关键函数：
  count_chinese_chars(text)      → int, 汉字字符数（CJK 统一汉字范围）
  char_type_ratio(text)          → dict, 中文/英文/数字/标点/其他各占比
  detect_script(text)            → "simplified" | "traditional" | "mixed" | "unknown"
  tokenize_by_char(text)         → List[str], 字符级分词（用于 n-gram 分析）
  compute_zh_ngrams(text, n)     → Counter, 字符级 n-gram 频率
  estimate_zh_tokens(text)       → int, 粗略估算 token 数（汉字≈1.5 tokens/字）
"""

import re
import unicodedata
from typing import Dict, List
from collections import Counter


# ── Unicode 范围常量 ──────────────────────────────────────────────

# CJK 统一汉字（含扩展 A/B/C/D/E/F）
_CJK_RANGES = [
    (0x4E00, 0x9FFF),    # CJK Unified Ideographs 基本区
    (0x3400, 0x4DBF),    # Extension A
    (0x20000, 0x2A6DF),  # Extension B
    (0x2A700, 0x2B73F),  # Extension C
    (0x2B740, 0x2B81F),  # Extension D
    (0x2B820, 0x2CEAF),  # Extension E
    (0x2CEB0, 0x2EBEF),  # Extension F
    (0xF900, 0xFAFF),    # CJK Compatibility Ideographs
]

# 繁体字高频字（判断是否为繁体文本的启发式特征）
_TRADITIONAL_CHARS = set(
    "們來時這個說國來後會對應也從於來他們來時後她後學"
    "現發現問題與為說來從這後學對話後來發現問題"
    # 高频繁体字
    "體樣進過點還實際於時從發與這說對國後學應現"
    "來後發現問題與為說來從這後學對話後來發現問題"
    "後來發現問題與為說來從這後學對話後來發現問題"
    "實際時間與為後發學應會現問題點體樣進過還"
    # 繁体特征字（与简体区别最明显）
    "愛藝術電話語書報紙時間問題學習工作生活"
    "歷史國際關係政治經濟文化社會發展變化"
)

# 简体字高频字（判断是否为简体文本）
_SIMPLIFIED_CHARS = set(
    "们来时这个说国来后会对应也从于来他们来时后她后学"
    "现发现问题与为说来从这后学对话后来发现问题"
    # 高频简体字
    "实际时间与为后发学应会现问题点体样进过还"
    "爱艺术电话语书报纸时间问题学习工作生活"
    "历史国际关系政治经济文化社会发展变化"
)

# 中文标点
_ZH_PUNCTUATION = set("，。！？；：「」『』【】〔〕〈〉《》〖〗、·…——～""''（）")

# 垃圾内容常见模式（正则）
_SPAM_PATTERNS = [
    re.compile(r"[\u2605\u2606]{3,}"),              # ★★★ 评分刷榜
    re.compile(r"(点击|立即|马上|快速)(下载|购买|领取|注册|登录){1}", re.U),  # 诱导点击
    re.compile(r"(免费|优惠|折扣|特价).{0,5}(领取|获取|下载)"),             # 促销诱导
    re.compile(r"(加微信|加QQ|加我|扫码).{0,10}(咨询|了解|获取)"),          # 引流话术
    re.compile(r"[0-9]{11}"),                        # 手机号（11位连续数字）
    re.compile(r"(http|https|www)[^\s]{3,50}"),      # URL（额外计数）
    re.compile(r"(.)\1{4,}"),                        # 重复字符 ≥5 次（如"哈哈哈哈哈"）
    re.compile(r"(转发|分享|点赞).{0,5}(抽奖|免费|赢)"),                    # 转发抽奖
    re.compile(r"SEO|关键词优化|百度排名|搜索引擎优化", re.I),              # SEO 话术
]


# ── 基础字符分析 ──────────────────────────────────────────────────

def _is_cjk(char: str) -> bool:
    """判断单个字符是否为 CJK 汉字。"""
    cp = ord(char)
    return any(lo <= cp <= hi for lo, hi in _CJK_RANGES)


def count_chinese_chars(text: str) -> int:
    """统计文本中 CJK 汉字字符数。"""
    return sum(1 for c in text if _is_cjk(c))


def char_type_ratio(text: str) -> Dict[str, float]:
    """
    分析文本字符类型分布。

    Returns:
        dict with keys: chinese, latin, digit, punctuation, whitespace, other
        values are ratios (0~1), sum ≈ 1.0
    """
    if not text:
        return {"chinese": 0, "latin": 0, "digit": 0, "punctuation": 0, "whitespace": 0, "other": 0}

    counts = {"chinese": 0, "latin": 0, "digit": 0, "punctuation": 0, "whitespace": 0, "other": 0}
    for c in text:
        if _is_cjk(c):
            counts["chinese"] += 1
        elif c.isalpha():
            counts["latin"] += 1
        elif c.isdigit():
            counts["digit"] += 1
        elif unicodedata.category(c).startswith("P") or c in _ZH_PUNCTUATION:
            counts["punctuation"] += 1
        elif c.isspace():
            counts["whitespace"] += 1
        else:
            counts["other"] += 1

    total = len(text)
    return {k: round(v / total, 4) for k, v in counts.items()}


def detect_script(text: str) -> str:
    """
    粗略判断文本是简体还是繁体中文。

    算法：统计文本中繁/简特征字出现次数，取多数。
    注意：简/繁都有大量相同汉字，此函数仅供参考，准确率约 80-85%。

    Returns:
        "simplified" | "traditional" | "mixed" | "unknown"
    """
    zh_chars = [c for c in text if _is_cjk(c)]
    if not zh_chars:
        return "unknown"

    trad_count = sum(1 for c in zh_chars if c in _TRADITIONAL_CHARS)
    simp_count  = sum(1 for c in zh_chars if c in _SIMPLIFIED_CHARS)

    if trad_count == 0 and simp_count == 0:
        return "unknown"

    total = trad_count + simp_count
    trad_ratio = trad_count / total
    simp_ratio = simp_count / total

    if trad_ratio > 0.65:
        return "traditional"
    if simp_ratio > 0.65:
        return "simplified"
    return "mixed"


# ── 分词与 N-gram ────────────────────────────────────────────────

def tokenize_by_char(text: str, keep_punct: bool = False) -> List[str]:
    """
    字符级分词（中文无空格分隔，字符即基本单位）。
    中文字符单独保留；ASCII 词语合并为词级 token；标点可选。

    Returns:
        list of tokens
    """
    tokens = []
    i = 0
    while i < len(text):
        c = text[i]
        if _is_cjk(c):
            tokens.append(c)
            i += 1
        elif c.isalnum():
            # 连续 ASCII 字母/数字合并
            j = i
            while j < len(text) and text[j].isalnum():
                j += 1
            tokens.append(text[i:j].lower())
            i = j
        elif c.isspace():
            i += 1
        else:
            if keep_punct:
                tokens.append(c)
            i += 1
    return tokens


def compute_zh_ngrams(text: str, n: int = 2) -> Counter:
    """
    计算字符级 n-gram（中文常用 bi-gram/tri-gram 分析重复度）。

    Returns:
        Counter {ngram_string: count}
    """
    chars = [c for c in text if _is_cjk(c)]
    ngrams = ["".join(chars[i:i+n]) for i in range(len(chars) - n + 1)]
    return Counter(ngrams)


# ── Token 数估算 ──────────────────────────────────────────────────

def estimate_zh_tokens(text: str) -> int:
    """
    粗略估算中文文本的 subword token 数（BPE/Unigram 分词假设）。

    经验公式（基于 tiktoken cl100k_base 统计）：
    - 中文字符：约 1.5 tokens/字（偶尔 2 个字合并为 1 token）
    - 英文词：约 1.3 tokens/词
    - 数字：约 1 token/位

    实际精确值需用 tokenizer，这里提供 ±15% 精度的快速估算。
    """
    zh_count  = count_chinese_chars(text)
    ratios    = char_type_ratio(text)
    total_len = len(text)

    latin_chars = int(total_len * ratios["latin"])
    digit_chars = int(total_len * ratios["digit"])

    # 英文词数估算（平均每词 5 字符）
    latin_words = latin_chars / 5

    return int(zh_count * 1.5 + latin_words * 1.3 + digit_chars * 1.0)


# ── 垃圾内容检测 ──────────────────────────────────────────────────

def count_spam_signals(text: str) -> Dict[str, int]:
    """
    统计文本中各类垃圾信号的出现次数。

    Returns:
        dict {pattern_name: count}
    """
    pattern_names = [
        "star_rating_spam", "click_bait", "promo_lure",
        "wechat_drain", "phone_number", "url_count",
        "char_repetition", "share_lucky_draw", "seo_terms"
    ]
    results = {}
    for name, pat in zip(pattern_names, _SPAM_PATTERNS):
        results[name] = len(pat.findall(text))
    return results


def compute_spam_score(text: str) -> float:
    """
    综合垃圾评分（0=干净, 1=高度垃圾）。
    用于快速排序，阈值建议 > 0.3 即过滤。
    """
    if not text:
        return 0.0
    signals = count_spam_signals(text)
    # 加权求和，归一化到 0~1
    weights = {
        "click_bait": 2.0,
        "promo_lure": 2.0,
        "wechat_drain": 3.0,
        "share_lucky_draw": 3.0,
        "seo_terms": 1.5,
        "char_repetition": 1.0,
        "phone_number": 0.5,
        "url_count": 0.3,
        "star_rating_spam": 0.5,
    }
    score = sum(weights.get(k, 1.0) * min(v, 5) for k, v in signals.items())
    return min(score / 20.0, 1.0)
