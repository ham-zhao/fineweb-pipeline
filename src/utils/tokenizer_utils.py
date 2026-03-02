"""
src/utils/tokenizer_utils.py
Tokenizer 工具：统计 token 数量、tokenize 文档（用于 Proxy Model 训练）。
使用 tiktoken（GPT-2 BPE tokenizer），无需额外模型文件。
"""

import tiktoken
import numpy as np
from pathlib import Path
from typing import Iterator, List, Optional
from tqdm import tqdm


def get_tokenizer(encoding: str = "gpt2"):
    """获取 tiktoken tokenizer（GPT-2 BPE）。"""
    return tiktoken.get_encoding(encoding)


def count_tokens(text: str, tokenizer=None) -> int:
    """统计单条文本的 token 数量。"""
    if tokenizer is None:
        tokenizer = get_tokenizer()
    return len(tokenizer.encode(text, disallowed_special=()))


def count_tokens_batch(texts: List[str], tokenizer=None) -> List[int]:
    """批量统计 token 数量（比逐条快）。"""
    if tokenizer is None:
        tokenizer = get_tokenizer()
    return [len(enc) for enc in tokenizer.encode_batch(texts, disallowed_special=())]


def estimate_total_tokens(jsonl_path: Path, text_field: str = "text", sample_size: int = 1000) -> int:
    """
    估算 JSONL 文件的总 token 数（采样估算，不读全部数据）。

    Args:
        jsonl_path: JSONL 文件路径
        text_field: 文本字段名
        sample_size: 采样条数

    Returns:
        估算的总 token 数
    """
    import json

    tokenizer = get_tokenizer()
    samples = []
    total_docs = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            total_docs += 1
            if total_docs <= sample_size:
                try:
                    doc = json.loads(line)
                    samples.append(doc.get(text_field, ""))
                except json.JSONDecodeError:
                    pass

    if not samples:
        return 0

    sample_tokens = count_tokens_batch(samples, tokenizer)
    avg_tokens = np.mean(sample_tokens)
    return int(avg_tokens * total_docs)


def tokenize_to_binary(
    jsonl_path: Path,
    output_path: Path,
    text_field: str = "text",
    max_seq_len: int = 1024,
    show_progress: bool = True,
) -> dict:
    """
    将 JSONL 文件 tokenize 并保存为二进制 numpy 格式（用于 Proxy Model 训练）。

    格式：uint16 numpy 数组，连续存储所有 token（用 EOT 分隔文档）。

    Args:
        jsonl_path: 输入 JSONL 文件
        output_path: 输出 .npy 文件
        text_field: 文本字段名
        max_seq_len: 单文档最大 token 数（超出截断）
        show_progress: 是否显示进度条

    Returns:
        统计信息字典
    """
    import json

    tokenizer = get_tokenizer()
    eot_token = tokenizer.eot_token  # GPT-2 的 end-of-text token

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_tokens = []
    total_docs = 0
    total_tokens = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    iterator = tqdm(lines, desc="  Tokenizing") if show_progress else lines

    for line in iterator:
        try:
            doc = json.loads(line)
            text = doc.get(text_field, "")
            if not text:
                continue

            tokens = tokenizer.encode(text, disallowed_special=())[:max_seq_len]
            all_tokens.extend(tokens)
            all_tokens.append(eot_token)
            total_docs += 1
            total_tokens += len(tokens)
        except (json.JSONDecodeError, Exception):
            pass

    # 保存为 uint16（GPT-2 词表大小 50257 < 65535）
    arr = np.array(all_tokens, dtype=np.uint16)
    np.save(output_path, arr)

    stats = {
        "total_docs": total_docs,
        "total_tokens": total_tokens,
        "output_file": str(output_path),
        "file_size_mb": output_path.stat().st_size / 1e6,
    }
    print(f"  ✅ Tokenize 完成: {total_docs:,} 文档, {total_tokens:,} tokens → {output_path}")
    return stats
