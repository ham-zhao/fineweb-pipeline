"""
src/utils/io.py
JSONL 读写工具函数（项目通用）。

产出文件：无（工具函数库）
"""

import json
from pathlib import Path
from typing import List, Dict


def read_jsonl(jsonl_path: Path, doc_limit: int = None) -> List[Dict]:
    """读取 JSONL 格式的预处理数据（跳过 WARC 解析步骤）。"""
    docs = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                docs.append(json.loads(line))
                if doc_limit and len(docs) >= doc_limit:
                    break
            except json.JSONDecodeError:
                pass
    return docs


def save_jsonl(docs: List[Dict], output_path: Path, desc: str = "") -> None:
    """保存文档列表为 JSONL 格式。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    print(f"  {'[' + desc + '] ' if desc else ''}已保存 {len(docs):,} 条 -> {output_path}")
