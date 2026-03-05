"""
src/utils/downloader.py
数据下载工具：Common Crawl WARC、FineWeb parquet、参考数据集。
"""

import os
import requests
import gzip
import shutil
from pathlib import Path
from typing import Optional
from tqdm import tqdm


def download_file(url: str, dest_path: Path, desc: str = "", chunk_size: int = 1024 * 1024) -> Path:
    """
    带进度条的文件下载。

    Args:
        url: 下载 URL
        dest_path: 保存路径（含文件名）
        desc: 进度条描述
        chunk_size: 每次读取的字节数（默认 1MB）

    Returns:
        保存后的文件路径
    """
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if dest_path.exists():
        print(f"  ⏭️  文件已存在，跳过下载: {dest_path.name}")
        return dest_path

    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    desc = desc or dest_path.name

    with open(dest_path, "wb") as f, tqdm(
        total=total,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=f"  ⬇️  {desc}",
    ) as bar:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            bar.update(len(chunk))

    print(f"  ✅ 下载完成: {dest_path} ({dest_path.stat().st_size / 1e6:.1f} MB)")
    return dest_path


def get_common_crawl_warc_url(
    crawl_id: str = "CC-MAIN-2024-10", segment: int = 0
) -> str:
    """
    获取 Common Crawl WARC 文件的 S3 URL。

    Args:
        crawl_id: CC dump ID，例如 "CC-MAIN-2024-10"
        segment: WARC 分段编号（0 表示第一个文件）

    Returns:
        WARC 文件的 S3 HTTP URL
    """
    # CC 的公开 paths.gz 文件列出了所有 WARC 路径
    # 实际项目中应先下载 paths.gz 再选择具体文件
    # 这里使用已知的一个样本 WARC 路径
    base = "https://data.commoncrawl.org"
    # 使用 CC-MAIN-2024-10 的第一个 WARC 文件
    path = f"crawl-data/{crawl_id}/segments/1707947473347.0/warc/CC-MAIN-20240220211055-20240221001055-00000.warc.gz"
    return f"{base}/{path}"


def download_warc_sample(dest_dir: Path, crawl_id: str = "CC-MAIN-2024-10") -> Path:
    """
    下载一个 WARC 样本文件（约 1-1.5GB 压缩）。

    Returns:
        下载的 WARC 文件路径
    """
    url = get_common_crawl_warc_url(crawl_id)
    filename = url.split("/")[-1]
    dest_path = Path(dest_dir) / filename

    print(f"📥 下载 Common Crawl WARC 文件")
    print(f"   Crawl: {crawl_id}")
    print(f"   URL: {url}")
    return download_file(url, dest_path, desc="CC WARC")


def download_fineweb_sample(dest_dir: Path, num_shards: int = 1) -> list[Path]:
    """
    从 HuggingFace 下载 FineWeb sample-10BT 的 parquet 分片。

    Args:
        dest_dir: 保存目录
        num_shards: 下载的分片数量（每片约 500MB）

    Returns:
        下载的文件路径列表
    """
    # FineWeb sample-10BT 的 HF 数据集文件
    base_url = "https://huggingface.co/datasets/HuggingFaceFW/fineweb/resolve/main/data/CC-MAIN-2024-10"
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for i in range(num_shards):
        filename = f"train-{i:05d}-of-00099.parquet"
        url = f"{base_url}/{filename}"
        dest_path = dest_dir / filename
        paths.append(download_file(url, dest_path, desc=f"FineWeb shard {i}"))

    return paths


def download_wikipedia_abstracts(dest_dir: Path, max_docs: int = 10000) -> Path:
    """
    下载 Wikipedia 摘要（正样本参考数据）。
    使用 HuggingFace wikipedia 数据集。

    Returns:
        保存的 JSONL 文件路径
    """
    import json
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    output_path = dest_dir / "wikipedia_abstracts.jsonl"

    if output_path.exists():
        print(f"  ⏭️  Wikipedia 摘要已存在: {output_path}")
        return output_path

    print(f"📥 下载 Wikipedia 摘要（前 {max_docs:,} 条）...")
    from datasets import load_dataset

    dataset = load_dataset(
        "wikimedia/wikipedia",
        "20231101.en",
        split="train",
        streaming=True,
    )

    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for doc in tqdm(dataset, total=max_docs, desc="  📖 Wikipedia"):
            # 只取摘要（第一段，以空行为分隔）
            text = doc["text"].split("\n\n")[0].strip()
            if len(text) > 100:  # 过滤过短的摘要
                f.write(json.dumps({"text": text, "source": "wikipedia", "title": doc["title"]}, ensure_ascii=False) + "\n")
                count += 1
            if count >= max_docs:
                break

    print(f"  ✅ Wikipedia 摘要: {count:,} 条 → {output_path}")
    return output_path
