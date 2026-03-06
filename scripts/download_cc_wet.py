#!/usr/bin/env python3
"""
scripts/download_cc_wet.py
从 Common Crawl WET 格式下载多 segment 随机采样数据

产出文件：
  data/raw/cc_wet_sample.jsonl - 采样后的文档（每行一条 JSON）

用法:
    python scripts/download_cc_wet.py --count 12000 --segments 10
    python scripts/download_cc_wet.py --count 3000 --segments 5

防休眠: caffeinate -i python scripts/download_cc_wet.py
"""

import sys
import os
import gzip
import json
import random
import argparse
import io
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError

sys.path.insert(0, str(Path(__file__).parent.parent))

CC_DUMP = "CC-MAIN-2024-10"
WET_PATHS_URL = f"https://data.commoncrawl.org/crawl-data/{CC_DUMP}/wet.paths.gz"
CC_BASE_URL = "https://data.commoncrawl.org/"


def parse_args():
    parser = argparse.ArgumentParser(description="Download CC WET sample data")
    parser.add_argument("--count", type=int, default=12000, help="Target document count")
    parser.add_argument("--segments", type=int, default=10, help="Number of segments to sample from")
    parser.add_argument("--output", default="data/raw/cc_wet_sample.jsonl", help="Output JSONL path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--min-text-len", type=int, default=50, help="Minimum text length in chars")
    parser.add_argument("--workers", type=int, default=5, help="Parallel download workers")
    return parser.parse_args()


def download_wet_paths(seed: int, num_segments: int) -> list:
    """Download and parse wet.paths.gz, return randomly selected segment paths."""
    print(f"  Downloading WET paths index: {WET_PATHS_URL}")
    req = Request(WET_PATHS_URL, headers={"User-Agent": "FineWebPipeline/1.0"})
    try:
        with urlopen(req, timeout=30) as resp:
            compressed = resp.read()
    except (HTTPError, URLError) as e:
        print(f"  ERROR: Failed to download paths index: {e}")
        sys.exit(1)

    paths = gzip.decompress(compressed).decode("utf-8").strip().split("\n")
    paths = [p.strip() for p in paths if p.strip() and p.strip().endswith(".warc.wet.gz")]
    print(f"  Found {len(paths):,} WET segments")

    rng = random.Random(seed)
    selected = rng.sample(paths, min(num_segments, len(paths)))
    print(f"  Selected {len(selected)} segments for sampling")
    return selected


def download_segment_docs(segment_path: str, target_count: int, min_text_len: int) -> list:
    """Stream-download one WET segment and extract documents."""
    from warcio.archiveiterator import ArchiveIterator

    url = CC_BASE_URL + segment_path
    segment_id = segment_path.split("/")[-1].split(".")[0]

    try:
        req = Request(url, headers={"User-Agent": "FineWebPipeline/1.0"})
        resp = urlopen(req, timeout=60)
    except (HTTPError, URLError) as e:
        print(f"    SKIP {segment_id}: {e}")
        return []

    docs = []
    try:
        for record in ArchiveIterator(resp):
            if record.rec_type != "conversion":
                continue
            warc_url = record.rec_headers.get_header("WARC-Target-URI", "")
            try:
                text = record.content_stream().read().decode("utf-8", errors="replace").strip()
            except Exception:
                continue

            if len(text) >= min_text_len:
                docs.append({
                    "text": text,
                    "url": warc_url,
                    "source": "common_crawl_wet",
                    "segment": segment_id,
                })

            if len(docs) >= target_count:
                break
    except Exception as e:
        print(f"    WARN {segment_id}: Stream error after {len(docs)} docs: {e}")
    finally:
        try:
            resp.close()
        except Exception:
            pass

    return docs


def main():
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  CC WET Download: {args.count:,} docs from {args.segments} segments")
    print(f"  Dump: {CC_DUMP}")
    print(f"{'='*60}")

    # Step 1: Get segment paths
    segment_paths = download_wet_paths(args.seed, args.segments)

    # Step 2: Download from each segment (parallel)
    from concurrent.futures import ThreadPoolExecutor, as_completed

    per_segment = args.count // len(segment_paths) + 1
    all_docs = []
    failed_segments = 0
    workers = min(args.workers, len(segment_paths))

    print(f"  Downloading with {workers} parallel workers...")

    def _download_one(i_seg):
        i, seg_path = i_seg
        docs = download_segment_docs(seg_path, per_segment, args.min_text_len)
        return i, seg_path, docs

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_download_one, (i, sp)): i for i, sp in enumerate(segment_paths)}
        for future in as_completed(futures):
            i, seg_path, docs = future.result()
            if docs:
                all_docs.extend(docs)
                print(f"    [{i+1}/{len(segment_paths)}] Got {len(docs):,} docs (total: {len(all_docs):,})")
            else:
                failed_segments += 1
                print(f"    [{i+1}/{len(segment_paths)}] FAILED (total failures: {failed_segments})")

    # Step 3: Shuffle and deduplicate by URL
    rng = random.Random(args.seed)
    rng.shuffle(all_docs)

    seen_urls = set()
    deduped = []
    for doc in all_docs:
        url_key = doc.get("url", "").strip().lower()
        if url_key not in seen_urls:
            seen_urls.add(url_key)
            deduped.append(doc)

    # Trim to target count
    final_docs = deduped[:args.count]

    # Step 4: Save
    with open(output_path, "w", encoding="utf-8") as f:
        for doc in final_docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    # Step 5: Stats
    segments_used = len(set(d.get("segment", "") for d in final_docs))
    domains = len(set(d.get("url", "").split("/")[2] if "/" in d.get("url", "") and len(d.get("url", "").split("/")) > 2 else "" for d in final_docs))
    avg_len = sum(len(d["text"]) for d in final_docs) / len(final_docs) if final_docs else 0

    print(f"\n{'='*60}")
    print(f"  Download complete!")
    print(f"  Output: {output_path} ({len(final_docs):,} docs)")
    print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"  Segments used: {segments_used}")
    print(f"  Unique domains: {domains:,}")
    print(f"  Avg text length: {avg_len:,.0f} chars")
    print(f"  Failed segments: {failed_segments}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
