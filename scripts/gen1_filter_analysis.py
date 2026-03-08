#!/usr/bin/env python3
"""
scripts/gen1_filter_analysis.py
Gen1 过滤器详细分析脚本

对原始数据逐文档运行各个过滤器，记录每条文档被哪个过滤器拦截，
保存详细分析结果供 NB02 读取。

产出文件：
  data/gen1_output/{mode}/gen1_filter_analysis.json
    - per_filter_examples: 每个过滤器的被过滤文档样例（5条/过滤器）
    - sub_filter_stats: 子过滤器级别的统计
    - language_distribution: 语言分布详情

用法:
  python scripts/gen1_filter_analysis.py
  python scripts/gen1_filter_analysis.py --run-mode full_run
"""

import sys
import json
import random
import re
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config_loader import load_run_config, get_output_path
from src.gen1.filters.url_filter import URLFilter
from src.gen1.filters.language_filter import LanguageFilter
from src.gen1.filters.quality_filter import QualityFilter, GopherQualityFilter, C4QualityFilter, FineWebQualityFilter
from src.gen1.filters.repetition_filter import GopherRepetitionFilter
from src.utils.io import read_jsonl


def clean_text_for_json(text: str) -> str:
    """Remove surrogate characters and truncate for JSON serialization."""
    text = re.sub(r'[\ud800-\udfff]', '', text)
    return text[:500].replace('\n', ' ')


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-mode", default=None)
    args = parser.parse_args()

    run_cfg = load_run_config(run_mode_override=args.run_mode)
    mode = run_cfg.get("run_mode", "smoke_test")
    doc_limit = run_cfg["doc_limit"]

    print(f"=== Gen1 Filter Analysis ({mode}, {doc_limit:,} docs) ===")

    # Load raw data
    if doc_limit > 12000:
        input_path = Path("data/raw/cc_wet_full.jsonl")
    else:
        input_path = Path("data/raw/cc_wet_sample.jsonl")

    docs = read_jsonl(input_path, doc_limit=doc_limit)
    print(f"Loaded {len(docs):,} docs from {input_path.name}")

    random.seed(42)

    # Initialize filters with same config as pipeline
    from src.utils.config_loader import load_pipeline_config
    pipe_cfg = load_pipeline_config(1)
    filter_cfg = pipe_cfg.get("filters", {})

    url_filter = URLFilter()
    lang_filter = LanguageFilter(
        target_language=filter_cfg.get("language_filter", {}).get("target_language", "en"),
        min_confidence=filter_cfg.get("language_filter", {}).get("min_confidence", 0.65),
    )

    goph_cfg = filter_cfg.get("gopher_quality", {})
    c4_cfg = filter_cfg.get("c4_quality", {})
    fw_cfg = filter_cfg.get("fineweb_quality", {})

    gopher_kwargs = {k: v for k, v in goph_cfg.items() if k != "enabled"}
    c4_kwargs = {}
    for key in ["min_lines", "min_words_per_line", "filter_javascript",
                 "filter_lorem_ipsum", "terminal_punct_min_ratio"]:
        if key in c4_cfg:
            c4_kwargs[key] = c4_cfg[key]
    fineweb_kwargs = {}
    if "max_lines_starting_with_bullet" in fw_cfg:
        fineweb_kwargs["max_bullet_lines_ratio"] = fw_cfg["max_lines_starting_with_bullet"]
    if "max_lines_ending_with_ellipsis" in fw_cfg:
        fineweb_kwargs["max_ellipsis_lines_ratio"] = fw_cfg["max_lines_ending_with_ellipsis"]
    if "min_alpha_words_ratio" in fw_cfg:
        fineweb_kwargs["min_alpha_words_ratio"] = fw_cfg["min_alpha_words_ratio"]

    gopher_filter = GopherQualityFilter(**gopher_kwargs)
    c4_filter = C4QualityFilter(**c4_kwargs)
    fineweb_filter = FineWebQualityFilter(**fineweb_kwargs)
    repetition_filter = GopherRepetitionFilter()

    # --- Step-by-step analysis (mirroring pipeline order) ---
    # We run each filter on the docs that would reach it (sequential cascade)
    analysis = {
        "mode": mode,
        "doc_limit": doc_limit,
        "total_input": len(docs),
    }

    # Track per-filter examples
    per_filter_examples = {}
    remaining = docs[:]

    # 1. URL Filter
    print("  Analyzing URL filter...")
    url_kept, url_filtered = [], []
    url_reasons = []
    for doc in remaining:
        should_filter, reason = url_filter.should_filter(doc.get("url", ""))
        if should_filter:
            url_filtered.append(doc)
            url_reasons.append(reason)
        else:
            url_kept.append(doc)

    url_reason_counts = Counter(r.split(":")[0] for r in url_reasons)
    url_detail_counts = Counter(url_reasons)
    per_filter_examples["url_filter"] = []
    for i, (doc, reason) in enumerate(zip(url_filtered, url_reasons)):
        if i >= 5:
            break
        per_filter_examples["url_filter"].append({
            "url": doc.get("url", "")[:120],
            "reason": reason,
            "text_preview": clean_text_for_json(doc.get("text", "")),
        })
    analysis["url_filter"] = {
        "input": len(remaining),
        "filtered": len(url_filtered),
        "output": len(url_kept),
        "reason_breakdown": dict(url_reason_counts.most_common()),
        "detail_breakdown": dict(url_detail_counts.most_common(30)),
    }
    remaining = url_kept

    # 2. Language Filter
    print("  Analyzing language filter...")
    lang_kept, lang_filtered = [], []
    lang_reasons = []
    lang_detected = []
    for doc in remaining:
        should_filter, lang, conf = lang_filter.should_filter(doc["text"])
        lang_detected.append((lang, conf))
        if should_filter:
            lang_filtered.append(doc)
            lang_reasons.append(f"{lang}(conf={conf:.2f})")
        else:
            lang_kept.append(doc)

    # Language distribution
    all_langs = Counter(l[0] for l in lang_detected)
    per_filter_examples["language_filter"] = []
    seen_langs = set()
    for doc, reason in zip(lang_filtered, lang_reasons):
        lang_code = reason.split("(")[0]
        if lang_code not in seen_langs and len(per_filter_examples["language_filter"]) < 5:
            seen_langs.add(lang_code)
            per_filter_examples["language_filter"].append({
                "url": doc.get("url", "")[:120],
                "reason": f"lang:{reason}",
                "text_preview": clean_text_for_json(doc.get("text", "")),
                "detected_lang": lang_code,
            })

    analysis["language_filter"] = {
        "input": len(remaining),
        "filtered": len(lang_filtered),
        "output": len(lang_kept),
        "language_distribution": dict(all_langs.most_common(20)),
        "english_count": all_langs.get("en", 0),
        "english_ratio": all_langs.get("en", 0) / sum(all_langs.values()) if all_langs else 0,
    }
    remaining = lang_kept

    # 3. Quality Filter (Gopher + C4 + FineWeb — separately analyzed)
    print("  Analyzing quality filter (Gopher/C4/FineWeb separately)...")
    qual_kept = []
    gopher_filtered, c4_filtered, fineweb_filtered = [], [], []
    gopher_reasons, c4_reasons, fineweb_reasons = [], [], []

    for doc in remaining:
        text = doc["text"]
        # Gopher check first
        passes, reason = gopher_filter.check(text)
        if not passes:
            gopher_filtered.append(doc)
            gopher_reasons.append(reason)
            continue
        # C4 check
        passes, reason = c4_filter.check(text)
        if not passes:
            c4_filtered.append(doc)
            c4_reasons.append(reason)
            continue
        # FineWeb check
        passes, reason = fineweb_filter.check(text)
        if not passes:
            fineweb_filtered.append(doc)
            fineweb_reasons.append(reason)
            continue
        qual_kept.append(doc)

    # Sub-filter stats
    gopher_reason_counts = Counter(r.split(":")[0] for r in gopher_reasons)
    c4_reason_counts = Counter(r.split(":")[0] for r in c4_reasons)
    fineweb_reason_counts = Counter(r.split(":")[0] for r in fineweb_reasons)

    # Examples per sub-filter
    for name, filtered_list, reasons_list in [
        ("gopher_quality", gopher_filtered, gopher_reasons),
        ("c4_quality", c4_filtered, c4_reasons),
        ("fineweb_quality", fineweb_filtered, fineweb_reasons),
    ]:
        per_filter_examples[name] = []
        seen_sub_reasons = set()
        for doc, reason in zip(filtered_list, reasons_list):
            sub_reason = reason.split(":")[0]
            if sub_reason not in seen_sub_reasons and len(per_filter_examples[name]) < 5:
                seen_sub_reasons.add(sub_reason)
                per_filter_examples[name].append({
                    "url": doc.get("url", "")[:120],
                    "reason": reason,
                    "text_preview": clean_text_for_json(doc.get("text", "")),
                })

    total_quality_filtered = len(gopher_filtered) + len(c4_filtered) + len(fineweb_filtered)
    analysis["quality_filter"] = {
        "input": len(remaining) + total_quality_filtered,
        "filtered": total_quality_filtered,
        "output": len(qual_kept),
        "sub_filters": {
            "gopher": {
                "filtered": len(gopher_filtered),
                "reason_breakdown": dict(gopher_reason_counts.most_common()),
            },
            "c4": {
                "filtered": len(c4_filtered),
                "reason_breakdown": dict(c4_reason_counts.most_common()),
            },
            "fineweb": {
                "filtered": len(fineweb_filtered),
                "reason_breakdown": dict(fineweb_reason_counts.most_common()),
            },
        },
    }
    remaining = qual_kept

    # 4. Repetition Filter
    print("  Analyzing repetition filter...")
    rep_kept, rep_filtered = [], []
    rep_reasons = []
    for doc in remaining:
        passes, reason = repetition_filter.check(doc["text"])
        if not passes:
            rep_filtered.append(doc)
            rep_reasons.append(reason)
        else:
            rep_kept.append(doc)

    rep_reason_counts = Counter(r.split(":")[0] for r in rep_reasons)
    per_filter_examples["repetition_filter"] = []
    seen_rep_reasons = set()
    for doc, reason in zip(rep_filtered, rep_reasons):
        sub_reason = reason.split(":")[0]
        if sub_reason not in seen_rep_reasons and len(per_filter_examples["repetition_filter"]) < 5:
            seen_rep_reasons.add(sub_reason)
            per_filter_examples["repetition_filter"].append({
                "url": doc.get("url", "")[:120],
                "reason": reason,
                "text_preview": clean_text_for_json(doc.get("text", "")),
            })

    analysis["repetition_filter"] = {
        "input": len(remaining) + len(rep_filtered),
        "filtered": len(rep_filtered),
        "output": len(rep_kept),
        "reason_breakdown": dict(rep_reason_counts.most_common()),
    }
    remaining = rep_kept

    analysis["final_output"] = len(remaining)
    analysis["per_filter_examples"] = per_filter_examples

    # Save
    output_dir = get_output_path(1, run_cfg)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "gen1_filter_analysis.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)

    print(f"\nSaved: {output_file}")
    print(f"  URL filter: {analysis['url_filter']['filtered']:,} filtered")
    print(f"  Language filter: {analysis['language_filter']['filtered']:,} filtered")
    print(f"  Quality filter: {analysis['quality_filter']['filtered']:,} filtered")
    print(f"    Gopher: {analysis['quality_filter']['sub_filters']['gopher']['filtered']:,}")
    print(f"    C4: {analysis['quality_filter']['sub_filters']['c4']['filtered']:,}")
    print(f"    FineWeb: {analysis['quality_filter']['sub_filters']['fineweb']['filtered']:,}")
    print(f"  Repetition filter: {analysis['repetition_filter']['filtered']:,} filtered")
    print(f"  Final output: {analysis['final_output']:,}")


if __name__ == "__main__":
    main()
