#!/usr/bin/env python3
"""
scripts/run_gen2.py
第二代 Model-based Filtering Pipeline — 独立运行脚本

产出文件：
  data/gen2_output/gen2_output.jsonl        - 过滤后文档
  data/gen2_output/gen2_stats.json          - 统计信息（含分数分布）
  data/gen2_output/gen2_stage_metrics.json   - 评估指标
  data/gen2_output/llm_labels.jsonl          - LLM 标注数据（如启用）

用法:
    python scripts/run_gen2.py                    # 从 Gen1 输出读取
    python scripts/run_gen2.py --from-raw         # 从原始数据开始
    python scripts/run_gen2.py --top-fraction 0.15  # 自定义保留比例
    python scripts/run_gen2.py --use-llm-labels   # 使用 LLM 标注训练分类器

防休眠: caffeinate -i python scripts/run_gen2.py
"""

import sys
import os
import argparse
import json
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config_loader import (
    load_run_config, load_pipeline_config, load_eval_config,
    load_api_config, get_output_path, print_config_summary
)
from src.gen1.pipeline import read_jsonl, read_warc_texts, save_jsonl, Gen1Pipeline
from src.gen2.quality_classifier import Gen2QualityClassifier
from src.gen2.pipeline import Gen2Pipeline
from src.evaluation.stage_tracker import StageTracker


def parse_args():
    parser = argparse.ArgumentParser(description="第二代 Model-based Filtering Pipeline")
    parser.add_argument("--run-config", default="configs/run_config.yaml")
    parser.add_argument("--config", default="configs/gen2_config.yaml")
    parser.add_argument("--from-raw", action="store_true", help="从原始数据开始（跳过 Gen1 输出）")
    parser.add_argument("--top-fraction", type=float, default=0.10, help="保留比例（默认 0.10 即 top-10%%）")
    parser.add_argument("--retrain-classifier", action="store_true", help="强制重新训练分类器")
    parser.add_argument("--use-llm-labels", action="store_true", help="使用 LLM 标注训练分类器（需 API 配置）")
    return parser.parse_args()


def _try_llm_labeling(gen1_docs, run_cfg, gen2_output_dir):
    """尝试 LLM 标注，返回标注结果列表。失败返回空列表。"""
    labels_path = gen2_output_dir / "llm_labels.jsonl"
    if labels_path.exists():
        print(f"  LLM 标注缓存存在: {labels_path}")
        labels = []
        with open(labels_path) as f:
            for line in f:
                try:
                    labels.append(json.loads(line))
                except Exception:
                    pass
        if labels:
            print(f"  加载 {len(labels)} 条已有标注")
            return labels

    try:
        api_cfg = load_api_config()
    except (ValueError, FileNotFoundError) as e:
        print(f"  API 未配置，跳过 LLM 标注: {e}")
        return []

    from src.gen2.llm_labeler import LLMLabeler
    labeler = LLMLabeler(
        api_config=api_cfg,
        concurrency=run_cfg.get("llm_label_concurrency", 5),
    )
    label_count = run_cfg.get("llm_label_count", 100)
    labels, stats = labeler.label_batch(
        gen1_docs,
        sample_count=label_count,
        random_seed=run_cfg.get("random_seed", 42),
    )
    if labels:
        gen2_output_dir.mkdir(parents=True, exist_ok=True)
        labeler.save_labels(labels, labels_path)
    return labels


def load_or_train_classifier(pipe_cfg, run_cfg, gen1_docs=None, gen2_output_dir=None,
                              retrain=False, use_llm=False) -> Gen2QualityClassifier:
    """加载已有分类器，或训练新分类器。支持 LLM 标注模式。"""
    model_path = pipe_cfg.get("classifier", {}).get("fasttext", {}).get(
        "model_path", "results/quality_scores/gen2_classifier.bin"
    )
    clf = Gen2QualityClassifier(model_path=None)

    if Path(model_path).exists() and not retrain:
        print(f"  Load existing Gen2 classifier: {model_path}")
        clf._load(model_path)
        return clf

    # 尝试 LLM 标注模式
    llm_labels = []
    if use_llm and gen1_docs and gen2_output_dir:
        llm_labels = _try_llm_labeling(gen1_docs, run_cfg, gen2_output_dir)

    if llm_labels:
        # LLM 标注模式：score >= 3 为正样本，score <= 1 为负样本
        print(f"  Training with LLM labels ({len(llm_labels)} samples)...")
        positive_texts = [l["text"] for l in llm_labels if l.get("llm_score", 0) >= 3]
        negative_texts = [l["text"] for l in llm_labels if l.get("llm_score", 0) <= 1]
        print(f"     LLM positive (score>=3): {len(positive_texts)} | negative (score<=1): {len(negative_texts)}")

        # 补充不足的样本
        if len(positive_texts) < 50:
            wiki_texts = _load_wiki_texts()
            positive_texts.extend(wiki_texts[:200])
            print(f"     Augmented with Wikipedia: total positive = {len(positive_texts)}")
        if len(negative_texts) < 50:
            raw_texts = _load_raw_texts(run_cfg)
            negative_texts.extend(raw_texts[:200])
            print(f"     Augmented with raw CC: total negative = {len(negative_texts)}")
    else:
        # 降级：Wikipedia 二分类模式
        print(f"  Training with Wikipedia binary classification (no LLM labels)...")
        positive_texts = _load_wiki_texts()
        negative_texts = _load_raw_texts(run_cfg)

    if not positive_texts:
        print("  WARNING: No positive samples, using synthetic data")
        positive_texts = [
            "The mitochondria is the powerhouse of the cell, converting nutrients into ATP.",
            "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
            "The French Revolution began in 1789 with the storming of the Bastille.",
        ] * 100
    if not negative_texts:
        print("  WARNING: No negative samples, using synthetic data")
        negative_texts = ["buy now discount sale cheap free click here"] * 100

    # 平衡正负样本
    min_count = min(len(positive_texts), len(negative_texts), 5000)
    rng = random.Random(run_cfg.get("random_seed", 42))
    positive_texts = rng.sample(positive_texts, min(min_count, len(positive_texts)))
    negative_texts = rng.sample(negative_texts, min(min_count, len(negative_texts)))

    fasttext_cfg = pipe_cfg.get("classifier", {}).get("fasttext", {})
    clf.train(
        positive_texts=positive_texts,
        negative_texts=negative_texts,
        output_path=model_path,
        dim=fasttext_cfg.get("dim", 64),
        wordNgrams=fasttext_cfg.get("wordNgrams", 2),
        lr=fasttext_cfg.get("lr", 0.1),
        epoch=fasttext_cfg.get("epoch", 5),
        minCount=fasttext_cfg.get("minCount", 5),
    )
    return clf


def _load_wiki_texts():
    """加载 Wikipedia 正样本。"""
    texts = []
    for path in [Path("data/reference/wikipedia_abstracts.jsonl"),
                 Path("data/reference/stackexchange_top.jsonl")]:
        if path.exists():
            with open(path) as f:
                for line in f:
                    try:
                        texts.append(json.loads(line)["text"])
                    except Exception:
                        pass
    return texts


def _load_raw_texts(run_cfg):
    """加载原始数据作为负样本。"""
    texts = []
    # 优先 CC WET
    raw_file = Path("data/raw/cc_wet_sample.jsonl")
    if not raw_file.exists():
        raw_files = list(Path("data/raw").glob("*.jsonl"))
        raw_file = raw_files[0] if raw_files else None
    if raw_file and raw_file.exists():
        doc_limit = run_cfg.get("doc_limit", 1000)
        with open(raw_file) as f:
            for i, line in enumerate(f):
                if i >= doc_limit:
                    break
                try:
                    texts.append(json.loads(line).get("text", ""))
                except Exception:
                    pass
    return texts


def main():
    args = parse_args()

    run_cfg = load_run_config(args.run_config)
    pipe_cfg = load_pipeline_config(2, args.config)

    print_config_summary(run_cfg)

    doc_limit = run_cfg.get("doc_limit")
    gen1_output_dir = get_output_path(1, run_cfg)
    gen2_output_dir = get_output_path(2, run_cfg)

    # ── 加载输入数据 ──────────────────────────────────────────
    gen1_file = gen1_output_dir / "gen1_output.jsonl"
    if not args.from_raw and gen1_file.exists():
        print(f"\nLoad Gen1 output: {gen1_file}")
        docs = read_jsonl(gen1_file, doc_limit=doc_limit)
    else:
        print(f"\nGen1 output not found, loading raw data...")
        cc_wet = Path("data/raw/cc_wet_sample.jsonl")
        raw_files = list(Path("data/raw").glob("*.warc.gz")) + list(Path("data/raw").glob("*.jsonl"))
        if cc_wet.exists():
            input_path = cc_wet
        elif raw_files:
            input_path = raw_files[0]
        else:
            print("ERROR: No input data found!")
            sys.exit(1)
        if input_path.suffix in (".gz",):
            docs = read_warc_texts(input_path, doc_limit=doc_limit)
        else:
            docs = read_jsonl(input_path, doc_limit=doc_limit)

    print(f"Loaded {len(docs):,} documents")

    # ── 训练/加载分类器 ───────────────────────────────────────
    classifier = load_or_train_classifier(
        pipe_cfg, run_cfg,
        gen1_docs=docs,
        gen2_output_dir=gen2_output_dir,
        retrain=args.retrain_classifier,
        use_llm=args.use_llm_labels,
    )

    # ── 运行 Pipeline ─────────────────────────────────────────
    tracker = StageTracker(load_eval_config(), run_cfg)

    pipeline = Gen2Pipeline(
        run_config=run_cfg,
        pipeline_config=pipe_cfg,
        classifier=classifier,
        stage_tracker=tracker,
    )

    result = pipeline.run(docs, top_fraction=args.top_fraction)

    # ── 保存输出 ─────────────────────────────────────────────
    gen2_output_dir.mkdir(parents=True, exist_ok=True)
    output_file = gen2_output_dir / "gen2_output.jsonl"
    pipeline.save_with_scores(result, output_file)

    stats_file = gen2_output_dir / "gen2_stats.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        import numpy as np
        stats_safe = dict(result["stats"])
        all_scores = result["all_scores"]
        hist_counts, hist_edges = np.histogram(all_scores, bins=50)
        stats_safe["score_histogram"] = {
            "counts": hist_counts.tolist(),
            "bin_edges": hist_edges.tolist(),
        }
        stats_safe["all_scores"] = [round(float(s), 4) for s in all_scores]
        json.dump(stats_safe, f, ensure_ascii=False, indent=2)

    tracker.save(str(gen2_output_dir / "gen2_stage_metrics.json"))

    print(f"\nGen2 Pipeline complete!")
    print(f"   Input: {result['stats']['input_count']:,} | Output: {result['stats']['output_count']:,}")
    print(f"   Retention: {result['stats']['retention_rate']:.1%}")
    print(f"   Output dir: {gen2_output_dir}")


if __name__ == "__main__":
    main()
