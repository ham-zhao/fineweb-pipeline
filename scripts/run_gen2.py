#!/usr/bin/env python3
"""
scripts/run_gen2.py
第二代 Model-based Filtering Pipeline — 独立运行脚本

用法:
    python scripts/run_gen2.py                    # 从 Gen1 输出读取
    python scripts/run_gen2.py --from-raw         # 从原始数据开始（不用 Gen1 输出）
    python scripts/run_gen2.py --top-fraction 0.15  # 自定义保留比例

防休眠: caffeinate -i python scripts/run_gen2.py
"""

import sys
import os
import argparse
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config_loader import (
    load_run_config, load_pipeline_config, load_eval_config,
    get_output_path, print_config_summary
)
from src.gen1.pipeline import read_jsonl, read_warc_texts, save_jsonl, Gen1Pipeline
from src.gen2.quality_classifier import Gen2QualityClassifier
from src.gen2.pipeline import Gen2Pipeline
from src.evaluation.stage_tracker import StageTracker
from src.evaluation.quality_classifier import EvalQualityClassifier


def parse_args():
    parser = argparse.ArgumentParser(description="第二代 Model-based Filtering Pipeline")
    parser.add_argument("--run-config", default="configs/run_config.yaml")
    parser.add_argument("--config", default="configs/gen2_config.yaml")
    parser.add_argument("--from-raw", action="store_true", help="从原始数据开始（跳过 Gen1 输出）")
    parser.add_argument("--top-fraction", type=float, default=0.10, help="保留比例（默认 0.10 即 top-10%）")
    parser.add_argument("--retrain-classifier", action="store_true", help="强制重新训练分类器")
    return parser.parse_args()


def load_or_train_classifier(pipe_cfg, run_cfg, retrain: bool = False) -> Gen2QualityClassifier:
    """加载已有分类器，或训练新分类器。"""
    model_path = pipe_cfg.get("classifier", {}).get("fasttext", {}).get(
        "model_path", "results/quality_scores/gen2_classifier.bin"
    )

    clf = Gen2QualityClassifier(model_path=None)

    if Path(model_path).exists() and not retrain:
        print(f"  📂 加载已有 Gen2 分类器: {model_path}")
        clf._load(model_path)
        return clf

    print(f"  🏋️  训练 Gen2 分类器（正样本: Wikipedia+ELI5，负样本: 原始CC）...")

    # 加载正样本（Wikipedia 摘要 + ELI5）
    positive_texts = []
    wiki_path = Path("data/reference/wikipedia_abstracts.jsonl")
    if wiki_path.exists():
        with open(wiki_path) as f:
            for line in f:
                try:
                    doc = json.loads(line)
                    positive_texts.append(doc["text"])
                except Exception:
                    pass
        print(f"     Wikipedia 正样本: {len(positive_texts):,} 条")

    eli5_path = Path("data/reference/stackexchange_top.jsonl")
    if eli5_path.exists():
        with open(eli5_path) as f:
            for line in f:
                try:
                    doc = json.loads(line)
                    positive_texts.append(doc["text"])
                except Exception:
                    pass
        print(f"     ELI5 正样本（含）: {len(positive_texts):,} 条")

    if not positive_texts:
        print("  ⚠️  未找到正样本数据，使用合成样本（仅用于 smoke_test）")
        positive_texts = [
            "The mitochondria is the powerhouse of the cell, converting nutrients into ATP through oxidative phosphorylation.",
            "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
            "The French Revolution began in 1789 with the storming of the Bastille, fundamentally changing European politics.",
        ] * 100

    # 负样本（原始 CC 数据）
    negative_texts = []
    raw_files = list(Path("data/raw").glob("*.jsonl"))
    doc_limit = run_cfg.get("doc_limit", 1000)
    for f in raw_files[:1]:
        with open(f) as fp:
            for i, line in enumerate(fp):
                if i >= doc_limit:
                    break
                try:
                    doc = json.loads(line)
                    negative_texts.append(doc.get("text", ""))
                except Exception:
                    pass

    if not negative_texts:
        print("  ⚠️  未找到负样本，使用合成样本")
        negative_texts = ["buy now discount sale cheap free click here"] * 100

    # 平衡正负样本
    min_count = min(len(positive_texts), len(negative_texts), 5000)
    import random
    random.seed(run_cfg.get("random_seed", 42))
    positive_texts = random.sample(positive_texts, min(min_count, len(positive_texts)))
    negative_texts = random.sample(negative_texts, min(min_count, len(negative_texts)))

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
        print(f"\n📂 读取 Gen1 输出: {gen1_file}")
        docs = read_jsonl(gen1_file, doc_limit=doc_limit)
    else:
        print(f"\n⚠️  Gen1 输出不存在，从原始数据读取...")
        raw_files = list(Path("data/raw").glob("*.warc.gz")) + list(Path("data/raw").glob("*.jsonl"))
        if not raw_files:
            print("❌ 未找到原始数据！请先运行 bash scripts/download_sample.sh")
            sys.exit(1)
        input_path = raw_files[0]
        if input_path.suffix in (".gz",):
            docs = read_warc_texts(input_path, doc_limit=doc_limit)
        else:
            docs = read_jsonl(input_path, doc_limit=doc_limit)

    print(f"✅ 读取 {len(docs):,} 条文档")

    # ── 训练/加载分类器 ───────────────────────────────────────
    classifier = load_or_train_classifier(pipe_cfg, run_cfg, args.retrain_classifier)

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
        # ndarray 不可 JSON 序列化，转为 list
        stats_safe = dict(result["stats"])
        json.dump(stats_safe, f, ensure_ascii=False, indent=2)

    tracker.save(str(gen2_output_dir / "gen2_stage_metrics.json"))

    print(f"\n✅ 第二代 Pipeline 完成！")
    print(f"   输入: {result['stats']['input_count']:,} | 输出: {result['stats']['output_count']:,}")
    print(f"   保留率: {result['stats']['retention_rate']:.1%}")
    print(f"   输出目录: {gen2_output_dir}")


if __name__ == "__main__":
    main()
