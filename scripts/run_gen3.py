#!/usr/bin/env python3
"""
scripts/run_gen3.py
第三代 Hybrid Pipeline + Data Recovery — 独立运行脚本

用法:
    python scripts/run_gen3.py                   # 标准运行
    python scripts/run_gen3.py --no-rephrase     # 跳过 LLM 改写（无需 API）
    python scripts/run_gen3.py --strategy union  # 指定集成策略

防休眠: caffeinate -i python scripts/run_gen3.py
"""

import sys
import argparse
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config_loader import (
    load_run_config, load_pipeline_config, load_eval_config,
    load_api_config, get_output_path, print_config_summary
)
from src.gen1.pipeline import read_jsonl, read_warc_texts
from src.gen2.quality_classifier import Gen2QualityClassifier
from src.gen3.classifier_ensemble import ClassifierEnsemble
from src.gen3.synthetic_rephraser import SyntheticRephraser
from src.gen3.pipeline import Gen3Pipeline
from src.evaluation.stage_tracker import StageTracker


def parse_args():
    parser = argparse.ArgumentParser(description="第三代 Hybrid Pipeline")
    parser.add_argument("--run-config", default="configs/run_config.yaml")
    parser.add_argument("--config", default="configs/gen3_config.yaml")
    parser.add_argument("--run-mode", default=None, choices=["smoke_test", "full_run"],
                        help="覆盖 run_config.yaml 中的 run_mode（不修改文件）")
    parser.add_argument("--no-rephrase", action="store_true", help="跳过 LLM 改写（不需要 API Key）")
    parser.add_argument("--strategy", default="union", choices=["union", "intersection", "weighted_avg"])
    return parser.parse_args()


def main():
    args = parse_args()

    run_cfg = load_run_config(args.run_config, run_mode_override=args.run_mode)
    pipe_cfg = load_pipeline_config(3, args.config)

    print_config_summary(run_cfg)

    doc_limit = run_cfg.get("doc_limit")
    gen1_output_dir = get_output_path(1, run_cfg)
    gen3_output_dir = get_output_path(3, run_cfg)

    # ── 加载输入数据（优先 Gen1 输出，否则原始数据）────────
    gen1_file = gen1_output_dir / "gen1_output.jsonl"
    if gen1_file.exists():
        print(f"\n📂 读取 Gen1 输出: {gen1_file}")
        docs = read_jsonl(gen1_file, doc_limit=doc_limit)
    else:
        raw_files = list(Path("data/raw").glob("*.warc.gz")) + list(Path("data/raw").glob("*.jsonl"))
        if not raw_files:
            print("❌ 未找到数据！请先运行 bash scripts/download_sample.sh && python scripts/run_gen1.py")
            sys.exit(1)
        docs = read_warc_texts(raw_files[0], doc_limit=doc_limit) if raw_files[0].suffix == ".gz" \
               else read_jsonl(raw_files[0], doc_limit=doc_limit)

    print(f"✅ 读取 {len(docs):,} 条文档")

    # ── 构建分类器集成 ────────────────────────────────────────
    print("\n  🔧 构建分类器集成...")
    ensemble = ClassifierEnsemble(
        strategy=args.strategy,
        union_threshold=pipe_cfg.get("classifier_ensemble", {}).get("union_threshold", 0.5),
    )

    # 分类器 1: DCLM 风格 fastText
    clf1_path = "results/quality_scores/gen2_classifier.bin"
    if Path(clf1_path).exists():
        clf1 = Gen2QualityClassifier(model_path=clf1_path)
        ensemble.add_fasttext_classifier("fasttext_dclm", clf1, weight=0.4)
    else:
        print(f"  ⚠️  Gen2 分类器不存在 ({clf1_path})，请先运行 run_gen2.py")
        print(f"     将只使用 TF-IDF+LR 分类器继续...")

    # 分类器 2: fasttext_edu (教育类正样本: Cosmopedia)
    edu_path = Path("data/reference/cosmopedia_edu.jsonl")
    if edu_path.exists():
        edu_texts = []
        with open(edu_path) as f:
            for i, line in enumerate(f):
                if i >= 5000:
                    break
                try:
                    edu_texts.append(json.loads(line)["text"])
                except Exception:
                    pass
        if edu_texts:
            negative_sample_edu = [d["text"] for d in docs[:min(len(edu_texts), 5000)]]
            clf_edu = Gen2QualityClassifier()
            clf_edu.train(
                positive_texts=edu_texts,
                negative_texts=negative_sample_edu,
                output_path="results/quality_scores/gen3_edu_classifier.bin",
                dim=64,
                wordNgrams=2,
            )
            ensemble.add_fasttext_classifier("fasttext_edu", clf_edu, weight=0.4)
    else:
        print(f"  ⚠️  Cosmopedia 教育文本不存在 ({edu_path})，跳过 fasttext_edu")
        print(f"     运行 bash scripts/download_sample.sh 下载")

    # 分类器 3: TF-IDF + LR (Wikipedia 正样本)
    wiki_texts = []
    wiki_path = Path("data/reference/wikipedia_abstracts.jsonl")
    if wiki_path.exists():
        with open(wiki_path) as f:
            for i, line in enumerate(f):
                if i >= 2000:
                    break
                try:
                    wiki_texts.append(json.loads(line)["text"])
                except Exception:
                    pass

    if wiki_texts:
        negative_sample = [d["text"] for d in docs[:min(len(wiki_texts), 2000)]]
        ensemble.train_tfidf_lr(
            name="tfidf_lr_wiki",
            positive_texts=wiki_texts,
            negative_texts=negative_sample,
            model_path="results/quality_scores/gen3_tfidf_lr.pkl",
            weight=0.2,
        )

    # ── 初始化改写器 ─────────────────────────────────────────
    rephraser = None
    if not args.no_rephrase:
        try:
            api_cfg = load_api_config()
            rephraser = SyntheticRephraser(api_cfg)
            print(f"  ✅ LLM 改写器就绪: {api_cfg['provider']}/{api_cfg['model']}")
        except ValueError as e:
            print(f"  ⚠️  {e}")
            print(f"  ⚠️  将跳过 LLM 改写（使用 --no-rephrase 明确跳过）")

    # ── 运行 Pipeline ─────────────────────────────────────────
    tracker = StageTracker(load_eval_config(), run_cfg)

    pipeline = Gen3Pipeline(
        run_config=run_cfg,
        pipeline_config=pipe_cfg,
        ensemble=ensemble,
        rephraser=rephraser,
        stage_tracker=tracker,
    )

    result = pipeline.run(docs)

    # ── 保存输出 ─────────────────────────────────────────────
    gen3_output_dir.mkdir(parents=True, exist_ok=True)
    pipeline.save_results(result, gen3_output_dir)
    tracker.save(str(gen3_output_dir / "gen3_stage_metrics.json"))

    print(f"\n✅ 第三代 Pipeline 完成！")
    print(f"   最终保留: {len(result['final_docs']):,} 条")
    print(f"   输出目录: {gen3_output_dir}")


if __name__ == "__main__":
    main()
