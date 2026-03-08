#!/usr/bin/env python3
"""
scripts/run_gen3.py
第三代 Hybrid Pipeline + Data Recovery — 独立运行脚本

⚠️  统一输入架构：Gen3 = Gen1 heuristic + 分类器集成 + 路由 + 改写
    始终从原始 CC WET 开始，内部先运行 Gen1 Pipeline，再应用 Gen3 逻辑。
    保留率口径 = Gen3 最终输出 / CC WET 原始输入。

用法:
    python scripts/run_gen3.py                   # 标准运行（CC WET → Gen1 → Gen3）
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
from src.gen1.pipeline import read_jsonl, read_warc_texts, Gen1Pipeline
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
    gen3_output_dir = get_output_path(3, run_cfg)

    # ── 加载原始 CC WET 数据（统一输入架构）───────────────────
    cc_wet = Path("data/raw/cc_wet_full.jsonl") if run_cfg.get("doc_limit", 0) > 12000 and Path("data/raw/cc_wet_full.jsonl").exists() else Path("data/raw/cc_wet_sample.jsonl")
    if not cc_wet.exists():
        raw_files = list(Path("data/raw").glob("*.warc.gz")) + list(Path("data/raw").glob("*.jsonl"))
        if raw_files:
            cc_wet = raw_files[0]
        else:
            print("❌ 未找到数据！请先运行 bash scripts/download_sample.sh")
            sys.exit(1)

    print(f"\n📂 读取原始 CC WET: {cc_wet}")
    if cc_wet.suffix in (".gz",):
        docs = read_warc_texts(cc_wet, doc_limit=doc_limit)
    else:
        docs = read_jsonl(cc_wet, doc_limit=doc_limit)

    raw_input_count = len(docs)
    print(f"✅ 原始输入: {raw_input_count:,} 条")

    # ── Step 1: Gen1 Heuristic 预处理 ──────────────────────────
    print(f"\n{'='*50}")
    print(f"  Step 1: Gen1 Heuristic 预处理")
    print(f"{'='*50}")
    gen1_pipe_cfg = load_pipeline_config(1)
    gen1_pipeline = Gen1Pipeline(
        run_config=run_cfg,
        pipeline_config=gen1_pipe_cfg,
    )
    docs = gen1_pipeline.run(docs)
    print(f"  Gen1 后剩余: {len(docs):,} 条 (Gen1 保留率: {len(docs)/raw_input_count:.1%})")

    # ── 构建分类器集成 ────────────────────────────────────────
    print("\n  🔧 构建分类器集成...")
    ensemble = ClassifierEnsemble(
        strategy=args.strategy,
        union_threshold=pipe_cfg.get("classifier_ensemble", {}).get("union_threshold", 0.5),
    )

    # ── 准备统一负样本（原始 CC WET）──────────────────────────
    # 为什么不用 Gen1 输出做负样本？
    #   Gen1 输出已经过语言过滤+URL过滤+质量过滤，是相对干净的英文文本。
    #   用它做负样本时，与 Wikipedia/Cosmopedia 的差距太小（分离度仅 0.14~0.44），
    #   分类器学不到有效的质量信号。
    # 使用原始 CC WET 做负样本（与 Gen2 一致）：
    #   - 分离度从 0.14 提升到 0.47（EDU），从 0.44 提升到 0.62（DCLM）
    #   - 三个分类器用同一来源的负样本，分数尺度自然一致
    #   - 推理时对 Gen1 输出做排序取 top-X%，只需相对排序正确
    raw_neg_path = Path("data/raw/cc_wet_full.jsonl") if run_cfg.get("doc_limit", 0) > 12000 and Path("data/raw/cc_wet_full.jsonl").exists() else Path("data/raw/cc_wet_sample.jsonl")
    negative_texts_unified = []
    if raw_neg_path.exists():
        with open(raw_neg_path) as f:
            for i, line in enumerate(f):
                if i >= 5000:
                    break
                try:
                    negative_texts_unified.append(json.loads(line).get("text", ""))
                except Exception:
                    pass
    if not negative_texts_unified:
        print("  ⚠️  原始 CC WET 不存在，降级使用 Gen1 输出做负样本（分离度可能不足）")
        negative_texts_unified = [d["text"] for d in docs[:5000]]
    print(f"  📊 统一负样本: 原始 CC WET {len(negative_texts_unified):,} 条")

    # 分类器 1: DCLM 风格 fastText（Gen3 独立训练，不复用 Gen2 模型）
    # 虽然 Gen2 也有 dclm 分类器，但 Gen3 需要独立训练以确保集成成员的分数校准一致。
    wiki_texts = []
    wiki_path = Path("data/reference/wikipedia_abstracts.jsonl")
    if wiki_path.exists():
        with open(wiki_path) as f:
            for line in f:
                try:
                    wiki_texts.append(json.loads(line)["text"])
                except Exception:
                    pass

    if wiki_texts:
        clf_dclm = Gen2QualityClassifier()
        clf_dclm.train(
            positive_texts=wiki_texts[:5000],
            negative_texts=negative_texts_unified,
            output_path="results/quality_scores/gen3_dclm_classifier.bin",
            dim=64,
            wordNgrams=2,
        )
        ensemble.add_fasttext_classifier("fasttext_dclm", clf_dclm, weight=0.4)
    else:
        print(f"  ⚠️  Wikipedia 不存在 ({wiki_path})，跳过 fasttext_dclm")

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
            clf_edu = Gen2QualityClassifier()
            clf_edu.train(
                positive_texts=edu_texts,
                negative_texts=negative_texts_unified,
                output_path="results/quality_scores/gen3_edu_classifier.bin",
                dim=64,
                wordNgrams=2,
            )
            ensemble.add_fasttext_classifier("fasttext_edu", clf_edu, weight=0.4)
    else:
        print(f"  ⚠️  Cosmopedia 教育文本不存在 ({edu_path})，跳过 fasttext_edu")
        print(f"     运行 bash scripts/download_sample.sh 下载")

    # 分类器 3: TF-IDF + LR (Wikipedia 正样本)
    if wiki_texts:
        ensemble.train_tfidf_lr(
            name="tfidf_lr_wiki",
            positive_texts=wiki_texts[:5000],
            negative_texts=negative_texts_unified,
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

    e2e_retention = len(result['final_docs']) / raw_input_count if raw_input_count > 0 else 0
    print(f"\n✅ 第三代 Pipeline 完成！")
    print(f"   原始 CC WET 输入: {raw_input_count:,}")
    print(f"   Gen1 heuristic 后: {len(docs):,}")
    print(f"   Gen3 最终输出: {len(result['final_docs']):,} 条")
    print(f"   端到端保留率: {e2e_retention:.1%} (Gen3输出/CC WET原始输入)")
    print(f"   输出目录: {gen3_output_dir}")


if __name__ == "__main__":
    main()
