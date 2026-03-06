#!/usr/bin/env python3
"""
scripts/run_gen1.py
第一代 Heuristic Filtering Pipeline — 独立运行脚本

用法:
    python scripts/run_gen1.py                          # 使用默认配置
    python scripts/run_gen1.py --config configs/gen1_config.yaml
    python scripts/run_gen1.py --input data/raw/sample.warc.gz

防休眠: caffeinate -i python scripts/run_gen1.py
"""

import sys
import os
import argparse
import json
import time
from pathlib import Path

# 确保项目根目录在 Python 路径中
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config_loader import (
    load_run_config, load_pipeline_config, load_eval_config,
    get_output_path, print_config_summary
)
from src.gen1.pipeline import Gen1Pipeline, read_warc_texts, read_jsonl, save_jsonl
from src.evaluation.stage_tracker import StageTracker
from src.evaluation.filter_auditor import FilterAuditor


def parse_args():
    parser = argparse.ArgumentParser(description="第一代 Heuristic Filtering Pipeline")
    parser.add_argument("--config", default="configs/gen1_config.yaml", help="Pipeline 配置文件")
    parser.add_argument("--run-config", default="configs/run_config.yaml", help="运行配置文件")
    parser.add_argument("--run-mode", default=None, choices=["smoke_test", "full_run"],
                        help="覆盖 run_config.yaml 中的 run_mode（不修改文件）")
    parser.add_argument("--input", default=None, help="输入数据文件路径（WARC 或 JSONL）")
    parser.add_argument("--output", default=None, help="输出目录路径")
    parser.add_argument("--no-eval", action="store_true", help="跳过评估（加快速度）")
    return parser.parse_args()


def main():
    args = parse_args()

    # ── 加载配置 ─────────────────────────────────────────────
    run_cfg = load_run_config(args.run_config, run_mode_override=args.run_mode)
    pipe_cfg = load_pipeline_config(1, args.config)
    eval_cfg = load_eval_config()

    print_config_summary(run_cfg)

    doc_limit = run_cfg.get("doc_limit")
    output_dir = Path(args.output) if args.output else get_output_path(1, run_cfg)

    # ── 初始化评估组件 ───────────────────────────────────────
    tracker = None
    auditor = None

    if not args.no_eval:
        run_mode = run_cfg.get("run_mode", "smoke_test")
        audit_dir = Path("data/reports/audit/gen1") / run_mode
        auditor = FilterAuditor(
            output_dir=str(audit_dir),
            audit_sample_size=run_cfg.get("audit_sample_size", 20),
            random_seed=run_cfg.get("random_seed", 42),
        )
        tracker = StageTracker(eval_cfg, run_cfg)

    # ── 加载输入数据 ──────────────────────────────────────────
    if args.input:
        input_path = Path(args.input)
    else:
        # 自动寻找输入文件，优先 CC WET 数据
        # full_run (100K) 使用 cc_wet_full.jsonl，其他模式用 cc_wet_sample.jsonl
        cc_wet_full = Path("data/raw/cc_wet_full.jsonl")
        cc_wet_file = Path("data/raw/cc_wet_sample.jsonl")
        warc_files = list(Path("data/raw").glob("*.warc.gz"))
        jsonl_files = list(Path("data/raw").glob("*.jsonl"))

        if doc_limit and doc_limit > 12000 and cc_wet_full.exists():
            input_path = cc_wet_full
        elif cc_wet_file.exists():
            input_path = cc_wet_file
        elif warc_files:
            input_path = warc_files[0]
        elif jsonl_files:
            input_path = jsonl_files[0]
        else:
            print("❌ 未找到输入数据！请先运行 bash scripts/download_sample.sh")
            print("   或使用 --input 参数指定文件路径")
            sys.exit(1)

    print(f"\n📂 输入文件: {input_path}")

    # 读取数据
    if input_path.suffix in (".gz", ".warc"):
        docs = read_warc_texts(input_path, doc_limit=doc_limit)
    elif input_path.suffix == ".jsonl":
        docs = read_jsonl(input_path, doc_limit=doc_limit)
    else:
        print(f"❌ 不支持的文件格式: {input_path.suffix}")
        sys.exit(1)

    if not docs:
        print("❌ 没有读取到任何文档！")
        sys.exit(1)

    print(f"✅ 读取 {len(docs):,} 条文档")

    # ── 运行 Pipeline ─────────────────────────────────────────
    pipeline = Gen1Pipeline(
        run_config=run_cfg,
        pipeline_config=pipe_cfg,
        stage_tracker=tracker,
        filter_auditor=auditor,
    )

    filtered_docs = pipeline.run(docs)

    # ── 保存输出 ─────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "gen1_output.jsonl"
    save_jsonl(filtered_docs, output_file, desc="Gen1 输出")

    # 保存 pipeline 统计
    stats_file = output_dir / "gen1_pipeline_stats.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump({
            "pipeline_stats": pipeline.get_pipeline_stats(),
            "input_count": len(docs),
            "output_count": len(filtered_docs),
            "retention_rate": len(filtered_docs) / len(docs) if docs else 0,
        }, f, ensure_ascii=False, indent=2)
    print(f"📊 统计报告: {stats_file}")

    # 保存评估指标
    if tracker:
        tracker.save(str(output_dir / "gen1_stage_metrics.json"))

    # 审计 CSV
    if auditor:
        auditor.export_audit_csv()
        auditor.generate_summary()

    print(f"\n✅ 第一代 Pipeline 完成！")
    print(f"   输出目录: {output_dir}")
    print(f"   保留: {len(filtered_docs):,}/{len(docs):,} ({len(filtered_docs)/len(docs):.1%})")


if __name__ == "__main__":
    main()
