"""
src/gen3/pipeline.py
第三代 Pipeline：Hybrid Pipeline + Data Recovery

三个核心创新的完整集成：
  1. 分类器集成（ClassifierEnsemble）→ 扩大高质量覆盖面
  2. 条件性 Heuristic Bypass（ConditionalBypass）→ 减少误杀
  3. 合成数据改写（SyntheticRephraser）→ 低质量数据回收

Nemotron-CC 的方法论指导原则（论文原文）：
  "从静态的、非学习的 heuristic pipeline，
   转向更具学习能力的 flywheel——
   数据改善模型，模型反过来改善数据。"
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np

from src.gen3.classifier_ensemble import ClassifierEnsemble
from src.gen3.conditional_bypass import ConditionalBypass
from src.gen3.synthetic_rephraser import SyntheticRephraser
from src.gen1.filters.quality_filter import QualityFilter
from src.utils.config_loader import load_pipeline_config


class Gen3Pipeline:
    """
    第三代 Hybrid Pipeline + Data Recovery。
    """

    def __init__(
        self,
        run_config: Dict,
        pipeline_config: Dict,
        ensemble: ClassifierEnsemble,
        rephraser: Optional[SyntheticRephraser] = None,
        stage_tracker=None,
        filter_auditor=None,
    ):
        self.run_cfg = run_config
        self.pipe_cfg = pipeline_config
        self.ensemble = ensemble
        self.rephraser = rephraser
        self.tracker = stage_tracker
        self.auditor = filter_auditor

        # 条件路由器
        bypass_cfg = pipeline_config.get("conditional_bypass", {})
        self.router = ConditionalBypass(
            high_quality_threshold=bypass_cfg.get("high_quality_threshold", 0.7),
            medium_quality_threshold=bypass_cfg.get("medium_quality_threshold", 0.3),
            rephrase_score_range=tuple(
                pipeline_config.get("synthetic_rephrasing", {})
                .get("rephrase_score_range", [0.1, 0.3])
            ),
        )

        # Heuristic 过滤器（只用于中等质量文档）
        # 必须使用 Gen1 config 中调整过的阈值（如 CC WET 的 terminal_punct_min_ratio=0.1,
        # min_alpha_ratio=0.5），否则默认阈值（0.7）会导致 93% 误杀率（论文预期 ~18%）。
        gen1_pipe_cfg = load_pipeline_config(1)
        gen1_filter_cfg = gen1_pipe_cfg.get("filters", {})
        goph_cfg = gen1_filter_cfg.get("gopher_quality", {})
        c4_cfg = gen1_filter_cfg.get("c4_quality", {})
        fw_cfg = gen1_filter_cfg.get("fineweb_quality", {})

        gopher_kwargs = {k: v for k, v in goph_cfg.items() if k != "enabled"}
        c4_kwargs = {}
        if "min_lines" in c4_cfg:
            c4_kwargs["min_lines"] = c4_cfg["min_lines"]
        if "min_words_per_line" in c4_cfg:
            c4_kwargs["min_words_per_line"] = c4_cfg["min_words_per_line"]
        if "filter_javascript" in c4_cfg:
            c4_kwargs["filter_javascript"] = c4_cfg["filter_javascript"]
        if "filter_lorem_ipsum" in c4_cfg:
            c4_kwargs["filter_lorem_ipsum"] = c4_cfg["filter_lorem_ipsum"]
        if "terminal_punct_min_ratio" in c4_cfg:
            c4_kwargs["terminal_punct_min_ratio"] = c4_cfg["terminal_punct_min_ratio"]
        fineweb_kwargs = {}
        if "max_lines_starting_with_bullet" in fw_cfg:
            fineweb_kwargs["max_bullet_lines_ratio"] = fw_cfg["max_lines_starting_with_bullet"]
        if "max_lines_ending_with_ellipsis" in fw_cfg:
            fineweb_kwargs["max_ellipsis_lines_ratio"] = fw_cfg["max_lines_ending_with_ellipsis"]
        if "min_alpha_words_ratio" in fw_cfg:
            fineweb_kwargs["min_alpha_words_ratio"] = fw_cfg["min_alpha_words_ratio"]

        self.heuristic_filter = QualityFilter(
            use_gopher=goph_cfg.get("enabled", True),
            use_c4=c4_cfg.get("enabled", True),
            use_fineweb=fw_cfg.get("enabled", True),
            gopher_kwargs=gopher_kwargs,
            c4_kwargs=c4_kwargs,
            fineweb_kwargs=fineweb_kwargs,
        )

        # 严格阈值过滤器（仅用于 bypass 价值分析）
        # bypass 分析回答的问题是："如果用论文标准的严格规则，会误杀多少高质量文档？"
        # 必须用严格阈值（论文默认值）才有意义，用宽松阈值会得到 0% 误杀的假象。
        self.strict_heuristic_filter = QualityFilter(
            use_gopher=True, use_c4=True, use_fineweb=True,
        )

        # 改写数量限制
        self.rewrite_count = run_config.get("rewrite_count", 50)

    def run(self, docs: List[Dict]) -> Dict:
        """
        执行第三代 Pipeline。

        Returns:
            dict，包含：
              - final_docs: 最终保留的文档（真实 + 合成）
              - routing_summary: 路由统计
              - rephrasing_stats: 改写统计
              - stage_stats: 各阶段统计
        """
        print(f"\n{'='*60}")
        print(f"  第三代 Pipeline 启动 | 输入: {len(docs):,} 条")
        print(f"{'='*60}")
        start = time.time()

        if self.tracker:
            self.tracker.record("gen3_input", [d["text"] for d in docs],
                                urls=[d.get("url", "") for d in docs])

        # ── Step 1: 分类器集成打分 ────────────────────────────
        print(f"\n  Step 1: 分类器集成打分...")
        texts = [d["text"] for d in docs]
        ensemble_scores, individual_scores = self.ensemble.score_batch(texts)

        for doc, score in zip(docs, ensemble_scores):
            doc["_ensemble_score"] = round(float(score), 4)

        # ── Step 2: 条件性路由 ────────────────────────────────
        print(f"\n  Step 2: 条件性 Heuristic Bypass 路由...")
        buckets = self.router.route(docs, ensemble_scores, self.heuristic_filter)

        # Bypass 价值分析（验证 Nemotron-CC 的 18.1% 发现）
        # 用严格阈值（论文默认值）分析：如果不做 bypass，严格 heuristic 会误杀多少高质量文档？
        bypass_analysis = self.router.compute_bypass_value(
            buckets["high_quality"], self.strict_heuristic_filter
        )

        # ── Step 3: 合成数据改写 ──────────────────────────────
        rephrased_docs = []
        rephrasing_stats = {"skipped": True}

        if self.rephraser and buckets["to_rephrase"]:
            rewrite_concurrency = self.run_cfg.get("rewrite_concurrency", 1)
            min_quality = self.pipe_cfg.get("synthetic_rephrasing", {}).get(
                "post_rephrase_filter", {}
            ).get("min_quality_score", 0.4)
            print(f"\n  Step 3: Synthetic rewrite (max {self.rewrite_count}, concurrency={rewrite_concurrency})...")
            print(f"     Post-rephrase quality gate: ensemble score >= {min_quality}")
            rephrased_docs, rephrasing_stats = self.rephraser.rephrase_batch(
                buckets["to_rephrase"],
                max_count=self.rewrite_count,
                eval_classifier=self.ensemble,
                min_quality_after=min_quality,
                concurrency=rewrite_concurrency,
            )
            print(f"     改写成功: {len(rephrased_docs)} 条")
        else:
            print(f"\n  Step 3: 跳过改写（未配置 API 或无待改写文档）")

        # ── Step 4: 合并最终结果 ──────────────────────────────
        final_docs = (
            buckets["high_quality"]
            + buckets["heuristic_passed"]
            + rephrased_docs
        )

        # ── 完成 ──────────────────────────────────────────────
        elapsed = time.time() - start
        routing_summary = self.router.get_summary(buckets, len(docs))

        print(f"\n{'='*60}")
        print(f"  ✅ 第三代 Pipeline 完成！")
        print(f"  输入: {len(docs):,} → 最终输出: {len(final_docs):,}")
        print(f"    真实高质量 (bypass): {len(buckets['high_quality']):,}")
        print(f"    真实中等 (heuristic通过): {len(buckets['heuristic_passed']):,}")
        print(f"    合成数据 (改写): {len(rephrased_docs):,}")
        print(f"  总保留率: {len(final_docs)/len(docs):.1%} | 耗时: {elapsed:.1f}s")
        print(f"{'='*60}")

        if self.tracker:
            self.tracker.record(
                "gen3_output",
                [d["text"] for d in final_docs],
                urls=[d.get("url", "") for d in final_docs],
            )

        return {
            "final_docs": final_docs,
            "routing_summary": routing_summary,
            "bypass_analysis": bypass_analysis,
            "rephrasing_stats": rephrasing_stats,
            "individual_scores": individual_scores,
            "ensemble_scores": ensemble_scores,
            "elapsed_seconds": elapsed,
        }

    def save_results(self, result: Dict, output_dir: Path) -> None:
        """保存第三代 Pipeline 结果（分文件保存真实和合成数据）。"""
        output_dir.mkdir(parents=True, exist_ok=True)

        # 保存全量最终文档
        all_path = output_dir / "gen3_output.jsonl"
        with open(all_path, "w", encoding="utf-8") as f:
            for doc in result["final_docs"]:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")

        # 保存路由摘要
        summary_path = output_dir / "gen3_routing_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({
                "routing": result["routing_summary"],
                "bypass_analysis": result["bypass_analysis"],
                "rephrasing": result["rephrasing_stats"],
                "elapsed_seconds": result["elapsed_seconds"],
            }, f, ensure_ascii=False, indent=2)

        print(f"  💾 Gen3 结果已保存: {output_dir}")
