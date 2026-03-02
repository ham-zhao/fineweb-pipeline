"""
src/gen3/conditional_bypass.py
第三代核心创新 2：条件性 Heuristic Bypass

Nemotron-CC 的关键发现：
  对 fastText 判定为高质量的文档，跳过 heuristic filter——
  因为 heuristic 会误杀 18.1% 的高质量 token（Nemotron-CC 原文数据）。

  为什么会误杀？举例：
  1. 代码文档：含有大量特殊字符，被 Gopher 的"alpha ratio"规则过滤
  2. 技术教程：含有很多"短行"（代码片段），被 C4 的行规则过滤
  3. 问答格式文本：平均句子短，被 Gopher 的 avg_sentence_length 过滤
  以上内容可能质量很高，但长相"不像"高质量文本（不符合 heuristic 的统计假设）

分层路由逻辑：
  score >= high_quality_threshold  →  直接保留（跳过 heuristic）
  medium_threshold <= score < high  →  应用 heuristic 过滤
  score < medium_threshold          →  送去 LLM 改写或丢弃
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class RoutingDecision:
    """单条文档的路由决策记录。"""
    doc_index: int
    ensemble_score: float
    route: str          # "high_quality" | "heuristic_filtered" | "heuristic_passed" | "rephrase" | "discard"
    heuristic_reason: str = ""
    final_action: str = ""  # "kept" | "filtered" | "rephrase" | "discarded"


class ConditionalBypass:
    """
    条件性 Heuristic Bypass 路由器。
    根据集成分类器的分数，将文档路由到不同处理路径。
    """

    def __init__(
        self,
        high_quality_threshold: float = 0.7,   # 直接保留（跳过 heuristic）
        medium_quality_threshold: float = 0.3,  # 中等：应用 heuristic
        # 低于 medium_threshold：送去改写（分数最高的那部分）或直接丢弃
        rephrase_score_range: Tuple[float, float] = (0.1, 0.3),
    ):
        """
        Args:
            high_quality_threshold: 高质量阈值（score >= 此值 → bypass heuristic）
            medium_quality_threshold: 中等质量阈值（score < 此值 → 改写/丢弃）
            rephrase_score_range: 改写区间（在低质量中选"最有潜力"的）
        """
        self.high_threshold = high_quality_threshold
        self.medium_threshold = medium_quality_threshold
        self.rephrase_min, self.rephrase_max = rephrase_score_range

    def route(
        self,
        docs: List[Dict],
        ensemble_scores: np.ndarray,
        quality_filter,       # QualityFilter 实例（第一代的规则过滤器）
    ) -> Dict[str, List]:
        """
        将所有文档按分数路由到不同处理桶。

        Returns:
            dict，包含：
              - high_quality: 直接保留（bypass heuristic）的文档
              - heuristic_passed: 经过 heuristic 且通过的文档
              - heuristic_filtered: 经过 heuristic 被过滤的文档
              - to_rephrase: 待 LLM 改写的文档
              - discarded: 直接丢弃的文档
              - routing_log: 每条文档的路由决策记录
        """
        buckets = {
            "high_quality": [],
            "heuristic_passed": [],
            "heuristic_filtered": [],
            "to_rephrase": [],
            "discarded": [],
        }
        routing_log: List[RoutingDecision] = []

        print(f"  🚦 条件性路由: {len(docs):,} 条文档")
        print(f"     高质量阈值: {self.high_threshold} | 中等阈值: {self.medium_threshold}")

        for i, (doc, score) in enumerate(zip(docs, ensemble_scores)):
            score = float(score)
            doc["_ensemble_score"] = round(score, 4)

            decision = RoutingDecision(doc_index=i, ensemble_score=score, route="")

            if score >= self.high_threshold:
                # 路径 A：高质量 → 直接保留，跳过 heuristic
                decision.route = "high_quality"
                decision.final_action = "kept"
                buckets["high_quality"].append(doc)

            elif score >= self.medium_threshold:
                # 路径 B：中等质量 → 应用 heuristic
                passes, reason = quality_filter.check(doc["text"])
                if passes:
                    decision.route = "heuristic_passed"
                    decision.final_action = "kept"
                    buckets["heuristic_passed"].append(doc)
                else:
                    decision.route = "heuristic_filtered"
                    decision.heuristic_reason = reason
                    decision.final_action = "filtered"
                    doc["_heuristic_reason"] = reason
                    buckets["heuristic_filtered"].append(doc)

            else:
                # 路径 C：低质量 → 改写或丢弃
                if self.rephrase_min <= score <= self.rephrase_max:
                    decision.route = "to_rephrase"
                    decision.final_action = "rephrase"
                    buckets["to_rephrase"].append(doc)
                else:
                    decision.route = "discard"
                    decision.final_action = "discarded"
                    buckets["discarded"].append(doc)

            routing_log.append(decision)

        # 打印路由统计
        self._print_routing_stats(buckets, routing_log, len(docs))
        buckets["routing_log"] = routing_log
        return buckets

    def _print_routing_stats(
        self,
        buckets: Dict,
        routing_log: List[RoutingDecision],
        total: int,
    ) -> None:
        print(f"\n  路由结果:")
        for name, docs_list in buckets.items():
            if name == "routing_log":
                continue
            n = len(docs_list)
            print(f"    {name:<22}: {n:>6,} ({n/total:.1%})")

        # 计算 bypass 回收率（高质量中有多少会被 heuristic 误杀）
        high_q = buckets["high_quality"]
        if high_q:
            heuristic_filter = self._simulate_heuristic_on_high_quality(
                buckets.get("high_quality", []), buckets
            )

    def _simulate_heuristic_on_high_quality(
        self, high_quality_docs: List[Dict], buckets: Dict
    ) -> None:
        """模拟：如果对高质量文档也应用 heuristic，会损失多少？（仅统计，不实际过滤）"""
        # 这个分析在 Notebook 04 Cell Group B 中展开
        pass

    def compute_bypass_value(
        self,
        high_quality_docs: List[Dict],
        quality_filter,
    ) -> Dict:
        """
        计算 bypass 的价值：高质量文档中会被 heuristic 误杀的比例。
        这验证 Nemotron-CC 的 18.1% 误杀率发现。

        Returns:
            dict，包含 would_be_filtered_rate 和样本
        """
        if not high_quality_docs:
            return {"would_be_filtered_rate": 0, "n_checked": 0}

        would_filter = []
        for doc in high_quality_docs:
            passes, reason = quality_filter.check(doc["text"])
            if not passes:
                would_filter.append({"text": doc["text"][:200], "reason": reason})

        rate = len(would_filter) / len(high_quality_docs)
        print(f"\n  🔬 Bypass 价值分析:")
        print(f"     高质量文档: {len(high_quality_docs):,} 条")
        print(f"     若应用 heuristic 会被误杀: {len(would_filter):,} 条 ({rate:.1%})")
        print(f"     对比 Nemotron-CC 论文发现: 18.1%")

        return {
            "high_quality_count": len(high_quality_docs),
            "would_be_filtered_count": len(would_filter),
            "would_be_filtered_rate": round(rate, 4),
            "sample_false_kills": would_filter[:5],
        }

    def get_summary(self, buckets: Dict, total: int) -> Dict:
        """生成路由摘要统计（用于 Notebook 漏斗图）。"""
        summary = {"total": total}
        for name, docs_list in buckets.items():
            if name == "routing_log":
                continue
            summary[name] = {
                "count": len(docs_list),
                "rate": round(len(docs_list) / total, 4) if total > 0 else 0,
            }
        summary["total_kept"] = (
            summary.get("high_quality", {}).get("count", 0)
            + summary.get("heuristic_passed", {}).get("count", 0)
        )
        return summary
