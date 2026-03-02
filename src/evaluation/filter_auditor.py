"""
src/evaluation/filter_auditor.py
过滤精度抽检器：对被过滤数据采样，生成 CSV 供人工标注，计算 Precision + 分析误杀案例。

核心用途：
  量化每个过滤器的"误杀率"——被过滤的数据中有多少实际上是高质量的？
  这是验证 Nemotron-CC 核心发现（heuristic filter 误杀 18.1% 高质量 token）的工具。

输出：
  - results/reports/audit/<filter_name>_audit.csv  （供人工标注）
  - results/reports/audit/audit_summary.json        （汇总统计）
"""

import json
import csv
import random
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime


class FilterAuditor:
    """
    过滤精度抽检器。

    典型工作流：
    1. 每个过滤器结束后，调用 record_filtered() 保存被过滤的样本
    2. 全部 pipeline 跑完后，调用 export_audit_csv() 导出 CSV
    3. 人工在 CSV 中填写 human_label（0=确实应该过滤, 1=误杀）
    4. 调用 compute_precision() 计算 Precision（越高越好）
    """

    def __init__(self, output_dir: str, audit_sample_size: int = 20, random_seed: int = 42):
        """
        Args:
            output_dir: 输出目录（如 "results/reports/audit"）
            audit_sample_size: 每个过滤器采样多少条（run_config 中的 audit_sample_size）
            random_seed: 随机种子
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.audit_sample_size = audit_sample_size
        self.random_seed = random_seed

        # 记录所有过滤器的被过滤文档
        self.filter_records: Dict[str, List[Dict]] = {}
        self.filter_stats: Dict[str, Dict] = {}

    def record_filtered(
        self,
        filter_name: str,
        filtered_texts: List[str],
        filtered_meta: Optional[List[Dict]] = None,
        total_docs_before: Optional[int] = None,
    ) -> None:
        """
        记录某个过滤器过滤掉的文档。

        Args:
            filter_name: 过滤器名称（如 "gopher_quality", "url_filter"）
            filtered_texts: 被过滤的文本列表
            filtered_meta: 额外元数据（如 URL、触发的规则）
            total_docs_before: 过滤前的总文档数（用于计算过滤率）
        """
        if not filtered_texts:
            return

        # 随机采样
        sample_size = min(self.audit_sample_size, len(filtered_texts))
        random.seed(self.random_seed)
        indices = random.sample(range(len(filtered_texts)), sample_size)

        records = []
        for idx in indices:
            meta = filtered_meta[idx] if filtered_meta else {}
            records.append({
                "filter_name": filter_name,
                "text_preview": filtered_texts[idx][:500].replace("\n", " "),  # 只存前 500 字
                "text_length_words": len(filtered_texts[idx].split()),
                "url": meta.get("url", ""),
                "filter_reason": meta.get("filter_reason", ""),
                "human_label": "",  # 留空，等人工填写（0=应过滤, 1=误杀）
                "human_comment": "",
            })

        self.filter_records[filter_name] = records

        # 统计信息
        self.filter_stats[filter_name] = {
            "total_filtered": len(filtered_texts),
            "filter_rate": len(filtered_texts) / total_docs_before if total_docs_before else None,
            "audit_sample_size": sample_size,
        }

        print(f"  📋 过滤审计记录: {filter_name} | "
              f"总过滤: {len(filtered_texts):,} | 抽检样本: {sample_size}")

    def export_audit_csv(self) -> Dict[str, Path]:
        """
        为每个过滤器导出 CSV 文件（供人工标注）。

        Returns:
            dict，key 为 filter_name，value 为 CSV 路径
        """
        paths = {}
        for filter_name, records in self.filter_records.items():
            csv_path = self.output_dir / f"{filter_name}_audit.csv"
            fieldnames = ["filter_name", "text_preview", "text_length_words",
                          "url", "filter_reason", "human_label", "human_comment"]

            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(records)

            paths[filter_name] = csv_path
            print(f"  📄 审计 CSV 已导出: {csv_path}")

        return paths

    def compute_precision(
        self,
        filter_name: Optional[str] = None,
    ) -> Dict:
        """
        读取人工标注后的 CSV，计算 Precision（需要先人工填写 human_label）。

        Precision = 正确过滤的文档数 / 总过滤文档数
        （human_label=0 表示"应该过滤"，=1 表示"误杀"）

        Args:
            filter_name: 指定过滤器（None 表示计算全部）

        Returns:
            dict，包含每个过滤器的 precision + 误杀案例
        """
        results = {}
        filter_names = [filter_name] if filter_name else list(self.filter_records.keys())

        for fname in filter_names:
            csv_path = self.output_dir / f"{fname}_audit.csv"
            if not csv_path.exists():
                continue

            correct = 0
            false_kill = 0
            false_kill_examples = []

            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    label = row.get("human_label", "").strip()
                    if label == "0":
                        correct += 1
                    elif label == "1":
                        false_kill += 1
                        false_kill_examples.append({
                            "text": row["text_preview"][:200],
                            "reason": row.get("filter_reason", ""),
                            "comment": row.get("human_comment", ""),
                        })

            total_labeled = correct + false_kill
            if total_labeled == 0:
                print(f"  ⚠️  {fname}: 尚未完成人工标注（human_label 列为空）")
                continue

            precision = correct / total_labeled
            results[fname] = {
                "precision": round(precision, 4),
                "false_kill_rate": round(false_kill / total_labeled, 4),
                "correct": correct,
                "false_kill": false_kill,
                "total_labeled": total_labeled,
                "false_kill_examples": false_kill_examples[:3],
            }

            print(f"  📊 {fname}: Precision={precision:.1%} | 误杀率={false_kill/total_labeled:.1%}")

        return results

    def auto_assess_with_classifier(
        self,
        filter_name: str,
        quality_classifier,
        high_quality_threshold: float = 0.7,
    ) -> Dict:
        """
        用评估分类器自动估算误杀率（不需要人工标注的快速估算版）。

        被分类器认为高质量的被过滤文档 = 估算的误杀
        注意：这只是估算，不能替代人工标注。

        Args:
            filter_name: 过滤器名称
            quality_classifier: EvalQualityClassifier 实例
            high_quality_threshold: 质量分数阈值（高于此值认为被误杀）

        Returns:
            dict，包含估算的误杀率和高质量被过滤样本
        """
        records = self.filter_records.get(filter_name, [])
        if not records:
            return {}

        texts = [r["text_preview"] for r in records]
        scores = quality_classifier.score_batch(texts)

        estimated_false_kill_mask = scores > high_quality_threshold
        estimated_false_kill_rate = float(estimated_false_kill_mask.mean())

        # 找出被误杀的高质量样本
        high_quality_examples = []
        for i, (score, record) in enumerate(zip(scores, records)):
            if score > high_quality_threshold:
                high_quality_examples.append({
                    "quality_score": round(float(score), 4),
                    "text": record["text_preview"][:200],
                    "filter_reason": record.get("filter_reason", ""),
                })

        print(f"  🔍 {filter_name} 估算误杀率: {estimated_false_kill_rate:.1%} "
              f"（{int(estimated_false_kill_mask.sum())}/{len(records)} 条被认为是高质量）")

        return {
            "filter_name": filter_name,
            "estimated_false_kill_rate": round(estimated_false_kill_rate, 4),
            "high_quality_threshold": high_quality_threshold,
            "estimated_false_kills": int(estimated_false_kill_mask.sum()),
            "total_audited": len(records),
            "high_quality_examples": high_quality_examples[:5],
        }

    def generate_summary(self) -> Dict:
        """生成所有过滤器的汇总统计，保存为 JSON。"""
        summary = {
            "generated_at": datetime.now().isoformat(),
            "filter_stats": self.filter_stats,
            "audit_csv_paths": {
                k: str(self.output_dir / f"{k}_audit.csv")
                for k in self.filter_records
            },
        }

        summary_path = self.output_dir / "audit_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"  ✅ 审计汇总已保存: {summary_path}")
        return summary
