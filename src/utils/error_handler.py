"""
src/utils/error_handler.py
Pipeline 统一错误处理

产出文件：
  data/{gen}_output/error_summary.json - 错误摘要
"""

import json
from pathlib import Path
from typing import Dict, Optional
from collections import Counter


class PipelineError(Exception):
    """Pipeline 关键错误（缺失依赖、配置错误等），应立即停止。"""
    pass


class ErrorAccumulator:
    """
    累积错误收集器：记录每条失败文档的信息，超过阈值自动停止。

    用法：
        acc = ErrorAccumulator(error_rate_threshold=0.20)
        for doc in docs:
            try:
                result = process(doc)
                acc.record_success()
            except Exception as e:
                acc.record_error(doc_id=doc.get("url", ""), error=str(e))
                if acc.should_stop():
                    raise PipelineError(acc.get_summary_text())
    """

    def __init__(self, error_rate_threshold: float = 0.20):
        self.threshold = error_rate_threshold
        self.total = 0
        self.success = 0
        self.errors = []
        self.error_reasons = Counter()

    def record_success(self):
        self.total += 1
        self.success += 1

    def record_error(self, doc_id: str = "", error: str = ""):
        self.total += 1
        reason = error[:100] if error else "unknown"
        self.errors.append({"doc_id": doc_id, "error": reason})
        self.error_reasons[reason] += 1

    @property
    def error_count(self) -> int:
        return len(self.errors)

    @property
    def error_rate(self) -> float:
        return self.error_count / self.total if self.total > 0 else 0.0

    def should_stop(self) -> bool:
        return self.total >= 10 and self.error_rate > self.threshold

    def get_summary(self) -> Dict:
        top_reasons = self.error_reasons.most_common(3)
        return {
            "total_processed": self.total,
            "success_count": self.success,
            "error_count": self.error_count,
            "error_rate": round(self.error_rate, 4),
            "top_3_errors": [{"reason": r, "count": c} for r, c in top_reasons],
            "threshold": self.threshold,
            "stopped_early": self.should_stop(),
        }

    def get_summary_text(self) -> str:
        s = self.get_summary()
        lines = [
            f"Pipeline Error Summary: {s['error_count']}/{s['total_processed']} failed ({s['error_rate']:.1%})",
        ]
        for item in s["top_3_errors"]:
            lines.append(f"  - {item['reason']} ({item['count']}x)")
        return "\n".join(lines)

    def save(self, output_path: str) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.get_summary(), f, ensure_ascii=False, indent=2)
