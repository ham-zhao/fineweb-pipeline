"""
src/gen1_zh/pipeline.py
中文 Gen1 Heuristic Pipeline

完整的中文预训练数据清洗流程：
  Step 1: 语言检测  — 确认文档主体为中文
  Step 2: 质量过滤  — ChineseQualityFilter（字符数/标点/重复/垃圾）
  Step 3: 简繁识别  — 记录简/繁体分布（不过滤，双语皆保留）
  Step 4: 去重前处理 — 标准化（全角→半角，多余空白压缩）

与英文 Gen1Pipeline 的关系：
  - 可独立使用，也可在英文 pipeline 后作第二道中文专项过滤
  - 评估指标与英文版一致（通过 StageTracker 记录）

使用方式：
    from src.gen1_zh.pipeline import ChineseGen1Pipeline
    pipeline = ChineseGen1Pipeline(run_config=cfg, pipeline_config=pipe_cfg)
    filtered = pipeline.run(docs)
    pipeline.save(filtered, Path("results/gen1_zh_output"))
"""

import re
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from src.gen1_zh.chinese_quality_filter import ChineseQualityFilter
from src.gen1_zh.chinese_text_utils import (
    count_chinese_chars,
    char_type_ratio,
    detect_script,
)


# ── 文本标准化 ────────────────────────────────────────────────────

# 全角→半角映射表（数字和英文字母）
_FULLWIDTH_MAP = str.maketrans(
    "ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ"
    "ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ"
    "０１２３４５６７８９",
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789",
)

_WHITESPACE_RE = re.compile(r"[ \t]+")
_BLANK_LINE_RE  = re.compile(r"\n{3,}")


def normalize_chinese_text(text: str) -> str:
    """
    标准化中文文本：
    1. 全角字母/数字 → 半角
    2. 连续空格/Tab 压缩为单空格
    3. 连续 3+ 个空行压缩为 2 个空行
    """
    text = text.translate(_FULLWIDTH_MAP)
    text = _WHITESPACE_RE.sub(" ", text)
    text = _BLANK_LINE_RE.sub("\n\n", text)
    return text.strip()


# ── 语言检测 ──────────────────────────────────────────────────────

def is_chinese_document(text: str, min_zh_ratio: float = 0.15) -> bool:
    """
    简单判断文档是否主体为中文（不依赖 langdetect/fastText）。
    中文字符占所有非空白字符的比例 ≥ min_zh_ratio。
    """
    if not text:
        return False
    ratios = char_type_ratio(text)
    return ratios["chinese"] >= min_zh_ratio


# ── Pipeline ──────────────────────────────────────────────────────

class ChineseGen1Pipeline:
    """
    中文 Gen1 Heuristic Pipeline。

    参数（来自 pipeline_config）：
      zh_pipeline:
        min_zh_ratio: 0.15         # 语言检测阈值
        normalize: true            # 是否做全角→半角标准化
        quality_filter:
          min_chinese_chars: 100
          max_chinese_chars: 50000
          min_zh_char_ratio: 0.20
          min_terminal_punct_ratio: 0.20
          max_duplicate_line_ratio: 0.50
          max_bigram_top_ratio: 0.30
          max_spam_score: 0.35
    """

    def __init__(
        self,
        run_config: Dict,
        pipeline_config: Optional[Dict] = None,
        stage_tracker=None,
    ):
        self.run_cfg = run_config
        self.pipe_cfg = pipeline_config or {}
        self.stage_tracker = stage_tracker

        zh_cfg = self.pipe_cfg.get("zh_pipeline", {})
        self.min_zh_ratio = zh_cfg.get("min_zh_ratio", 0.15)
        self.normalize = zh_cfg.get("normalize", True)

        # 质量过滤器参数
        qf_cfg = zh_cfg.get("quality_filter", {})
        self.quality_filter = ChineseQualityFilter(
            min_chinese_chars=qf_cfg.get("min_chinese_chars", 100),
            max_chinese_chars=qf_cfg.get("max_chinese_chars", 50000),
            min_zh_char_ratio=qf_cfg.get("min_zh_char_ratio", 0.20),
            min_terminal_punct_ratio=qf_cfg.get("min_terminal_punct_ratio", 0.20),
            max_duplicate_line_ratio=qf_cfg.get("max_duplicate_line_ratio", 0.50),
            max_bigram_top_ratio=qf_cfg.get("max_bigram_top_ratio", 0.30),
            max_spam_score=qf_cfg.get("max_spam_score", 0.35),
        )

        self._stats: List[Dict] = []

    def _record(self, step: str, docs: List[Dict], input_count: int) -> None:
        """记录步骤统计并通知 StageTracker。"""
        s = {
            "step": step,
            "input": input_count,
            "output": len(docs),
            "filtered": input_count - len(docs),
            "retention": round(len(docs) / input_count, 4) if input_count else 0,
        }
        self._stats.append(s)
        print(
            f"  [{step:30s}] {input_count:6,} → {len(docs):6,} "
            f"(-{s['filtered']:,}, {s['retention']:.1%})",
            flush=True,
        )
        if self.stage_tracker:
            texts = [d.get("text", "") for d in docs[:200]]
            urls  = [d.get("url", "") for d in docs[:200]]
            self.stage_tracker.record(f"zh_{step}", texts, urls)

    def run(self, docs: List[Dict]) -> List[Dict]:
        """
        执行完整的中文 Gen1 Pipeline。

        Returns:
            过滤后的文档列表（text 字段已标准化）
        """
        start = time.time()
        print(f"\n🇨🇳 中文 Gen1 Pipeline 开始 | 输入: {len(docs):,} 条", flush=True)
        print("=" * 55, flush=True)

        current = docs

        # ── Step 1: 语言检测 ─────────────────────────────────
        before = len(current)
        current = [
            d for d in current
            if is_chinese_document(d.get("text", ""), self.min_zh_ratio)
        ]
        self._record("language_detect", current, before)

        # ── Step 2: 文本标准化（可选）────────────────────────
        if self.normalize:
            for doc in current:
                doc["text"] = normalize_chinese_text(doc.get("text", ""))
            print(f"  [{'normalize':30s}] 全角→半角 + 空白压缩 完成", flush=True)

        # ── Step 3: 中文质量过滤 ────────────────────────────
        before = len(current)
        passed_docs = []
        fail_reasons = {}
        for doc in current:
            ok, reason = self.quality_filter.check(doc.get("text", ""))
            if ok:
                passed_docs.append(doc)
            else:
                key = reason.split(":")[0]
                fail_reasons[key] = fail_reasons.get(key, 0) + 1
        current = passed_docs
        self._record("quality_filter_zh", current, before)

        # 打印失败原因分布
        if fail_reasons:
            print("     过滤原因分布:", flush=True)
            for k, v in sorted(fail_reasons.items(), key=lambda x: -x[1])[:5]:
                print(f"       {k}: {v:,}", flush=True)

        # ── Step 4: 简繁体统计 ───────────────────────────────
        script_dist = {"simplified": 0, "traditional": 0, "mixed": 0, "unknown": 0}
        for doc in current:
            script = detect_script(doc.get("text", ""))
            script_dist[script] = script_dist.get(script, 0) + 1
            doc["zh_script"] = script  # 元数据注入，方便下游使用
        print(
            f"  [{'script_detection':30s}] 简: {script_dist['simplified']:,} | "
            f"繁: {script_dist['traditional']:,} | 混: {script_dist['mixed']:,}",
            flush=True,
        )

        elapsed = time.time() - start
        print("=" * 55, flush=True)
        print(
            f"✅ 中文 Gen1 完成 | 保留: {len(current):,}/{len(docs):,} "
            f"({len(current)/len(docs):.1%}) | 耗时: {elapsed:.1f}s",
            flush=True,
        )
        return current

    def get_stats(self) -> List[Dict]:
        return self._stats

    def save(self, docs: List[Dict], output_dir: Path) -> None:
        """保存过滤后的文档和统计信息。"""
        output_dir.mkdir(parents=True, exist_ok=True)

        out_file = output_dir / "gen1_zh_output.jsonl"
        with open(out_file, "w", encoding="utf-8") as f:
            for doc in docs:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
        print(f"  📄 输出: {out_file} ({len(docs):,} 条)", flush=True)

        stats_file = output_dir / "gen1_zh_stats.json"
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "pipeline": "ChineseGen1Pipeline",
                    "output_count": len(docs),
                    "steps": self._stats,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"  📊 统计: {stats_file}", flush=True)
