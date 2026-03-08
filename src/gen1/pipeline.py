"""
src/gen1/pipeline.py
第一代 Pipeline：Heuristic Filtering

执行顺序（按 FineWeb 实际顺序）：
  Step 0: WARC 读取 + Trafilatura 文本提取
  Step 1: URL Filter
  Step 2: Language Filter
  Step 3: Gopher + C4 + FineWeb Quality Filter
  Step 4: Gopher Repetition Filter
  Step 5: PII Formatter（脱敏）
  Step 6: Toxicity Filter

每步记录：过滤前后文档数 + 质量指标（通过 StageTracker）
"""

import json
import time
import random
from pathlib import Path
from typing import List, Dict, Optional, Iterator
from tqdm import tqdm

from src.utils.config_loader import load_run_config, load_pipeline_config, load_eval_config, get_output_path
from src.gen1.url_dedup import url_dedup
from src.gen1.filters.url_filter import URLFilter
from src.gen1.filters.language_filter import LanguageFilter
from src.gen1.filters.quality_filter import QualityFilter
from src.gen1.filters.repetition_filter import GopherRepetitionFilter
from src.gen1.filters.pii_filter import PIIFilter
from src.gen1.filters.toxicity_filter import ToxicityFilter


def read_wet_texts(wet_path: Path, doc_limit: int = None) -> List[Dict]:
    """
    读取 WET 文件（WARC Encapsulated Text）——CC 纯文本格式。
    WET 记录类型为 "conversion"（区别于 WARC 的 "response"），直接包含纯文本。
    """
    from warcio.archiveiterator import ArchiveIterator

    docs = []
    print(f"  Reading WET: {wet_path.name}")

    with open(wet_path, "rb") as f:
        for record in ArchiveIterator(f):
            if record.rec_type != "conversion":
                continue
            url = record.rec_headers.get_header("WARC-Target-URI", "")
            text = record.content_stream().read().decode("utf-8", errors="replace").strip()
            if len(text) > 50:
                docs.append({
                    "text": text,
                    "url": url,
                    "source": "common_crawl_wet",
                })
            if doc_limit and len(docs) >= doc_limit:
                break

    print(f"  WET extracted: {len(docs):,} docs")
    return docs


def read_warc_texts(warc_path: Path, doc_limit: int = None) -> List[Dict]:
    """
    读取 WARC 文件，用 Trafilatura 提取文本。

    Returns:
        list of dict，每条包含 {"text": ..., "url": ..., "date": ...}
    """
    import trafilatura
    from warcio.archiveiterator import ArchiveIterator

    docs = []
    print(f"  📂 读取 WARC: {warc_path.name}")

    with open(warc_path, "rb") as f:
        for record in tqdm(ArchiveIterator(f), desc="  解析 WARC"):
            if record.rec_type != "response":
                continue

            url = record.rec_headers.get_header("WARC-Target-URI", "")
            content = record.content_stream().read()

            # 只处理 HTML 内容
            if b"<html" not in content.lower()[:1000]:
                continue

            # Trafilatura 提取正文
            text = trafilatura.extract(
                content,
                include_tables=False,
                include_images=False,
                include_links=False,
                no_fallback=False,
            )
            if text and len(text.strip()) > 50:
                docs.append({
                    "text": text.strip(),
                    "url": url,
                    "source": "common_crawl",
                })

            if doc_limit and len(docs) >= doc_limit:
                break

    print(f"  ✅ 提取文本: {len(docs):,} 条文档")
    return docs


# re-export from canonical location for backward compatibility
from src.utils.io import read_jsonl, save_jsonl  # noqa: F401


class Gen1Pipeline:
    """
    第一代 Heuristic Filtering Pipeline。
    按顺序执行 6 个过滤步骤，每步记录统计信息。
    """

    def __init__(
        self,
        run_config: Dict,
        pipeline_config: Dict,
        stage_tracker=None,
        filter_auditor=None,
    ):
        self.run_cfg = run_config
        self.pipe_cfg = pipeline_config
        self.tracker = stage_tracker
        self.auditor = filter_auditor
        self.stats: List[Dict] = []   # 每步的统计记录

        # 初始化各过滤器
        filter_cfg = pipeline_config.get("filters", {})

        self.url_filter = URLFilter(
            blacklist_domains=filter_cfg.get("url_filter", {}).get("blacklist_domains", []),
            blacklist_keywords=filter_cfg.get("url_filter", {}).get("blacklist_keywords", []),
        ) if filter_cfg.get("url_filter", {}).get("enabled", True) else None

        self.lang_filter = LanguageFilter(
            target_language=filter_cfg.get("language_filter", {}).get("target_language", "en"),
            min_confidence=filter_cfg.get("language_filter", {}).get("min_confidence", 0.65),
        ) if filter_cfg.get("language_filter", {}).get("enabled", True) else None

        goph_cfg = filter_cfg.get("gopher_quality", {})
        c4_cfg = filter_cfg.get("c4_quality", {})
        fw_cfg = filter_cfg.get("fineweb_quality", {})

        # Pass config thresholds through to sub-filters
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

        self.quality_filter = QualityFilter(
            use_gopher=goph_cfg.get("enabled", True),
            use_c4=c4_cfg.get("enabled", True),
            use_fineweb=fw_cfg.get("enabled", True),
            gopher_kwargs=gopher_kwargs,
            c4_kwargs=c4_kwargs,
            fineweb_kwargs=fineweb_kwargs,
        )

        rep_cfg = filter_cfg.get("gopher_repetition", {})
        self.repetition_filter = GopherRepetitionFilter(
        ) if rep_cfg.get("enabled", True) else None

        pii_cfg = filter_cfg.get("pii_filter", {})
        self.pii_filter = PIIFilter(
            mode=pii_cfg.get("mode", "mask"),
        ) if pii_cfg.get("enabled", True) else None

        tox_cfg = filter_cfg.get("toxicity_filter", {})
        self.toxicity_filter = ToxicityFilter(
            toxicity_threshold=tox_cfg.get("toxicity_threshold", 0.85),
            severe_toxicity_threshold=tox_cfg.get("severe_toxicity_threshold", 0.5),
            action=tox_cfg.get("action", "filter"),
        ) if tox_cfg.get("enabled", True) else None

    def _record_step(self, step_name: str, docs: List[Dict], filtered_docs: List[Dict]) -> Dict:
        """记录单步统计，并通知 auditor 保存被过滤样本。"""
        from collections import Counter
        stat = {
            "step": step_name,
            "before": len(docs) + len(filtered_docs),
            "after": len(docs),
            "filtered": len(filtered_docs),
            "filter_rate": len(filtered_docs) / (len(docs) + len(filtered_docs))
                          if (len(docs) + len(filtered_docs)) > 0 else 0,
        }

        # Collect sub-filter reason breakdown from _filter_reason field
        if filtered_docs:
            reasons = [d.get("_filter_reason", "unknown") for d in filtered_docs]
            # Top-level category (e.g. "gopher", "c4", "fineweb", "dup_line_fraction")
            top_reasons = Counter(r.split(":")[0] for r in reasons)
            stat["reason_breakdown"] = dict(top_reasons.most_common())
            # Detailed reasons (e.g. "gopher:too_short:42<50")
            detail_reasons = Counter(r for r in reasons)
            stat["detail_breakdown"] = dict(detail_reasons.most_common(100))

        self.stats.append(stat)

        # 通知 auditor
        if self.auditor and filtered_docs:
            audit_size = self.run_cfg.get("audit_sample_size", 20)
            sample = random.sample(filtered_docs, min(audit_size, len(filtered_docs)))
            self.auditor.record_filtered(
                filter_name=step_name,
                filtered_texts=[d["text"] for d in sample],
                filtered_meta=[{"url": d.get("url", ""), "filter_reason": d.get("_filter_reason", "")} for d in sample],
                total_docs_before=stat["before"],
            )

        print(f"  [{step_name}] {stat['before']:,} → {stat['after']:,} | "
              f"过滤: {stat['filtered']:,} ({stat['filter_rate']:.1%})")

        return stat

    def run(self, docs: List[Dict]) -> List[Dict]:
        """
        执行完整的第一代 Pipeline。

        Args:
            docs: 输入文档列表（来自 WARC 或 JSONL）

        Returns:
            过滤后的文档列表
        """
        print(f"\n{'='*60}")
        print(f"  第一代 Pipeline 启动 | 输入: {len(docs):,} 条")
        print(f"  模式: {self.run_cfg.get('run_mode', 'smoke_test')}")
        print(f"{'='*60}")
        start_time = time.time()

        # StageTracker 记录基线
        if self.tracker:
            self.tracker.record("gen1_input", [d["text"] for d in docs],
                                urls=[d.get("url", "") for d in docs])

        # ── Step 0: URL Dedup ─────────────────────────────────
        docs, dedup_stats = url_dedup(docs)
        self.stats.append({
            "step": "url_dedup",
            "before": dedup_stats["input_count"],
            "after": dedup_stats["output_count"],
            "filtered": dedup_stats["removed_count"],
            "filter_rate": dedup_stats["dedup_rate"],
        })
        print(f"  [url_dedup] {dedup_stats['input_count']:,} → {dedup_stats['output_count']:,} | "
              f"去重: {dedup_stats['removed_count']:,} ({dedup_stats['dedup_rate']:.1%})")

        # ── Step 1: URL Filter ────────────────────────────────
        if self.url_filter:
            kept, filtered = [], []
            for doc in docs:
                should_filter, reason = self.url_filter.should_filter(doc.get("url", ""))
                if should_filter:
                    doc["_filter_reason"] = reason
                    filtered.append(doc)
                else:
                    kept.append(doc)
            self._record_step("url_filter", kept, filtered)
            docs = kept

        # ── Step 2: Language Filter ───────────────────────────
        if self.lang_filter:
            kept, filtered = [], []
            for doc in docs:
                should_filter, lang, conf = self.lang_filter.should_filter(doc["text"])
                doc["_detected_lang"] = lang
                doc["_lang_confidence"] = conf
                if should_filter:
                    doc["_filter_reason"] = f"lang:{lang}(conf={conf:.2f})"
                    filtered.append(doc)
                else:
                    kept.append(doc)
            self._record_step("language_filter", kept, filtered)
            docs = kept

        if self.tracker:
            self.tracker.record("gen1_after_lang", [d["text"] for d in docs],
                                urls=[d.get("url", "") for d in docs])

        # ── Step 3: Quality Filter (Gopher + C4 + FineWeb) ───
        kept, filtered = [], []
        for doc in docs:
            passes, reason = self.quality_filter.check(doc["text"])
            if not passes:
                doc["_filter_reason"] = reason
                filtered.append(doc)
            else:
                kept.append(doc)
        self._record_step("quality_filter", kept, filtered)
        docs = kept

        if self.tracker:
            self.tracker.record("gen1_after_quality", [d["text"] for d in docs],
                                urls=[d.get("url", "") for d in docs])

        # ── Step 4: Repetition Filter ─────────────────────────
        if self.repetition_filter:
            kept, filtered = [], []
            for doc in docs:
                passes, reason = self.repetition_filter.check(doc["text"])
                if not passes:
                    doc["_filter_reason"] = reason
                    filtered.append(doc)
                else:
                    kept.append(doc)
            self._record_step("repetition_filter", kept, filtered)
            docs = kept

        # ── Step 5: PII Formatter ─────────────────────────────
        if self.pii_filter:
            kept, filtered = [], []
            for doc in docs:
                processed_text, action, pii_stats = self.pii_filter.process(doc["text"])
                if processed_text is None:
                    doc["_filter_reason"] = f"pii_density:{action}"
                    filtered.append(doc)
                else:
                    doc["text"] = processed_text  # 更新为脱敏后的文本
                    if pii_stats:
                        doc["_pii_masked"] = pii_stats
                    kept.append(doc)
            self._record_step("pii_filter", kept, filtered)
            docs = kept

        # ── Step 6: Toxicity Filter ───────────────────────────
        if self.toxicity_filter:
            kept, filtered = [], []
            masks, scores, reasons = self.toxicity_filter.filter_batch(
                [d["text"] for d in docs]
            )
            for doc, should_filter, score, reason in zip(docs, masks, scores, reasons):
                doc["_toxicity_score"] = round(score, 4)
                if should_filter:
                    doc["_filter_reason"] = reason
                    filtered.append(doc)
                else:
                    kept.append(doc)
            self._record_step("toxicity_filter", kept, filtered)
            docs = kept

        # ── 完成 ──────────────────────────────────────────────
        elapsed = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"  ✅ 第一代 Pipeline 完成！")
        print(f"  输入: {self.stats[0]['before'] if self.stats else 0:,} 条")
        print(f"  输出: {len(docs):,} 条")
        print(f"  总保留率: {len(docs)/self.stats[0]['before']:.1%}" if self.stats else "")
        print(f"  耗时: {elapsed:.1f}s")
        print(f"{'='*60}")

        if self.tracker:
            self.tracker.record("gen1_output", [d["text"] for d in docs],
                                urls=[d.get("url", "") for d in docs])

        return docs

    def get_pipeline_stats(self) -> List[Dict]:
        """返回各步骤统计（用于 Notebook 绘制瀑布图）。"""
        return self.stats
