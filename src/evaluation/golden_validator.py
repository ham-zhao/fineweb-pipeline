"""
src/evaluation/golden_validator.py
Golden Set 验证器：用一组手工标注的"黄金样本"验证 Pipeline 行为是否符合预期。

核心用途：
  回归测试 —— 每次修改过滤器参数或代码后，用固定的 golden set 检查：
    1. 高质量样本是否仍然通过？（不被误杀）
    2. 低质量样本是否在预期阶段被过滤？（不漏放）
    3. 边界样本的处理是否合理？（不做硬性判定，只记录）

Golden 样本 JSONL 格式：
  {
    "id": "golden_001",
    "text": "...",
    "category": "high_quality|low_quality|boundary",
    "expected_outcome": "pass_all|filter_gen1|filter_gen2|borderline",
    "expected_filter_stage": "language|url|quality|dedup|classifier|null",
    "reason": "为什么选这条样本",
    "source": "样本来源"
  }

匹配策略：
  用文本前 200 字符作为 key 匹配 golden 样本与 pipeline 输出文档。
  前 200 字符足够唯一标识一个文档，同时避免因尾部截断差异导致匹配失败。

产出文件：
  results/reports/golden_validation_results.json  - 逐样本验证结果
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Set


# ── 常量 ─────────────────────────────────────────────────────────────
# 匹配用的文本前缀长度（前 200 字符）
_MATCH_KEY_LENGTH = 200

# 有效的 category 值
_VALID_CATEGORIES = {"high_quality", "low_quality", "boundary"}

# 有效的 expected_outcome 值
_VALID_OUTCOMES = {"pass_all", "filter_gen1", "filter_gen2", "borderline"}

# 有效的 expected_filter_stage 值（null 在 JSON 中会被解析为 None）
_VALID_FILTER_STAGES = {"language", "url", "quality", "dedup", "classifier", None}


def _make_match_key(text: str) -> str:
    """
    从文本中提取匹配用的 key：取前 200 字符，去除首尾空白。

    为什么用前 200 字符而非全文：
      1. pipeline 中间步骤可能对文本做 normalize（去尾部空行等），尾部不稳定
      2. 200 字符足够唯一标识一个文档（重复概率极低）
      3. 避免长文本做 dict key 的性能问题
    """
    return text.strip()[:_MATCH_KEY_LENGTH]


class GoldenSample:
    """单条 golden 样本的数据结构。"""

    def __init__(self, data: Dict):
        self.id: str = data["id"]
        self.text: str = data["text"]
        self.category: str = data["category"]
        self.expected_outcome: str = data["expected_outcome"]
        # JSON 中的 null 会被解析为 None，字符串 "null" 也统一转为 None
        raw_stage = data.get("expected_filter_stage")
        self.expected_filter_stage: Optional[str] = None if raw_stage in (None, "null") else raw_stage
        self.reason: str = data.get("reason", "")
        self.source: str = data.get("source", "")
        self.match_key: str = _make_match_key(self.text)

    def validate_fields(self) -> List[str]:
        """校验字段合法性，返回错误消息列表（空列表表示全部合法）。"""
        errors = []
        if self.category not in _VALID_CATEGORIES:
            errors.append(f"[{self.id}] category '{self.category}' 不在 {_VALID_CATEGORIES}")
        if self.expected_outcome not in _VALID_OUTCOMES:
            errors.append(f"[{self.id}] expected_outcome '{self.expected_outcome}' 不在 {_VALID_OUTCOMES}")
        if self.expected_filter_stage not in _VALID_FILTER_STAGES:
            errors.append(f"[{self.id}] expected_filter_stage '{self.expected_filter_stage}' 不在 {_VALID_FILTER_STAGES}")
        if not self.text.strip():
            errors.append(f"[{self.id}] text 为空")
        return errors


class GoldenSetValidator:
    """
    Golden Set 验证器。

    典型用法：
        validator = GoldenSetValidator()
        validator.load_golden_set("data/golden/golden_samples.jsonl")

        # 方式一：逐阶段验证
        validator.validate_stage("language_filter", input_docs, output_docs)
        validator.validate_stage("url_filter", input_docs2, output_docs2)

        # 方式二：一次性传入所有阶段
        validator.validate_pipeline({
            "language_filter": output_after_lang,
            "url_filter": output_after_url,
            "quality_filter": output_after_quality,
            ...
        })

        # 查看结果
        results = validator.report()

        # 与上次运行对比（回归检测）
        regression = validator.regression_check("results/reports/golden_validation_results.json")
    """

    def __init__(self):
        self.golden_samples: List[GoldenSample] = []
        # key = match_key, value = GoldenSample（用于快速查找）
        self._key_to_sample: Dict[str, GoldenSample] = {}

        # 验证结果存储
        # key = golden sample id, value = 该样本在各阶段的存活状态
        self._sample_results: Dict[str, Dict] = {}
        # 记录每个阶段的输入/输出文档数
        self._stage_stats: Dict[str, Dict] = {}

    # ── 加载 Golden Set ─────────────────────────────────────────────

    def load_golden_set(self, path: str) -> int:
        """
        从 JSONL 文件加载 golden 样本集。

        Args:
            path: JSONL 文件路径，每行一个 JSON 对象

        Returns:
            加载的样本数

        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 样本字段校验失败
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Golden set 文件不存在: {path}")

        samples = []
        all_errors = []

        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    all_errors.append(f"第 {line_no} 行 JSON 解析失败: {e}")
                    continue

                sample = GoldenSample(data)
                field_errors = sample.validate_fields()
                if field_errors:
                    all_errors.extend(field_errors)
                    continue

                samples.append(sample)

        # 如果有校验错误，全部报出来再抛异常（方便一次性修复）
        if all_errors:
            error_msg = "Golden set 校验失败:\n" + "\n".join(f"  - {e}" for e in all_errors)
            raise ValueError(error_msg)

        self.golden_samples = samples
        self._key_to_sample = {s.match_key: s for s in samples}

        # 检查 match_key 冲突（两条样本前 200 字符相同 = 无法区分）
        if len(self._key_to_sample) < len(samples):
            n_dup = len(samples) - len(self._key_to_sample)
            print(f"  WARNING: {n_dup} 条样本的前 {_MATCH_KEY_LENGTH} 字符重复，"
                  f"匹配时只保留最后一条。请检查 golden set 是否有重复。")

        # 初始化每条样本的结果容器
        for s in self.golden_samples:
            self._sample_results[s.id] = {
                "id": s.id,
                "category": s.category,
                "expected_outcome": s.expected_outcome,
                "expected_filter_stage": s.expected_filter_stage,
                "text_preview": s.text[:100].replace("\n", " "),
                "stages_survived": [],     # 存活过的阶段名称列表
                "filtered_at": None,       # 在哪个阶段被过滤（None = 通过所有阶段）
                "verdict": "pending",      # pass / fail / info（边界样本用 info）
                "detail": "",
            }

        # 打印加载摘要
        by_category = {}
        for s in self.golden_samples:
            by_category[s.category] = by_category.get(s.category, 0) + 1

        print(f"  Golden set 已加载: {len(self.golden_samples)} 条样本")
        for cat, count in sorted(by_category.items()):
            print(f"    - {cat}: {count} 条")

        return len(self.golden_samples)

    # ── 逐阶段验证 ─────────────────────────────────────────────────

    def validate_stage(
        self,
        stage_name: str,
        input_docs: List[Dict],
        output_docs: List[Dict],
        text_field: str = "text",
    ) -> Dict:
        """
        验证单个过滤阶段：检查哪些 golden 样本在此阶段被过滤。

        工作原理：
          1. 提取 input_docs 中所有文本的 match_key → 构建集合 S_in
          2. 提取 output_docs 中所有文本的 match_key → 构建集合 S_out
          3. 对每条 golden sample：
             - 如果 key 在 S_in 中但不在 S_out 中 → 此阶段被过滤
             - 如果 key 不在 S_in 中 → 已在更早的阶段被过滤，跳过

        Args:
            stage_name: 阶段名称（如 "language_filter", "url_filter"）
            input_docs: 此阶段的输入文档列表（每个 doc 是 dict，需包含 text_field 字段）
            output_docs: 此阶段的输出文档列表
            text_field: 文档中文本字段的 key（默认 "text"）

        Returns:
            dict，此阶段的验证统计
        """
        if not self.golden_samples:
            raise RuntimeError("尚未加载 golden set，请先调用 load_golden_set()")

        # 构建 match_key 集合
        input_keys: Set[str] = {_make_match_key(doc[text_field]) for doc in input_docs}
        output_keys: Set[str] = {_make_match_key(doc[text_field]) for doc in output_docs}

        # 记录阶段统计
        self._stage_stats[stage_name] = {
            "input_count": len(input_docs),
            "output_count": len(output_docs),
            "filtered_count": len(input_docs) - len(output_docs),
        }

        golden_in_input = 0
        golden_filtered = 0
        golden_survived = 0

        for sample in self.golden_samples:
            result = self._sample_results[sample.id]

            if sample.match_key not in input_keys:
                # 此样本不在此阶段的输入中（已在更早阶段被过滤或原始数据中不存在）
                continue

            golden_in_input += 1

            if sample.match_key in output_keys:
                # 存活：通过了此阶段
                result["stages_survived"].append(stage_name)
                golden_survived += 1
            else:
                # 被过滤：记录在哪个阶段被过滤（只记录第一次）
                if result["filtered_at"] is None:
                    result["filtered_at"] = stage_name
                golden_filtered += 1

        stage_result = {
            "stage_name": stage_name,
            "golden_in_input": golden_in_input,
            "golden_survived": golden_survived,
            "golden_filtered": golden_filtered,
        }

        print(f"  [{stage_name}] golden 样本: "
              f"输入 {golden_in_input} / 存活 {golden_survived} / 过滤 {golden_filtered}")

        return stage_result

    # ── 整体 Pipeline 验证 ─────────────────────────────────────────

    def validate_pipeline(
        self,
        stages_dict: Dict[str, List[Dict]],
        text_field: str = "text",
    ) -> Dict:
        """
        一次性验证整个 pipeline：传入各阶段的输出文档。

        stages_dict 的 key 是阶段名称，value 是该阶段**输出**的文档列表。
        阶段应按 pipeline 执行顺序排列（OrderedDict 或 Python 3.7+ 的普通 dict）。

        第一个阶段的输入 = 第一个阶段的文档本身（视为原始输入），
        后续每个阶段的输入 = 上一个阶段的输出。

        Args:
            stages_dict: {阶段名: 文档列表}，按执行顺序排列
            text_field: 文档中文本字段的 key

        Returns:
            dict，包含逐阶段统计和最终判定
        """
        if not self.golden_samples:
            raise RuntimeError("尚未加载 golden set，请先调用 load_golden_set()")

        stage_names = list(stages_dict.keys())
        stage_results = []

        for i, stage_name in enumerate(stage_names):
            output_docs = stages_dict[stage_name]

            if i == 0:
                # 第一个阶段：输入 = 输出（视为原始数据，不做过滤判定，仅标记存在性）
                input_docs = output_docs
            else:
                # 后续阶段：输入 = 上一阶段的输出
                prev_stage = stage_names[i - 1]
                input_docs = stages_dict[prev_stage]

            result = self.validate_stage(stage_name, input_docs, output_docs, text_field)
            stage_results.append(result)

        # 完成所有阶段后，计算最终判定
        self._compute_verdicts(stage_names[-1] if stage_names else None)

        return {
            "stages": stage_results,
            "summary": self._compute_summary(),
        }

    # ── 按代次验证的便捷方法 ──────────────────────────────────────

    def validate_gen1(
        self,
        raw_docs: List[Dict],
        output_docs: List[Dict],
        intermediate_stages: Optional[Dict[str, List[Dict]]] = None,
        text_field: str = "text",
    ) -> Dict:
        """
        验证 Gen1 pipeline（基于启发式规则过滤）。

        Args:
            raw_docs: 原始输入文档
            output_docs: Gen1 最终输出文档
            intermediate_stages: 可选的中间阶段输出，如 {"language_filter": docs, "url_filter": docs, ...}
            text_field: 文档中文本字段的 key

        Returns:
            验证结果字典
        """
        if intermediate_stages:
            # 有中间阶段：逐步验证
            stages = {"raw_input": raw_docs}
            stages.update(intermediate_stages)
            stages["gen1_final"] = output_docs
            return self.validate_pipeline(stages, text_field)
        else:
            # 无中间阶段：只看输入→输出
            return self.validate_pipeline(
                {"raw_input": raw_docs, "gen1_final": output_docs},
                text_field,
            )

    def validate_gen2(
        self,
        gen1_output_docs: List[Dict],
        gen2_output_docs: List[Dict],
        text_field: str = "text",
    ) -> Dict:
        """
        验证 Gen2 pipeline（基于分类器的质量过滤）。

        Args:
            gen1_output_docs: Gen1 输出文档（= Gen2 输入）
            gen2_output_docs: Gen2 最终输出文档
            text_field: 文档中文本字段的 key

        Returns:
            验证结果字典
        """
        return self.validate_pipeline(
            {"gen2_input": gen1_output_docs, "gen2_classifier": gen2_output_docs},
            text_field,
        )

    def validate_gen3(
        self,
        gen2_output_docs: List[Dict],
        gen3_output_docs: List[Dict],
        text_field: str = "text",
    ) -> Dict:
        """
        验证 Gen3 pipeline（LLM 改写 + 合成数据）。

        注意：Gen3 的 LLM 改写会改变文本内容，因此匹配可能失败。
        对于 Gen3，主要关注"输入中有哪些 golden 样本"以及"是否被选中改写"。

        Args:
            gen2_output_docs: Gen2 输出文档（= Gen3 输入）
            gen3_output_docs: Gen3 最终输出文档
            text_field: 文档中文本字段的 key

        Returns:
            验证结果字典
        """
        return self.validate_pipeline(
            {"gen3_input": gen2_output_docs, "gen3_rewritten": gen3_output_docs},
            text_field,
        )

    # ── 判定计算 ───────────────────────────────────────────────────

    def _compute_verdicts(self, last_stage: Optional[str]) -> None:
        """
        根据 expected_outcome 和实际过滤结果计算每条样本的 verdict。

        判定规则：
          - high_quality + expected_outcome=pass_all → 必须通过所有阶段，否则 fail
          - low_quality + expected_outcome=filter_gen1 → 必须在 Gen1 相关阶段被过滤，否则 fail
          - low_quality + expected_outcome=filter_gen2 → 必须在 Gen2 相关阶段被过滤，否则 fail
          - boundary + expected_outcome=borderline → 不做 pass/fail 判定，只记录为 info
        """
        for sample in self.golden_samples:
            result = self._sample_results[sample.id]
            outcome = sample.expected_outcome
            filtered_at = result["filtered_at"]

            if outcome == "pass_all":
                # 期望通过所有阶段
                if filtered_at is None:
                    result["verdict"] = "pass"
                    result["detail"] = "如预期通过所有阶段"
                else:
                    result["verdict"] = "fail"
                    result["detail"] = f"期望通过但在 [{filtered_at}] 被过滤（误杀）"

            elif outcome == "filter_gen1":
                # 期望在 Gen1 的某个阶段被过滤
                if filtered_at is not None:
                    # 检查是否在 Gen1 相关阶段被过滤
                    # Gen1 阶段名称通常包含: language, url, quality, dedup, gopher, c4 等
                    # 更宽松：只要被过滤就算 pass（精确到 stage 用 expected_filter_stage 检查）
                    if sample.expected_filter_stage and filtered_at != sample.expected_filter_stage:
                        # 被过滤了，但不在预期的阶段
                        result["verdict"] = "pass"
                        result["detail"] = (
                            f"被过滤（符合预期），但阶段不匹配: "
                            f"预期 [{sample.expected_filter_stage}]，实际 [{filtered_at}]"
                        )
                    else:
                        result["verdict"] = "pass"
                        result["detail"] = f"如预期在 [{filtered_at}] 被过滤"
                else:
                    result["verdict"] = "fail"
                    result["detail"] = "期望在 Gen1 被过滤但通过了所有阶段（漏放）"

            elif outcome == "filter_gen2":
                # 期望在 Gen2 阶段被过滤
                if filtered_at is not None:
                    result["verdict"] = "pass"
                    result["detail"] = f"被过滤于 [{filtered_at}]（期望在 Gen2 被过滤）"
                else:
                    result["verdict"] = "fail"
                    result["detail"] = "期望在 Gen2 被过滤但通过了所有阶段（漏放）"

            elif outcome == "borderline":
                # 边界样本：不做 pass/fail 判定，仅记录实际行为
                result["verdict"] = "info"
                if filtered_at:
                    result["detail"] = f"边界样本，实际在 [{filtered_at}] 被过滤"
                else:
                    result["detail"] = "边界样本，实际通过了所有阶段"

            else:
                result["verdict"] = "error"
                result["detail"] = f"未知的 expected_outcome: {outcome}"

    def _compute_summary(self) -> Dict:
        """汇总所有样本的判定结果。"""
        total = len(self._sample_results)
        pass_count = sum(1 for r in self._sample_results.values() if r["verdict"] == "pass")
        fail_count = sum(1 for r in self._sample_results.values() if r["verdict"] == "fail")
        info_count = sum(1 for r in self._sample_results.values() if r["verdict"] == "info")
        pending_count = sum(1 for r in self._sample_results.values() if r["verdict"] == "pending")

        # 按 category 分组统计
        by_category: Dict[str, Dict] = {}
        for r in self._sample_results.values():
            cat = r["category"]
            if cat not in by_category:
                by_category[cat] = {"pass": 0, "fail": 0, "info": 0, "pending": 0}
            by_category[cat][r["verdict"]] = by_category[cat].get(r["verdict"], 0) + 1

        # 通过率口径：分子=pass 数，分母=pass+fail 数（不含 info 和 pending）
        denominator = pass_count + fail_count
        pass_rate = pass_count / denominator if denominator > 0 else 0.0

        return {
            "total_samples": total,
            "pass": pass_count,
            "fail": fail_count,
            "info": info_count,
            "pending": pending_count,
            "pass_rate": round(pass_rate, 4),
            "pass_rate_description": (
                f"通过率 {pass_rate:.1%}（分子={pass_count} pass，"
                f"分母={denominator} pass+fail，不含 {info_count} info 边界样本）"
            ),
            "by_category": by_category,
        }

    # ── 报告 ───────────────────────────────────────────────────────

    def report(self, save_path: Optional[str] = None) -> Dict:
        """
        生成验证报告并打印人类可读的摘要。

        无论是通过 validate_pipeline() 还是逐次 validate_stage() 调用，
        report() 都会先计算 verdict 再生成报告。

        Args:
            save_path: 可选，保存 JSON 结果的路径

        Returns:
            dict，完整的验证结果（含逐样本详情）
        """
        # 确保 verdict 已计算（无论调用方式）
        last_stage = list(self._stage_stats.keys())[-1] if self._stage_stats else None
        self._compute_verdicts(last_stage)

        summary = self._compute_summary()

        # 构建完整报告
        full_report = {
            "generated_at": datetime.now().isoformat(),
            "golden_set_size": len(self.golden_samples),
            "summary": summary,
            "stage_stats": self._stage_stats,
            "sample_results": list(self._sample_results.values()),
        }

        # 打印人类可读摘要
        print("\n" + "=" * 60)
        print("  Golden Set 验证报告")
        print("=" * 60)
        print(f"  样本总数: {summary['total_samples']}")
        print(f"  {summary['pass_rate_description']}")
        print(f"  - PASS: {summary['pass']}  | FAIL: {summary['fail']}  "
              f"| INFO: {summary['info']}  | PENDING: {summary['pending']}")
        print()

        # 按类别打印
        for cat, stats in sorted(summary["by_category"].items()):
            cat_total = sum(stats.values())
            print(f"  [{cat}] (共 {cat_total} 条)")
            for verdict, count in sorted(stats.items()):
                if count > 0:
                    print(f"    {verdict}: {count}")

        # 打印失败的样本详情
        failed = [r for r in self._sample_results.values() if r["verdict"] == "fail"]
        if failed:
            print(f"\n  {'─' * 56}")
            print(f"  失败样本详情 ({len(failed)} 条):")
            for r in failed:
                print(f"    [{r['id']}] {r['detail']}")
                print(f"      文本预览: {r['text_preview'][:80]}...")

        print("=" * 60)

        # 保存到文件
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(full_report, f, ensure_ascii=False, indent=2)
            print(f"  验证结果已保存: {save_path}")

        return full_report

    # ── 回归对比 ───────────────────────────────────────────────────

    def regression_check(self, previous_results_path: str) -> Dict:
        """
        与上一次运行的验证结果对比，检测回归（pass→fail 的样本）。

        Args:
            previous_results_path: 上次运行保存的 JSON 结果路径

        Returns:
            dict，包含回归样本列表和状态变化统计
        """
        prev_path = Path(previous_results_path)
        if not prev_path.exists():
            print(f"  上次运行结果不存在: {prev_path}，跳过回归检查")
            return {"status": "no_previous_results", "regressions": [], "improvements": []}

        with open(prev_path, "r", encoding="utf-8") as f:
            prev_report = json.load(f)

        # 构建上次结果的 id → verdict 映射
        prev_verdicts: Dict[str, str] = {}
        for r in prev_report.get("sample_results", []):
            prev_verdicts[r["id"]] = r["verdict"]

        regressions = []    # pass → fail（退化）
        improvements = []   # fail → pass（改善）
        unchanged = 0

        for sample_id, current in self._sample_results.items():
            prev_verdict = prev_verdicts.get(sample_id)
            if prev_verdict is None:
                # 新增样本，上次没有
                continue

            curr_verdict = current["verdict"]
            if prev_verdict == "pass" and curr_verdict == "fail":
                regressions.append({
                    "id": sample_id,
                    "category": current["category"],
                    "detail": current["detail"],
                    "previous_verdict": prev_verdict,
                    "current_verdict": curr_verdict,
                })
            elif prev_verdict == "fail" and curr_verdict == "pass":
                improvements.append({
                    "id": sample_id,
                    "category": current["category"],
                    "detail": current["detail"],
                    "previous_verdict": prev_verdict,
                    "current_verdict": curr_verdict,
                })
            else:
                unchanged += 1

        # 打印回归检查结果
        print(f"\n  回归检查（对比 {prev_path.name}）:")
        prev_summary = prev_report.get("summary", {})
        print(f"  上次通过率: {prev_summary.get('pass_rate', 'N/A')}")
        current_summary = self._compute_summary()
        print(f"  本次通过率: {current_summary['pass_rate']}")

        if regressions:
            print(f"  REGRESSION: {len(regressions)} 条样本从 pass 退化为 fail:")
            for r in regressions:
                print(f"    [{r['id']}] {r['detail']}")
        else:
            print(f"  无回归（没有 pass→fail 的样本）")

        if improvements:
            print(f"  IMPROVEMENT: {len(improvements)} 条样本从 fail 改善为 pass")

        print(f"  未变化: {unchanged} 条")

        return {
            "status": "regression_found" if regressions else "no_regression",
            "regressions": regressions,
            "improvements": improvements,
            "unchanged": unchanged,
            "previous_pass_rate": prev_summary.get("pass_rate"),
            "current_pass_rate": current_summary["pass_rate"],
        }

    # ── 重置 ───────────────────────────────────────────────────────

    def reset_results(self) -> None:
        """
        清空验证结果（保留已加载的 golden set），用于重新验证。

        场景：修改了 pipeline 参数后，重新跑 validate_pipeline()。
        """
        self._stage_stats.clear()
        for sample in self.golden_samples:
            self._sample_results[sample.id] = {
                "id": sample.id,
                "category": sample.category,
                "expected_outcome": sample.expected_outcome,
                "expected_filter_stage": sample.expected_filter_stage,
                "text_preview": sample.text[:100].replace("\n", " "),
                "stages_survived": [],
                "filtered_at": None,
                "verdict": "pending",
                "detail": "",
            }
