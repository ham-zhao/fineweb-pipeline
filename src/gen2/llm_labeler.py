"""
src/gen2/llm_labeler.py
Gen2 LLM 标注模块：用 LLM 对文档打质量分（0-5）

产出文件：
  data/gen2_output/llm_labels.jsonl - 标注结果
"""

import json
import time
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed


QUALITY_RUBRIC = """Rate the educational and informational quality of this text on a scale of 0-5:

0 = Garbage: spam, ads, navigation menus, gibberish, or non-natural language
1 = Very Low: readable but no useful information (e.g., social media chatter, clickbait)
2 = Low: some information but poorly written, repetitive, or mostly opinion without evidence
3 = Medium: reasonably informative, covers a topic adequately, but not exceptional
4 = High: well-written, informative, provides clear explanations with examples or evidence
5 = Excellent: expert-level content, comprehensive, educational, suitable for textbook or encyclopedia

Respond with ONLY a single integer (0-5), nothing else."""


class LLMLabeler:
    """用 LLM 对文档打质量分，用于训练 Gen2 fastText 回归模型。"""

    def __init__(self, api_config: Dict, concurrency: int = 5):
        self.api_cfg = api_config
        self.provider = api_config["provider"]
        self.model = api_config.get("model", "claude-haiku-4-5-20251001")
        self.max_tokens = 10
        self.temperature = 0.0
        self.concurrency = concurrency
        self.max_retries = api_config.get("rephrasing", {}).get("max_retries", 3)

    def _label_single_anthropic(self, text: str) -> Tuple[Optional[int], str]:
        """用 Anthropic API 标注单条文档。"""
        import anthropic
        client = anthropic.Anthropic(api_key=self.api_cfg["api_key"])
        try:
            message = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=QUALITY_RUBRIC,
                messages=[{"role": "user", "content": text[:2000]}],
            )
            score_text = message.content[0].text.strip()
            score = int(score_text[0]) if score_text and score_text[0].isdigit() else None
            if score is not None and 0 <= score <= 5:
                return score, "success"
            return None, f"invalid_score:{score_text[:20]}"
        except Exception as e:
            return None, f"api_error:{str(e)[:50]}"

    def _label_single(self, text: str) -> Tuple[Optional[int], str]:
        """标注单条文档，支持重试。"""
        for attempt in range(self.max_retries):
            if self.provider == "anthropic":
                score, status = self._label_single_anthropic(text)
            else:
                return None, f"unsupported_provider:{self.provider}"

            if score is not None:
                return score, status
            if attempt < self.max_retries - 1:
                time.sleep(1 * (attempt + 1))

        return None, "max_retries_exceeded"

    def label_batch(
        self,
        docs: List[Dict],
        sample_count: int = 100,
        random_seed: int = 42,
    ) -> Tuple[List[Dict], Dict]:
        """
        批量标注文档质量分。

        Args:
            docs: Gen1 输出文档列表
            sample_count: 标注样本数（从 docs 中随机抽样）
            random_seed: 随机种子

        Returns:
            (标注结果列表, 统计字典)
        """
        rng = random.Random(random_seed)
        sample = rng.sample(docs, min(sample_count, len(docs)))

        print(f"  LLM 标注: {len(sample)} 条文档 (并发={self.concurrency})")
        print(f"  Provider: {self.provider} | Model: {self.model}")

        results = []
        stats = {"total": len(sample), "success": 0, "failed": 0, "score_distribution": {}}

        def _process(doc):
            score, status = self._label_single(doc["text"])
            return doc, score, status

        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            futures = {executor.submit(_process, doc): i for i, doc in enumerate(sample)}
            for future in as_completed(futures):
                doc, score, status = future.result()
                if score is not None:
                    results.append({
                        "text": doc["text"],
                        "url": doc.get("url", ""),
                        "llm_score": score,
                    })
                    stats["success"] += 1
                    stats["score_distribution"][str(score)] = stats["score_distribution"].get(str(score), 0) + 1
                else:
                    stats["failed"] += 1

        stats["success_rate"] = stats["success"] / stats["total"] if stats["total"] > 0 else 0
        print(f"  标注完成: {stats['success']}/{stats['total']} 成功 ({stats['success_rate']:.0%})")
        print(f"  分数分布: {stats['score_distribution']}")

        return results, stats

    def save_labels(self, labels: List[Dict], output_path: Path) -> None:
        """保存标注结果。"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for label in labels:
                f.write(json.dumps(label, ensure_ascii=False) + "\n")
        print(f"  标注数据保存: {output_path} ({len(labels)} 条)")
