"""
src/gen3/synthetic_rephraser.py
第三代核心创新 3：合成数据改写（Synthetic Rephrasing）

Nemotron-CC 的洞见：
  "低质量数据不是垃圾，而是待改写的原材料。"
  通过 LLM API 将低质量文本改写为高质量文本，实现数据回收。

  效果（Nemotron-CC）：
  - 改写后的合成数据 quality_score 提升 0.15-0.3（平均）
  - 与真实高质量数据混合后，不降低模型性能
  - 使得总可用 unique token 数量增加 4 倍（相比只用 top-10% 过滤）

  飞轮效应（论文原文）：
  "数据改善模型，模型反过来改善数据。"
  改写后的合成数据可用于训练更好的分类器，
  更好的分类器能更准确地找出值得改写的数据。

实现：调用 configs/api_config.yaml 中配置的 LLM API
"""

import time
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm


class SyntheticRephraser:
    """
    使用 LLM API 改写低质量文本。
    支持 Anthropic / OpenAI / DeepSeek 三个 provider。
    """

    def __init__(self, api_config: Dict):
        """
        Args:
            api_config: load_api_config() 的返回值
        """
        self.api_cfg = api_config
        self.provider = api_config["provider"]
        self.model = api_config.get("model", "claude-haiku-4-5-20251001")
        self.max_tokens = api_config.get("max_tokens", 1024)
        self.temperature = api_config.get("temperature", 0.3)

        rephrase_cfg = api_config.get("rephrasing", {})
        self.system_prompt = rephrase_cfg.get(
            "system_prompt",
            "你是一位专业的文本质量改写专家。将低质量网络文本改写为高质量、信息密集的内容。直接输出改写结果。"
        )
        self.user_prompt_template = rephrase_cfg.get(
            "user_prompt_template",
            "请改写以下文本：\n\n{text}"
        )
        self.requests_per_minute = rephrase_cfg.get("requests_per_minute", 50)
        self.max_retries = rephrase_cfg.get("max_retries", 3)
        self.retry_delay = rephrase_cfg.get("retry_delay", 2)

        # 速率限制
        self._request_interval = 60.0 / self.requests_per_minute
        self._last_request_time = 0.0

        print(f"  🔧 SyntheticRephraser 初始化: provider={self.provider}, model={self.model}")

    def _rephrase_anthropic(self, text: str) -> str:
        """调用 Anthropic API 改写文本。"""
        import anthropic
        client = anthropic.Anthropic(api_key=self.api_cfg["api_key"])

        message = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=self.system_prompt,
            messages=[{
                "role": "user",
                "content": self.user_prompt_template.format(text=text[:3000])  # 截断避免超限
            }],
        )
        return message.content[0].text.strip()

    def _rephrase_openai(self, text: str) -> str:
        """调用 OpenAI API 改写文本。"""
        from openai import OpenAI
        client = OpenAI(api_key=self.api_cfg["api_key"])

        response = client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.user_prompt_template.format(text=text[:3000])},
            ],
        )
        return response.choices[0].message.content.strip()

    def _rephrase_deepseek(self, text: str) -> str:
        """调用 DeepSeek API 改写文本（兼容 OpenAI 格式）。"""
        from openai import OpenAI
        client = OpenAI(
            api_key=self.api_cfg["api_key"],
            base_url="https://api.deepseek.com/v1",
        )
        response = client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.user_prompt_template.format(text=text[:3000])},
            ],
        )
        return response.choices[0].message.content.strip()

    def rephrase_single(self, text: str) -> Tuple[Optional[str], str]:
        """
        改写单条文本。

        Returns:
            (rephrased_text, status)
            status: "success" | "api_error" | "too_short"
        """
        if len(text.split()) < 30:
            return None, "too_short"

        # 速率限制
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self._request_interval:
            time.sleep(self._request_interval - elapsed)

        for attempt in range(self.max_retries):
            try:
                self._last_request_time = time.time()

                if self.provider == "anthropic":
                    result = self._rephrase_anthropic(text)
                elif self.provider == "openai":
                    result = self._rephrase_openai(text)
                elif self.provider == "deepseek":
                    result = self._rephrase_deepseek(text)
                else:
                    return None, f"unknown_provider:{self.provider}"

                if result and len(result.split()) >= 20:
                    return result, "success"
                return None, "empty_result"

            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    return None, f"api_error:{str(e)[:50]}"

        return None, "max_retries_exceeded"

    def rephrase_batch(
        self,
        docs: List[Dict],
        max_count: Optional[int] = None,
        eval_classifier=None,   # 改写后用评估分类器验证质量
        min_quality_after: float = 0.4,
    ) -> Tuple[List[Dict], Dict]:
        """
        批量改写文档。

        Args:
            docs: 待改写文档列表
            max_count: 最多改写多少条（None = 全部）
            eval_classifier: 用于验证改写后质量的分类器
            min_quality_after: 改写后最低质量分（低于此值则不保留改写版本）

        Returns:
            (rephrased_docs, stats)
        """
        target_docs = docs[:max_count] if max_count else docs
        print(f"\n  ✍️  LLM 改写: {len(target_docs)} 条文档")
        print(f"     Provider: {self.provider} | Model: {self.model}")

        rephrased_docs = []
        stats = {
            "total_input": len(target_docs),
            "success": 0,
            "failed": 0,
            "quality_filtered": 0,
            "total_api_calls": 0,
            "status_counts": {},
        }

        for doc in tqdm(target_docs, desc="  ✍️  改写"):
            original_text = doc["text"]
            rephrased, status = self.rephrase_single(original_text)
            stats["total_api_calls"] += 1
            stats["status_counts"][status] = stats["status_counts"].get(status, 0) + 1

            if rephrased and status == "success":
                # 可选：验证改写后质量
                if eval_classifier:
                    post_score = eval_classifier.score(rephrased)
                    if post_score < min_quality_after:
                        stats["quality_filtered"] += 1
                        continue

                new_doc = dict(doc)
                new_doc["text"] = rephrased
                new_doc["_original_text"] = original_text[:200]  # 保留原文片段
                new_doc["_is_synthetic"] = True
                new_doc["_rephrase_status"] = status
                rephrased_docs.append(new_doc)
                stats["success"] += 1
            else:
                stats["failed"] += 1

        stats["success_rate"] = round(stats["success"] / len(target_docs), 4) if target_docs else 0
        print(f"\n  ✅ 改写完成: {stats['success']}/{len(target_docs)} 成功")
        print(f"     成功率: {stats['success_rate']:.1%} | 质量过滤: {stats['quality_filtered']}")

        return rephrased_docs, stats

    def compute_before_after_comparison(
        self,
        original_docs: List[Dict],
        rephrased_docs: List[Dict],
        eval_classifier,
    ) -> Dict:
        """
        计算改写前后的质量指标对比（Notebook 04 核心分析）。
        """
        original_texts = [d["text"] for d in original_docs[:len(rephrased_docs)]]
        rephrased_texts = [d["text"] for d in rephrased_docs]

        orig_scores = eval_classifier.score_batch(original_texts)
        reph_scores = eval_classifier.score_batch(rephrased_texts)

        improvement = reph_scores - orig_scores
        return {
            "original_quality_mean": round(float(orig_scores.mean()), 4),
            "rephrased_quality_mean": round(float(reph_scores.mean()), 4),
            "quality_improvement_mean": round(float(improvement.mean()), 4),
            "quality_improvement_p50": round(float(
                __import__("numpy").percentile(improvement, 50)
            ), 4),
            "improved_fraction": round(float((improvement > 0).mean()), 4),
            "n_compared": len(original_texts),
        }
