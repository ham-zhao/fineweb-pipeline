"""
src/evaluation/perplexity_scorer.py
Perplexity 打分器（基于 GPT-2 Small）

⚠️  Perplexity 的双面性——使用时务必理解：
───────────────────────────────────────────────
Perplexity 衡量的是"文本的可预测性"，不是"文本的质量"。

  PPL 过高（>10000） → 乱码、非自然文本、随机字符（坏）
  PPL 过低（<10）    → 极度可预测/重复/模板化内容（也坏）
    典型反例：SEO 堆砌文本（"买XX找XX，XX专业XX"）的 PPL 可能比 Wikipedia 还低，
    因为它极度可预测——语言模型"非常确定"下一个词是什么。

结论：Perplexity 必须配合其他指标使用，不能单独作为质量信号。
DCLM 论文也证明：单独用 Perplexity 过滤效果不如 fastText 分类器。
───────────────────────────────────────────────
"""

import numpy as np
import torch
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from tqdm import tqdm


def _get_device() -> str:
    """自动选择最优设备（MPS > CUDA > CPU）。"""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


class PerplexityScorer:
    """
    基于 GPT-2 Small 的文档级 Perplexity 打分器。
    支持 MPS（Mac M 系列 GPU）加速。
    """

    def __init__(self, model_name: str = "gpt2", device: str = "auto", max_tokens: int = 512):
        """
        Args:
            model_name: HuggingFace 模型名（"gpt2" 或 "gpt2-medium" 等）
            device: "auto" | "mps" | "cuda" | "cpu"
            max_tokens: 文档截断长度（防 OOM）
        """
        self.model_name = model_name
        self.device = _get_device() if device == "auto" else device
        self.max_tokens = max_tokens
        self.model = None
        self.tokenizer = None

        print(f"  🖥️  Perplexity 打分器: device={self.device}, model={model_name}, max_tokens={max_tokens}")

    def _load_model(self) -> None:
        """懒加载模型（第一次调用 score 时才加载）。"""
        if self.model is not None:
            return

        from transformers import GPT2LMHeadModel, GPT2TokenizerFast
        print(f"  📦 加载 GPT-2 模型（{self.model_name}）...")
        self.tokenizer = GPT2TokenizerFast.from_pretrained(self.model_name)
        self.model = GPT2LMHeadModel.from_pretrained(self.model_name)
        self.model.eval()
        self.model.to(self.device)
        print(f"  ✅ GPT-2 已加载到 {self.device}")

    def score(self, text: str) -> float:
        """
        计算单条文本的 Perplexity。

        Returns:
            float，Perplexity 值（越低越可预测，越高越不可预测）
            若文本过短（<10 个 token），返回 float('inf')
        """
        self._load_model()

        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_tokens,
        )
        input_ids = encoding["input_ids"].to(self.device)

        if input_ids.shape[1] < 10:
            return float("inf")

        with torch.no_grad():
            outputs = self.model(input_ids, labels=input_ids)
            loss = outputs.loss  # 交叉熵 loss
            perplexity = torch.exp(loss).item()

        return float(perplexity)

    def score_batch(
        self,
        texts: List[str],
        batch_size: int = 8,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        批量计算 Perplexity（推荐使用，比逐条快约 batch_size 倍）。

        Returns:
            np.ndarray，shape=(len(texts),)
        """
        self._load_model()

        scores = []
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="  📊 Perplexity 打分", unit="batch")

        for i in iterator:
            batch = texts[i:i + batch_size]
            batch_scores = [self.score(t) for t in batch]
            scores.extend(batch_scores)

        return np.array(scores)

    def compute_statistics(self, scores: np.ndarray) -> Dict:
        """
        计算 Perplexity 统计量。

        Returns:
            dict，包含 mean, median, p25, p75, p90 等分位数
        """
        valid = scores[np.isfinite(scores)]
        if len(valid) == 0:
            return {}

        return {
            "mean": float(np.mean(valid)),
            "median": float(np.median(valid)),
            "p25": float(np.percentile(valid, 25)),
            "p75": float(np.percentile(valid, 75)),
            "p90": float(np.percentile(valid, 90)),
            "p95": float(np.percentile(valid, 95)),
            "n_valid": int(len(valid)),
            "n_inf": int(np.sum(~np.isfinite(scores))),
        }

    def identify_anomalies(
        self,
        texts: List[str],
        scores: np.ndarray,
        low_ppl_threshold: float = 10.0,
        high_ppl_threshold: float = 5000.0,
        n_samples: int = 5,
    ) -> Dict:
        """
        识别 Perplexity 异常文档（过低 = 模板化，过高 = 乱码）。

        Returns:
            dict，包含低 PPL 和高 PPL 的样本文档
        """
        low_ppl_mask = scores < low_ppl_threshold
        high_ppl_mask = scores > high_ppl_threshold

        def sample_texts(mask, n):
            idxs = np.where(mask)[0]
            if len(idxs) == 0:
                return []
            selected = idxs[:n]
            return [{"text": texts[i][:200], "ppl": float(scores[i])} for i in selected]

        return {
            "low_ppl_count": int(np.sum(low_ppl_mask)),
            "high_ppl_count": int(np.sum(high_ppl_mask)),
            "low_ppl_samples": sample_texts(low_ppl_mask, n_samples),   # 过低 = 模板化
            "high_ppl_samples": sample_texts(high_ppl_mask, n_samples), # 过高 = 乱码
        }
