"""
src/evaluation/kenlm_scorer.py
KenLM Perplexity 打分器（基于 Wikipedia 预训练模型）

工业标准方法（CCNet/FineWeb/Dolma/RedPajama 均采用）：
──────────────────────────────────────────────────────────
1. 用 Wikipedia 文本训练 KenLM n-gram 语言模型
2. 对每篇文档计算 perplexity：越低 = 越像 Wikipedia = 质量越高
3. 按 perplexity 分桶：head（高质量）/ middle / tail（低质量）

与 GPT-2 Perplexity 的区别：
  - KenLM：CPU 运行，极快（万条/秒），n-gram 模型，仅衡量"词序列概率"
  - GPT-2：需要 GPU/MPS，较慢，Transformer 模型，语义理解更深
  CCNet 原文用 KenLM，工业界沿用至今。

模型来源：edugp/kenlm (HuggingFace)
  - wikipedia/en.arpa.bin：英文 Wikipedia 5-gram 模型
  - wikipedia/en.sp.model：SentencePiece tokenizer
──────────────────────────────────────────────────────────
"""

import re
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm


# CCNet 预处理：数字归零 + 标点归一化
def _normalize_text(text: str) -> str:
    """CCNet 标准预处理：替换数字为 0，规范化空白。"""
    text = re.sub(r'\d', '0', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


class KenLMScorer:
    """
    基于 KenLM 的 Wikipedia perplexity 打分器。

    用法：
        scorer = KenLMScorer(model_dir="data/models")
        scores = scorer.score_batch(texts)
        stats = scorer.compute_statistics(scores)
        buckets = scorer.bucket_analysis(scores)
    """

    # CCNet 分桶阈值（基于 Wikipedia 英文数据的经验值）
    DEFAULT_BUCKET_THRESHOLDS = {
        "head": 300,     # PPL < 300：高质量，类 Wikipedia
        "middle": 1000,  # 300 <= PPL < 1000：中等质量
        # PPL >= 1000：tail，低质量
    }

    def __init__(
        self,
        model_dir: str = "data/models",
        model_file: str = "wikipedia/en.arpa.bin",
        sp_model_file: str = "wikipedia/en.sp.model",
    ):
        self.model_dir = Path(model_dir)
        self.model_path = self.model_dir / model_file
        self.sp_model_path = self.model_dir / sp_model_file
        self._model = None
        self._sp = None

    def _load(self):
        """懒加载 KenLM 模型和 SentencePiece tokenizer。"""
        if self._model is not None:
            return

        import kenlm

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"KenLM 模型未找到: {self.model_path}\n"
                f"请运行: python3 -c \"\n"
                f"from huggingface_hub import hf_hub_download\n"
                f"hf_hub_download('edugp/kenlm', 'wikipedia/en.arpa.bin', local_dir='data/models')\n"
                f"hf_hub_download('edugp/kenlm', 'wikipedia/en.sp.model', local_dir='data/models')\n"
                f"\""
            )

        print(f"  加载 KenLM 模型: {self.model_path.name}")
        self._model = kenlm.Model(str(self.model_path))

        if self.sp_model_path.exists():
            import sentencepiece
            self._sp = sentencepiece.SentencePieceProcessor()
            self._sp.Load(str(self.sp_model_path))
            print(f"  加载 SentencePiece: {self.sp_model_path.name}")
        else:
            print(f"  SentencePiece 模型不存在，使用空格分词")
            self._sp = None

    def _tokenize(self, text: str) -> str:
        """用 SentencePiece 或空格分词。"""
        text = _normalize_text(text)
        if self._sp is not None:
            tokens = self._sp.EncodeAsPieces(text)
            return " ".join(tokens)
        return text

    def score(self, text: str) -> float:
        """
        计算单条文档的 KenLM perplexity。

        Returns:
            float，perplexity 值。越低 = 越像 Wikipedia。
            文本过短（< 5 词）返回 inf。
        """
        self._load()
        tokenized = self._tokenize(text)
        words = tokenized.split()
        if len(words) < 5:
            return float("inf")

        log_score = self._model.score(tokenized)
        # KenLM 返回 log10 概率，转换为 perplexity
        n_words = len(words)
        ppl = 10 ** (-log_score / n_words)
        return float(ppl)

    def score_batch(
        self,
        texts: List[str],
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        批量计算 perplexity（KenLM 在 CPU 上极快，无需 batch GPU）。

        Returns:
            np.ndarray, shape=(len(texts),)
        """
        self._load()
        scores = []
        iterator = enumerate(texts)
        if show_progress:
            iterator = tqdm(iterator, total=len(texts), desc="  KenLM PPL 打分", unit="doc")

        for _, text in iterator:
            scores.append(self.score(text))

        return np.array(scores)

    def compute_statistics(self, scores: np.ndarray) -> Dict:
        """计算 perplexity 统计量。"""
        valid = scores[np.isfinite(scores)]
        if len(valid) == 0:
            return {"n_valid": 0, "n_inf": int(len(scores))}

        return {
            "mean": float(np.mean(valid)),
            "median": float(np.median(valid)),
            "p10": float(np.percentile(valid, 10)),
            "p25": float(np.percentile(valid, 25)),
            "p75": float(np.percentile(valid, 75)),
            "p90": float(np.percentile(valid, 90)),
            "n_valid": int(len(valid)),
            "n_inf": int(np.sum(~np.isfinite(scores))),
        }

    def bucket_analysis(
        self,
        scores: np.ndarray,
        thresholds: Optional[Dict[str, float]] = None,
    ) -> Dict:
        """
        CCNet 风格分桶：head / middle / tail。

        Args:
            scores: perplexity 分数数组
            thresholds: {"head": 300, "middle": 1000} 或自定义

        Returns:
            dict，包含各桶的数量、占比
        """
        if thresholds is None:
            thresholds = self.DEFAULT_BUCKET_THRESHOLDS

        valid = scores[np.isfinite(scores)]
        total = len(valid)
        if total == 0:
            return {}

        head_th = thresholds["head"]
        mid_th = thresholds["middle"]

        head_count = int(np.sum(valid < head_th))
        mid_count = int(np.sum((valid >= head_th) & (valid < mid_th)))
        tail_count = int(np.sum(valid >= mid_th))

        return {
            "head": {"count": head_count, "ratio": head_count / total,
                     "description": f"PPL < {head_th} (高质量，类 Wikipedia)"},
            "middle": {"count": mid_count, "ratio": mid_count / total,
                       "description": f"{head_th} <= PPL < {mid_th} (中等质量)"},
            "tail": {"count": tail_count, "ratio": tail_count / total,
                     "description": f"PPL >= {mid_th} (低质量)"},
            "total": total,
            "thresholds": thresholds,
        }
