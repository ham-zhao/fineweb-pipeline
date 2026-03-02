"""
src/proxy_model/evaluator.py
Proxy Model 评估器

加载 run_proxy_training.py 训练的 GPT-2 125M checkpoint，
提供困惑度计算、文本生成和简单 downstream 评估接口。

供 Notebook 09 调用。
"""

import json
import math
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple


def _get_device() -> str:
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class ProxyModelEvaluator:
    """
    对训练好的 Proxy Model checkpoint 进行评估。

    参数：
        model_path: model.pt 文件路径（由 run_proxy_training.py 生成）
        device: 推理设备（默认自动检测 MPS/CUDA/CPU）
    """

    def __init__(self, model_path: str, device: Optional[str] = None):
        import torch
        self.model_path = Path(model_path)
        self.device = device or _get_device()

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"模型文件不存在: {self.model_path}\n"
                "请先运行: caffeinate -i python scripts/run_proxy_training.py"
            )

        print(f"加载模型: {self.model_path} (device={self.device})")
        ckpt = torch.load(self.model_path, map_location="cpu")
        self._config = ckpt["config"]

        # 重建模型架构（与 run_proxy_training.py 中定义一致）
        self.model = self._build_model(
            vocab_size=self._config["vocab_size"],
            seq_len=self._config["seq_len"],
        )
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        # 加载 tokenizer
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"✅ 模型就绪 | 参数: {n_params/1e6:.1f}M | 设备: {self.device}")

    # ── 模型重建（与 run_proxy_training.py 保持一致）─────────────

    def _build_model(self, vocab_size: int, seq_len: int):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        import math as _math

        class CausalSelfAttention(nn.Module):
            def __init__(self, n_embd, n_head, seq_len, dropout=0.0):
                super().__init__()
                self.n_head = n_head
                self.n_embd = n_embd
                self.c_attn = nn.Linear(n_embd, 3 * n_embd)
                self.c_proj = nn.Linear(n_embd, n_embd)
                self.dropout = nn.Dropout(dropout)
                self.register_buffer(
                    "bias", torch.tril(torch.ones(seq_len, seq_len)).view(1, 1, seq_len, seq_len)
                )

            def forward(self, x):
                B, T, C = x.size()
                q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
                nh, hs = self.n_head, C // self.n_head
                q = q.view(B, T, nh, hs).transpose(1, 2)
                k = k.view(B, T, nh, hs).transpose(1, 2)
                v = v.view(B, T, nh, hs).transpose(1, 2)
                att = (q @ k.transpose(-2, -1)) * (1.0 / _math.sqrt(hs))
                att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
                att = F.softmax(att, dim=-1)
                y = self.dropout(att) @ v
                return self.c_proj(y.transpose(1, 2).contiguous().view(B, T, C))

        class MLP(nn.Module):
            def __init__(self, n_embd):
                super().__init__()
                self.c_fc   = nn.Linear(n_embd, 4 * n_embd)
                self.c_proj = nn.Linear(4 * n_embd, n_embd)
                self.act    = nn.GELU()

            def forward(self, x):
                return self.c_proj(self.act(self.c_fc(x)))

        class Block(nn.Module):
            def __init__(self, n_embd, n_head, seq_len):
                super().__init__()
                self.ln_1 = nn.LayerNorm(n_embd)
                self.attn = CausalSelfAttention(n_embd, n_head, seq_len)
                self.ln_2 = nn.LayerNorm(n_embd)
                self.mlp  = MLP(n_embd)

            def forward(self, x):
                x = x + self.attn(self.ln_1(x))
                x = x + self.mlp(self.ln_2(x))
                return x

        class GPT2(nn.Module):
            def __init__(self, vocab_size, seq_len, n_layer=12, n_head=12, n_embd=768):
                super().__init__()
                self.seq_len = seq_len
                self.transformer = nn.ModuleDict({
                    "wte":  nn.Embedding(vocab_size, n_embd),
                    "wpe":  nn.Embedding(seq_len, n_embd),
                    "drop": nn.Dropout(0.0),
                    "h":    nn.ModuleList([Block(n_embd, n_head, seq_len) for _ in range(n_layer)]),
                    "ln_f": nn.LayerNorm(n_embd),
                })
                self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
                self.transformer["wte"].weight = self.lm_head.weight

            def forward(self, idx, targets=None):
                import torch as _torch
                B, T = idx.size()
                pos = _torch.arange(T, device=idx.device)
                x = self.transformer["drop"](
                    self.transformer["wte"](idx) + self.transformer["wpe"](pos)
                )
                for block in self.transformer["h"]:
                    x = block(x)
                x = self.transformer["ln_f"](x)
                logits = self.lm_head(x)
                loss = None
                if targets is not None:
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
                return logits, loss

        return GPT2(
            vocab_size=vocab_size,
            seq_len=seq_len,
            n_layer=self._config.get("n_layer", 12),
            n_head=self._config.get("n_head", 12),
            n_embd=self._config.get("n_embd", 768),
        )

    # ── 核心评估方法 ──────────────────────────────────────────────

    def compute_perplexity(
        self,
        texts: List[str],
        max_length: Optional[int] = None,
        batch_size: int = 4,
    ) -> Tuple[float, List[float]]:
        """
        计算文本列表的平均困惑度。

        Returns:
            (mean_perplexity, per_text_perplexities)
        """
        import torch
        seq_len = self._config["seq_len"]
        max_length = max_length or seq_len

        per_ppls = []
        for text in texts:
            ids = self.tokenizer.encode(text, add_special_tokens=True)
            ids = ids[:max_length]
            if len(ids) < 2:
                continue
            x = torch.tensor(ids[:-1], dtype=torch.long, device=self.device).unsqueeze(0)
            y = torch.tensor(ids[1:],  dtype=torch.long, device=self.device).unsqueeze(0)
            with torch.no_grad():
                _, loss = self.model(x, y)
            ppl = math.exp(min(loss.item(), 20))
            per_ppls.append(ppl)

        if not per_ppls:
            return float("nan"), []
        mean_ppl = sum(per_ppls) / len(per_ppls)
        return mean_ppl, per_ppls

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 40,
    ) -> str:
        """
        从 prompt 开始生成文本（用于定性分析）。

        Returns:
            生成的完整文本（含 prompt）
        """
        import torch

        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        seq_len = self._config["seq_len"]
        x = torch.tensor(ids, dtype=torch.long, device=self.device).unsqueeze(0)

        with torch.no_grad():
            for _ in range(max_new_tokens):
                x_cond = x[:, -seq_len:]
                logits, _ = self.model(x_cond)
                logits = logits[:, -1, :] / temperature
                if top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, -1:]] = float("-inf")
                probs = torch.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
                x = torch.cat([x, next_id], dim=1)
                if next_id.item() == self.tokenizer.eos_token_id:
                    break

        return self.tokenizer.decode(x[0].tolist(), skip_special_tokens=True)

    def completion_accuracy(
        self,
        prompts_and_answers: List[Dict[str, str]],
    ) -> Dict:
        """
        简易 completion 任务准确率评估。

        输入格式：
            [{"prompt": "The capital of France is", "answer": "Paris"}, ...]

        通过比较正确答案 vs 随机答案的 token 概率来判断。

        Returns:
            {"accuracy": float, "n_correct": int, "n_total": int, "details": list}
        """
        import torch

        correct = 0
        details = []

        for item in prompts_and_answers:
            prompt  = item["prompt"]
            answer  = item["answer"]
            wrongs  = item.get("wrong_answers", ["no", "unknown", "none"])

            prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            answer_ids = self.tokenizer.encode(" " + answer, add_special_tokens=False)

            # 计算正确答案的平均 log-prob
            def score_completion(answer_tok_ids: List[int]) -> float:
                full_ids = prompt_ids + answer_tok_ids
                x = torch.tensor(full_ids[:-1], device=self.device).unsqueeze(0)
                y = torch.tensor(full_ids[1:],  device=self.device).unsqueeze(0)
                with torch.no_grad():
                    _, loss = self.model(x, y)
                return -loss.item()  # log-prob（越高越好）

            correct_score = score_completion(answer_ids)
            wrong_scores = [
                score_completion(self.tokenizer.encode(" " + w, add_special_tokens=False))
                for w in wrongs
            ]

            is_correct = correct_score > max(wrong_scores) if wrong_scores else True
            if is_correct:
                correct += 1

            details.append({
                "prompt": prompt[:60],
                "answer": answer,
                "correct_score": round(correct_score, 4),
                "max_wrong_score": round(max(wrong_scores), 4) if wrong_scores else None,
                "is_correct": is_correct,
            })

        n = len(prompts_and_answers)
        return {
            "accuracy": round(correct / n, 4) if n else 0,
            "n_correct": correct,
            "n_total": n,
            "details": details,
        }

    # ── 便捷加载方法 ─────────────────────────────────────────────

    @classmethod
    def load_all(cls, proxy_dir: str = "results/proxy_models") -> Dict[str, "ProxyModelEvaluator"]:
        """
        加载 proxy_dir 下所有子目录中的 model.pt。

        Returns:
            {"raw": evaluator, "gen1": evaluator, "gen3": evaluator}
        """
        evaluators = {}
        base = Path(proxy_dir)
        for sub in ["raw", "gen1", "gen3"]:
            model_path = base / sub / "model.pt"
            if model_path.exists():
                try:
                    evaluators[sub] = cls(str(model_path))
                except Exception as e:
                    print(f"  ⚠️  加载 {sub} 失败: {e}")
        return evaluators

    @staticmethod
    def load_train_stats(proxy_dir: str = "results/proxy_models") -> Dict[str, Dict]:
        """加载各数据集的训练统计 JSON。"""
        stats = {}
        base = Path(proxy_dir)
        for sub in ["raw", "gen1", "gen3"]:
            f = base / sub / "train_stats.json"
            if f.exists():
                with open(f) as fp:
                    stats[sub] = json.load(fp)
        return stats
