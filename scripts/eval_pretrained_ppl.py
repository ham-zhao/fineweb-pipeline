"""
用预训练 GPT-2 Small 计算各数据集的 Perplexity 分布。
不需要训练，直接用 HuggingFace 预训练权重做推理。

逻辑：高质量文本更接近自然语言分布 → PPL 更低
      垃圾文本/乱码/非英文 → PPL 更高

产出文件：
  results/proxy_models/pretrained_ppl_stats.json  - 各数据集 PPL 统计
"""

import json
import sys
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation.perplexity_scorer import PerplexityScorer


def load_texts(path: Path, max_docs: int = 500) -> list:
    """加载 JSONL 文件的 text 字段，随机采样 max_docs 条。"""
    docs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
                text = d.get("text", "")
                if text and len(text.strip()) > 50:
                    docs.append(text)
            except Exception:
                pass
    if len(docs) > max_docs:
        random.seed(42)
        docs = random.sample(docs, max_docs)
    return docs


def main():
    # 数据集配置
    datasets = {
        "raw": ROOT / "data/raw/cc_wet_full.jsonl",
        "gen1": ROOT / "data/gen1_output/full_run/gen1_output.jsonl",
        "gen2": ROOT / "data/gen2_output/full_run/gen2_output.jsonl",
        "gen3": ROOT / "data/gen3_output/full_run/gen3_output.jsonl",
    }

    # 采样数量：每个数据集最多 300 条（平衡速度和统计意义）
    SAMPLE_SIZE = 300

    # 检查文件存在
    for name, path in datasets.items():
        if not path.exists():
            print(f"  ⚠️  {name}: {path} 不存在，跳过")

    available = {k: v for k, v in datasets.items() if v.exists()}
    if not available:
        print("无可用数据集")
        sys.exit(1)

    # 加载数据
    all_texts = {}
    for name, path in available.items():
        texts = load_texts(path, max_docs=SAMPLE_SIZE)
        all_texts[name] = texts
        print(f"  {name}: 加载 {len(texts)} 条文档")

    # 初始化 scorer
    scorer = PerplexityScorer(model_name="gpt2", device="auto", max_tokens=512)

    # 计算 PPL
    results = {}
    for name, texts in all_texts.items():
        print(f"\n{'='*60}")
        print(f"  计算 {name} PPL（{len(texts)} 条）...")
        print(f"{'='*60}")

        scores = scorer.score_batch(texts, batch_size=1, show_progress=True)
        stats = scorer.compute_statistics(scores)

        results[name] = {
            "n_docs": len(texts),
            **stats,
        }

        print(f"  {name}: mean={stats['mean']:.1f}, median={stats['median']:.1f}, "
              f"p25={stats['p25']:.1f}, p75={stats['p75']:.1f}")

    # 计算相对改善
    if "raw" in results:
        raw_median = results["raw"]["median"]
        for name, stats in results.items():
            if name != "raw":
                improvement = (raw_median - stats["median"]) / raw_median * 100
                results[name]["ppl_reduction_vs_raw_pct"] = round(improvement, 1)
                print(f"  {name} vs raw: PPL 降低 {improvement:.1f}%")

    # 保存结果
    out_dir = ROOT / "results/proxy_models"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "pretrained_ppl_stats.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  结果已保存: {out_file}")

    # 打印汇总表
    print(f"\n{'='*70}")
    print(f"  {'数据集':<8} {'文档数':>6} {'Mean PPL':>10} {'Median PPL':>12} {'P25':>8} {'P75':>8}")
    print(f"  {'-'*62}")
    for name in ["raw", "gen1", "gen2", "gen3"]:
        if name in results:
            r = results[name]
            print(f"  {name:<8} {r['n_docs']:>6} {r['mean']:>10.1f} {r['median']:>12.1f} "
                  f"{r['p25']:>8.1f} {r['p75']:>8.1f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
