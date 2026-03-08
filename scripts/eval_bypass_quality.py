#!/usr/bin/env python3
"""
scripts/eval_bypass_quality.py
用 Claude Opus 评估 bypass 被严格 heuristic 误杀的高质量文档。

核心问题：分类器判定为高质量（ensemble_score >= 0.7）但严格 heuristic 会过滤的文档，
        到底是"真正高质量但被误杀"还是"分类器误判"？

产出文件：
  data/gen3_output/full_run/bypass_quality_eval.json - Claude Opus 评估结果
"""

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")


def evaluate_with_claude(texts_with_meta: list, api_key: str, batch_size: int = 5) -> list:
    """用 Claude API 评估文档质量。"""
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    results = []
    for i in range(0, len(texts_with_meta), batch_size):
        batch = texts_with_meta[i:i+batch_size]
        batch_texts = []
        for j, item in enumerate(batch):
            batch_texts.append(f"--- Document {j+1} ---\n{item['text'][:1500]}\n")

        prompt = f"""Please evaluate the quality of the following {len(batch)} web documents for use as pre-training data for language models.

For each document, provide:
1. quality_score (0.0-1.0): Overall quality for LLM pre-training
2. quality_label: "high" (>0.7), "medium" (0.3-0.7), or "low" (<0.3)
3. brief_reason: 1-sentence reason (in English)

Scoring criteria:
- High (>0.7): Well-written, informative, coherent, suitable for training
- Medium (0.3-0.7): Some useful content but has issues (formatting, partial content, etc.)
- Low (<0.3): Boilerplate, navigation text, spam, non-content, garbled text

{"".join(batch_texts)}

Respond in JSON format only:
[{{"doc_id": 1, "quality_score": 0.X, "quality_label": "...", "brief_reason": "..."}}, ...]"""

        try:
            response = client.messages.create(
                model="claude-sonnet-4-6",  # Use Sonnet for cost efficiency
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            resp_text = response.content[0].text
            # Parse JSON from response
            # Handle potential markdown code blocks
            if "```json" in resp_text:
                resp_text = resp_text.split("```json")[1].split("```")[0]
            elif "```" in resp_text:
                resp_text = resp_text.split("```")[1].split("```")[0]

            batch_results = json.loads(resp_text)
            for j, result in enumerate(batch_results):
                idx = i + j
                result["ensemble_score"] = texts_with_meta[idx]["ensemble_score"]
                result["kill_reason"] = texts_with_meta[idx]["kill_reason"]
                result["url"] = texts_with_meta[idx].get("url", "")
                results.append(result)

            print(f"  Batch {i//batch_size + 1}/{(len(texts_with_meta) + batch_size - 1)//batch_size}: "
                  f"{len(batch_results)} evaluated")
        except Exception as e:
            print(f"  Batch {i//batch_size + 1} error: {e}")
            for j in range(len(batch)):
                idx = i + j
                results.append({
                    "doc_id": j + 1,
                    "quality_score": -1,
                    "quality_label": "error",
                    "brief_reason": str(e),
                    "ensemble_score": texts_with_meta[idx]["ensemble_score"],
                    "kill_reason": texts_with_meta[idx]["kill_reason"],
                })

        # Rate limiting
        time.sleep(1)

    return results


def main():
    # Load samples
    samples_path = ROOT / "data/gen3_output/full_run/bypass_quality_samples.json"
    if not samples_path.exists():
        print(f"Error: {samples_path} not found. Run the extraction step first.")
        sys.exit(1)

    with open(samples_path) as f:
        samples = json.load(f)
    print(f"Loaded {len(samples)} bypass-killed samples for evaluation")

    # Get API key
    api_key = os.environ.get("FINEWEB_API_KEY", "")
    if not api_key:
        print("Error: FINEWEB_API_KEY not set")
        sys.exit(1)

    # Evaluate
    print(f"\nEvaluating with Claude Sonnet 4.6...")
    results = evaluate_with_claude(samples, api_key, batch_size=5)

    # Summarize
    valid_results = [r for r in results if r["quality_score"] >= 0]
    if not valid_results:
        print("No valid results")
        sys.exit(1)

    scores = [r["quality_score"] for r in valid_results]
    labels = [r["quality_label"] for r in valid_results]

    high_count = sum(1 for l in labels if l == "high")
    medium_count = sum(1 for l in labels if l == "medium")
    low_count = sum(1 for l in labels if l == "low")

    print(f"\n{'='*60}")
    print(f"  Claude Opus 质量评估结果 ({len(valid_results)} docs)")
    print(f"{'='*60}")
    print(f"  High quality:   {high_count:3d} ({high_count/len(valid_results)*100:.1f}%)")
    print(f"  Medium quality: {medium_count:3d} ({medium_count/len(valid_results)*100:.1f}%)")
    print(f"  Low quality:    {low_count:3d} ({low_count/len(valid_results)*100:.1f}%)")
    print(f"  Mean score:     {sum(scores)/len(scores):.3f}")
    print(f"{'='*60}")
    print(f"\n  真正被误杀率（Claude 判定 high 但 heuristic 会杀）: "
          f"{high_count}/{len(valid_results)} = {high_count/len(valid_results)*100:.1f}%")

    # Save results
    output = {
        "summary": {
            "total_evaluated": len(valid_results),
            "high_quality": high_count,
            "medium_quality": medium_count,
            "low_quality": low_count,
            "mean_score": round(sum(scores)/len(scores), 3),
            "true_false_kill_rate": round(high_count/len(valid_results), 3),
        },
        "details": results,
    }

    out_path = ROOT / "data/gen3_output/full_run/bypass_quality_eval.json"
    with open(out_path, "w") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
