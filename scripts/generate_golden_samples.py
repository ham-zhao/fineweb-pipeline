"""
Auto-generate golden samples for fineweb-pipeline regression testing.

产出文件：
  data/reference/golden_samples.jsonl  - 10 golden samples across 3 categories

Usage:
  source .venv/bin/activate
  python3 scripts/generate_golden_samples.py
"""

import json
import re
import sys
from pathlib import Path
from collections import Counter

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_CC_WET = PROJECT_ROOT / "data" / "raw" / "cc_wet_sample.jsonl"
GEN1_OUTPUT = PROJECT_ROOT / "data" / "gen1_output" / "smoke_test" / "gen1_output.jsonl"
GEN2_OUTPUT = PROJECT_ROOT / "data" / "gen2_output" / "smoke_test" / "gen2_output.jsonl"
WIKIPEDIA = PROJECT_ROOT / "data" / "reference" / "wikipedia_abstracts.jsonl"
EVAL_CLASSIFIER = PROJECT_ROOT / "results" / "quality_scores" / "eval_classifier.bin"
OUTPUT_FILE = PROJECT_ROOT / "data" / "reference" / "golden_samples.jsonl"

MAX_TEXT_LEN = 2000  # Truncate stored text to this length


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_jsonl(path: Path) -> list[dict]:
    """Load all records from a JSONL file."""
    docs = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


def truncate_text(text: str, max_len: int = MAX_TEXT_LEN) -> str:
    """Truncate text to max_len characters, adding ellipsis if truncated."""
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def latin_ratio(text: str) -> float:
    """Fraction of characters that are basic Latin letters (a-z, A-Z)."""
    if not text:
        return 0.0
    latin_chars = sum(1 for c in text if c.isascii() and c.isalpha())
    return latin_chars / len(text)


def alpha_ratio(text: str) -> float:
    """Fraction of characters that are alphabetic (any script)."""
    if not text:
        return 0.0
    alpha_chars = sum(1 for c in text if c.isalpha())
    return alpha_chars / len(text)


def word_count(text: str) -> int:
    """Count whitespace-delimited words."""
    return len(text.split())


def repetition_score(text: str) -> float:
    """Estimate repetition: fraction of 5-grams that are duplicates."""
    words = text.split()
    if len(words) < 5:
        return 0.0
    ngrams = [" ".join(words[i:i + 5]) for i in range(len(words) - 4)]
    counts = Counter(ngrams)
    if not ngrams:
        return 0.0
    duplicated = sum(c - 1 for c in counts.values() if c > 1)
    return duplicated / len(ngrams)


def special_char_ratio(text: str) -> float:
    """Fraction of characters that are non-alphanumeric and non-whitespace."""
    if not text:
        return 0.0
    special = sum(1 for c in text if not c.isalnum() and not c.isspace())
    return special / len(text)


def eval_score_fasttext(model, text: str) -> float:
    """Score a document using the eval fasttext classifier. Returns P(__label__high)."""
    # fasttext expects single-line input
    clean = text.replace("\n", " ").strip()
    if not clean:
        return 0.0
    labels, probs = model.predict(clean, k=2)
    # Find the probability for __label__high
    for label, prob in zip(labels, probs):
        if label == "__label__high":
            return float(prob)
    return 0.0


def make_sample(
    id_str: str,
    text: str,
    category: str,
    expected_outcome: str,
    expected_filter_stage: str | None,
    reason: str,
    source: str,
    eval_score: float | None = None,
    gen2_score: float | None = None,
) -> dict:
    """Construct a golden sample record."""
    record = {
        "id": id_str,
        "text": truncate_text(text),
        "text_length": len(text),
        "category": category,
        "expected_outcome": expected_outcome,
        "expected_filter_stage": expected_filter_stage,
        "reason": reason,
        "source": source,
    }
    if eval_score is not None:
        record["eval_score"] = round(eval_score, 4)
    if gen2_score is not None:
        record["gen2_score"] = round(gen2_score, 4)
    return record


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("Golden Sample Generator for fineweb-pipeline")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("\n[1/5] Loading data sources...")

    raw_docs = load_jsonl(RAW_CC_WET)
    print(f"  Raw CC WET:   {len(raw_docs):,} docs")

    gen1_docs = load_jsonl(GEN1_OUTPUT)
    print(f"  Gen1 output:  {len(gen1_docs):,} docs")

    gen2_docs = load_jsonl(GEN2_OUTPUT)
    print(f"  Gen2 output:  {len(gen2_docs):,} docs")

    wiki_docs = load_jsonl(WIKIPEDIA)
    print(f"  Wikipedia:    {len(wiki_docs):,} docs")

    # ------------------------------------------------------------------
    # 2. Load eval classifier (fasttext)
    # ------------------------------------------------------------------
    print("\n[2/5] Loading eval classifier...")
    ft_model = None
    try:
        import fasttext
        fasttext.FastText.eprint = lambda x: None  # suppress warnings
        ft_model = fasttext.load_model(str(EVAL_CLASSIFIER))
        print(f"  Loaded fasttext model: {EVAL_CLASSIFIER.name}")
        print(f"  Labels: {ft_model.labels}")
    except Exception as e:
        print(f"  WARNING: Could not load fasttext model: {e}")
        print("  Will use heuristic scoring instead.")

    def score_doc(text: str) -> float:
        """Score a document: use fasttext if available, else heuristics."""
        if ft_model is not None:
            return eval_score_fasttext(ft_model, text)
        # Heuristic fallback: combine alpha_ratio and word_count
        ar = alpha_ratio(text)
        wc = min(word_count(text) / 500.0, 1.0)  # normalize to [0,1]
        return 0.5 * ar + 0.5 * wc

    # ------------------------------------------------------------------
    # 3. Build lookup sets for filtering
    # ------------------------------------------------------------------
    print("\n[3/5] Building lookups and scoring...")

    # Build set of Gen1 text prefixes (first 200 chars) for fast matching
    gen1_prefixes = set()
    for doc in gen1_docs:
        gen1_prefixes.add(doc["text"][:200])

    gen2_prefixes = set()
    for doc in gen2_docs:
        gen2_prefixes.add(doc["text"][:200])

    # Identify raw docs NOT in Gen1 (filtered out by Gen1)
    raw_filtered_by_gen1 = []
    for doc in raw_docs:
        prefix = doc["text"][:200]
        if prefix not in gen1_prefixes:
            raw_filtered_by_gen1.append(doc)
    print(f"  Raw docs filtered by Gen1: {len(raw_filtered_by_gen1):,}")

    # Identify Gen1 docs NOT in Gen2 (filtered out by Gen2)
    gen1_filtered_by_gen2 = []
    for doc in gen1_docs:
        prefix = doc["text"][:200]
        if prefix not in gen2_prefixes:
            gen1_filtered_by_gen2.append(doc)
    print(f"  Gen1 docs filtered by Gen2: {len(gen1_filtered_by_gen2):,}")

    # Score all Gen2 docs with eval classifier
    gen2_scored = []
    for doc in gen2_docs:
        es = score_doc(doc["text"])
        gen2_scored.append({**doc, "_eval_score": es})
    gen2_scored.sort(key=lambda x: x["_eval_score"], reverse=True)

    # Score Gen1 docs with eval classifier for boundary selection
    gen1_scored = []
    for doc in gen1_docs:
        es = score_doc(doc["text"])
        gen1_scored.append({**doc, "_eval_score": es})
    gen1_scored.sort(key=lambda x: x["_eval_score"])

    # Score Gen1-filtered-by-Gen2 docs
    gen1_filt_scored = []
    for doc in gen1_filtered_by_gen2:
        es = score_doc(doc["text"])
        gen1_filt_scored.append({**doc, "_eval_score": es})
    gen1_filt_scored.sort(key=lambda x: x["_eval_score"])

    # ------------------------------------------------------------------
    # 4. Select golden samples
    # ------------------------------------------------------------------
    print("\n[4/5] Selecting golden samples...")
    golden_samples = []

    # --- HIGH QUALITY (3 samples): top Gen2 output by eval score ---
    print("\n  --- HIGH QUALITY (from Gen2 top-10% by eval score) ---")
    for i, doc in enumerate(gen2_scored[:3]):
        sample = make_sample(
            id_str=f"golden_{i + 1:03d}",
            text=doc["text"],
            category="high_quality",
            expected_outcome="pass_all",
            expected_filter_stage=None,
            reason=(
                f"Top-{i + 1} eval classifier score ({doc['_eval_score']:.4f}) among Gen2 output. "
                f"Survived all Gen1 heuristic filters and Gen2 classifier (gen2_score={doc['_gen2_score']:.4f}). "
                f"Word count: {word_count(doc['text'])}, alpha ratio: {alpha_ratio(doc['text']):.2f}."
            ),
            source="gen2_output/smoke_test",
            eval_score=doc["_eval_score"],
            gen2_score=doc["_gen2_score"],
        )
        golden_samples.append(sample)
        print(f"    {sample['id']}: eval={doc['_eval_score']:.4f}, gen2={doc['_gen2_score']:.4f}, "
              f"words={word_count(doc['text'])}")

    # --- LOW QUALITY (3 samples): diverse failure modes from raw filtered ---
    print("\n  --- LOW QUALITY (from raw CC WET, filtered by Gen1) ---")

    # 4a. Non-English sample: lowest Latin ratio
    non_english_candidates = sorted(raw_filtered_by_gen1, key=lambda d: latin_ratio(d["text"]))
    non_eng_doc = None
    for cand in non_english_candidates:
        lr = latin_ratio(cand["text"])
        if lr < 0.3 and word_count(cand["text"]) > 10:
            non_eng_doc = cand
            break
    if non_eng_doc is None and non_english_candidates:
        non_eng_doc = non_english_candidates[0]

    if non_eng_doc:
        lr = latin_ratio(non_eng_doc["text"])
        sample = make_sample(
            id_str="golden_004",
            text=non_eng_doc["text"],
            category="low_quality",
            expected_outcome="filter_gen1",
            expected_filter_stage="language_filter",
            reason=(
                f"Non-English content with very low Latin character ratio ({lr:.2f}). "
                f"Expected to be caught by language detection filter in Gen1. "
                f"Word count: {word_count(non_eng_doc['text'])}."
            ),
            source="raw/cc_wet_sample",
            eval_score=score_doc(non_eng_doc["text"]),
        )
        golden_samples.append(sample)
        print(f"    {sample['id']}: latin_ratio={lr:.2f}, words={word_count(non_eng_doc['text'])}, "
              f"reason=non-English")
    else:
        print("    WARNING: Could not find non-English sample")

    # 4b. Very short sample (<50 words)
    short_candidates = [d for d in raw_filtered_by_gen1 if word_count(d["text"]) < 50 and word_count(d["text"]) > 3]
    short_candidates.sort(key=lambda d: word_count(d["text"]))
    short_doc = None
    # Prefer one that's in English (to isolate the "too short" failure mode)
    for cand in short_candidates:
        if latin_ratio(cand["text"]) > 0.5:
            short_doc = cand
            break
    if short_doc is None and short_candidates:
        short_doc = short_candidates[0]

    if short_doc:
        wc = word_count(short_doc["text"])
        sample = make_sample(
            id_str="golden_005",
            text=short_doc["text"],
            category="low_quality",
            expected_outcome="filter_gen1",
            expected_filter_stage="gopher_length_filter",
            reason=(
                f"Very short document with only {wc} words (Gopher min is typically ~50-100 words). "
                f"Expected to be caught by length-based heuristic filters in Gen1. "
                f"Alpha ratio: {alpha_ratio(short_doc['text']):.2f}."
            ),
            source="raw/cc_wet_sample",
            eval_score=score_doc(short_doc["text"]),
        )
        golden_samples.append(sample)
        print(f"    {sample['id']}: word_count={wc}, reason=too_short")
    else:
        print("    WARNING: Could not find short sample")

    # 4c. Spam/junk content: high repetition or high special char ratio or very low alpha
    spam_candidates = []
    for doc in raw_filtered_by_gen1:
        text = doc["text"]
        wc = word_count(text)
        if wc < 10:
            continue
        rep = repetition_score(text)
        spc = special_char_ratio(text)
        ar = alpha_ratio(text)
        # Score "junkiness": high repetition, high special chars, low alpha
        junk_score = rep * 2.0 + spc * 2.0 + (1.0 - ar) * 1.0
        spam_candidates.append((doc, junk_score, rep, spc, ar))

    spam_candidates.sort(key=lambda x: x[1], reverse=True)

    # Avoid picking the same doc as non-English or short
    used_prefixes = set()
    for s in golden_samples:
        used_prefixes.add(s["text"][:200])

    spam_doc = None
    spam_info = None
    for cand, js, rep, spc, ar in spam_candidates:
        if cand["text"][:200] not in used_prefixes:
            spam_doc = cand
            spam_info = (js, rep, spc, ar)
            break

    if spam_doc:
        js, rep, spc, ar = spam_info
        sample = make_sample(
            id_str="golden_006",
            text=spam_doc["text"],
            category="low_quality",
            expected_outcome="filter_gen1",
            expected_filter_stage="quality_heuristic_filter",
            reason=(
                f"Spam/junk content with high junk score ({js:.2f}). "
                f"Repetition: {rep:.2f}, special char ratio: {spc:.2f}, alpha ratio: {ar:.2f}. "
                f"Expected to be caught by quality heuristic filters (C4/Gopher) in Gen1."
            ),
            source="raw/cc_wet_sample",
            eval_score=score_doc(spam_doc["text"]),
        )
        golden_samples.append(sample)
        print(f"    {sample['id']}: junk_score={js:.2f}, rep={rep:.2f}, special={spc:.2f}, "
              f"alpha={ar:.2f}, reason=spam/junk")
    else:
        print("    WARNING: Could not find spam/junk sample")

    # --- BOUNDARY (4 samples) ---
    print("\n  --- BOUNDARY (edge cases) ---")

    # Update used prefixes
    used_prefixes = set()
    for s in golden_samples:
        used_prefixes.add(s["text"][:200])

    # 5a. Gen1 doc with lowest eval score (barely passed heuristics)
    boundary_1 = None
    for doc in gen1_scored:
        if doc["text"][:200] not in used_prefixes:
            boundary_1 = doc
            break

    if boundary_1:
        sample = make_sample(
            id_str="golden_007",
            text=boundary_1["text"],
            category="boundary",
            expected_outcome="borderline",
            expected_filter_stage="gen2_classifier",
            reason=(
                f"Lowest eval score ({boundary_1['_eval_score']:.4f}) among Gen1 output. "
                f"Passed all heuristic filters but has lowest quality as measured by independent evaluator. "
                f"Word count: {word_count(boundary_1['text'])}, alpha ratio: {alpha_ratio(boundary_1['text']):.2f}."
            ),
            source="gen1_output/smoke_test",
            eval_score=boundary_1["_eval_score"],
        )
        golden_samples.append(sample)
        used_prefixes.add(boundary_1["text"][:200])
        print(f"    {sample['id']}: eval={boundary_1['_eval_score']:.4f}, "
              f"reason=lowest_eval_in_gen1")

    # 5b. Gen1 doc that was filtered by Gen2 (low classifier score)
    boundary_2 = None
    for doc in gen1_filt_scored:
        if doc["text"][:200] not in used_prefixes:
            boundary_2 = doc
            break

    if boundary_2:
        sample = make_sample(
            id_str="golden_008",
            text=boundary_2["text"],
            category="boundary",
            expected_outcome="borderline",
            expected_filter_stage="gen2_classifier",
            reason=(
                f"Passed Gen1 heuristic filters but filtered out by Gen2 classifier. "
                f"Eval score: {boundary_2['_eval_score']:.4f}. "
                f"Represents content that passes basic quality checks but fails statistical quality model. "
                f"Word count: {word_count(boundary_2['text'])}."
            ),
            source="gen1_output/smoke_test (filtered by gen2)",
            eval_score=boundary_2["_eval_score"],
        )
        golden_samples.append(sample)
        used_prefixes.add(boundary_2["text"][:200])
        print(f"    {sample['id']}: eval={boundary_2['_eval_score']:.4f}, "
              f"reason=passed_gen1_filtered_gen2")

    # 5c. Mixed quality signals: decent word count + reasonable alpha but odd formatting
    # Look for Gen1 docs with moderate eval score and some formatting issues
    mid_idx = len(gen1_scored) // 2
    boundary_3 = None
    for doc in gen1_scored[mid_idx:]:
        text = doc["text"]
        prefix = text[:200]
        if prefix in used_prefixes:
            continue
        # Look for mixed signals: reasonable length, moderate alpha, but some oddity
        wc = word_count(text)
        ar = alpha_ratio(text)
        spc = special_char_ratio(text)
        # Mixed = moderate quality + some unusual characteristic
        if 100 < wc < 1000 and 0.4 < ar < 0.7 and spc > 0.05:
            boundary_3 = doc
            break
    # Fallback: just pick a mid-range Gen1 doc
    if boundary_3 is None:
        for doc in gen1_scored[mid_idx:mid_idx + 20]:
            if doc["text"][:200] not in used_prefixes:
                boundary_3 = doc
                break

    if boundary_3:
        text = boundary_3["text"]
        sample = make_sample(
            id_str="golden_009",
            text=text,
            category="boundary",
            expected_outcome="borderline",
            expected_filter_stage=None,
            reason=(
                f"Mixed quality signals: word count={word_count(text)}, "
                f"alpha ratio={alpha_ratio(text):.2f}, special char ratio={special_char_ratio(text):.2f}. "
                f"Eval score: {boundary_3['_eval_score']:.4f}. "
                f"Content has decent substance but unusual formatting characteristics."
            ),
            source="gen1_output/smoke_test",
            eval_score=boundary_3["_eval_score"],
        )
        golden_samples.append(sample)
        used_prefixes.add(text[:200])
        print(f"    {sample['id']}: eval={boundary_3['_eval_score']:.4f}, "
              f"words={word_count(text)}, alpha={alpha_ratio(text):.2f}, "
              f"reason=mixed_quality_signals")

    # 5d. Just above Gen2 threshold (barely kept in Gen2)
    gen2_by_gen2_score = sorted(gen2_scored, key=lambda x: x["_gen2_score"])
    boundary_4 = None
    for doc in gen2_by_gen2_score:
        if doc["text"][:200] not in used_prefixes:
            boundary_4 = doc
            break

    if boundary_4:
        sample = make_sample(
            id_str="golden_010",
            text=boundary_4["text"],
            category="boundary",
            expected_outcome="borderline",
            expected_filter_stage=None,
            reason=(
                f"Just above Gen2 classifier threshold. Gen2 score: {boundary_4['_gen2_score']:.4f} "
                f"(threshold ~0.693). Eval score: {boundary_4['_eval_score']:.4f}. "
                f"Barely survived Gen2 selection — small changes in classifier or threshold could flip the outcome. "
                f"Word count: {word_count(boundary_4['text'])}."
            ),
            source="gen2_output/smoke_test",
            eval_score=boundary_4["_eval_score"],
            gen2_score=boundary_4["_gen2_score"],
        )
        golden_samples.append(sample)
        used_prefixes.add(boundary_4["text"][:200])
        print(f"    {sample['id']}: gen2_score={boundary_4['_gen2_score']:.4f}, "
              f"eval={boundary_4['_eval_score']:.4f}, reason=barely_above_gen2_threshold")

    # ------------------------------------------------------------------
    # 5. Save and summarize
    # ------------------------------------------------------------------
    print(f"\n[5/5] Saving {len(golden_samples)} golden samples to {OUTPUT_FILE}...")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for sample in golden_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    file_size = OUTPUT_FILE.stat().st_size
    print(f"  Saved: {OUTPUT_FILE} ({file_size / 1024:.1f} KB)")

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("GOLDEN SAMPLES SUMMARY")
    print("=" * 70)
    print(f"{'ID':<12} {'Category':<14} {'Outcome':<14} {'Filter Stage':<24} {'Eval':>6} {'Gen2':>6} {'Words':>6}")
    print("-" * 90)
    for s in golden_samples:
        eval_str = f"{s.get('eval_score', 0):.3f}" if s.get("eval_score") is not None else "  n/a"
        gen2_str = f"{s.get('gen2_score', 0):.3f}" if s.get("gen2_score") is not None else "  n/a"
        filter_stage = s["expected_filter_stage"] or "null"
        # Estimate word count from stored (possibly truncated) text
        wc = word_count(s["text"])
        print(f"{s['id']:<12} {s['category']:<14} {s['expected_outcome']:<14} "
              f"{filter_stage:<24} {eval_str:>6} {gen2_str:>6} {wc:>6}")

    # Category summary
    cat_counts = Counter(s["category"] for s in golden_samples)
    print(f"\nBy category: {dict(cat_counts)}")
    print(f"Total: {len(golden_samples)} golden samples")

    print("\n" + "=" * 70)
    print("Sample reasons:")
    print("-" * 70)
    for s in golden_samples:
        print(f"  {s['id']}: {s['reason'][:120]}")

    print(f"\nDone. Output: {OUTPUT_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
