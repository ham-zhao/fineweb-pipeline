#!/usr/bin/env bash
# scripts/download_sample.sh
# 下载实验所需的样本数据
#
# 包含：
#   1. Common Crawl WARC 文件（约 1-1.5GB 压缩）
#   2. FineWeb sample-10BT 的 1 个 parquet 分片（对比参考）
#   3. 正样本参考数据：Wikipedia 摘要（分类器训练用）
#
# 用法: bash scripts/download_sample.sh
# 预计耗时: 10-20 分钟（取决于网速）
# 注意: 使用 caffeinate 防止 Mac 休眠

set -euo pipefail

# 防止 Mac 休眠
if command -v caffeinate &> /dev/null; then
    caffeinate -i -w $$ &
    CAFFEINATE_PID=$!
    trap "kill $CAFFEINATE_PID 2>/dev/null || true" EXIT
fi

echo "=================================================="
echo "  fineweb-pipeline 数据下载"
echo "  $(date)"
echo "=================================================="

# ── 目录确认 ─────────────────────────────────────────────
mkdir -p data/raw data/reference

# ── 1. Common Crawl WARC ────────────────────────────────
echo ""
echo "1️⃣  下载 Common Crawl WARC 文件..."
echo "   来源: CC-MAIN-2024-10（2024年第一个 Crawl）"

WARC_URL="https://data.commoncrawl.org/crawl-data/CC-MAIN-2024-10/segments/1707947473347.0/warc/CC-MAIN-20240220211055-20240221001055-00000.warc.gz"
WARC_FILE="data/raw/CC-MAIN-20240220211055-20240221001055-00000.warc.gz"

if [ -f "$WARC_FILE" ]; then
    echo "   ⏭️  WARC 文件已存在，跳过下载: $WARC_FILE"
else
    echo "   ⬇️  下载中（约 1-1.5GB，10-15 分钟）..."
    curl -L --progress-bar \
         --retry 3 --retry-delay 5 \
         -o "$WARC_FILE" \
         "$WARC_URL"
    echo "   ✅ WARC 下载完成: $WARC_FILE"
    ls -lh "$WARC_FILE"
fi

# ── 2. FineWeb parquet 分片 ─────────────────────────────
echo ""
echo "2️⃣  下载 FineWeb sample-10BT parquet 分片..."
echo "   来源: HuggingFace HuggingFaceFW/fineweb"

FINEWEB_DIR="data/reference/fineweb_sample"
mkdir -p "$FINEWEB_DIR"
FINEWEB_FILE="$FINEWEB_DIR/train-00000-of-00009.parquet"

if [ -f "$FINEWEB_FILE" ]; then
    echo "   ⏭️  FineWeb 分片已存在，跳过下载"
else
    echo "   ⬇️  下载中（约 400-600MB）..."
    curl -L --progress-bar \
         --retry 3 --retry-delay 5 \
         -H "Accept: application/json" \
         -o "$FINEWEB_FILE" \
         "https://huggingface.co/datasets/HuggingFaceFW/fineweb/resolve/main/data/CC-MAIN-2024-10/train-00000-of-00009.parquet"
    echo "   ✅ FineWeb 分片下载完成: $FINEWEB_FILE"
    ls -lh "$FINEWEB_FILE"
fi

# ── 3. 正样本参考数据（Python 下载）────────────────────
echo ""
echo "3️⃣  下载正样本参考数据（Wikipedia 摘要）..."
echo "   使用 HuggingFace datasets 库（需要 Python 环境已激活）"

source .venv/bin/activate 2>/dev/null || true

python3 - <<'PYEOF'
import sys
sys.path.insert(0, ".")
from src.utils.downloader import download_wikipedia_abstracts
from pathlib import Path

output_path = download_wikipedia_abstracts(
    dest_dir=Path("data/reference"),
    max_docs=10000,
)
print(f"✅ Wikipedia 摘要: {output_path}")
PYEOF

# ── 4. StackExchange 高赞回答（可选）────────────────────
echo ""
echo "4️⃣  下载 StackExchange 高赞回答（正样本补充）..."

python3 - <<'PYEOF'
import json
from pathlib import Path
from datasets import load_dataset

output_path = Path("data/reference/stackexchange_top.jsonl")
if output_path.exists():
    print(f"  ⏭️  StackExchange 数据已存在: {output_path}")
else:
    print("  ⬇️  下载 StackExchange ELI5 高赞回答...")
    ds = load_dataset("eli5", split="train_eli5", streaming=True)
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for doc in ds:
            # 取得分最高的回答
            answers = doc.get("answers", {})
            texts = answers.get("text", [])
            scores = answers.get("score", [])
            if texts and scores:
                best_idx = scores.index(max(scores))
                text = texts[best_idx].strip()
                if len(text.split()) >= 50:  # 至少 50 词
                    f.write(json.dumps({"text": text, "source": "eli5_stackexchange"}, ensure_ascii=False) + "\n")
                    count += 1
            if count >= 5000:
                break
    print(f"  ✅ StackExchange 高赞回答: {count:,} 条 → {output_path}")
PYEOF

# ── 完成 ─────────────────────────────────────────────────
echo ""
echo "=================================================="
echo "✅ 数据下载完成！"
echo ""
echo "文件清单:"
ls -lh data/raw/ 2>/dev/null || echo "  data/raw/: (空)"
ls -lh data/reference/ 2>/dev/null || echo "  data/reference/: (空)"
echo ""
echo "下一步："
echo "  # 验证数据可读（smoke_test 模式）"
echo "  python scripts/run_gen1.py"
echo "=================================================="
