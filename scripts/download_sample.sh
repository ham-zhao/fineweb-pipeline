#!/usr/bin/env bash
# scripts/download_sample.sh
# 下载实验所需的样本数据
#
# 包含：
#   1. Common Crawl WARC 文件（约 1-1.5GB 压缩）
#   2. 正样本参考数据：Wikipedia 摘要（分类器训练用）
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

# ── 2. 正样本参考数据（Python 下载）────────────────────
echo ""
echo "2️⃣  下载正样本参考数据（Wikipedia 摘要）..."
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

# ── 3. 教育类正样本（Cosmopedia）────────────────────────
echo ""
echo "3️⃣  下载 Cosmopedia 教育类文本（Gen3 分类器正样本）..."
echo "   使用 HuggingFace Cosmopedia 数据集 openstax 子集"

python3 - <<'PYEOF'
import sys
sys.path.insert(0, ".")
from src.utils.downloader import download_cosmopedia_samples
from pathlib import Path

output_path = download_cosmopedia_samples(
    dest_dir=Path("data/reference"),
    max_docs=5000,
)
print(f"✅ Cosmopedia 教育文本: {output_path}")
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
