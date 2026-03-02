#!/usr/bin/env bash
# setup.sh — 一键环境安装脚本
# 用法: bash setup.sh
# 预计耗时: 5-10 分钟（首次安装）
# 平台: macOS (M 系列芯片), Python 3.10+

set -euo pipefail

echo "=================================================="
echo "  fineweb-pipeline 环境安装"
echo "  预训练数据清洗三代方法论对比实验"
echo "=================================================="

# ── 检查 Python 版本 ────────────────────────────────────
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]); then
    echo "❌ 需要 Python 3.10+，当前版本: $PYTHON_VERSION"
    echo "   推荐使用: brew install python@3.11"
    exit 1
fi
echo "✅ Python 版本: $PYTHON_VERSION"

# ── 创建虚拟环境 ─────────────────────────────────────────
VENV_DIR=".venv"
if [ ! -d "$VENV_DIR" ]; then
    echo ""
    echo "📦 创建虚拟环境 .venv ..."
    python3 -m venv "$VENV_DIR"
fi

# 激活虚拟环境
source "$VENV_DIR/bin/activate"
echo "✅ 虚拟环境已激活: $VENV_DIR"

# ── 升级 pip ────────────────────────────────────────────
echo ""
echo "⬆️  升级 pip ..."
pip install --upgrade pip --quiet

# ── 安装依赖 ─────────────────────────────────────────────
echo ""
echo "📦 安装 Python 依赖（约 3-5 分钟）..."
pip install -r requirements.txt --quiet

# ── 验证关键包 ───────────────────────────────────────────
echo ""
echo "🔍 验证关键依赖..."

python3 -c "import datatrove; print(f'  ✅ datatrove {datatrove.__version__}')" 2>/dev/null || echo "  ⚠️  datatrove 安装失败，请手动检查"
python3 -c "import fasttext; print('  ✅ fasttext')" 2>/dev/null || echo "  ⚠️  fasttext-wheel 安装失败"
python3 -c "import torch; print(f'  ✅ torch {torch.__version__} | MPS 可用: {torch.backends.mps.is_available()}')" 2>/dev/null || echo "  ⚠️  torch 安装失败"
python3 -c "import trafilatura; print(f'  ✅ trafilatura {trafilatura.__version__}')" 2>/dev/null || echo "  ⚠️  trafilatura 安装失败"
python3 -c "import anthropic; print(f'  ✅ anthropic {anthropic.__version__}')" 2>/dev/null || echo "  ⚠️  anthropic 安装失败"
python3 -c "import sklearn; print(f'  ✅ scikit-learn {sklearn.__version__}')" 2>/dev/null || echo "  ⚠️  scikit-learn 安装失败"

# ── 安装 Jupyter kernel ──────────────────────────────────
echo ""
echo "📓 安装 Jupyter kernel ..."
python3 -m ipykernel install --user --name fineweb-pipeline --display-name "Python (fineweb-pipeline)" --quiet
echo "  ✅ Jupyter kernel: fineweb-pipeline"

# ── 创建必要目录（确保 data/ 下子目录存在）─────────────
echo ""
echo "📁 确认数据目录..."
mkdir -p data/raw data/gen1_output data/gen2_output data/gen3_output data/reference
mkdir -p results/figures results/reports/audit results/quality_scores results/proxy_models
echo "  ✅ 数据目录就绪"

# ── 检查 API 配置 ────────────────────────────────────────
echo ""
API_KEY=$(python3 -c "import yaml; c=yaml.safe_load(open('configs/api_config.yaml')); print(c.get('api_key',''))" 2>/dev/null || echo "")
if [ "$API_KEY" = "YOUR_API_KEY_HERE" ] || [ -z "$API_KEY" ]; then
    echo "⚠️  提醒：请在 configs/api_config.yaml 中填入你的 API Key"
    echo "   第三代 Pipeline 的 LLM 改写环节需要 API Key"
else
    echo "✅ API Key 已配置"
fi

# ── 完成提示 ─────────────────────────────────────────────
echo ""
echo "=================================================="
echo "✅ 环境安装完成！"
echo ""
echo "使用方法："
echo "  # 激活环境"
echo "  source .venv/bin/activate"
echo ""
echo "  # 下载样本数据（约 15 分钟）"
echo "  bash scripts/download_sample.sh"
echo ""
echo "  # 运行三代 Pipeline（smoke_test 模式）"
echo "  python scripts/run_gen1.py"
echo "  python scripts/run_gen2.py"
echo "  python scripts/run_gen3.py"
echo ""
echo "  # 启动 Jupyter"
echo "  jupyter lab notebooks/"
echo "=================================================="
