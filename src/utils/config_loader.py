"""
src/utils/config_loader.py
统一配置加载器：所有 pipeline 和 notebook 通过此模块读取配置，
确保 run_mode 切换（smoke_test / full_run）对全局生效。
两档运行模式：smoke_test (12K docs) / full_run (100K docs)。
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent.parent / ".env")
except ImportError:
    pass


# 项目根目录（从本文件位置推断）
PROJECT_ROOT = Path(__file__).parent.parent.parent


def _load_yaml(path: Path) -> Dict[str, Any]:
    """加载 YAML 文件并返回字典。"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_run_config(config_path: Optional[str] = None, run_mode_override: Optional[str] = None) -> Dict[str, Any]:
    """
    加载运行模式配置，返回当前 run_mode 下的参数字典。

    Args:
        config_path: 自定义配置文件路径（可选）
        run_mode_override: 命令行覆盖 run_mode（可选，优先于 YAML 文件中的值）

    Returns:
        dict，包含 run_mode、当前模式的所有参数、以及 paths 配置。

    Example:
        >>> cfg = load_run_config()
        >>> print(cfg["run_mode"])          # "smoke_test"
        >>> print(cfg["doc_limit"])         # 1000
        >>> print(cfg["paths"]["raw_data"]) # "data/raw"
    """
    path = Path(config_path) if config_path else PROJECT_ROOT / "configs" / "run_config.yaml"
    raw = _load_yaml(path)

    run_mode = run_mode_override or raw["run_mode"]
    if run_mode not in ("smoke_test", "full_run"):
        raise ValueError(f"run_mode 必须是 'smoke_test' 或 'full_run'，当前值: {run_mode}")

    # 将当前模式的参数提升到顶层，方便直接访问
    mode_params = raw[run_mode]
    result = {
        "run_mode": run_mode,
        "random_seed": raw.get("random_seed", 42),
        "paths": raw.get("paths", {}),
        **mode_params,
    }
    return result


def load_api_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    加载 LLM API 配置，返回当前 provider 的完整配置。

    Returns:
        dict，包含 provider、api_key、以及当前 provider 的模型参数。
    """
    path = Path(config_path) if config_path else PROJECT_ROOT / "configs" / "api_config.yaml"
    raw = _load_yaml(path)

    provider = raw["provider"]
    api_key = os.environ.get("FINEWEB_API_KEY", raw.get("api_key", ""))

    if api_key == "YOUR_API_KEY_HERE" or not api_key:
        raise ValueError(
            "API Key 未配置！请在 configs/api_config.yaml 中设置 api_key，"
            "或通过环境变量 FINEWEB_API_KEY 传入。"
        )

    provider_config = raw.get(provider, {})
    return {
        "provider": provider,
        "api_key": api_key,
        "rephrasing": raw.get("rephrasing", {}),
        **provider_config,
    }


def load_pipeline_config(generation: int, config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    加载指定代次的 pipeline 配置。

    Args:
        generation: 1, 2 或 3
        config_path: 自定义配置路径（可选）

    Returns:
        pipeline 配置字典。
    """
    if generation not in (1, 2, 3):
        raise ValueError(f"generation 必须是 1, 2 或 3，当前值: {generation}")

    if config_path:
        path = Path(config_path)
    else:
        path = PROJECT_ROOT / "configs" / f"gen{generation}_config.yaml"

    return _load_yaml(path)


def load_eval_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """加载评估体系配置。"""
    path = Path(config_path) if config_path else PROJECT_ROOT / "configs" / "eval_config.yaml"
    return _load_yaml(path)


def get_output_path(generation: int, run_cfg: Optional[Dict] = None) -> Path:
    """
    获取指定代次的输出目录路径（绝对路径）。

    不同 run_mode 的输出隔离在子目录中，防止互相覆盖：
        data/gen1_output/smoke_test/
        data/gen1_output/full_run/

    raw_data (generation=0) 不分 run_mode（两档共用同一份原始数据）。

    Args:
        generation: 0 = 原始数据, 1/2/3 = 各代输出
        run_cfg: load_run_config() 的返回值（可选，不传则自动加载）
    """
    if run_cfg is None:
        run_cfg = load_run_config()

    paths = run_cfg.get("paths", {})
    run_mode = run_cfg.get("run_mode", "smoke_test")

    mapping = {
        0: paths.get("raw_data", "data/raw"),
        1: paths.get("gen1_output", "data/gen1_output"),
        2: paths.get("gen2_output", "data/gen2_output"),
        3: paths.get("gen3_output", "data/gen3_output"),
    }

    base = PROJECT_ROOT / mapping[generation]

    # raw_data 不分 run_mode
    if generation == 0:
        return base

    return base / run_mode


def print_config_summary(run_cfg: Optional[Dict] = None) -> None:
    """打印当前运行配置摘要（Notebook 开头调用，方便确认模式）。"""
    if run_cfg is None:
        run_cfg = load_run_config()

    mode = run_cfg["run_mode"]
    desc = run_cfg.get("description", "")

    print(f"{'=' * 50}")
    print(f"  当前运行模式: {mode.upper()}")
    print(f"  {desc}")
    print(f"{'─' * 50}")
    print(f"  doc_limit       : {run_cfg.get('doc_limit', 'N/A'):,}")
    print(f"  eval_sample_size: {run_cfg.get('eval_sample_size', 'N/A'):,}")
    print(f"  audit_sample_size: {run_cfg.get('audit_sample_size', 'N/A'):,}")
    print(f"  rewrite_count   : {run_cfg.get('rewrite_count', 'N/A'):,}")
    print(f"  random_seed     : {run_cfg.get('random_seed', 42)}")
    print(f"  output_subdir   : .../<run_mode>/ = .../{mode}/")
    print(f"{'=' * 50}")
