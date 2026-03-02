"""
src/gen1_zh/__init__.py
中文预训练数据清洗模块

在 Gen1 英文 Heuristic Pipeline 的基础上，针对中文文本特点进行适配：
- 使用字符数（而非 word count）作为长度指标
- 简繁体中文均支持
- 中文垃圾内容模式（SEO/广告/低质量 UGC）专项过滤

使用方式：
    from src.gen1_zh.pipeline import ChineseGen1Pipeline
    pipeline = ChineseGen1Pipeline(run_config=cfg, pipeline_config=pipe_cfg)
    filtered_docs = pipeline.run(docs)
"""

from src.gen1_zh.chinese_quality_filter import ChineseQualityFilter
from src.gen1_zh.pipeline import ChineseGen1Pipeline

__all__ = ["ChineseQualityFilter", "ChineseGen1Pipeline"]
