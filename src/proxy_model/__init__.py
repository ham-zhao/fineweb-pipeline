"""
src/proxy_model/__init__.py
Proxy Model 评估模块

提供对 scripts/run_proxy_training.py 训练产物的加载和评估能力，
供 Notebook 09 及后续分析脚本使用。

使用方式：
    from src.proxy_model.evaluator import ProxyModelEvaluator
    ev = ProxyModelEvaluator("results/proxy_models/gen3/model.pt")
    ppl = ev.compute_perplexity(text_list)
"""

from src.proxy_model.evaluator import ProxyModelEvaluator

__all__ = ["ProxyModelEvaluator"]
