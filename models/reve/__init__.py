# coding=utf-8
"""REVE model with mechanistic interpretability support."""

from .configuration_reve import ReveConfig
from .modeling_reve import (            
    Reve,
    TransformerBackbone,
    TransformerBlock,
    Attention,
    FeedForward,
    GEGLU,
    RMSNorm,
    FourierEmb4D,
)
from .hooked_reve import (
    HookedSAEReve,
    get_deep_attr,
    set_deep_attr,
)

__all__ = [
    "ReveConfig",
    "Reve",
    "TransformerBackbone",
    "TransformerBlock",
    "Attention",
    "FeedForward",
    "GEGLU",
    "RMSNorm",
    "FourierEmb4D",
    "HookedSAEReve",
    "get_deep_attr",
    "set_deep_attr",
]
