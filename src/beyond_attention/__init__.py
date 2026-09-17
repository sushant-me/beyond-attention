"""A selective state-space model, implemented from scratch and measured honestly."""

from .model import (
    AttentionBlock,
    LanguageModel,
    RMSNorm,
    SelectiveSSMBlock,
    build_pair,
    count_parameters,
)
from .ssm import (
    selective_scan,
    selective_scan_associative,
    selective_scan_chunked,
    selective_scan_reference,
)
from .tasks import Batch, accuracy, mqar_batch, vocabulary_for
from .train import Result, train

__version__ = "0.1.0"

__all__ = [
    "AttentionBlock", "LanguageModel", "RMSNorm", "SelectiveSSMBlock",
    "build_pair", "count_parameters",
    "selective_scan", "selective_scan_associative", "selective_scan_chunked",
    "selective_scan_reference",
    "Batch", "accuracy", "mqar_batch", "vocabulary_for",
    "Result", "train",
    "__version__",
]
