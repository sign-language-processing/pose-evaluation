from typing import TypeVar

import torch

from pose_evaluation.metrics.base import BaseMetric

# Define a type alias for embeddings (e.g., torch.Tensor)
Embedding = TypeVar("Embedding", bound=torch.Tensor)

EmbeddingMetric = BaseMetric[Embedding]
