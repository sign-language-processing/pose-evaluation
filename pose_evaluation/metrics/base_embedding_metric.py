from typing import TypeVar
from pose_evaluation.metrics.base import BaseMetric
import torch

# Define a type alias for embeddings (e.g., torch.Tensor)
Embedding = TypeVar("Embedding", bound=torch.Tensor)

EmbeddingMetric = BaseMetric[Embedding]