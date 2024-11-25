from pose_evaluation.metrics.base_embedding_metric import NumpyArrayEmbeddingMetric
from typing import Literal
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cosine
import torch
import torch.nn.functional as F


class SignCLIPEmbeddingDistanceMetric(NumpyArrayEmbeddingMetric):
    def __init__(self, kind: str = "cosine", device: torch.device | str = "cuda"):
        """
        Initializes the metric with the specified distance type and device.

        Args:
            kind (str): The type of distance metric, either 'cosine' or 'l2'.
            device (torch.device | str): The device to use ('cuda' or 'cpu').
        """
        self.kind = kind
        self.device = torch.device(device) if isinstance(device, str) else device

    def score_all(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Computes the pairwise distance matrix for the provided embeddings.

        Args:
            embeddings (torch.Tensor): A 2D tensor of shape (N, D), where N is the number
                                        of embeddings and D is the feature dimension.

        Returns:
            torch.Tensor: A 2D tensor of shape (N, N) containing pairwise distances.
        """
        # Move embeddings to the specified device
        embeddings = embeddings.to(self.device)

        if self.kind == "cosine":
            # Normalize embeddings to unit norm
            embeddings = F.normalize(embeddings, p=2, dim=1)
            # Compute pairwise cosine similarity
            similarity_matrix = torch.matmul(embeddings, embeddings.T)  # Shape: (N, N)
            distance_matrix = 1 - similarity_matrix  # Cosine distance = 1 - cosine similarity
        elif self.kind == "l2":
            # Compute pairwise L2 distance using broadcasting
            diff = embeddings[:, None, :] - embeddings[None, :, :]  # Shape: (N, N, D)
            distance_matrix = torch.norm(diff, dim=2)  # Shape: (N, N)
        else:
            raise ValueError(f"Unsupported distance metric: {self.kind}")

        return distance_matrix
