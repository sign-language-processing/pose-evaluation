from typing import Literal, Union, List
import torch
import torch.nn.functional as F
import numpy as np

from pose_evaluation.metrics.base_embedding_metric import EmbeddingMetric


class EmbeddingDistanceMetric(EmbeddingMetric):
    def __init__(self, kind: Literal["cosine", "l2"] = "cosine", device: Union[torch.device, str] = None):
        """
        Initialize the embedding distance metric.

        Args:
            kind (Literal["cosine", "l2"]): The type of distance metric.
            device (torch.device | str): The device to use for computation. If None, automatically detects.
        """
        super().__init__(f"EmbeddingDistanceMetric {kind}", higher_is_better=False)
        self.kind = kind
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device) if isinstance(device, str) else device

    def _to_tensor(self, data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Convert input to a PyTorch tensor if it is a NumPy array.

        Args:
            data (np.ndarray | torch.Tensor): Input data.

        Returns:
            torch.Tensor: Tensor on the correct device.
        """
        if isinstance(data, np.ndarray):
            data = torch.tensor(data, dtype=torch.float32)
        return data.to(self.device)

    def score(self, hypothesis: Union[np.ndarray, torch.Tensor], reference: Union[np.ndarray, torch.Tensor]) -> float:
        """
        Compute the distance between two embeddings.

        Args:
            hypothesis (np.ndarray | torch.Tensor): A single embedding vector.
            reference (np.ndarray | torch.Tensor): Another single embedding vector.

        Returns:
            float: The calculated distance.
        """
        hypothesis = self._to_tensor(hypothesis)
        reference = self._to_tensor(reference)

        if self.kind == "cosine":
            # Normalize both embeddings to unit length
            hypothesis = F.normalize(hypothesis, p=2, dim=0)
            reference = F.normalize(reference, p=2, dim=0)
            # Cosine similarity, converted to distance
            similarity = torch.dot(hypothesis, reference).item()
            return 1 - similarity
        elif self.kind == "l2":
            # L2 distance
            return torch.norm(hypothesis - reference).item()
        else:
            raise ValueError(f"Unsupported distance metric: {self.kind}")

    def score_all(
        self,
        hypotheses: List[Union[np.ndarray, torch.Tensor]],
        references: List[Union[np.ndarray, torch.Tensor]],
        progress_bar: bool = True,
    ) -> torch.Tensor:
        """
        Compute the pairwise distance between all hypotheses and references. Expects 2D inputs.

        Args:
            hypotheses (list[np.ndarray | torch.Tensor]): List of hypothesis embeddings.
            references (list[np.ndarray | torch.Tensor]): List of reference embeddings.
            progress_bar (bool): Whether to display a progress bar.

        Returns:
            torch.Tensor, distance matrix. Row i is the distances of hypotheses[i] to all rows of references
        """
        # Convert inputs to tensors and stack
        hypotheses = torch.stack([self._to_tensor(h) for h in hypotheses])
        references = torch.stack([self._to_tensor(r) for r in references])

        if self.kind == "cosine":
            # Normalize the tensors along the feature dimension (dim=1)
            normalized_hypotheses = F.normalize(hypotheses, dim=1)
            normalized_references = F.normalize(references, dim=1)

            # Calculate cosine similarity between all hypothesis-reference pairs
            cosine_similarities = torch.matmul(normalized_hypotheses, normalized_references.T)

            # Convert cosine similarities to cosine distances
            distance_matrix = 1 - cosine_similarities
        elif self.kind == "l2":
            # Use broadcasting to calculate pairwise L2 distances
            diff = hypotheses[:, None, :] - references[None, :, :]
            distance_matrix = torch.norm(diff, dim=2)
        else:
            raise ValueError(f"Unsupported distance metric: {self.kind}")

        return distance_matrix.cpu()
