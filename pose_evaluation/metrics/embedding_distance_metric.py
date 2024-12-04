from typing import Literal, List
import torch
from torch import Tensor
import numpy as np
from sentence_transformers import util as st_util
from pose_evaluation.metrics.base_embedding_metric import EmbeddingMetric


# Useful reference: https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/util.py#L31
# * Helper functions such as batch_to_device, _convert_to_tensor, _convert_to_batch, _convert_to_batch_tensor
# * a whole semantic search function, with chunking and top_k

# See also pgvector's C implementation: https://github.com/pgvector/pgvector/blob/master/src/vector.c
# * cosine_distance: https://github.com/pgvector/pgvector/blob/master/src/vector.c#L658
# * l2_distance https://github.com/pgvector/pgvector/blob/master/src/vector.c#L566


class EmbeddingDistanceMetric(EmbeddingMetric):
    def __init__(
        self,
        kind: Literal["cosine", "euclidean", "dot"] = "cosine",
        device: torch.device | str = None,
        dtype=torch.float64,
    ):
        """
        Initialize the embedding distance metric.

        Args:
            kind (Literal["cosine", "euclidean"]): The type of distance metric.
            device (torch.device | str): The device to use for computation. If None, automatically detects.
        """
        super().__init__(f"EmbeddingDistanceMetric {kind}", higher_is_better=False)
        self.kind = kind
        if device is None:
            self.device = torch.device(st_util.get_device_name())
        else:
            self.device = torch.device(device) if isinstance(device, str) else device

        self.dtype = dtype

    def _to_device_tensor(self, data: list | np.ndarray | Tensor, dtype=None) -> Tensor:
        if dtype is None:
            dtype = self.dtype
        return st_util._convert_to_tensor(data).to(device=self.device, dtype=dtype)

    def _to_batch_tensor_on_device(self, data: list | np.ndarray | Tensor, dtype=None) -> Tensor:
        if dtype is None:
            dtype = self.dtype
        return st_util._convert_to_batch_tensor(data).to(device=self.device, dtype=dtype)

    def score(
        self,
        hypothesis: list | np.ndarray | Tensor,
        reference: list | np.ndarray | Tensor,
    ) -> float:
        """
        Compute the distance between two embeddings.

        Args:
            hypothesis (list| np.ndarray | Tensor): A single embedding vector.
            reference (list| np.ndarray | Tensor): Another single embedding vector.

        Returns:
            float: The calculated distance.
        """
        if hypothesis is None or reference is None:
            raise ValueError("Neither 'hypothesis' nor 'reference' can be None.")

        try:
            hypothesis = self._to_batch_tensor_on_device(hypothesis)
            reference = self._to_batch_tensor_on_device(reference)
        except RuntimeError as e:
            raise TypeError(f"Inputs must support conversion to device tensors: {e}") from e
        return self.score_all(hypothesis, reference).item()

    def score_all(
        self,
        hypotheses: List[list | np.ndarray | Tensor],
        references: List[list | np.ndarray | Tensor],
        progress_bar: bool = True,
    ) -> Tensor:
        """
        Compute the pairwise distance between all hypotheses and references.
        Expects 2D inputs, where each element in the second dimension is one embedding

        Args:
            hypotheses (list[list| np.ndarray | Tensor]): List of hypothesis embeddings.
            references (list[list| np.ndarray | Tensor]): List of reference embeddings.
            progress_bar (bool): Whether to display a progress bar.

        Returns:
            Tensor, distance matrix. Row i is the distances of hypotheses[i] to all rows of references
        """
        # Convert inputs to tensors and stack
        hypotheses = torch.stack([self._to_device_tensor(h) for h in hypotheses])
        references = torch.stack([self._to_device_tensor(r) for r in references])

        if self.kind == "dot":
            distance_matrix = self.dot_product(hypotheses, references)

        elif self.kind == "cosine":
            distance_matrix = self.cosine_distances(hypotheses, references)

        elif self.kind == "euclidean":
            distance_matrix = self.euclidean_distances(hypotheses, references)

        elif self.kind == "manhattan":
            distance_matrix = self.manhattan_distances(hypotheses, references)

        else:
            raise ValueError(f"Unsupported distance metric: {self.kind}")

        return distance_matrix

    def dot_product(self, hypotheses: list | np.ndarray | Tensor, references: list | np.ndarray | Tensor) -> Tensor:
        # TODO: test if this gives the same thing as previous matmul implementation, see stack overflow link below:
        # https://stackoverflow.com/questions/73924697/whats-the-difference-between-torch-mm-torch-matmul-and-torch-mul
        return st_util.dot_score(hypotheses, references)

    def euclidean_similarities(
        self, hypotheses: list | np.ndarray | Tensor, references: list | np.ndarray | Tensor
    ) -> Tensor:
        """
        Returns the negative L2 norm/euclidean distances, which is what sentence-transformers uses for similarities.
        """
        return st_util.euclidean_sim(hypotheses, references)

    def euclidean_distances(
        self, hypotheses: list | np.ndarray | Tensor, references: list | np.ndarray | Tensor
    ) -> Tensor:
        """
        Seeing as how sentence-transformers just negates the distances to get "similarities",
        We can re-negate to get them positive again.
        """
        return -self.euclidean_similarities(hypotheses, references)

    def cosine_similarities(
        self, hypotheses: list | np.ndarray | Tensor, references: list | np.ndarray | Tensor
    ) -> Tensor:
        """
        Calculates cosine similarities, which can be thought of as the angle between two embeddings.
        The min value is -1 (least similar/pointing directly away), and the max is 1 (exactly the same angle).
        """
        return st_util.cos_sim(hypotheses, references)

    def cosine_distances(
        self, hypotheses: list | np.ndarray | Tensor, references: list | np.ndarray | Tensor
    ) -> Tensor:
        """
        Converts cosine similarities to distances by simply subtracting from 1.
        Max distance is 2, min distance is 0.
        """
        return 1 - self.cosine_similarities(hypotheses, references)

    def manhattan_similarities(
        self, hypotheses: list | np.ndarray | Tensor, references: list | np.ndarray | Tensor
    ) -> Tensor:
        """
        Get the L1/Manhattan similarities, aka negative distances.
        """
        return st_util.manhattan_sim(hypotheses, references)

    def manhattan_distances(
        self, hypotheses: list | np.ndarray | Tensor, references: list | np.ndarray | Tensor
    ) -> Tensor:
        """
        Sentence transformers defines similarity as negative distances.
        We can re-negate to recover the distances.
        """
        return -self.manhattan_similarities(hypotheses, references)
