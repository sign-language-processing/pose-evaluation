import logging
from typing import Literal, List, Union

import numpy as np
import torch
from sentence_transformers import util as st_util
from torch import Tensor
from torch.types import Number

from pose_evaluation.metrics.base_embedding_metric import EmbeddingMetric

# Useful reference: https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/util.py#L31
# * Helper functions such as batch_to_device, _convert_to_tensor, _convert_to_batch, _convert_to_batch_tensor
# * a whole semantic search function, with chunking and top_k

# See also pgvector's C implementation: https://github.com/pgvector/pgvector/blob/master/src/vector.c
# * cosine_distance: https://github.com/pgvector/pgvector/blob/master/src/vector.c#L658
# * l2_distance https://github.com/pgvector/pgvector/blob/master/src/vector.c#L566

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ValidDistanceKinds = Literal["cosine", "euclidean", "manhattan", "dot"]
TensorConvertableType = Union[List, np.ndarray, Tensor]


class EmbeddingDistanceMetric(EmbeddingMetric):
    def __init__(
        self,
        kind: ValidDistanceKinds = "cosine",
        device: Union[torch.device, str] = None,
        dtype=None,
    ):
        """
        Args:
            kind (ValidDistanceKinds): The type of distance metric, e.g. "cosine", or "euclidean".
            device (Union[torch.device, str]): The device to use for computation.
                If None, automatically detects.
            dtype (torch.dtype): The data type to use for tensors.
                If None, uses torch.get_default_dtype()
        """
        super().__init__(f"EmbeddingDistanceMetric {kind}", higher_is_better=False)
        self.kind = kind
        if device is None:
            self.device = torch.device(st_util.get_device_name())
        else:
            self.device = torch.device(device) if isinstance(device, str) else device

        if dtype is None:
            dtype = torch.get_default_dtype()

        self.dtype = dtype

        # Dispatch table for metric computations
        self._metric_dispatch = {
            "cosine": self.cosine_distances,
            "euclidean": self.euclidean_distances,
            "dot": self.dot_product,
            "manhattan": self.manhattan_distances,
        }

    def to(self, device: Union[torch.device, str]) -> None:
        """
        Explicitly set the device used for tensors.
        """
        self.device = torch.device(device)
        logger.info(f"Device set to: {self.device}")
        return self

    def _to_batch_tensor_on_device(self, data: TensorConvertableType) -> Tensor:
        """
        Convert input data to a batch tensor on the specified device.

        Returns:
            Tensor: Batch tensor representation of the data on the specified device.
        """
        # better performance this way, see https://github.com/pytorch/pytorch/issues/13918
        if isinstance(data, list) and all(isinstance(x, np.ndarray) for x in data):
            data = np.asanyarray(data)

        if isinstance(data, list) and all(isinstance(x, torch.Tensor) for x in data):
            # prevents ValueError: only one element tensors can be converted to Python scalars
            # https://stackoverflow.com/questions/55050717/converting-list-of-tensors-to-tensors-pytorch
            data = torch.stack(data)

        return st_util._convert_to_batch_tensor(data).to(
            device=self.device, dtype=self.dtype
        )

    def score(self, hypothesis: TensorConvertableType, reference: TensorConvertableType) -> Number:
        """
        Compute the distance between two embeddings.

        Returns:
            Number: The calculated distance.

        """
        return self.score_all(hypothesis, reference).item()

    def score_all(
        self,
        hypotheses: Union[List[TensorConvertableType], Tensor],
        references: Union[List[TensorConvertableType], Tensor],
        progress_bar: bool = True,
    ) -> Tensor:
        """
        Compute the distance between all hypotheses and all references.

        Expects 2D inputs. If not already Tensors, will attempt to convert them.

        Returns:
            Tensor: Distance matrix. Row `i` is the distances of `hypotheses[i]` to all rows of `references`.
                Shape is be NxM, where N is the number of hypotheses, and M is the number of references

        Raises:
            TypeError: If either hypotheses or references cannot be converted to a batch tensor
            ValueError: If the specified metric is unsupported.
        """
        try:
            hypotheses = self._to_batch_tensor_on_device(hypotheses)
            references = self._to_batch_tensor_on_device(references)
        except RuntimeError as e:
            raise TypeError(
                f"Inputs must support conversion to device tensors: {e}"
            ) from e

        assert (
            hypotheses.ndim == 2 and references.ndim == 2
        ), f"score_all received non-2D input: hypotheses: {hypotheses.shape}, references: {references.shape}"

        return self._metric_dispatch[self.kind](hypotheses, references)

    def dot_product(
        self, hypotheses: TensorConvertableType, references: TensorConvertableType
    ) -> Tensor:
        """
        Compute the dot product between embeddings.
        Uses sentence_transformers.util.dot_score
        """
        # https://stackoverflow.com/questions/73924697/whats-the-difference-between-torch-mm-torch-matmul-and-torch-mul
        return st_util.dot_score(hypotheses, references)

    def euclidean_similarities(
        self, hypotheses: TensorConvertableType, references: TensorConvertableType
    ) -> Tensor:
        """
        Returns the negative L2 norm/euclidean distances, which is what sentence-transformers uses for similarities.
        Uses sentence_transformers.util.euclidean_sim
        """
        return st_util.euclidean_sim(hypotheses, references)

    def euclidean_distances(
        self, hypotheses: TensorConvertableType, references: TensorConvertableType
    ) -> Tensor:
        """
        Seeing as how sentence-transformers just negates the distances to get "similarities",
        We can re-negate to get them positive again.
        Uses sentence_transformers.util.euclidean_similarities
        """
        return -self.euclidean_similarities(hypotheses, references)

    def cosine_similarities(
        self, hypotheses: TensorConvertableType, references: TensorConvertableType
    ) -> Tensor:
        """
        Calculates cosine similarities, which can be thought of as the angle between two embeddings.
        The min value is -1 (least similar/pointing directly away), and the max is 1 (exactly the same angle).
        Uses sentence_transformers.util.cos_sim
        """
        return st_util.cos_sim(hypotheses, references)

    def cosine_distances(
        self, hypotheses: TensorConvertableType, references: TensorConvertableType
    ) -> Tensor:
        """
        Converts cosine similarities to distances by simply subtracting from 1.
        Max distance is 2, min distance is 0.
        """
        return 1 - self.cosine_similarities(hypotheses, references)

    def manhattan_similarities(
        self, hypotheses: TensorConvertableType, references: TensorConvertableType
    ) -> Tensor:
        """
        Get the L1/Manhattan similarities, aka negative distances.
        Uses sentence_transformers.util.manhattan_sim
        """
        return st_util.manhattan_sim(hypotheses, references)

    def manhattan_distances(
        self, hypotheses: TensorConvertableType, references: TensorConvertableType
    ) -> Tensor:
        """
        Convert Manhattan similarities to distances.
        Sentence transformers defines similarity as negative distances.
        We can re-negate to recover the distances.
        """
        return -self.manhattan_similarities(hypotheses, references)
