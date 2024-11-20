from numpy import ndarray
import torch
import torch.nn.functional as F
from pose_evaluation.metrics.base import BaseMetric

class NumpyArrayEmbeddingMetric(BaseMetric[ndarray]):
    def __init__(self, name: str, higher_is_better: bool = True, kind: str = "cosine", device: torch.device | str = None):
        # Call the base class __init__ to initialize 'name' and 'higher_is_better'
        super().__init__(name, higher_is_better)
        
        self.kind = kind

        if device is None:
            if torch.cuda.is_available():
                self.cuda()
        else:
            self.device = torch.device(device) if isinstance(device, str) else device

    def score(self, hypothesis: ndarray, reference: ndarray) -> float:
        if self.kind == "cosine":
            return F.cosine_similarity(hypothesis, reference)
        elif self.kind == "l2":
            return F.pairwise_distance(hypothesis, reference, p=2)
