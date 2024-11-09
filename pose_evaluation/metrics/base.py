# pylint: disable=undefined-variable
from tqdm import tqdm


class BaseMetric[T]:
    """Base class for all metrics."""

    def __init__(self, name: str, higher_is_better: bool = True):
        self.name = name
        self.higher_is_better = higher_is_better

    def __call__(self, hypothesis: T, reference: T) -> float:
        return self.score(hypothesis, reference)

    def score(self, hypothesis: T, reference: T) -> float:
        raise NotImplementedError

    def score_max(self, hypothesis: T, references: list[T]) -> float:
        all_scores = self.score_all([hypothesis], references)
        return max(max(scores) for scores in all_scores)

    def validate_corpus_score_input(self, hypotheses: list[T], references: list[list[T]]):
        # This method is designed to avoid mistakes in the use of the corpus_score method
        for reference in references:
            assert len(hypotheses) == len(reference), \
                "Hypothesis and reference must have the same number of instances"

    def corpus_score(self, hypotheses: list[T], references: list[list[T]]) -> float:
        # Default implementation: average over sentence scores
        self.validate_corpus_score_input(hypotheses, references)
        transpose_references = list(zip(*references))
        return sum(self.score_max(h, r) for h, r in zip(hypotheses, transpose_references)) / len(hypotheses)

    def score_all(self, hypotheses: list[T], references: list[T], progress_bar=True) -> list[list[float]]:
        # Default implementation: call the score function for each hypothesis-reference pair
        return [[self.score(h, r) for r in references]
                for h in tqdm(hypotheses, disable=not progress_bar or len(hypotheses) == 1)]

    def __str__(self):
        return self.name
