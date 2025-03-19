from abc import ABC, abstractmethod
from typing import Any, Callable, Generic, Sequence, TypeVar

from tqdm import tqdm

T = TypeVar("T")


class Signature:
    """Represents reproducibility signatures for metrics. Inspired by sacreBLEU"""

    def __init__(self, name: str, args: dict):

        self._abbreviated = {"name": "n", "higher_is_better": "hb"}

        self.signature_info = {"name": name, **args}

    def update(self, key: str, value: Any):
        self.signature_info[key] = value

    def update_abbr(self, key: str, abbr: str):
        self._abbreviated.update({key: abbr})

    def update_signature_and_abbr(self, key: str, abbr: str, args: dict):
        self.update_abbr(key, abbr)
        self.signature_info.update({key: args.get(key, None)})

    def format(self, short: bool = False) -> str:
        parts = []
        # Always print the "name" value first, if available.
        name_value = self.signature_info.get("name")
        if name_value is not None:
            parts.append(str(name_value))
        # Process all other keys.
        for key, value in self.signature_info.items():
            if key == "name" or value is None:
                continue
            # Handle nested signature objects and wrap them in curly braces.
            if hasattr(value, "get_signature"):
                nested_signature = value.get_signature()
                if isinstance(nested_signature, Signature):
                    value = "{" + nested_signature.format(short=short) + "}"
            elif isinstance(value, list) and all(hasattr(v, "get_signature") for v in value):
                value = "[" + ",".join(v.get_signature().format(short=short) for v in value) + "]"

            if isinstance(value, bool):
                value = "yes" if value else "no"
            if isinstance(value, Callable):
                value = value.__name__
            abbreviated_key = self._abbreviated.get(key, key) if short else key
            parts.append(f"{abbreviated_key}:{value}")
        return "|".join(parts)

    def __str__(self) -> str:
        return self.format()

    def __repr__(self) -> str:
        return self.format()


class Score:
    """Inspired by Sacrebleu, a base score class which can add signature information after the value."""

    def __init__(self, name: str, score: float, signature: str) -> None:
        self.name = name
        self.score = score
        self.signature = signature

    def __str__(self):
        return f"{self.signature} = {self.score}"

    def format(
        self,
        width: int = 2,
        score_only: bool = False,
    ) -> str:

        sc = f"{self.score:.{width}f}"

        full_score = f"{self.signature}" if self.signature else self.name
        full_score = f"{full_score} = {sc}"

        if score_only:
            return sc
        return full_score

    def __repr__(self):
        return self.format()


class BaseMetric(ABC, Generic[T]):  # Ensure it extends ABC
    """Base class for all metrics."""

    _SIGNATURE_TYPE = Signature

    def __init__(self, name: str, higher_is_better: bool = False):
        self.name = name
        self.higher_is_better = higher_is_better

    def __call__(self, hypothesis: T, reference: T) -> float:
        return self.score(hypothesis, reference)

    @abstractmethod
    def score(self, hypothesis: T, reference: T) -> float:
        raise NotImplementedError

    def score_with_signature(
        self,
        hypothesis: T,
        reference: T,
        short: bool = False,
    ) -> Score:
        return Score(
            name=self.name,
            score=self.score(hypothesis, reference),
            signature=self.get_signature().format(short=short),
        )

    def score_max(self, hypothesis: T, references: Sequence[T]) -> float:
        all_scores = self.score_all([hypothesis], references)
        return max(max(scores) for scores in all_scores)

    def validate_corpus_score_input(self, hypotheses: Sequence[T], references: Sequence[Sequence[T]]):
        # This method is designed to avoid mistakes in the use of the corpus_score method
        for reference in references:
            assert len(hypotheses) == len(reference), "Hypothesis and reference must have the same number of instances"

    def corpus_score(self, hypotheses: Sequence[T], references: Sequence[list[T]]) -> float:
        """Default implementation: average over sentence scores."""
        self.validate_corpus_score_input(hypotheses, references)
        transpose_references = list(zip(*references))
        scores = [self.score_max(h, r) for h, r in zip(hypotheses, transpose_references)]
        return sum(scores) / len(hypotheses)

    def score_all(self, hypotheses: Sequence[T], references: Sequence[T], progress_bar=False) -> list[list[float]]:
        """Call the score function for each hypothesis-reference pair."""
        return [
            [self.score(h, r) for r in references]
            for h in tqdm(hypotheses, disable=not progress_bar or len(hypotheses) == 1)
        ]

    def score_all_with_signature(
        self,
        hypotheses: Sequence[T],
        references: Sequence[T],
        progress_bar=False,
        short: bool = False,
    ) -> list[list[Score]]:
        return [
            [self.score_with_signature(h, r, short=short) for r in references]
            for h in tqdm(hypotheses, disable=not progress_bar or len(hypotheses) == 1)
        ]

    def __str__(self):
        return str(self.get_signature())

    def get_signature(self) -> Signature:
        return self._SIGNATURE_TYPE(self.name, self.__dict__)
