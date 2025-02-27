# pylint: disable=undefined-variable
from typing import Any, Callable, Sequence
from tqdm import tqdm


class Signature:
    """Represents reproducibility signatures for metrics. Inspired by sacreBLEU
    """
    def __init__(self, name:str, args: dict):

        self._abbreviated = {
            "name":"n",
            "higher_is_better":"hb"
        }

        self.signature_info = {"name": name, **args}

    def update(self, key: str, value: Any):
        self.signature_info[key] = value

    def update_abbr(self, key, abbr):
        self._abbreviated.update({
            key: abbr
        })


    def update_signature_and_abbr(self, key:str, abbr:str, args:dict):
        self.update_abbr(key, abbr)

        self.signature_info.update({
            key: args.get(key, None)
        })

    def format(self, short: bool = False) -> str:
        pairs = []
        keys = list(self.signature_info.keys())
        for name in keys:
            value = self.signature_info[name]
            if value is not None:
                # Check for nested signature objects
                if hasattr(value, "get_signature"):
                    # Wrap nested signatures in brackets
                    nested_signature = value.get_signature()
                    if isinstance(nested_signature, Signature):
                        nested_signature = nested_signature.format(short=short)
                    value = f"{{{nested_signature}}}"
                if isinstance(value, bool):
                    # Replace True/False with yes/no
                    value = "yes" if value else "no"
                if isinstance(value, Callable):
                    value = value.__name__

                # if the abbreviation is not defined, use the full name as a fallback.
                abbreviated_name = self._abbreviated.get(name, name)
                final_name = abbreviated_name if short else name
                pairs.append(f"{final_name}:{value}")

        return "|".join(pairs)

    def __str__(self):
        return self.format()

    def __repr__(self):
        return self.format()

class ScoreWithSignature(float):
    __slots__ = ("_signature",)  # Explicitly allow the attribute

    def __new__(cls, value, signature):
        obj = super().__new__(cls, value)  # Create the float instance
        obj._signature = signature  # Store signature object
        return obj

    def __str__(self):
        return f"{self._signature.format()} = {float(self)}"

    def __repr__(self):
        return f"ScoreWithSignature({super().__repr__()}, signature={repr(self._signature)})"


class BaseMetric[T]:
    """Base class for all metrics."""
    # Each metric should define its Signature class' name here
    _SIGNATURE_TYPE = Signature

    def __init__(self, name: str, higher_is_better: bool = False):
        self.name = name
        self.higher_is_better = higher_is_better

    def __call__(self, hypothesis: T, reference: T) -> float:
        return self.score(hypothesis, reference)

    def score(self, hypothesis: T, reference: T) -> float:
        raise NotImplementedError

    def score_with_signature(self, hypothesis: T, reference: T) -> ScoreWithSignature:
        return ScoreWithSignature(self.score(hypothesis, reference), self.get_signature())


    def score_max(self, hypothesis: T, references: Sequence[T]) -> float:
        all_scores = self.score_all([hypothesis], references)
        return max(max(scores) for scores in all_scores)

    def validate_corpus_score_input(self, hypotheses: Sequence[T], references: Sequence[Sequence[T]]):
        # This method is designed to avoid mistakes in the use of the corpus_score method
        for reference in references:
            assert len(hypotheses) == len(reference), \
                "Hypothesis and reference must have the same number of instances"

    def corpus_score(self, hypotheses: Sequence[T], references: Sequence[list[T]]) -> float:
        # Default implementation: average over sentence scores
        self.validate_corpus_score_input(hypotheses, references)
        transpose_references = list(zip(*references))
        scores = [self.score_max(h, r) for h, r in zip(hypotheses, transpose_references)]
        return sum(scores) / len(hypotheses)

    def score_all(self, hypotheses: Sequence[T], references: Sequence[T], progress_bar=True) -> list[list[float]]:
        # Default implementation: call the score function for each hypothesis-reference pair
        return [[self.score(h, r) for r in references]
                for h in tqdm(hypotheses, disable=not progress_bar or len(hypotheses) == 1)]

    def __str__(self):
        return self.name

    def get_signature(self) -> Signature:
        return self._SIGNATURE_TYPE(self.name, self.__dict__)
