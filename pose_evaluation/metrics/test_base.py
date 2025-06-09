import pytest

from pose_evaluation.metrics.base import BaseMetric, Score, Signature

# --- Dummy setup ---


class DummyMetric(BaseMetric[str]):
    def __init__(self, name="Dummy", higher_is_better=False, extra_arg=None):
        super().__init__(name, higher_is_better)
        self.extra_arg = extra_arg
        self._private_stuff = "SHOULD_NOT_BE_INCLUDED"

    def score(self, hypothesis: str, reference: str) -> float:
        return 1.23


# --- Signature Tests ---


def test_signature_format_default():
    sig = Signature("MetricName", {"higher_is_better": True, "foo": "bar"})
    result = sig.format()
    assert "MetricName" in result
    assert "foo:bar" in result
    assert "hb:yes" not in result  # short form not requested


def test_signature_format_short():
    sig = Signature("MetricName", {"higher_is_better": True, "foo": "bar"})
    sig.update_abbr("foo", "f")
    result = sig.format(short=True)
    assert "n:MetricName" not in result  # name shown as value only
    assert "f:bar" in result
    assert "hb:yes" in result


def test_signature_skips_none_and_private():
    sig = Signature("MetricName", {"visible": "yes", "_private": "no", "maybe": None})
    result = sig.format()
    assert "visible:yes" in result
    assert "_private:no" not in result
    assert "maybe:None" not in result


# --- Score Tests ---


def test_score_repr_and_format():
    score = Score("Dummy", 0.8765, "sig:stuff")
    assert str(score) == "sig:stuff = 0.8765"
    assert score.format(width=1) == "sig:stuff = 0.9"
    assert score.format(score_only=True) == "0.88"


# --- Metric Tests ---


@pytest.fixture
def dummy_metric():
    return DummyMetric(name="TestMetric", higher_is_better=True, extra_arg="X")


def test_score_and_signature(dummy_metric):
    result = dummy_metric.score_with_signature("a", "b")
    assert isinstance(result, Score)
    assert result.name == "TestMetric"
    assert result.score == 1.23
    assert "extra_arg:X" in result.signature
    assert "_private" not in result.signature


def test_score_all_and_max(dummy_metric):
    h = ["a", "b"]
    r = ["x", "y"]
    scores = dummy_metric.score_all(h, r)
    assert len(scores) == 2
    assert all(len(row) == 2 for row in scores)
    assert all(score == 1.23 for row in scores for score in row)

    max_score = dummy_metric.score_max("a", r)
    assert max_score == 1.23


def test_score_all_with_signature(dummy_metric):
    h = ["a"]
    r = ["x", "y"]
    results = dummy_metric.score_all_with_signature(h, r, short=True)
    assert isinstance(results[0][0], Score)
    assert "TestMetric" in results[0][0].signature or "n:TestMetric" in results[0][0].signature


def test_corpus_score_and_validation(dummy_metric):
    hypotheses = ["a", "b"]
    references = [["x", "y"], ["x", "y"]]
    score = dummy_metric.corpus_score(hypotheses, references)
    assert score == 1.23


def test_signature_nested_objects():
    class DummySubMetric:
        def get_signature(self):
            return Signature("Sub", {"param": 42})

    sig = Signature("Main", {"sub": DummySubMetric()})
    formatted = sig.format()
    assert "{Sub|param:42}" in formatted


def test_signature_list_of_nested():
    class DummySub:
        def get_signature(self):
            return Signature("X", {"v": 1})

    nested = [DummySub(), DummySub()]
    sig = Signature("Metric", {"subs": nested})
    formatted = sig.format()
    assert formatted.count("X|v:1") == 2
