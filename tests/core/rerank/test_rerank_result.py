import pytest

from core.errors import TransientError
from core.rerank import RankedDocument, RerankResult, TransientRerankError


def test_accepts_results_sorted_best_first() -> None:
    result = RerankResult(ranked=[RankedDocument(2, 0.9), RankedDocument(0, 0.4)])

    assert [doc.index for doc in result.ranked] == [2, 0]


def test_rejects_unsorted_results() -> None:
    with pytest.raises(ValueError, match="best first"):
        RerankResult(ranked=[RankedDocument(0, 0.4), RankedDocument(2, 0.9)])


def test_rejects_duplicate_documents() -> None:
    with pytest.raises(ValueError, match="only once"):
        RerankResult(ranked=[RankedDocument(1, 0.9), RankedDocument(1, 0.4)])


def test_provider_errors_share_one_retryable_base() -> None:
    from core.embeddings import TransientEmbeddingError
    from core.llm import TransientLLMError

    for error in (TransientLLMError, TransientEmbeddingError, TransientRerankError):
        assert issubclass(error, TransientError)
