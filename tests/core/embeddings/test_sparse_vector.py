import pytest

from core.embeddings import SparseVector


def test_from_weights_sorts_and_drops_zeros() -> None:
    vector = SparseVector.from_weights({42: 0.5, 7: 1.25, 13: 0.0}, dim=250_002)

    assert vector.indices == (7, 42)
    assert vector.values == (1.25, 0.5)
    assert vector.dim == 250_002


def test_empty_vector_is_allowed() -> None:
    vector = SparseVector.from_weights({}, dim=10)

    assert vector.indices == ()
    assert vector.values == ()


@pytest.mark.parametrize(
    ("indices", "values", "dim", "message"),
    [
        ((1, 2), (0.5,), 10, "same length"),
        ((2, 1), (0.5, 0.5), 10, "strictly increasing"),
        ((1, 1), (0.5, 0.5), 10, "strictly increasing"),
        ((-1, 2), (0.5, 0.5), 10, r"\[0, 10\)"),
        ((1, 10), (0.5, 0.5), 10, r"\[0, 10\)"),
        ((1,), (0.0,), 10, "zero weights"),
        ((), (), 0, "dim must be positive"),
    ],
)
def test_rejects_vectors_pgvector_would_reject(
    indices: tuple[int, ...], values: tuple[float, ...], dim: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        SparseVector(indices=indices, values=values, dim=dim)
