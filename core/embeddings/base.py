"""Provider-agnostic interfaces for dense and sparse embeddings.

Interfaces only: no implementation ships yet. The local TEI (bge-m3) and hosted providers
arrive with the retrieval phase.

Queries and documents are embedded through separate methods because many models treat
them differently: multilingual-e5 needs "query: " / "passage: " prefixes, Cohere takes an
`input_type`, and bge-m3 treats both the same. The caller shouldn't have to know which.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

from core.errors import TransientError


class TransientEmbeddingError(TransientError):
    """A retryable embedding-provider failure."""


@dataclass(frozen=True)
class DenseEmbeddings:
    vectors: list[list[float]]
    # Tokens billed by the provider; 0 when the provider doesn't report them (local models).
    input_tokens: int = 0
    cost_usd: float = 0.0


@dataclass(frozen=True)
class SparseVector:
    """A sparse vector in the model's vocabulary space: token id -> weight.

    Maps directly onto pgvector's `sparsevec`, which wants sorted, unique, 0-based indices
    below `dim` and drops zero weights.
    """

    indices: tuple[int, ...]
    values: tuple[float, ...]
    dim: int

    def __post_init__(self) -> None:
        if len(self.indices) != len(self.values):
            raise ValueError("indices and values must have the same length")
        if self.dim <= 0:
            raise ValueError("dim must be positive")
        if any(b <= a for a, b in zip(self.indices, self.indices[1:], strict=False)):
            raise ValueError("indices must be strictly increasing")
        if self.indices and (self.indices[0] < 0 or self.indices[-1] >= self.dim):
            raise ValueError(f"indices must be in [0, {self.dim})")
        if any(value == 0 for value in self.values):
            raise ValueError("zero weights must be dropped, not stored")

    @classmethod
    def from_weights(cls, weights: dict[int, float], dim: int) -> "SparseVector":
        """Build from an unordered token-id -> weight mapping, dropping zero weights."""
        kept = sorted((index, value) for index, value in weights.items() if value != 0)
        return cls(
            indices=tuple(index for index, _ in kept),
            values=tuple(value for _, value in kept),
            dim=dim,
        )


@dataclass(frozen=True)
class SparseEmbeddings:
    vectors: list[SparseVector]
    input_tokens: int = 0
    cost_usd: float = 0.0


class DenseEmbedder(ABC):
    @property
    @abstractmethod
    def model(self) -> str:
        """Identifier stored next to each vector, so a model change forces a reindex."""

    @property
    @abstractmethod
    def dimensions(self) -> int: ...

    @abstractmethod
    async def embed_documents(self, texts: Sequence[str]) -> DenseEmbeddings: ...

    @abstractmethod
    async def embed_query(self, text: str) -> DenseEmbeddings: ...


class SparseEmbedder(ABC):
    """Learned lexical weights (e.g. bge-m3 sparse), used as one of the keyword backends."""

    @property
    @abstractmethod
    def model(self) -> str: ...

    @property
    @abstractmethod
    def vocabulary_size(self) -> int: ...

    @abstractmethod
    async def embed_documents(self, texts: Sequence[str]) -> SparseEmbeddings: ...

    @abstractmethod
    async def embed_query(self, text: str) -> SparseEmbeddings: ...


__all__ = [
    "DenseEmbedder",
    "DenseEmbeddings",
    "SparseEmbedder",
    "SparseEmbeddings",
    "SparseVector",
    "TransientEmbeddingError",
]
