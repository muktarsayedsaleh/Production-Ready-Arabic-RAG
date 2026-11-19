"""Provider-agnostic interface for rerankers (cross-encoders).

Interface only: no implementation ships yet. The local TEI (bge-reranker-v2-m3) and
hosted providers arrive with the retrieval phase.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

from core.errors import TransientError


class TransientRerankError(TransientError):
    """A retryable reranker-provider failure."""


@dataclass(frozen=True)
class RankedDocument:
    # Position of the document in the list passed to `rerank`.
    index: int
    # Relevance score. Comparable within one call only; scales differ between models.
    score: float


@dataclass(frozen=True)
class RerankResult:
    # Best first. At most `top_k` entries when `top_k` was given.
    ranked: list[RankedDocument]
    input_tokens: int = 0
    cost_usd: float = 0.0

    def __post_init__(self) -> None:
        scores = [doc.score for doc in self.ranked]
        if scores != sorted(scores, reverse=True):
            raise ValueError("ranked documents must be sorted by score, best first")
        indices = [doc.index for doc in self.ranked]
        if len(set(indices)) != len(indices):
            raise ValueError("each document may appear only once")


class Reranker(ABC):
    @property
    @abstractmethod
    def model(self) -> str: ...

    @abstractmethod
    async def rerank(
        self, query: str, documents: Sequence[str], *, top_k: int | None = None
    ) -> RerankResult: ...
