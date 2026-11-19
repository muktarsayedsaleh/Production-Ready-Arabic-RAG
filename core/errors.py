class TransientError(Exception):
    """A provider failure worth retrying later: rate limits, overload, timeouts, network.

    Every provider (LLM, embeddings, reranker) maps its retryable failures to a subclass
    of this, so callers can retry without knowing which provider is behind the interface.
    """
