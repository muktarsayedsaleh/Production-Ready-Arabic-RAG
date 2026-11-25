"""Checks on the committed synthetic internal documents."""

import pytest

from evals.dataset import Caller, Document, Language, load_corpus
from evals.wikipedia import CORPUS_DIR

INTERNAL_DIR = CORPUS_DIR / "internal"


def internal_documents() -> dict[str, Document]:
    return {
        doc.id: doc for doc in load_corpus(CORPUS_DIR).values() if doc.path.startswith("internal/")
    }


def text(doc_id: str) -> str:
    return (CORPUS_DIR / internal_documents()[doc_id].path).read_text(encoding="utf-8")


def test_every_internal_document_is_cc0_access_controlled_and_titled() -> None:
    docs = internal_documents()
    assert len(docs) >= 10
    for doc in docs.values():
        assert doc.license == "CC0-1.0"
        assert doc.source_url is None
        assert doc.acl is not None, f"{doc.id} has no ACL"
        assert text(doc.id).startswith(f"# {doc.title}\n")
    assert "CC0 1.0" in (INTERNAL_DIR / "LICENSE").read_text(encoding="utf-8")


def test_internal_documents_cover_every_language() -> None:
    assert {doc.language for doc in internal_documents().values()} == set(Language)


@pytest.mark.parametrize(
    ("identifier", "doc_id"),
    [
        ("INV-2047", "internal-invoice-inv-2047"),
        ("INV-2048", "internal-invoice-inv-2048"),
        ("ORDER-123456", "internal-order-123456"),
        ("ORDER-123465", "internal-order-123465"),
        ("TKT-5521", "internal-ticket-tkt-5521"),
    ],
)
def test_each_identifier_appears_only_in_its_own_document(identifier: str, doc_id: str) -> None:
    # Near-identical ids (INV-2047 / INV-2048, ORDER-123456 / ORDER-123465) are the hard
    # negatives for exact-id lookups, so each id must be in exactly one document.
    containing = [d for d in internal_documents() if identifier in text(d)]
    assert containing == [doc_id]


def test_travel_policy_has_a_stale_version() -> None:
    docs = internal_documents()
    assert docs["internal-policy-travel-2025"].superseded_by == "internal-policy-travel-2026"
    assert "300" in text("internal-policy-travel-2025")
    assert "450" in text("internal-policy-travel-2026")


def test_acls_separate_the_callers() -> None:
    docs = internal_documents()
    furat = Caller(user="u-furat", groups=frozenset({"employees"}))
    fatima = Caller(user="u-fatima", groups=frozenset({"employees"}))
    finance = Caller(groups=frozenset({"employees", "finance"}))

    def readable(caller: Caller) -> set[str]:
        return {doc_id for doc_id, doc in docs.items() if caller.may_read(doc.acl)}

    policies = {doc_id for doc_id in docs if doc_id.startswith("internal-policy-")}
    assert readable(furat) == policies | {"internal-ticket-tkt-5521"}
    assert readable(fatima) == policies | {"internal-order-123456"}
    assert readable(finance) == policies | {
        "internal-invoice-inv-2047",
        "internal-invoice-inv-2048",
    }
    assert not any(
        caller.may_read(docs["internal-hr-salary-bands"].acl) for caller in (furat, fatima, finance)
    )
