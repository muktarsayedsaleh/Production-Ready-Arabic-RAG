from pathlib import Path
from typing import Any

import pytest
import yaml

from evals.dataset import (
    MANIFEST_FILE,
    Acl,
    Caller,
    DatasetError,
    Expected,
    Language,
    Tag,
    load_corpus,
    load_dataset,
)

PUBLIC_DOC: dict[str, Any] = {
    "id": "wiki-ar-riyadh",
    "title": "الرياض",
    "language": "ar",
    "path": "wikipedia/riyadh.md",
    "source_url": "https://ar.wikipedia.org/w/index.php?oldid=1",
    "revision": "1",
    "license": "CC-BY-SA-4.0",
    "version_date": "2025-01-01",
}

FINANCE_DOC: dict[str, Any] = {
    "id": "internal-inv-2047",
    "title": "Invoice INV-2047",
    "language": "mixed",
    "path": "internal/inv-2047.md",
    "license": "CC0-1.0",
    "version_date": "2025-03-01",
    "acl": {"groups": ["finance"]},
}

ANSWER_CASE: dict[str, Any] = {
    "id": "ar-riyadh-capital",
    "question": "ما عاصمة السعودية؟",
    "language": "ar",
    "reference_answer": "الرياض",
    "source_doc_ids": ["wiki-ar-riyadh"],
    "tags": ["paraphrase"],
}


def write_dataset(
    tmp_path: Path,
    documents: list[dict[str, Any]],
    cases: list[dict[str, Any]] | dict[str, list[dict[str, Any]]],
    *,
    create_files: bool = True,
) -> tuple[Path, Path]:
    corpus = tmp_path / "corpus"
    cases_dir = tmp_path / "cases"
    corpus.mkdir()
    cases_dir.mkdir()
    (corpus / MANIFEST_FILE).write_text(
        yaml.safe_dump({"documents": documents}, allow_unicode=True), encoding="utf-8"
    )
    if create_files:
        for doc in documents:
            file = corpus / doc["path"]
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text("text", encoding="utf-8")
    files = cases if isinstance(cases, dict) else {"cases.yaml": cases}
    for name, items in files.items():
        (cases_dir / name).write_text(
            yaml.safe_dump({"cases": items}, allow_unicode=True), encoding="utf-8"
        )
    return corpus, cases_dir


def problems_for(
    tmp_path: Path,
    documents: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    **kwargs: Any,
) -> list[str]:
    with pytest.raises(DatasetError) as info:
        load_dataset(*write_dataset(tmp_path, documents, cases, **kwargs))
    return info.value.problems


def case(**overrides: Any) -> dict[str, Any]:
    return {**ANSWER_CASE, **overrides}


# --- Loading


def test_loads_documents_and_cases(tmp_path: Path) -> None:
    dataset = load_dataset(*write_dataset(tmp_path, [PUBLIC_DOC], [ANSWER_CASE]))

    doc = dataset.documents["wiki-ar-riyadh"]
    assert doc.language is Language.AR
    assert doc.acl is None
    [loaded] = dataset.cases
    assert loaded.question == "ما عاصمة السعودية؟"
    assert loaded.expected is Expected.ANSWER
    assert loaded.tags == {Tag.PARAPHRASE}
    assert loaded.caller == Caller()


def test_loads_cases_from_every_yaml_file_in_order(tmp_path: Path) -> None:
    files = {
        "b.yaml": [case(id="second")],
        "a.yaml": [case(id="first")],
        "notes.txt": [],
    }
    corpus, cases_dir = write_dataset(tmp_path, [PUBLIC_DOC], files)

    dataset = load_dataset(corpus, cases_dir)

    assert [c.id for c in dataset.cases] == ["first", "second"]


def test_counts_by_tag_and_language_include_zeros(tmp_path: Path) -> None:
    cases = [
        case(id="a", tags=["paraphrase", "morphology"]),
        case(id="b", tags=["paraphrase"], language="en"),
    ]
    dataset = load_dataset(*write_dataset(tmp_path, [PUBLIC_DOC], cases))

    by_tag = dataset.counts_by_tag()
    assert by_tag[Tag.PARAPHRASE] == 2
    assert by_tag[Tag.MORPHOLOGY] == 1
    assert by_tag[Tag.LEAKAGE] == 0
    assert set(by_tag) == set(Tag)
    assert dataset.counts_by_language() == {Language.AR: 1, Language.EN: 1, Language.MIXED: 0}


# --- File and schema problems


def test_reports_missing_manifest_and_cases(tmp_path: Path) -> None:
    with pytest.raises(DatasetError) as info:
        load_dataset(tmp_path / "nope", tmp_path / "also-nope")

    text = "\n".join(info.value.problems)
    assert "cannot read" in text
    assert "no case files" in text


def test_reports_wrong_top_level_shape(tmp_path: Path) -> None:
    corpus, cases_dir = write_dataset(tmp_path, [PUBLIC_DOC], [ANSWER_CASE])
    (cases_dir / "cases.yaml").write_text("- just a list\n", encoding="utf-8")

    with pytest.raises(DatasetError, match="top-level 'cases' list"):
        load_dataset(corpus, cases_dir)


@pytest.mark.parametrize(
    ("overrides", "field"),
    [
        ({"language": "fr"}, "language"),
        ({"tags": ["made-up"]}, "tags"),
        ({"question": "   "}, "question"),
        ({"id": "Has Spaces"}, "id"),
        ({"expected": "maybe"}, "expected"),
        ({"surprise": 1}, "surprise"),
    ],
)
def test_schema_errors_name_the_case_and_field(
    tmp_path: Path, overrides: dict[str, Any], field: str
) -> None:
    problems = problems_for(tmp_path, [PUBLIC_DOC], [case(**overrides)])

    label = overrides.get("id", ANSWER_CASE["id"])
    assert any(str(label) in p and field in p for p in problems), problems


def test_reports_every_problem_not_just_the_first(tmp_path: Path) -> None:
    cases = [case(id="a", language="fr"), case(id="b", source_doc_ids=["missing"])]

    problems = problems_for(tmp_path, [PUBLIC_DOC], cases)

    assert len(problems) >= 2


def test_duplicate_ids(tmp_path: Path) -> None:
    problems = problems_for(tmp_path, [PUBLIC_DOC, PUBLIC_DOC], [ANSWER_CASE, ANSWER_CASE])

    assert any("duplicate document id" in p for p in problems)
    assert any("duplicate case id" in p for p in problems)


def test_missing_document_file(tmp_path: Path) -> None:
    problems = problems_for(tmp_path, [PUBLIC_DOC], [ANSWER_CASE], create_files=False)

    assert any("not found" in p for p in problems)


def test_document_path_outside_corpus(tmp_path: Path) -> None:
    doc = {**PUBLIC_DOC, "path": "../outside.md"}
    (tmp_path / "outside.md").write_text("x", encoding="utf-8")

    problems = problems_for(tmp_path, [doc], [ANSWER_CASE], create_files=False)

    assert any("outside the corpus" in p for p in problems)


# --- Document rules


def test_superseded_by_must_point_to_a_newer_known_document(tmp_path: Path) -> None:
    old = {**PUBLIC_DOC, "id": "old", "path": "old.md", "superseded_by": "unknown"}
    same_day = {**PUBLIC_DOC, "id": "older", "path": "older.md", "superseded_by": "wiki-ar-riyadh"}

    problems = problems_for(tmp_path, [PUBLIC_DOC, old, same_day], [ANSWER_CASE])

    assert any("superseded_by unknown" in p for p in problems)
    assert any("is not newer" in p for p in problems)


# --- Case rules


def test_unknown_source_document(tmp_path: Path) -> None:
    problems = problems_for(tmp_path, [PUBLIC_DOC], [case(source_doc_ids=["nope"])])

    assert any("unknown document 'nope'" in p for p in problems)


def test_answer_case_needs_sources_and_reference(tmp_path: Path) -> None:
    problems = problems_for(
        tmp_path, [PUBLIC_DOC], [case(source_doc_ids=[], reference_answer=None)]
    )

    assert any("needs source_doc_ids" in p for p in problems)
    assert any("needs a reference_answer" in p for p in problems)


def test_abstain_case_must_not_list_sources(tmp_path: Path) -> None:
    problems = problems_for(tmp_path, [PUBLIC_DOC], [case(expected="abstain")])

    assert any("must not have source_doc_ids" in p for p in problems)


def test_unanswerable_case_loads(tmp_path: Path) -> None:
    unanswerable = case(
        id="unanswerable-1",
        expected="abstain",
        reference_answer=None,
        source_doc_ids=[],
        tags=["unanswerable"],
    )

    dataset = load_dataset(*write_dataset(tmp_path, [PUBLIC_DOC], [unanswerable]))

    assert dataset.cases[0].expected is Expected.ABSTAIN


def test_unanswerable_tag_needs_abstain(tmp_path: Path) -> None:
    problems = problems_for(tmp_path, [PUBLIC_DOC], [case(tags=["unanswerable"])])

    assert any("needs expected 'abstain'" in p for p in problems)


def test_caller_must_be_able_to_read_source_documents(tmp_path: Path) -> None:
    problems = problems_for(
        tmp_path, [PUBLIC_DOC, FINANCE_DOC], [case(source_doc_ids=["internal-inv-2047"])]
    )

    assert any("may not read source document" in p for p in problems)


def test_valid_leakage_case(tmp_path: Path) -> None:
    leakage = case(
        id="leak-1",
        question="What is the total of invoice INV-2047?",
        language="en",
        expected="abstain",
        reference_answer=None,
        source_doc_ids=[],
        forbidden_doc_ids=["internal-inv-2047"],
        tags=["leakage", "exact-id"],
        caller={"user": "sara", "groups": ["support"]},
    )

    dataset = load_dataset(*write_dataset(tmp_path, [PUBLIC_DOC, FINANCE_DOC], [leakage]))

    assert dataset.cases[0].forbidden_doc_ids == ("internal-inv-2047",)


def test_leakage_rules(tmp_path: Path) -> None:
    readable = case(
        id="readable",
        expected="abstain",
        source_doc_ids=[],
        forbidden_doc_ids=["internal-inv-2047"],
        tags=["leakage"],
        caller={"groups": ["finance"]},
    )
    no_forbidden = case(id="no-forbidden", tags=["leakage"])
    untagged = case(id="untagged", forbidden_doc_ids=["internal-inv-2047"])

    problems = problems_for(tmp_path, [PUBLIC_DOC, FINANCE_DOC], [readable, no_forbidden, untagged])

    assert any("readable" in p and "may read forbidden document" in p for p in problems)
    assert any("no-forbidden" in p and "needs forbidden_doc_ids" in p for p in problems)
    assert any("untagged" in p and "only for cases tagged 'leakage'" in p for p in problems)


def test_source_cannot_also_be_forbidden(tmp_path: Path) -> None:
    both = case(forbidden_doc_ids=["wiki-ar-riyadh"], tags=["leakage"])

    problems = problems_for(tmp_path, [PUBLIC_DOC], [both])

    assert any("both a source and forbidden" in p for p in problems)


def test_personal_case_needs_a_user(tmp_path: Path) -> None:
    problems = problems_for(tmp_path, [PUBLIC_DOC], [case(tags=["personal"])])

    assert any("needs a caller user" in p for p in problems)


def test_multi_hop_needs_two_sources(tmp_path: Path) -> None:
    problems = problems_for(tmp_path, [PUBLIC_DOC], [case(tags=["multi-hop"])])

    assert any("at least two source documents" in p for p in problems)


def test_stale_doc_case_needs_a_superseding_source(tmp_path: Path) -> None:
    old = {
        **PUBLIC_DOC,
        "id": "wiki-ar-riyadh-2020",
        "path": "old.md",
        "version_date": "2020-01-01",
        "superseded_by": "wiki-ar-riyadh",
    }
    good = case(id="good", tags=["stale-doc"])
    bad = case(id="bad", tags=["stale-doc"], source_doc_ids=["wiki-ar-riyadh-2020"])

    problems = problems_for(tmp_path, [PUBLIC_DOC, old], [good, bad])

    assert not any("good" in p for p in problems)
    assert any("bad" in p and "supersedes an older one" in p for p in problems)


# --- ACL semantics


@pytest.mark.parametrize(
    ("caller", "acl", "allowed"),
    [
        (Caller(), None, True),
        (Caller(), Acl(groups=frozenset({"finance"})), False),
        (Caller(user="sara"), Acl(users=frozenset({"sara"})), True),
        (Caller(user="omar"), Acl(users=frozenset({"sara"})), False),
        (Caller(groups=frozenset({"finance", "hr"})), Acl(groups=frozenset({"finance"})), True),
        (Caller(groups=frozenset({"support"})), Acl(groups=frozenset({"finance"})), False),
        # An empty ACL means nobody, not everybody.
        (Caller(user="sara", groups=frozenset({"finance"})), Acl(), False),
    ],
)
def test_may_read(caller: Caller, acl: Acl | None, allowed: bool) -> None:
    assert caller.may_read(acl) is allowed


def test_load_corpus_reads_the_manifest_without_cases(tmp_path: Path) -> None:
    corpus, _ = write_dataset(tmp_path, [PUBLIC_DOC], [])
    assert list(load_corpus(corpus)) == ["wiki-ar-riyadh"]


def test_load_corpus_can_skip_the_file_check(tmp_path: Path) -> None:
    corpus, _ = write_dataset(tmp_path, [PUBLIC_DOC], [], create_files=False)
    with pytest.raises(DatasetError, match="not found"):
        load_corpus(corpus)
    assert list(load_corpus(corpus, check_files=False)) == ["wiki-ar-riyadh"]


def test_load_corpus_still_rejects_paths_outside_the_corpus(tmp_path: Path) -> None:
    corpus, _ = write_dataset(
        tmp_path, [{**PUBLIC_DOC, "path": "../escape.md"}], [], create_files=False
    )
    with pytest.raises(DatasetError, match="outside the corpus"):
        load_corpus(corpus, check_files=False)
