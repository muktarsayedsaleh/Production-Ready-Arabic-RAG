"""Golden dataset format and loader.

Two kinds of files:

- A documents manifest (`manifest.yaml` in the corpus directory) listing every document
  the eval indexes: id, title, language, file path, source, license, version date, and an
  optional ACL.
- Case files (`*.yaml` in the cases directory), each holding a list of golden cases:
  question, language, reference answer, the documents that answer it, tags, the caller
  identity the question is asked with, and the expected behaviour.

Relevance is recorded per document, not per chunk, so the golden set survives chunking
changes: a retrieved chunk counts as relevant when its document is in `source_doc_ids`.

`load_dataset` checks every cross-reference and every tag rule, and reports all problems
at once rather than stopping at the first.
"""

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError

MANIFEST_FILE = "manifest.yaml"

Identifier = Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9._-]*$")]
NonEmpty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class Language(StrEnum):
    AR = "ar"
    EN = "en"
    MIXED = "mixed"


class Tag(StrEnum):
    # Asked by a specific user about their own data.
    PERSONAL = "personal"
    # The caller must not see the forbidden documents; retrieval must return none of them.
    LEAKAGE = "leakage"
    # The answer needs two or more documents.
    MULTI_HOP = "multi-hop"
    # An older version of a source document is also in the corpus; answering from it is
    # faithful to the context but incorrect.
    STALE_DOC = "stale-doc"
    # The corpus doesn't contain the answer; the system must say it doesn't know.
    UNANSWERABLE = "unanswerable"
    # Worded with little lexical overlap with the source document.
    PARAPHRASE = "paraphrase"
    # Looks up an exact identifier (invoice number, order code, ticket id).
    EXACT_ID = "exact-id"
    # Uses a different derivation of an Arabic root than the source document.
    MORPHOLOGY = "morphology"


class Expected(StrEnum):
    # Answer from the source documents.
    ANSWER = "answer"
    # Say "I don't know": the answer isn't in anything the caller may see.
    ABSTAIN = "abstain"


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Acl(_Model):
    """Who may read a document. A document without an ACL is public."""

    users: frozenset[NonEmpty] = frozenset()
    groups: frozenset[NonEmpty] = frozenset()


class Caller(_Model):
    """The identity a question is asked with. The default is an anonymous caller."""

    user: NonEmpty | None = None
    groups: frozenset[NonEmpty] = frozenset()

    def may_read(self, acl: Acl | None) -> bool:
        if acl is None:
            return True
        return (self.user is not None and self.user in acl.users) or bool(self.groups & acl.groups)


class Document(_Model):
    id: Identifier
    title: NonEmpty
    language: Language
    # Relative to the corpus directory.
    path: NonEmpty
    source_url: NonEmpty | None = None
    # Source revision (e.g. a Wikipedia oldid), so the text can be fetched again.
    revision: NonEmpty | None = None
    # SPDX identifier, e.g. "CC-BY-SA-4.0" or "CC0-1.0".
    license: NonEmpty
    version_date: date
    acl: Acl | None = None
    # Set on an older version of a document: the id of the version that replaces it.
    superseded_by: Identifier | None = None


class Case(_Model):
    id: Identifier
    question: NonEmpty
    language: Language
    expected: Expected = Expected.ANSWER
    # Required when `expected` is "answer".
    reference_answer: NonEmpty | None = None
    source_doc_ids: tuple[Identifier, ...] = ()
    # Documents that must never be retrieved for this caller.
    forbidden_doc_ids: tuple[Identifier, ...] = ()
    tags: frozenset[Tag] = frozenset()
    caller: Caller = Field(default_factory=Caller)


class DatasetError(Exception):
    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        super().__init__("invalid eval dataset:\n" + "\n".join(f"  - {p}" for p in problems))


@dataclass(frozen=True)
class Dataset:
    documents: dict[str, Document]
    cases: list[Case]

    def counts_by_tag(self) -> dict[Tag, int]:
        counts = Counter(tag for case in self.cases for tag in case.tags)
        return {tag: counts[tag] for tag in Tag}

    def counts_by_language(self) -> dict[Language, int]:
        counts = Counter(case.language for case in self.cases)
        return {language: counts[language] for language in Language}


def load_dataset(corpus_dir: Path, cases_dir: Path) -> Dataset:
    problems: list[str] = []
    documents = _load_documents(corpus_dir, problems)
    cases = _load_cases(cases_dir, problems)
    problems += _check_documents(documents, corpus_dir)
    problems += _check_cases(cases, documents)
    if problems:
        raise DatasetError(problems)
    return Dataset(documents=documents, cases=cases)


def load_corpus(corpus_dir: Path, *, check_files: bool = True) -> dict[str, Document]:
    """Load and check the documents manifest alone, without any cases.

    With `check_files=False` the document files may be missing, for tools that create them.
    """
    problems: list[str] = []
    documents = _load_documents(corpus_dir, problems)
    problems += _check_documents(documents, corpus_dir, check_files=check_files)
    if problems:
        raise DatasetError(problems)
    return documents


def _read_yaml(path: Path, key: str, problems: list[str]) -> list[Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        problems.append(f"{path}: cannot read: {exc}")
        return []
    items = data.get(key) if isinstance(data, dict) else None
    if not isinstance(items, list):
        problems.append(f"{path}: expected a top-level '{key}' list")
        return []
    return items


def _parse[M: BaseModel](
    model: type[M], items: Iterable[Any], source: Path, problems: list[str]
) -> list[M]:
    parsed: list[M] = []
    for position, item in enumerate(items):
        try:
            parsed.append(model.model_validate(item))
        except ValidationError as exc:
            label = item.get("id", f"#{position}") if isinstance(item, dict) else f"#{position}"
            for error in exc.errors():
                field = ".".join(str(part) for part in error["loc"]) or "(root)"
                problems.append(f"{source}: {label}: {field}: {error['msg']}")
    return parsed


def _load_documents(corpus_dir: Path, problems: list[str]) -> dict[str, Document]:
    manifest = corpus_dir / MANIFEST_FILE
    documents: dict[str, Document] = {}
    for document in _parse(
        Document, _read_yaml(manifest, "documents", problems), manifest, problems
    ):
        if document.id in documents:
            problems.append(f"{manifest}: duplicate document id {document.id!r}")
        documents[document.id] = document
    return documents


def _load_cases(cases_dir: Path, problems: list[str]) -> list[Case]:
    files = sorted(cases_dir.glob("*.yaml"))
    if not files:
        problems.append(f"{cases_dir}: no case files (*.yaml)")
    cases: list[Case] = []
    seen: set[str] = set()
    for path in files:
        for case in _parse(Case, _read_yaml(path, "cases", problems), path, problems):
            if case.id in seen:
                problems.append(f"{path}: duplicate case id {case.id!r}")
            seen.add(case.id)
            cases.append(case)
    return cases


def _check_documents(
    documents: dict[str, Document], corpus_dir: Path, *, check_files: bool = True
) -> list[str]:
    problems = []
    root = corpus_dir.resolve()
    for doc in documents.values():
        file = (corpus_dir / doc.path).resolve()
        if not file.is_relative_to(root):
            problems.append(f"document {doc.id}: path {doc.path!r} is outside the corpus")
        elif check_files and not file.is_file():
            problems.append(f"document {doc.id}: file {doc.path!r} not found")
        if doc.superseded_by is not None:
            newer = documents.get(doc.superseded_by)
            if newer is None:
                problems.append(f"document {doc.id}: superseded_by unknown {doc.superseded_by!r}")
            elif newer.id == doc.id:
                problems.append(f"document {doc.id}: cannot supersede itself")
            elif newer.version_date <= doc.version_date:
                problems.append(
                    f"document {doc.id}: superseded_by {newer.id!r} is not newer "
                    f"({newer.version_date} <= {doc.version_date})"
                )
    return problems


def _check_cases(cases: list[Case], documents: dict[str, Document]) -> list[str]:
    problems = []
    superseding = {doc.superseded_by for doc in documents.values() if doc.superseded_by}
    for case in cases:
        where = f"case {case.id}"
        for doc_id in (*case.source_doc_ids, *case.forbidden_doc_ids):
            if doc_id not in documents:
                problems.append(f"{where}: unknown document {doc_id!r}")
        if set(case.source_doc_ids) & set(case.forbidden_doc_ids):
            problems.append(f"{where}: a document is both a source and forbidden")

        if case.expected is Expected.ANSWER:
            if not case.source_doc_ids:
                problems.append(f"{where}: expected 'answer' needs source_doc_ids")
            if case.reference_answer is None:
                problems.append(f"{where}: expected 'answer' needs a reference_answer")
        elif case.source_doc_ids:
            problems.append(f"{where}: expected 'abstain' must not have source_doc_ids")

        for doc_id in case.source_doc_ids:
            doc = documents.get(doc_id)
            if doc is not None and not case.caller.may_read(doc.acl):
                problems.append(f"{where}: caller may not read source document {doc_id!r}")
        for doc_id in case.forbidden_doc_ids:
            doc = documents.get(doc_id)
            if doc is not None and case.caller.may_read(doc.acl):
                problems.append(f"{where}: caller may read forbidden document {doc_id!r}")

        problems += [f"{where}: {p}" for p in _check_tags(case, superseding)]
    return problems


def _check_tags(case: Case, superseding: set[str]) -> list[str]:
    problems = []
    tags = case.tags
    if Tag.UNANSWERABLE in tags and case.expected is not Expected.ABSTAIN:
        problems.append("tag 'unanswerable' needs expected 'abstain'")
    if Tag.LEAKAGE in tags:
        if not case.forbidden_doc_ids:
            problems.append("tag 'leakage' needs forbidden_doc_ids")
    elif case.forbidden_doc_ids:
        problems.append("forbidden_doc_ids is only for cases tagged 'leakage'")
    if Tag.PERSONAL in tags and case.caller.user is None:
        problems.append("tag 'personal' needs a caller user")
    if Tag.MULTI_HOP in tags and len(case.source_doc_ids) < 2:
        problems.append("tag 'multi-hop' needs at least two source documents")
    if Tag.STALE_DOC in tags and not superseding & set(case.source_doc_ids):
        problems.append("tag 'stale-doc' needs a source document that supersedes an older one")
    return problems
