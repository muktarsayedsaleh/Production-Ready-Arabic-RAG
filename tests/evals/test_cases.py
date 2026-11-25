"""Checks on the committed golden cases."""

from pathlib import Path

import pytest

from evals.dataset import Dataset, Expected, Tag, load_dataset
from evals.wikipedia import CORPUS_DIR

CASES_DIR = Path(__file__).parents[2] / "evals" / "cases"

# Minimum number of cases per tag, so a later edit can't quietly drop a category.
MINIMUM_PER_TAG = 3


@pytest.fixture(scope="module")
def dataset() -> Dataset:
    return load_dataset(CORPUS_DIR, CASES_DIR)


def counts_tables(dataset: Dataset) -> str:
    rows = ["| Tag | Cases |", "|---|---|"]
    rows += [f"| {tag} | {count} |" for tag, count in dataset.counts_by_tag().items()]
    rows += ["", "| Language | Cases |", "|---|---|"]
    rows += [f"| {lang} | {count} |" for lang, count in dataset.counts_by_language().items()]
    rows.append(f"| total | {len(dataset.cases)} |")
    return "\n".join(rows)


def test_readme_counts_match_the_cases(dataset: Dataset) -> None:
    readme = (CASES_DIR / "README.md").read_text(encoding="utf-8")
    expected = counts_tables(dataset)
    assert expected in readme, f"update the counts in evals/cases/README.md to:\n\n{expected}"


@pytest.mark.parametrize("tag", list(Tag))
def test_every_tag_has_cases(dataset: Dataset, tag: Tag) -> None:
    assert dataset.counts_by_tag()[tag] >= MINIMUM_PER_TAG


def test_every_language_has_cases(dataset: Dataset) -> None:
    assert all(count > 0 for count in dataset.counts_by_language().values())


def test_answers_and_abstentions_are_both_present(dataset: Dataset) -> None:
    expected = {case.expected for case in dataset.cases}
    assert expected == set(Expected)
