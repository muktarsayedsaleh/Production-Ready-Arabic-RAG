from collections.abc import Callable
from pathlib import Path

import pytest
import yaml

from evals import wikipedia
from evals.dataset import MANIFEST_FILE, Language, load_corpus
from evals.wikipedia import (
    ATTRIBUTION_FILE,
    CORPUS_DIR,
    Revision,
    attribution,
    html_to_markdown,
    permalink_parts,
    wikipedia_documents,
)


def test_keeps_headings_paragraphs_and_lists() -> None:
    html = """
    <p>Riyadh is the <b>capital</b> of Saudi Arabia.</p>
    <div class="mw-heading mw-heading2"><h2 id="History">History</h2>
      <span class="mw-editsection">[<a>edit</a>]</span></div>
    <p>First   paragraph<br>continues.</p>
    <h3>Districts</h3>
    <ul><li>Al Olaya</li><li>Al Malaz</li></ul>
    """
    assert html_to_markdown(html) == (
        "Riyadh is the capital of Saudi Arabia.\n\n"
        "## History\n\n"
        "First paragraph continues.\n\n"
        "### Districts\n\n"
        "- Al Olaya\n- Al Malaz"
    )


def test_drops_tables_references_and_navigation() -> None:
    html = """
    <div class="hatnote">For other uses, see Riyadh (disambiguation).</div>
    <table class="infobox"><tr><td><p>Population 7 million</p></td></tr></table>
    <p>Text<sup class="reference"><a>[1]</a></sup> here.</p>
    <style>.x { color: red }</style>
    <figure><img src="a.png"><figcaption><p>A caption</p></figcaption></figure>
    <div class="navbox"><ul><li>Other cities</li></ul></div>
    """
    assert html_to_markdown(html) == "Text here."


def test_nested_skipped_elements_do_not_end_the_skip_early() -> None:
    html = "<table><tr><td><table><tr><td>inner</td></tr></table><p>still</p></td></tr></table>"
    assert html_to_markdown(html + "<p>after</p>") == "after"


def test_decodes_entities() -> None:
    assert html_to_markdown("<p>Tom &amp; Jerry&nbsp;&#1593;</p>") == "Tom & Jerry ع"


@pytest.mark.parametrize("heading", ["References", "See also", "المراجع", "انظر أيضاً", "انظر ايضًا"])
def test_stops_at_trailing_sections(heading: str) -> None:
    html = f"<p>Body.</p><h2>{heading}</h2><p>Smith 2020.</p><h2>Later</h2><p>More.</p>"
    assert html_to_markdown(html) == "Body."


def test_drops_headings_of_empty_sections() -> None:
    html = """
    <h2>Gallery</h2><table><tr><td>images</td></tr></table>
    <h2>Bids</h2><h3>Saudi Arabia</h3><h3>Rejected</h3><ul><li>ASEAN</li></ul>
    <h2>Empty at the end</h2>
    """
    assert html_to_markdown(html) == "## Bids\n\n### Rejected\n\n- ASEAN"


def test_permalink_parts() -> None:
    corpus = load_corpus(CORPUS_DIR)
    assert permalink_parts(corpus["wiki-ar-riyadh"]) == ("ar", "78062326")


def test_permalink_must_match_the_revision() -> None:
    doc = load_corpus(CORPUS_DIR)["wiki-ar-riyadh"]
    with pytest.raises(ValueError, match="permalink"):
        permalink_parts(doc.model_copy(update={"revision": "1"}))
    with pytest.raises(ValueError, match="permalink"):
        permalink_parts(doc.model_copy(update={"source_url": "https://ar.wikipedia.org/wiki/x"}))


def _corpus_with_one_doc(tmp_path: Path, version_date: str = "2026-11-24") -> Path:
    manifest = {
        "documents": [
            {
                "id": "wiki-ar-riyadh",
                "title": "الرياض",
                "language": "ar",
                "path": "wikipedia/wiki-ar-riyadh.md",
                "source_url": "https://ar.wikipedia.org/w/index.php?oldid=42",
                "revision": "42",
                "license": "CC-BY-SA-4.0",
                "version_date": version_date,
            }
        ]
    }
    (tmp_path / "wikipedia").mkdir()
    (tmp_path / MANIFEST_FILE).write_text(yaml.safe_dump(manifest, allow_unicode=True))
    return tmp_path


def _fake_fetch(
    title: str = "الرياض", timestamp: str = "2026-11-24T20:31:22Z"
) -> tuple[Callable[[str, str], Revision], list[tuple[str, str]]]:
    calls: list[tuple[str, str]] = []

    def fetch_revision(lang: str, revision: str) -> Revision:
        calls.append((lang, revision))
        return Revision(title=title, timestamp=timestamp, html="<p>عاصمة السعودية.</p>")

    return fetch_revision, calls


def test_fetch_writes_the_document_and_the_attribution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = _corpus_with_one_doc(tmp_path)
    fake, calls = _fake_fetch()
    monkeypatch.setattr(wikipedia, "fetch_revision", fake)
    wikipedia.fetch(corpus)
    assert calls == [("ar", "42")]
    text = (corpus / "wikipedia" / "wiki-ar-riyadh.md").read_text(encoding="utf-8")
    assert text == "# الرياض\n\nعاصمة السعودية.\n"
    assert "oldid=42" in (corpus / "wikipedia" / ATTRIBUTION_FILE).read_text(encoding="utf-8")


def test_fetch_rejects_a_title_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    corpus = _corpus_with_one_doc(tmp_path)
    monkeypatch.setattr(wikipedia, "fetch_revision", _fake_fetch(title="جدة")[0])
    with pytest.raises(ValueError, match="title"):
        wikipedia.fetch(corpus)


def test_fetch_rejects_a_date_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    corpus = _corpus_with_one_doc(tmp_path, version_date="2026-01-01")
    monkeypatch.setattr(wikipedia, "fetch_revision", _fake_fetch()[0])
    with pytest.raises(ValueError, match="revision date"):
        wikipedia.fetch(corpus)


# The committed corpus.


def test_committed_corpus_is_valid() -> None:
    docs = wikipedia_documents(load_corpus(CORPUS_DIR))
    assert len(docs) >= 25
    for doc in docs:
        assert doc.license == "CC-BY-SA-4.0"
        permalink_parts(doc)
        assert (CORPUS_DIR / doc.path).read_text(encoding="utf-8").startswith(f"# {doc.title}\n")


def test_committed_corpus_covers_both_languages() -> None:
    languages = {doc.language for doc in wikipedia_documents(load_corpus(CORPUS_DIR))}
    assert {Language.AR, Language.EN} <= languages


def test_committed_corpus_has_a_stale_revision_pair() -> None:
    corpus = load_corpus(CORPUS_DIR)
    old = corpus["wiki-ar-world-cup-2034-2024-11"]
    assert old.superseded_by == "wiki-ar-world-cup-2034"
    assert corpus[old.superseded_by].title == old.title


def test_committed_attribution_is_up_to_date() -> None:
    expected = attribution(wikipedia_documents(load_corpus(CORPUS_DIR)))
    actual = (CORPUS_DIR / wikipedia.WIKIPEDIA_DIR / ATTRIBUTION_FILE).read_text(encoding="utf-8")
    assert actual == expected


def test_committed_corpus_has_its_license() -> None:
    text = (CORPUS_DIR / wikipedia.WIKIPEDIA_DIR / "LICENSE").read_text(encoding="utf-8")
    assert text.startswith("Attribution-ShareAlike 4.0 International")
