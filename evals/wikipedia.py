"""Fetch pinned Wikipedia revisions for the eval corpus.

The manifest (`evals/corpus/manifest.yaml`) is the source of truth. Every document under
`wikipedia/` names a revision id and a permalink; this script fetches that exact revision,
converts the rendered HTML to Markdown, writes it to the document's path, and regenerates
`wikipedia/ATTRIBUTION.md`.

    uv run python -m evals.wikipedia            # fetch every Wikipedia document
    uv run python -m evals.wikipedia --only ID  # fetch one

The fetched files are committed. Fetching again later can give slightly different text even
for the same revision, because Wikipedia renders old revisions with today's templates. The
eval always runs on the committed files; this script exists so they can be checked and
rebuilt, not to be run before every eval.

Conversion keeps headings, paragraphs and list items, and drops tables, infoboxes, images,
reference markers, navigation boxes, and the trailing sections (references, see also,
external links). ATTRIBUTION.md records this, as CC BY-SA requires changes to be indicated.
"""

import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path

from core.text.arabic.normalize import normalize
from evals.dataset import Document, load_corpus

CORPUS_DIR = Path(__file__).parent / "corpus"
WIKIPEDIA_DIR = "wikipedia"
ATTRIBUTION_FILE = "ATTRIBUTION.md"

# Wikimedia's API policy asks for a User-Agent that identifies the client and how to reach
# its maintainers.
USER_AGENT = (
    "Production-Ready-Arabic-RAG-corpus-fetch/0.1 "
    "(https://github.com/muktarsayedsaleh/Production-Ready-Arabic-RAG)"
)
# Seconds between API requests, to stay well below Wikimedia's rate limits.
REQUEST_INTERVAL = 1.0

_PERMALINK = re.compile(
    r"^https://(?P<lang>[a-z]+)\.wikipedia\.org/w/index\.php\?oldid=(?P<rev>\d+)$"
)

# Sections from which everything to the end of the article is dropped. Headings are compared
# after Arabic normalization, so spelling variants (أيضا, ايضًا, أيضاً) match one entry.
_STOP_SECTIONS = frozenset(
    normalize(heading).casefold()
    for heading in {
        "references",
        "notes",
        "see also",
        "external links",
        "further reading",
        "bibliography",
        "sources",
        "citations",
        "footnotes",
        "notes and references",
        "المراجع",
        "مراجع",
        "المصادر",
        "مصادر",
        "الهوامش",
        "هوامش",
        "ملاحظات",
        "وصلات خارجية",
        "روابط خارجية",
        "انظر أيضا",
        "قراءات إضافية",
        "قراءة إضافية",
        "ببليوغرافيا",
        "المراجع والمصادر",
    }
)

# Elements dropped together with everything inside them.
_SKIP_TAGS = frozenset(
    {"table", "style", "script", "figure", "img", "math", "sup", "noscript", "link", "meta"}
)
_SKIP_CLASSES = frozenset(
    {
        "mw-editsection",
        "reflist",
        "references",
        "navbox",
        "hatnote",
        "metadata",
        "ambox",
        "noprint",
        "thumb",
        "gallery",
        "mw-empty-elt",
        "toc",
        "shortdescription",
        "sistersitebox",
        "side-box",
        "mwe-math-element",
    }
)
_HEADINGS = {"h2": "##", "h3": "###", "h4": "####", "h5": "####", "h6": "####"}
_BLOCKS = frozenset({"p", "li", "dd", "dt", "blockquote", *_HEADINGS})
_VOID = frozenset({"br", "hr", "img", "link", "meta", "wbr", "input", "source", "col", "area"})


class _Converter(HTMLParser):
    """Rendered Wikipedia HTML to Markdown: headings, paragraphs and list items only."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[str] = []
        self._skip_depth = 0
        self._block: str | None = None
        self._text: list[str] = []
        self._stopped = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _VOID:
            if tag == "br" and not self._skip_depth:
                self._text.append(" ")
            return
        if self._skip_depth:
            self._skip_depth += 1
            return
        classes = set((dict(attrs).get("class") or "").split())
        if tag in _SKIP_TAGS or classes & _SKIP_CLASSES:
            self._skip_depth = 1
            return
        if tag in _BLOCKS:
            self._flush()
            self._block = tag

    def handle_endtag(self, tag: str) -> None:
        if tag in _VOID:
            return
        if self._skip_depth:
            self._skip_depth -= 1
            return
        if tag == self._block:
            self._flush()

    def handle_data(self, data: str) -> None:
        if not self._skip_depth and self._block is not None:
            self._text.append(data)

    def close(self) -> None:
        super().close()
        self._flush()

    def _flush(self) -> None:
        block, text = self._block, " ".join("".join(self._text).split())
        self._block, self._text = None, []
        if block is None or not text or self._stopped:
            return
        if block in _HEADINGS:
            if normalize(text).casefold() in _STOP_SECTIONS:
                self._stopped = True
                return
            self.blocks.append(f"{_HEADINGS[block]} {text}")
        elif block == "li":
            self.blocks.append(f"- {text}")
        else:
            self.blocks.append(text)


def html_to_markdown(html: str) -> str:
    converter = _Converter()
    converter.feed(html)
    converter.close()
    return "\n\n".join(_merge_lists(_drop_empty_sections(converter.blocks)))


def _heading_level(block: str) -> int | None:
    marks = block.split(" ", 1)[0]
    return len(marks) if marks and set(marks) == {"#"} else None


def _drop_empty_sections(blocks: list[str]) -> list[str]:
    """Drop headings whose section ended up empty (e.g. a gallery or a table section)."""
    kept: list[str] = []
    for block in reversed(blocks):
        level = _heading_level(block)
        if level is not None:
            following = _heading_level(kept[-1]) if kept else 0
            if following is not None and following <= level:
                continue
        kept.append(block)
    return kept[::-1]


def _merge_lists(blocks: list[str]) -> Iterator[str]:
    """Join consecutive list items into one block so a list stays together."""
    items: list[str] = []
    for block in blocks:
        if block.startswith("- "):
            items.append(block)
            continue
        if items:
            yield "\n".join(items)
            items = []
        yield block
    if items:
        yield "\n".join(items)


@dataclass(frozen=True)
class Revision:
    title: str
    timestamp: str
    html: str


def _api(lang: str, params: dict[str, str]) -> dict[str, object]:
    query = urllib.parse.urlencode({**params, "format": "json", "formatversion": "2"})
    request = urllib.request.Request(
        f"https://{lang}.wikipedia.org/w/api.php?{query}", headers={"User-Agent": USER_AGENT}
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        data: dict[str, object] = json.load(response)
    time.sleep(REQUEST_INTERVAL)
    if "error" in data:
        raise RuntimeError(f"{lang}.wikipedia.org API error: {data['error']}")
    return data


def fetch_revision(lang: str, revision: str) -> Revision:
    meta = _api(
        lang, {"action": "query", "prop": "revisions", "revids": revision, "rvprop": "timestamp"}
    )
    page = meta["query"]["pages"][0]  # type: ignore[index]
    parsed = _api(lang, {"action": "parse", "oldid": revision, "prop": "text"})
    return Revision(
        title=page["title"],
        timestamp=page["revisions"][0]["timestamp"],
        html=parsed["parse"]["text"],  # type: ignore[index]
    )


def permalink_parts(document: Document) -> tuple[str, str]:
    """The wiki language and revision id from a document's permalink."""
    match = _PERMALINK.match(document.source_url or "")
    if match is None or match["rev"] != document.revision:
        raise ValueError(
            f"document {document.id}: source_url must be the permalink "
            f"https://<lang>.wikipedia.org/w/index.php?oldid=<revision>"
        )
    return match["lang"], match["rev"]


def wikipedia_documents(documents: dict[str, Document]) -> list[Document]:
    return sorted(
        (doc for doc in documents.values() if doc.path.startswith(f"{WIKIPEDIA_DIR}/")),
        key=lambda doc: doc.id,
    )


def attribution(documents: list[Document]) -> str:
    lines = [
        "# Attribution",
        "",
        "The files in this directory are adapted from Wikipedia articles, at the revisions",
        "linked below. Each article's authors are listed in its page history.",
        "",
        "Changes: the rendered HTML of each revision was converted to Markdown. Headings,",
        "paragraphs and list items were kept. Tables, infoboxes, images, captions, reference",
        "markers, navigation boxes, and the trailing sections (references, notes, see also,",
        "external links) were removed.",
        "",
        "Licensed under CC BY-SA 4.0, see LICENSE in this directory.",
        "",
        "| File | Article | Revision | Date | History |",
        "|---|---|---|---|---|",
    ]
    for doc in documents:
        lang, revision = permalink_parts(doc)
        title = urllib.parse.quote(doc.title.replace(" ", "_"))
        history = f"https://{lang}.wikipedia.org/w/index.php?title={title}&action=history"
        lines.append(
            f"| {Path(doc.path).name} | {doc.title} ({lang}) | "
            f"[{revision}]({doc.source_url}) | {doc.version_date} | [history]({history}) |"
        )
    return "\n".join(lines) + "\n"


def fetch(corpus_dir: Path, only: str | None = None) -> None:
    documents = wikipedia_documents(load_corpus(corpus_dir, check_files=False))
    for doc in documents:
        if only is not None and doc.id != only:
            continue
        lang, revision = permalink_parts(doc)
        fetched = fetch_revision(lang, revision)
        if fetched.title != doc.title:
            raise ValueError(f"document {doc.id}: title is {fetched.title!r}, not {doc.title!r}")
        if fetched.timestamp[:10] != doc.version_date.isoformat():
            raise ValueError(
                f"document {doc.id}: revision date is {fetched.timestamp[:10]}, "
                f"manifest says {doc.version_date}"
            )
        text = f"# {doc.title}\n\n{html_to_markdown(fetched.html)}\n"
        (corpus_dir / doc.path).write_text(text, encoding="utf-8")
        print(f"{doc.id}: {len(text):,} characters", file=sys.stderr)
    (corpus_dir / WIKIPEDIA_DIR / ATTRIBUTION_FILE).write_text(
        attribution(documents), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--corpus", type=Path, default=CORPUS_DIR)
    parser.add_argument("--only", metavar="ID", help="fetch only this document id")
    args = parser.parse_args()
    fetch(args.corpus, args.only)


if __name__ == "__main__":
    main()
