# Golden cases

Questions the eval asks, with the documents that answer them and a reference answer. The
format is defined and validated in `evals/dataset.py`; the corpus is in `evals/corpus/`.

- `wikipedia.yaml`: questions over the public Wikipedia documents, asked anonymously.
- `internal.yaml`: questions over the synthetic internal documents, asked by an employee,
  a customer, or a member of the finance or IT support group. The callers are listed at the
  top of the file.

Version 0. Drafted from the corpus text; every case is reviewed by a human before it is
committed, and every reference answer is checked against its source document.

## Tags

- `paraphrase`: worded with little overlap with the source, so keyword search alone
  struggles.
- `exact-id`: looks up an identifier such as INV-2047 or ORDER-123456. The corpus has
  near-identical ids (INV-2048, ORDER-123465) as hard negatives, so dense search alone
  struggles.
- `morphology`: uses a different derivation of an Arabic root than the source (مؤلفات and
  ألّف, إدراج and أُدرجت). A comment on each case names both forms.
- `stale-doc`: an older version of the source is also indexed and gives a different answer
  (the 2024-11 revision of the 2034 World Cup article, version 1 of the travel policy).
- `multi-hop`: needs two documents.
- `personal`: a caller asking about their own data.
- `leakage`: the answer exists, but only in documents the caller may not read. Retrieval
  must return none of them, and the answer must be "I don't know".
- `unanswerable`: the corpus doesn't contain the answer.

Personal and leakage cases need permission-aware retrieval (Phase C). Until it exists the
eval reports them as not yet runnable rather than dropping them.

## Counts

Checked against the loader by `tests/evals/test_cases.py`. A case can have several tags.

| Tag | Cases |
|---|---|
| personal | 5 |
| leakage | 4 |
| multi-hop | 4 |
| stale-doc | 4 |
| unanswerable | 4 |
| paraphrase | 11 |
| exact-id | 13 |
| morphology | 10 |

| Language | Cases |
|---|---|
| ar | 25 |
| en | 19 |
| mixed | 7 |
| total | 51 |

## Adding a case

1. Find the answer in a document under `evals/corpus/` and copy the facts into
   `reference_answer`. Don't answer from memory.
2. List every document that answers it in `source_doc_ids`, not just the first one found.
3. Set `caller` to who is asking. Internal documents are not public.
4. Add tags only when they hold: a `paraphrase` question really avoids the source's
   wording, a `morphology` question really uses another form of the root.
5. Run `uv run pytest tests/evals` and update the counts above.
