"""Shared paragraph-level chunking of data/index.csv's item files.

Used by both RAG.step11_make_qa (to locate a gold passage for each question) and
RAG.step12_evaluate_retrievers (to build the retrieval corpus), so the two always
agree on what a "passage" is.

A chunk is one blank-line-separated block of an item file. That's also the unit
step10_build_pdfs.tableize() operates on: a financial-statement table's heading and
its rows have no blank lines between them, so a whole table section (e.g. "Assets"
plus its balance-sheet rows) falls out as one chunk -- a natural retrieval unit,
not an arbitrary fixed-size window.

The parsed real filings have no blank lines at all (one line per paragraph or table
row), so a real Item would otherwise be a single chunk of 10,000+ words. Those files
are instead packed into ~REAL_CHUNK_WORDS-word chunks of consecutive lines
(pack_lines). evals.compare_sets uses the same chunk_text, so the scorecard, the RAG
answers, step11/step12 and the chatbot all see the same passages.
"""

import csv
from dataclasses import dataclass
from pathlib import Path

from method_2.config import RAW_DIR


INDEX_PATH = RAW_DIR.parent / "index.csv"
REAL_CHUNK_WORDS = 80  # target chunk size when an item file has no blank lines (see pack_lines)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    ticker: str
    company_name: str
    fiscal_year: int
    item: str
    source: str  # "real" or "synthetic"
    path: str
    text: str


def chunk_text(text: str) -> list[str]:
    """Split on blank lines, dropping any empty fragments; a file with no blank lines
    (the parsed real filings) is packed line by line instead (pack_lines).
    """
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paras) == 1 and "\n" in paras[0]:
        paras = pack_lines([line.strip() for line in paras[0].splitlines() if line.strip()])
    return paras


def pack_lines(lines: list[str], target_words: int = REAL_CHUNK_WORDS) -> list[str]:
    """Join consecutive lines into chunks of about `target_words` words.

    A parsed real filing has one line per paragraph or table row (median 5 words), so
    single lines are too small to be a passage. Packing them gives chunks near the
    size of a long synthetic paragraph (synthetic Item 7 paragraphs: median 24 words,
    90th percentile 57). A line longer than the target stays a chunk of its own.
    """
    chunks, current, words = [], [], 0
    for line in lines:
        n = len(line.split())
        if current and words + n > target_words:
            chunks.append("\n".join(current))
            current, words = [], 0
        current.append(line)
        words += n
    if current:
        chunks.append("\n".join(current))
    return chunks


def load_chunks(
    index_path: Path = INDEX_PATH,
    *,
    sources: set[str] | None = None,
    tickers: set[str] | None = None,
) -> list[Chunk]:
    """Return every chunk from matching rows in data/index.csv, in file order.

    `sources`/`tickers` filter at the index-row level, before any file is read --
    the real corpus alone is ~730 rows including some 100,000+-word items, so
    filtering after reading every file (as an earlier version of this function did)
    cost minutes for callers that only need a handful of real companies (see
    RAG.step12_evaluate_retrievers's 6-company sample).
    """
    with open(index_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if sources is not None:
        rows = [r for r in rows if r["source"] in sources]
    if tickers is not None:
        rows = [r for r in rows if r["ticker"] in tickers]

    chunks = []
    for row in rows:
        path = RAW_DIR.parent / row["path"]
        text = path.read_text(encoding="utf-8")
        for i, para in enumerate(chunk_text(text)):
            chunk_id = f"{row['ticker']}_{row['fiscal_year']}_{row['item']}_{i}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    ticker=row["ticker"],
                    company_name=row["company_name"],
                    fiscal_year=int(row["fiscal_year"]),
                    item=row["item"],
                    source=row["source"],
                    path=row["path"],
                    text=para,
                ),
            )
    return chunks
