"""The one question bank every eval uses: fact questions plus analytical questions,
asked the same way of every set.

Every company gets the same questions, whichever method wrote its filing: ten fact
questions (QUESTIONS, a figure the statements report) and six analytical questions
(ANALYTICAL, a ratio that takes two figures, usually from different statements, e.g.
return on equity = net income / stockholders' equity). The answer to each comes from
the company's statements (financials.json for a synthetic company, XBRL headline
figures for a real one), never from the text, so a set that never states its revenue
still gets the revenue question and scores a miss on it. That keeps the denominator
fixed per company and makes sets comparable. evals.section_scorecard,
evals.rag_answers, evals.compare_sets and RAG.step11_make_qa (data/qa_pairs.csv for
step12, the dashboard and the chatbot) all build their questions here.

Per question this records the true value and the chunks that make it answerable (the
same figure matching as method_2.step09's numeric gate: dollars within 1%, percentages
within 0.25 points):
- a fact question: every chunk whose text states the value (gold_chunk_ids);
- an analytical question: per input figure, every chunk stating it
  (input_gold_chunk_ids, groups split by "|"); gold_chunk_ids is empty. It is
  answerable when every input is stated. A chunk "stating the ratio" isn't used: a
  percentage within 0.25 points of a ratio turns up by chance in most filings.
A question with nothing answerable is unanswerable from the corpus. hit() is the one
retrieval rule every caller uses: a fact question is hit when a stating chunk is in
the retrieved list; an analytical one when a chunk for every input is.

Scores, for a list of questions and the chunks they are searched over (score()):
- fact_coverage / analytical_coverage / qa_coverage: share of the fact, analytical
  and all questions that are answerable from the chunks.
- fact_recall_at_10 / analytical_recall_at_10 / qa_recall_at_10: share hit by BM25's
  top 10. Unanswerable questions count as misses, so recall can't exceed coverage.

And one score on the text itself, label-aware, with no chance matches:
- fact_correct_pct: of the figures the text attaches to one of the ten facts' labels
  ("revenues of $X", "net margin was Y%", "Total assets | Z" table rows), the share
  equal to that fact's current or prior-year value or their difference. A figure
  next to "revenue" must be the revenue line, not any number in the statements.

Run from implementations/synthetic_10k; used by evals.section_scorecard and
evals.compare_sets.
"""

import csv
import re
from collections.abc import Iterable
from pathlib import Path

from method_1.financials import DURATION_METRICS, INSTANT_METRICS
from method_2.step09_validate_filings import (
    PERCENT_RE,
    PROSE_DOLLAR_RE,
    UNIT_SCALE,
    closest_match,
    extract_figures,
    to_float,
)
from RAG.chunks import Chunk
from RAG.retrieval import Bm25Retriever


K = 10

# (fact key, question template, is_percent). Every fact exists in both methods'
# financials.json and in Method 1's XBRL extract of real filings.
QUESTIONS: list[tuple[str, str, bool]] = [
    ("revenue", "What were {name}'s total revenues for fiscal {fy}?", False),
    ("net_income", "What was {name}'s net income for fiscal {fy}?", False),
    ("operating_income", "What was {name}'s operating income for fiscal {fy}?", False),
    ("total_assets", "What were {name}'s total assets at the end of fiscal {fy}?", False),
    ("total_liabilities", "What were {name}'s total liabilities at the end of fiscal {fy}?", False),
    ("stockholders_equity", "What was {name}'s total stockholders' equity at the end of fiscal {fy}?", False),
    ("cash", "How much cash and cash equivalents did {name} hold at the end of fiscal {fy}?", False),
    ("operating_cash_flow", "How much cash did {name} generate from operating activities in fiscal {fy}?", False),
    ("net_margin", "What was {name}'s net margin for fiscal {fy}?", True),
    ("operating_margin", "What was {name}'s operating margin for fiscal {fy}?", True),
]

# (key, question template, numerator fact, denominator fact). All are percentages of
# the denominator; the template names the formula so the question is unambiguous.
ANALYTICAL: list[tuple[str, str, str, str]] = [
    (
        "return_on_equity",
        "What was {name}'s return on equity (net income as a percentage of total stockholders' equity) for fiscal {fy}?",
        "net_income",
        "stockholders_equity",
    ),
    (
        "return_on_assets",
        "What was {name}'s return on assets (net income as a percentage of total assets) for fiscal {fy}?",
        "net_income",
        "total_assets",
    ),
    (
        "liabilities_to_assets",
        "What share of {name}'s total assets were financed by liabilities (total liabilities as a percentage of total assets) at the end of fiscal {fy}?",
        "total_liabilities",
        "total_assets",
    ),
    (
        "cash_to_assets",
        "What percentage of {name}'s total assets was held in cash and cash equivalents at the end of fiscal {fy}?",
        "cash",
        "total_assets",
    ),
    (
        "operating_cash_flow_margin",
        "What was {name}'s operating cash flow margin (cash from operating activities as a percentage of total revenues) for fiscal {fy}?",
        "operating_cash_flow",
        "revenue",
    ),
    (
        "cash_conversion",
        "What was {name}'s cash conversion (cash from operating activities as a percentage of net income) for fiscal {fy}?",
        "operating_cash_flow",
        "net_income",
    ),
]
ANALYTICAL_TOLERANCE = 0.5  # percentage points (or 2% of the value, if larger) for a computed ratio

# Leaf names in a synthetic financials.json (either method) for each dollar fact.
SYNTHETIC_KEYS = {
    "revenue": ("income_statement", "revenues"),
    "net_income": ("income_statement", "net_income"),
    "operating_income": ("income_statement", "operating_income"),
    "total_assets": ("balance_sheet", "total_assets"),
    "total_liabilities": ("balance_sheet", "total_liabilities"),
    "stockholders_equity": ("balance_sheet", "total_stockholders_equity"),
    "cash": ("balance_sheet", "cash_and_equivalents"),
    "operating_cash_flow": ("cash_flow_statement", "cash_from_operations"),
}


# How each fact is labelled in prose or a table row. Matched case-insensitively.
LABELS = {
    "revenue": r"(?:total |net )?revenues?|net sales|total sales",
    "net_income": r"net income|net earnings|net loss",
    "operating_income": r"operating income|income from operations|operating loss",
    "total_assets": r"total assets",
    "total_liabilities": r"total liabilities(?! and)",
    "stockholders_equity": r"(?:total )?(?:stockholders|shareholders)['’]? equity",
    "cash": r"cash and cash equivalents",
    "operating_cash_flow": r"(?:net )?cash (?:provided by|generated (?:from|by)|from) operating activities|operating cash flows?",
    "net_margin": r"net (?:profit )?margin",
    "operating_margin": r"operating margin",
}
ANY_LABEL = re.compile(r"\b(?:" + "|".join(LABELS.values()) + r")\b", re.IGNORECASE)
FULL_DOLLARS = 100_000  # a bare table figure at least this big is in full dollars
# "$4,049,858,186": a full-dollar figure with no unit word (two or more thousands groups).
BARE_DOLLAR_RE = re.compile(
    r"\$\s?\(?(\d{1,3}(?:,\d{3}){2,}(?:\.\d+)?)\)?(?!\s*(?:thousand|million|billion))",
    re.IGNORECASE,
)
WINDOW = 120  # characters after a label in which its figure must appear
SHARE_OF = re.compile(r"\bor\s+[\d.]+%,?\s+of\b", re.IGNORECASE)
# A label followed by one of these names a slice of the fact ("revenues from customers
# outside the United States", "net income attributable to noncontrolling interests").
SUB_LINE_TAIL = re.compile(
    r"\s+(?:stream|base|from customers|outside|attributable|in the \w+ segment|by segment|from (?:our|the) \w+ segment)",
    re.IGNORECASE,
)
# Words that may precede a headline label; anything else ("product", "cost of",
# "segment") names a sub-line.
HEADLINE_LEAD = {
    "our",
    "the",
    "its",
    "total",
    "consolidated",
    "company's",
    "company’s",
    "fiscal",
    "annual",
    "reported",
    "while",
    "with",
    "as",
    "and",
    "where",
}
COMPARISON = re.compile(
    r"compar|prior|previous|last year|increase|decrease|grew|declin|rose|fell|\bfrom\b|\bversus\b|\bvs\b",
    re.IGNORECASE,
)


def find_leaf(d: dict, key: str) -> float | None:
    """Return the first numeric value stored under `key` anywhere in `d`, skipping prior-year blocks."""
    for k, v in d.items():
        if k == key and isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, dict) and k != "prior_year":
            found = find_leaf(v, key)
            if found is not None:
                return found
    return None


def with_margins(facts: dict[str, float | None]) -> dict[str, float | None]:
    """Add net and operating margin (percent of revenue) to a dict of dollar facts."""
    revenue = facts.get("revenue")
    for margin, numerator in (("net_margin", "net_income"), ("operating_margin", "operating_income")):
        value = facts.get(numerator)
        facts[margin] = value / revenue * 100 if revenue and value is not None else None
    return facts


def synthetic_facts(financials: dict) -> dict[str, float | None]:
    """The ten facts from a synthetic company's financials.json."""
    facts = {fact: find_leaf(financials.get(section, {}), leaf) for fact, (section, leaf) in SYNTHETIC_KEYS.items()}
    return with_margins(facts)


def load_xbrl_metrics(financials_csv: Path) -> dict[tuple[str, int], dict[str, float]]:
    """Every XBRL metric in Method 1's financials.csv, per (ticker, fiscal year); {} if
    the file is missing.
    """
    out: dict[tuple[str, int], dict[str, float]] = {}
    if not financials_csv.exists():
        return out
    with open(financials_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            metrics = {}
            for m in [*DURATION_METRICS, *INSTANT_METRICS]:
                try:
                    metrics[m] = float(row[m])
                except (KeyError, TypeError, ValueError):
                    continue
            out[(row["ticker"], int(row["fiscal_year"]))] = metrics
    return out


def real_facts(xbrl: dict[str, float]) -> dict[str, float | None]:
    """The ten facts from a real filing's XBRL headline figures (Method 1's financials.csv names)."""
    names = {"revenue": "revenue", "stockholders_equity": "stockholders_equity"}
    facts = {fact: xbrl.get(names.get(fact, fact)) for fact in SYNTHETIC_KEYS}
    return with_margins(facts)


def cell_readings(cell: str) -> list[float]:
    """A table cell's dollar readings: in thousands and in millions (tables state their
    unit in a header the row doesn't carry), plus as written for a figure big enough to
    be in full dollars (Method 2's "$59,990,031").
    """
    v = to_float(cell.strip("*$ "))
    if v is None:
        return []
    return [v * 1e3, v * 1e6] + ([v] if abs(v) >= FULL_DOLLARS else [])


def dollar_readings(text: str) -> list[float]:
    """Every dollar figure in `text`: prose with a unit word, bare full-dollar prose
    ("$4,049,858,186"), and table cells in " | ", markdown or tab-separated rows.
    """
    values = [v for v, _ in extract_figures(text)[0]]
    values += [v for m in BARE_DOLLAR_RE.finditer(text) if (v := to_float(m.group(1))) is not None]
    for line in text.splitlines():
        cells = table_cells(line)
        if cells:
            values += [r for cell in cells[1:] for r in cell_readings(cell)]
    return values


def stating_chunks(value: float, chunks: Iterable[Chunk], *, is_percent: bool) -> list[str]:
    """Ids of every chunk whose text states `value`."""
    ids = []
    for chunk in chunks:
        found = [v for v, _ in extract_figures(chunk.text)[1]] if is_percent else dollar_readings(chunk.text)
        if any(closest_match(v, {value}, relative=not is_percent) for v in found):
            ids.append(chunk.chunk_id)
    return ids


def analytical_value(facts: dict[str, float | None], numerator: str, denominator: str) -> float | None:
    """The ratio numerator / denominator * 100, or None when either is missing or the denominator
    isn't positive (a ratio over negative equity or a net loss means nothing).
    """
    top, bottom = facts.get(numerator), facts.get(denominator)
    if top is None or not bottom or bottom <= 0:
        return None
    return top / bottom * 100


def make_questions(
    company_id: str,
    name: str,
    fy: int,
    facts: dict[str, float | None],
    chunks: list[Chunk],
) -> list[dict]:
    """The fixed question bank for one company (fact, then analytical), with each
    answer's stating chunks.

    A fact the statements don't have (a bank's operating income, a real filing missing
    an XBRL tag) is skipped, and so is a ratio that needs it.
    """
    rows = []
    by_id = {c.chunk_id: c for c in chunks}
    gold_by_fact: dict[str, list[str]] = {}

    def row(style: str, fact: str, question: str, value: float, is_percent: bool, gold: list[str], inputs: str) -> dict:
        items = {by_id[g].item for g in gold} | {
            by_id[g].item for group in inputs.split("|") for g in group.split(";") if g
        }
        return {
            "company_id": company_id,
            "fiscal_year": fy,
            "style": style,
            "fact": fact,
            "question": question,
            "answer": round(value, 2),
            "is_percent": is_percent,
            "gold_chunk_ids": ";".join(gold),
            "input_gold_chunk_ids": inputs,
            "gold_items": ";".join(sorted(items)),
        }

    for fact, template, is_percent in QUESTIONS:
        value = facts.get(fact)
        if value is None:
            continue
        gold = stating_chunks(value, chunks, is_percent=is_percent)
        gold_by_fact[fact] = gold
        rows.append(row("factual", fact, template.format(name=name, fy=fy), value, is_percent, gold, ""))
    for key, template, numerator, denominator in ANALYTICAL:
        value = analytical_value(facts, numerator, denominator)
        if value is None:
            continue
        inputs = "|".join(";".join(gold_by_fact.get(f, [])) for f in (numerator, denominator))
        rows.append(row("analytical", key, template.format(name=name, fy=fy), value, True, [], inputs))
    return rows


def hit(q: dict, retrieved: Iterable[str] | None) -> bool:
    """True if `retrieved` (chunk ids; None = every chunk) answers `q`: a chunk stating
    the answer, or, for an analytical question, a chunk stating each input figure.
    """
    pool = None if retrieved is None else set(retrieved)

    def present(ids: str) -> bool:
        found = [g for g in ids.split(";") if g]
        return bool(found) if pool is None else any(g in pool for g in found)

    if present(q.get("gold_chunk_ids", "")):
        return True
    groups = q.get("input_gold_chunk_ids", "")
    return bool(groups) and all(present(group) for group in groups.split("|"))


def answer_ok(q: dict, value: float | None) -> bool:
    """True if `value` is the right answer to `q`: within the figure tolerance for a
    fact, within ANALYTICAL_TOLERANCE points (or 2%) for a computed ratio.
    """
    if value is None:
        return False
    truth = float(q["answer"])
    if q.get("style") == "analytical":
        return abs(value - truth) <= max(ANALYTICAL_TOLERANCE, abs(truth) * 0.02)
    return closest_match(value, {truth}, relative=not q["is_percent"])


def contextual(chunk: Chunk) -> str:
    """The chunk as indexed: prefixed with its company, year and Item, so a question
    naming the company can find that company's passage among look-alike ones.
    """
    return f"{chunk.company_name}, fiscal {chunk.fiscal_year}, Item {chunk.item}:\n{chunk.text}"


def score(questions: list[dict], chunks: list[Chunk], k: int = K) -> dict[str, float | int | None]:
    """Coverage and BM25 recall@k of `questions` over `chunks`, for the fact questions,
    the analytical ones and all of them (see the module docstring).
    """
    ids = {c.chunk_id for c in chunks}
    bm25 = Bm25Retriever([c.chunk_id for c in chunks], [contextual(c) for c in chunks]) if chunks else None
    out: dict[str, float | int | None] = {"n_questions": len(questions)}
    tallies = {"fact": [0, 0, 0], "analytical": [0, 0, 0], "qa": [0, 0, 0]}  # n, answerable, hit
    for q in questions:
        groups = ["qa", "fact" if q.get("style", "factual") == "factual" else "analytical"]
        ok = hit(q, ids)
        found = ok and bm25 is not None and hit(q, bm25.rank(q["question"])[:k])
        for g in groups:
            tallies[g][0] += 1
            tallies[g][1] += ok
            tallies[g][2] += found
    for g, (n, covered, hits) in tallies.items():
        if g != "qa":
            out[f"n_{g}_questions"] = n
        out[f"{g}_coverage"] = round(covered / n, 3) if n else None
        out[f"{g}_recall_at_{k}"] = round(hits / n, 3) if n else None
    return out


def first_percent(text: str) -> float | None:
    m = PERCENT_RE.search(text)
    return to_float(m.group(0)) if m else None


def first_dollar(text: str) -> float | None:
    m = PROSE_DOLLAR_RE.search(text)
    bare = BARE_DOLLAR_RE.search(text)
    if bare and (not m or bare.start() < m.start()):
        return to_float(bare.group(1))
    if not m:
        return None
    amount = to_float(m.group(0)[: -len(m.group(1))])
    return amount * UNIT_SCALE[m.group(1).lower()] if amount is not None else None


def headline_label(line: str, start: int) -> bool:
    """True if the label starting at `start` is the headline line, not a sub-line.

    "product revenues", "cost of revenues", "segment net income" and the like name a
    different figure, so the word before the label must be one of HEADLINE_LEAD (or a
    year, or nothing).
    """
    before = re.findall(r"[\w'’]+", line[max(0, start - 30) : start])
    return not before or before[-1].lower() in HEADLINE_LEAD or before[-1].isdigit()


def table_cells(line: str) -> list[str] | None:
    """Cells of a " | " (synthetic) or tab-separated (real) table row, "$" cells dropped."""
    sep = "|" if " | " in line or line.lstrip().startswith("|") else "\t" if "\t" in line else None
    if sep is None:
        return None
    cells = [c.strip().strip("*").strip() for c in line.split(sep)]
    return [c for c in cells if c not in ("", "$")]


def attached_figures(text: str, fact: str, *, is_percent: bool) -> list[tuple[list[float], str]]:
    """Figures the text attaches to `fact`'s headline label, each as (readings, context).

    Prose: the first figure of the right kind within WINDOW characters after the label
    in the same sentence. Tables: a row whose first cell is the label. A table figure
    has no unit, so it gets two readings (thousands and millions); prose has one.
    """
    label = re.compile(rf"\b(?:{LABELS[fact]})\b", re.IGNORECASE)
    found = []
    for line in text.splitlines():
        cells = table_cells(line)
        if cells is not None:
            percent_row = len(cells) > 2 and cells[2].startswith("%")
            if not is_percent and len(cells) > 1 and not percent_row and label.fullmatch(cells[0].strip("*: ")):
                # Every value column: segment tables put the total last, comparative
                # tables put prior years after the current one.
                readings = [r for cell in cells[1:] for r in cell_readings(cell)]
                if readings:
                    found.append((readings, line))
            continue
        for m in label.finditer(line):
            if not headline_label(line, m.start()):
                continue
            if SUB_LINE_TAIL.match(line, m.end()):
                continue
            sentence = re.split(r"(?<=[.;])\s", line[m.end() : m.end() + WINDOW], maxsplit=1)[0]
            # Stop at the next fact's label: a figure after it belongs to that fact.
            sentence = ANY_LABEL.split(sentence, maxsplit=1)[0]
            if SHARE_OF.search(sentence):  # "$X million, or 89%, of our total assets": a slice
                continue
            v = first_percent(sentence) if is_percent else first_dollar(sentence)
            if v is not None:
                found.append(([v], line[max(0, m.start() - 80) : m.end() + len(sentence)]))
    return found


def correctness(
    texts: Iterable[str],
    facts: dict[str, float | None],
    prior: dict[str, float | None],
    fy: int | None = None,
) -> tuple[int, int]:
    """(correct, attached) headline-label figures in `texts` for one company.

    A figure is correct if it equals the fact's current value, or, when the prior year
    is known, the prior value or the change. When the prior year is unknown (Method 2's
    one-year statements, real filings without a prior XBRL row) a figure that doesn't
    match the current value but sits in a comparison ("compared to", another year,
    "increased by") can't be checked, so it is left out rather than counted wrong.
    """
    correct = total = 0
    other_year = re.compile(rf"\b(?!{fy}\b)(?:19|20)\d\d\b") if fy else re.compile(r"\b(?:19|20)\d\d\b")
    for fact, _, is_percent in QUESTIONS:
        value = facts.get(fact)
        if value is None:
            continue
        allowed = {value}
        before = prior.get(fact)
        if before is not None:
            allowed |= {before, value - before}
        for text in texts:
            for readings, context in attached_figures(text, fact, is_percent=is_percent):
                ok = any(closest_match(v, allowed, relative=not is_percent) for v in readings)
                if not ok and before is None and (COMPARISON.search(context) or other_year.search(context)):
                    continue
                total += 1
                correct += ok
    return correct, total


def synthetic_prior_facts(financials: dict) -> dict[str, float | None]:
    """Prior-year facts from a two-year financials.json (Method 1), else all None."""
    prior_block = financials.get("cash_flow_statement", {}).get("prior_year", {})
    facts = {fact: prior_block.get(f"{leaf}_prior") for fact, (_, leaf) in SYNTHETIC_KEYS.items()}
    return with_margins({k: (float(v) if v is not None else None) for k, v in facts.items()})
