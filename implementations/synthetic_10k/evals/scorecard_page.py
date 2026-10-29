r"""Build the dashboard's scorecard page from the scorecard and the RAG answers.

Reads section_scorecard's scorecard.csv (an "all" row per set for the whole filing,
plus one row per set and Item) and, if given, rag_answers' summary.csv files, and
writes a static page with dashboard/scorecard.css inlined. Retrieval and RAG answers
are reported once per set over the whole filing; the per-section tables keep the
text metrics. The best synthetic set per metric is shaded with the scorecard's own
rule (evals.section_scorecard.pick), and every metric is defined at the bottom.

Run from implementations/synthetic_10k:
    uv run python -m evals.scorecard_page --scorecard evals/results/day3_scorecard/scorecard.csv \
        --rag bm25=evals/results/day3_rag_bm25/summary.csv --rag hybrid=evals/results/day3_rag/summary.csv
"""

import argparse
import contextlib
import csv
import html
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from evals.section_scorecard import ALL, ITEM_LABELS, REAL, WEIGHTS, WHOLE_WEIGHTS, pick  # noqa: E402


DEFAULT_OUT = ROOT / "dashboard" / "scorecard.html"
CSS = ROOT / "dashboard" / "scorecard.css"
FINDINGS = HERE / "scorecard_findings.html"  # optional hand-written <li> items

SET_LABELS = {
    "m1_gemini": ("Method 1", "Gemini 3 Flash"),
    "m2_gemini": ("Method 2", "Gemini 3 Flash"),
    "m1_gpt4o": ("Method 1", "GPT-4o"),
    "m2_gpt4o": ("Method 2", "GPT-4o"),
    "m1_opus": ("Method 1", "Claude Opus 5.5"),
    "m2_opus": ("Method 2", "Claude Opus 5.5"),
    REAL: ("Real 10-Ks", "20-filing sample"),
}
DIRECTIONS = {
    "high": "higher is better",
    "low": "lower is better",
    "real": "closer to real",
    "one": "closer to 1",
    "le_real": "at or below real",
    "": "shown, not ranked",
}
SITENAV = """<nav class="sitenav" aria-label="Dashboard pages">
  <a href="index.html">Results dashboard</a>
  <a href="scorecard.html" aria-current="page">Scorecard</a>
  <a href="financial_method_comparison.html">Financial method comparison</a>
  <a href="chatbot.html">Chatbot</a>
</nav>"""

# Whole-filing retrieval columns (BM25, no LLM): metric, header.
RETRIEVAL_COLUMNS = [
    ("n_questions", "Questions"),
    ("fact_coverage", "Facts answerable"),
    ("analytical_coverage", "Analytical answerable"),
    ("fact_recall_at_10", "Fact recall@10"),
    ("analytical_recall_at_10", "Analytical recall@10"),
    ("qa_recall_at_10", "All questions recall@10"),
    ("self_retrieval_mrr", "Self-retrieval MRR"),
    ("fact_correct_pct", "Numbers correct %"),
]
# RAG answer columns per run: metric, header, direction.
RAG_COLUMNS = [
    ("rag_answer_correct_pct", "Correct %, all", "high"),
    ("rag_fact_correct_pct", "Correct %, fact", "high"),
    ("rag_analytical_correct_pct", "Correct %, analytical", "high"),
    ("retrieval_hit_pct", "Retrieval hit %", "high"),
    ("rag_abstain_pct", "Abstained %", "low"),
]
# Rows of each Item's table: group, metric, label.
ITEM_METRICS = [
    ("Numbers", "fact_correct_pct", "Numbers correct %"),
    ("Numbers", "n_attached", "Labelled figures"),
    ("Realism", "length_ratio", "Length vs real"),
    ("Realism", "vocab_jaccard", "Vocabulary overlap"),
    ("Realism", "figures_per_1k", "Figures per 1k words"),
    ("Diversity and privacy", "cross_doc_overlap", "Repetition across filings"),
    ("Diversity and privacy", "copied_from_real", "Copied from real"),
    ("Shown, not ranked", "traced_pct", "Figures traced %"),
    ("Shown, not ranked", "median_words", "Median words"),
]
# Metric definitions: group, label, direction key, definition.
DEFINITIONS = [
    (
        "Question bank",
        "Questions",
        "",
        "Every company gets the same questions, whichever method wrote its filing: ten fact questions (total revenues, "
        "net income, operating income, total assets, total liabilities, stockholders' equity, cash and cash "
        "equivalents, operating cash flow, net margin, operating margin) and six analytical ones that combine two "
        "figures (return on equity, return on assets, liabilities as a share of assets, cash as a share of assets, "
        "operating cash flow as a share of revenue and of net income). The answers come from the company's "
        "statements (financials.json, or XBRL for real filings), never from the text, and are written by "
        "qa/fact_questions.py, not by a model. A fact a company doesn't have (a bank's operating income) drops that "
        "question and the ratios that need it.",
    ),
    (
        "Retrieval, whole filing (BM25, no LLM)",
        "Facts answerable",
        "",
        "Share of fact questions whose answer is stated somewhere in the filing: a passage states the value, dollars "
        "within 1%, percentages within 0.25 points. A question whose answer isn't stated can never be answered, so "
        "this caps every retrieval and RAG score.",
    ),
    (
        "Retrieval, whole filing (BM25, no LLM)",
        "Analytical answerable",
        "",
        "Share of analytical questions the filing makes answerable: both input figures are stated (each in some "
        "passage). A passage giving the ratio itself isn't counted: a percentage that close turns up by chance.",
    ),
    (
        "Retrieval, whole filing (BM25, no LLM)",
        "Fact / Analytical / All questions recall@10",
        "high",
        "BM25 keyword search over every passage of the set (each indexed with its company, fiscal year and Item), "
        "the question as the query. A question is hit when the top 10 passages contain its answer: a passage stating "
        "the value, or for an analytical question a passage for each input figure. Every question counts, so "
        "unanswerable ones are misses and recall can't exceed answerable.",
    ),
    (
        "Retrieval, whole filing (BM25, no LLM)",
        "Self-retrieval MRR",
        "real",
        "A sentence from a passage, with the company's name removed, is the query; the score is the mean reciprocal "
        "rank of finding that same passage among all the set's passages. Low means passages repeat across "
        "companies; the real sample sets the target.",
    ),
    (
        "RAG answers, whole filing (retrieval + LLM)",
        "Correct %, all / fact / analytical",
        "high",
        "Each question goes through the RAG pipeline: retrieval returns the top 10 passages, an LLM "
        "(gemini-3.1-flash-lite-preview, the same for every set) answers from them only, computing a ratio when the "
        "question asks for one, and the number is checked against the statements (dollars within 1%, percentages "
        "within 0.25 points, computed ratios within 0.5 points or 2%). The end-to-end score: does a RAG system get "
        "the right answer from this corpus?",
    ),
    (
        "RAG answers, whole filing (retrieval + LLM)",
        "Retrieval hit %",
        "high",
        "Share of questions whose answer is in the 10 passages the LLM saw (the same hit rule as recall@10). When it "
        "is, the LLM is almost always right, so correct % tracks this closely.",
    ),
    (
        "RAG answers, whole filing (retrieval + LLM)",
        "Abstained %",
        "low",
        "Share of questions where the LLM said the passages don't give the answer instead of guessing. Correct + "
        "abstained + wrong = 100%, so a high abstain rate means retrieval missed, not that the model made things up.",
    ),
    (
        "RAG answers, whole filing (retrieval + LLM)",
        "bm25 / hybrid",
        "",
        "The retriever of each RAG run: BM25 keyword search alone, or BM25 fused with a semantic embedding retriever "
        "(potion-retrieval-32M) by reciprocal rank fusion.",
    ),
    (
        "Numbers",
        "Numbers correct %",
        "high",
        'Of the figures the text attaches to one of the ten facts\' labels ("net income of $X", a "Total assets" '
        "table row), the share equal to that fact's value in the statements (or its prior-year value or change). "
        'Sub-lines like "product revenues" or segment rows don\'t count. A hallucination check on the text itself; '
        "not scored under 5 labelled figures. For real filings it is a floor: real text rounds and cites figures "
        "the XBRL extract can't verify.",
    ),
    ("Numbers", "Labelled figures", "", "How many labelled figures the numbers-correct check covered."),
    (
        "Realism",
        "Length vs real",
        "one",
        "The section's median word count over the real sample's median for the same section.",
    ),
    (
        "Realism",
        "Vocabulary overlap",
        "high",
        "Jaccard overlap of the section's 1,000 most frequent terms with the real sections' 1,000 most frequent terms.",
    ),
    ("Realism", "Figures per 1k words", "real", "Dollar and percent figures per 1,000 words (median per filing)."),
    (
        "Diversity and privacy",
        "Repetition across filings",
        "le_real",
        "Median share of a filing's 8-word sequences that also appear in the same section of another filing in the "
        "set. High means the generator reuses the same sentences across companies.",
    ),
    (
        "Diversity and privacy",
        "Copied from real",
        "le_real",
        "Median share of a filing's 8-word sequences found in the same section of the real corpus. The real column is "
        "a held-out real filing against the others: the legal boilerplate real filings share.",
    ),
    (
        "Shown, not ranked",
        "Figures traced %",
        "",
        "Share of all figures within tolerance of any value derivable from the statements. Not ranked: about a "
        "quarter of random numbers would pass, and it ignores what a figure is labelled as.",
    ),
    (
        "Ranking",
        "Suggested set / shading",
        "",
        "Per metric, each set earns points by rank (weight × number of sets it beats); the suggested set per section "
        "has the most points over the text metrics, with numbers correct counting twice. Shaded = best synthetic set "
        "on that metric.",
    ),
]


def read_csv(path: Path) -> list[dict]:
    """Rows of a CSV, numbers as floats where they parse."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for k, v in row.items():
                if k in ("set", "item"):
                    continue
                if v in ("", None):
                    row[k] = None
                else:
                    with contextlib.suppress(ValueError):
                        row[k] = float(v)
            rows.append(row)
    return rows


def cell(v) -> str:
    """Format a number for a table cell."""
    if v is None:
        return '<span class="na">not scored</span>'
    return f"{v:,.0f}" if float(v).is_integer() and abs(v) >= 10 else f"{v:g}"


def set_head(name: str) -> str:
    """Column header for a set."""
    method, model = SET_LABELS.get(name, (name, ""))
    cls = ' class="num real-col"' if name == REAL else ' class="num"'
    return f'<th{cls}><span class="sh-model">{html.escape(model)}</span><span class="sh-m">{html.escape(method)}</span></th>'


def chip(name: str | None) -> str:
    """A set name as a coloured chip."""
    if not name:
        return "<span class='muted'>none</span>"
    method, model = SET_LABELS.get(name, (name, ""))
    kind = "m1" if name.startswith("m1") else "m2"
    return f'<span class="chip {kind}">{html.escape(method)} · {html.escape(model)}</span>'


def best_by(rows: list[dict], metric: str, direction: str) -> str | None:
    """The synthetic set with the best value of `metric`."""
    scored = [(r[metric], r["set"]) for r in rows if r.get(metric) is not None]
    if len(scored) < 2:
        return None
    return (max(scored) if direction == "high" else min(scored))[1]


def set_table(order: list[str], heads: list[tuple[str, str]], values, best: dict[str, str]) -> str:
    """A table with one row per set: `values(name, key)` gives each cell."""
    th = "".join(f'<th class="num">{html.escape(h)}</th>' for _, h in heads)
    body = []
    for name in order:
        method, model = SET_LABELS.get(name, (name, ""))
        tds = []
        for key, _ in heads:
            classes = ["num"] + (["best"] if name != REAL and best.get(key) == name else [])
            tds.append(f'<td class="{" ".join(classes)}">{cell(values(name, key))}</td>')
        cls = ' class="real-col"' if name == REAL else ""
        body.append(f'<tr{cls}><th scope="row">{html.escape(method)} · {html.escape(model)}</th>{"".join(tds)}</tr>')
    return f"""<div class="scroll"><table class="grid wide">
    <thead><tr><th class="mcol">Set</th>{th}</tr></thead><tbody>{"".join(body)}</tbody></table></div>"""


def retrieval_table(rows: list[dict], sets: list[str]) -> str:
    """Whole-filing retrieval metrics (BM25, no LLM)."""
    by_set = {r["set"]: r for r in rows if r["item"] == ALL}
    _, best = pick([by_set[s] for s in sets if s in by_set], by_set.get(REAL), WHOLE_WEIGHTS)

    def values(name: str, key: str):
        r = by_set.get(name, {})
        if key == "n_questions" and r.get("n_questions") is not None:
            return f"{r['n_questions']:.0f} ({r.get('n_fact_questions') or 0:.0f} + {r.get('n_analytical_questions') or 0:.0f})"
        return r.get(key)

    return set_table([*sets, REAL], RETRIEVAL_COLUMNS, values, best)


def rag_table(rag: dict[str, dict[str, dict]], sets: list[str]) -> str:
    """Whole-filing RAG answers, one column group per run."""
    heads = [(f"{run}|{col}", f"{label} ({run})") for run in rag for col, label, _ in RAG_COLUMNS]
    direction = {col: d for col, _, d in RAG_COLUMNS}
    best = {}
    for run, by_set in rag.items():
        for col, _, d in RAG_COLUMNS:
            best[f"{run}|{col}"] = best_by(
                [by_set[s] for s in sets if s in by_set],
                col,
                direction[col] if d else "high",
            )

    def values(name: str, key: str):
        run, col = key.split("|")
        return rag[run].get(name, {}).get(col)

    return set_table([*sets, REAL], heads, values, best)


def cell_text(v) -> str:
    """cell() for values that may already be formatted strings."""
    return html.escape(v) if isinstance(v, str) else cell(v)


def item_section(item: str, rows: list[dict], sets: list[str]) -> str:
    """One Item's table of text metrics, with the suggested set and per-metric best."""
    item_rows = [r for r in rows if r["item"] == item and r["set"] in sets]
    real_row = next((r for r in rows if r["item"] == item and r["set"] == REAL), None)
    if not item_rows:
        return ""
    suggested, best = pick(item_rows, real_row)
    shown = [*item_rows, *([real_row] if real_row else [])]
    body, group = [], None
    for g, metric, label in ITEM_METRICS:
        if g != group:
            body.append(f'<tr class="grp"><td colspan="{len(shown) + 1}">{html.escape(g)}</td></tr>')
            group = g
        weight, direction = WEIGHTS.get(metric, (0, ""))
        sub = " · ".join(x for x in (DIRECTIONS[direction], f"weight {weight}" if weight else "") if x)
        cells = []
        for r in shown:
            classes = ["num"]
            if r["set"] == REAL:
                classes.append("real-col")
            elif best.get(metric) == r["set"]:
                classes.append("best")
            cells.append(f'<td class="{" ".join(classes)}">{cell(r.get(metric))}</td>')
        body.append(
            f'<tr><th scope="row"><span class="mname">{html.escape(label)}</span><span class="mdir">{html.escape(sub)}</span></th>{"".join(cells)}</tr>',
        )
    label = ITEM_LABELS[item]
    number, _, name = label.partition(" ")[2].partition(" ")
    return f"""
<section class="item" id="item-{item}">
  <div class="item-head">
    <h2><span class="ino">Item {html.escape(number)}</span>{html.escape(name)}</h2>
    <dl class="picks"><div><dt>Suggested set</dt><dd>{chip(suggested)}</dd></div>
    <div><dt>Most numbers correct</dt><dd>{chip(best.get("fact_correct_pct"))}</dd></div></dl>
  </div>
  <div class="scroll"><table class="grid">
    <thead><tr><th class="mcol">Metric</th>{"".join(set_head(r["set"]) for r in shown)}</tr></thead>
    <tbody>{"".join(body)}</tbody>
  </table></div>
</section>"""


def definitions() -> str:
    """The metric definitions table."""
    body, group = [], None
    for g, label, direction, text in DEFINITIONS:
        if g != group:
            body.append(f'<tr class="grp"><td colspan="3">{html.escape(g)}</td></tr>')
            group = g
        body.append(
            f'<tr><td class="dname">{html.escape(label)}</td><td class="ddir">{html.escape(DIRECTIONS[direction])}</td>'
            f"<td>{html.escape(text)}</td></tr>",
        )
    return f"""<table class="defs-table"><thead><tr><th>Metric</th><th>Direction</th><th>What it measures and how</th></tr></thead>
  <tbody>{"".join(body)}</tbody></table>"""


def page(rows: list[dict], rag: dict[str, dict[str, dict]]) -> str:
    """The whole page."""
    sets = [s for s in SET_LABELS if s != REAL and any(r["set"] == s for r in rows)]
    findings = FINDINGS.read_text(encoding="utf-8") if FINDINGS.exists() else ""
    toc = "".join(
        f'<a href="#item-{i}">Item {ITEM_LABELS[i].split(" ")[1]} <span>{html.escape(ITEM_LABELS[i].split(" ", 2)[2])}</span></a>'
        for i in ITEM_LABELS
    )
    rag_section = (
        f"""<section id="rag">
  <h2>RAG answers: retrieval + LLM</h2>
  <p class="muted">The same questions through the RAG pipeline (top 10 passages, gemini-3.1-flash-lite-preview answering from them, the same for every set), scored against the statements. bm25 = keyword retrieval; hybrid = keyword fused with semantic embeddings.</p>
  {rag_table(rag, sets)}
</section>"""
        if rag
        else ""
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Synthetic 10-K Scorecard</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,500;6..72,600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
{CSS.read_text(encoding="utf-8")}
</style>
</head>
<body>
<div class="wrap">

{SITENAV}

<header>
  <span class="eyebrow">Vector Institute · Synthetic Data Bootcamp · Day 3 · Oct 29, 2026</span>
  <h1>Synthetic 10-K Scorecard</h1>
  <p class="lede">Six synthetic sets (two methods, three models) and a sample of real 10-Ks, asked the same ten fact and six analytical questions per company. Retrieval and RAG answers are scored over the whole filing; the text is scored section by section.</p>
</header>
{f'<section><h2>Key findings</h2><ul class="findings">{findings}</ul></section>' if findings else ""}
<section id="retrieval">
  <h2>Retrieval over the whole filing</h2>
  <p class="muted">BM25 keyword search over every passage of a set, no LLM. Questions = fact + analytical; the answers come from each company's statements, never from the text, so a filing that never states a fact scores a miss. See <a href="#definitions">metric definitions</a>.</p>
  {retrieval_table(rows, sets)}
  <p class="note">Banks have no operating income, so their operating-income, operating-margin and ratio questions that need it are dropped; real filings get the questions their XBRL figures support.</p>
</section>
{rag_section}
<section>
  <h2>Text, section by section</h2>
  <p class="muted">Shaded = best synthetic set on that metric. The suggested set is the weighted rank across the ranked metrics; numbers correct counts twice.</p>
  <nav class="toc">{toc}</nav>
</section>
{"".join(item_section(i, rows, sets) for i in ITEM_LABELS)}
<section id="definitions">
  <h2>Metric definitions</h2>
  {definitions()}
</section>
<footer><span>Built by <code>evals/scorecard_page.py</code> from <code>evals/section_scorecard.py</code> and <code>evals/rag_answers.py</code>; questions from <code>qa/fact_questions.py</code>.</span></footer>
</div>
</body>
</html>
"""


def main() -> None:
    """Write the page."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scorecard", type=Path, required=True, help="section_scorecard's scorecard.csv")
    parser.add_argument("--rag", action="append", default=[], metavar="NAME=PATH", help="a rag_answers summary.csv")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    rows = read_csv(args.scorecard)
    rag = {}
    for spec in args.rag:
        name, path = spec.split("=", 1)
        rag[name] = {r["set"]: r for r in read_csv(Path(path))}
    args.out.write_text(page(rows, rag), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
