"""Phase 9: build data/qa_pairs.csv from the shared question bank (qa.fact_questions),
one row per question with an exact answer and its gold passages (chunk ids, see
RAG.chunks).

The standard bank ("bank" = "standard") is the same one evals.section_scorecard and
evals.rag_answers score: for every company, synthetic and real alike, ten fact
questions (a figure the statements report) and six analytical ones (a ratio of two
figures, e.g. return on equity = net income / stockholders' equity). Answers come
from the company's statements (financials.json for the 20 synthetic companies, Method
1's XBRL extract for the real sample), never from the text; the gold passages are
every chunk that states the answer, and for an analytical question every chunk that
states each input figure (gold_chunk_ids, input_gold_chunk_ids). A question with no
gold passage stays in, so step12 scores it as a miss, as the scorecard does.

The scenario bank ("bank" = "scenario") keeps Method 2's scenario questions
(method_2.step06b_add_scenarios: "why did margins compress?", "what company did it
acquire?"), grounded by a keyword or phrase search of the company's own narrative.
Only Method 2's scenario companies have them, so they aren't comparable across sets;
they feed the chatbot's scenario demo (RAG.chatbot), not the retrieval scores.

Real questions use the 6-company real sample step12 searches (REAL_SAMPLE): for each,
the latest fiscal year that has both parsed text and XBRL figures in --real-financials
(Method 1's financials.csv); a company with neither is skipped with a warning.

Run from implementations/synthetic_10k (after method_2.step03_build_index):
    uv run python -m RAG.step11_make_qa [--real-financials ../../data/sec10k/financials.csv]
"""

import argparse
import csv
import io
import json
import re
from pathlib import Path

from method_2.config import RAW_DIR
from method_2.step09_validate_filings import (
    SCENARIO_ANCHOR_FACT,
    SCENARIO_KEYWORDS,
    closest_match,
    extract_figures,
    to_float,
)
from method_2.utils import setup_logging, write_atomic
from qa.fact_questions import load_xbrl_metrics, make_questions, real_facts, synthetic_facts

from RAG.chunks import Chunk, load_chunks


logger = setup_logging("make_qa")

SYNTHETIC_DIR = RAW_DIR.parent / "synthetic"
QA_PATH = RAW_DIR.parent / "qa_pairs.csv"
QA_FIELDS = [
    "qa_id",
    "source",
    "bank",
    "ticker",
    "fiscal_year",
    "item",
    "style",
    "fact",
    "question",
    "answer",
    "gold_chunk_id",
    "gold_chunk_ids",
    "input_gold_chunk_ids",
    "is_percent",
]
DEFAULT_REAL_FINANCIALS = RAW_DIR.parents[4] / "data" / "sec10k" / "financials.csv"
# The real companies step12's real and mixed scopes search (RAG.step12's REAL_SAMPLE_TICKERS).
REAL_SAMPLE = ["AEP", "BEN", "USB", "CVS", "NOC", "CRWD"]

# One "why" question per keyword-grounded scenario type (method_2.step06b_add_scenarios),
# phrased so answering it requires reading the passage that connects the scenario's
# cause to its effect, not just matching a category label. The two anchor-fact types
# (mergers_and_acquisitions, product_launch) are handled separately below since their
# answer is a proper noun, not a matched keyword.
SCENARIO_KEYWORD_QUESTIONS = {
    "competitive_displacement": "Why did {name}'s operating margin come under pressure during fiscal {fy}?",
    "margin_crisis": "Why did {name}'s margins compress to historically thin levels in fiscal {fy}?",
    "capital_structure": "What is driving {name}'s elevated financial leverage?",
    "cash_burn": "Why did {name} record an operating loss in fiscal {fy} rather than sustained profitability?",
    "regulatory_rate_case": "Why did {name} fall short of recovering its full requested costs in fiscal {fy}?",
    "weather_event": "Why did {name}'s fiscal {fy} capital spending include incremental restoration costs?",
    "regulatory_pricing_pressure": "Why did regulatory changes affect {name}'s margins during fiscal {fy}?",
    "regulatory_consent_order": "Why did {name}'s compliance-related costs increase during fiscal {fy}?",
}

# For these two scenario types, the narrative states the scenario's driver and its
# financial-statement outcome in the *same* paragraph (verified for BIDZ and WJE, the
# only two companies with these types, at generation time -- not assumed). That makes
# a richer question answerable from one passage: the exact phrase here is the
# qualitative severity language the narrative uses, not a bare number, so answering it
# requires reading past the figures to the sentence that characterizes them.
SEVERITY_PHRASE_QUESTIONS = {
    "margin_crisis": ("How severe does {name} describe its fiscal {fy} margin compression?", "thinnest margins"),
    "cash_burn": (
        "What does {name}'s fiscal {fy} operating loss reflect about its stage of growth, per its MD&A?",
        "ahead of sustained profitability",
    ),
}

# Verified present in the risk-factor item of every company with the matching
# scenario type -- not guaranteed by construction, checked directly in the narrative,
# same as every other grounding in this file.
ACQUISITION_RISK_PHRASE = "may not realize the anticipated benefits"
PRODUCT_LAUNCH_RISK_PHRASE = "may not realize the expected benefits"


def extract_answer_token(ctx: str, target_value: float, *, is_percent: bool) -> str:
    """Pull the short span matching `target_value` out of `ctx`.

    For prose, `ctx` from extract_figures is already the exact figure token (e.g.
    "$16,233.4 million"). For a table row, `ctx` is the whole "label | value[ | pct]"
    line; this picks out just the one cell that matches, not the whole row.
    """
    if " | " not in ctx:
        return ctx
    for part in [p.strip() for p in ctx.split(" | ")[1:]]:
        value = to_float(part)
        if value is None:
            continue
        scaled = value if is_percent else value * 1e6
        if closest_match(scaled, {target_value}, relative=not is_percent):
            return part
    return ctx


def find_figure_chunk(value: float, chunks: list[Chunk], *, is_percent: bool) -> tuple[Chunk, str] | None:
    """Return (chunk, exact answer span) for the first chunk whose text states `value`."""
    for chunk in chunks:
        dollars, percents = extract_figures(chunk.text)
        for v, ctx in percents if is_percent else dollars:
            if closest_match(v, {value}, relative=not is_percent):
                return chunk, extract_answer_token(ctx, value, is_percent=is_percent)
    return None


def find_phrase_chunk(phrase: str, chunks: list[Chunk]) -> Chunk | None:
    """Return the first chunk whose text contains `phrase` (case-insensitive)."""
    lowered = phrase.lower()
    return next((c for c in chunks if lowered in c.text.lower()), None)


def qa_row(
    source: str,
    ticker: str,
    fy: int,
    item: str,
    style: str,
    question: str,
    answer: str,
    gold_chunk_id: str,
) -> dict:
    return {
        "source": source,
        "ticker": ticker,
        "fiscal_year": fy,
        "item": item,
        "style": style,
        "question": question,
        "answer": answer,
        "gold_chunk_id": gold_chunk_id,
    }


def scenario_qa_rows(spec: dict, company_chunks: list[Chunk]) -> list[dict]:
    """Questions grounded in spec.json's scenario (method_2.step06b_add_scenarios); [] for baseline."""
    scenario = spec.get("scenario", {})
    scenario_type = scenario.get("type", "baseline")
    if scenario_type == "baseline":
        return []
    name, fy, ticker, facts = spec["fictional_name"], spec["fiscal_year"], spec["fictional_id"], scenario["facts"]
    rows = []

    if scenario_type in SCENARIO_ANCHOR_FACT:
        anchor_value = facts[SCENARIO_ANCHOR_FACT[scenario_type]]
        chunk = next((c for c in company_chunks if anchor_value in c.text), None)
        if chunk:
            question = (
                f"What company did {name} acquire during fiscal {fy}?"
                if scenario_type == "mergers_and_acquisitions"
                else f"What new product did {name} launch during fiscal {fy}?"
            )
            rows.append(qa_row("synthetic", ticker, fy, chunk.item, "factual", question, anchor_value, chunk.chunk_id))
        if scenario_type == "mergers_and_acquisitions":
            found = find_figure_chunk(float(facts["purchase_price"]), company_chunks, is_percent=False)
            if found:
                price_chunk, price_answer = found
                question = f"How much did {name} pay to acquire {facts['acquired_name']}?"
                rows.append(
                    qa_row(
                        "synthetic",
                        ticker,
                        fy,
                        price_chunk.item,
                        "factual",
                        question,
                        price_answer,
                        price_chunk.chunk_id,
                    ),
                )
            risk_chunk = find_phrase_chunk(ACQUISITION_RISK_PHRASE, company_chunks)
            if risk_chunk:
                question = (
                    f"What integration risk does {name} face following its acquisition of {facts['acquired_name']}?"
                )
                rows.append(
                    qa_row(
                        "synthetic",
                        ticker,
                        fy,
                        risk_chunk.item,
                        "analytical",
                        question,
                        ACQUISITION_RISK_PHRASE,
                        risk_chunk.chunk_id,
                    ),
                )
        if scenario_type == "product_launch":
            risk_chunk = find_phrase_chunk(PRODUCT_LAUNCH_RISK_PHRASE, company_chunks)
            if risk_chunk:
                question = f"What risk does {name} face related to its launch of {facts['product_name']}?"
                rows.append(
                    qa_row(
                        "synthetic",
                        ticker,
                        fy,
                        risk_chunk.item,
                        "analytical",
                        question,
                        PRODUCT_LAUNCH_RISK_PHRASE,
                        risk_chunk.chunk_id,
                    ),
                )
        return rows

    template = SCENARIO_KEYWORD_QUESTIONS.get(scenario_type)
    keywords = SCENARIO_KEYWORDS.get(scenario_type, [])
    if not template:
        return rows
    for chunk in company_chunks:
        lowered = chunk.text.lower()
        matched_keyword = next((k for k in keywords if k in lowered), None)
        if matched_keyword:
            m = re.search(re.escape(matched_keyword) + r"\w*", chunk.text, re.IGNORECASE)
            answer = m.group(0) if m else matched_keyword
            rows.append(
                qa_row(
                    "synthetic",
                    ticker,
                    fy,
                    chunk.item,
                    "analytical",
                    template.format(name=name, fy=fy),
                    answer,
                    chunk.chunk_id,
                ),
            )

            severity = SEVERITY_PHRASE_QUESTIONS.get(scenario_type)
            if severity:
                severity_template, phrase = severity
                m2 = re.search(re.escape(phrase), chunk.text, re.IGNORECASE)
                if m2:
                    rows.append(
                        qa_row(
                            "synthetic",
                            ticker,
                            fy,
                            chunk.item,
                            "analytical",
                            severity_template.format(name=name, fy=fy),
                            m2.group(0),
                            chunk.chunk_id,
                        ),
                    )
            break
    return rows


def format_answer(value: float, *, is_percent: bool) -> str:
    """An answer as the dashboard shows it."""
    return f"{value:.2f}%" if is_percent else f"${value / 1e6:,.1f} million"


def standard_rows(source: str, ticker: str, name: str, fy: int, facts: dict, chunks: list[Chunk]) -> list[dict]:
    """The standard bank for one company as qa_pairs rows."""
    by_id = {c.chunk_id: c for c in chunks}
    rows = []
    for q in make_questions(ticker, name, fy, facts, chunks):
        first = next((g for g in q["gold_chunk_ids"].split(";") if g), None) or next(
            (g for g in re.split(r"[;|]", q["input_gold_chunk_ids"]) if g),
            "",
        )
        row = qa_row(
            source,
            ticker,
            fy,
            by_id[first].item if first else "",
            q["style"],
            q["question"],
            format_answer(q["answer"], is_percent=q["is_percent"]),
            first,
        )
        row.update(
            bank="standard",
            fact=q["fact"],
            gold_chunk_ids=q["gold_chunk_ids"],
            input_gold_chunk_ids=q["input_gold_chunk_ids"],
            is_percent=q["is_percent"],
        )
        rows.append(row)
    return rows


def synthetic_qa_rows(spec_path: Path, all_chunks: list[Chunk]) -> list[dict]:
    """The standard bank and the scenario questions for one synthetic company."""
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    financials = json.loads((spec_path.parent / "financials.json").read_text(encoding="utf-8"))
    ticker, fy, name = spec["fictional_id"], spec["fiscal_year"], spec["fictional_name"]
    company_chunks = [c for c in all_chunks if c.source == "synthetic" and c.ticker == ticker and c.fiscal_year == fy]
    rows = standard_rows("synthetic", ticker, name, fy, synthetic_facts(financials), company_chunks)
    rows.extend({**r, "bank": "scenario"} for r in scenario_qa_rows(spec, company_chunks))
    return rows


def real_qa_rows(chunks: list[Chunk], financials_csv: Path) -> list[dict]:
    """The standard bank for each REAL_SAMPLE filing, from its XBRL figures."""
    metrics = load_xbrl_metrics(financials_csv)
    if not metrics:
        logger.warning("No XBRL figures in %s; no real questions", financials_csv)
        return []
    rows = []
    for ticker in REAL_SAMPLE:
        years = {c.fiscal_year for c in chunks if c.ticker == ticker} & {y for t, y in metrics if t == ticker}
        if not years:
            logger.warning("No real filing with both text and XBRL figures for %s; skipped", ticker)
            continue
        fy = max(years)
        mine = [c for c in chunks if c.ticker == ticker and c.fiscal_year == fy]
        rows.extend(
            standard_rows("real", ticker, mine[0].company_name, fy, real_facts(metrics.get((ticker, fy), {})), mine),
        )
    return rows


def main() -> None:
    """Build data/qa_pairs.csv from the synthetic corpus and the real sample."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--real-financials",
        type=Path,
        default=DEFAULT_REAL_FINANCIALS,
        help="Method 1's financials.csv",
    )
    args = parser.parse_args()

    synthetic_chunks = load_chunks(sources={"synthetic"})
    rows = []
    for spec_path in sorted(SYNTHETIC_DIR.glob("*/*/spec.json")):
        rows.extend(synthetic_qa_rows(spec_path, synthetic_chunks))
    real_chunks = load_chunks(sources={"real"}, tickers=set(REAL_SAMPLE))
    rows.extend(real_qa_rows(real_chunks, args.real_financials))

    for i, row in enumerate(rows):
        row["qa_id"] = f"qa_{i:04d}"

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=QA_FIELDS)
    writer.writeheader()
    writer.writerows([{field: row.get(field, "") for field in QA_FIELDS} for row in rows])
    write_atomic(QA_PATH, buffer.getvalue())

    counts = {}
    for r in rows:
        key = f"{r['source']}/{r['bank']}/{r['style']}"
        counts[key] = counts.get(key, 0) + 1
    logger.info("Wrote %d QA pairs to %s: %s", len(rows), QA_PATH, counts)


if __name__ == "__main__":
    main()
