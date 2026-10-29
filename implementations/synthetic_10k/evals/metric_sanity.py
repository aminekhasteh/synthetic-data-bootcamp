r"""Test each scorecard metric on its own, against controls whose answer is known.

section_scorecard.py reports eight metrics per Item. A metric is only worth showing if it
moves the right way when the text is made better or worse on purpose. For one synthetic
set and the real corpus, this runs every metric on the set as it is and on deliberately
broken copies of it, then says whether the metric reacted as expected:

- traced_pct: every figure in the text nudged off (x1.37 for amounts, +7.3 points for
  percentages), and each filing checked against another company's statements. Both
  should fall well below the baseline.
- fact_recall_at_10: the company's name blanked out of every chunk (should fall, since
  the questions name the company), and each question pasted into its gold chunk (an
  oracle: should reach about 1).
- self_retrieval_mrr: every company given the same text (should fall to about 1/n).
- vocab_jaccard: real filings split into two halves (should be high), and the set with
  every word spelled backwards (should be near 0).
- cross_doc_overlap: every company given the same text (should be 1).
- copied_from_real: real filings that are in the index (should be about 1), and real
  filings held out of it (the real baseline).
- figures_per_1k and length_ratio are plain counts; they are printed, not tested.

Then, from an existing scorecard.csv, it prints each set's weighted-rank points per Item,
so the "suggested set" in scorecard.md can be traced back to the metrics that drove it.

Output: a pass/fail table on stdout and metric_sanity.md under --out. BM25 only, no API.

Run from implementations/synthetic_10k, in the repo's uv environment:
    uv run python -m evals.metric_sanity \
        --set m1_gemini=data/gemini20/method_1/synthetic/filings \
        --scorecard evals/results/day2_scorecard/scorecard.csv
"""

import argparse
import csv
import dataclasses
import random
import re
import statistics
from collections import Counter
from pathlib import Path

import numpy as np
from method_2.utils import setup_logging, write_atomic

from evals.compare_sets import (
    DEFAULT_REAL_DIR,
    DEFAULT_REAL_FINANCIALS,
    ITEMS,
    company_chunks,
    financial_fact_rows,
    item_texts,
    load_real_filings,
    load_real_records,
    load_set,
)
from evals.section_scorecard import (
    WEIGHTS,
    RealItems,
    cross_doc_overlap,
    fact_recall_by_item,
    self_retrieval_mrr,
    top_terms,
    traced_share,
)


logger = setup_logging("metric_sanity")

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "results" / "metric_sanity"
NUMBER_RE = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?(\s?%)?")


def nudge_figures(text: str) -> str:
    """Move every amount by x1.37 and every percentage by +7.3 points, leaving years alone."""

    def repl(m: re.Match) -> str:
        whole, frac, pct = m.group(1), m.group(2) or "", m.group(3) or ""
        value = float(whole.replace(",", "") + frac)
        if not pct and not frac and 1900 <= value <= 2100:
            return m.group(0)
        decimals = len(frac) - 1 if frac else 0
        new = value + 7.3 if pct else value * 1.37
        out = (
            f"{new:,.{max(decimals, 1 if pct else 0)}f}"
            if "," in whole
            else f"{new:.{max(decimals, 1 if pct else 0)}f}"
        )
        return out + pct

    return NUMBER_RE.sub(repl, text)


def reverse_words(text: str) -> str:
    """Spell every word backwards: same lengths and layout, no shared vocabulary."""
    return re.sub(r"[A-Za-z]+", lambda m: m.group(0)[::-1], text)


def jaccard(a: list[str], b: list[str]) -> float:
    ta, tb = top_terms(a), top_terms(b)
    return round(len(ta & tb) / len(ta | tb), 3) if ta | tb else 0.0


def traced_pct(pairs: list[tuple[str, dict]]) -> float | None:
    traced = total = 0
    for text, financials in pairs:
        a, b = traced_share(text, financials)
        traced, total = traced + a, total + b
    return round(traced / total * 100, 1) if total else None


def check(rows: list[dict], metric: str, item: str, control: str, value, expect: str, ok: bool | None) -> None:
    rows.append(
        {
            "metric": metric,
            "item": item,
            "control": control,
            "value": value,
            "expect": expect,
            "result": "n/a" if ok is None else ("PASS" if ok else "FAIL"),
        },
    )


def run_controls(companies: list, real: RealItems, seed: int) -> list[dict]:
    rows: list[dict] = []
    texts = {c.company_id: item_texts(c.folder) for c in companies}
    n = len(companies)

    # fact_recall_at_10: baseline, names blanked, oracle.
    chunks, qa_rows = [], []
    for c in companies:
        mine = company_chunks(
            c.company_id,
            c.name,
            c.fiscal_year,
            {i: c.folder / f"item_{i}.txt" for i in texts[c.company_id]},
            "synthetic",
        )
        chunks.extend(mine)
        qa_rows.extend(financial_fact_rows("synthetic", c.company_id, c.name, c.fiscal_year, c.financials, mine))
    names = {c.company_id: c.name for c in companies}
    blanked = [
        dataclasses.replace(ch, text=re.sub(re.escape(names[ch.ticker]), "the Company", ch.text, flags=re.IGNORECASE))
        for ch in chunks
    ]
    gold_q: dict[str, list[str]] = {}
    for q in qa_rows:
        for gid in list(filter(None, q.get("gold_chunk_ids", "").split(";"))) or [q["gold_chunk_id"]]:
            gold_q.setdefault(gid, []).append(q["question"])
    oracle = [dataclasses.replace(ch, text=ch.text + "\n" + " ".join(gold_q.get(ch.chunk_id, []))) for ch in chunks]
    base_r, blank_r, oracle_r = (fact_recall_by_item(qa_rows, cs) for cs in (chunks, blanked, oracle))

    for item in ITEMS:
        docs = [(c, texts[c.company_id][item]) for c in companies if item in texts[c.company_id]]
        if len(docs) < 2:
            continue
        item_texts_ = [t for _, t in docs]

        base = traced_pct([(t, c.financials) for c, t in docs])
        nudged = traced_pct([(nudge_figures(t), c.financials) for c, t in docs])
        rotated = traced_pct([(t, docs[(i + 1) % len(docs)][0].financials) for i, (_, t) in enumerate(docs)])
        check(rows, "traced_pct", item, "as generated", base, "high for a grounded method", None)
        if base is not None and base >= 20:
            check(
                rows,
                "traced_pct",
                item,
                "figures nudged",
                nudged,
                "< half of baseline",
                nudged is not None and nudged < base / 2,
            )
            check(
                rows,
                "traced_pct",
                item,
                "other company's statements",
                rotated,
                "< half of baseline",
                rotated is not None and rotated < base / 2,
            )

        b = base_r.get(item, (None, 0))[0]
        if b is not None:
            bl, orc = blank_r.get(item, (None, 0))[0], oracle_r.get(item, (None, 0))[0]
            check(rows, "fact_recall_at_10", item, "as generated", b, "", None)
            check(
                rows,
                "fact_recall_at_10",
                item,
                "company name blanked",
                bl,
                "below baseline",
                bl is not None and bl < b,
            )
            check(
                rows,
                "fact_recall_at_10",
                item,
                "question pasted into gold chunk",
                orc,
                ">= 0.9",
                orc is not None and orc >= 0.9,
            )

        same = [(c.company_id, c.name, docs[0][1]) for c, _ in docs]
        mrr = self_retrieval_mrr([(c.company_id, c.name, t) for c, t in docs], seed)
        mrr_same = self_retrieval_mrr(same, seed)
        check(rows, "self_retrieval_mrr", item, "as generated", mrr, "", None)
        check(
            rows,
            "self_retrieval_mrr",
            item,
            "every company same text",
            mrr_same,
            f"<= {2 / len(docs):.2f} (about 1/n)",
            mrr_same is not None and mrr_same <= 2 / len(docs),
        )

        real_texts = list(real.texts[item].values())
        if len(real_texts) >= 4:
            half = len(real_texts) // 2
            j_real = jaccard(real_texts[:half], real_texts[half:])
            j_set = jaccard(item_texts_, real_texts)
            j_rev = jaccard([reverse_words(t) for t in item_texts_], real_texts)
            check(rows, "vocab_jaccard", item, "real half vs real half", j_real, ">= 0.6", j_real >= 0.6)
            check(
                rows,
                "vocab_jaccard",
                item,
                "as generated vs real",
                j_set,
                "between the two controls",
                j_rev < j_set <= j_real + 0.05,
            )
            check(rows, "vocab_jaccard", item, "words spelled backwards", j_rev, "<= 0.05", j_rev <= 0.05)

        ov = cross_doc_overlap(item_texts_)
        ov_same = cross_doc_overlap([item_texts_[0]] * len(item_texts_))
        check(rows, "cross_doc_overlap", item, "as generated", ov, "", None)
        check(
            rows,
            "cross_doc_overlap",
            item,
            "every company same text",
            ov_same,
            ">= 0.99",
            ov_same is not None and ov_same >= 0.99,
        )

        if len(real.index.get(item, [])):
            copied = round(statistics.median([real.copied_share(t, item) for t in item_texts_]), 3)
            indexed = round(statistics.median([real.copied_share(t, item) for t in real_texts[:n]]), 3)
            holdout = [
                float(np.isin(h, real.other_index[item]).mean()) for h in real.holdout_hashes.get(item, []) if len(h)
            ]
            ho = round(statistics.median(holdout), 3) if holdout else None
            check(rows, "copied_from_real", item, "as generated", copied, "", None)
            check(
                rows,
                "copied_from_real",
                item,
                "real filing already in the index",
                indexed,
                ">= 0.95",
                indexed >= 0.95,
            )
            check(
                rows,
                "copied_from_real",
                item,
                "real filing held out (the real baseline)",
                ho,
                "well below 1",
                ho is not None and ho < 0.5,
            )
    return rows


def points_table(scorecard_csv: Path) -> list[str]:
    """Each set's weighted-rank points per Item, and the metric that gave each the most."""
    with open(scorecard_csv, encoding="utf-8") as f:
        rows = [
            {k: (float(v) if v not in ("", None) and k not in ("set", "item") else v or None) for k, v in r.items()}
            for r in csv.DictReader(f)
        ]
    out = ["| Item | Set | Points | Top contributing metrics |", "|---|---|---|---|"]
    for item in ITEMS:
        item_rows = [r for r in rows if r["item"] == item and r["set"] != "real"]
        real_row = next((r for r in rows if r["item"] == item and r["set"] == "real"), None)
        if not item_rows:
            continue
        per_metric: dict[str, Counter] = {}
        for metric, (weight, direction) in WEIGHTS.items():
            target = real_row.get(metric) if real_row else None
            scored = []
            for r in item_rows:
                v = r.get(metric)
                if v is None or (direction in ("real", "le_real") and target is None):
                    continue
                key = {"high": -v, "one": abs(v - 1), "real": abs(v - (target or 0))}.get(
                    direction,
                    max(v - (target or 0), 0) * 10 + abs(v - (target or 0)),
                )
                scored.append((key, r["set"]))
            if len(scored) < 2:
                continue
            for rank, (_, name) in enumerate(sorted(scored)):
                per_metric.setdefault(name, Counter())[metric] += weight * (len(scored) - 1 - rank)
        totals = sorted(((sum(c.values()), s) for s, c in per_metric.items()), reverse=True)
        for total, s in totals:
            top = ", ".join(f"{m} {p:g}" for m, p in per_metric[s].most_common(3))
            out.append(f"| {item} | {s} | {total:g} | {top} |")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", required=True, metavar="NAME=PATH", help="the synthetic set to test the metrics on")
    parser.add_argument("--real-dir", type=Path, default=DEFAULT_REAL_DIR)
    parser.add_argument("--real-financials", type=Path, default=DEFAULT_REAL_FINANCIALS)
    parser.add_argument("--scorecard", type=Path, help="a scorecard.csv to explain the suggested picks of")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    name, path = args.set.split("=", 1)
    companies = load_set(name, Path(path), args.out)
    filings = load_real_filings(args.real_dir, load_real_records(args.real_financials))
    holdout = {(f.ticker, f.fiscal_year) for f in random.Random(args.seed).sample(filings, min(20, len(filings)))}
    real = RealItems(filings, holdout=holdout)

    rows = run_controls(companies, real, args.seed)
    lines = [
        f"# Metric sanity checks on {name}",
        "",
        "| Metric | Item | Control | Value | Expect | Result |",
        "|---|---|---|---|---|---|",
    ]
    lines += [
        f"| {r['metric']} | {r['item']} | {r['control']} | {r['value']} | {r['expect']} | {r['result']} |" for r in rows
    ]
    fails = [r for r in rows if r["result"] == "FAIL"]
    lines += ["", f"{sum(r['result'] == 'PASS' for r in rows)} passed, {len(fails)} failed."]
    if args.scorecard:
        lines += ["", f"## Where the suggested picks in {args.scorecard} come from", "", *points_table(args.scorecard)]
    args.out.mkdir(parents=True, exist_ok=True)
    write_atomic(args.out / "metric_sanity.md", "\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
