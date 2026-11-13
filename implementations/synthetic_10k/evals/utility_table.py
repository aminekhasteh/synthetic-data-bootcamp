"""Utility table for the final deck: whole-filing retrieval metrics on the shared
question bank, the same setup as the Day 3 scorecard (evals.section_scorecard) and
the deck's bootstrap CIs.

Setup (identical for every set and the real sample):
- Questions: qa.fact_questions' bank, 10 fact + 6 analytical per company, every
  question counted (one the text never states is a miss).
- Index: every chunk of the set's filings, prefixed with company/year/Item
  (qa.fact_questions.contextual), searched with BM25.
- Real: the scorecard's 20-filing real sample (seed 7), same questions from XBRL.

Metrics per question, over the BM25 ranking:
- coverage: answer stated anywhere in the filing (qa.fact_questions.hit over all chunks)
- recall@5 / @10: hit() within the top 5 / 10 (an analytical question needs a chunk for
  each input figure)
- MRR: 1 / the first rank k at which hit(top k) holds; 0 if never
- nDCG@10: binary relevance over every gold chunk (answer chunks plus input chunks),
  as RAG.step12 and evals.compare_sets define it; 0 for a question with no gold chunk
- self-retrieval MRR: evals.section_scorecard's diversity probe (a sentence of a chunk,
  company name removed, must find its own chunk), 5 chunks per filing
- transfer (optional, --semantic): best retriever by MRR among BM25, potion-retrieval-32M
  and their RRF hybrid, and Spearman correlation of the three MRRs with real's

Method rows pool each method's three LLM sets (question-weighted, as the deck's
bootstrap_ci.py --group). 95% CIs: cluster bootstrap over companies, stratified by set,
2,000 resamples, percentile; the index stays fixed.

Run from implementations/synthetic_10k:
    uv run python -m evals.utility_table --out evals/results/day3_utility \
        [--real-dir ../../data/sec10k/parsed --real-financials ../../data/sec10k/financials.csv] [--semantic]
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from qa.fact_questions import contextual, hit, load_xbrl_metrics, make_questions, real_facts, synthetic_facts
from RAG.retrieval import Bm25Retriever, SemanticRetriever, ndcg_at_k, reciprocal_rank, reciprocal_rank_fusion, tokenize

from evals.compare_sets import company_chunks, item_texts, load_real_filings, load_real_records, load_set
from evals.section_scorecard import QUERIES_PER_DOC, query_sentence


SETS = {
    "m1_gemini": "data/gemini20/method_1/synthetic/filings",
    "m1_gpt4o": "data/gpt4o20/method_1/synthetic/filings",
    "m1_opus": "data/opus20/method_1/synthetic/filings",
    "m2_gemini": "data/gemini20/method_2/synthetic",
    "m2_gpt4o": "data/gpt4o20/method_2/synthetic",
    "m2_opus": "data/opus20/method_2/synthetic",
}
GROUPS = {"method_1": ["m1_gemini", "m1_gpt4o", "m1_opus"], "method_2": ["m2_gemini", "m2_gpt4o", "m2_opus"]}
METRICS = ["coverage", "recall_at_5", "recall_at_10", "mrr", "ndcg_at_10", "self_mrr"]


def gold_ids(q: dict) -> set[str]:
    ids = f"{q.get('gold_chunk_ids', '')};{q.get('input_gold_chunk_ids', '')}".replace("|", ";")
    return {g for g in ids.split(";") if g}


def first_hit_rank(q: dict, ranking: list[str]) -> int | None:
    """Smallest k with hit(q, ranking[:k]), by the same rule as qa.fact_questions.hit."""
    pos = {cid: i + 1 for i, cid in enumerate(ranking)}
    best = None

    def group_rank(ids: str) -> int | None:
        ranks = [pos[g] for g in ids.split(";") if g and g in pos]
        return min(ranks) if ranks else None

    direct = group_rank(q.get("gold_chunk_ids", ""))
    if direct is not None:
        best = direct
    groups = q.get("input_gold_chunk_ids", "")
    if groups:
        rs = [r for g in groups.split("|") if (r := group_rank(g)) is not None]
        if len(rs) == len(groups.split("|")):
            best = max(rs) if best is None else min(best, max(rs))
    return best


def score_questions(questions: list[dict], chunks: list, retrievers: dict) -> dict[str, list[dict]]:
    """Per retriever, one row per question: company, coverage, r5, r10, rr, ndcg."""
    ids = {c.chunk_id for c in chunks}
    rankings = {name: {} for name in retrievers}
    for q in questions:
        for name, rank_fn in retrievers.items():
            rankings[name][id(q)] = rank_fn(q["question"])
    out = {}
    for name in retrievers:
        rows = []
        for q in questions:
            ranking = rankings[name][id(q)]
            ok = hit(q, ids)
            k = first_hit_rank(q, ranking) if ok else None
            gold = gold_ids(q)
            rows.append(
                {
                    "company": str(q["company_id"]),
                    "coverage": float(ok),
                    "recall_at_5": float(k is not None and k <= 5),
                    "recall_at_10": float(k is not None and k <= 10),
                    "mrr": 1.0 / k if k else 0.0,
                    "ndcg_at_10": ndcg_at_k(ranking, gold, 10) if gold else 0.0,
                },
            )
        out[name] = rows
    return out


def self_retrieval(chunks: list, seed: int) -> list[dict]:
    """evals.section_scorecard.self_retrieval_mrr, kept per query (with its company)."""
    docs: dict[str, list] = defaultdict(list)
    for c in chunks:
        docs[c.ticker].append(c)
    flat = [(f"{t}_{i}", cs[0].company_name, c.text, t) for t, cs in docs.items() for i, c in enumerate(cs)]
    bm25 = Bm25Retriever([c[0] for c in flat], [c[2] for c in flat])
    rng = random.Random(seed)
    by_doc: dict[str, list] = defaultdict(list)
    for c in flat:
        by_doc[c[0].rsplit("_", 1)[0]].append(c)
    rows = []
    for doc_chunks in by_doc.values():
        for chunk_id, name, text, ticker in rng.sample(doc_chunks, min(QUERIES_PER_DOC, len(doc_chunks))):
            query = query_sentence(text, {w.lower() for w in tokenize(name)}, rng)
            if query:
                rows.append({"company": ticker, "self_mrr": reciprocal_rank(bm25.rank(query), {chunk_id})})
    return rows


def make_retrievers(chunks: list, semantic: bool) -> dict:
    bm25 = Bm25Retriever([c.chunk_id for c in chunks], [contextual(c) for c in chunks])
    out = {"bm25": bm25.rank}
    if semantic:
        sem = SemanticRetriever([c.chunk_id for c in chunks], [c.text for c in chunks])
        out["semantic"] = sem.rank
        out["hybrid_rrf"] = lambda text: reciprocal_rank_fusion([bm25.rank(text), sem.rank(text)])
    return out


def synthetic_set(path: Path, name: str, out: Path, semantic: bool, seed: int):
    chunks, questions = [], []
    for c in load_set(name, path, out):
        texts = item_texts(c.folder)
        mine = company_chunks(
            c.company_id,
            c.name,
            c.fiscal_year,
            {i: c.folder / f"item_{i}.txt" for i in texts},
            "synthetic",
        )
        chunks.extend(mine)
        questions.extend(make_questions(c.company_id, c.name, c.fiscal_year, synthetic_facts(c.financials), mine))
    return score_questions(questions, chunks, make_retrievers(chunks, semantic)), self_retrieval(chunks, seed)


def real_set(real_dir: Path, financials: Path, n: int, seed: int, semantic: bool):
    records = load_real_records(financials)
    filings = load_real_filings(real_dir, records)
    sample = random.Random(seed).sample(filings, min(n, len(filings)))
    metrics = load_xbrl_metrics(financials)
    chunks, questions = [], []
    for f in sample:
        mine = company_chunks(f.ticker, f.name, f.fiscal_year, f.paths, "real")
        chunks.extend(mine)
        questions.extend(
            make_questions(
                f.ticker,
                f.name,
                f.fiscal_year,
                real_facts(metrics.get((f.ticker, f.fiscal_year), {})),
                mine,
            ),
        )
    return score_questions(questions, chunks, make_retrievers(chunks, semantic)), self_retrieval(chunks, seed)


def aggregate(per_set: dict[str, list[dict]], metric: str, picks: dict[str, list[str]] | None = None) -> float:
    """Pooled mean of `metric` over the sets' rows; `picks` = resampled companies per set."""
    vals = []
    for s, rows in per_set.items():
        by_c: dict[str, list[float]] = defaultdict(list)
        for r in rows:
            by_c[r["company"]].append(r[metric])
        companies = picks[s] if picks else list(by_c)
        for c in companies:
            vals.extend(by_c.get(c, []))
    return float(np.mean(vals)) if vals else float("nan")


def bootstrap(per_set: dict[str, list[dict]], metric: str, n_boot: int, seed: int) -> tuple[float, float]:
    rng = random.Random(seed)
    companies = {s: sorted({r["company"] for r in rows}) for s, rows in per_set.items()}
    stats = []
    for _ in range(n_boot):
        picks = {s: [rng.choice(cs) for _ in cs] for s, cs in companies.items()}
        stats.append(aggregate(per_set, metric, picks))
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3:
        return None
    rx, ry = np.argsort(np.argsort(xs)), np.argsort(np.argsort(ys))
    if np.std(rx) == 0 or np.std(ry) == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--real-dir", type=Path)
    p.add_argument("--real-financials", type=Path)
    p.add_argument("--real-sample", type=int, default=20)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--semantic", action="store_true")
    p.add_argument("--out", type=Path, default=Path("evals/results/day3_utility"))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    q_rows: dict[str, dict[str, list[dict]]] = {}  # set -> retriever -> rows
    s_rows: dict[str, list[dict]] = {}
    for name, path in SETS.items():
        q_rows[name], s_rows[name] = synthetic_set(Path(path), name, args.out, args.semantic, args.seed)
    if args.real_dir and args.real_financials:
        q_rows["real"], s_rows["real"] = real_set(
            args.real_dir,
            args.real_financials,
            args.real_sample,
            args.seed,
            args.semantic,
        )

    units = {s: [s] for s in q_rows} | GROUPS
    results = []
    for unit, members in units.items():
        if not all(m in q_rows for m in members):
            continue
        row: dict = {"unit": unit, "n_companies": sum(len({r["company"] for r in q_rows[m]["bm25"]}) for m in members)}
        row["n_questions"] = sum(len(q_rows[m]["bm25"]) for m in members)
        for metric in METRICS:
            per = {m: s_rows[m] for m in members} if metric == "self_mrr" else {m: q_rows[m]["bm25"] for m in members}
            lo, hi = bootstrap(per, metric, args.n_boot, args.seed)
            row[metric] = round(aggregate(per, metric), 4)
            row[f"{metric}_lo"], row[f"{metric}_hi"] = round(lo, 4), round(hi, 4)
        # per-retriever MRR (transfer)
        for r in q_rows[members[0]]:
            row[f"mrr_{r}"] = round(aggregate({m: q_rows[m][r] for m in members}, "mrr"), 4)
        results.append(row)

    if "real" in q_rows and len(q_rows["real"]) > 1:
        real_mrr = {r: aggregate({"real": q_rows["real"][r]}, "mrr") for r in q_rows["real"]}
        real_best = max(real_mrr, key=lambda k: real_mrr[k])
        for row in results:
            own = {r: float(row[f"mrr_{r}"]) for r in real_mrr}
            row["best_retriever"] = max(own, key=lambda k: own[k])
            row["best_agrees_with_real"] = row["best_retriever"] == real_best
            row["spearman_vs_real"] = spearman([real_mrr[r] for r in real_mrr], [own[r] for r in real_mrr])

    fields = list(dict.fromkeys(k for r in results for k in r))
    with open(args.out / "utility.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)
    (args.out / "utility.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    for res in results:
        print({k: v for k, v in res.items() if not k.endswith(("_lo", "_hi"))})


if __name__ == "__main__":
    main()
