"""Phase 9: evaluate BM25, a real semantic embedding retriever, and their
reciprocal-rank-fusion hybrid against data/qa_pairs.csv, across three corpus scopes
(PLAN.md Phase 9): synthetic only, real only, and mixed.

"Real" here means the same 6-company sample data/qa_pairs.csv's real questions come
from (AEP, BEN, USB, CVS, NOC, CRWD) -- not all 50 real companies, which would mean
embedding on the order of 500,000 paragraphs. The questions are the shared standard
bank (qa.fact_questions via step11: fact and analytical, every question counted, so an
unanswerable one is a miss), scored with the same hit rule as the scorecard. See PLAN.md Phase 9 and
RAG.retrieval's module docstring for the semantic retriever itself (fastembed +
minishlab/potion-retrieval-32M, a static embedding model, not a transformer).

Output: data/retrieval_eval.csv, one row per (experiment, retriever).

Run from implementations/synthetic_10k (after RAG.step11_make_qa):
    uv run python -m RAG.step12_evaluate_retrievers
"""

import csv
import io
import re

from method_2.config import RAW_DIR
from method_2.utils import setup_logging, write_atomic
from qa.fact_questions import contextual, hit

from RAG.chunks import Chunk, load_chunks
from RAG.retrieval import (
    Bm25Retriever,
    SemanticRetriever,
    ndcg_at_k,
    reciprocal_rank,
    reciprocal_rank_fusion,
)


logger = setup_logging("evaluate_retrievers")

QA_PATH = RAW_DIR.parent / "qa_pairs.csv"
EVAL_PATH = RAW_DIR.parent / "retrieval_eval.csv"
EVAL_FIELDS = ["experiment", "retriever", "n_questions", "recall_at_5", "recall_at_10", "mrr", "ndcg_at_10"]

REAL_SAMPLE_TICKERS = {"AEP", "BEN", "USB", "CVS", "NOC", "CRWD"}


def load_qa_pairs() -> list[dict]:
    """The standard bank's questions (fact and analytical); the scenario bank is the
    chatbot demo's, not comparable across companies, so it isn't scored here.
    """
    with open(QA_PATH, encoding="utf-8") as f:
        return [q for q in csv.DictReader(f) if q.get("bank", "standard") == "standard"]


def qa_for(qa_pairs: list[dict], experiment: str) -> list[dict]:
    if experiment == "synthetic_only":
        return [q for q in qa_pairs if q["source"] == "synthetic"]
    if experiment == "real_only":
        return [q for q in qa_pairs if q["source"] == "real"]
    return qa_pairs


def gold_ids(q: dict) -> set[str]:
    """Every chunk that helps answer `q`: its stating chunks and its input figures' chunks."""
    return {g for g in re.split(r"[;|]", f"{q['gold_chunk_ids']};{q['input_gold_chunk_ids']}") if g}


def evaluate(qa_pairs: list[dict], rankings_by_qa: dict[str, list[str]], name: str) -> dict:
    """Recall@k by qa.fact_questions.hit (the rule every eval shares: an unanswerable
    question is a miss, an analytical one needs each input), MRR and nDCG over gold_ids.
    """
    n = len(qa_pairs)
    recall5, recall10, rr_sum, ndcg_sum = 0.0, 0.0, 0.0, 0.0
    for q in qa_pairs:
        ranking = rankings_by_qa[q["qa_id"]]
        gold = gold_ids(q)
        recall5 += hit(q, ranking[:5])
        recall10 += hit(q, ranking[:10])
        rr_sum += reciprocal_rank(ranking, gold) if gold else 0.0
        ndcg_sum += ndcg_at_k(ranking, gold, 10) if gold else 0.0
    return {
        "retriever": name,
        "n_questions": n,
        "recall_at_5": recall5 / n,
        "recall_at_10": recall10 / n,
        "mrr": rr_sum / n,
        "ndcg_at_10": ndcg_sum / n,
    }


def load_corpus(experiment: str) -> list[Chunk]:
    """Load only the index rows this experiment needs (see module docstring)."""
    if experiment == "synthetic_only":
        return load_chunks(sources={"synthetic"})
    if experiment == "real_only":
        return load_chunks(sources={"real"}, tickers=REAL_SAMPLE_TICKERS)
    return load_chunks(sources={"synthetic"}) + load_chunks(sources={"real"}, tickers=REAL_SAMPLE_TICKERS)


def run_experiment(experiment: str, all_qa: list[dict]) -> list[dict]:
    chunks = load_corpus(experiment)
    qa_pairs = qa_for(all_qa, experiment)
    if not qa_pairs:
        logger.warning("%s: no QA pairs in this corpus scope; skipping", experiment)
        return []

    chunk_ids = [c.chunk_id for c in chunks]
    texts = [c.text for c in chunks]
    logger.info("%s: embedding %d chunks for the semantic retriever", experiment, len(texts))
    # BM25 indexes each chunk with its company/year/Item prefix, as the scorecard and
    # evals.rag_answers do; the semantic retriever embeds the raw text (rag_answers).
    bm25 = Bm25Retriever(chunk_ids, [contextual(c) for c in chunks])
    semantic = SemanticRetriever(chunk_ids, texts)

    bm25_rankings, semantic_rankings, hybrid_rankings = {}, {}, {}
    for q in qa_pairs:
        b = bm25.rank(q["question"])
        s = semantic.rank(q["question"])
        bm25_rankings[q["qa_id"]] = b
        semantic_rankings[q["qa_id"]] = s
        hybrid_rankings[q["qa_id"]] = reciprocal_rank_fusion([b, s])

    rows = []
    for name, rankings in [("bm25", bm25_rankings), ("semantic", semantic_rankings), ("hybrid_rrf", hybrid_rankings)]:
        result = evaluate(qa_pairs, rankings, name)
        result["experiment"] = experiment
        rows.append(result)
    return rows


def main() -> None:
    """Run all three Phase 9 experiments and write data/retrieval_eval.csv."""
    all_qa = load_qa_pairs()
    logger.info("Loaded %d QA pairs", len(all_qa))

    rows = []
    for experiment in ["synthetic_only", "real_only", "mixed"]:
        rows.extend(run_experiment(experiment, all_qa))

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=EVAL_FIELDS)
    writer.writeheader()
    writer.writerows([{k: r[k] for k in EVAL_FIELDS} for r in rows])
    write_atomic(EVAL_PATH, buffer.getvalue())

    logger.info("Wrote %d rows to %s", len(rows), EVAL_PATH)
    for r in rows:
        logger.info(
            "%-15s %-12s n=%-4d recall@5=%.2f recall@10=%.2f mrr=%.2f ndcg@10=%.2f",
            r["experiment"],
            r["retriever"],
            r["n_questions"],
            r["recall_at_5"],
            r["recall_at_10"],
            r["mrr"],
            r["ndcg_at_10"],
        )


if __name__ == "__main__":
    main()
