"""Retrieval-only evaluation: ColPali vs CLIP.

This doesn't hit any generation API, so it works without quota. For each
QA pair, we query the retriever with the *question text* and check whether
the gold image is returned in the top-k. We also do an image -> image
self-retrieval check (the gold image should be its own nearest neighbour).

Metrics:
  - recall@1, recall@5
  - mean reciprocal rank (MRR)
  - mean query latency

Usage:
    python -m scripts.eval_retrieval --qa data/qa_dataset/qa.jsonl --k 5
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.config import QA_DATASET_DIR
from src.retrieval.retriever import Retriever


def _load_qa(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _rank(hits, gold_id: str) -> int | None:
    for i, h in enumerate(hits, 1):
        if h.image_id == gold_id:
            return i
    return None


def evaluate_one(backend: str, qa_rows: list[dict], k: int) -> dict:
    retriever = Retriever.colpali(top_k=k) if backend == "colpali" else Retriever.clip(top_k=k)

    text_ranks: list[int | None] = []
    img_ranks: list[int | None] = []
    text_lats: list[float] = []
    img_lats: list[float] = []

    seen_images: set[str] = set()
    for r in qa_rows:
        gold = r["image_id"]
        # Text -> image (the actual RAG retrieval step)
        t0 = time.perf_counter()
        hits = retriever.search_by_text(r["question"], k=k)
        text_lats.append(time.perf_counter() - t0)
        text_ranks.append(_rank(hits, gold))

        # Image -> image self-retrieval (sanity check) — once per image only
        if gold in seen_images:
            continue
        seen_images.add(gold)
        try:
            t0 = time.perf_counter()
            hits = retriever.search_by_image(r["image_path"], k=k)
            img_lats.append(time.perf_counter() - t0)
            img_ranks.append(_rank(hits, gold))
        except Exception as e:
            print(f"  image search failed for {gold}: {e}")

    def _r_at(n: int, ranks: list[int | None]) -> float:
        if not ranks:
            return float("nan")
        hits = sum(1 for r in ranks if r is not None and r <= n)
        return hits / len(ranks)

    def _mrr(ranks: list[int | None]) -> float:
        if not ranks:
            return float("nan")
        return sum(1.0 / r if r else 0.0 for r in ranks) / len(ranks)

    return {
        "backend": backend,
        "n_queries": len(text_ranks),
        "n_unique_images": len(img_ranks),
        "text->image recall@1": round(_r_at(1, text_ranks), 3),
        "text->image recall@5": round(_r_at(5, text_ranks), 3),
        "text->image MRR": round(_mrr(text_ranks), 3),
        "image->image recall@1": round(_r_at(1, img_ranks), 3),
        "image->image recall@5": round(_r_at(5, img_ranks), 3),
        "image->image MRR": round(_mrr(img_ranks), 3),
        "avg text query (s)": round(sum(text_lats) / max(1, len(text_lats)), 2),
        "avg image query (s)": round(sum(img_lats) / max(1, len(img_lats)), 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", default=str(QA_DATASET_DIR / "qa.jsonl"))
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    qa = _load_qa(Path(args.qa))
    if args.limit is not None:
        qa = qa[: args.limit]
    print(f"loaded {len(qa)} QA pairs covering {len({r['image_id'] for r in qa})} images")

    results = []
    for backend in ("clip", "colpali"):
        print(f"\n=== {backend} ===")
        try:
            r = evaluate_one(backend, qa, args.k)
        except Exception as e:
            print(f"  {backend} eval failed: {e}")
            continue
        results.append(r)
        for k, v in r.items():
            print(f"  {k}: {v}")

    out = Path("outputs/retrieval_eval.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
