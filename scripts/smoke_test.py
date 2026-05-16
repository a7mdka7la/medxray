"""Quick end-to-end smoke test: report generation + RAG QA on one sample.

Uses Gemini as the VLM (so it works without GPU) and CLIP as the retriever
(faster to build than ColPali). Prints both outputs so you can eyeball them.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Force Gemini path before any pipeline imports.
os.environ.setdefault("MEDXRAY_GENERATOR", "gemini")

import pandas as pd

from src.pipeline.rag_qa import RAGQAPipeline
from src.pipeline.report_generation import ReportGenerator


def main() -> None:
    df = pd.read_csv("data/sample_reports.csv")
    img_path = df.iloc[0]["image_path"]
    print(f"image: {img_path}")
    print(f"gold report:\n{df.iloc[0]['text']}\n")

    # --- Report generation ---
    print("=" * 70)
    print("REPORT GENERATION — clip_rag")
    print("=" * 70)
    gen = ReportGenerator()
    res = gen.generate(img_path, strategy="clip_rag", k=3)
    print(f"latency: {res.latency_s:.2f}s")
    print(f"retrieved: {[h.image_id for h in res.retrieved]}")
    print(f"\n{res.raw_text}\n")
    print(f"parsed sections: {list(res.sections.keys())}")

    # --- RAG QA ---
    print("\n" + "=" * 70)
    print("RAG QA — clip backend")
    print("=" * 70)
    pipe = RAGQAPipeline(backend="clip")
    q = "Is there any evidence of cardiac abnormality in this image?"
    qres = pipe.answer(img_path, q, k=3)
    print(f"Q: {q}")
    print(f"latency: {qres.latency_s:.2f}s")
    print(f"retrieved: {[h.image_id for h in qres.retrieved]}")
    print(f"\nA: {qres.answer}\n")


if __name__ == "__main__":
    main()
