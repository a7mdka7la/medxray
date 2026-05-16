"""Evaluation metrics for report generation and QA.

We use:
  * BLEU-4         (sacrebleu)
  * ROUGE-L        (rouge-score)
  * BERTScore F1   (bert-score)
  * Clinical token accuracy — overlap of a small clinical-term lexicon
    between hypothesis and reference. Cheap stand-in for CheXbert.

All metrics return floats in [0, 1] (BLEU/ROUGE/BERTScore are scaled to 1.0
even if the underlying libraries return 0-100).
"""

from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np

from ..utils.text_utils import parse_structured_report


# A pragmatic, hand-picked set of high-frequency chest-X-ray findings.
CLINICAL_TERMS = {
    "pneumothorax",
    "effusion",
    "consolidation",
    "edema",
    "atelectasis",
    "cardiomegaly",
    "opacity",
    "infiltrate",
    "pneumonia",
    "nodule",
    "mass",
    "fracture",
    "lesion",
    "tube",
    "catheter",
    "device",
    "calcification",
    "emphysema",
    "fibrosis",
    "no acute",
    "normal",
    "clear",
    "stable",
    "unchanged",
    "improved",
    "worsened",
    "bilateral",
    "left",
    "right",
}


def _safe_lower(s: str) -> str:
    return (s or "").lower()


def bleu_score(hyps: Sequence[str], refs: Sequence[str]) -> float:
    try:
        from sacrebleu import corpus_bleu

        bleu = corpus_bleu(list(hyps), [list(refs)])
        return float(bleu.score) / 100.0
    except Exception:
        return float("nan")


def rouge_l_score(hyps: Sequence[str], refs: Sequence[str]) -> float:
    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = [scorer.score(r, h)["rougeL"].fmeasure for h, r in zip(hyps, refs)]
        return float(np.mean(scores)) if scores else float("nan")
    except Exception:
        return float("nan")


def bertscore_f1(hyps: Sequence[str], refs: Sequence[str], model: str = "distilbert-base-uncased") -> float:
    try:
        from bert_score import score as _bs

        _, _, F1 = _bs(list(hyps), list(refs), model_type=model, verbose=False)
        return float(F1.mean().item())
    except Exception:
        return float("nan")


def clinical_token_accuracy(hyps: Sequence[str], refs: Sequence[str]) -> float:
    """For each (hyp, ref) pair, compute F1 over the intersection with
    `CLINICAL_TERMS`. Mean across the corpus.
    """
    f1s: list[float] = []
    for h, r in zip(hyps, refs):
        h_set = {t for t in CLINICAL_TERMS if t in _safe_lower(h)}
        r_set = {t for t in CLINICAL_TERMS if t in _safe_lower(r)}
        if not h_set and not r_set:
            f1s.append(1.0)
            continue
        if not h_set or not r_set:
            f1s.append(0.0)
            continue
        tp = len(h_set & r_set)
        precision = tp / len(h_set)
        recall = tp / len(r_set)
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        f1s.append(f1)
    return float(np.mean(f1s)) if f1s else float("nan")


def section_scores(hyps: Sequence[str], refs: Sequence[str]) -> dict:
    """Per-section metrics on FINDINGS / IMPRESSION."""
    parsed_h = [parse_structured_report(x) for x in hyps]
    parsed_r = [parse_structured_report(x) for x in refs]

    def _pull(section: str, side: list[dict]) -> list[str]:
        return [d.get(section, "") for d in side]

    out: dict = {}
    for sec in ("findings", "impression"):
        h_sec = _pull(sec, parsed_h)
        r_sec = _pull(sec, parsed_r)
        out[sec] = {
            "bleu4": bleu_score(h_sec, r_sec),
            "rouge_l": rouge_l_score(h_sec, r_sec),
            "clinical_f1": clinical_token_accuracy(h_sec, r_sec),
        }
    return out
