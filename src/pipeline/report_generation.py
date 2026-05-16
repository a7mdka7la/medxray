"""Report Generation Mode.

Three strategies are exposed under one class. The strategy is chosen at
runtime so the demo app can switch without reloading the whole process.

  - `medgemma_direct`: zero-shot generation with the VLM.
  - `colpali_rag`:     retrieve k similar reports with ColPali, condition
                       the VLM on them, generate.
  - `clip_rag`:        same as above but with a CLIP retriever (baseline).

The VLM generator is configurable. By default we pick `MedGemmaWrapper`
(the assignment requirement) but fall back to `GeminiVLMWrapper` when
MedGemma cannot be loaded locally (e.g. low RAM/disk). Set the
`MEDXRAY_GENERATOR` env var to `medgemma` or `gemini` to force one.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Union

from ..models.gemini_vlm import GeminiVLMWrapper
from ..models.medgemma import MedGemmaWrapper
from ..retrieval.retriever import Retriever, RetrievalHit
from ..utils.text_utils import parse_structured_report

Strategy = Literal["medgemma_direct", "colpali_rag", "clip_rag"]
VLMGenerator = Union[MedGemmaWrapper, GeminiVLMWrapper]


def _default_generator() -> VLMGenerator:
    """Pick a generator based on env var, falling back gracefully if the
    local MedGemma load fails."""
    pref = os.getenv("MEDXRAY_GENERATOR", "auto").lower()
    if pref == "gemini":
        return GeminiVLMWrapper().load()
    if pref == "medgemma":
        return MedGemmaWrapper().load()
    # auto: try MedGemma, fall back to Gemini on any failure.
    try:
        return MedGemmaWrapper().load()
    except Exception as e:
        print(f"[ReportGenerator] MedGemma load failed ({e}); falling back to Gemini.")
        return GeminiVLMWrapper().load()


@dataclass
class ReportResult:
    strategy: str
    raw_text: str
    sections: dict
    retrieved: List[RetrievalHit] = field(default_factory=list)
    latency_s: float = 0.0

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "raw_text": self.raw_text,
            "sections": self.sections,
            "retrieved": [h.to_dict() for h in self.retrieved],
            "latency_s": self.latency_s,
        }


class ReportGenerator:
    def __init__(
        self,
        generator: Optional[VLMGenerator] = None,
        colpali_retriever: Optional[Retriever] = None,
        clip_retriever: Optional[Retriever] = None,
    ) -> None:
        self.generator = generator
        self.colpali = colpali_retriever
        self.clip = clip_retriever

    # Lazy loaders so the demo app can prepare in the background.
    def _ensure_generator(self) -> VLMGenerator:
        if self.generator is None:
            self.generator = _default_generator()
        return self.generator

    def _ensure_colpali(self) -> Retriever:
        if self.colpali is None:
            self.colpali = Retriever.colpali()
        return self.colpali

    def _ensure_clip(self) -> Retriever:
        if self.clip is None:
            self.clip = Retriever.clip()
        return self.clip

    # ------------------------------------------------------------------
    def generate(
        self,
        image,
        strategy: Strategy = "medgemma_direct",
        indication: Optional[str] = None,
        k: int = 3,
    ) -> ReportResult:
        import time

        t0 = time.perf_counter()
        hits: List[RetrievalHit] = []
        if strategy == "medgemma_direct":
            text = self._ensure_generator().generate_report(image, indication=indication)
        elif strategy in {"colpali_rag", "clip_rag"}:
            retriever = self._ensure_colpali() if strategy == "colpali_rag" else self._ensure_clip()
            hits = retriever.search_by_image(image, k=k)
            text = self._ensure_generator().generate_report_with_context(
                image, retrieved_reports=[h.report for h in hits]
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        elapsed = time.perf_counter() - t0
        return ReportResult(
            strategy=strategy,
            raw_text=text,
            sections=parse_structured_report(text),
            retrieved=hits,
            latency_s=elapsed,
        )
