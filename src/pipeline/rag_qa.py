"""RAG-based QA pipeline.

Given `(image, question)`:

  1. Encode the image with the chosen retriever (ColPali by default).
  2. Pull `top_k` most similar reports from the index.
  3. Prompt MedGemma with the question + the retrieved reports + the image.
  4. Return the grounded answer along with the retrieved evidence.

We also expose an optional second pass: retrieve again with the question text
to catch reports that mention the specific finding the user is asking about.
The two retrieval sets are deduped before being passed to the generator.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Union

from ..models.gemini_vlm import GeminiVLMWrapper
from ..models.medgemma import MedGemmaWrapper
from ..retrieval.retriever import Retriever, RetrievalHit

Backend = Literal["colpali", "clip"]
VLMGenerator = Union[MedGemmaWrapper, GeminiVLMWrapper]


def _default_generator() -> VLMGenerator:
    pref = os.getenv("MEDXRAY_GENERATOR", "auto").lower()
    if pref == "gemini":
        return GeminiVLMWrapper().load()
    if pref == "medgemma":
        return MedGemmaWrapper().load()
    try:
        return MedGemmaWrapper().load()
    except Exception as e:
        print(f"[RAGQAPipeline] MedGemma load failed ({e}); falling back to Gemini.")
        return GeminiVLMWrapper().load()


@dataclass
class QAResult:
    backend: str
    question: str
    answer: str
    retrieved: List[RetrievalHit] = field(default_factory=list)
    latency_s: float = 0.0

    def to_dict(self) -> dict:
        return {
            "backend": self.backend,
            "question": self.question,
            "answer": self.answer,
            "retrieved": [h.to_dict() for h in self.retrieved],
            "latency_s": self.latency_s,
        }


class RAGQAPipeline:
    def __init__(
        self,
        retriever: Optional[Retriever] = None,
        generator: Optional[VLMGenerator] = None,
        backend: Backend = "colpali",
    ) -> None:
        self.backend = backend
        self._retriever = retriever
        self._generator = generator

    def _ensure_retriever(self) -> Retriever:
        if self._retriever is None:
            self._retriever = (
                Retriever.colpali() if self.backend == "colpali" else Retriever.clip()
            )
        return self._retriever

    def _ensure_generator(self) -> VLMGenerator:
        if self._generator is None:
            self._generator = _default_generator()
        return self._generator

    # ------------------------------------------------------------------
    def answer(
        self,
        image,
        question: str,
        k: int = 4,
        use_text_pass: bool = True,
    ) -> QAResult:
        import time

        t0 = time.perf_counter()
        retriever = self._ensure_retriever()

        hits = retriever.search_by_image(image, k=k)
        if use_text_pass:
            text_hits = retriever.search_by_text(question, k=k)
            seen = {h.image_id for h in hits}
            hits = hits + [h for h in text_hits if h.image_id not in seen]
            hits = hits[: max(k, 4)]

        answer = self._ensure_generator().answer_question(
            image=image,
            question=question,
            retrieved_reports=[h.report for h in hits],
        )
        elapsed = time.perf_counter() - t0
        return QAResult(
            backend=self.backend,
            question=question,
            answer=answer,
            retrieved=hits,
            latency_s=elapsed,
        )
