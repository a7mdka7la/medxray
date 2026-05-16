"""Unified retriever interface.

A retriever wraps (a) an encoder, (b) a vector store, and (c) the corpus
metadata (so callers get back full reports, not just IDs).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np

from ..config import INDEX_DIR, RUNTIME
from ..models.clip_model import CLIPWrapper
from ..models.colpali import ColPaliWrapper
from ..utils.image_utils import to_pil
from .vector_store import ColPaliStore, FlatCLIPStore, _MetaRow


@dataclass
class RetrievalHit:
    image_id: str
    image_path: str
    report: str
    score: float

    def to_dict(self) -> dict:
        return {
            "image_id": self.image_id,
            "image_path": self.image_path,
            "report": self.report,
            "score": self.score,
        }


class Retriever:
    """Strategy-pattern wrapper. Use `Retriever.colpali()` or
    `Retriever.clip()` factory methods to build a configured retriever.
    """

    def __init__(self, backend: str, encoder, store, top_k: int = RUNTIME.retrieval_top_k) -> None:
        self.backend = backend
        self.encoder = encoder
        self.store = store
        self.top_k = top_k

    # ---- factories ---------------------------------------------------------
    @classmethod
    def colpali(cls, store: ColPaliStore | None = None, top_k: int | None = None) -> "Retriever":
        store = store or ColPaliStore.load(INDEX_DIR / "colpali")
        encoder = ColPaliWrapper().load()
        return cls("colpali", encoder, store, top_k or RUNTIME.retrieval_top_k)

    @classmethod
    def clip(cls, store: FlatCLIPStore | None = None, top_k: int | None = None) -> "Retriever":
        store = store or FlatCLIPStore.load(INDEX_DIR / "clip")
        encoder = CLIPWrapper().load()
        return cls("clip", encoder, store, top_k or RUNTIME.retrieval_top_k)

    # ---- queries -----------------------------------------------------------
    def search_by_image(self, image, k: int | None = None) -> List[RetrievalHit]:
        k = k or self.top_k
        if self.backend == "colpali":
            q = self.encoder.encode_image_as_query(to_pil(image))
        elif self.backend == "clip":
            q = self.encoder.encode_images([to_pil(image)])[0]
        else:
            raise ValueError(self.backend)
        return self._hits(self.store.search(q, k=k))

    def search_by_text(self, text: str, k: int | None = None) -> List[RetrievalHit]:
        """Text-to-image retrieval. ColPali handles this natively; CLIP uses
        its text encoder.
        """
        k = k or self.top_k
        if self.backend == "colpali":
            q = self.encoder.encode_queries([text])[0]
        elif self.backend == "clip":
            q = self.encoder.encode_texts([text])[0]
        else:
            raise ValueError(self.backend)
        return self._hits(self.store.search(q, k=k))

    def _hits(self, raw: list[tuple[int, float]]) -> List[RetrievalHit]:
        out: list[RetrievalHit] = []
        for idx, score in raw:
            row: _MetaRow = self.store.meta[idx]
            out.append(
                RetrievalHit(
                    image_id=row.image_id,
                    image_path=row.image_path,
                    report=row.report,
                    score=score,
                )
            )
        return out
