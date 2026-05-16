"""ColPali wrapper for late-interaction visual document retrieval.

ColPali was trained to embed page images and queries into a shared
multi-vector space, then score with a MaxSim-style late interaction. We use
the `colpali-engine` library for loading and scoring, and store the
multi-vector embeddings in memory (small N) or on disk.

For chest X-rays we treat each `(image, paired_report)` row as a "document":
the image is what gets indexed, the report is the payload returned at
retrieval time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..config import MODELS, RUNTIME
from ..utils.image_utils import resize_keep_aspect, to_pil


@dataclass
class ColPaliWrapper:
    model_name: str = MODELS.colpali
    device: str = RUNTIME.device
    _model: object = None
    _processor: object = None

    def load(self) -> "ColPaliWrapper":
        if self._model is not None:
            return self
        import torch
        from colpali_engine.models import ColPali, ColPaliProcessor

        dtype = torch.bfloat16 if self.device != "cpu" else torch.float32
        kwargs: dict = {"torch_dtype": dtype}
        if RUNTIME.hf_token:
            kwargs["token"] = RUNTIME.hf_token
        # transformers 5.x lazy-loads to meta tensors and then expects you
        # to supply a device_map. Map everything to the chosen device.
        kwargs["device_map"] = {"": self.device} if self.device != "cpu" else "cpu"

        self._model = ColPali.from_pretrained(self.model_name, **kwargs)
        self._processor = ColPaliProcessor.from_pretrained(self.model_name, **kwargs)
        self._model.eval()
        return self

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode_images(self, images: Sequence) -> np.ndarray:
        """Return a list of multi-vector embeddings, one per image.

        We pad to the longest sequence inside this batch and return a
        ragged `object` ndarray so callers can store per-image matrices.
        """
        import torch

        assert self._model is not None and self._processor is not None
        pil_imgs = [resize_keep_aspect(to_pil(im)) for im in images]
        batch = self._processor.process_images(pil_imgs).to(self._model.device)
        with torch.inference_mode():
            emb = self._model(**batch)
        emb = emb.detach().to(torch.float32).cpu().numpy()
        return np.asarray([row for row in emb], dtype=object)

    def encode_queries(self, queries: Sequence[str]) -> np.ndarray:
        import torch

        assert self._model is not None and self._processor is not None
        batch = self._processor.process_queries(list(queries)).to(self._model.device)
        with torch.inference_mode():
            emb = self._model(**batch)
        emb = emb.detach().to(torch.float32).cpu().numpy()
        return np.asarray([row for row in emb], dtype=object)

    # ------------------------------------------------------------------
    # Scoring (late interaction)
    # ------------------------------------------------------------------

    @staticmethod
    def late_interaction_score(query_emb: np.ndarray, doc_emb: np.ndarray) -> float:
        """MaxSim score: for each query token, take the max similarity with
        any document token, then sum across query tokens.

        `query_emb`: (Nq, D)
        `doc_emb`:   (Nd, D)
        """
        q = np.asarray(query_emb, dtype=np.float32)
        d = np.asarray(doc_emb, dtype=np.float32)
        sims = q @ d.T  # (Nq, Nd)
        return float(sims.max(axis=1).sum())

    def score(self, queries_emb: np.ndarray, docs_emb: np.ndarray) -> np.ndarray:
        """Pairwise score matrix shape `(len(queries), len(docs))`."""
        scores = np.zeros((len(queries_emb), len(docs_emb)), dtype=np.float32)
        for i, q in enumerate(queries_emb):
            for j, d in enumerate(docs_emb):
                scores[i, j] = self.late_interaction_score(q, d)
        return scores

    # Convenience: take a single image as a "query" — for image-to-image search
    def encode_image_as_query(self, image) -> np.ndarray:
        return self.encode_images([image])[0]
