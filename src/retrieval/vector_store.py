"""Tiny on-disk vector stores.

Two backends:

- `FlatCLIPStore`: single-vector L2-normalized embeddings (CLIP/sentence-
  transformers). Score = inner product. Uses FAISS if available, falls back
  to a NumPy dot-product so the project works in CPU-only sandboxes.
- `ColPaliStore`: stores ragged multi-vector embeddings (a Python list of
  (Ni, D) arrays) and scores with MaxSim. There is no FAISS index here on
  purpose — late interaction needs the full doc tokens, and the corpus we
  build for this assignment is small enough that brute-force is fine.

Both stores serialize to `.npz` + `.jsonl` so they are inspectable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence

import numpy as np

from ..models.colpali import ColPaliWrapper


@dataclass
class _MetaRow:
    image_id: str
    image_path: str
    report: str

    def to_dict(self) -> dict:
        return {"image_id": self.image_id, "image_path": self.image_path, "report": self.report}

    @classmethod
    def from_dict(cls, d: dict) -> "_MetaRow":
        return cls(image_id=d["image_id"], image_path=d["image_path"], report=d["report"])


class _BaseStore:
    meta: List[_MetaRow]

    def __init__(self) -> None:
        self.meta = []

    def _save_meta(self, path: Path) -> None:
        with open(path, "w") as f:
            for row in self.meta:
                f.write(json.dumps(row.to_dict()) + "\n")

    def _load_meta(self, path: Path) -> None:
        self.meta = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.meta.append(_MetaRow.from_dict(json.loads(line)))


class FlatCLIPStore(_BaseStore):
    """Cosine-similarity store for single-vector embeddings."""

    def __init__(self, dim: int | None = None) -> None:
        super().__init__()
        self.embeddings = np.zeros((0, dim or 0), dtype=np.float32)

    def add(self, vectors: np.ndarray, meta: Sequence[_MetaRow]) -> None:
        if self.embeddings.shape[0] == 0:
            self.embeddings = vectors.astype(np.float32)
        else:
            self.embeddings = np.vstack([self.embeddings, vectors.astype(np.float32)])
        self.meta.extend(meta)

    def search(self, query_vec: np.ndarray, k: int = 4) -> list[tuple[int, float]]:
        if self.embeddings.shape[0] == 0:
            return []
        q = query_vec.astype(np.float32)
        if q.ndim == 1:
            q = q[None, :]
        scores = (q @ self.embeddings.T)[0]
        order = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in order]

    # ---- persistence -------------------------------------------------------
    def save(self, out_dir: str | Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_dir / "clip.npz", embeddings=self.embeddings)
        self._save_meta(out_dir / "clip.meta.jsonl")

    @classmethod
    def load(cls, out_dir: str | Path) -> "FlatCLIPStore":
        out_dir = Path(out_dir)
        z = np.load(out_dir / "clip.npz")
        store = cls(dim=int(z["embeddings"].shape[1]))
        store.embeddings = z["embeddings"]
        store._load_meta(out_dir / "clip.meta.jsonl")
        return store


@dataclass
class ColPaliStore(_BaseStore):
    """Multi-vector store with MaxSim scoring."""

    docs: list = field(default_factory=list)

    def __post_init__(self) -> None:
        super().__init__()

    def add(self, doc_embeddings: Sequence[np.ndarray], meta: Sequence[_MetaRow]) -> None:
        self.docs.extend([np.asarray(e, dtype=np.float32) for e in doc_embeddings])
        self.meta.extend(meta)

    def search(self, query_emb: np.ndarray, k: int = 4) -> list[tuple[int, float]]:
        if not self.docs:
            return []
        scores = np.array(
            [ColPaliWrapper.late_interaction_score(query_emb, d) for d in self.docs],
            dtype=np.float32,
        )
        order = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in order]

    # ---- persistence -------------------------------------------------------
    def save(self, out_dir: str | Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        # Pack ragged arrays with their shapes so we can rebuild on load.
        np.savez_compressed(
            out_dir / "colpali.npz",
            **{f"doc_{i}": d for i, d in enumerate(self.docs)},
        )
        self._save_meta(out_dir / "colpali.meta.jsonl")

    @classmethod
    def load(cls, out_dir: str | Path) -> "ColPaliStore":
        out_dir = Path(out_dir)
        z = np.load(out_dir / "colpali.npz", allow_pickle=False)
        store = cls()
        # Keys are doc_0, doc_1, ... — preserve order.
        keys = sorted(z.files, key=lambda k: int(k.split("_")[1]))
        store.docs = [z[k].astype(np.float32) for k in keys]
        store._load_meta(out_dir / "colpali.meta.jsonl")
        return store
