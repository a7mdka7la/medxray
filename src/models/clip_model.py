"""CLIP wrapper used as a retrieval baseline.

We use the HF `CLIPModel` so both image and text encoders are in one object.
Embeddings are L2-normalized so cosine similarity = dot product, which lets
us hand them to FAISS later if we want an exact-search index.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..config import MODELS, RUNTIME
from ..utils.image_utils import to_pil


@dataclass
class CLIPWrapper:
    model_name: str = MODELS.clip
    device: str = RUNTIME.device
    _model: object = None
    _processor: object = None

    def load(self) -> "CLIPWrapper":
        if self._model is not None:
            return self
        from transformers import CLIPModel, CLIPProcessor

        self._model = CLIPModel.from_pretrained(self.model_name)
        self._processor = CLIPProcessor.from_pretrained(self.model_name)
        if self.device != "cpu":
            self._model = self._model.to(self.device)
        self._model.eval()
        return self

    def _normalise(self, t):
        return t / t.norm(dim=-1, keepdim=True).clamp(min=1e-12)

    def _extract_tensor(self, out) -> "torch.Tensor":  # type: ignore[name-defined]
        """transformers 4.x returns a tensor from get_image_features /
        get_text_features. Some 5.x builds return a `BaseModelOutput`-like
        wrapper; handle both."""
        import torch

        if isinstance(out, torch.Tensor):
            return out
        for attr in ("image_embeds", "text_embeds", "pooler_output", "last_hidden_state"):
            v = getattr(out, attr, None)
            if isinstance(v, torch.Tensor):
                return v if v.ndim == 2 else v.mean(dim=1)
        raise TypeError(f"Unexpected CLIP output type: {type(out)}")

    def encode_images(self, images: Sequence) -> np.ndarray:
        import torch

        assert self._model is not None and self._processor is not None
        pil = [to_pil(im) for im in images]
        batch = self._processor(images=pil, return_tensors="pt").to(self._model.device)
        with torch.inference_mode():
            out = self._model.get_image_features(**batch)
        feats = self._normalise(self._extract_tensor(out))
        return feats.detach().cpu().numpy().astype(np.float32)

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        import torch

        assert self._model is not None and self._processor is not None
        batch = self._processor(
            text=list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self._model.device)
        with torch.inference_mode():
            out = self._model.get_text_features(**batch)
        feats = self._normalise(self._extract_tensor(out))
        return feats.detach().cpu().numpy().astype(np.float32)
