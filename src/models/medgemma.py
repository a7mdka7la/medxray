"""MedGemma wrapper.

Loads `google/medgemma-4b-it` (image+text instruction-tuned) and exposes a
single `.generate(image, prompt) -> str` method. The constructor does not
touch the network — call `.load()` first so a missing checkpoint shows up
when you actually try to use the model, not when the package is imported.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from PIL import Image

from ..config import MODELS, RUNTIME
from ..utils.image_utils import resize_keep_aspect, to_pil


SYSTEM_REPORT = (
    "You are a board-certified radiologist drafting a concise chest X-ray "
    "report. Only describe what is visible. Be specific about laterality, "
    "anatomic location, and severity. Do not invent demographic information."
)

SYSTEM_QA = (
    "You are a board-certified radiologist. Answer the user's question about "
    "the chest X-ray using ONLY the provided image and retrieved reports. "
    "If the evidence is insufficient, say so explicitly."
)


@dataclass
class MedGemmaWrapper:
    model_name: str = MODELS.medgemma
    device: str = RUNTIME.device
    dtype: str = "bfloat16"
    _model: object = None
    _processor: object = None

    def load(self) -> "MedGemmaWrapper":
        if self._model is not None:
            return self
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText

        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        torch_dtype = dtype_map.get(self.dtype, torch.bfloat16)

        kwargs = {"torch_dtype": torch_dtype}
        if RUNTIME.hf_token:
            kwargs["token"] = RUNTIME.hf_token

        self._processor = AutoProcessor.from_pretrained(self.model_name, **kwargs)
        self._model = AutoModelForImageTextToText.from_pretrained(self.model_name, **kwargs)
        if self.device != "cpu":
            self._model = self._model.to(self.device)
        self._model.eval()
        return self

    def _chat(self, system: str, user_text: str, image: Image.Image) -> str:
        import torch

        assert self._model is not None and self._processor is not None, "call .load() first"
        image = resize_keep_aspect(image)

        messages = [
            {"role": "system", "content": [{"type": "text", "text": system}]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image", "image": image},
                ],
            },
        ]
        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model.device, dtype=self._model.dtype)

        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            generated = self._model.generate(
                **inputs,
                max_new_tokens=RUNTIME.max_new_tokens,
                do_sample=False,
            )
        new_tokens = generated[0][input_len:]
        return self._processor.decode(new_tokens, skip_special_tokens=True).strip()

    # ---- public surface ----------------------------------------------------

    def generate_report(self, image, indication: Optional[str] = None) -> str:
        prompt = (
            "Write a chest X-ray report for the attached image. Use exactly "
            "three sections in this order: FINDINGS, IMPRESSION, RECOMMENDATIONS. "
            "Each section header must be on its own line followed by a colon."
        )
        if indication:
            prompt += f"\n\nClinical indication: {indication.strip()}"
        return self._chat(SYSTEM_REPORT, prompt, to_pil(image))

    def generate_report_with_context(self, image, retrieved_reports: list[str]) -> str:
        ctx = "\n\n".join(f"[REF {i + 1}]\n{r.strip()}" for i, r in enumerate(retrieved_reports))
        prompt = (
            "Below are reports for chest X-rays that the retriever found similar "
            "to the attached image. Use them ONLY as stylistic and clinical priors; "
            "do not copy unrelated findings. Now write a fresh report for the "
            "attached image with three sections (FINDINGS, IMPRESSION, "
            "RECOMMENDATIONS).\n\n"
            f"Retrieved references:\n{ctx}"
        )
        return self._chat(SYSTEM_REPORT, prompt, to_pil(image))

    def answer_question(self, image, question: str, retrieved_reports: list[str]) -> str:
        ctx = "\n\n".join(f"[REF {i + 1}]\n{r.strip()}" for i, r in enumerate(retrieved_reports))
        prompt = (
            f"Question: {question.strip()}\n\n"
            "Use the attached image as primary evidence. Cite the retrieved "
            "reports below as supporting context where relevant; do not be swayed "
            "by them if they disagree with the image. Give a short answer "
            "(1-3 sentences) followed by a brief justification.\n\n"
            f"Retrieved reports:\n{ctx}"
        )
        return self._chat(SYSTEM_QA, prompt, to_pil(image))
