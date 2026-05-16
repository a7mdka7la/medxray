"""Gemini multimodal generator wrapper.

Mirrors the interface of `MedGemmaWrapper` so the pipelines can swap one
for the other without code changes. Use this wrapper when local GPU/RAM
isn't available — it offloads generation to the Gemini API.

Rate-limited and retry-aware: the free tier is 5 RPM for `gemini-2.5-flash`
and ~30 RPM for `gemini-2.5-flash-lite`. We default to the lite variant
and back off on 429s.
"""

from __future__ import annotations

import io
import re
import time
from dataclasses import dataclass
from typing import Optional

from PIL import Image

from ..config import RUNTIME
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
class GeminiVLMWrapper:
    model: str = "gemini-2.5-flash"
    min_interval_s: float = 7.0  # ~8 RPM, safely below the 10 RPM free-tier
    max_retries: int = 5
    _client: object = None
    _last_call_t: float = 0.0

    def load(self) -> "GeminiVLMWrapper":
        if self._client is not None:
            return self
        if not RUNTIME.google_api_key:
            raise RuntimeError("GOOGLE_API_KEY missing — set it in your .env or env vars.")
        from google import genai

        self._client = genai.Client(api_key=RUNTIME.google_api_key)
        return self

    # ---- internal --------------------------------------------------------
    def _wait(self) -> None:
        gap = time.monotonic() - self._last_call_t
        if gap < self.min_interval_s:
            time.sleep(self.min_interval_s - gap)
        self._last_call_t = time.monotonic()

    @staticmethod
    def _img_to_part(image: Image.Image) -> dict:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return {"inline_data": {"mime_type": "image/png", "data": buf.getvalue()}}

    def _generate(self, system: str, user_text: str, image: Image.Image) -> str:
        assert self._client is not None, "call .load() first"
        image = resize_keep_aspect(image)

        from google.genai import types as gtypes

        contents = [
            gtypes.Content(
                role="user",
                parts=[
                    gtypes.Part.from_bytes(data=_img_bytes(image), mime_type="image/png"),
                    gtypes.Part.from_text(text=user_text),
                ],
            )
        ]
        config = gtypes.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=RUNTIME.max_new_tokens,
            temperature=0.2,
        )

        last_err = None
        for attempt in range(self.max_retries):
            self._wait()
            try:
                resp = self._client.models.generate_content(
                    model=self.model, contents=contents, config=config
                )
                return (getattr(resp, "text", "") or "").strip()
            except Exception as e:
                last_err = e
                msg = str(e)
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                    m = re.search(r"retryDelay'?\s*:\s*'?(\d+(?:\.\d+)?)s", msg)
                    delay = float(m.group(1)) + 1.0 if m else (5 * (attempt + 1))
                    time.sleep(delay)
                    continue
                raise
        raise last_err if last_err else RuntimeError("Gemini request failed")

    # ---- public surface --------------------------------------------------
    def generate_report(self, image, indication: Optional[str] = None) -> str:
        prompt = (
            "Write a chest X-ray report for the attached image. Use exactly "
            "three sections in this order: FINDINGS, IMPRESSION, RECOMMENDATIONS. "
            "Each section header must be on its own line followed by a colon."
        )
        if indication:
            prompt += f"\n\nClinical indication: {indication.strip()}"
        return self._generate(SYSTEM_REPORT, prompt, to_pil(image))

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
        return self._generate(SYSTEM_REPORT, prompt, to_pil(image))

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
        return self._generate(SYSTEM_QA, prompt, to_pil(image))


def _img_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()
