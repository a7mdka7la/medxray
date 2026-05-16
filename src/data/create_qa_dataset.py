"""Synthetic chest X-ray QA dataset generator.

MIMIC-CXR ships images and reports but no QA pairs. We synthesise them by
prompting a strong LLM with each report and asking for a small set of
clinically meaningful question/answer/rationale triples. The LLM never sees
the image — it only sees the gold report — so the QA pair is grounded in the
report and we can use it as a noisy ground-truth for evaluating the RAG QA
mode.

Two backends are supported:

  * `gemini`  — Google's Gemini API (default, fast, cheap, requires a key)
  * `medgemma` — local MedGemma in text-only mode (slower, free, requires
                 the HF gated model)

The output is a JSONL file with one record per QA pair:

    {
      "image_id":  "...",
      "image_path": "...",
      "question":  "...",
      "answer":    "...",
      "rationale": "...",
      "source_report": "..."
    }

Usage:
    python -m src.data.create_qa_dataset \\
        --reports data/sample_reports.csv \\
        --out data/qa_dataset/qa.jsonl \\
        --per-report 3 \\
        --backend gemini
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd
from tqdm import tqdm

from ..config import RUNTIME


PROMPT_TEMPLATE = """You are creating a question/answer dataset for chest
X-ray analysis from a radiologist's report. Given the report below, write
EXACTLY {n} diverse question-answer pairs that a clinician might realistically
ask about the X-ray. Mix categories: yes/no, location, severity, comparison,
differential.

Rules:
- Questions must be answerable from the report alone.
- Answers must be short (1-2 sentences) and clinically precise.
- Provide a short rationale citing the report span you used.
- Do NOT include the patient's identifiers.
- Output ONLY a JSON list of objects with keys: question, answer, rationale.

REPORT:
\"\"\"
{report}
\"\"\"

JSON:"""


@dataclass
class QARecord:
    image_id: str
    image_path: str
    question: str
    answer: str
    rationale: str
    source_report: str

    def to_dict(self) -> dict:
        return self.__dict__


# ----------------------------------------------------------------------
# Backends
# ----------------------------------------------------------------------


class _GeminiBackend:
    """Uses the modern `google.genai` SDK.

    The free-tier quota for `gemini-2.5-flash` is only 5 RPM, so this class
    throttles requests and retries on 429s with the server-supplied delay.
    The default model is `gemini-2.5-flash-lite-preview-09-2025` which has
    a more generous free quota.
    """

    def __init__(
        self,
        model: str = "gemini-2.5-flash-lite",
        min_interval_s: float = 6.5,
        max_retries: int = 5,
    ) -> None:
        if not RUNTIME.google_api_key:
            raise RuntimeError("GOOGLE_API_KEY missing — set it in your .env or env vars.")
        from google import genai

        self.client = genai.Client(api_key=RUNTIME.google_api_key)
        self.model = model
        self.min_interval_s = min_interval_s
        self.max_retries = max_retries
        self._last_call_t = 0.0

    def _wait(self) -> None:
        import time

        gap = time.monotonic() - self._last_call_t
        if gap < self.min_interval_s:
            time.sleep(self.min_interval_s - gap)
        self._last_call_t = time.monotonic()

    def __call__(self, prompt: str) -> str:
        import re
        import time

        last_err = None
        for attempt in range(self.max_retries):
            self._wait()
            try:
                resp = self.client.models.generate_content(model=self.model, contents=prompt)
                return getattr(resp, "text", "") or ""
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


class _MedGemmaTextBackend:
    """Use MedGemma as a text-only generator (no image attached)."""

    def __init__(self) -> None:
        from ..models.medgemma import MedGemmaWrapper

        self.wrapper = MedGemmaWrapper().load()

    def __call__(self, prompt: str) -> str:
        # We pass a 1x1 blank image so the chat template stays consistent.
        from PIL import Image

        blank = Image.new("RGB", (8, 8), (0, 0, 0))
        return self.wrapper._chat("You are a careful clinical text assistant.", prompt, blank)


def _make_backend(name: str):
    if name == "gemini":
        return _GeminiBackend()
    if name == "medgemma":
        return _MedGemmaTextBackend()
    raise ValueError(f"Unknown backend: {name}")


# ----------------------------------------------------------------------
# JSON extraction (LLMs sometimes wrap in code fences)
# ----------------------------------------------------------------------


_JSON_BLOCK = re.compile(r"\[(?:[^\[\]]|\[[^\[\]]*\])*\]", re.S)


def _extract_json_list(text: str) -> list:
    # First try fenced ```json
    if "```" in text:
        parts = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.S)
        for p in parts:
            try:
                obj = json.loads(p.strip())
                if isinstance(obj, list):
                    return obj
            except Exception:
                continue
    m = _JSON_BLOCK.search(text)
    if not m:
        return []
    try:
        return json.loads(m.group(0))
    except Exception:
        return []


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------


def generate_for_report(backend, report: str, per_report: int) -> List[dict]:
    prompt = PROMPT_TEMPLATE.format(n=per_report, report=report.strip())
    raw = backend(prompt)
    items = _extract_json_list(raw)
    cleaned: list[dict] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        q, a = it.get("question"), it.get("answer")
        if not q or not a:
            continue
        cleaned.append(
            {
                "question": str(q).strip(),
                "answer": str(a).strip(),
                "rationale": str(it.get("rationale", "")).strip(),
            }
        )
    return cleaned


def run(reports_csv: Path, out_path: Path, per_report: int, backend_name: str, limit: int | None) -> None:
    df = pd.read_csv(reports_csv)
    if limit is not None:
        df = df.head(limit)

    backend = _make_backend(backend_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_pairs = 0
    with open(out_path, "w") as f:
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"QA via {backend_name}"):
            try:
                pairs = generate_for_report(backend, row["text"], per_report)
            except Exception as e:
                tqdm.write(f"[skip] {row.get('image_id')}: {e}")
                continue
            for p in pairs:
                rec = QARecord(
                    image_id=str(row["image_id"]),
                    image_path=str(row["image_path"]),
                    source_report=str(row["text"]),
                    **p,
                )
                f.write(json.dumps(rec.to_dict()) + "\n")
                n_pairs += 1
    print(f"Wrote {n_pairs} QA pairs -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-report", type=int, default=3)
    ap.add_argument("--backend", choices=["gemini", "medgemma"], default="gemini")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    run(Path(args.reports), Path(args.out), args.per_report, args.backend, args.limit)


if __name__ == "__main__":
    main()
