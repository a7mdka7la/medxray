"""Pull the unsloth/Radiology_mini dataset from HF, keep only chest X-rays
(by caption keyword), extract images to disk, write sample_reports.csv.

This is the fallback dataset used when MIMIC-CXR is unavailable.
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import sys
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm

from src.config import SAMPLE_IMAGES_DIR, SAMPLE_REPORTS_CSV
from src.utils.text_utils import normalize_report


CHEST_KEYWORDS = (
    "chest",
    "thorax",
    "thoracic",
    "lung",
    "pulmonary",
    "pleural",
    "cardiac",
    "cardiomediastin",
    "ribs",
    "pneumon",
    "pneumothorax",
    "effusion",
    "atelectasis",
    "consolidation",
    "bronchi",
    "trachea",
    "mediastin",
)


def _is_chest_caption(caption: str) -> bool:
    c = caption.lower()
    return any(k in c for k in CHEST_KEYWORDS)


def _to_structured_report(caption: str) -> str:
    """The captions in Radiology_mini are free-form. Wrap them into the
    FINDINGS/IMPRESSION/RECOMMENDATIONS template so the rest of our
    pipeline (which parses sections) works.
    """
    caption = normalize_report(caption)
    # Heuristic: first 2 sentences -> findings, rest -> impression.
    parts = [p.strip() for p in caption.replace(";", ".").split(".") if p.strip()]
    if not parts:
        return ""
    if len(parts) == 1:
        findings = parts[0]
        impression = parts[0]
    else:
        split = max(1, len(parts) // 2)
        findings = ". ".join(parts[:split]).strip() + "."
        impression = ". ".join(parts[split:]).strip() + "."
    return (
        f"FINDINGS: {findings}\n"
        f"IMPRESSION: {impression}\n"
        f"RECOMMENDATIONS: Clinical correlation."
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=50, help="Max number of chest X-rays to keep")
    ap.add_argument("--max-side", type=int, default=896, help="Resize images so the long side is at most this many px")
    args = ap.parse_args()

    token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        env_file = Path(".env")
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("HUGGINGFACE_TOKEN="):
                    token = line.split("=", 1)[1].strip()
                    break
    if not token:
        sys.exit("HUGGINGFACE_TOKEN missing.")

    from huggingface_hub import hf_hub_download

    print("downloading Radiology_mini parquet…")
    pq_path = hf_hub_download(
        repo_id="unsloth/Radiology_mini",
        filename="data/train-00000-of-00001.parquet",
        repo_type="dataset",
        token=token,
    )
    print("parquet:", pq_path)

    table = pq.read_table(pq_path)
    df = table.to_pandas()
    print(f"loaded {len(df)} rows from parquet")

    SAMPLE_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    kept: list[dict] = []
    seen_ids: set[str] = set()
    for _, row in tqdm(df.iterrows(), total=len(df), desc="filtering"):
        if len(kept) >= args.limit:
            break
        caption = str(row.get("caption") or "")
        if not _is_chest_caption(caption):
            continue
        image_id = str(row.get("image_id") or f"radmini_{len(kept):04d}")
        if image_id in seen_ids:
            continue
        img_dict = row.get("image")
        if not img_dict or not img_dict.get("bytes"):
            continue
        try:
            img = Image.open(io.BytesIO(img_dict["bytes"])).convert("RGB")
        except Exception:
            continue

        # Resize so disk usage stays bounded.
        w, h = img.size
        if max(w, h) > args.max_side:
            if w >= h:
                new_w, new_h = args.max_side, int(h * args.max_side / w)
            else:
                new_h, new_w = args.max_side, int(w * args.max_side / h)
            img = img.resize((new_w, new_h), Image.LANCZOS)

        out_path = SAMPLE_IMAGES_DIR / f"{image_id}.png"
        img.save(out_path, format="PNG", optimize=True)

        kept.append(
            {
                "image_id": image_id,
                "image_path": str(out_path),
                "text": _to_structured_report(caption),
            }
        )
        seen_ids.add(image_id)

    if not kept:
        sys.exit("no chest X-ray rows passed the filter")

    SAMPLE_REPORTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(SAMPLE_REPORTS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_id", "image_path", "text"])
        w.writeheader()
        for r in kept:
            w.writerow(r)
    print(f"wrote {len(kept)} rows -> {SAMPLE_REPORTS_CSV}")


if __name__ == "__main__":
    main()
