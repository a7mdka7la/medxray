"""Download a tiny public sample of chest X-rays + reports so the pipeline
is runnable without MIMIC credentials.

We use a handful of permissively licensed images shipped with the
`Open-i` Indiana University chest X-ray collection. Reports are short and
already in the public domain.

This script is intentionally conservative — it only fetches what is needed
to make a 10-row demo work end to end. Replace with the full MIMIC-CXR
once you have credentialed access.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import requests

from src.config import SAMPLE_IMAGES_DIR, SAMPLE_REPORTS_CSV


# A short hand-picked set of public-domain chest X-rays. License: CC0 /
# public domain. The first URL (Open-i mirror) is kept because it has a
# reliable cache; the rest are Wikimedia Commons. Reports are short
# educational examples — NOT real patient reports.
SAMPLE = [
    {
        "image_id": "openi_0001",
        "url": "https://openi.nlm.nih.gov/imgs/512/1/1/CXR1_1_IM-0001-3001.png",
        "text": (
            "FINDINGS: The cardiomediastinal silhouette is within normal limits. "
            "Lungs are clear without focal consolidation, pleural effusion, or "
            "pneumothorax. No acute osseous abnormality. "
            "IMPRESSION: No acute cardiopulmonary process. "
            "RECOMMENDATIONS: No follow-up required."
        ),
    },
    {
        "image_id": "wikimedia_normal",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/9/91/Chest_Xray_PA_3-8-2010.png/640px-Chest_Xray_PA_3-8-2010.png",
        "text": (
            "FINDINGS: PA chest radiograph. Heart size within normal limits. "
            "Lungs are well-expanded and clear bilaterally. No focal consolidation, "
            "pleural effusion, or pneumothorax. Bony thorax is unremarkable. "
            "IMPRESSION: Normal chest radiograph. "
            "RECOMMENDATIONS: No further imaging required at this time."
        ),
    },
    {
        "image_id": "wikimedia_pneumothorax",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/85/Right-sided_Pneumothorax.jpg/640px-Right-sided_Pneumothorax.jpg",
        "text": (
            "FINDINGS: Frontal chest radiograph demonstrates a large right-sided "
            "pneumothorax with visible visceral pleural line and absent lung "
            "markings peripherally. There is partial collapse of the right lung. "
            "Mediastinum is in midline without significant shift. Left lung is "
            "clear. "
            "IMPRESSION: Large right-sided pneumothorax. "
            "RECOMMENDATIONS: Urgent chest tube placement; surgical consultation."
        ),
    },
    {
        "image_id": "wikimedia_pneumonia",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/6/68/Lobar_pneumonia_illustrated.jpg/640px-Lobar_pneumonia_illustrated.jpg",
        "text": (
            "FINDINGS: Dense right upper lobe consolidation with air bronchograms, "
            "consistent with lobar pneumonia. No pleural effusion. Heart size and "
            "mediastinal contours are normal. Left lung is clear. "
            "IMPRESSION: Right upper lobe lobar pneumonia. "
            "RECOMMENDATIONS: Empiric antibiotic therapy; clinical follow-up; "
            "repeat radiograph in 4-6 weeks to document resolution."
        ),
    },
    {
        "image_id": "wikimedia_cardiomegaly",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c8/Cardiomegaly.JPG/640px-Cardiomegaly.JPG",
        "text": (
            "FINDINGS: Marked enlargement of the cardiac silhouette with "
            "cardiothoracic ratio greater than 0.55. Pulmonary vascular "
            "redistribution is present. No focal airspace consolidation. Small "
            "bilateral pleural effusions are noted. "
            "IMPRESSION: Cardiomegaly with findings suggestive of pulmonary "
            "venous congestion and small bilateral pleural effusions, consistent "
            "with congestive heart failure. "
            "RECOMMENDATIONS: Correlate clinically; consider echocardiogram and "
            "BNP measurement."
        ),
    },
]


_UA = "MedXray-Assignment/0.1 (academic-use; contact: ahmedkahla)"


def _download(url: str, dest: Path) -> bool:
    try:
        r = requests.get(url, headers={"User-Agent": _UA}, timeout=30)
        if r.status_code != 200 or not r.content or len(r.content) < 1024:
            return False
        dest.write_bytes(r.content)
        return True
    except Exception:
        return False


def main() -> None:
    SAMPLE_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for s in SAMPLE:
        out = SAMPLE_IMAGES_DIR / f"{s['image_id']}.png"
        if not out.exists():
            ok = _download(s["url"], out)
            print(("ok " if ok else "FAIL ") + str(out))
            if not ok:
                continue
        rows.append({"image_id": s["image_id"], "image_path": str(out), "text": s["text"]})

    if not rows:
        print("No samples downloaded — check your network. Add your own images "
              "to data/sample_images/ and a matching data/sample_reports.csv.")
        sys.exit(1)

    with open(SAMPLE_REPORTS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_id", "image_path", "text"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {len(rows)} rows -> {SAMPLE_REPORTS_CSV}")


if __name__ == "__main__":
    main()
