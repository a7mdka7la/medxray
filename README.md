# MedXray — Multi-Modal Chest X-Ray System

A two-mode medical AI system that combines vision and language models for chest X-ray
analysis. Built for the Multimedia Assignment 2.

- **Mode 1 — Report Generation:** image in, structured radiology report out.
- **Mode 2 — RAG QA:** image plus a clinical question in, grounded answer out.

The project compares three different model families across both modes:
ColPali (visual document retriever), MedGemma (medical VLM), and CLIP
(general vision-language baseline).

---

## Author
- Ahmed Kahla
- ID: 202202231

---

## Project layout
```
medxray/
├── src/
│   ├── config.py                 # central paths and model names
│   ├── models/                   # thin wrappers around each model
│   │   ├── medgemma.py           # MedGemma VLM (generation)
│   │   ├── colpali.py            # ColPali visual document retriever
│   │   └── clip_model.py         # CLIP baseline retriever
│   ├── retrieval/
│   │   ├── vector_store.py       # local FAISS-style index helpers
│   │   └── retriever.py          # unified retriever interface
│   ├── pipeline/
│   │   ├── report_generation.py  # image -> report (3 strategies)
│   │   └── rag_qa.py             # image + question -> grounded answer
│   ├── evaluation/
│   │   ├── metrics.py            # BLEU / ROUGE / BERTScore / clinical accuracy
│   │   └── compare_models.py     # cross-model evaluation runner
│   ├── data/
│   │   ├── prepare_dataset.py    # MIMIC-CXR loader + cleaning
│   │   └── create_qa_dataset.py  # synthetic QA generation from reports
│   └── utils/image_utils.py
├── app/
│   ├── streamlit_app.py          # main demo
│   └── cli.py                    # terminal demo
├── scripts/
│   ├── build_index.py
│   ├── run_evaluation.py
│   └── download_sample_data.py
├── data/                         # local data / indices (gitignored)
├── docs/REPORT.md                # short architecture + comparison report
├── requirements.txt
└── .env.example
```

---

## Quick start

### 1) Environment
Python 3.10+ is recommended. Tested with 3.10–3.14.
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Credentials
MedGemma is gated on Hugging Face. You also need a Gemini key if you want to
either (a) generate the synthetic QA dataset or (b) use Gemini as the
fall-back VLM generator (recommended on machines without a CUDA GPU).

Copy `.env.example` to `.env` and fill in:
```
HUGGINGFACE_TOKEN=hf_xxx
GOOGLE_API_KEY=AIza...
```

Make sure to accept the MedGemma license here:
https://huggingface.co/google/medgemma-4b-it

### Choosing a VLM generator

Both pipelines pick a generator at runtime based on the `MEDXRAY_GENERATOR`
environment variable:

- `MEDXRAY_GENERATOR=medgemma` — force MedGemma-4B (needs ~10GB RAM, GPU
  strongly recommended).
- `MEDXRAY_GENERATOR=gemini`   — force Gemini-2.5-flash-lite via API
  (works on any laptop, needs a `GOOGLE_API_KEY`).
- *unset / `auto`* (default) — try MedGemma; if loading fails (out of
  RAM, weights not downloaded, license not accepted), fall back to Gemini.

The retriever side (ColPali vs CLIP) is unaffected by this — both retrievers
run locally either way.

### 3) Data
The full MIMIC-CXR dataset is huge and credentialed. For development, place a
small subset in `data/sample_images/` and the matching reports in
`data/sample_reports.csv` (must contain `image_id` and `text` columns).

A helper is provided to download a small public sample to verify the pipeline:
```bash
python -m scripts.download_sample_data
```

### 4) Build the index (for RAG and retrieval-augmented report generation)
```bash
python -m scripts.build_index --backend colpali     # default
python -m scripts.build_index --backend clip        # baseline
```

### 5) Generate the synthetic QA dataset from the reports
There is no QA dataset shipped with MIMIC-CXR, so we create one. The script
prompts Gemini (or a local MedGemma) to produce 3 question/answer pairs per
report along with a short rationale.
```bash
python -m src.data.create_qa_dataset \
  --reports data/sample_reports.csv \
  --out data/qa_dataset/qa.jsonl \
  --per-report 3
```

### 6) Run the demo
```bash
streamlit run app/streamlit_app.py
```
Or the CLI:
```bash
python -m app.cli report --image path/to/cxr.jpg --model medgemma
python -m app.cli qa --image path/to/cxr.jpg --question "Is there pleural effusion?"
```

---

## Mode 1 — Report generation

Three strategies are implemented and compared:

| Strategy | Vision encoder | Text generator | What it does |
|----------|----------------|----------------|--------------|
| `medgemma_direct` | MedGemma vision tower | MedGemma | One-shot image → report |
| `colpali_rag`     | ColPali  | MedGemma | Retrieve k similar reports, then generate conditioned on them |
| `clip_rag`        | CLIP     | MedGemma | Same as above but with a CLIP retriever (baseline) |

Outputs follow a fixed structured template: `Findings`, `Impression`,
`Recommendations`.

---

## Mode 2 — RAG QA

For an `(image, question)` pair we:

1. Encode the image with the retriever (ColPali by default, CLIP as a baseline).
2. Pull the top-`k` most similar reports from the index.
3. Build a grounded prompt: question + retrieved reports + image.
4. Generate the answer with MedGemma.

The system also returns the retrieved evidence so answers stay auditable.

---

## Model comparison

`scripts/run_evaluation.py` runs every model on a held-out split and writes a
markdown report into `docs/REPORT.md` (Findings / Impression BLEU + ROUGE-L +
BERTScore, plus clinical-accuracy on the QA set). The short written report
discusses where each model wins, where it fails, and the practical trade-offs.

---

## Notes / limitations

- MedGemma checkpoint loading and ColPali both require sizable downloads. The
  repository structure is designed so you can swap any model wrapper for a
  hosted API (Vertex AI, Hugging Face Inference Endpoints, etc.) by changing
  one class — see `src/models/medgemma.py:MedGemmaWrapper.load()`.
- The QA dataset is synthetic and not clinically validated. It is intended for
  pipeline development and qualitative comparison, not for medical decisions.
- The system is for academic use only. It is **not** a diagnostic tool.
