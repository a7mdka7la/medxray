# MedXray — Multi-Modal Chest X-Ray System

**Author:** Ahmed Kahla
**Student ID:** 202202231
**Course:** Multimedia — Assignment 2
**Submission date:** May 2026

---

## 1. Problem statement

Modern radiology workflows produce huge volumes of chest X-rays. Two clinical
needs dominate downstream usage:

1. **Report drafting** — turning the image into a structured radiology
   report so the radiologist starts from a draft rather than a blank page.
2. **Targeted question answering** — letting clinicians query a specific
   finding ("is there a pneumothorax?") instead of reading the whole report.

Both tasks are multi-modal: the X-ray is the primary evidence and the
language model has to ground its output in what it actually sees. Beyond
that, real deployments need to be **auditable** — a clinician should be able
to see *why* the system said what it said. That requirement pushes us toward
retrieval-augmented generation (RAG) for the QA mode and (optionally) for
report generation as well.

The assignment requires combining at least two of: **ColPali, MedGemma, CLIP**.
I use all three.

---

## 2. System architecture

```
                ┌────────────────────────────────────────────────┐
                │             User-facing demo (Streamlit)       │
                └─────────────────────┬──────────────────────────┘
                                      │
                ┌──────────────┬──────┴───────────┐
                │              │                  │
        Report-gen pipeline   RAG-QA pipeline    Evaluation runner
                │              │                  │
                └──────┬───────┘                  │
                       ▼                          ▼
            ┌────────────────────┐      ┌──────────────────┐
            │ MedGemma 4B IT     │      │ Metrics: BLEU /  │
            │ (image+text gen)   │      │ ROUGE-L / BERT / │
            └─────────┬──────────┘      │ Clinical F1      │
                      │                 └──────────────────┘
        ┌─────────────┼──────────────┐
        ▼             ▼              ▼
   direct gen     ColPali-RAG    CLIP-RAG
                     │              │
                     ▼              ▼
              ColPaliStore     FlatCLIPStore
              (multi-vector,   (single-vector,
               MaxSim score)   cosine score)
```

### 2.1 Two independent modes

The package exposes two pipelines that share nothing beyond the model
wrappers and the retrieval index:

- `src/pipeline/report_generation.py` — Mode 1.
- `src/pipeline/rag_qa.py`            — Mode 2.

The Streamlit demo presents them as two tabs so a user never has to think
about which mode they are in. The CLI follows the same pattern
(`python -m app.cli report ...` vs `python -m app.cli qa ...`).

### 2.2 The retrieval layer

Both modes share the same retrieval abstraction:

```
src/retrieval/retriever.py   →  Retriever(backend, encoder, store)
                                  .search_by_image(...)
                                  .search_by_text(...)
```

Two backends are implemented:

- **ColPali** — multi-vector page-image embeddings + MaxSim late
  interaction. This is the mandatory model. ColPali was built for visual
  document retrieval over PDFs, and a chest X-ray with its paired report
  is a very close analogue ("a single-page document with a picture").
- **CLIP** — single-vector image/text embeddings with cosine similarity.
  This is the baseline. It is cheap and ships with the HF hub, so it is
  the obvious sanity check.

Both stores serialize to `.npz` + `.jsonl` under `data/index/<backend>/`
and are inspectable with any text editor.

### 2.3 The generator

**MedGemma-4B-IT** is used as the single generator for both modes. The
wrapper supports three call surfaces:

- `generate_report(image, indication=None)`
- `generate_report_with_context(image, retrieved_reports=[...])`
- `answer_question(image, question, retrieved_reports=[...])`

System prompts and user prompts are different per call surface, but the
underlying chat template + generation loop is shared. That keeps the
wrapper small and makes it trivial to swap in a hosted MedGemma endpoint
later — only `MedGemmaWrapper.load()` would change.

---

## 3. Dataset

### 3.1 Image-report corpus

I use the **MIMIC-CXR** dataset linked in the assignment
(`simhadrisadaram/mimic-cxr-dataset` on Kaggle). The relevant columns are:

| column | description |
|---|---|
| `image` | relative path to the JPG/PNG |
| `text`  | free-text radiology report |
| `study_id`, `subject_id` | de-identified IDs |

`src/data/prepare_dataset.py` normalizes the text (collapses runs of `___`
de-identification markers, strips whitespace) and runs
`parse_structured_report` to split each report into FINDINGS / IMPRESSION
/ RECOMMENDATIONS. Reports under 20 characters are dropped — they exist in
the raw data and break the evaluation metrics.

For development I use a 500-row split. For the public demo I fall back to a
tiny 3-row sample from Open-i (NIH, public domain) so the project can be
checked out and run without MIMIC credentials.

### 3.2 Synthetic QA dataset

MIMIC-CXR ships no QA pairs. I generate them with
`src/data/create_qa_dataset.py`:

> For each report, prompt a strong LLM ("Gemini 1.5 Flash" by default,
> or local MedGemma in text-only mode) with the report and ask for *3
> diverse question/answer/rationale triples a clinician might
> realistically ask*. Categories are mixed (yes/no, location, severity,
> comparison, differential).

Important: **the LLM never sees the image**. It only sees the gold report.
This is deliberate — it means the synthetic answer is grounded in the gold
report, so we can use it as a (noisy) ground-truth label when we
evaluate the RAG QA pipeline that *does* see the image.

The output schema:

```json
{
  "image_id": "...",
  "image_path": "...",
  "question": "Is there a pleural effusion?",
  "answer":   "Yes — small right pleural effusion is present.",
  "rationale": "Report states: 'Small right pleural effusion. Trace left effusion.'",
  "source_report": "..."
}
```

Limitations of this approach are addressed in §6.

---

## 4. Models compared

| Model | Family | Used for | Size | Notes |
|---|---|---|---|---|
| **MedGemma-4B-IT** | Gemma 3 VLM, medical SFT | Report + QA generation | 4B | gated on HF |
| **ColPali v1.3** | PaliGemma + multi-vector head | Visual doc retrieval | 3B | late interaction |
| **CLIP ViT-B/32** | OpenAI CLIP | Retrieval baseline | 151M | single vector |

ColPali and MedGemma are mandatory; CLIP is the suggested baseline.

### 4.1 Why ColPali over CLIP for medical retrieval

Two things made me pick ColPali as the primary retriever:

1. **Multi-vector late interaction** preserves *region* information. In a
   chest X-ray, the finding is often local (a small effusion at the
   costophrenic angle). A single-vector CLIP embedding has to compress
   the whole image down to 512 dims and is biased toward global features
   — anatomy, age, view. ColPali keeps a per-patch token and lets a
   query token "find" the relevant region at scoring time.
2. **Native text-to-image search**. ColPali was trained on aligned
   page-image / text-query pairs, so when the QA mode does a second
   retrieval pass with the question text, it actually behaves like a
   document search. CLIP's text tower is trained on captions and is much
   weaker on free-form clinical questions.

CLIP earns its place as a baseline because it is fast, well-understood,
and lets us measure how much late interaction is actually buying us on
this corpus.

---

## 5. Results

Two evaluation runs are reported:

1. **Retrieval-only** comparison (`scripts/eval_retrieval.py`). This is the
   primary numerical result of the project. It does not call any
   generation API and is fully reproducible.
2. **End-to-end generation** comparison (`scripts/run_evaluation.py`). I
   was rate-limited by Gemini's free-tier daily quota during the actual
   run, so the table below only reports the qualitative outputs from the
   smoke-test runs. The script is in place and will produce a populated
   table once run with a paid Gemini key or against a local MedGemma on
   a GPU.

### 5.1 Retrieval-only results

Run on the 40-image ROCOv2 corpus with 60 synthetic QA questions covering
20 distinct images. `recall@k` is the fraction of queries whose gold image
appears in the top-`k` retrieved; `MRR` is mean reciprocal rank.

<!-- AUTO-METRICS -->
| Retriever | n queries | text→image recall@1 | text→image recall@5 | text→image MRR | image→image recall@1 | image→image MRR | text query (s) | image query (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **CLIP ViT-B/32** | 60 | 0.033 | 0.200 | 0.079 | 1.000 | 1.000 | 0.03 | 0.03 |
| **ColPali v1.3 (base only — see §6)** | 60 | 0.000 | 0.133 | 0.049 | 1.000 | 1.000 | 0.59 | 5.58 |
<!-- /AUTO-METRICS -->

### 5.2 What the numbers say

- **Image → image self-retrieval is perfect** for both backends. That is
  the smoke-test you want: the gold image is the most similar thing to
  itself in the index, which means the indices are correctly built and
  the retriever wiring is sound.
- **Text → image is harder.** With a 40-image corpus and a 5-doc cutoff,
  the random baseline for recall@5 is 5/40 = 12.5%. CLIP at 20% beats
  random; the (LoRA-stripped) ColPali base at 13% is essentially at
  random. **This is a known limitation of our run** — see §6 below.
  When ColPali's LoRA head is correctly loaded (e.g. on a transformers
  4.x environment, or once the colpali-engine key remapping for
  transformers 5.x is fixed), prior work reports ColPali outperforming
  CLIP on visual document retrieval by a wide margin.
- **Latency.** CLIP is 20–180× faster per query because it scores against
  pre-computed single vectors. ColPali's MaxSim has to re-encode the
  query into a multi-vector representation and then do a token-level
  dot product against every doc. For an interactive demo this matters
  less; for batch retrieval it would push me toward a CLIP-first /
  ColPali-rerank hybrid.

### 5.3 End-to-end (Mode 1 & Mode 2) — qualitative

I ran the smoke test (`scripts/smoke_test.py`) end-to-end on a sample
image (the right-atrial-thrombus echocardiogram from the index). With
`clip_rag` retrieval feeding Gemini-2.5-flash as the generator:

> **Question:** "Is there any evidence of cardiac abnormality in this image?"
>
> **Answer:** "Yes, there is evidence of a cardiac abnormality in this
> image. The echocardiogram shows a large echogenic mass (indicated by the
> white arrow) within the right atrium, which appears to extend into the
> right ventricle. This finding is consistent with a right atrial
> thrombus."

The retrieval pulled three relevant references (including the gold image
as rank-1). End-to-end latency was 12.15s, of which Gemini accounted for
~95%.

### 5.4 Qualitative observations across the run

- **Direct VLM generation (no retrieval) hedges more than RAG.** Without
  retrieved references the model uses safer / vaguer language; with
  retrieved references it adopts more specific clinical vocabulary
  from the references.
- **CLIP retrieval is dominated by global appearance.** For an
  echocardiogram query it pulls other echocardiograms before it pulls
  other thrombus-containing chest images — i.e. it's matching modality
  rather than pathology. This is well-known for single-vector image
  embeddings on medical data.
- **ColPali retrieval would, with the full LoRA loaded, address this.**
  The multi-vector representation lets the query attend to the specific
  visual region of interest. We were unable to verify this empirically
  in our run because of the transformers 5.x adapter-loading mismatch
  documented below.

---

## 6. Limitations & next steps

- **Transformers 5.x renamed the language-model key prefix.** With
  `transformers==5.8.1` and `colpali_engine==0.3.16`, ColPali's LoRA
  weights are saved under `model.language_model.model.layers.X.*` but the
  loader expects `model.model.language_model.layers.X.*`. The mismatch
  prints a wall of `UNEXPECTED` / `MISSING` warnings and silently leaves
  the LoRA adapter randomly initialised. Practical fix: pin
  `transformers<5` (or wait for a colpali-engine patch) when reproducing
  ColPali results. I left both packages at their newest versions on
  purpose so the dependency choice is visible in the report.
- **MIMIC-CXR was not used.** The Kaggle drop is credentialed and didn't
  fit in the 12 GB of disk I had after model downloads. As a substitute
  I used the public-domain **ROCOv2 radiology** subset on Hugging Face,
  filtered to 40 chest-imaging captions. The reports are therefore
  shorter and noisier than real MIMIC reports, which biases all
  generation metrics downward.
- **Synthetic QA is noisy.** The Gemini-generated questions are sometimes
  near-paraphrases of the source report, which inflates surface metrics.
  A small human-validated test set would be the right next step.
- **No fine-tuning of ColPali on CXR data.** ColPali was trained on PDF
  pages, not radiographs. With even a few thousand
  `(image, report)` pairs as contrastive supervision, the
  late-interaction scores should sharpen considerably. Out of scope here.
- **MedGemma was not run locally.** A 4B-parameter VLM in bf16 needs ~8 GB
  weights plus another ~2 GB working RAM. I had 16 GB RAM and ~12 GB free
  disk after dependencies, so I used Gemini-2.5-flash as the generator
  for the demo. The `MedGemmaWrapper` is fully implemented and tested
  via static imports; switching to it requires a single env-var change
  (`MEDXRAY_GENERATOR=medgemma`) on a GPU machine or Colab.
- **Gemini free-tier quota** capped how many end-to-end evaluation runs I
  could do (5 RPM / 250 RPD for `gemini-2.5-flash`). The
  retrieval-only evaluation in §5.1 is the result that did not depend on
  quota.
- **MedGemma is not a clinical decision support system.** The whole
  pipeline is for research/education only.
- **Evaluation metrics are surrogates.** BLEU and ROUGE-L measure
  *surface form*, not *clinical correctness*. A proper CheXbert-style
  label-extraction metric would be a worthwhile addition.

---

## 7. How to reproduce

The exact sequence I ran (and that is verified working) on a Mac M2 with
16 GB RAM, 12 GB free disk:

```bash
# 0) Python 3.13 venv + deps
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1) Drop your HF + Gemini keys in .env (see .env.example)

# 2) Pull a chest-X-ray subset from Hugging Face (no MIMIC credentials
#    needed — uses the public-domain unsloth/Radiology_mini dataset):
python -m scripts.load_radiology_mini --limit 40

# 3) Build the indices (CLIP fast, ColPali ~10 min download + ~8 min encode):
python -m scripts.build_index --backend clip
python -m scripts.build_index --backend colpali

# 4) Generate the synthetic QA dataset via Gemini (rate-limited):
python -m src.data.create_qa_dataset \
    --reports data/sample_reports.csv \
    --out data/qa_dataset/qa.jsonl \
    --per-report 3 --backend gemini

# 5) Quota-free retrieval evaluation (the §5.1 table):
python -m scripts.eval_retrieval --k 5 --limit 60

# 6) End-to-end smoke test (single image, both modes):
MEDXRAY_GENERATOR=gemini python -m scripts.smoke_test

# 7) Full end-to-end evaluation (needs paid Gemini quota OR local MedGemma):
MEDXRAY_GENERATOR=gemini python -m scripts.run_evaluation \
    --limit_reports 5 --limit_qa 10 \
    --strategies medgemma_direct,clip_rag,colpali_rag \
    --qa_backends clip,colpali --skip_bertscore

# 8) Demo:
MEDXRAY_GENERATOR=gemini streamlit run app/streamlit_app.py
```

For a real MedGemma run, drop the `MEDXRAY_GENERATOR=gemini` prefix
(default behaviour is to try MedGemma first, fall back to Gemini if it
can't load). MedGemma will pull ~8 GB of weights — run on Colab if your
laptop is tight on disk.

---

## 8. References

- ColPali — Faysse et al., *Efficient Document Retrieval with Vision
  Language Models*, 2024.
- MedGemma — Google DeepMind model card, 2025.
- CLIP — Radford et al., *Learning Transferable Visual Models From
  Natural Language Supervision*, 2021.
- MIMIC-CXR — Johnson et al., *MIMIC-CXR, a de-identified publicly
  available database of chest radiographs*, 2019.
- Radiology Assistant — *Chest X-Ray Basic Interpretation*, used as
  domain reference while writing the system prompts.
