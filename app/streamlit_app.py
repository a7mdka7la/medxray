"""Streamlit demo for both modes.

Run with:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

# Make src/ importable when run from the project root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
from PIL import Image

from src.pipeline.rag_qa import RAGQAPipeline
from src.pipeline.report_generation import ReportGenerator


st.set_page_config(page_title="MedXray — Multi-Modal CXR", page_icon="🩻", layout="wide")


@st.cache_resource(show_spinner="Loading MedGemma + retrievers (one-off)…")
def get_report_generator() -> ReportGenerator:
    return ReportGenerator()


@st.cache_resource(show_spinner="Loading RAG QA pipeline…")
def get_qa_pipeline(backend: str) -> RAGQAPipeline:
    return RAGQAPipeline(backend=backend)


def _show_evidence(hits: list) -> None:
    if not hits:
        st.info("No retrieved evidence (index empty or direct generation).")
        return
    for i, h in enumerate(hits, 1):
        with st.expander(f"Reference {i} — score {h.score:.3f}  ·  {h.image_id}", expanded=(i == 1)):
            c1, c2 = st.columns([1, 2])
            try:
                c1.image(h.image_path, use_container_width=True)
            except Exception:
                c1.write(h.image_path)
            c2.markdown(h.report)


def render_report_mode() -> None:
    st.subheader("Report Generation Mode")
    st.caption("Upload a chest X-ray. The system writes a structured radiology report.")

    file = st.file_uploader("Chest X-ray", type=["png", "jpg", "jpeg"])
    indication = st.text_input("Optional clinical indication", placeholder="e.g. cough, fever, post-op chest pain")

    strategy = st.radio(
        "Strategy",
        options=["medgemma_direct", "colpali_rag", "clip_rag"],
        horizontal=True,
        help="Direct VLM, retrieval-augmented (ColPali), or retrieval-augmented (CLIP baseline).",
    )

    if file and st.button("Generate report", type="primary"):
        image = Image.open(io.BytesIO(file.getvalue())).convert("RGB")
        col_img, col_out = st.columns([1, 2])
        col_img.image(image, use_container_width=True)

        with st.spinner(f"Generating with {strategy}…"):
            gen = get_report_generator()
            result = gen.generate(image, strategy=strategy)

        col_out.markdown("### Generated report")
        col_out.code(result.raw_text, language="text")
        col_out.caption(f"latency: {result.latency_s:.2f}s")

        with col_out.expander("Parsed sections"):
            st.json(result.sections)

        st.divider()
        st.markdown("### Retrieved evidence")
        _show_evidence(result.retrieved)


def render_qa_mode() -> None:
    st.subheader("QA Mode (RAG)")
    st.caption("Ask a clinical question about the uploaded X-ray. Answers are grounded in retrieved reports.")

    file = st.file_uploader("Chest X-ray", type=["png", "jpg", "jpeg"], key="qa_file")
    question = st.text_input("Question", placeholder="e.g. Is there evidence of pleural effusion?")
    backend = st.selectbox("Retriever", options=["colpali", "clip"])
    k = st.slider("Top-k retrieved reports", 1, 8, 4)

    if file and question and st.button("Answer question", type="primary"):
        image = Image.open(io.BytesIO(file.getvalue())).convert("RGB")
        col_img, col_out = st.columns([1, 2])
        col_img.image(image, use_container_width=True)

        with st.spinner("Retrieving + answering…"):
            pipe = get_qa_pipeline(backend)
            result = pipe.answer(image, question, k=k)

        col_out.markdown("### Answer")
        col_out.write(result.answer)
        col_out.caption(f"backend: {backend}  ·  latency: {result.latency_s:.2f}s")

        st.divider()
        st.markdown("### Retrieved evidence")
        _show_evidence(result.retrieved)


def main() -> None:
    st.title("🩻 MedXray — Multi-Modal Chest X-Ray System")
    st.markdown(
        "**Two modes, three models compared.** Choose a tab below. "
        "Built for the Multimedia Assignment 2 by **Ahmed Kahla** (ID: 202202231)."
    )

    tab_report, tab_qa, tab_about = st.tabs(["Report Generation", "RAG QA", "About / Models"])

    with tab_report:
        render_report_mode()
    with tab_qa:
        render_qa_mode()
    with tab_about:
        st.markdown(
            """
**Models used**

| Model | Role | Why |
|---|---|---|
| **MedGemma 4B IT** | image+text generation | medical-domain VLM — primary generator |
| **ColPali v1.3** | visual document retrieval | late-interaction multi-vector retriever |
| **CLIP ViT-B/32** | retrieval baseline | open-domain reference for comparison |

**Pipeline**

- *Report mode:* image → (optional retrieval) → MedGemma → structured report.
- *QA mode:* (image + question) → retriever (image AND text passes) → MedGemma.

**Disclaimer.** Research/education only. Not a medical device.
            """
        )


if __name__ == "__main__":
    main()
