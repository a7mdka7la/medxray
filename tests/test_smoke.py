"""Import-level smoke tests.

These do not load any of the large models — they only check that every
module imports cleanly and the small pieces (text utils, store
serialization, metric helpers without external deps) behave.
"""

from __future__ import annotations

import numpy as np

from src.utils.text_utils import normalize_report, parse_structured_report
from src.evaluation.metrics import clinical_token_accuracy
from src.retrieval.vector_store import ColPaliStore, FlatCLIPStore, _MetaRow


def test_normalize_report() -> None:
    raw = "FINDINGS:  small ___ effusion.\n\nIMPRESSION:   stable."
    out = normalize_report(raw)
    assert "___" not in out
    assert "  " not in out


def test_parse_structured_report() -> None:
    text = (
        "FINDINGS: clear lungs.\n"
        "IMPRESSION: no acute findings.\n"
        "RECOMMENDATIONS: none."
    )
    parsed = parse_structured_report(text)
    assert "clear" in parsed["findings"].lower()
    assert "no acute" in parsed["impression"].lower()
    assert "none" in parsed["recommendations"].lower()


def test_clinical_f1_perfect() -> None:
    f1 = clinical_token_accuracy(
        ["small right pleural effusion"],
        ["right pleural effusion noted"],
    )
    assert f1 > 0.5


def test_clinical_f1_empty() -> None:
    assert clinical_token_accuracy([""], [""]) == 1.0


def test_flat_store_roundtrip(tmp_path) -> None:
    store = FlatCLIPStore(dim=4)
    vecs = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]], dtype=np.float32)
    meta = [
        _MetaRow("a", "a.png", "report a"),
        _MetaRow("b", "b.png", "report b"),
    ]
    store.add(vecs, meta)
    store.save(tmp_path)
    again = FlatCLIPStore.load(tmp_path)
    assert again.embeddings.shape == (2, 4)
    assert again.meta[0].image_id == "a"

    hits = again.search(np.array([1.0, 0, 0, 0], dtype=np.float32), k=1)
    assert hits[0][0] == 0


def test_colpali_store_roundtrip(tmp_path) -> None:
    store = ColPaliStore()
    d1 = np.random.rand(5, 8).astype(np.float32)
    d2 = np.random.rand(7, 8).astype(np.float32)
    store.add([d1, d2], [_MetaRow("a", "a.png", "ra"), _MetaRow("b", "b.png", "rb")])
    store.save(tmp_path)
    again = ColPaliStore.load(tmp_path)
    assert len(again.docs) == 2
    assert again.docs[0].shape == (5, 8)
    assert again.meta[1].report == "rb"
