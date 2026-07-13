# Objective: Test coverage for rag router behavior and regressions.
"""Test coverage for rag router behavior and regressions.

This test module verifies expected behavior, regression boundaries, and failure
handling for the corresponding runtime component.
"""


from io import BytesIO
from types import SimpleNamespace

import pytest
from app.routers import rag_router as rr
from fastapi import HTTPException, UploadFile


def test_chunk_text_basic():
    """Testa chunk text basic."""
    txt = "a " * 1000
    chunks = rr.chunk_text(txt, chunk_size=100, overlap=10)
    assert len(chunks) > 1
    assert all(isinstance(c, str) for c in chunks)


def test_extract_text_from_pdf_error(monkeypatch):
    """Testa extract text from pdf error."""
    import sys

    fake_fitz = SimpleNamespace(open=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("bad pdf")))
    monkeypatch.setitem(sys.modules, "fitz", fake_fitz)
    with pytest.raises(HTTPException) as exc:
        rr.extract_text_from_pdf(b"%PDF-invalid")
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_summarize_text_success_and_fallback(monkeypatch):
    """Testa summarize text success and fallback."""
    async def _ok(**kwargs):
        """Execute the ok routine.

This helper encapsulates one focused step used by the surrounding workflow."""
        return "TÍTULO: Manual\nRESUMO: Guia curto.", {}

    monkeypatch.setattr(rr, "call_model", _ok)
    title, summary = await rr.summarize_text("texto")
    assert title == "Manual"
    assert "Guia curto" in summary

    async def _empty(**kwargs):
        """Execute the empty routine.

This helper encapsulates one focused step used by the surrounding workflow."""
        return "", {}

    monkeypatch.setattr(rr, "call_model", _empty)
    title2, summary2 = await rr.summarize_text("texto")
    assert title2 == "Documento Processado"
    assert "indisponível" in summary2


@pytest.mark.asyncio
async def test_add_doc_txt_success(monkeypatch):
    """Testa add doc txt success."""
    async def _summary(_txt):
        """Execute the summary routine.

This helper encapsulates one focused step used by the surrounding workflow."""
        return "Meu título", "Meu resumo"

    inserted = []

    async def _add_document(**kwargs):
        """Execute the add document routine.

This helper encapsulates one focused step used by the surrounding workflow."""
        inserted.append(kwargs)
        return True

    monkeypatch.setattr(rr, "summarize_text", _summary)
    monkeypatch.setattr(rr, "add_document", _add_document)

    up = UploadFile(filename="base.txt", file=BytesIO(b"linha 1\nlinha 2"))
    out = await rr.add_doc(up)
    assert out["file"] == "base.txt"
    assert out["fragments_total"] >= 1
    assert out["fragments_inserted"] == out["fragments_total"]
    assert inserted


@pytest.mark.asyncio
async def test_add_doc_validation_and_empty(monkeypatch):
    """Testa add doc validation and empty."""
    bad = UploadFile(filename="x.csv", file=BytesIO(b"abc"))
    with pytest.raises(HTTPException) as exc1:
        await rr.add_doc(bad)
    assert exc1.value.status_code == 400

    empty = UploadFile(filename="x.txt", file=BytesIO(b"   "))
    with pytest.raises(HTTPException) as exc2:
        await rr.add_doc(empty)
    assert exc2.value.status_code == 400


@pytest.mark.asyncio
async def test_ingest_text_success_and_fail(monkeypatch):
    """Testa ingest text success and fail."""
    req = rr.IngestRequest(text="abc", doc_id="d1", metadata={"a": 1}, collection_name="course_1")

    async def _ok(**kwargs):
        """Execute the ok routine.

This helper encapsulates one focused step used by the surrounding workflow."""
        return True

    monkeypatch.setattr(rr, "add_document", _ok)
    out = await rr.ingest_text(req)
    assert out["status"] == "ok"
    assert req.metadata["target_collection"] == "course_1"

    async def _fail(**kwargs):
        """Execute the fail routine.

This helper encapsulates one focused step used by the surrounding workflow."""
        return False

    monkeypatch.setattr(rr, "add_document", _fail)
    with pytest.raises(HTTPException) as exc:
        await rr.ingest_text(rr.IngestRequest(text="a", doc_id="d2", metadata={}))
    assert exc.value.status_code == 500
