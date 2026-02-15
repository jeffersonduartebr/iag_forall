from io import BytesIO
from types import SimpleNamespace

import pytest
from fastapi import HTTPException, UploadFile

from app.routers import rag_router as rr


def test_chunk_text_basic():
    txt = "a " * 1000
    chunks = rr.chunk_text(txt, chunk_size=100, overlap=10)
    assert len(chunks) > 1
    assert all(isinstance(c, str) for c in chunks)


def test_extract_text_from_pdf_error(monkeypatch):
    monkeypatch.setattr(rr.fitz, "open", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("bad pdf")))
    with pytest.raises(HTTPException) as exc:
        rr.extract_text_from_pdf(b"%PDF-invalid")
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_summarize_text_success_and_fallback(monkeypatch):
    async def _ok(**kwargs):
        return "TÍTULO: Manual\nRESUMO: Guia curto.", {}

    monkeypatch.setattr(rr, "call_model", _ok)
    title, summary = await rr.summarize_text("texto")
    assert title == "Manual"
    assert "Guia curto" in summary

    async def _empty(**kwargs):
        return "", {}

    monkeypatch.setattr(rr, "call_model", _empty)
    title2, summary2 = await rr.summarize_text("texto")
    assert title2 == "Documento Processado"
    assert "indisponível" in summary2


@pytest.mark.asyncio
async def test_add_doc_txt_success(monkeypatch):
    async def _summary(_txt):
        return "Meu título", "Meu resumo"

    inserted = []

    async def _add_document(**kwargs):
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
    req = rr.IngestRequest(text="abc", doc_id="d1", metadata={"a": 1}, collection_name="course_1")

    async def _ok(**kwargs):
        return True

    monkeypatch.setattr(rr, "add_document", _ok)
    out = await rr.ingest_text(req)
    assert out["status"] == "ok"
    assert req.metadata["target_collection"] == "course_1"

    async def _fail(**kwargs):
        return False

    monkeypatch.setattr(rr, "add_document", _fail)
    with pytest.raises(HTTPException) as exc:
        await rr.ingest_text(rr.IngestRequest(text="a", doc_id="d2", metadata={}))
    assert exc.value.status_code == 500
