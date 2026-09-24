# Objective: Small value types embedded in QueryRequest (kept apart so schemas.py stays under its SLOC ceiling).
"""Value types embedded in ``QueryRequest``."""

from __future__ import annotations

from typing import Annotated, Dict, Optional, Union

from pydantic import BaseModel, Field

#: One metadata key of a RAG scope. Lowercase identifiers only, so a client can never smuggle a
#: Chroma operator (``$and``, ``$in``...) into the ``where`` clause built by ``services.rag_scope``.
RagFilterKey = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")]

#: Flat metadata equalities; ``tenant_id`` is always overridden by the caller's resolved tenant.
RagFilter = Annotated[
    Optional[Dict[RagFilterKey, Union[str, int, float, bool]]],
    Field(
        max_length=8,
        description=(
            "Escopo do RAG: igualdades de metadados (ex.: {disciplina, periodo}) combinadas com o tenant. "
            "Com escopo, enable_rag_for_answer é honrado sem bypass heurístico (exceto com `messages`)."
        ),
    ),
]


class WorkloadHints(BaseModel):
    """Optional client hints; complexity is always detected at runtime."""
    theme: Annotated[Optional[str], Field(max_length=128, description="Benchmark or domain theme for telemetry.")] = None
    benchmark_id: Annotated[Optional[str], Field(max_length=128, description="Catalog entry id when known.")] = None
    expected_tokens: Annotated[Optional[int], Field(ge=32, le=32000, description="Desired response length floor.")] = None
