# Objective: Exact-value tests for the rubric parser, weights and judge orchestration.
"""services.judge_rubric: parsing edge cases, weights, scoring mode and the judge loop.

Complements test_judge_rubric.py (end-to-end through judges.py) by calling the
helpers directly, so mutation testing sees each comparison and constant pinned.
"""

import pytest
from app.services import judge_rubric as jr

DIMS = jr.RUBRIC_DIMENSIONS


def _rating(c, a, al):
    return {"clareza": c, "acuracia": a, "alinhamento": al}


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("Clareza", "clareza"),
        (" Acurácia Conceitual ", "acuracia"),
        ("_alinhamento_", "alinhamento"),
        ("Alinhamento-Pedagógico", "alinhamento"),
        ("clarity", "clareza"),
        ("fluencia", None),
        (3, None),
    ],
)
def test_canonical_key(key, expected):
    assert jr._canonical_key(key) == expected


def test_extract_json_prefers_scores_tag_and_skips_invalid_blocks():
    text = 'antes {"x": } <scores>{"clareza": 1, "acuracia": 2, "alinhamento": 3}</scores> {"clareza": 9}'
    assert jr._extract_json_object(text) == {"clareza": 1, "acuracia": 2, "alinhamento": 3}
    assert jr._extract_json_object('{"a": } e depois {"clareza": 7}') == {"clareza": 7}
    assert jr._extract_json_object("[1, 2]") is None
    assert jr._extract_json_object("sem json") is None


def test_parse_rubric_scores_skips_bad_values_and_nan():
    text = '{"clareza": "oito", "clareza_coesao": 8, "acuracia": NaN, "accuracy": 11, "alinhamento": -2, "extra": 5}'
    assert jr.parse_rubric_scores(text) == _rating(8.0, 10.0, 0.0)
    assert jr.parse_rubric_scores('{"clareza": 5, "acuracia": NaN, "alinhamento": 5}') is None
    assert jr.parse_rubric_scores("") is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ({"clareza": 0.2, "acuracia": 0.3, "alinhamento": 0.5}, {"clareza": 0.2, "acuracia": 0.3, "alinhamento": 0.5}),
        ('{"clareza": 0, "acuracia": 1, "alinhamento": 1}', {"clareza": 0.0, "acuracia": 1.0, "alinhamento": 1.0}),
        (
            '{"peso": "x", "clareza": 1, "acuracia": 1, "alinhamento": 2}',
            {"clareza": 1.0, "acuracia": 1.0, "alinhamento": 2.0},
        ),
        (
            '{"extra": 5, "clareza": 1, "acuracia": 1, "alinhamento": 1}',
            {"clareza": 1.0, "acuracia": 1.0, "alinhamento": 1.0},
        ),
        ('{"clareza": -1, "acuracia": 1, "alinhamento": 1}', jr.DEFAULT_RUBRIC_WEIGHTS),  # negativo descartado
        ('{"clareza": 0, "acuracia": 0, "alinhamento": 0}', jr.DEFAULT_RUBRIC_WEIGHTS),  # soma zero
        ('{"clareza": 1, "acuracia": 1}', jr.DEFAULT_RUBRIC_WEIGHTS),  # falta dimensão
        ("{quebrado", jr.DEFAULT_RUBRIC_WEIGHTS),
        (b'{"clareza": 1, "acuracia": 1, "alinhamento": 1}', {"clareza": 1.0, "acuracia": 1.0, "alinhamento": 1.0}),
        (42, jr.DEFAULT_RUBRIC_WEIGHTS),
        (None, jr.DEFAULT_RUBRIC_WEIGHTS),
    ],
)
def test_parse_rubric_weights(raw, expected):
    assert jr.parse_rubric_weights(raw) == expected


def test_weighted_quality_normalizes_by_total_and_falls_back_on_zero():
    scores = _rating(10.0, 5.0, 0.0)
    assert jr.weighted_quality(scores, {"clareza": 0.1, "acuracia": 0.1, "alinhamento": 0.3}) == pytest.approx(3.0)
    assert jr.weighted_quality(scores, {"clareza": 1.0, "acuracia": 1.0, "alinhamento": 0.0}) == pytest.approx(7.5)
    assert jr.weighted_quality(scores, dict.fromkeys(DIMS, 0.0)) == pytest.approx(0.3 * 10 + 0.5 * 5)


def test_combine_ratings_exact_payload():
    out = jr.combine_ratings([_rating(8, 6, 4), None, _rating(6, 8, 8)], jr.DEFAULT_RUBRIC_WEIGHTS)
    assert out["dimensions"] == {"clareza": 7.0, "acuracia": 7.0, "alinhamento": 6.0}
    assert out["quality"] == pytest.approx(0.3 * 7 + 0.5 * 7 + 0.2 * 6)
    assert out["judge_quality"] == pytest.approx([6.2, 7.4])
    assert out["dispersion"]["quality_std"] == pytest.approx(0.6)
    assert out["dispersion"]["by_dimension"] == {"clareza": 1.0, "acuracia": 1.0, "alinhamento": 2.0}
    assert (out["n_judges"], out["aggregate"]) == (2, "mean")

    single = jr.combine_ratings([_rating(5, 5, 5)], jr.DEFAULT_RUBRIC_WEIGHTS, aggregate="median")
    assert single["dispersion"] == {"quality_std": 0.0, "by_dimension": dict.fromkeys(DIMS, 0.0)}
    assert single["aggregate"] == "median"
    assert jr.combine_ratings([None, {}], jr.DEFAULT_RUBRIC_WEIGHTS) is None


def test_build_rubric_prompt_blocks():
    plain = jr.build_rubric_prompt("Q?", "A.")
    assert "PERGUNTA: Q?" in plain and "RESPOSTA DO MODELO: A." in plain
    assert "GABARITO" not in plain and "RAG" not in plain and "IMAGEM" not in plain
    assert "Avalie a precisão factual e lógica." in plain
    assert '{"clareza": <0-10>, "acuracia": <0-10>, "alinhamento": <0-10>}' in plain

    full = jr.build_rubric_prompt("Q?", "A.", reference="42", rag_context="ctx", image_description="gato")
    assert "GABARITO OFICIAL (GROUND TRUTH): 42" in full
    assert "Compare com o GABARITO OFICIAL" in full
    assert "CONTEXTO ADICIONAL (RAG):\nctx" in full and "DESCRIÇÃO DA IMAGEM:\ngato" in full


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, "rubric"),
        ("binary", "binary"),
        (" BINARY ", "binary"),
        ("Rubric", "rubric"),
        ("outro", "rubric"),
        ("", "rubric"),
    ],
)
def test_judge_scoring_mode(value, expected):
    class _S:
        def get(self, key, default=None):
            assert key == "JUDGE_SCORING_MODE" and default == "rubric"
            return value

    assert jr.judge_scoring_mode(_S()) == expected


def test_judge_scoring_mode_defaults_on_error():
    class _Broken:
        def get(self, *a):
            raise RuntimeError("redis down")

    assert jr.judge_scoring_mode(_Broken()) == "rubric"


@pytest.mark.asyncio
async def test_rate_with_rubric_success_failure_and_unparseable():
    calls = []

    async def call_model(**kwargs):
        calls.append(kwargs)
        return '{"clareza": 7, "acuracia": 8, "alinhamento": 9}', {"latency": 1.2}

    rating, meta = await jr.rate_with_rubric(call_model, "j1", "prompt", 0.0, 256)
    assert rating == _rating(7.0, 8.0, 9.0) and meta == {"latency": 1.2}
    assert calls == [{"model": "j1", "prompt": "prompt", "temperature": 0.0, "max_tokens": 256}]

    async def garbled(**kwargs):
        return "sem notas", "meta-invalida"

    assert await jr.rate_with_rubric(garbled, "j1", "p", 0.0, 10) == (None, {})

    async def broken(**kwargs):
        raise TimeoutError

    assert await jr.rate_with_rubric(broken, "j1", "p", 0.0, 10) == (None, {})


def _rater(table):
    async def rate(model):
        return table.get(model, (None, {}))

    return rate


@pytest.mark.asyncio
async def test_score_with_rubric_agreement_keeps_mean_without_meta():
    seen, meta_calls = [], []
    rate = _rater({"a": (_rating(8, 8, 8), {"i": 1}), "b": (_rating(6, 6, 6), {"i": 2})})
    out = await jr.score_with_rubric(
        ["a", "b"],
        rate,
        jr.DEFAULT_RUBRIC_WEIGHTS,
        meta_model=lambda: meta_calls.append(1) or "meta",
        disagreement=2.0,  # ΔQ = 2 não é > 2
        on_rating=lambda m, r, info: seen.append((m, info)),
    )
    assert out["judges"] == ["a", "b"] and out["aggregate"] == "mean"
    assert out["quality"] == pytest.approx(7.0)
    assert seen == [("a", {"i": 1}), ("b", {"i": 2})] and meta_calls == []


@pytest.mark.asyncio
async def test_score_with_rubric_disagreement_uses_meta_median():
    rate = _rater({"a": (_rating(9, 9, 9), {}), "b": (_rating(2, 2, 2), {}), "meta": (_rating(4, 5, 6), {})})
    seen = []
    out = await jr.score_with_rubric(
        ["a", "b"], rate, jr.DEFAULT_RUBRIC_WEIGHTS, meta_model=lambda: "meta", on_rating=lambda m, r, i: seen.append(m)
    )
    assert out["judges"] == ["a", "b", "meta"] and out["aggregate"] == "median"
    assert out["dimensions"] == {"clareza": 4.0, "acuracia": 5.0, "alinhamento": 6.0}
    assert seen == ["a", "b", "meta"]


@pytest.mark.asyncio
async def test_score_with_rubric_meta_failure_and_edge_cases():
    rate = _rater({"a": (_rating(9, 9, 9), {}), "b": (_rating(2, 2, 2), {})})
    out = await jr.score_with_rubric(["a", "b"], rate, jr.DEFAULT_RUBRIC_WEIGHTS, meta_model=lambda: "meta")
    assert out["judges"] == ["a", "b"] and out["aggregate"] == "mean"  # meta falhou: fica a média

    no_meta = await jr.score_with_rubric(["a", "b"], rate, jr.DEFAULT_RUBRIC_WEIGHTS)
    assert no_meta["aggregate"] == "mean"

    one = await jr.score_with_rubric(["a", "x"], rate, jr.DEFAULT_RUBRIC_WEIGHTS, meta_model=lambda: "b")
    assert one["judges"] == ["a"] and one["n_judges"] == 1

    assert await jr.score_with_rubric(["x", "y"], rate, jr.DEFAULT_RUBRIC_WEIGHTS) is None
