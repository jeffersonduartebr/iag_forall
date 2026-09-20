# Objective: Tests for the human anchoring CLI that turns pre-classifications into the golden ground truth.
"""The review CLI produces the ground truth, so its bookkeeping has to be exact.

What matters is that a decision is never lost (the state is saved after each
one), never invented (an item the human skipped must not appear in the ground
truth), and that the final report separates what the human accepted from what
they overrode.
"""

import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


from scripts.benchmark_v2 import review_cli  # noqa: E402

ITEMS = {
    "a": {
        "id": "a", "query": "Quanto e 15% de 180?", "discipline": "matematica", "topic": "porcentagem",
        "partition": "canonical", "steps_required": 1, "complexity_gold": "BAIXA", "requires_rag": False,
    },
    "b": {
        "id": "b", "query": "Texto longo " * 200, "discipline": "matematica", "topic": "porcentagem",
        "partition": "verbosity_trap", "base_id": "a", "steps_required": 1, "complexity_gold": "BAIXA",
        "requires_rag": False,
    },
}


def verdict(item_id, model, label):
    return {
        "item_id": item_id,
        "model": model,
        "rotulo_complexidade_final": label,
        "dependencia_ancoragem": "NAO",
        "analise_inferencia": "um passo",
        "analise_distratores": "nenhum",
    }


VERDICTS = {
    "a": [verdict("a", "m1", "BAIXA"), verdict("a", "m2", "BAIXA")],
    "b": [verdict("b", "m1", "ALTA"), verdict("b", "m2", "MEDIA")],
}


def scripted(*answers):
    """A reader that replays the given keystrokes."""
    remaining = list(answers)

    def _read(prompt=""):
        if not remaining:
            raise EOFError
        return remaining.pop(0)

    return _read


# ---------------------------------------------------------------------------
# Proposed label
# ---------------------------------------------------------------------------


def test_the_proposal_is_the_label_both_classifiers_agree_on():
    assert review_cli.proposed_label(VERDICTS["a"], ITEMS["a"]) == "BAIXA"


def test_a_contested_label_proposes_the_median_of_the_proposals():
    assert review_cli.proposed_label(VERDICTS["b"], ITEMS["b"]) == "ALTA"


def test_without_any_verdict_the_generator_label_is_proposed():
    assert review_cli.proposed_label([], ITEMS["a"]) == "BAIXA"


def test_a_failed_verdict_does_not_count_as_a_proposal():
    failed = [{"item_id": "a", "model": "m1", "rotulo_complexidade_final": None, "error": "timeout"}]
    assert review_cli.proposed_label(failed, ITEMS["a"]) == "BAIXA"


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


def run_session(keys, queue=("a", "b"), state=None, tmp_path=None):
    path = tmp_path / "state.json"
    return review_cli.review_session(
        list(queue), ITEMS, VERDICTS, state or {"decisions": {}}, path, reader=scripted(*keys)
    ), path


def test_enter_accepts_the_proposed_label(tmp_path):
    state, _ = run_session(["", ""], tmp_path=tmp_path)
    assert state["decisions"]["a"] == {"label": "BAIXA", "source": "aceito"}
    assert state["decisions"]["b"]["label"] == "ALTA"


def test_a_key_overrides_the_proposal(tmp_path):
    state, _ = run_session(["m", "b"], tmp_path=tmp_path)
    assert state["decisions"]["a"] == {"label": "MEDIA", "source": "humano"}
    assert state["decisions"]["b"] == {"label": "BAIXA", "source": "humano"}


def test_skipping_leaves_no_decision(tmp_path):
    state, _ = run_session(["s", "s"], tmp_path=tmp_path)
    assert state["decisions"] == {}


def test_quitting_stops_the_queue_and_saves(tmp_path):
    state, path = run_session(["b", "q"], tmp_path=tmp_path)
    assert list(state["decisions"]) == ["a"]
    assert json.loads(path.read_text(encoding="utf-8"))["decisions"]["a"]["label"] == "BAIXA"


def test_flagging_records_a_broken_item(tmp_path):
    state, _ = run_session(["f", "s"], tmp_path=tmp_path)
    assert state["decisions"]["a"]["flagged"] is True
    assert state["decisions"]["a"]["source"] == "defeituoso"


def test_a_note_is_attached_and_the_item_stays_open(tmp_path):
    # 'n' asks for the note text, then the item is presented again.
    state, _ = run_session(["n", "enunciado ambiguo", "a", "s"], tmp_path=tmp_path)
    assert state["decisions"]["a"]["note"] == "enunciado ambiguo"
    assert state["decisions"]["a"]["label"] == "ALTA"


def test_an_unknown_key_reprompts_without_deciding(tmp_path):
    state, _ = run_session(["z", "b", "s"], tmp_path=tmp_path)
    assert state["decisions"]["a"]["label"] == "BAIXA"


def test_the_state_is_written_after_every_decision(tmp_path):
    _, path = run_session(["b", "q"], tmp_path=tmp_path)
    assert path.exists()


def test_an_interrupted_session_keeps_what_was_decided(tmp_path):
    # The reader runs out of answers, which raises EOFError mid-session.
    state, path = run_session([""], tmp_path=tmp_path)
    assert list(state["decisions"]) == ["a"]
    assert json.loads(path.read_text(encoding="utf-8"))["decisions"]["a"]


# ---------------------------------------------------------------------------
# Queue
# ---------------------------------------------------------------------------


def records_for_queue():
    return [v for verdicts in VERDICTS.values() for v in verdicts]


def test_the_queue_puts_the_contested_item_first():
    queue = review_cli.build_queue(records_for_queue(), ITEMS, ["m1", "m2"], {}, only_conflicts=False)
    assert queue[0] == "b"


def test_the_queue_skips_items_already_decided():
    queue = review_cli.build_queue(
        records_for_queue(), ITEMS, ["m1", "m2"], {"b": {"label": "ALTA"}}, only_conflicts=False
    )
    assert "b" not in queue


def test_only_conflicts_drops_the_routine_items():
    queue = review_cli.build_queue(records_for_queue(), ITEMS, ["m1", "m2"], {}, only_conflicts=True)
    assert queue == ["b"]


# ---------------------------------------------------------------------------
# Ground truth and report
# ---------------------------------------------------------------------------


def test_ground_truth_holds_only_reviewed_items(tmp_path):
    out = tmp_path / "gt.jsonl"
    decisions = {"a": {"label": "MEDIA", "source": "humano"}}
    written = review_cli.write_ground_truth(out, ITEMS, decisions)
    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line]
    assert written == 1
    assert [r["id"] for r in records] == ["a"]
    assert records[0]["complexity_human"] == "MEDIA"
    assert records[0]["review_source"] == "humano"


def test_ground_truth_keeps_the_original_item_fields(tmp_path):
    out = tmp_path / "gt.jsonl"
    review_cli.write_ground_truth(out, ITEMS, {"a": {"label": "BAIXA", "source": "aceito"}})
    record = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert record["query"] == ITEMS["a"]["query"]
    assert record["complexity_gold"] == "BAIXA"


def test_the_report_separates_accepted_from_overridden():
    decisions = {
        "a": {"label": "BAIXA", "source": "aceito"},
        "b": {"label": "BAIXA", "source": "humano"},
    }
    report = review_cli.final_report(ITEMS, VERDICTS, decisions)
    assert report["reviewed"] == 2
    assert report["overridden"] == 1
    assert report["labels"] == {"BAIXA": 2}


def test_the_report_gives_a_kappa_per_partition():
    """Agreement on canonical items versus traps is the trap partition's result."""
    decisions = {
        "a": {"label": "BAIXA", "source": "aceito"},
        "b": {"label": "BAIXA", "source": "humano"},
    }
    report = review_cli.final_report(ITEMS, VERDICTS, decisions)
    assert set(report["kappa_human_vs_llm"]) == {"canonical", "verbosity_trap"}


def test_the_report_counts_flagged_items():
    decisions = {"a": {"label": "BAIXA", "source": "defeituoso", "flagged": True}}
    assert review_cli.final_report(ITEMS, VERDICTS, decisions)["flagged"] == 1


# ---------------------------------------------------------------------------
# Presentation
# ---------------------------------------------------------------------------


def test_a_long_statement_is_shortened_but_keeps_both_ends():
    text = "INICIO " + ("x " * 900) + "FIM"
    shortened = review_cli.shorten(text)
    assert shortened.startswith("INICIO")
    assert shortened.rstrip().endswith("FIM")
    assert len(shortened) < len(text)


def test_a_short_statement_is_left_alone():
    assert review_cli.shorten("curto") == "curto"


def test_the_screen_shows_both_verdicts_and_the_generator_depth():
    screen = review_cli.render(ITEMS["b"], VERDICTS["b"], "[1/2]", expand=False)
    assert "m1" in screen and "m2" in screen
    assert "verbosity_trap" in screen
    assert "passos do gerador: 1" in screen
