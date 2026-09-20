# Objective: db_init must fail when the migration fails.
"""``service_completed_successfully`` is only worth anything if failure exits non-zero.

The command joined its steps with ``;`` and ended with an ``echo``. The exit
code of ``bash -c`` is the exit code of the *last* command, and an ``echo``
always returns 0 — so ``alembic upgrade head`` could fail, the container would
print "✅ Banco pronto!", exit 0, and the ``db_init: condition:
service_completed_successfully`` that the API depends on was satisfied anyway.
The API then started against a half-migrated schema, and the inserts that
needed the new columns failed at runtime, swallowed by ``persist_log``.

These tests pin the shell semantics rather than the exact text, because the
property is about exit codes, not about how the command is spelled.
"""

import shutil
import subprocess

import pytest
import yaml

COMPOSE = "docker-compose.yml"


@pytest.fixture(scope="module")
def db_init_command():
    with open(COMPOSE, encoding="utf-8") as handle:
        return yaml.safe_load(handle)["services"]["db_init"]["command"]


def test_the_steps_are_chained_not_sequenced(db_init_command):
    """``;`` runs the next step whatever the previous one did."""
    assert "&&" in db_init_command
    body = db_init_command.split('"', 1)[1] if '"' in db_init_command else db_init_command
    assert ";" not in body


def test_the_shell_exits_on_the_first_error(db_init_command):
    assert db_init_command.strip().startswith("bash -ec") or "set -e" in db_init_command


def test_both_migration_steps_are_still_run(db_init_command):
    assert "alembic upgrade head" in db_init_command
    assert "python -m app.db_manager" in db_init_command


def test_the_api_still_waits_for_it():
    """The fix is worthless if nothing consumes the exit code."""
    with open(COMPOSE, encoding="utf-8") as handle:
        compose = yaml.safe_load(handle)
    waiters = [
        name
        for name, svc in compose["services"].items()
        if (svc.get("depends_on") or {}).get("db_init", {}).get("condition")
        == "service_completed_successfully"
    ]
    assert waiters, "ninguém espera pelo db_init; o código de saída não é lido"


# ---------------------------------------------------------------------------
# The shell semantics themselves
# ---------------------------------------------------------------------------


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash indisponível")
class TestShellSemantics:
    """Runs the two spellings against a real bash, with a failing middle step."""

    @staticmethod
    def run(script: str) -> int:
        return subprocess.run(["bash", "-c", script], capture_output=True).returncode

    def test_the_old_spelling_hid_the_failure(self):
        """This is the defect, reproduced: exit 0 with a failed step."""
        assert self.run("echo a; false; echo b; echo pronto;") == 0

    def test_the_new_spelling_propagates_it(self):
        assert self.run("bash -ec 'echo a && false && echo b && echo pronto'") != 0

    def test_the_new_spelling_still_succeeds_when_everything_works(self):
        assert self.run("bash -ec 'echo a && true && echo b && echo pronto'") == 0
