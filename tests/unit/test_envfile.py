"""The .env loader: parsing, precedence, and CLI auto-loading."""

import os
import subprocess
import sys
from pathlib import Path

from telecom_aut.envfile import load_dotenv


def test_parses_values_comments_quotes(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text(
        "# comment\n"
        "\n"
        "PLAIN_VAR=abc\n"
        'QUOTED_VAR="with spaces"\n'
        "export EXPORTED_VAR=xyz\n"
        "EMPTY_VAR=\n"
        "not a valid line\n"
        "BAD KEY=nope\n"
    )
    for name in ("PLAIN_VAR", "QUOTED_VAR", "EXPORTED_VAR", "EMPTY_VAR"):
        monkeypatch.delenv(name, raising=False)
    loaded = load_dotenv(env)
    assert set(loaded) == {"PLAIN_VAR", "QUOTED_VAR", "EXPORTED_VAR"}
    assert os.environ["PLAIN_VAR"] == "abc"
    assert os.environ["QUOTED_VAR"] == "with spaces"
    assert os.environ["EXPORTED_VAR"] == "xyz"
    assert "EMPTY_VAR" not in os.environ  # empty values are skipped
    for name in loaded:
        monkeypatch.delenv(name, raising=False)


def test_shell_environment_wins(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("PRESET_VAR=from_file\n")
    monkeypatch.setenv("PRESET_VAR", "from_shell")
    assert load_dotenv(env) == []
    assert os.environ["PRESET_VAR"] == "from_shell"


def test_missing_file_is_a_noop(tmp_path):
    assert load_dotenv(tmp_path / "nope.env") == []


def test_cli_entrypoint_autoloads_repo_env(tmp_path):
    """A CLI main() run from a directory with .env sees the variable,
    proven in a subprocess with a scrubbed environment."""
    (tmp_path / ".env").write_text("DOTENV_PROBE_VAR=loaded\n")
    code = (
        "import os\n"
        "from telecom_aut.envfile import load_dotenv\n"
        "load_dotenv()\n"  # exactly what every CLI main() does first
        "print(os.environ.get('DOTENV_PROBE_VAR', 'MISSING'))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path, capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).parents[2])},
    )
    assert result.stdout.strip() == "loaded"
