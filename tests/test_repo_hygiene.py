import re
import shutil
import subprocess
from pathlib import Path

import pytest

from core.config import ENV_VARS

ROOT = Path(__file__).resolve().parent.parent
GIT = shutil.which("git")
HAS_REPO = GIT is not None and (ROOT / ".git").exists()

# Looks like a real Groq key. Built from parts so this file never matches itself.
KEY_PATTERN = re.compile("gsk" + "_[A-Za-z0-9]{20,}")


def _env_example_pairs():
    pairs = {}
    for line in (ROOT / ".env.example").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            name, _, value = line.partition("=")
            pairs[name.strip()] = value.strip()
    return pairs


def test_env_example_has_no_credentials_and_documents_every_variable():
    pairs = _env_example_pairs()
    assert pairs["GROQ_API_KEY"] == ""
    assert set(pairs) == set(ENV_VARS)
    assert not KEY_PATTERN.search((ROOT / ".env.example").read_text())


def test_gitignore_excludes_secrets_but_keeps_the_example():
    text = (ROOT / ".gitignore").read_text().splitlines()
    assert ".env" in text
    assert "!.env.example" in text


@pytest.mark.skipif(not HAS_REPO, reason="not a git checkout")
def test_git_ignores_env_files_but_not_the_example():
    def ignored(path):
        return subprocess.run([GIT, "-C", str(ROOT), "check-ignore", "-q", path]).returncode == 0

    assert ignored(".env")
    assert ignored(".env.local")
    assert ignored(".streamlit/secrets.toml")
    assert not ignored(".env.example")


@pytest.mark.skipif(not HAS_REPO, reason="not a git checkout")
def test_no_real_looking_keys_in_any_non_ignored_file():
    out = subprocess.run(
        [GIT, "-C", str(ROOT), "ls-files", "--cached", "--others", "--exclude-standard"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    assert ".env" not in out
    offenders = [
        f
        for f in out
        if (ROOT / f).is_file() and KEY_PATTERN.search((ROOT / f).read_text(errors="ignore"))
    ]
    assert offenders == []
