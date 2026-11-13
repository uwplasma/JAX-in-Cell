"""Tracked text must not contain machine paths, personal handles or AI-attribution strings."""

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = {
    "machine path": r"/Users/|/home/(?!runner)[a-z]|/private/tmp|/data\d+/|C:\\Users",
    "personal handle": r"rogeriojorge|rjorge",
    "AI attribution": r"Co-Authored-By|Generated with|Claude|Anthropic|ChatGPT|Codex|Copilot",
}
HANDLE_OK = {"CITATION.cff", "LICENSE", "pyproject.toml", ".zenodo.json"}


def _tracked():
    out = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True
    ).stdout.decode()
    return [p for p in out.split("\0") if p and p != Path(__file__).relative_to(ROOT).as_posix()]


def test_tracked_text_is_clean():
    bad = []
    for rel in _tracked():
        try:
            text = (ROOT / rel).read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        text = re.sub(r"github\.com/rogeriojorge|github\.com/uwplasma", "", text)
        for name, pattern in PATTERNS.items():
            if name == "personal handle" and rel.rsplit("/", 1)[-1] in HANDLE_OK:
                continue
            for m in re.finditer(pattern, text):
                bad.append(f"{rel}: {name}: {m.group(0)}")
    assert not bad, "\n".join(bad[:20])
