"""The command line surface.

These are cheap smoke tests, and they exist because a silent editing mistake
once left `--no-turn` referenced in the code but never registered on the parser.
Nothing caught it: every unit test passed, and the command died with an
AttributeError the first time anyone ran it.

They use only the commands that need no models.
"""

from __future__ import annotations

import pytest

from hearwrite.cli import build_parser, main


def test_version(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    assert "hearwrite" in capsys.readouterr().out


def test_demo_runs_with_no_models(capsys):
    assert main(["demo", "--policy", "dictation"]) == 0
    assert "transcript:" in capsys.readouterr().out


@pytest.mark.parametrize("policy", ["conversation", "dictation", "agent"])
def test_demo_accepts_every_preset(policy, capsys):
    assert main(["demo", "--policy", policy]) == 0
    capsys.readouterr()


def test_demo_json_is_one_event_per_line(capsys):
    assert main(["demo", "--json"]) == 0
    import json

    for line in capsys.readouterr().out.strip().splitlines():
        json.loads(line)


def test_policies_lists_every_preset(capsys):
    assert main(["policies"]) == 0
    out = capsys.readouterr().out
    for name in ("conversation", "dictation", "agent"):
        assert name in out


def test_models_reports_licence_and_checksum(capsys):
    assert main(["models"]) == 0
    out = capsys.readouterr().out
    assert "sha256 pinned" in out
    assert "UNPINNED" not in out, "a registry model has no checksum"


def test_unknown_policy_is_rejected():
    with pytest.raises(SystemExit):
        main(["demo", "--policy", "nonsense"])


@pytest.mark.parametrize(
    "argv",
    [
        ["transcribe", "x.wav"],
        ["transcribe", "x.wav", "--no-vad", "--no-turn"],
        ["transcribe", "x.wav", "--engine", "whisper", "--language", "en"],
        ["transcribe", "x.wav", "--policy", "conversation", "--speaker-model", "m"],
        ["bench", "x.wav", "--engine", "sherpa"],
        ["endpoints", "somewhere", "--turn-model", "smart-turn"],
        ["serve", "--port", "9999", "--max-sessions", "2", "--policy", "agent"],
        ["demo", "--chunk", "0.5", "--json"],
    ],
)
def test_every_flag_the_parser_defines_actually_parses(argv):
    assert build_parser().parse_args(argv) is not None


#: Every attribute the command implementations read off `args`, per command.
#: If a flag is referenced in code but never registered, the command dies with
#: an AttributeError the first time it runs. That has happened.
READS = {
    "transcribe": [
        "path",
        "policy",
        "engine",
        "model",
        "speaker_model",
        "language",
        "chunk",
        "threads",
        "json",
        "no_vad",
        "no_turn",
    ],
    "bench": ["path", "policy", "engine", "model", "speaker_model", "language", "threads"],
    "endpoints": ["path", "turn_model"],
    "serve": ["host", "port", "policy", "model", "max_sessions", "punctuate", "record"],
    "demo": ["policy", "chunk", "json"],
}


@pytest.mark.parametrize("command,attributes", READS.items())
def test_every_attribute_the_code_reads_is_registered(command, attributes):
    """The regression guard for the `--no-turn` bug."""
    argv = [command] + (["x"] if command in {"transcribe", "bench", "endpoints"} else [])
    args = build_parser().parse_args(argv)
    missing = [a for a in attributes if not hasattr(args, a)]
    assert not missing, f"`hearwrite {command}` reads unregistered args: {missing}"


def test_missing_file_is_a_usage_error(capsys):
    assert main(["transcribe", "/nonexistent/file.wav"]) == 2
    assert "no such file" in capsys.readouterr().err


def test_archive_extraction_refuses_an_unsafe_python():
    """No fallback path may extract a downloaded tarball without the filter.

    tarfile's 'data' filter refuses absolute paths, parent traversal and device
    nodes. It arrived in Python 3.11.4, which is why requires-python says
    3.11.4 rather than 3.11. If someone installs past that floor anyway, the
    download must fail loudly rather than quietly extract without protection.
    """
    import tarfile

    from hearwrite import models

    assert hasattr(tarfile, "data_filter"), (
        "this interpreter predates the tarfile security backport"
    )
    source = __import__("inspect").getsource(models._extract)
    assert 'filter="data"' in source
    assert "ModelError" in source, "there is no guard for an interpreter without it"
