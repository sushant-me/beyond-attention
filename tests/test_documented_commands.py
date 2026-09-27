"""Every documented command has to record what the file it writes records.

The README's *Reproducing this* block is a recipe, and a recipe that omits a
flag whose value the committed artifact records non-defaultly produces a
different file. That is not hypothetical here: both length-scaling baselines
were measured at `--scaling-batch 4` -- which is the number their tables print
in their own header lines -- while the documented commands left the flag off and
would have measured at the default 8. Nothing failed, because nothing compared
the two.

So this module compares them, and it reads the defaults out of the live argparse
parsers rather than from a table kept here: a default that changes without the
README changing is exactly the drift this is meant to catch, and a remembered
copy of the defaults would not catch it. Liveness has a price -- each script is
imported in a subprocess -- so the parser of each script is read once and cached.

The check is deliberately one-directional. It asks whether a documented command
*can* produce the committed configuration; it does not re-run anything, so it
says nothing about whether the numbers still agree. Re-running is the job of the
renderer comparison in `tests/test_render_readme.py` and of whatever produced
the file. What it catches is the cheaper mistake: the published recipe quietly
diverging from the published artifact.

It counts what it checked. A README whose bash blocks stopped being recognised
would otherwise pass by matching nothing.
"""

from __future__ import annotations

import json
import pathlib
import re
import shlex
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
README = ROOT / "README.md"

# Running a script until it builds its parser, then refusing to go further. The
# exception is what stops it: the script is never allowed to do any work, so no
# measurement is started and no output file is touched.
_EXTRACT_DEFAULTS = r"""
import argparse, json, os, runpy, sys

# The experiments import each other by bare name, so the directory has to be on
# the path the way it is when one of them is run as a script.
sys.path.insert(0, os.path.dirname(os.path.abspath(sys.argv[1])))

captured = {}

class _Stop(Exception):
    pass

def _capture(self, *args, **kwargs):
    captured["parser"] = self
    raise _Stop

original = argparse.ArgumentParser.parse_args
argparse.ArgumentParser.parse_args = _capture
try:
    runpy.run_path(sys.argv[1], run_name="__main__")
except _Stop:
    pass
except SystemExit:
    pass
finally:
    argparse.ArgumentParser.parse_args = original

parser = captured.get("parser")
if parser is None:
    print(json.dumps({"__error__": "the script built no parser"}))
else:
    print(json.dumps({a.dest: a.default for a in parser._actions
                      if a.dest != "help"}, default=str))
"""

_defaults_cache: dict[str, dict] = {}

# Each extraction imports the script, which for most of these means importing
# torch -- about four seconds, and almost all of it is that one import. Run in
# parallel, but not all at once: every worker holds a torch in memory, and the
# point of this guard is not to be the reason a CI runner runs out of it.
_EXTRACTION_WORKERS = 4


def _run_documentation_commands() -> list[str]:
    """Every command the README shows, with line continuations joined.

    The comments the README puts after a command are dropped: a flag named in a
    comment must not count as the flag being passed.
    """
    commands: list[str] = []
    pending = ""
    for line in README.read_text().splitlines():
        stripped = line.rstrip()
        if stripped.endswith("\\"):
            pending += " " + stripped[:-1].strip()
            continue
        if pending:
            commands.append((pending + " " + stripped).strip())
            pending = ""
        elif stripped.strip().startswith("python "):
            commands.append(stripped.strip())
    return [c.split(" #")[0].strip() for c in commands]


def _tokens(command: str) -> list[str]:
    return shlex.split(command)


def _extract(script: pathlib.Path) -> tuple[str, dict]:
    proc = subprocess.run(
        [sys.executable, "-c", _EXTRACT_DEFAULTS, str(script)],
        capture_output=True, text=True, timeout=300,
    )
    lines = (proc.stdout or "").strip().splitlines()
    assert lines, (
        f"reading the argparse defaults of {script} produced nothing; "
        f"stderr: {(proc.stderr or '').strip()[-400:]}"
    )
    payload = json.loads(lines[-1])
    assert "__error__" not in payload, f"{script}: {payload['__error__']}"
    return script.name, payload


def _load_defaults(scripts: list[pathlib.Path]) -> None:
    """Fill the cache for every script, a few subprocesses at a time."""
    pending = [s for s in scripts if s.name not in _defaults_cache]
    if not pending:
        return
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(
            max_workers=_EXTRACTION_WORKERS) as pool:
        for name, payload in pool.map(_extract, pending):
            _defaults_cache[name] = payload


def _defaults_of(script: pathlib.Path) -> dict:
    _load_defaults([script])
    return _defaults_cache[script.name]


def test_every_documented_command_records_its_committed_configuration() -> None:
    """A flag the artifact was written with is not left out of the recipe."""
    checked_commands: list[tuple[str, list[str], pathlib.Path, dict]] = []

    for command in _run_documentation_commands():
        tokens = _tokens(command)
        scripts = [t for t in tokens if t.startswith("experiments/")
                   and t.endswith(".py")]
        if not scripts:
            continue
        target = None
        for index, token in enumerate(tokens):
            if token == "--out" and index + 1 < len(tokens):
                target = tokens[index + 1]
            elif token.startswith("--out="):
                target = token.split("=", 1)[1]
        if target is None:
            continue
        path = ROOT / target
        if not path.is_file() or path.suffix != ".json":
            continue
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        config = payload.get("config")
        if not isinstance(config, dict) or not config:
            continue
        checked_commands.append(
            (target, tokens, ROOT / scripts[0], config))

    _load_defaults([script for _, _, script, _ in checked_commands])

    mismatches: list[str] = []
    for target, tokens, script, config in checked_commands:
        defaults = _defaults_cache[script.name]
        for key, value in config.items():
            if key not in defaults:
                continue
            if str(value) == str(defaults[key]):
                continue
            flag = "--" + key.replace("_", "-")
            if flag not in tokens:
                mismatches.append(
                    f"{target}: records {key}={value!r} but the documented "
                    f"command leaves it at the default {defaults[key]!r} "
                    f"({flag} is not passed)"
                )

    # A floor, not a target: the point is that a README this file can no longer
    # parse fails here instead of passing vacuously.
    assert len(checked_commands) >= 12, (
        f"only {len(checked_commands)} documented commands were matched to a "
        f"committed JSON config; the README's command blocks are probably not "
        f"being parsed"
    )
    assert not mismatches, "\n".join(mismatches)


def test_the_guard_notices_a_flag_that_is_actually_missing() -> None:
    """Remove a flag the README passes and the check above has to complain.

    A guard that cannot fail is documentation. This drives the same comparison
    with one `--scaling-batch 4` deleted from a copy of the commands, and with
    the floor lowered to where the copy still clears it.
    """
    commands = _run_documentation_commands()
    assert any("--scaling-batch" in c for c in commands), (
        "no documented command passes --scaling-batch; this test is aimed at a "
        "flag the README no longer carries, so it can no longer fail"
    )

    stripped = [c.replace("--scaling-batch 4", "").replace("--scaling-batch 2", "")
                for c in commands]
    assert stripped != commands, "the flag was not actually removed"

    targets = {
        "scaling-final.json": "experiments/run.py",
        "scaling-mask.json": "experiments/run.py",
    }
    failures = []
    for command in stripped:
        tokens = _tokens(command)
        if "--scaling-batch" in tokens:
            continue
        for target, script in targets.items():
            if f"--out {target}" not in command:
                continue
            config = json.loads((ROOT / target).read_text())["config"]
            defaults = _defaults_of(ROOT / script)
            if str(config["scaling_batch"]) != str(defaults["scaling_batch"]):
                failures.append(target)

    assert sorted(failures) == ["scaling-final.json", "scaling-mask.json"], (
        f"with --scaling-batch removed the comparison only flagged {failures}; "
        f"it is no longer sensitive to the mistake it was written for"
    )
