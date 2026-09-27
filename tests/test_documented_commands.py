"""Every documented command has to record what the file it writes records.

The README's *Reproducing this* block is a recipe, and a recipe that omits a
flag whose value the committed artifact records non-defaultly -- or that passes
the right flag with the wrong value -- produces a different file. That is not
hypothetical here: both length-scaling baselines were measured at
`--scaling-batch 4`, which is the number their tables print in their own header
lines, while the documented commands left the flag off and would have measured
at the default 8. Nothing failed, because nothing compared the two.

So this module compares them, and it reads the defaults out of the live argparse
parsers rather than from a table kept here: a default that changes without the
README changing is exactly the drift this is meant to catch, and a remembered
copy of the defaults would not catch it. Liveness has a price -- each script is
imported in a subprocess -- so the parser of each script is read once, a few at a
time, and cached.

The check is deliberately one-directional. It asks whether a documented command
*can* produce the committed configuration; it does not re-run anything, so it
says nothing about whether the numbers still agree. Re-running is the job of the
renderer comparison in `tests/test_render_readme.py` and of whatever produced the
file. What it catches is the cheaper mistake: the published recipe quietly
diverging from the published artifact.

It counts what it checked. A README whose bash blocks stopped being recognised
would otherwise pass by matching nothing, so the comparison returns the number of
commands it matched and the test asserts a floor before asserting the result.
The three mutation tests below drive that same comparison rather than restating
its logic, so a guard that stopped being sensitive fails here too.
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


def _flags_and_values(tokens: list[str]) -> dict[str, list[str]]:
    """flag -> the value tokens that follow it, empty for a store_true flag.

    Long flags only: the commands are launched with `python` and sometimes `-u`,
    and neither takes a value that could be mistaken for one. Flag values are
    the tokens up to the next `--`, which is how argparse reads them too, so a
    list flag such as `--pairs 2 4 8 16` collects all four.
    """
    found: dict[str, list[str]] = {}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token.startswith("--") and "=" in token:
            flag, _, value = token.partition("=")
            found.setdefault(flag, []).append(value)
        elif token.startswith("--"):
            values: list[str] = []
            index += 1
            while index < len(tokens) and not tokens[index].startswith("--"):
                values.append(tokens[index])
                index += 1
            found.setdefault(token, []).extend(values)
            continue
        index += 1
    return found


def _same_value(command_values: list[str], recorded) -> bool:
    """Does what the command passes match what the file recorded?

    Compared as text, because the recorded value has already been through
    argparse's `type` and the command line has not -- but numbers are compared
    as numbers, so a recipe may write `5e-3` where the file records `0.005`.
    """
    expected = recorded if isinstance(recorded, list) else [recorded]
    if len(command_values) != len(expected):
        return False
    for given, wanted in zip(command_values, expected):
        if str(given) == str(wanted):
            continue
        try:
            if float(given) == float(wanted):
                continue
        except (TypeError, ValueError):
            pass
        return False
    return True


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


def compare(commands: list[str]) -> tuple[int, list[str]]:
    """Match each command to the committed file it writes; report disagreements.

    Returns how many commands could be matched -- the caller asserts a floor on
    it, so that a README this can no longer parse fails loudly -- and the list of
    places where a command would not reproduce the file it points at.
    """
    checked: list[tuple[str, list[str], pathlib.Path, dict]] = []

    for command in commands:
        tokens = _tokens(command)
        scripts = [t for t in tokens if t.startswith("experiments/")
                   and t.endswith(".py")]
        if not scripts:
            continue
        passed = _flags_and_values(tokens)
        target = (passed.get("--out") or [None])[0]
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
        checked.append((target, tokens, ROOT / scripts[0], config))

    _load_defaults([script for _, _, script, _ in checked])

    mismatches: list[str] = []
    for target, tokens, script, config in checked:
        defaults = _defaults_cache[script.name]
        passed = _flags_and_values(tokens)
        for key, value in config.items():
            if key not in defaults:
                continue
            default = defaults[key]
            flag = "--" + key.replace("_", "-")

            # A store_true flag has no value: its default is a bool, and what
            # the file records is whether the flag was given at all.
            if isinstance(default, bool):
                if bool(value) != (flag in passed):
                    given = "passes" if flag in passed else "does not pass"
                    mismatches.append(
                        f"{target}: records {key}={value!r} but the documented "
                        f"command {given} {flag}")
                continue

            if flag not in passed:
                # Absent, so the file has to agree with the default.
                if str(value) != str(default):
                    mismatches.append(
                        f"{target}: records {key}={value!r} but the documented "
                        f"command leaves it at the default {default!r} "
                        f"({flag} is not passed)")
                continue

            # Present: present with the right value, or the recipe still does
            # not reproduce the file. `--scaling-batch 8` against a file that
            # records 4 is the same defect as omitting the flag.
            if not _same_value(passed[flag], value):
                mismatches.append(
                    f"{target}: records {key}={value!r} but the documented "
                    f"command passes {flag} {' '.join(passed[flag])!r}")

    return len(checked), mismatches


def test_every_documented_command_records_its_committed_configuration() -> None:
    """A flag the artifact was written with is not left out, or given wrongly."""
    checked, mismatches = compare(_run_documentation_commands())
    assert checked >= 12, (
        f"only {checked} documented commands were matched to a committed JSON "
        f"config; the README's command blocks are probably not being parsed"
    )
    assert not mismatches, "\n".join(mismatches)


def test_the_reader_understands_the_commands_it_is_given() -> None:
    """The parsing the comparison rests on, checked directly.

    If `_flags_and_values` stopped collecting list values, or `_same_value`
    started accepting everything, the tests below would still pass while the main
    comparison quietly checked less.
    """
    passed = _flags_and_values(
        _tokens("python experiments/run.py --pairs 2 4 8 16 --steps 3000 "
                "--skip-sweep --out=results.json --scaling-batch 4"))
    assert passed["--pairs"] == ["2", "4", "8", "16"]
    assert passed["--steps"] == ["3000"]
    assert passed["--skip-sweep"] == []
    assert passed["--out"] == ["results.json"]
    assert _same_value(["4"], 4)
    assert _same_value(["5e-3"], 0.005)
    assert _same_value(["attention"], ["attention"])
    assert not _same_value(["8"], 4)
    assert not _same_value(["2", "4"], [2, 4, 8, 16])


def test_the_guard_notices_a_flag_that_is_actually_missing() -> None:
    """Delete every `--scaling-batch` and the comparison has to complain.

    A guard that cannot fail is documentation. This is the mistake that was
    actually there: both scaling baselines printed batch=4 in their headers while
    the documented commands would have measured at the default 8.
    """
    commands = _run_documentation_commands()
    stripped = [re.sub(r"--scaling-batch \d+", "", c) for c in commands]
    assert stripped != commands, "no command carried --scaling-batch to remove"

    _, mismatches = compare(stripped)
    named = {m.split(":")[0] for m in mismatches if "--scaling-batch is not passed" in m}
    assert named == {"results.json", "scaling-final.json", "scaling-mask.json",
                     "control-attention.json"}, (
        f"removing --scaling-batch only flagged {sorted(named)}")
    assert all("records scaling_batch" in m for m in mismatches), mismatches


def test_the_guard_notices_a_flag_given_with_the_wrong_value() -> None:
    """Presence is not enough: the value has to be the recorded one."""
    commands = _run_documentation_commands()
    wrong = [c.replace("--scaling-batch 4", "--scaling-batch 8") for c in commands]
    assert wrong != commands, "no command carried --scaling-batch 4 to change"

    _, mismatches = compare(wrong)
    named = {m.split(":")[0] for m in mismatches if "passes --scaling-batch" in m}
    assert named == {"results.json", "scaling-final.json", "scaling-mask.json"}, (
        f"a wrong --scaling-batch only flagged {sorted(named)}")
    assert all("'8'" in m for m in mismatches), mismatches


def test_the_guard_notices_a_store_true_flag_the_file_says_was_not_given() -> None:
    """`skip_scaling` is False in the scaling runs, so passing it is a defect."""
    commands = _run_documentation_commands()
    added = [
        c + " --skip-scaling" if "--out scaling-mask.json" in c else c
        for c in commands
    ]
    assert added != commands, "no command wrote scaling-mask.json"

    _, mismatches = compare(added)
    assert mismatches and all("skip_scaling" in m for m in mismatches), mismatches
    assert mismatches[0].startswith("scaling-mask.json:"), mismatches
