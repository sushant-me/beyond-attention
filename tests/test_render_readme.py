"""The README's generated regions are generated, and this is the check.

The repository's central claim about its own numbers is that every published
figure comes from a JSON results file through `experiments/render_readme.py`.
That claim is only as good as the weakest generated region, and it was false in
one place: the results block contained a section written by hand, so re-running
the documented results command **deleted** it -- the renderer's guard checks for
the markers its own tables produce, and a section it cannot produce is not one it
notices losing.

So the check here is not "the README mentions the right numbers". It is:

* each `BEGIN`/`END` region of the README is **byte-identical** to what the
  renderer produces from the committed JSON, which fails the moment hand-written
  text sits inside a rendered region;
* rendering is deterministic back to back, so the comparison means what it says;
* and both blocks that contain a section the renderer can only produce from a
  second file -- the results block and the gate block -- **refuse** to write when
  that file is not given, rather than silently dropping the section. The
  length-extrapolation bug and the straight-through subsection are the same bug,
  tested as the specific bug in each place it can recur.

`BLOCKS` is every `BEGIN`/`END` pair in the README. It was not: the gate block
was missing from it, so the guarantee above was false for one of the regions it
is stated over, and a hand-written edit inside that region would have survived
until someone re-rendered it.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
README = ROOT / "README.md"


def _load_renderer():
    spec = importlib.util.spec_from_file_location(
        "render_readme_for_tests", ROOT / "experiments" / "render_readme.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


render_readme = _load_renderer()


def _json(name: str) -> dict:
    return json.loads((ROOT / name).read_text())


def _block(text: str, begin: str, end: str) -> str:
    """The block exactly as the renderer writes it, markers included."""
    assert begin in text and end in text, f"README has no {begin} block"
    _, rest = text.split(begin, 1)
    body, _ = rest.split(end, 1)
    return f"{begin}{body}{end}"


def _results_render() -> str:
    return render_readme.render(
        _json("results.json"),
        [_json("scaling-final.json"), _json("scaling-mask.json")],
        _json("control-attention.json"),
        _json("scan-inner.json"),
        _json("length-extrapolation.json"),
    )


BLOCKS = (
    ("results", render_readme.BEGIN, render_readme.END, _results_render),
    ("voice", render_readme.VOICE_BEGIN, render_readme.VOICE_END,
     lambda: render_readme.voice_section(_json("voice-affect.json"))),
    ("agent", render_readme.AGENT_BEGIN, render_readme.AGENT_END,
     lambda: render_readme.agent_section(_json("agent-loop.json"))),
    ("memory", render_readme.MEMORY_BEGIN, render_readme.MEMORY_END,
     lambda: render_readme.memory_section(_json("long-memory.json"))),
    ("emotion", render_readme.EMOTION_BEGIN, render_readme.EMOTION_END,
     lambda: render_readme.emotion_section(_json("emotion-classifier.json"))),
    # The gate block was the one rendered region this tuple did not cover, so
    # the file's own headline guarantee -- "each BEGIN/END region is
    # byte-identical to the renderer's output" -- was false for it, and an edit
    # inside it would have survived until someone re-rendered.
    ("learned-gate", render_readme.LEARNED_GATE_BEGIN,
     render_readme.LEARNED_GATE_END,
     lambda: render_readme.learned_gate_section(
         _json("learned-gate.json"), _json("straight-through.json"))),
    ("inner-scan-sweep", render_readme.INNER_SWEEP_BEGIN,
     render_readme.INNER_SWEEP_END,
     lambda: render_readme.inner_scan_sweep_section(
         _json("scan-inner-scaling.json"))),
)


@pytest.mark.parametrize("name, begin, end, produce", BLOCKS,
                         ids=[block[0] for block in BLOCKS])
def test_every_rendered_block_matches_its_results_file(
        name: str, begin: str, end: str, produce) -> None:
    """A rendered region contains the renderer's output and nothing else.

    Hand-written content inside a rendered block fails here, which is the point:
    the alternative is discovering it when the documented command deletes it.
    """
    text = README.read_text()
    rendered = produce()
    expected = f"{begin}\n{rendered}\n{end}"
    assert _block(text, begin, end) == expected, (
        f"the {name} block is not what the renderer produces from its JSON; "
        f"either the JSON changed and the README was not re-rendered, or "
        f"hand-written content is inside a rendered region"
    )


def test_every_block_renders_deterministically_back_to_back() -> None:
    for name, _, _, produce in BLOCKS:
        first, second = produce(), produce()
        assert first == second, f"the {name} render is not deterministic"


def test_the_results_block_is_not_hand_written_anywhere() -> None:
    """The specific failure: a section the renderer had to learn to produce."""
    results = _block(README.read_text(), render_readme.BEGIN,
                     render_readme.END)
    assert "### Does either model read longer than it trained?" in results
    assert "| x train |" in results
    # The section's numbers are the JSON's, not the README's. Counted, because
    # this loop is the whole assertion: an empty payload would make it pass
    # without checking a single number, which is the failure mode a guard that
    # reads a file it never confirmed is non-empty always has.
    payload = _json("length-extrapolation.json")
    checked = 0
    for model in ("attention", "ssm"):
        for pairs, row in payload["extrapolated"][model].items():
            assert f"{row['mean']:.3f}" in results, (model, pairs)
            checked += 1
    assert checked >= 8, f"only {checked} extrapolated rows were checked"


def _results_argv(copy: pathlib.Path, *, omit: str | None = None,
                  control: pathlib.Path | None = None) -> list[str]:
    """The documented results command, optionally missing or over a file."""
    pairs = [
        ("--mqar", ROOT / "results.json"),
        ("--scaling", ROOT / "scaling-final.json"),
        ("--scaling", ROOT / "scaling-mask.json"),
        ("--control", control or ROOT / "control-attention.json"),
        ("--scan-inner", ROOT / "scan-inner.json"),
        ("--length-extrapolation", ROOT / "length-extrapolation.json"),
    ]
    argv = ["render_readme.py"]
    for flag, path in pairs:
        if flag == omit:
            continue
        argv += [flag, str(path)]
    return argv + ["--readme", str(copy)]


@pytest.mark.parametrize("omit", ["--scaling", "--control", "--scan-inner",
                                  "--length-extrapolation"])
def test_the_results_command_refuses_to_delete_any_of_its_sections(
        omit: str, tmp_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch) -> None:
    """The documented command refuses rather than dropping a section.

    This was the bug found in the repository, and it was fixed for one file: the
    guard covered `--length-extrapolation` only, while `--control` and
    `--scan-inner` still exited 0 and replaced their tables with a placeholder.
    One of those casualties is the control that falsifies the headline. Each of
    the four is omitted in turn, against a copy of the README, and the copy must
    come back byte-identical every time because the run refused.
    """
    copy = tmp_path / "README.md"
    original = README.read_text()
    copy.write_text(original)

    monkeypatch.setattr(sys, "argv", _results_argv(copy, omit=omit))
    assert render_readme.main() == 1, f"{omit} was not required"
    assert copy.read_text() == original, "a refused render must not write"
    assert ("### Does either model read longer than it trained?"
            in copy.read_text())


def test_the_results_render_with_the_file_still_produces_the_block(
        tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    copy = tmp_path / "README.md"
    copy.write_text(README.read_text())
    monkeypatch.setattr(sys, "argv", _results_argv(copy))
    assert render_readme.main() == 0
    assert copy.read_text() == README.read_text(), \
        "the documented results command must reproduce the committed README"


@pytest.mark.parametrize("body", ['{"mqar": {}}', "{}"])
def test_the_results_render_refuses_a_payload_it_cannot_render(
        body: str, tmp_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch) -> None:
    """A file that is present but unusable is a refusal, not an empty table.

    The flag check catches a *missing* file. This catches the two ways a file can
    be there and still not render -- a payload whose rows are empty, which would
    otherwise be published as "_control: not run_", and a payload that is not the
    shape the renderer reads, which would otherwise be a traceback. Both are
    passed as the control file, and the copy must come back byte-identical.
    """
    bad = tmp_path / "control-attention.json"
    bad.write_text(body)

    copy = tmp_path / "README.md"
    original = README.read_text()
    copy.write_text(original)
    monkeypatch.setattr(sys, "argv", _results_argv(copy, control=bad))
    assert render_readme.main() == 1
    assert copy.read_text() == original, "a refused render must not write"


def test_the_gate_command_refuses_to_delete_the_straight_through_subsection(
        tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The same bug, in the other block that can have it.

    The straight-through subsection is rendered into the gate block, so a
    `--learned-gate` render without `--straight-through` would replace a
    published result with a missing-file note. The command is run against a copy
    and the copy must come back byte-identical, because the run refused.
    """
    copy = tmp_path / "README.md"
    original = README.read_text()
    copy.write_text(original)

    monkeypatch.setattr(sys, "argv", [
        "render_readme.py",
        "--learned-gate", str(ROOT / "learned-gate.json"),
        "--readme", str(copy),
    ])
    assert render_readme.main() == 1
    assert copy.read_text() == original, "a refused render must not write"
    assert "### The straight-through hard gate" in copy.read_text()


def test_the_gate_render_with_the_file_produces_the_block(
        tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    copy = tmp_path / "README.md"
    copy.write_text(README.read_text())
    monkeypatch.setattr(sys, "argv", [
        "render_readme.py",
        "--learned-gate", str(ROOT / "learned-gate.json"),
        "--straight-through", str(ROOT / "straight-through.json"),
        "--readme", str(copy),
    ])
    assert render_readme.main() == 0
    assert copy.read_text() == README.read_text(), \
        "the documented gate command must reproduce the committed README"


def test_the_straight_through_rows_are_the_jsons_not_the_readmes() -> None:
    """The subsection's numbers come from `straight-through.json`.

    Three of them, because they are the argument: the harness control that has to
    reproduce before the result is readable, the result itself, and the untrained
    control that says the hard forward pass does nothing without training.
    """
    text = README.read_text()
    _, rest = text.split(render_readme.LEARNED_GATE_BEGIN, 1)
    block, _ = rest.split(render_readme.LEARNED_GATE_END, 1)

    payload = _json("straight-through.json")
    conditions = payload["conditions"]
    assert f"{conditions['soft_gate_raw']['rate']:.3f}" in block
    assert f"{conditions['soft_gate_sharpened']['rate']:.3f}" in block
    trained = conditions["straight_through_trained"]["spread"]
    assert f"**{trained['mean']:.3f}**" in block, trained
    for field in ("min", "max", "stdev"):
        assert f"{trained[field]:.3f}" in block, (field, trained)
    assert f"{len(trained['rates'])} seeds" in block, trained

    # The result is the loss on every seed, and the payload is where it comes
    # from: a renderer that hard-coded 29.19 would pass the assertions above.
    losses = {row["final_loss"] for row in payload["seeds"]}
    assert len(losses) == 1, losses
    assert f"{losses.pop():.2f}" in block
    assert "writes to every slot" in block


def test_the_straight_through_table_refuses_a_payload_that_cannot_argue() -> None:
    """No controls, no table -- the row it is there to earn is missing.

    The claim the table makes is not "the hard gate scored 0.160"; it is "0.420
    and 1.000 came out of the same loop, and the hard gate scored 0.160 anyway".
    A payload with only the straight-through rows cannot make that claim, and the
    KeyError is the intended outcome rather than a fallback rendering zeroes.
    """
    payload = _json("straight-through.json")
    stripped = {"config": payload["config"],
                "conditions": {"straight_through_trained":
                               payload["conditions"]["straight_through_trained"]}}
    with pytest.raises(KeyError):
        render_readme.straight_through_table(stripped)

    assert render_readme.straight_through_table(None) == \
        "_straight-through attempt: missing_"
    assert render_readme.straight_through_table({}) == \
        "_straight-through attempt: no rows_"


def test_the_inner_scan_sweep_is_the_jsons_and_emphasises_the_fastest() -> None:
    """The sweep's table was the last one in Results with typed-in numbers.

    Every cell is checked against `scan-inner-scaling.json` rather than against
    the README, and the row that is bold has to be the row that is actually
    fastest at the longest length -- an emphasis that is written into the render
    rather than derived would survive the measurement reversing.
    """
    block = _block(README.read_text(), render_readme.INNER_SWEEP_BEGIN,
                   render_readme.INNER_SWEEP_END)
    payload = _json("scan-inner-scaling.json")
    results = payload["results"]
    longest = max(payload["config"]["lengths"])

    configurations = sorted({key.rsplit("@L", 1)[0] for key in results
                             if key != "_exponents"})
    checked = 0
    for key in configurations:
        inner, chunk = key.split("@")
        times = {k: results[f"{k}@L{longest}"]["seconds"]
                 for k in configurations}
        fastest = min(times, key=times.get)
        expected = (f"| **{inner} @ chunk {chunk}** | "
                    f"**{results['_exponents'][key]:.2f}** | "
                    f"**{times[key]:.2f} s** | "
                    f"{results[f'{key}@L{longest}']['peak_rss_kb'] / 1024:,.0f} |"
                    if key == fastest else
                    f"| {inner} @ chunk {chunk} | "
                    f"{results['_exponents'][key]:.2f} | {times[key]:.2f} s | "
                    f"{results[f'{key}@L{longest}']['peak_rss_kb'] / 1024:,.0f} |")
        assert expected in block, (key, expected)
        checked += 1
    assert checked == len(configurations) >= 4, checked

    # The per-length evidence the sentence rests on, from the same file.
    for length in payload["config"]["lengths"]:
        at_length = {k: results[f"{k}@L{length}"]["seconds"]
                     for k in configurations}
        best = min(at_length, key=at_length.get)
        assert f"{length:,}: {at_length[best]:.2f} s" in block, length


def test_the_sweep_verdict_changes_when_the_fastest_configuration_does() -> None:
    """The sentence is a claim about this run, so it has to be able to fail.

    Making one configuration much slower at the shortest length moves the winner
    there. The renderer must then say that the fastest configuration is not the
    same at every length, instead of keeping the sentence it was written with.
    """
    payload = _json("scan-inner-scaling.json")
    shortest = min(payload["config"]["lengths"])
    baseline = render_readme.inner_scan_sweep_section(payload)
    assert "at **every** length measured" in baseline

    payload["results"][f"loop@64@L{shortest}"]["seconds"] = 999.0
    flipped = render_readme.inner_scan_sweep_section(payload)
    assert "at **every** length measured" not in flipped
    assert "not the same at every length" in flipped
    assert f"{shortest:,}: " in flipped


def test_the_sweep_section_handles_a_payload_it_cannot_render() -> None:
    assert "not run_" in render_readme.inner_scan_sweep_section(None)
    assert render_readme.inner_scan_sweep_section({}) == \
        "_inner-scan sweep: no rows_"
