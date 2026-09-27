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

import copy
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


def test_the_reference_column_reports_its_own_spread() -> None:
    """The reference column was one model, and the sentence said so.

    With a single reference model the only spread on screen was the extrapolated
    column's, and the bullet compared the gap against that -- the right
    comparison only while the reference is one draw. The reference runs three
    seeds now, so it carries a spread of its own and the sentence has to be about
    both. Left as a branch rather than a rewrite: a run with `--reference-seeds 0`
    is still possible and must not claim a spread it does not have.
    """
    payload = _json("length-extrapolation.json")
    section = render_readme.extrapolation_section(payload)
    assert "either spread" in section, section
    checked = 0
    for model in ("attention", "ssm"):
        for pairs, row in payload["reference"][model].items():
            assert len(row["per_seed"]) > 1, (model, pairs)
            assert f"{row['mean']:.3f} ±{row['spread']:.3f}" in section, (model, pairs)
            checked += 1
    assert checked >= 4, f"only {checked} reference cells were checked"

    single = copy.deepcopy(payload)
    for model in single["reference"].values():
        for row in model.values():
            row["per_seed"] = row["per_seed"][:1]
            row["spread"] = 0.0
    weaker = render_readme.extrapolation_section(single)
    assert "either spread" not in weaker
    assert "The reference is 1 seed, so a gap smaller than the spread" in weaker


def test_the_reference_attribution_follows_the_sign_of_the_gap() -> None:
    """Which model lost ground is the interpretation, so it is derived.

    A model scoring below one trained at that length lost something to
    extrapolation; one scoring above it did not. The sentence used to attribute
    every larger gap to "part extrapolation", which was true while both models
    lost ground and is false for whichever one did not -- and the committed run
    is the case where one of each occurs.
    """
    payload = _json("length-extrapolation.json")
    mixed = render_readme.extrapolation_section(payload)
    assert "the two gaps run opposite ways" in mixed, mixed
    assert "ssm (0.018) scores below" in mixed, mixed
    assert "attention (0.044) scores above it" in mixed, mixed

    # Both references far above the extrapolated rows: both lost ground.
    both_cost = copy.deepcopy(payload)
    for model in both_cost["reference"].values():
        for row in model.values():
            row["mean"] = 0.9
    costs = render_readme.extrapolation_section(both_cost)
    assert "run opposite ways" not in costs
    assert "part extrapolation and part how hard MQAR" in costs

    # Both below: neither lost ground, so neither gap is an extrapolation cost.
    neither = copy.deepcopy(payload)
    for model in neither["reference"].values():
        for row in model.values():
            row["mean"] = 0.02
    free = render_readme.extrapolation_section(neither)
    assert "run opposite ways" not in free
    assert "task difficulty rather than extrapolation" in free
    assert "part extrapolation" not in free, free


def _results_argv(copy: pathlib.Path, *, omit: str | None = None,
                  control: pathlib.Path | None = None,
                  scan_inner: pathlib.Path | None = None) -> list[str]:
    """The documented results command, optionally missing or over a file."""
    pairs = [
        ("--mqar", ROOT / "results.json"),
        ("--scaling", ROOT / "scaling-final.json"),
        ("--scaling", ROOT / "scaling-mask.json"),
        ("--control", control or ROOT / "control-attention.json"),
        ("--scan-inner", scan_inner or ROOT / "scan-inner.json"),
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
        omit: str, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str]) -> None:
    """The documented command refuses rather than dropping a section.

    This was the bug found in the repository, and it was fixed for one file: the
    guard covered `--length-extrapolation` only, while `--control` and
    `--scan-inner` still exited 0 and replaced their tables with a placeholder.
    One of those casualties is the control that falsifies the headline. Each of
    the four is omitted in turn, against a copy of the README, and the copy must
    come back byte-identical every time because the run refused.

    The refusal has to **name the flag**, which is what makes this test pin the
    guard it is named for. The property is defended more than once -- a missing
    file also renders as a placeholder that the sentinel check catches -- so
    without this assertion, deleting the `required` guard entirely left the test
    green while the command's failure message degraded to nothing at all.
    """
    copy = tmp_path / "README.md"
    original = README.read_text()
    copy.write_text(original)

    monkeypatch.setattr(sys, "argv", _results_argv(copy, omit=omit))
    assert render_readme.main() == 1, f"{omit} was not required"
    assert omit in capsys.readouterr().err, (
        f"the refusal does not say {omit} is missing, so it is not the guard "
        f"this test is named for that refused")
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
        tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str]) -> None:
    """The same bug, in the other block that can have it.

    The straight-through subsection is rendered into the gate block, so a
    `--learned-gate` render without `--straight-through` would replace a
    published result with a missing-file note. The command is run against a copy
    and the copy must come back byte-identical, because the run refused — and the
    refusal has to name the flag, for the same reason as above: the property is
    defended twice, so without that the test passes whichever guard fires.
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
    assert "--straight-through" in capsys.readouterr().err
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

    # A sweep with one length cannot support this section: there is no
    # least-squares fit over a single point, and "at every length measured" would
    # mean one length. It used to publish exactly that, over a payload the
    # documented `scan_inner.py --lengths 8192` produces.
    single = _json("scan-inner-scaling.json")
    keep = max(single["config"]["lengths"])
    single["config"]["lengths"] = [keep]
    single["results"] = {k: v for k, v in single["results"].items()
                         if k.endswith(f"@L{keep}") or k == "_exponents"}
    rendered = render_readme.inner_scan_sweep_section(single)
    assert "too few lengths_" in rendered
    assert "least squares" not in rendered

    # The near-identical name: `--scan-inner-scaling scan-inner.json` is the
    # mistake this flag invites, and the single-length comparison file has no
    # `lengths`, so it is the same refusal rather than a traceback.
    assert "too few lengths_" in render_readme.inner_scan_sweep_section(
        _json("scan-inner.json"))


def test_a_refused_allocation_is_a_row_not_a_crash() -> None:
    """`scan_inner.py` records a refused allocation as a result without timings.

    A renderer that indexed `["seconds"]` crashed on the experiment's own output
    rather than publishing the refusal. The row is an em dash, and it is excluded
    from every comparison rather than counted as zero.
    """
    payload = _json("scan-inner-scaling.json")
    longest = max(payload["config"]["lengths"])
    refused = f"vectorized@64@L{longest}"
    payload["results"][refused] = {
        "inner": "vectorized", "chunk": 64, "length": longest,
        "failed": True, "note": "exit 137",
    }
    rendered = render_readme.inner_scan_sweep_section(payload)
    assert "| vectorized @ chunk 64 | 0.69 | — | — |" in rendered
    # The configuration that did complete is still the fastest, and the one that
    # refused is not silently treated as an instant zero.
    assert "| **loop @ chunk 64** |" in rendered
    assert "0.00 s" not in rendered

    every = _json("scan-inner-scaling.json")
    for key, row in list(every["results"].items()):
        if isinstance(row, dict):
            every["results"][key] = dict(row, failed=True)
            every["results"][key].pop("seconds", None)
            every["results"][key].pop("peak_rss_kb", None)
    assert "no rows_" in render_readme.inner_scan_sweep_section(every)


def test_the_default_the_sweep_names_is_the_models_own_default() -> None:
    """The sentence says "the default", so it has to be the real one.

    The renderer keeps `DEFAULT_SCAN` as a constant rather than importing the
    model, because importing it would pull torch into rendering. That leaves one
    way for the sentence to be false: the constant drifting from the class. A run
    where `vectorized` is fastest must say the shipped default is *not* it, which
    the renderer only gets right if this constant is right.
    """
    import inspect

    from beyond_attention.model import SelectiveSSMBlock

    parameters = inspect.signature(SelectiveSSMBlock.__init__).parameters
    assert render_readme.DEFAULT_SCAN == (
        parameters["scan_inner"].default, parameters["scan_chunk"].default), (
        "DEFAULT_SCAN has drifted from SelectiveSSMBlock's own defaults, so the "
        "sweep section names the wrong configuration as shipped")

    payload = _json("scan-inner-scaling.json")
    for key in list(payload["results"]):
        if not isinstance(payload["results"][key], dict):
            continue
        payload["results"][key]["seconds"] = (
            0.5 if key.startswith("vectorized@64") else 99.0)
    rendered = render_readme.inner_scan_sweep_section(payload)
    assert "The shipped default is `loop` at chunk 64" in rendered
    assert "So the default is not a compromise" not in rendered
    # The emphasis is the data's too: hardcoding the bold row to `loop@64` is
    # invisible against the committed payload, where loop@64 happens to win.
    assert "| **vectorized @ chunk 64** |" in rendered
    assert "| **loop @ chunk 64** |" not in rendered


def test_the_sweep_movers_are_computed_and_signed() -> None:
    """The closing comparison is derived, both in size and in direction.

    It used to hardcode loop-first and first-chunk-versus-last, so a run where
    the vectorised scan won printed a negative percentage, and a single-chunk run
    printed a 1.0x chunk effect for a chunk compared with itself.
    """
    payload = _json("scan-inner-scaling.json")
    results, config = payload["results"], payload["config"]
    longest, first = max(config["lengths"]), config["chunks"][0]
    block = _block(README.read_text(), render_readme.INNER_SWEEP_BEGIN,
                   render_readme.INNER_SWEEP_END)

    times = {"loop": results[f"loop@{first}@L{longest}"]["seconds"],
             "vectorized": results[f"vectorized@{first}@L{longest}"]["seconds"]}
    fast, slow = sorted(times, key=times.get)
    assert f"{100 * (times[slow] / times[fast] - 1):.0f}%" in block
    assert f"{times[fast]:.2f} → {times[slow]:.2f} s" in block
    assert f"{fast} → {slow}" in block

    # The percentage has to follow the payload, not just be present: against the
    # committed file "21%" is also what a hardcoded 21% would print. Making the
    # slow configuration exactly three times the fast one must print 200%.
    tripled = _json("scan-inner-scaling.json")
    tripled["results"][f"{slow}@{first}@L{longest}"]["seconds"] = 3 * times[fast]
    movers = render_readme.inner_scan_sweep_section(tripled).split(
        "moves the result.")[1]
    assert "about 200%" in movers, movers
    assert f"{3 * times[fast]:.2f} s" in movers, movers

    last = config["chunks"][-1]
    loop_short = times["loop"]
    loop_long = results[f"loop@{last}@L{longest}"]["seconds"]
    assert f"{loop_long / loop_short:.1f}x" in block

    # Reversed, so the same sentence has to come out with the direction flipped
    # and no negative percentage.
    payload["results"][f"loop@{first}@L{longest}"]["seconds"] = 99.0
    flipped = render_readme.inner_scan_sweep_section(payload)
    movers = flipped.split("moves the result.")[1]
    assert "vectorized → loop" in movers
    assert "% (" in movers and "-" not in movers.split("% (")[0].split()[-1]

    # One chunk is no chunk effect: the clause is absent rather than 1.0x.
    payload = _json("scan-inner-scaling.json")
    payload["config"]["chunks"] = [first]
    payload["results"] = {k: v for k, v in payload["results"].items()
                          if f"@{first}@" in k or k == "_exponents"}
    single = render_readme.inner_scan_sweep_section(payload)
    assert "Changing the **chunk**" not in single
    assert "Changing the inner scan" in single


@pytest.mark.parametrize("body", [
    "{}",
    '{"config": {"lengths": [8192]}, '
    '"results": {"loop@64@L8192": {"seconds": 1.0}}}',
])
def test_the_sweep_block_refuses_a_payload_it_cannot_publish(
        body: str, tmp_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch) -> None:
    """The newest block's guard, which had no test of its own.

    Deleting the whole guard left the suite green, so nothing pinned the
    behaviour that the sweep region is never written from a payload that renders
    as a placeholder. The README copy must come back byte-identical.
    """
    bad = tmp_path / "sweep.json"
    bad.write_text(body)
    copy = tmp_path / "README.md"
    original = README.read_text()
    copy.write_text(original)
    monkeypatch.setattr(sys, "argv", [
        "render_readme.py", "--scan-inner-scaling", str(bad),
        "--readme", str(copy),
    ])
    assert render_readme.main() == 1
    assert copy.read_text() == original, "a refused render must not write"


def test_the_sweep_block_accepts_the_single_chunk_run_it_used_to_refuse(
        tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A false refusal: the guard's marker depended on which row was bold.

    `"| loop @ chunk "` is absent when the loop row is the emphasised fastest
    one, so a complete single-chunk sweep was refused for a reason that had
    nothing to do with the payload's validity.
    """
    payload = _json("scan-inner-scaling.json")
    first = payload["config"]["chunks"][0]
    payload["config"]["chunks"] = [first]
    payload["results"] = {k: v for k, v in payload["results"].items()
                          if f"@{first}@" in k or k == "_exponents"}
    path = tmp_path / "sweep.json"
    path.write_text(json.dumps(payload))

    copy = tmp_path / "README.md"
    copy.write_text(README.read_text())
    monkeypatch.setattr(sys, "argv", [
        "render_readme.py", "--scan-inner-scaling", str(path),
        "--readme", str(copy),
    ])
    assert render_readme.main() == 0
    block = _block(copy.read_text(), render_readme.INNER_SWEEP_BEGIN,
                   render_readme.INNER_SWEEP_END)
    assert "| **loop @ chunk 64** |" in block


def test_the_inner_scan_table_reads_both_key_shapes() -> None:
    """`scan_inner.py` renamed its result keys and this renderer did not notice.

    The committed `scan-inner.json` was produced before `5497438` added the
    `@L{length}` suffix, so it holds `loop@64`; a run of the script as it stands
    holds `loop@64@L8192`. The renderer read only the first, so regenerating the
    file -- which the README's own reproduction procedure does -- produced a
    table with a header and no rows, and the command exited 0.
    """
    committed = _json("scan-inner.json")
    rendered = render_readme.scan_inner_table(committed)
    rows = [line for line in rendered.splitlines() if line.startswith("| loop")]
    assert len(rows) == 2, rendered

    # The same run, in the shape the script writes today.
    current = {"config": dict(committed["config"], length=8192),
               "results": {f"{key}@L8192": row
                           for key, row in committed["results"].items()}}
    rendered = render_readme.scan_inner_table(current)
    rows = [line for line in rendered.splitlines() if line.startswith("| loop")]
    assert len(rows) == 2, rendered
    # And the two shapes render the same table, not two different ones.
    assert rendered == render_readme.scan_inner_table(committed)


def test_an_inner_scan_file_with_no_readable_row_is_refused(
        tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The failure the guard could not see, because it checked a header.

    `| inner scan |` is emitted whenever the section renders at all, so it is
    present in a table with nothing under it. The refusal now also requires a
    row, which is what makes this fail instead of quietly replacing four
    published rows with an empty table.
    """
    payload = _json("scan-inner.json")
    payload["results"] = {"loop_64": payload["results"]["loop@64"]}
    unreadable = tmp_path / "scan-inner.json"
    unreadable.write_text(json.dumps(payload))

    copy = tmp_path / "README.md"
    original = README.read_text()
    copy.write_text(original)
    monkeypatch.setattr(sys, "argv", _results_argv(copy, scan_inner=unreadable))
    assert render_readme.main() == 1
    assert copy.read_text() == original, "a refused render must not write"
    assert "| loop | 64 |" in copy.read_text()
