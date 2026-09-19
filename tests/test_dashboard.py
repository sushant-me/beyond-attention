"""The generated dashboard: self-contained, deterministic, and about the data.

A published page is the easiest artifact in a repository to let drift, because it
looks finished whether or not it is. So the tests here are the three properties
the page claims, and the third is the one that can fail loudly:

**Self-contained.** The page must reference nothing outside itself -- no CDN, no
external font, no image, no request of any kind -- which the test checks by
grepping the output for the shapes such a reference takes. That control can fail,
and it is meant to: a helpful-looking `<link>` would otherwise be invisible until
someone opened the file offline. The SVG roots carry no `xmlns` for this reason,
since the HTML parser supplies the namespace and the declaration's value is the
one string the page must not contain.

**Deterministic.** Rendering the same inputs twice produces identical bytes, so a
diff of the page is a diff of the measurements and not of a timestamp.

**Generated, and current.** Every number asserted below is read out of
`voice-affect.json` and `agent-loop.json` and required to appear in the page, the
chart paths are required to have one point per frame the JSON has a value for,
and the sentence about a control is required to match the control's actual cells
-- including, when they disagree, a count of the cells where they disagree. The
last test renders the page again and requires the committed file to match it, so
a results file that changes without the page being re-rendered fails here rather
than in a reader's browser.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import re
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
RENDERER = ROOT / "experiments" / "render_dashboard.py"
VOICE = ROOT / "voice-affect.json"
AGENT = ROOT / "agent-loop.json"
PAGE = ROOT / "dashboard.html"

# Every shape an external reference takes in an HTML document. The first four are
# the ones the task names; the rest are the ways the same guarantee is usually
# broken by accident.
FORBIDDEN_REFERENCES = (
    "http://", "https://", "//cdn", "<script src=",
    "<link", "<img", "@import", "url(",
)

SECTION_MARKERS = (
    "<!-- DASHBOARD:VOICE:BEGIN -->", "<!-- DASHBOARD:VOICE:END -->",
    "<!-- DASHBOARD:CONTROL:BEGIN -->", "<!-- DASHBOARD:CONTROL:END -->",
    "<!-- DASHBOARD:AGENT:BEGIN -->", "<!-- DASHBOARD:AGENT:END -->",
    "<!-- DASHBOARD:RESULTS:BEGIN -->", "<!-- DASHBOARD:RESULTS:END -->",
    "<!-- DASHBOARD:LIMITS:BEGIN -->", "<!-- DASHBOARD:LIMITS:END -->",
    "<!-- DASHBOARD:FOOTER:BEGIN -->", "<!-- DASHBOARD:FOOTER:END -->",
)

SECTION_IDS = ("voice", "voicing-control", "agent", "results", "limitations",
               "regenerate")


def renderer():
    """The renderer as a module, so its formatters are the test's formatters."""
    spec = importlib.util.spec_from_file_location("render_dashboard", RENDERER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def rd():
    return renderer()


@pytest.fixture(scope="module")
def voice() -> dict:
    return json.loads(VOICE.read_text())


@pytest.fixture(scope="module")
def agent() -> dict:
    return json.loads(AGENT.read_text())


@pytest.fixture(scope="module")
def page(rd, voice, agent) -> str:
    return rd.render(voice, agent)


# --------------------------------------------------------------------------
# Self-contained
# --------------------------------------------------------------------------

def test_the_page_references_nothing_outside_itself(page: str) -> None:
    for forbidden in FORBIDDEN_REFERENCES:
        assert forbidden not in page, f"the page references {forbidden!r}"


def test_the_page_carries_its_own_style_and_script(page: str) -> None:
    """The other half of self-contained: there has to be something to inline."""
    assert "<style>" in page and "</style>" in page
    assert "<script>" in page and "</script>" in page
    assert page.count("<svg") >= 6  # four condition charts and two control charts


# --------------------------------------------------------------------------
# Deterministic
# --------------------------------------------------------------------------

def test_rendering_the_same_inputs_twice_is_byte_identical(rd, voice, agent) -> None:
    first = rd.render(voice, agent)
    second = rd.render(voice, agent)

    assert first == second
    assert first.encode() == second.encode()


def test_the_renderer_writes_the_file_the_command_line_documents(
    tmp_path: pathlib.Path,
) -> None:
    out = tmp_path / "dashboard.html"
    result = subprocess.run(
        [sys.executable, str(RENDERER), "--voice", str(VOICE),
         "--agent", str(AGENT), "--out", str(out)],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )

    assert result.returncode == 0, result.stderr
    assert out.is_file() and out.stat().st_size > 10_000
    assert "wrote" in result.stdout
    written = out.read_text()
    for marker in SECTION_MARKERS:
        assert marker in written, marker
    for section_id in SECTION_IDS:
        assert f'id="{section_id}"' in written, section_id


def test_an_input_missing_a_panel_is_refused_rather_than_rendered(
    tmp_path: pathlib.Path, voice: dict,
) -> None:
    """The renderer's guard, as a control that can fail.

    A page rendered from a results file without the frame tracks would silently
    lose its charts, and it would look finished.
    """
    narrowed = tmp_path / "narrow.json"
    narrowed.write_text(json.dumps({key: value for key, value in voice.items()
                                    if key != "frame_tracks"}))
    result = subprocess.run(
        [sys.executable, str(RENDERER), "--voice", str(narrowed),
         "--agent", str(AGENT), "--out", str(tmp_path / "out.html")],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )

    assert result.returncode == 1
    assert "refusing to render" in result.stderr
    assert not (tmp_path / "out.html").exists()


# --------------------------------------------------------------------------
# Generated: the numbers on the page are the numbers in the files
# --------------------------------------------------------------------------

def test_the_voice_panel_carries_the_measured_descriptors(rd, page: str,
                                                          voice: dict) -> None:
    for condition, entry in voice["conditions"].items():
        assert condition in page
        for key in ("f0_mean", "f0_std", "energy_std_voiced", "voiced_ratio",
                    "speaking_rate", "jitter"):
            value = entry["descriptors"][key]
            unit, digits = rd.DESCRIPTOR_UNITS[key]
            shown = (rd.fmt_pct(value) if unit == "share"
                     else rd.fmt(value, digits))
            assert shown in page, (condition, key, shown)

    separability = voice["separability"]
    assert rd.fmt_rate(separability["accuracy"]) in page
    assert rd.fmt_rate(separability["chance"]) in page
    assert separability["classes"] == list(voice["conditions"])


def test_the_voicing_control_numbers_are_on_the_page(rd, page: str,
                                                     voice: dict) -> None:
    probe = voice["voicing_probe"]
    threshold = probe["threshold"]
    assert rd.fmt_compact(threshold) in page

    noise = probe["series"]["white_noise"]
    tone = probe["series"]["tone"]
    assert noise["voiced_ratio"] == 0.0, "the probe noise was judged voiced"
    assert noise["max_confidence"] < threshold
    assert tone["max_confidence"] > threshold

    assert rd.fmt(noise["max_confidence"], 4) in page
    assert rd.fmt(tone["max_confidence"], 3) in page
    assert rd.fmt_rate(noise["voiced_ratio"]) in page
    assert rd.fmt_rate(tone["voiced_ratio"]) in page
    # And the summary over the file's own noise draws, which is a different
    # measurement from the drawn probe.
    control = voice["controls"]["white_noise_voicing"]
    assert rd.fmt(control["max_confidence"], 4) in page
    assert rd.fmt_rate(control["voiced_ratio_max"]) in page


def test_every_chart_is_drawn_from_the_frame_data(page: str,
                                                  voice: dict) -> None:
    """One point per frame with a value, and none invented for the gaps.

    This is the check that the chart is the measurement: a path with a different
    number of points than the JSON has values would still look like a chart.
    """
    tracks = voice["frame_tracks"]
    conditions = voice["separability"]["classes"]
    f0_paths = re.findall(r'<path class="line f0"[^>]*d="([^"]*)"', page)
    rms_paths = re.findall(r'<path class="line rms"[^>]*d="([^"]*)"', page)

    assert len(f0_paths) == len(conditions)
    assert len(rms_paths) == len(conditions)
    for condition, f0_d, rms_d in zip(conditions, f0_paths, rms_paths):
        track = tracks["conditions"][condition]
        expected = sum(1 for value in track["f0_hz"] if value is not None)
        assert len(re.findall(r"[ML][-\d.]+,", f0_d)) == expected, condition
        assert len(re.findall(r"[ML][-\d.]+,", rms_d)) == len(track["rms"])
        assert len(track["f0_hz"]) == len(track["rms"]) == len(track["voiced_share"])
        assert len(track["f0_hz"]) == len(tracks["times_s"])


def test_the_staircase_table_carries_every_measured_cell(rd, page: str,
                                                         agent: dict) -> None:
    config = agent["config"]
    for family in config["families"]:
        assert f"<code>{family}</code>" in page
        for budget in config["budgets"]:
            cell = agent["solve_rate"][f"{family}@{budget}"]
            assert rd.fmt_rate(cell["rate"]) in page
            assert f'({cell["solved"]}/{cell["total"]})' in page


def test_every_control_appears_with_its_numbers(rd, page: str,
                                                agent: dict) -> None:
    for name, value in agent["controls"].items():
        assert name in page, name
        if isinstance(value, dict) and value and all(
            isinstance(cell, dict) and "rate" in cell for cell in value.values()
        ):
            for cell in value.values():
                assert rd.fmt_rate(cell["rate"]) in page


def test_the_page_states_the_scalar_carry_result_the_file_holds(rd, page: str,
                                                                agent: dict) -> None:
    """The negative has to be on the page, and it has to match the data.

    Which sentence is correct depends on the results file: a control can tie the
    agent row for row, or differ in a counted number of cells. The test computes
    the same comparison the renderer does and requires the page to say the thing
    the numbers support -- so the page cannot quietly claim the flattering one.
    """
    solve = agent["solve_rate"]
    cells = agent["controls"]["scalar_carry"]
    shared = [key for key in solve if key in cells]
    same = [key for key in shared if cells[key]["rate"] == solve[key]["rate"]]

    assert "scalar_carry" in page
    if len(same) == len(shared):
        assert f"row for row: all {len(shared)} cells" in page
    else:
        assert f"reproduces the agent's rate in {len(same)} of the {len(shared)}" \
               in page
        assert len(shared) - len(same) == 1  # the file's own count, spelled out
        differing = next(key for key in shared if key not in same)
        assert differing in page
        assert rd.fmt_rate(cells[differing]["rate"]) in page


def test_the_agent_panel_narrates_the_trace_step_by_step(page: str,
                                                         agent: dict) -> None:
    for trace_key in ("example_trace", "selective_example_trace"):
        trace = agent.get(trace_key)
        if not trace:
            continue
        assert trace["text"] in page, trace_key
        for step in trace["steps"]:
            assert f"step {step['index']}" in page
            assert step["instruction"] in page
            assert step["action"] in page
            for key, value in step["args"].items():
                assert f"{key}={value}" in page
            assert str(step["result"]) in page
        assert str(trace["answer"]) in page


def test_every_condition_parameters_appear_with_their_condition(page: str,
                                                                voice: dict) -> None:
    for entry in voice["conditions"].values():
        base = entry["parameters"]["f0_base"]
        assert f"F0 target {base:g} Hz" in page


def test_the_limitations_panel_says_what_is_not_established(page: str) -> None:
    """The four statements the page is required to make, checked as text.

    These are claims about the work rather than numbers, so this is the one test
    that greps for prose -- it is checking that a required section was not
    dropped, not that a measurement is right.
    """
    limitations = page.split("<!-- DASHBOARD:LIMITS:BEGIN -->")[1]
    for phrase in (
        "no generalisation beyond the closed task family",
        "no real-speech validation",
        "synthetic-parameter recovery",
        "in a gradient path",
        "no optimiser, no loss and no backward pass",
        "no tokenizer",
    ):
        assert phrase.lower() in limitations.lower(), phrase


def test_the_footer_states_the_regeneration_command_and_the_sources(page: str) -> None:
    footer = page.split("<!-- DASHBOARD:FOOTER:BEGIN -->")[1]
    for fragment in ("experiments/voice_affect.py", "experiments/agent_loop.py",
                     "experiments/render_dashboard.py", "voice-affect.json",
                     "agent-loop.json"):
        assert fragment in footer, fragment


# --------------------------------------------------------------------------
# Freshness: the committed page is what the committed results files produce
# --------------------------------------------------------------------------

def test_the_committed_page_is_what_the_renderer_produces_from_these_files(
    rd, voice, agent,
) -> None:
    """Re-render and compare. A stale page is a page that lies.

    If this fails, the numbers moved and the page did not. The fix is the
    command in the page's own footer:

        python experiments/render_dashboard.py --voice voice-affect.json \\
            --agent agent-loop.json --out dashboard.html
    """
    assert PAGE.is_file(), "dashboard.html has not been rendered"
    on_disk = PAGE.read_text()

    assert on_disk == rd.render(voice, agent), (
        "dashboard.html is stale: re-run experiments/render_dashboard.py"
    )
