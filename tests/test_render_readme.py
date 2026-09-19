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
* and the results render **refuses** to write the block when the
  length-extrapolation file is not given, rather than silently dropping the
  section -- the specific bug, tested as the specific bug.
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
    # The section's numbers are the JSON's, not the README's.
    payload = _json("length-extrapolation.json")
    for model in ("attention", "ssm"):
        for pairs, row in payload["extrapolated"][model].items():
            assert f"{row['mean']:.3f}" in results, (model, pairs)


def test_the_results_command_refuses_to_delete_the_section(
        tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the extrapolation file the render fails instead of dropping it.

    This is the bug that was found in the repository, reproduced as a test: the
    documented results command is run against a copy of the README, and the copy
    must come back byte-identical because the run refused.
    """
    copy = tmp_path / "README.md"
    original = README.read_text()
    copy.write_text(original)

    monkeypatch.setattr(sys, "argv", [
        "render_readme.py",
        "--mqar", str(ROOT / "results.json"),
        "--scaling", str(ROOT / "scaling-final.json"),
        "--scaling", str(ROOT / "scaling-mask.json"),
        "--control", str(ROOT / "control-attention.json"),
        "--scan-inner", str(ROOT / "scan-inner.json"),
        "--readme", str(copy),
    ])
    assert render_readme.main() == 1
    assert copy.read_text() == original, "a refused render must not write"
    assert ("### Does either model read longer than it trained?"
            in copy.read_text())


def test_the_results_render_with_the_file_still_produces_the_block(
        tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    copy = tmp_path / "README.md"
    copy.write_text(README.read_text())
    monkeypatch.setattr(sys, "argv", [
        "render_readme.py",
        "--mqar", str(ROOT / "results.json"),
        "--scaling", str(ROOT / "scaling-final.json"),
        "--scaling", str(ROOT / "scaling-mask.json"),
        "--control", str(ROOT / "control-attention.json"),
        "--scan-inner", str(ROOT / "scan-inner.json"),
        "--length-extrapolation", str(ROOT / "length-extrapolation.json"),
        "--readme", str(copy),
    ])
    assert render_readme.main() == 0
    assert copy.read_text() == README.read_text(), \
        "the documented results command must reproduce the committed README"
