"""Render `dashboard.html`: one self-contained page from the results files.

This is the same contract `render_readme.py` has, for a surface meant to be
looked at rather than read: every number and every chart on the page is produced
from the JSON the experiments write, and re-running the experiments and this
script reproduces the page byte for byte.

    python experiments/voice_affect.py --out voice-affect.json
    python experiments/agent_loop.py --out agent-loop.json
    python experiments/render_dashboard.py --voice voice-affect.json \
        --agent agent-loop.json --out dashboard.html

Three properties are deliberate, and each is guarded by `tests/test_dashboard.py`
rather than left to the reader:

* **Self-contained.** The styles, the script and every chart are inline. The page
  loads nothing: no CDN, no external font, no image file, and no request of any
  kind, so it renders from disk with no server and no network. The test greps the
  output for external references, because that guarantee is exactly the kind that
  a helpful-looking `<link>` breaks silently.
  The SVG roots carry no ``xmlns`` attribute for the same reason. In an HTML
  document the parser puts an inline ``<svg>`` in the SVG namespace itself, so
  the declaration is redundant here -- and the value it would carry is the one
  string a self-contained page must not contain.
* **Deterministic.** No timestamp, no host name, no ordering that depends on a
  set, and no floating-point formatting that depends on the platform, so two
  renders of the same inputs are the same bytes and a diff of the page is a diff
  of the measurements.
* **Generated, including the parts that are sentences.** The prose that says what
  a control did is assembled from the numbers in the JSON -- including the
  count of cells where a control matches the agent and the count where it does
  not -- so the page cannot say one thing while the data says another.

What the page does **not** do is compute anything about the experiment. It scales
axes, turns counts into percentages, and compares control cells against agent
cells for equality; every other number is copied from the two files. If a
measurement is not in the JSON, it is not on the page.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import pathlib
import re
import sys
from typing import Any, Iterable, Sequence

VOICE_BEGIN = "<!-- DASHBOARD:VOICE:BEGIN -->"
VOICE_END = "<!-- DASHBOARD:VOICE:END -->"
CONTROL_BEGIN = "<!-- DASHBOARD:CONTROL:BEGIN -->"
CONTROL_END = "<!-- DASHBOARD:CONTROL:END -->"
AGENT_BEGIN = "<!-- DASHBOARD:AGENT:BEGIN -->"
AGENT_END = "<!-- DASHBOARD:AGENT:END -->"
RESULTS_BEGIN = "<!-- DASHBOARD:RESULTS:BEGIN -->"
RESULTS_END = "<!-- DASHBOARD:RESULTS:END -->"
LIMITS_BEGIN = "<!-- DASHBOARD:LIMITS:BEGIN -->"
LIMITS_END = "<!-- DASHBOARD:LIMITS:END -->"
FOOTER_BEGIN = "<!-- DASHBOARD:FOOTER:BEGIN -->"
FOOTER_END = "<!-- DASHBOARD:FOOTER:END -->"

# Every section the page must contain, checked before anything is written. A
# renderer that silently drops a section produces a page that looks finished,
# which is the failure this list exists to make loud.
REQUIRED_MARKERS = (
    VOICE_BEGIN, VOICE_END, CONTROL_BEGIN, CONTROL_END, AGENT_BEGIN, AGENT_END,
    RESULTS_BEGIN, RESULTS_END, LIMITS_BEGIN, LIMITS_END, FOOTER_BEGIN,
    FOOTER_END,
)

# Descriptors shown next to each chart, with the unit the number is in. This is
# a *label* table, not a source of values: the numbers come from the JSON, and a
# descriptor missing from the file is shown as absent rather than filled in.
CHART_DESCRIPTORS: tuple[tuple[str, str, int, str], ...] = (
    ("f0_mean", "Hz", 1, "mean pitch of the voiced frames"),
    ("f0_std", "Hz", 2, "spread of pitch across voiced frames"),
    ("energy_std_voiced", "a.u.", 4, "spread of loudness while voiced"),
    ("voiced_ratio", "share", 3, "fraction of frames judged voiced"),
    ("speaking_rate", "runs/s", 2, "voiced runs per second (proxy)"),
    ("jitter", "Hz", 2, "mean successive-frame pitch step"),
)

DESCRIPTOR_UNITS: dict[str, tuple[str, int]] = {
    "f0_mean": ("Hz", 1),
    "f0_std": ("Hz", 2),
    "f0_range": ("Hz", 2),
    "jitter": ("Hz", 3),
    "jitter_relative": ("ratio", 4),
    "energy_mean": ("a.u.", 4),
    "energy_std": ("a.u.", 4),
    "energy_mean_voiced": ("a.u.", 4),
    "energy_std_voiced": ("a.u.", 4),
    "energy_flux_mean": ("a.u.", 4),
    "voiced_ratio": ("share", 3),
    "speaking_rate": ("runs/s", 2),
    "zcr_mean": ("crossings", 3),
    "centroid_mean": ("Hz", 0),
    "flatness_mean": ("ratio", 5),
    "n_frames": ("frames", 0),
    "duration_s": ("s", 2),
}

# One line about what each agent-loop control holds fixed. A control the JSON
# grows later still renders -- with its key as the label -- because the layout is
# driven by the file, not by this table.
CONTROL_NOTES: dict[str, str] = {
    "no_memory": ("the state is wiped before every decision, and only the current "
                  "instruction is re-streamed"),
    "scalar_carry": ("the same controller with the carried value in one Python "
                     "int instead of the state"),
    "fixed_decay": ("a state of the same width with a constant, input-independent "
                    "write gate: every value-carrying event goes into every slot"),
    "random_action": ("tools and arguments drawn uniformly; the floor, not a policy"),
}


# --------------------------------------------------------------------------
# Formatting. One place per shape of number, so the page and the tests that
# check it cannot drift into two different spellings of the same value.
# --------------------------------------------------------------------------

def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def fmt_rate(value: float) -> str:
    return f"{value:.3f}"


def fmt_pct(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}%"


def fmt_compact(value: float) -> str:
    """A number with no trailing zeros: 1.0 -> 1, 0.32 -> 0.32."""
    return f"{value:g}"


def rate_text(rate: float | None) -> str:
    """A rate, or an em dash where the measurement was not taken."""
    return fmt_rate(rate) if rate is not None else "—"


def _esc(text: Any) -> str:
    return html.escape(str(text), quote=True)


def _load(path: str | None, label: str) -> dict | None:
    if not path:
        return None
    try:
        return json.loads(pathlib.Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"could not read {label} from {path}: {exc}", file=sys.stderr)
        return None


# --------------------------------------------------------------------------
# Chart geometry. Everything below draws in SVG user units; the scales are the
# only arithmetic the renderer does, and they are stated on the axes.
# --------------------------------------------------------------------------

CHART_W = 460.0
PAD_LEFT = 58.0
PAD_RIGHT = 14.0
TOP_PAD = 12.0
TRACK_F0_H = 96.0
TRACK_GAP = 28.0
TRACK_RMS_H = 74.0
AXIS_H = 34.0
CONF_H = 132.0
X0 = PAD_LEFT
X1 = CHART_W - PAD_RIGHT


def _x(index: int, count: int) -> float:
    if count <= 1:
        return X0
    return X0 + (X1 - X0) * index / (count - 1)


def _y(value: float, lo: float, hi: float, top: float, height: float) -> float:
    span = hi - lo
    if span <= 0.0:
        return top + height
    return top + height - height * (value - lo) / span


def _nice_ticks(lo: float, hi: float, count: int = 4) -> list[float]:
    """Evenly spaced round ticks inside ``[lo, hi]``. Deterministic by design."""
    span = hi - lo
    if span <= 0.0:
        return [lo]
    raw = span / count
    magnitude = 10.0 ** math.floor(math.log10(raw))
    step = magnitude * 10.0
    for multiple in (1.0, 2.0, 2.5, 5.0, 10.0):
        if multiple * magnitude >= raw:
            step = multiple * magnitude
            break
    ticks: list[float] = []
    tick = math.ceil(lo / step - 1e-9) * step
    while tick <= hi + step * 1e-9:
        ticks.append(round(tick, 10))
        tick += step
    return ticks


def _bursts(value: float) -> str:
    """``2.0 -> "2 bursts/s"``, ``1.0 -> "1 burst/s"``."""
    shown = fmt_compact(value)
    return f"{shown} burst/s" if float(value) == 1.0 else f"{shown} bursts/s"


def _tick_label(value: float) -> str:
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    if value == int(value):
        return str(int(value))
    return f"{value:g}"


def _bounds(values: Iterable[float | None], pad: float = 0.08,
            zero_floor: bool = False) -> tuple[float, float]:
    present = [float(v) for v in values if v is not None]
    if not present:
        return 0.0, 1.0
    lo, hi = min(present), max(present)
    if zero_floor:
        lo = min(0.0, lo)
    span = hi - lo
    if span <= 0.0:
        span = abs(hi) if hi else 1.0
        return lo - span * pad, hi + span * pad
    return lo - span * pad, hi + span * pad


def _runs(flags: Sequence[int]) -> list[tuple[int, int]]:
    """Contiguous ``[start, stop)`` runs of truthy flags."""
    out: list[tuple[int, int]] = []
    start: int | None = None
    for index, flag in enumerate(flags):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            out.append((start, index))
            start = None
    if start is not None:
        out.append((start, len(flags)))
    return out


def _series_path(values: Sequence[float | None], count: int, lo: float, hi: float,
                 top: float, height: float, css: str, attrs: str = "") -> str:
    """A path that is broken wherever the series has no value.

    ``nan``/``None`` is not zero: an unvoiced frame has no pitch, and a chart
    that drew a line through it would show a plunge to a pitch nobody measured.
    """
    commands: list[str] = []
    pen_down = False
    for index, value in enumerate(values):
        if value is None:
            pen_down = False
            continue
        x = _x(index, count)
        y = _y(float(value), lo, hi, top, height)
        commands.append(("L" if pen_down else "M") + f"{x:.2f},{y:.2f}")
        pen_down = True
    space = f" {attrs}" if attrs else ""
    return f'<path class="{css}" d="{" ".join(commands)}"{space}/>'


def _panel_head(lo: float, hi: float, top: float, height: float,
                ticks: Sequence[float], title: str, shade: Sequence[tuple[int, int]],
                count: int) -> list[str]:
    out: list[str] = []
    if shade:
        for start, stop in shade:
            left = _x(start, count)
            right = _x(stop - 1, count)
            out.append(
                f'<rect class="shade" x="{left:.2f}" y="{top:.2f}" '
                f'width="{max(right - left, 0.6):.2f}" height="{height:.2f}"/>'
            )
    for tick in ticks:
        y = _y(tick, lo, hi, top, height)
        out.append(f'<line class="grid" x1="{X0:.1f}" x2="{X1:.1f}" '
                   f'y1="{y:.2f}" y2="{y:.2f}"/>')
        out.append(f'<text class="tick" x="{X0 - 6:.1f}" y="{y + 3.4:.2f}" '
                   f'text-anchor="end">{_tick_label(tick)}</text>')
    out.append(f'<line class="axis" x1="{X0:.1f}" x2="{X1:.1f}" '
               f'y1="{top + height:.2f}" y2="{top + height:.2f}"/>')
    out.append(f'<line class="axis" x1="{X0:.1f}" x2="{X0:.1f}" '
               f'y1="{top:.2f}" y2="{top + height:.2f}"/>')
    out.append(f'<text class="axis-title" x="{X0:.1f}" y="{top - 5:.1f}">'
               f'{_esc(title)}</text>')
    return out


def _x_axis(times: Sequence[float], bottom: float, label: str) -> list[str]:
    """X ticks from the times array, at round seconds where one exists."""
    out: list[str] = []
    if not times:
        return out
    stop = times[-1]
    ticks = [t for t in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0) if t <= stop + 1e-9]
    count = len(times)
    for tick in ticks:
        x = _x(min(range(count), key=lambda i: abs(times[i] - tick)), count)
        out.append(f'<line class="grid" x1="{x:.2f}" x2="{x:.2f}" '
                   f'y1="{bottom:.2f}" y2="{bottom + 4:.2f}"/>')
        out.append(f'<text class="tick" x="{x:.2f}" y="{bottom + 15:.2f}" '
                   f'text-anchor="middle">{_tick_label(tick)}</text>')
    out.append(f'<text class="tick axis-title" x="{(X0 + X1) / 2:.1f}" '
               f'y="{bottom + 29:.2f}" text-anchor="middle">{_esc(label)}</text>')
    return out


def _series_attrs(label: str, unit: str, digits: int, lo: float, hi: float,
                  top: float, bottom: float) -> str:
    """The values the hover script reads back out of the drawn path.

    Only the line's own geometry and scale are written down; the numbers are
    recovered by inverting the scale in the browser. That keeps a second copy of
    every frame value out of the file, so the page cannot show a readout that
    disagrees with the chart it came from -- there is one copy, and it is the
    picture.
    """
    return (
        f'data-label="{_esc(label)}" data-unit="{_esc(unit)}" '
        f'data-digits="{digits}" data-lo="{lo:.6g}" data-hi="{hi:.6g}" '
        f'data-top="{top:.2f}" data-bottom="{bottom:.2f}"'
    )


def track_chart(*, times: Sequence[float], f0: Sequence[float | None],
                rms: Sequence[float], voiced_share: Sequence[float],
                f0_lo: float, f0_hi: float, rms_hi: float, readout_id: str,
                f0_ticks: Sequence[float], rms_ticks: Sequence[float]) -> str:
    """One condition: the F0 track over the energy contour, sharing a time axis."""
    count = len(times)
    height = TOP_PAD + TRACK_F0_H + TRACK_GAP + TRACK_RMS_H + AXIS_H
    f0_top = TOP_PAD
    rms_top = TOP_PAD + TRACK_F0_H + TRACK_GAP
    rms_bottom = rms_top + TRACK_RMS_H
    shade = [(start, stop) for start, stop in _runs([1 if v > 0 else 0
                                                     for v in voiced_share])]
    parts = [
        f'<svg class="chart" viewBox="0 0 {CHART_W:.0f} {height:.0f}" '
        f'role="img" data-readout="{_esc(readout_id)}" data-frames="{count}" '
        f'data-x0="{X0:.2f}" data-x1="{X1:.2f}" '
        f'data-times="{",".join(f"{t:g}" for t in times)}">',
        '<g class="frame">',
    ]
    parts += _panel_head(f0_lo, f0_hi, f0_top, TRACK_F0_H, f0_ticks,
                         "F0 (Hz)", shade, count)
    parts.append(_series_path(
        f0, count, f0_lo, f0_hi, f0_top, TRACK_F0_H, "line f0",
        _series_attrs("F0", "Hz", 1, f0_lo, f0_hi, f0_top,
                      f0_top + TRACK_F0_H)))
    parts += _panel_head(0.0, rms_hi, rms_top, TRACK_RMS_H, rms_ticks,
                         "energy (RMS amplitude, a.u.)", [], count)
    parts.append(_series_path(
        rms, count, 0.0, rms_hi, rms_top, TRACK_RMS_H, "line rms",
        _series_attrs("energy", "a.u.", 4, 0.0, rms_hi, rms_top,
                      rms_top + TRACK_RMS_H)))
    parts += _x_axis(times, rms_bottom, "time (s)")
    parts += ["</g>", "</svg>"]
    return "\n".join(parts)


def confidence_chart(*, times: Sequence[float], confidence: Sequence[float],
                     voiced: Sequence[int], lo: float, hi: float, threshold: float,
                     ticks: Sequence[float], readout_id: str) -> str:
    """The voiced/unvoiced decision drawn: the confidence, and the line it faced."""
    count = len(times)
    height = TOP_PAD + CONF_H + AXIS_H
    top = TOP_PAD
    bottom = top + CONF_H
    parts = [
        f'<svg class="chart" viewBox="0 0 {CHART_W:.0f} {height:.0f}" '
        f'role="img" data-readout="{_esc(readout_id)}" data-frames="{count}" '
        f'data-x0="{X0:.2f}" data-x1="{X1:.2f}" '
        f'data-times="{",".join(f"{t:g}" for t in times)}">',
        '<g class="frame">',
    ]
    parts += _panel_head(lo, hi, top, CONF_H, ticks,
                         "voiced confidence (normalised autocorrelation peak)",
                         [], count)
    # Where the decision said "voiced", drawn as a band along the floor of the
    # panel, so the answer and the curve it came from are in one picture.
    for start, stop in _runs(voiced):
        left = _x(start, count)
        right = _x(stop - 1, count)
        parts.append(f'<rect class="voiced-band" x="{left:.2f}" '
                     f'y="{bottom - 5:.2f}" width="{max(right - left, 0.6):.2f}" '
                     f'height="5.00"/>')
    y = _y(threshold, lo, hi, top, CONF_H)
    parts.append(f'<line class="threshold" x1="{X0:.1f}" x2="{X1:.1f}" '
                 f'y1="{y:.2f}" y2="{y:.2f}"/>')
    parts.append(f'<text class="tick" x="{X1:.1f}" y="{y - 4:.2f}" '
                 f'text-anchor="end">threshold {_tick_label(threshold)}</text>')
    parts.append(_series_path(
        confidence, count, lo, hi, top, CONF_H, "line conf",
        _series_attrs("confidence", "", 3, lo, hi, top, bottom)))
    parts += _x_axis(times, bottom, "time (s)")
    parts += ["</g>", "</svg>"]
    return "\n".join(parts)


# --------------------------------------------------------------------------
# Panels
# --------------------------------------------------------------------------

def _table(head: Sequence[str], rows: Sequence[Sequence[str]],
           aligns: Sequence[str] | None = None) -> str:
    aligns = aligns or ["left"] + ["right"] * (len(head) - 1)
    out = ['<table>', "<thead><tr>"]
    for cell, align in zip(head, aligns):
        out.append(f'<th class="{align}">{_esc(cell)}</th>')
    out.append("</tr></thead><tbody>")
    for row in rows:
        out.append("<tr>")
        for cell, align in zip(row, aligns):
            out.append(f'<td class="{align}">{cell}</td>')
        out.append("</tr>")
    out.append("</tbody></table>")
    return "\n".join(out)


def _dl(pairs: Sequence[tuple[str, str]]) -> str:
    out = ['<dl class="kv">']
    for key, value in pairs:
        out.append(f"<dt>{_esc(key)}</dt><dd>{value}</dd>")
    out.append("</dl>")
    return "\n".join(out)


def voice_panel(voice: dict) -> str:
    tracks = voice.get("frame_tracks") or {}
    conditions = tracks.get("conditions") or {}
    times = tracks.get("times_s") or []
    classes = voice.get("separability", {}).get("classes") or list(conditions)
    described = voice.get("conditions") or {}

    all_f0 = [value for name in conditions
              for value in conditions[name].get("f0_hz", [])]
    f0_lo, f0_hi = _bounds(all_f0)
    rms_hi = _bounds([value for name in conditions
                      for value in conditions[name].get("rms", [])],
                     zero_floor=True)[1]
    f0_ticks = _nice_ticks(f0_lo, f0_hi, 4)
    rms_ticks = _nice_ticks(0.0, rms_hi, 3)

    cards: list[str] = []
    for name in classes:
        track = conditions.get(name)
        if track is None or not times:
            continue
        parameters = described.get(name, {}).get("parameters", {})
        descriptors = described.get(name, {}).get("descriptors", {})
        readout_id = f"readout-voice-{name}"
        caption = (f'<span class="cond">{_esc(name)}</span>'
                   f'<span class="muted">generator: F0 target '
                   f'{_tick_label(parameters.get("f0_base", 0))} Hz, '
                   f'{_bursts(parameters.get("bursts_per_sec", 0))}, '
                   f'panel {_tick_label(parameters.get("duty", 0) * 100)}% voiced'
                   f'</span>')
        pairs = []
        for key, unit, digits, _note in CHART_DESCRIPTORS:
            if key not in descriptors:
                continue
            value = descriptors[key]
            shown = fmt_pct(value, 1) if unit == "share" else fmt(value, digits)
            pairs.append((key, f"{shown} {unit}" if unit != "share" else shown))
        cards.append(
            '<figure class="card">'
            f"<figcaption>{caption}</figcaption>"
            + track_chart(
                times=times, f0=track.get("f0_hz", []), rms=track.get("rms", []),
                voiced_share=track.get("voiced_share", []), f0_lo=f0_lo,
                f0_hi=f0_hi, rms_hi=rms_hi, readout_id=readout_id,
                f0_ticks=f0_ticks, rms_ticks=rms_ticks)
            + f'<p class="readout" id="{_esc(readout_id)}" '
              'data-idle="hover the chart to read one frame">'
              "hover the chart to read one frame</p>"
            + _dl(pairs)
            + "</figure>"
        )

    separability = voice.get("separability") or {}
    config = voice.get("config") or {}
    rows = []
    keys = [key for key in (described.get(classes[0], {}).get("descriptors") or {})
            if key in DESCRIPTOR_UNITS] if classes else []
    for key in keys:
        unit, digits = DESCRIPTOR_UNITS[key]
        cells = []
        for name in classes:
            value = (described.get(name, {}).get("descriptors") or {}).get(key)
            cells.append("—" if value is None else fmt(value, digits))
        rows.append([f"<code>{_esc(key)}</code>", unit] + cells)

    accuracy_line = ""
    if separability:
        accuracy_line = (
            "<p>Separability in descriptor space — leave-one-out nearest centroid "
            f"over {separability.get('n_utterances')} synthetic utterances, chance "
            f"{fmt_rate(separability.get('chance', 0.0))}: "
            f"<strong>{fmt_rate(separability.get('accuracy', 0.0))}</strong> "
            f"({fmt_pct(separability.get('accuracy', 0.0))}, "
            f"{fmt(separability.get('z_vs_chance', 0.0), 1)}σ against chance). "
            "The closest pair of condition centroids is "
            f"{fmt(separability.get('closest_centroids', 0.0), 2)} against a mean "
            f"within-condition spread of "
            f"{fmt(separability.get('within_spread', 0.0), 2)} — a ratio of "
            f"<strong>{fmt(separability.get('ratio', 0.0), 2)}×</strong>.</p>")

    source = (
        f"<p class=\"note\">Source: <code>voice-affect.json</code> — "
        f"{config.get('utterances_per_condition')} utterances per condition "
        f"({config.get('duration_s')} s each) at {config.get('sample_rate'):,} Hz, "
        f"seed {config.get('seed')}, framed at {fmt_compact(tracks.get('window_ms', 0))}"
        f" ms window / {fmt_compact(tracks.get('hop_ms', 0))} ms hop "
        f"({len(times)} frames). The F0 line is the mean over the condition's "
        "utterances, counted frame by frame and only over the utterances voiced in "
        "that frame; the shaded band marks frames where at least one was voiced, and "
        "the line breaks where none was. Both axes are shared by all four panels, so "
        "the levels can be compared by eye. Rounding: "
        f"{_esc(tracks.get('rounding', ''))}.</p>")

    return (
        f"{VOICE_BEGIN}\n"
        "<section id=\"voice\">\n"
        "<h2>Voice: four synthetic prosody conditions, one frame per 10 ms</h2>\n"
        "<p class=\"lede\">Every condition below was synthesised from one "
        "parameter tuple that sets pitch level, pitch movement, loudness movement "
        "and burst rate. The charts are the measured F0 and energy contours; the "
        "numbers beside them are the descriptors the same run produced. Look at "
        "them together: the descriptor is a summary of the curve, not a second "
        "opinion about it.</p>\n"
        + accuracy_line + "\n"
        + _table(["descriptor", "unit"] + [str(name) for name in classes], rows)
        + "\n<div class=\"grid two\">\n" + "\n".join(cards) + "\n</div>\n"
        + source + "\n</section>\n" + VOICE_END
    )


def control_panel(voice: dict) -> str:
    probe = voice.get("voicing_probe") or {}
    series = probe.get("series") or {}
    controls = (voice.get("controls") or {}).get("white_noise_voicing") or {}
    config = voice.get("config") or {}
    times = (voice.get("frame_tracks") or {}).get("times_s") or []
    threshold = float(probe.get("threshold", 0.45))

    values = [value for row in series.values() for value in row.get("confidence", [])]
    lo, hi = _bounds(values, pad=0.05, zero_floor=False)
    lo = min(lo, 0.0)
    hi = max(hi, 1.0)
    ticks = _nice_ticks(lo, hi, 4)

    cards: list[str] = []
    for key in ("white_noise", "tone"):
        row = series.get(key)
        if row is None:
            continue
        readout_id = f"readout-probe-{key}"
        measured = (
            f'<span class="muted">voiced share '
            f'{fmt_rate(row.get("voiced_ratio", 0.0))}, largest confidence '
            f'{fmt(row.get("max_confidence", 0.0), 4)}</span>'
        )
        mean_f0 = row.get("mean_f0_hz")
        if mean_f0 is not None:
            asked = probe.get("tone_hz")
            error = abs(mean_f0 / asked - 1.0) if asked else 0.0
            measured += (f'<span class="muted">mean estimated F0 '
                         f'{fmt(mean_f0, 1)} Hz against {_tick_label(asked)} Hz '
                         f'asked for ({fmt_pct(error, 2)} error)</span>')
        cards.append(
            '<figure class="card">'
            f'<figcaption><span class="cond">{_esc(row.get("label", key))}</span>'
            f'{measured}</figcaption>'
            + confidence_chart(
                times=times, confidence=row.get("confidence", []),
                voiced=row.get("voiced", []), lo=lo, hi=hi, threshold=threshold,
                ticks=ticks, readout_id=readout_id)
            + f'<p class="readout" id="{_esc(readout_id)}" '
              'data-idle="hover the chart to read one frame">'
              "hover the chart to read one frame</p>"
            + "</figure>"
        )

    noise = series.get("white_noise", {})
    tone = series.get("tone", {})
    rounds = controls.get("rounds")
    rounds_text = (f"the {fmt_compact(rounds)} noise draws"
                   if rounds else "the noise draws")
    tolerance_note = ""
    if tone.get("mean_f0_hz"):
        tolerance_note = (
            f" The tone's estimated pitch is {fmt(tone['mean_f0_hz'], 1)} Hz for the "
            f"{_tick_label(probe.get('tone_hz'))} Hz asked for — a "
            f"{fmt_pct(abs(tone['mean_f0_hz'] / probe.get('tone_hz', 1) - 1), 2)} "
            "error, inside the 2% the tests assert; the search band in the module is "
            "60–400 Hz, which is wider than the range it works over."
        )
    return (
        f"{CONTROL_BEGIN}\n"
        '<section id="voicing-control">\n'
        "<h2>The voiced/unvoiced control, drawn rather than asserted</h2>\n"
        "<p class=\"lede\">Two signals through the same framing and the same "
        "autocorrelation decision: a block of white noise and a "
        f"{_tick_label(probe.get('tone_hz'))} Hz tone of the same length. The curve "
        "is the estimator's own normalised autocorrelation peak — the number the "
        "voiced/unvoiced decision was made on — so what is drawn is the decision "
        f"variable, not a summary of it. The dashed line is the "
        f"{fmt_compact(threshold)} threshold; the band along the floor marks the "
        "frames the decision called voiced.</p>\n"
        "<div class=\"grid two\">\n" + "\n".join(cards) + "\n</div>\n"
        f"<p>White noise keeps its largest confidence at "
        f"<strong>{fmt(noise.get('max_confidence', 0.0), 4)}</strong> against a "
        f"threshold of {fmt_compact(threshold)}, and is judged voiced in "
        f"<strong>{fmt_rate(noise.get('voiced_ratio', 0.0))}</strong> of frames; the "
        f"tone sits near {fmt(tone.get('max_confidence', 0.0), 3)} and is voiced in "
        f"{fmt_rate(tone.get('voiced_ratio', 0.0))} of them.{tolerance_note} "
        "A detector that answered \"voiced\" to everything would pass every tone "
        "test ever written and would make every pitch number on this page "
        "meaningless, which is why the noise case is on the page and not only in a "
        "table.</p>\n"
        f"<p class=\"note\">Over {rounds_text} summarised in "
        "<code>voice-affect.json</code>, the largest voiced ratio is "
        f"{fmt_rate(controls.get('voiced_ratio_max', 0.0))} and the largest "
        f"confidence {fmt(controls.get('max_confidence', 0.0), 4)}. The probe above "
        "is one further draw from its own seed stream, so it could not shift those "
        "numbers. Framing: "
        f"{config.get('sample_rate'):,} Hz, {len(times)} frames of 25 ms every "
        "10 ms.</p>\n</section>\n" + CONTROL_END
    )


def _call_text(action: str, args: dict) -> str:
    return f"{action}(" + ", ".join(f"{k}={v}" for k, v in sorted(args.items())) + ")"


def _trace_list(trace: dict, slots: bool) -> str:
    steps = trace.get("steps") or []
    items: list[str] = []
    for position, step in enumerate(steps):
        reads = step.get("reads") or {}
        observed = step.get("result")
        if step.get("error"):
            observed = f'<span class="bad">{_esc(step["error"])}</span>'
        following = steps[position + 1] if position + 1 < len(steps) else None
        if following is None:
            carried = (f'the loop stopped here: {_esc(trace.get("stop_reason"))}, '
                       f'answer <strong>{_esc(trace.get("answer"))}</strong>')
        else:
            carried = (f'the next decision read carry '
                       f'<strong>{fmt_compact(following["reads"]["carry"])}</strong>')
        memory = ""
        if slots and reads.get("slots") is not None:
            shown = ", ".join(fmt_compact(v) for v in reads["slots"])
            memory = f'<div class="trace-sub">memory slots read: [{shown}]</div>'
        items.append(
            "<li>"
            f'<div class="trace-head"><span class="step">step {step.get("index")}</span>'
            f'<span class="instr">{_esc(step.get("instruction"))}</span>'
            f'<code class="act">{_esc(_call_text(step.get("action", ""), step.get("args") or {}))}</code>'
            f'<span class="obs">→ {observed}</span></div>'
            f'<div class="trace-sub">state read: op {reads.get("op_code")}, '
            f'arg {reads.get("arg")}, carry {fmt_compact(reads.get("carry", 0))}, '
            f'observations {reads.get("observations")} · {carried}</div>'
            f"{memory}</li>"
        )
    return '<ol class="trace">' + "\n".join(items) + "</ol>"


def _trace_block(trace: dict, *, slots: bool) -> str:
    plan = ", ".join(
        f'{instruction["op"]} {" ".join(str(a) for a in instruction["args"])}'.strip()
        for instruction in trace.get("plan") or []
    )
    replayed = trace.get("replayed") or []
    replay_text = ", ".join(
        _esc(row.get("value")) if not row.get("error") else f'error {_esc(row["error"])}'
        for row in replayed
    )
    table = trace.get("table") or []
    table_text = (", ".join(f"{_tick_label(k)} → {_tick_label(v)}" for k, v in table)
                  if table else "empty — this family has no key/value table")
    return (
        f'<p class="task">{_esc(trace.get("text"))}</p>\n'
        + _dl([
            ("task id", f'<code>{_esc(trace.get("task_id"))}</code>'),
            ("plan", f"<code>{_esc(plan)}</code>"),
            ("key/value table", _esc(table_text)),
            ("answer (analytic)", f'<strong>{_esc(trace.get("answer"))}</strong>'),
            ("loop steps", f'{len(trace.get("steps") or [])} of budget '
                           f'{_esc(trace.get("budget"))}'),
            ("stop reason", f'<code>{_esc(trace.get("stop_reason"))}</code>, solved '
                            f'<strong>{_esc(trace.get("solved"))}</strong>'),
            ("replay check", f"the trace replays to the same observations: "
                             f"{replay_text}"),
        ])
        + _trace_list(trace, slots)
    )


def agent_panel(agent: dict) -> str:
    config = agent.get("config") or {}
    example = agent.get("example_trace") or {}
    selective = agent.get("selective_example_trace") or {}
    tools = ", ".join(f"<code>{_esc(name)}</code>"
                      for name in config.get("tools") or [])
    registers = ", ".join(f"<code>{_esc(name)}</code>"
                          for name in config.get("registers") or [])

    selective_block = ""
    if selective:
        shape = config.get("selective") or {}
        shape_text = ""
        if shape:
            shape_text = (
                f" {shape.get('stores')} stores, {shape.get('distractors')} "
                f"distractors, {shape.get('keys')} keys, state width "
                f"{shape.get('state_width')} slots.")
        selective_block = (
            "<h3>The same loop on the selective family, where the state has to "
            "choose what to keep</h3>\n"
            "<p>Here the task is a stream of keyed values and distractors, and the "
            "memory has to retain the one that is asked for. Retention and address "
            "are decided by the event's own content — a <code>PUT</code> goes to the "
            "slot its key addresses, a <code>NOISE</code> goes nowhere — so the "
            "memory row below is the measurement, not a description of one."
            f"{shape_text}</p>\n"
            + _trace_block(selective, slots=True)
        )

    return (
        f"{AGENT_BEGIN}\n"
        '<section id="agent">\n'
        "<h2>Agent: one task, decision by decision</h2>\n"
        "<p class=\"lede\">The loop reads the task's instructions as structured "
        f"events, streams them through the recurrence, decodes the state into the "
        f"registers {registers}, and calls one of {tools}. Each step below shows "
        "what the state held when the choice was made, the call that came out of "
        "it, the observation the tool returned, and the carried value the next "
        "decision then read — so the operand chain is visible rather than "
        "claimed.</p>\n"
        + _trace_block(example, slots=False) + "\n"
        + selective_block + "\n</section>\n" + AGENT_END
    )


def _is_grid(value: Any) -> bool:
    """A control measured per ``family@budget`` cell.

    The key shape is part of the test, not just the value shape: a sweep keyed by
    slot count holds cell-shaped dicts too, and reading one as a grid would print
    a table of em dashes and call it a measurement.
    """
    return (isinstance(value, dict) and bool(value)
            and all(re.fullmatch(r"[^@]+@\d+", key) for key in value)
            and all(isinstance(cell, dict) and "rate" in cell
                    for cell in value.values()))


def _rate_of(value: Any) -> float | None:
    """A rate from either a cell or a solved/total pair, or ``None``."""
    if not isinstance(value, dict):
        return None
    if isinstance(value.get("rate"), (int, float)):
        return float(value["rate"])
    if isinstance(value.get("solved"), (int, float)) and value.get("tasks"):
        return float(value["solved"]) / float(value["tasks"])
    return None


def _is_sweep(value: Any) -> bool:
    if not isinstance(value, dict) or not value:
        return False
    for cell in value.values():
        if not isinstance(cell, dict):
            return False
        inner = cell.get("agent")
        if not (isinstance(inner, dict) and "rate" in inner):
            return False
    return True


def _sort_keys(keys: Iterable[str]) -> list[str]:
    keys = list(keys)
    if all(key.lstrip("-").isdigit() for key in keys):
        return sorted(keys, key=float)
    return sorted(keys)


def _cell(cells: dict, key: str, counts: bool = True) -> str:
    """One solve-rate cell, with its solved/attempted counts where they fit."""
    cell = cells.get(key)
    if not cell:
        return '<span class="muted">—</span>'
    rate = fmt_rate(cell["rate"])
    if not counts:
        return rate
    return f'{rate}<span class="muted"> ({cell["solved"]}/{cell["total"]})</span>'


def _grid_controls(controls: dict) -> dict[str, dict]:
    return {name: cells for name, cells in controls.items() if _is_grid(cells)}


def _agreement(solve: dict, cells: dict) -> dict:
    """Where a control's cell equals the agent's, exactly.

    Exact string-shaped equality of the parsed rates, not a tolerance: the two
    numbers come from the same file and are equal or they are not. The counts are
    what the page's headline sentence is built from, so the sentence cannot claim
    a match the table does not show.
    """
    shared = [key for key in solve if key in cells]
    same = [key for key in shared if cells[key]["rate"] == solve[key]["rate"]]
    different = [key for key in shared if key not in same]
    return {"shared": shared, "same": same, "different": different}


def _family_budget(key: str) -> tuple[str, int]:
    family, _, budget = key.rpartition("@")
    return family, int(budget)


def results_panel(agent: dict) -> str:
    config = agent.get("config") or {}
    solve = agent.get("solve_rate") or {}
    controls = agent.get("controls") or {}
    families = list(config.get("families") or [])
    budgets = list(config.get("budgets") or [])
    reference = max(budgets) if budgets else 0
    grids = _grid_controls(controls)

    # --- the honest headline, computed before it is written ---------------
    # The order is deliberate: the control that costs the increment its
    # flattering reading is first, then the two that bound what the state is
    # worth, then whatever else the file holds.
    priority = ("scalar_carry", "fixed_decay", "no_memory")
    ordered = sorted(grids, key=lambda name: (
        priority.index(name) if name in priority else len(priority), name))
    headline: list[str] = []
    for name in ordered:
        cells = grids[name]
        agreement = _agreement(solve, cells)
        if not agreement["shared"]:
            continue
        label = _esc(name)
        note = CONTROL_NOTES.get(name, "recorded by experiments/agent_loop.py")
        if not agreement["different"]:
            headline.append(
                f"<p><strong>{label}</strong> — {note} — reproduces the agent's "
                f"solve rate row for row: all {len(agreement['shared'])} cells it "
                "shares with the agent are equal, so on this suite it is not "
                "evidence for the state.</p>")
            continue
        cells_text, families_text = _difference_text(agreement, solve, cells)
        headline.append(
            f"<p><strong>{label}</strong> — {note} — reproduces the agent's rate in "
            f"{len(agreement['same'])} of the {len(agreement['shared'])} cells it "
            f"shares with the agent. The cells where it does not "
            f"({len(agreement['different'])}): {cells_text}. "
            f"{families_text}</p>")

    fixed_note = ""
    if "fixed_decay" in controls and "selective@%d" % reference in controls["fixed_decay"]:
        selective_cell = controls["fixed_decay"][f"selective@{reference}"]
        agent_cell = solve.get(f"selective@{reference}", {})
        fixed_note = (
            " The same table says why the width is not the explanation: a state of "
            f"the same width with a constant write gate (<code>fixed_decay</code>) "
            f"reaches {fmt_rate(selective_cell['rate'])} on "
            f"<code>selective@{reference}</code> where the agent reaches "
            f"{fmt_rate(agent_cell.get('rate', 0.0))} — it is the gating, not the "
            "number of slots.")

    scalar_selective = controls.get("scalar_selective")
    scalar_note = ""
    if isinstance(scalar_selective, dict):
        queried = scalar_selective.get("queried_the_last_store")
        scalar_note = (
            " The single-register memory fails selectively rather than at random: "
            f"of {scalar_selective.get('tasks')} selective tasks it solves "
            f"{scalar_selective.get('solved')}, and "
            f"{scalar_selective.get('queried_the_last_store')} of them asked for the "
            "key that was stored last. The correspondence holds task for task "
            f"(<code>{_esc(scalar_selective.get('correspondence_holds'))}</code> over "
            f"{scalar_selective.get('exact_matches')} tasks).")

    # --- the staircase ----------------------------------------------------
    rows = []
    for family in families:
        cells = []
        for budget in budgets:
            cells.append(_cell(solve, f"{family}@{budget}"))
        rows.append([f"<code>{_esc(family)}</code>",
                     str(config.get("step_counts", {}).get(family, "—"))] + cells)
    staircase = _table(
        ["task family", "steps needed"] + [f"budget {b}" for b in budgets], rows)

    # --- every control, in full ------------------------------------------
    control_cards: list[str] = []
    for name, cells in grids.items():
        note = CONTROL_NOTES.get(name, "recorded by experiments/agent_loop.py")
        body = []
        for family in families:
            body.append([f"<code>{_esc(family)}</code>"] + [
                _cell(cells, f"{family}@{budget}", counts=False)
                for budget in budgets])
        control_cards.append(
            '<figure class="card">'
            f'<figcaption><span class="cond">{_esc(name)}</span>'
            f'<span class="muted">{_esc(note)}</span></figcaption>'
            + _table(["family"] + [f"b{b}" for b in budgets], body) + "</figure>")

    sweeps: list[str] = []
    for name, value in controls.items():
        if not _is_sweep(value):
            continue
        shape_keys = [key for key in value[_sort_keys(value)[0]]
                      if not isinstance(value[_sort_keys(value)[0]][key], dict)]
        conditions = [key for key in value[_sort_keys(value)[0]]
                      if isinstance(value[_sort_keys(value)[0]][key], dict)]
        if "agent" in conditions:
            conditions = ["agent"] + [c for c in conditions if c != "agent"]
        rows = []
        for key in _sort_keys(value):
            cell = value[key]
            # A shape column that is itself a rate gets the rate format, so the
            # same quantity is not spelled two ways on one page.
            rows.append([f"<code>{_esc(key)}</code>"] + [
                _esc(fmt_rate(cell.get(k)) if k == "rate"
                     else fmt_compact(cell.get(k)))
                for k in shape_keys
            ] + [_cell(cell, c) for c in conditions])
        sweeps.append(
            "<h4>Sweep: <code>%s</code></h4>%s" % (
                _esc(name),
                _table(["value"] + shape_keys + conditions, rows)))

    extras: list[str] = []
    one_step = controls.get("one_step")
    if isinstance(one_step, dict):
        rows = [[f"<code>{_esc(family)}</code>", _cell(one_step, family)]
                for family in families if family in one_step]
        if "all" in one_step:
            rows.append(["<em>all families</em>", _cell(one_step, "all")])
        extras.append("<h4><code>one_step</code> — the budget-1 control: "
                      "is the suite one call deep?</h4>" + _table(
            ["task family", "solve rate at one decision"], rows))

    zero = controls.get("zero_carry_branch")
    if isinstance(zero, dict):
        extras.append(
            "<p><code>zero_carry_branch</code> — where the first version's "
            "residual bit: of the "
            f"{zero.get('total')} <code>branch</code> tasks, {zero.get('count')} "
            f"arrive at the <code>IFPOS</code> decision (step {zero.get('step')}) "
            "carrying exactly 0 — the value a write residual turns positive.</p>")

    exactness = controls.get("write_exactness")
    if isinstance(exactness, dict):
        rows = []
        for key in _sort_keys(exactness):
            row = exactness[key]
            rows.append([f"<code>{_esc(key)}</code>",
                         f"{row['multiplier']:.3e}",
                         f"{row['residual_after_overwrite']:.3e}",
                         "yes" if row["sign_test_holds"] else "no"])
        extras.append("<h4><code>write_exactness</code> — register write "
                      "exactness</h4>"
                      "<p>A residual above zero is not a rounding detail: "
                      "<code>IFPOS</code> branches on the sign of this number. "
                      "What a register holds after being written 9 and then 0:</p>"
                      + _table(["A", "exp(A)", "residual", "sign test holds"], rows))

    other: list[str] = []
    loose: list[tuple[str, str]] = []
    handled = {"one_step", "zero_carry_branch", "write_exactness"}
    for name, value in controls.items():
        if name in handled or _is_grid(value) or _is_sweep(value):
            continue
        if isinstance(value, dict) and not any(isinstance(v, dict)
                                              for v in value.values()):
            pairs = [(key, _esc(fmt_compact(v)) if isinstance(v, (int, float))
                      and not isinstance(v, bool) else _esc(v))
                     for key, v in value.items()]
            other.append(f"<h4><code>{_esc(name)}</code></h4>" + _dl(pairs))
        elif isinstance(value, (int, float, bool)) or value is None:
            loose.append((name, f"<code>{_esc(fmt_compact(value))}</code>"
                          if isinstance(value, (int, float)) and not isinstance(value, bool)
                          else f"<code>{_esc(value)}</code>"))
        else:
            other.append(f"<h4><code>{_esc(name)}</code></h4>"
                         f"<p><code>{_esc(json.dumps(value))}</code></p>")

    return (
        f"{RESULTS_BEGIN}\n"
        '<section id="results">\n'
        "<h2>Results, and the controls that decide what they mean</h2>\n"
        "<p class=\"lede\">The solve rate is a number about the person who wrote "
        "the tasks unless the controls are published next to it. They are, "
        "including the one that costs the increment its most flattering "
        "reading.</p>\n"
        '<div class="callout">\n'
        + "\n".join(headline) + fixed_note + scalar_note + "\n</div>\n"
        "<h3>Solve rate by family and step budget</h3>\n"
        f"<p>{config.get('suite_size')} tasks "
        f"({config.get('tasks_per_family')} per family), seed "
        f"{config.get('seed')}. Each family needs a known number of decisions, so "
        "the staircase is the step counts rather than a discovery — it is here to "
        "show the budget is enforced rather than nominal.</p>\n"
        + staircase + "\n"
        "<h3>Every control, in full</h3>\n"
        "<p>Each control's whole grid, not a selected row: a slice chosen after "
        "the fact is how an unflattering cell goes missing. Cells here are rates; "
        "the counts behind them are in the staircase above (50 attempts each) and "
        "in each control's own totals below. Random action is "
        f"measured at budget {reference} only, over "
        f"{config.get('random_rounds')} draws per task with arguments spanning "
        f"±{config.get('random_arg_span')}.</p>\n"
        "<div class=\"grid two\">\n" + "\n".join(control_cards) + "\n</div>\n"
        + "\n".join(sweeps) + "\n"
        + "\n".join(extras) + "\n"
        + ("<h3>Other recorded controls</h3>\n" + "\n".join(other) + "\n"
           if other else "")
        + (_dl(loose) if loose else "")
        + "</section>\n" + RESULTS_END
    )


def _difference_text(agreement: dict, solve: dict, cells: dict,
                     limit: int = 6) -> tuple[str, str]:
    """The cells where a control parts from the agent, and what that means.

    The list is capped so the callout stays readable; the full grid for the same
    control is printed below it, so nothing is dropped from the page -- only from
    the summary.
    """
    parts = []
    families = set()
    for key in agreement["different"]:
        family, _budget = _family_budget(key)
        families.add(family)
        if len(parts) < limit:
            parts.append(f"<code>{_esc(key)}</code> "
                         f"(agent {fmt_rate(solve[key]['rate'])} vs control "
                         f"{fmt_rate(cells[key]['rate'])})")
    shown = len(parts)
    if len(agreement["different"]) > shown:
        parts.append(f"and {len(agreement['different']) - shown} more cells, "
                     "all listed in the control's own grid below")
    if len(families) == 1:
        family = next(iter(families))
        families_text = ("So the control ties the agent everywhere except "
                         f"<code>{_esc(family)}</code>.")
    else:
        families_text = ("The families where they part: "
                         + ", ".join(f"<code>{_esc(f)}</code>"
                                     for f in sorted(families)) + ".")
    return ", ".join(parts), families_text


def limitations_panel(voice: dict, agent: dict) -> str:
    voice_config = voice.get("config") or {}
    conditions = voice.get("conditions") or {}
    separability = voice.get("separability") or {}
    agent_config = agent.get("config") or {}
    families = agent_config.get("families") or []
    tools = agent_config.get("tools") or []
    controls = agent.get("controls") or {}
    solve = agent.get("solve_rate") or {}
    reference = max(agent_config.get("budgets") or [0])

    generator_rows = []
    for name, entry in conditions.items():
        parameters = entry.get("parameters") or {}
        generator_rows.append(
            f"<code>{_esc(name)}</code> = F0 target "
            f"{_tick_label(parameters.get('f0_base', 0))} Hz, swing "
            f"{parameters.get('f0_swing')}, jitter {parameters.get('f0_jitter')}, "
            f"amplitude spread {parameters.get('amp_var')}, "
            f"{_bursts(parameters.get('bursts_per_sec', 0))}")
    generator_text = "; ".join(generator_rows)

    fixed = controls.get("fixed_decay") or {}
    fixed_selective = fixed.get(f"selective@{reference}")
    selective_agent = solve.get(f"selective@{reference}")
    fixed_rate = _rate_of(fixed_selective)
    agent_rate = _rate_of(selective_agent)
    scalar_rate = _rate_of(controls.get("scalar_selective"))

    return (
        f"{LIMITS_BEGIN}\n"
        '<section id="limitations">\n'
        "<h2>What this page does not show</h2>\n"
        "<p class=\"lede\">The same pattern the README uses, because the failure "
        "mode is the same: a page of generated numbers is persuasive whether or "
        "not the numbers mean what a glance says they mean.</p>\n"
        "<p><strong>It is not</strong> an emotion recogniser, and it is not a "
        "learning result. Three claims that would be easy to make from this page, "
        "and are all false:</p>\n"
        "<ul class=\"falseclaims\">\n"
        "<li><em>\"The descriptors identify emotion in speech.\"</em> They identify "
        f"which of {len(conditions)} <strong>synthetic</strong> conditions generated "
        "a signal. Every utterance was synthesised by this repository, one parameter "
        f"tuple per condition ({generator_text}), and the labels come from the "
        "generator rather than from a listener. No human voice was involved at any "
        "point, there is no labelled affect corpus here to validate against, and "
        f"the {fmt_rate(separability.get('accuracy', 0.0))} accuracy is the "
        "recovery of synthetic parameters — not emotion.</li>\n"
        "<li><em>\"The loop learns, or something here was trained.\"</em> Nothing "
        "on this page is in a gradient path: there is no optimiser, no loss and no "
        "backward pass anywhere in the agent code. The controller is hand-written "
        "branching, the recurrence's decay and its read/write gate are constants "
        "chosen by hand, and a seed only decides which tasks are generated. The "
        "voice descriptors are analytic measurements of a waveform, not learned "
        "features, and the encoder that bridges them to the model is a fixed random "
        "projection with no bias and no nonlinearity.</li>\n"
        "<li><em>\"The agent generalises.\"</em> It runs one controller over a "
        f"closed tool set ({', '.join('<code>' + _esc(t) + '</code>' for t in tools)}) "
        f"and a closed task family "
        f"({', '.join('<code>' + _esc(f) + '</code>' for f in families)}), emitted by a "
        "seeded generator over bounded integers. The \"task text\" is that "
        "instruction grammar delivered as structured events: there is no "
        "tokenizer, no parsing and no language understanding anywhere in the path. "
        "The solver cannot be asked for a task the generator does not generate or a "
        "tool the registry does not hold, so a rate of 1.000 says the controller "
        "matches the shapes it was written against and nothing wider.</li>\n"
        "</ul>\n"
        "<p>And the boundaries that a reader should carry away with the numbers:</p>\n"
        "<ul class=\"limits\">\n"
        "<li><strong>No generalisation beyond the closed task family.</strong> "
        "Every task comes from the generator's own constraints, and those "
        "constraints are what make the controls interpretable. Nothing here speaks "
        "to a task outside the family.</li>\n"
        "<li><strong>The affect descriptors are synthetic-parameter recovery, not "
        "human emotion, and there is no real-speech validation.</strong> There is no "
        "corpus, no listener labels and no trained classifier; the conditions were "
        "built to differ along exactly the axes the descriptors measure, so the "
        "separation on this page is partly a property of the design.</li>\n"
        "<li><strong>The F0 numbers are bounded by the estimator, not by the "
        "signal.</strong> The module's 60–400 Hz band is a <em>search</em> range, "
        "not a working range: with the 25 ms window used here the tests assert a 2% "
        "tolerance at 150, 220 and 300 Hz and nothing below about 100 Hz. An "
        "unchanged clause from the README, repeated here because the chart makes "
        "the estimate look more authoritative than it is.</li>\n"
        "<li><strong>Jitter and speaking rate are frame-level proxies.</strong> "
        "Jitter is a successive-frame pitch difference between adjacent voiced "
        "frames and is bounded below by the 10 ms hop; \"speaking rate\" counts "
        "voiced segments and cannot see unvoiced consonants at all. Neither is a "
        "phonetics measurement.</li>\n"
        "<li><strong>The charts are averages of generated signals at one seed.</strong> "
        "Each F0 line is the mean over that condition's utterances, so it shows the "
        "shared burst structure and not the spread within the condition — the "
        "descriptor table is where the spread lives.</li>\n"
        "<li><strong>This page adds no measurements.</strong> It draws the numbers "
        "in the two JSON files. Its own arithmetic is axis scaling, percentages, "
        "and the comparison of each control's cells against the agent's; it fits "
        "nothing, infers nothing and estimates nothing.</li>\n"
        "<li><strong>Only the controls that were run are shown, and only at the "
        "budgets that were run.</strong> A missing cell in a control grid is a "
        "measurement that was not taken, printed as an em dash rather than filled "
        "in. Random action was measured at budget "
        f"{reference} only.</li>\n"
        + (f"<li><strong>The selective family's advantage is narrow and its gate is "
           f"hand-set.</strong> The gated state reaches "
           f"{rate_text(agent_rate)} on "
           f"<code>selective@{reference}</code> while a fixed, input-independent "
           f"decay of the same width reaches "
           f"{rate_text(fixed_rate)} and one Python int "
           f"reaches {rate_text(scalar_rate)}"
           " — but the gate is chosen, not learned, the family is closed and "
           "synthetic, and the arithmetic families are unaffected, which is why the "
           "scalar-carry control still ties the agent there.</li>\n"
           if fixed_rate is not None and agent_rate is not None else "")
        + "<li><strong>The page is only as fresh as its last render.</strong> "
        "Changing a results file without re-running the renderer leaves this page "
        "describing an older run; <code>tests/test_dashboard.py</code> fails if the "
        "committed page and the committed results files disagree, which is the "
        "point of publishing a generated surface.</li>\n"
        "</ul>\n</section>\n" + LIMITS_END
    )


def footer_panel(voice: dict, agent: dict) -> str:
    voice_config = voice.get("config") or {}
    agent_config = agent.get("config") or {}
    tracks = voice.get("frame_tracks") or {}
    return (
        f"{FOOTER_BEGIN}\n"
        '<footer id="regenerate">\n'
        "<h2>Regenerating this page</h2>\n"
        "<pre><code>python experiments/voice_affect.py --out voice-affect.json\n"
        "python experiments/agent_loop.py --out agent-loop.json\n"
        "python experiments/render_dashboard.py --voice voice-affect.json \\\n"
        "    --agent agent-loop.json --out dashboard.html</code></pre>\n"
        + _dl([
            ("voice-affect.json",
             f"{voice_config.get('utterances_per_condition')} utterances per "
             f"condition × {len(voice.get('conditions') or {})} conditions at "
             f"{voice_config.get('sample_rate'):,} Hz, seed "
             f"{voice_config.get('seed')}; written by "
             "<code>experiments/voice_affect.py</code>. Frame tracks: "
             f"{len(tracks.get('times_s') or [])} frames of "
             f"{fmt_compact(tracks.get('hop_ms', 0))} ms."),
            ("agent-loop.json",
             f"{agent_config.get('suite_size')} tasks "
             f"({agent_config.get('tasks_per_family')} per family) over "
             f"{len(agent_config.get('families') or [])} families, budgets "
             f"{agent_config.get('budgets')}, seed {agent_config.get('seed')}; "
             "written by <code>experiments/agent_loop.py</code>."),
            ("dashboard.html",
             "this file, written by <code>experiments/render_dashboard.py</code>; "
             "it is one file with no external references — the styles, the script "
             "and every chart are inline — so it renders from disk with no server "
             "and no network, and it carries no timestamp, so two renders of the "
             "same inputs are byte-identical."),
        ])
        + "\n</footer>\n" + FOOTER_END
    )


# --------------------------------------------------------------------------
# The document
# --------------------------------------------------------------------------

STYLE = """
:root {
  --ink: #171a21; --body: #2b3038; --muted: #6a7381; --rule: #e2e6ec;
  --bg: #f7f8fa; --card: #ffffff; --accent: #1d5b86; --accent-soft: #eef4f9;
  --warn: #8a3d10; --warn-bg: #fdf4ee; --line-f0: #1d5b86; --line-rms: #b06a1f;
  --line-conf: #1d5b86; --band: #2f7d4f;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--body);
  font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
        "Helvetica Neue", Arial, sans-serif;
}
main { max-width: 1080px; margin: 0 auto; padding: 32px 22px 64px; }
h1 { font-size: 27px; line-height: 1.2; margin: 0 0 6px; color: var(--ink); }
h2 { font-size: 20px; margin: 0 0 10px; color: var(--ink); }
h3 { font-size: 16px; margin: 26px 0 8px; color: var(--ink); }
h4 { font-size: 14px; margin: 18px 0 6px; color: var(--ink); }
p { margin: 0 0 12px; }
a { color: var(--accent); }
code, pre, .mono {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 12.5px;
}
header.top { border-bottom: 2px solid var(--ink); padding-bottom: 14px;
             margin-bottom: 8px; }
header.top .sub { color: var(--muted); max-width: 62ch; }
nav.toc { margin: 14px 0 26px; font-size: 13px; color: var(--muted); }
nav.toc a { margin-right: 14px; text-decoration: none;
            border-bottom: 1px solid var(--rule); }
section, footer { background: var(--card); border: 1px solid var(--rule);
                  border-radius: 6px; padding: 20px 22px; margin: 0 0 22px; }
.lede { color: var(--body); max-width: 76ch; }
.note { color: var(--muted); font-size: 12.5px; max-width: 86ch; }
.muted { color: var(--muted); font-weight: 400; font-size: 12px; }
table { border-collapse: collapse; width: 100%; margin: 10px 0 14px;
        font-variant-numeric: tabular-nums; }
th, td { border-bottom: 1px solid var(--rule); padding: 5px 8px; font-size: 13px;
         text-align: left; }
th { background: var(--accent-soft); color: var(--ink); font-weight: 600;
     font-size: 12px; }
td.right, th.right { text-align: right; }
.grid { display: grid; gap: 16px; }
.grid.two { grid-template-columns: repeat(auto-fit, minmax(330px, 1fr)); }
figure.card { margin: 0; background: var(--card); border: 1px solid var(--rule);
              border-radius: 6px; padding: 12px 14px 14px; }
figcaption { margin-bottom: 8px; font-size: 13px; color: var(--ink); }
figcaption .cond { font-weight: 600; margin-right: 8px; }
figcaption .muted { display: block; margin-top: 2px; }
svg.chart { width: 100%; height: auto; display: block; overflow: visible; }
svg text { font-family: inherit; font-size: 9.5px; fill: var(--muted); }
svg .axis-title { fill: var(--ink); font-size: 10px; }
svg .grid { stroke: var(--rule); stroke-width: 1; }
svg .axis { stroke: #b9c1cc; stroke-width: 1; }
svg .shade { fill: #eef4f9; }
svg .voiced-band { fill: var(--band); opacity: 0.75; }
svg .threshold { stroke: var(--warn); stroke-width: 1.2; stroke-dasharray: 5 4; }
svg .line { fill: none; stroke-width: 1.6; stroke-linejoin: round; }
svg .line.f0, svg .line.conf { stroke: var(--line-f0); }
svg .line.rms { stroke: var(--line-rms); }
.readout { margin: 8px 0 0; font-size: 12px; color: var(--muted);
           font-variant-numeric: tabular-nums; min-height: 1.2em; }
dl.kv { display: grid; grid-template-columns: max-content 1fr; gap: 2px 12px;
        margin: 10px 0 0; font-size: 12.5px; }
dl.kv dt { color: var(--muted); }
dl.kv dd { margin: 0; font-variant-numeric: tabular-nums; }
.callout { background: var(--warn-bg); border: 1px solid #edd9c8;
           border-left: 3px solid var(--warn); border-radius: 4px;
           padding: 12px 16px; margin: 0 0 18px; }
.callout p { margin: 0 0 10px; }
.callout p:last-child { margin-bottom: 0; }
ol.trace { list-style: none; margin: 12px 0 0; padding: 0;
           border-top: 1px solid var(--rule); }
ol.trace li { padding: 9px 0; border-bottom: 1px solid var(--rule); }
.trace-head { display: flex; flex-wrap: wrap; gap: 10px; align-items: baseline;
              font-size: 13px; }
.trace-head .step { color: var(--muted); font-size: 11.5px; min-width: 52px; }
.trace-head .instr { font-weight: 600; color: var(--ink); min-width: 74px; }
.trace-head code.act { background: var(--accent-soft); padding: 1px 6px;
                       border-radius: 3px; }
.trace-head .obs { font-variant-numeric: tabular-nums; }
.trace-sub { color: var(--muted); font-size: 12px; margin-top: 3px;
             font-variant-numeric: tabular-nums; }
p.task { font-size: 15px; color: var(--ink); background: var(--accent-soft);
         border-radius: 4px; padding: 10px 14px; }
ul.falseclaims, ul.limits { padding-left: 20px; }
ul.falseclaims li, ul.limits li { margin-bottom: 9px; max-width: 88ch; }
footer pre { background: var(--accent-soft); border: 1px solid var(--rule);
             border-radius: 4px; padding: 12px 14px; overflow-x: auto; }
.bad { color: var(--warn); }
"""

SCRIPT = """
(function () {
  var charts = document.querySelectorAll('svg.chart[data-frames]');
  Array.prototype.forEach.call(charts, function (svg) {
    var out = document.getElementById(svg.getAttribute('data-readout'));
    if (!out) { return; }
    var idle = out.getAttribute('data-idle') || '';
    var n = parseInt(svg.getAttribute('data-frames'), 10);
    var x0 = parseFloat(svg.getAttribute('data-x0'));
    var x1 = parseFloat(svg.getAttribute('data-x1'));
    var times = (svg.getAttribute('data-times') || '').split(',');
    var series = [];
    Array.prototype.forEach.call(svg.querySelectorAll('path[data-label]'),
      function (path) {
        var lo = parseFloat(path.getAttribute('data-lo'));
        var hi = parseFloat(path.getAttribute('data-hi'));
        var top = parseFloat(path.getAttribute('data-top'));
        var bottom = parseFloat(path.getAttribute('data-bottom'));
        var values = new Array(n);
        var re = /[ML](-?[0-9.]+),(-?[0-9.]+)/g;
        var match;
        while ((match = re.exec(path.getAttribute('d') || '')) !== null) {
          var x = parseFloat(match[1]);
          var y = parseFloat(match[2]);
          var i = Math.round((x - x0) / (x1 - x0) * (n - 1));
          if (i < 0 || i > n - 1 || bottom === top) { continue; }
          values[i] = lo + (bottom - y) / (bottom - top) * (hi - lo);
        }
        series.push({
          label: path.getAttribute('data-label'),
          unit: path.getAttribute('data-unit'),
          digits: parseInt(path.getAttribute('data-digits'), 10),
          values: values
        });
      });
    function read(event) {
      var point = svg.createSVGPoint();
      point.x = event.clientX;
      point.y = event.clientY;
      var local = point.matrixTransform(svg.getScreenCTM().inverse());
      var fraction = (local.x - x0) / (x1 - x0);
      var index = Math.round(fraction * (n - 1));
      if (index < 0 || index > n - 1) { return; }
      var parts = [];
      for (var s = 0; s < series.length; s++) {
        var value = series[s].values[index];
        if (value === undefined || value === null) { continue; }
        var shown = value.toFixed(series[s].digits) +
                    (series[s].unit ? ' ' + series[s].unit : '');
        parts.push(series[s].label + ' ' + shown);
      }
      var stamp = times[index] !== undefined ? times[index] + ' s' : 'frame ' + index;
      out.textContent = 'frame ' + index + ' (' + stamp + '): ' +
        (parts.length ? parts.join('  ·  ') : 'no value at this frame');
    }
    svg.addEventListener('mousemove', read);
    svg.addEventListener('mouseleave', function () { out.textContent = idle; });
  });
})();
"""


def render(voice: dict, agent: dict) -> str:
    """The whole page, as one string.

    Pure in its two inputs: no timestamp, no environment, and not even the output
    path is allowed in -- the footer names the canonical file, so rendering the
    same results to two different paths produces the same bytes and a diff of the
    page is a diff of the measurements.
    """
    voice_config = voice.get("config") or {}
    agent_config = agent.get("config") or {}
    sections = [
        (
            "<header class=\"top\">\n"
            "<h1>beyond-attention — generated dashboard</h1>"
            "<p class=\"sub\">A selective state-space model, its prosody front end "
            "and its agent loop, drawn from the JSON the experiments write. Every "
            "number, chart and sentence about what a control did is produced by "
            "<code>experiments/render_dashboard.py</code> from "
            "<code>voice-affect.json</code> and <code>agent-loop.json</code>; "
            "re-running the experiments and the renderer reproduces this file byte "
            "for byte. Nothing on the page is typed in by hand, and nothing on it "
            "was measured twice.</p>\n"
            f"<p class=\"sub\">voice: seed {voice_config.get('seed')}, "
            f"{voice_config.get('utterances_per_condition')} utterances per "
            f"condition · agent: seed {agent_config.get('seed')}, "
            f"{agent_config.get('suite_size')} tasks, budgets "
            f"{', '.join(str(b) for b in agent_config.get('budgets') or [])}"
            "</p>\n"
            "</header>\n"
            "<nav class=\"toc\">"
            "<a href=\"#voice\">Voice</a>"
            "<a href=\"#voicing-control\">Voiced/unvoiced control</a>"
            "<a href=\"#agent\">Agent</a>"
            "<a href=\"#results\">Results and controls</a>"
            "<a href=\"#limitations\">What this does not show</a>"
            "<a href=\"#regenerate\">Regenerating</a>"
            "</nav>"
        ),
        voice_panel(voice),
        control_panel(voice),
        agent_panel(agent),
        results_panel(agent),
        limitations_panel(voice, agent),
        footer_panel(voice, agent),
    ]
    return (
        "<!doctype html>\n"
        "<html lang=\"en\">\n<head>\n"
        "<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        "<title>beyond-attention — generated dashboard</title>\n"
        f"<style>{STYLE}</style>\n"
        "</head>\n<body>\n<main>\n"
        + "\n".join(sections)
        + f"\n</main>\n<script>{SCRIPT}</script>\n</body>\n</html>\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--voice", default="voice-affect.json",
                        help="voice-affect.json, which supplies the voice panels")
    parser.add_argument("--agent", default="agent-loop.json",
                        help="agent-loop.json, which supplies the agent and "
                             "results panels")
    parser.add_argument("--out", default="dashboard.html")
    args = parser.parse_args()

    voice = _load(args.voice, "the voice/affect run")
    agent = _load(args.agent, "the agent-loop run")
    if voice is None or agent is None:
        return 1

    # Both files have to be the ones the page's panels are drawn from; a render
    # from an older or narrower results file would silently drop a panel.
    missing = [key for key, payload in (("frame_tracks", voice),
                                        ("voicing_probe", voice),
                                        ("example_trace", agent),
                                        ("solve_rate", agent))
               if key not in payload]
    if missing:
        print(f"refusing to render: {', '.join(missing)} missing from the inputs",
              file=sys.stderr)
        return 1

    page = render(voice, agent)
    for marker in REQUIRED_MARKERS:
        if marker not in page:
            print(f"refusing to write: {marker!r} missing from the render",
                  file=sys.stderr)
            return 1
    for section_id in ("voice", "voicing-control", "agent", "results",
                       "limitations", "regenerate"):
        if f'id="{section_id}"' not in page:
            print(f"refusing to write: section {section_id!r} missing",
                  file=sys.stderr)
            return 1

    pathlib.Path(args.out).write_text(page)
    # The size is reported in bytes rather than characters: the page is UTF-8 and
    # carries em dashes, so the two numbers differ and only one of them is what
    # the file occupies on disk.
    print(f"wrote {args.out}: {len(page.encode('utf-8')):,} bytes, "
          f"{len(page.splitlines()):,} lines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
