"""The README's measurement tables that the renderer does not produce.

`tests/test_render_readme.py` covers every `BEGIN`/`END` region byte for byte.
That is not the same claim as "every number in the README is checked", and the
gap is where four measurement tables live: they carry figures from committed
JSON files but sit inside narrative sections, outside every rendered region, so
nothing failed when the JSON moved.

They are pinned here rather than rendered, and the reason is in the tables
themselves. Rendering them would put interpretation into the renderer:

* the training-variant table's spread column reads "~2e-5" and "~2.8, ten orders
  of magnitude" -- prose judgements about a spread, not a formatted number;
* the carried-state table is explicitly arithmetic rather than measurement
  ("Every figure above is `numel * 4`"), and two of its seven rows extend past
  every committed file, so a renderer would have to reimplement the model's
  cache-size formula rather than read a measurement;
* the measured-memory table's labels are sentences, and one of its cells carries
  its own arithmetic cross-check in a parenthesis.

The README reserves interpretation for hand-written text and says so. So these
tables stay hand-written, and this module is what stops them drifting: each
figure is recomputed from the JSON and compared. It counts what it checked, so a
parse that silently matched nothing cannot pass.

One test here is not about a table at all: the seed count behind the headline
argument, which the prose states in words ("on three seeds") and which no
renderer can see.
"""

from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
README = ROOT / "README.md"


def _json(name: str) -> dict:
    return json.loads((ROOT / name).read_text())


def _table(header: str) -> list[list[str]]:
    """The rows of the README table whose header line starts with `header`.

    An anchor rather than a line number: a line number would silently point at
    different rows after an edit above it. If the header is reworded, this raises
    with the wording it was looking for, which is a message a maintainer can act
    on rather than a stale index.
    """
    lines = README.read_text().splitlines()
    for start, line in enumerate(lines):
        if line.strip().startswith(header):
            rows: list[list[str]] = []
            for row in lines[start + 1:]:
                if not row.strip().startswith("|"):
                    break
                cells = [cell.strip() for cell in row.strip().strip("|").split("|")]
                if set("".join(cells)) <= set("-: "):  # the alignment row
                    continue
                rows.append(cells)
            assert rows, f"the table under {header!r} has a header and no rows"
            return rows
    raise AssertionError(
        f"no README table header starts with {header!r}; if it was reworded, "
        f"update the anchor here so these figures stay pinned"
    )


def _numbers(text: str) -> list[float]:
    """Every float in a cell, with thousands separators removed.

    Tolerant of the surrounding prose on purpose: the point is to pin the
    figures, not the words around them.
    """
    return [float(v.replace(",", "")) for v in _tokens(text)]


def _tokens(text: str) -> list[str]:
    """The numeric tokens in a cell, as written."""
    import re

    return re.findall(r"-?\d[\d,]*\.?\d*(?:e-?\d+)?", text.replace("**", ""))


def _printed_at_its_own_precision(token: str, want: float) -> float:
    """The JSON value rounded to the number of digits the README printed.

    A fixed relative tolerance is the wrong instrument here: this table's values
    span seventeen orders of magnitude and are quoted to three significant
    figures in one row and six in the other, so a tolerance loose enough for
    `6.09e-17` against `6.0908e-17` would accept a genuinely different number in
    the same row. Rounding the JSON value at the printed precision instead checks
    the claim the README is actually making -- that this is that number, to the
    digits shown.
    """
    mantissa = token.lower().split("e")[0].lstrip("-").replace(".", "").lstrip("0")
    return float(f"{want:.{max(len(mantissa), 1) - 1}e}")


def _row_containing(rows: list[list[str]], needle: str) -> list[str]:
    matches = [row for row in rows if needle in row[0]]
    assert len(matches) == 1, f"{needle!r} matched {len(matches)} rows, wanted 1"
    return matches[0]


def _close(got: float, want: float, *, label: str, rel: float = 1e-3) -> None:
    scale = max(abs(want), 1e-30)
    assert abs(got - want) / scale < rel, f"{label}: README says {got}, JSON says {want}"


def test_the_training_variant_losses_are_the_committed_ones() -> None:
    """The five-seed losses that say annealing is the numerically sensitive row.

    This table is evidence for the cross-platform anomaly the README documents,
    so a wrong figure here would change which row is blamed for it.
    """
    rows = _table("| variant | final losses, seeds 0–4 | spread |")
    reports = _json("learned-gate.json")["training_reports"]
    checked = 0

    for needle, kind in (("raw, no annealing", "raw"), ("annealed", "anneal")):
        row = _row_containing(rows, needle)
        if kind == "raw":
            losses = [r["final_loss"] for r in reports if r["kind"] == "raw"]
        else:
            losses = [r["final_loss"] for r in reports if r.get("anneal_to")]
        assert len(losses) == 5, (needle, len(losses))

        printed = _tokens(row[1])
        assert len(printed) == 5, (needle, row[1], printed)
        for token, want in zip(printed, losses):
            got = float(token.replace(",", ""))
            assert got == _printed_at_its_own_precision(token, want), (
                f"{needle}: README prints {token}, the committed value is {want!r}")
            checked += 1

        # The spread column is a judgement, but its magnitude is not: it has to
        # be the spread of the numbers in the same row.
        spread = max(losses) - min(losses)
        printed_spread = _numbers(row[2])[0]
        if spread < 1e-3:
            # printed as "~2e-5", so only the exponent is comparable
            assert abs(printed_spread / spread - 1) < 1.0, (
                f"{needle}: README says ~{printed_spread:g}, JSON spread is "
                f"{spread:g}")
        else:
            _close(printed_spread, spread, label=f"{needle} spread", rel=0.2)
        checked += 1

    assert checked == 12, checked


def test_the_carried_state_table_is_the_committed_one() -> None:
    """The state-vs-cache table, including the two rows no file reaches.

    Five rows come from `stream-cost.json`. The 262,144 and 1,048,576 rows are
    arithmetic on the same shape, so what is checked for them is that they carry
    the file's own proportionality rather than a convenient number -- and the
    longest one is independently confirmed by `long-context.json`.
    """
    rows = _table("| context length | SSM state | attention KV cache |")
    cost = _json("stream-cost.json")
    contrast = _json("long-context.json")["contrast"]

    unit = {"B": 1, "KiB": 1024, "MiB": 1024 ** 2, "GiB": 1024 ** 3}

    def size(cell: str) -> tuple[int, int]:
        """`4,864 elts · 19 KiB` -> (4864 elements, 19456 bytes)."""
        elements, _, printed = cell.partition("·")
        count, name = printed.split()
        return int(elements.split()[0].replace(",", "")), int(float(count) * unit[name])

    parsed: dict[int, tuple[int, int, int, int]] = {}
    for row in rows:
        length = int(row[0].replace(",", ""))
        ssm_elts, ssm_bytes = size(row[1])
        attn_elts, attn_bytes = size(row[2])
        parsed[length] = (ssm_elts, attn_elts, ssm_bytes, attn_bytes)
    assert len(parsed) == 7, sorted(parsed)

    checked = 0
    for ssm, attn in zip(cost["ssm"], cost["attention"]):
        length = ssm["length"]
        got = parsed[length]
        assert got[0] == ssm["state_elements"], (length, got, ssm)
        assert got[1] == attn["state_elements"], (length, got, attn)
        assert got[2] == ssm["state_bytes"], (length, got, ssm)
        assert got[3] == attn["state_bytes"], (length, got, attn)
        assert got[2] == got[0] * 4 and got[3] == got[1] * 4, (length, got)
        checked += 1

    # The two rows past the file: same proportionality, and the longest is
    # confirmed by a second file that streamed it.
    ratio = cost["attention"][0]["state_elements"] // cost["attention"][0]["length"]
    state = cost["ssm"][0]["state_elements"]
    for length in (262_144, 1_048_576):
        ssm_elts, attn_elts, ssm_bytes, attn_bytes = parsed[length]
        assert ssm_elts == state, (length, ssm_elts, state)
        assert attn_elts == length * ratio, (length, attn_elts, length * ratio)
        assert ssm_bytes == ssm_elts * 4 and attn_bytes == attn_elts * 4, length
        checked += 1
    assert parsed[contrast["longest_streamed"]][3] == contrast["attention_kv_cache_bytes"]
    assert parsed[contrast["longest_streamed"]][2] == contrast["ssm_state_bytes"]

    assert checked == 7, checked


def test_the_measured_memory_table_is_the_committed_one() -> None:
    """The three figures that make the flat RSS line mean something."""
    rows = _table("| | measured |")
    memory = _json("stream-memory.json")
    contrast = _json("long-context.json")["contrast"]
    checked = 0

    row = _row_containing(rows, "SSM streamed")
    rss_mb = memory["ssm_stream"]["rss_after_warmup_min"] / 1024 ** 2
    _close(_numbers(row[1])[0], rss_mb, label="SSM streamed RSS")
    # The spread is printed to two decimals, so it is compared at two decimals.
    # A relative tolerance is the wrong instrument: "0.00%" is the correct
    # rendering of a 0.003% spread, and `0.003` against `0.00` is a 100% error by
    # that measure. A re-run of the experiment produced 8 KB of growth over a
    # 245 MB stream, which is exactly that case.
    printed_spread = _numbers(row[1])[1]
    actual_spread = round(
        memory["ssm_stream"]["rss_after_warmup_growth_fraction"] * 100, 2)
    assert printed_spread == actual_spread, (
        f"the README prints {printed_spread}% and the JSON says {actual_spread}%")
    checked += 2

    row = _row_containing(rows, "positive control")
    grown_mb = memory["cache_growth_control"]["rss_after_warmup_growth_bytes"] / 1024 ** 2
    _close(_numbers(row[1])[0], grown_mb, label="positive control growth")
    # The control has to have grown, or "the SSM's RSS did not move" is a
    # statement about an instrument that cannot see movement.
    assert grown_mb > 100, grown_mb
    checked += 1

    row = _row_containing(rows, "attention KV cache at")
    cache_mb = contrast["attention_kv_cache_bytes"] / 1024 ** 2
    printed = _numbers(row[1])
    assert len(printed) == 2, (row[1], printed)
    _close(printed[0], cache_mb, label="attention KV cache")
    _close(printed[1], cache_mb, label="attention KV cache arithmetic")
    checked += 3

    assert checked == 6, checked


def test_the_parallel_forward_wall_is_the_committed_one() -> None:
    """Three lengths, one of which is a refusal rather than a number.

    The refusal is the result, so the test requires that the README records it as
    a failure and does not print a peak figure for it.
    """
    rows = _table("| length | attention parallel forward | peak RSS |")
    measured = _json("long-context.json")["attention_forward"]
    checked = 0

    def find(length: int) -> list[str]:
        return [row for row in rows if int(row[0].replace(",", "")) == length][0]

    for entry in measured:
        row = find(entry["length"])
        if entry["ok"]:
            assert row[1] == "ok", (entry["length"], row)
            _close(_numbers(row[2])[0], entry["peak_rss_bytes"] / 1024 ** 2,
                   label=f"peak RSS at {entry['length']}")
            checked += 2
        else:
            assert "RuntimeError" in row[1], (entry["length"], row)
            assert row[2] == "—", (
                f"{entry['length']} refused the allocation, so its peak RSS is "
                f"not a measurement, but the README prints {row[2]!r}")
            checked += 2

    assert checked == 6, checked


def test_the_headline_sweeps_are_more_than_one_seed() -> None:
    """The README's central argument is stated over three seeds.

    "That headline is false, and the control shows it" is the claim the whole
    repository turns on: at the matched 3,000-step budget the Transformer
    degrades where the state-space model does not, and the same Transformer given
    20,000 steps reaches 1.000 at 8 pairs. Both were one seed until this was
    changed. A single seed cannot show whether a row is stable, and a
    regeneration with `--seeds 0` would drop the ± column while the prose went on
    making a claim about three seeds -- a mismatch no other test here can see,
    because the rendered table would agree with whatever file it was given.

    The run count is read from the rows as well as the config, so a file whose
    config claims three seeds while its rows carry one fails too.
    """
    for name in ("results.json", "control-attention.json"):
        payload = _json(name)
        seeds = payload["config"]["seeds"]
        assert len(seeds) >= 3, (
            f"{name} records seeds={seeds}; the README's headline result is "
            f"stated over three seeds and its spread column is rendered from them")
        counts = {len(row["accuracy_runs"]) for row in payload["mqar"].values()}
        assert counts == {len(seeds)}, (
            f"{name}: config says {len(seeds)} seeds but the rows carry "
            f"{sorted(counts)} runs each")
