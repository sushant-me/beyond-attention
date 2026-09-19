# beyond-attention

A selective state-space model — **Mamba's S6 recurrence** — implemented from
scratch, with the arithmetic checked three independent ways, and a
parameter-matched comparison against a Transformer that produced a result I did
not expect and a control that explains it.

## What this is, and what it is not

**It is** a working, tested implementation of a non-attention sequence model,
plus a measurement of it. Every number below is generated from a JSON results
file by `experiments/render_readme.py`; re-running the experiment and the
renderer reproduces the tables.

**It is not** a new architecture, and not a replacement for attention. Three
claims that would be easy to make from the main table, and are all false:

* *"The state-space model recalls associations better than the Transformer."*
  The main table says exactly that — and the control falsifies it.
* *"It is faster because it is linear."* Asymptotically yes, and **not against a
  good baseline.** Against attention written with the fused kernel this
  implementation is ~2.6× slower and ~2.8× lighter-is-worse at every length up to
  32,768 tokens. Against attention written with an explicit causal mask it does
  cross over. Both are measured and both are in [Results](#results), because
  reporting one of them would be choosing the answer.
* *"It is a new model."* It is S6, written out longhand so it can be tested.

## The recurrence

A state-space model keeps a fixed-size state and updates it as it reads:

```
a_t = exp(delta_t * A)                    # (B, L, D, N), in (0, 1)
b_t = delta_t * B_t * x_t                 # (B, L, D, N)
h_t = a_t * h_{t-1} + b_t                 # (B, D, N),  h_{-1} = 0
y_t = sum_n C_t[n] * h_t[d, n]            # (B, L, D)
```

`A` is stored in log space and negated, so every decay is in `(0, 1)` and the
recurrence is stable for *any* parameter value the optimiser picks. `delta`, `B`
and `C` are produced from the input, which is what makes the scan **selective**:
the model chooses per token how much to write into the state, how much to decay
it, and what to read back. A non-selective SSM uses constants for those three and
cannot do content-based lookup at all.

## Correctness

The recurrence is simple enough to get subtly wrong, and the usual way that
survives is that a fast path is compared only against a loss curve — which
descends happily for a model that is quietly forgetting everything past its chunk
boundary. So there are three implementations sharing no code:

| function | method | role |
|---|---|---|
| `selective_scan_reference` | explicit loop over time | the definition |
| `selective_scan_chunked` | short loop per chunk, then scan the summaries | what a kernel does |
| `selective_scan_associative` | Hillis-Steele scan over the `(a, b)` monoid | log₂(L) parallel steps |

The tests assert all three are **equal to float64 tolerance** on random,
degenerate and adversarial inputs, at chunk sizes that do and do not divide the
sequence length, with gradients matching between the loop and the chunked path.

Two properties are tested independently of numerical agreement, because
agreement cannot catch them:

* **Causality** — change an input at position `t` and require every output before
  `t` to be bit-for-bit unchanged, for all three paths, at chunk sizes strictly
  smaller than the cut. An earlier version of this test listed only two of the
  three implementations, so the chunked path — the one that can leak across a
  chunk boundary — was never tested by it, and a deliberately reversed chunk scan
  passed. The parametrisation was the bug.
* **Long-range propagation** — the depthwise convolution has a kernel of 2, so
  position 0 reaches positions 0 and 1 and nothing else. A test changes only
  position 0 and requires the output at position 11 to move; if it does not, the
  block is a local filter wearing a state-space model's name.

Every test was mutation-checked. Breaking the monoid, dropping the carry between
chunks, reversing the chunk scan, removing the convolution's causal padding and
removing the attention mask each fail a *different, specific* test — five
mutations, five distinct failures. 56 tests, all green.

## Results

Both architectures get the same data, optimiser, learning-rate schedule, step
budget and seed; the parameter counts are printed in the table and asserted in
the tests. The task is **multi-query associative recall**: a sequence of
key/value pairs, then a query key, and the model must emit the paired value. Keys
and values come from disjoint halves of the vocabulary, so a model cannot score
by predicting a likely token — it has to have stored the association.

MQAR is the right test because it is *exact* (accuracy, not a loss to interpret)
and *position-invariant*, so a Transformer with no positional encoding is a fair
baseline rather than a crippled one.

<!-- RESULTS:BEGIN -->
**Main sweep** — d_model=64, layers=2, steps=3000, seeds=[0], batch=32, lr=0.005

| pairs in context | model | parameters | accuracy (exact match) | chance | best accuracy at an unseen size |
|---|---|---|---|---|---|
| 2 | attention | 67,968 | 1.000 | 0.500 | 0.769 (at 4) |
| 2 | ssm | 67,584 | 1.000 | 0.500 | 0.548 (at 4) |
| 4 | attention | 67,968 | 0.413 | 0.250 | 0.554 (at 2) |
| 4 | ssm | 67,584 | 1.000 | 0.250 | 1.000 (at 2) |
| 8 | attention | 67,968 | 0.312 | 0.125 | 0.568 (at 2) |
| 8 | ssm | 67,584 | 1.000 | 0.125 | 0.975 (at 4) |
| 16 | attention | 67,968 | 0.203 | 0.062 | 0.000 (at 2) |
| 16 | ssm | 67,584 | 0.186 | 0.062 | 0.000 (at 2) |

The control below is the same architecture at the same size, with the step budget raised and nothing else changed.

**Control: attention, 20,000 steps** — d_model=64, layers=2, steps=20000, seeds=[0], batch=32, lr=0.005

| pairs in context | model | parameters | accuracy (exact match) | chance | best accuracy at an unseen size |
|---|---|---|---|---|---|
| 8 | attention | 67,968 | 1.000 | 0.125 | 0.000 (at 16) |
| 16 | attention | 67,968 | 0.175 | 0.062 | 0.000 (at 8) |

### Does either model read longer than it trained?

The "unseen size" column above is not a clean measure of that, and it took building the experiment below to see why. `mqar_batch` sizes its key space from the pair count by default: trained at 2 pairs a model meets keys 1–8 and values 9–16, while a 16-pair batch hands it keys 1–16 and values 17–32 — token ids it has never seen, in roles it has never seen them in. Every `0.000` in that column is an evaluation at a *longer* length than training, which is precisely the case the column was meant to measure. Those zeros are a vocabulary mismatch, not a failure to generalise.

Holding the key space fixed so the vocabulary is identical at every length changes the picture completely. Both models are trained at **2 pairs (6 tokens)** — the longest length at which *both* solve the task outright — then evaluated with frozen weights at the lengths below (up to **5.7x**):

| pairs | tokens | x train | attention | ssm | chance | a model trained at that length: attention | ssm |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 6 | 1.0x | 1.000 ±0.000 | 1.000 ±0.000 | 1/16 | 1.000 | 1.000 |
| 4 | 10 | 1.7x | 0.793 ±0.007 | 0.541 ±0.007 | 1/16 | — | — |
| 8 | 18 | 3.0x | 0.459 ±0.005 | 0.297 ±0.010 | 1/16 | — | — |
| 16 | 34 | 5.7x | 0.240 ±0.009 | 0.176 ±0.014 | 1/16 | 0.205 | 0.197 |

Three seeds, spread shown, nothing retrained between rows.

What this says, including the parts that are unflattering:

* **Neither architecture extrapolates here.** Both are perfect at the length they trained on (2 pairs) and both fall below 1.000 at every longer length — attention to 0.793 and the SSM to 0.541 at 4 pairs, the shortest step past training.
* **Attention decays more slowly than the SSM** at every step — 0.793 against 0.541 at 4 pairs, 0.459 against 0.297 at 8 pairs, 0.240 against 0.176 at 16 pairs. On this task the state-space model is the weaker of the two past its training length, which is the opposite of what the architecture's reputation would predict.
* **Part of the decay is task difficulty, and the reference column is what says so.** A model trained from scratch at the longest length reaches attention 0.240 against 0.205 (gap 0.035, three-seed spread ±0.009); ssm 0.176 against 0.197 (gap 0.021, three-seed spread ±0.014). The reference is 1 seed, so a gap smaller than the spread is not a difference and a larger one is part extrapolation and part how hard MQAR is at that length for a two-layer, `d_model`-64 model — which no column here separates. Without it the curve looks like a generalisation result and is not one.
* The SSM does fit *longer training lengths* better than attention: in the main sweep it reaches 1.000 at 8 pairs where attention reaches 0.312. So "fits long sequences when trained on them" and "generalises to longer ones when trained short" are separate properties, and the two architectures sit on opposite sides of them.

One training length, one task, one model size. This measures MQAR at 2 pairs, 6 tokens, d_model=64, 2 layers, on three seeds.

### Length scaling

Activation memory is the *rise in current RSS* sampled while the forward and backward pass runs, one fresh process per point. It is not a difference of `ru_maxrss` high-water marks: that value is reported out of `signal_struct`, which a forked child inherits from its parent, so a child of a large parent starts with the parent's peak already recorded and its own allocations are invisible. The first version of this probe measured that way and printed zero, or the same constant, for every configuration.

**attention baseline: causal_mode=sdpa** (batch=4, d_model=64, layers=2)

| sequence length | model | forward+backward (s) | peak activation memory (MB) | MB per token |
|---|---|---|---|---|
| 256 | attention | 0.014 | 26.6 | 0.0260 |
| 256 | ssm | 0.821 | 125.8 | 0.1228 |
| 1024 | attention | 0.057 | 68.7 | 0.0168 |
| 1024 | ssm | 1.394 | 189.6 | 0.0463 |
| 4096 | attention | 0.486 | 236.0 | 0.0144 |
| 4096 | ssm | 3.514 | 660.2 | 0.0403 |
| 8192 | attention | 1.708 | 455.9 | 0.0139 |
| 8192 | ssm | 7.835 | 1234.2 | 0.0377 |
| 16384 | attention | 6.163 | 896.3 | 0.0137 |
| 16384 | ssm | 21.101 | 2378.6 | 0.0363 |
| 32768 | attention | 23.526 | 1653.1 | 0.0126 |
| 32768 | ssm | 61.683 | 4650.1 | 0.0355 |

**attention baseline: causal_mode=mask** (batch=4, d_model=64, layers=2)

| sequence length | model | forward+backward (s) | peak activation memory (MB) | MB per token |
|---|---|---|---|---|
| 8192 | attention | 3.600 | 983.4 | 0.0300 |
| 8192 | ssm | 8.106 | 1227.3 | 0.0375 |
| 16384 | attention | 15.085 | 2977.7 | 0.0454 |
| 16384 | ssm | 20.669 | 1747.5 | 0.0267 |
| 32768 | attention | 57.554 | 9121.7 | 0.0696 |
| 32768 | ssm | 55.857 | 4652.3 | 0.0355 |

### Which inner scan to use

The recurrence inside each chunk can be run as a sequential loop, or as a Hillis-Steele scan vectorised along the chunk axis. The loop does `O(chunk)` work and the scan does `O(chunk * log2(chunk))`, so on a CPU the scan is doing more arithmetic to remove interpreter overhead that is cheaper than the arithmetic it adds.

length=8192, batch=4, d_inner=128, d_state=16, threads=4

| inner scan | chunk | chunks | seconds | peak MB | agrees with loop |
|---|---|---|---|---|---|
| loop | 64 | 128 | 3.89 | 450.6 | yes (1.4e-07) |
| vectorized | 64 | 128 | 4.31 | 3302.8 | yes (1.4e-07) |
| loop | 256 | 32 | 16.09 | 483.9 | yes (1.3e-08) |
| vectorized | 256 | 32 | 8.56 | 4898.0 | yes (1.3e-08) |

```
parameter check at the sweep vocabulary: ssm=67,584 vs attention=67,968
main sweep wall time: 1814.6s
control wall time: 271.1s
```
<!-- RESULTS:END -->

## The inner-scan comparison, and a correction to my own earlier claim

The table above is generated by `experiments/scan_inner.py`, which was written
because I expected the vectorised scan to win and I wanted the negative result on
the record rather than in a commit message.

**The sequential loop at chunk=64 is the only configuration that is best on both
axes** — 3.89 s and 451 MB. The vectorised scan is not uniformly worse: at
chunk=256 it is **1.9× faster** than the loop at the same chunk (8.56 s vs
16.09 s), but it pays about **ten times the memory** to get there (4,898 MB vs
484 MB), because Hillis-Steele holds several full `(B, chunk, D, N)` tensors at
once. It never wins on both axes, so the loop stays the default.

Two corrections are worth recording, because both were mistakes of mine:

* **My first version of this experiment gave the two paths different random
  inputs** — it drew fresh tensors inside the measurement — so the "difference"
  column it printed between them was meaningless. A comparison that cannot fail.
  The inputs now come from a fixed seed and each process reports a checksum,
  which the driver compares; every row above agrees to float32 resolution
  (1.4e-07). That column is what makes the speed numbers mean anything.
* **My first conclusion was that the vectorised scan was slower outright.** That
  came from timings taken while a `torch.compile` job was running concurrently
  and competing for CPU — loop@64 measured 27.9 s under contention and 3.89 s on
  an idle machine, a 7× error. The memory result survived; the time result
  reversed. Timings from a loaded machine are not measurements.

### Does the answer change with length?

The table above is one length. `scan_inner.py` now sweeps several, because a
configuration that wins at 8,192 tokens need not win at 1,024 — and the chunk
size is the setting most likely to behave that way.

Cost is fitted as `length ** exponent` by least squares over 1,024 / 2,048 /
4,096 / 8,192 (batch=4, d_inner=128, d_state=16):

| configuration | exponent | at 8,192 | peak MB |
|---|---:|---:|---:|
| **loop @ chunk 64** | **0.61** | **3.58 s** | 449 |
| vectorized @ chunk 64 | 0.69 | 4.33 s | 3,305 |
| loop @ chunk 256 | 0.89 | 14.93 s | 489 |
| vectorized @ chunk 256 | 0.81 | 8.91 s | 4,827 |

`loop` at chunk 64 is the fastest configuration at **every** length measured
(1,024: 0.99 s, 2,048: 1.21 s, 4,096: 1.79 s, 8,192: 3.58 s), so the default is
not a compromise that happens to hold at one size.

The more useful number is what actually moves the result. Changing the inner
scan at a fixed chunk changes the time by about 20% (3.58 → 4.33 s at chunk 64).
Changing the **chunk** at a fixed inner scan changes it by **4.2x** (3.58 →
14.93 s at chunk 256). The chunk is the setting worth tuning — and it is the one
the single-length table could not show, which is why the sweep exists.

An exponent below 1.0 means cost is growing *slower* than the sequence, because
fixed per-call overhead is still being amortised over this range. It is not
evidence that the scan is sublinear; it should approach 1.0 for long enough
inputs, which is what `O(L)` implies.

**This corrects a claim I made in the Limitations below.** I reported the step
cost as growing "roughly as `L^1.9`", fitted over four points between 6 and 34
tokens. At those lengths per-step overhead dominates, and the ratio of two noisy
small numbers is not an exponent. Measured properly over 1,024–8,192 the scan
grows at **0.61** and is nowhere near quadratic. The old figure was wrong.

## The result, and why the obvious reading of it is wrong

At the matched 3,000-step budget the state-space model is clearly ahead: it
reaches 1.000 up to 8 pairs in context, while the Transformer degrades from 1.000
at 2 pairs to 0.203 at 16. The natural headline is *"a selective state-space model
recalls associations better than a Transformer"*, and this repository's main
table supports it.

**That headline is false, and the control shows it.** The same Transformer, at
the same size and with nothing changed but the step budget raised to 20,000,
reaches **1.000** at 8 pairs. Nothing about the architecture changed; it simply
needed longer to get there.

This is a known effect rather than a surprise: associative recall in a
Transformer is implemented by an *induction head*, a two-layer circuit that takes
many steps to form, and it is one of the last things to appear during training.
The 3,000-step sweep therefore measured **how quickly each architecture learns
this task**, not what it is capable of — and those are different questions that
produce the same-looking table.

The distinction is the most useful thing in this repository, because it is
invisible without the control and the wrong version is the more interesting
story. A paper — or a README — that stopped at the main table would be reporting
a training-budget artefact as an architectural result. The honest summary is:

* **Learning speed on MQAR, at this scale: the SSM is faster.** It fits this task
  in a fraction of the steps.
* **Capability on MQAR, at this scale: both architectures get there.** For 8
  pairs, given enough steps, either can do it.
* **Neither solves 16 pairs** at this model size, on any budget tried.

## Which baseline you choose decides the answer

Two comparisons, both measured the same way, and they disagree.

**Against attention written with an explicit causal mask** (a perfectly ordinary
way to write it, and what `nn.MultiheadAttention` needs for a causal mask), the
state-space model **crosses over**. At 32,768 tokens it is marginally faster
(55.9 s vs 57.6 s) and about **2× lighter** (4,652 MB vs 9,122 MB). The explicit
mask materialises the `L × L` score matrix, and its memory per token climbs from
0.030 to 0.070 MB across the range while the SSM's stays flat at 0.036.

**Against attention written with the fused kernel** — `F.scaled_dot_product_attention`
with `is_causal=True`, which never materialises that matrix — **there is no
crossover up to 32,768 tokens.** Attention is ~2.6× faster (23.5 s vs 61.7 s) and
~2.8× lighter (1,653 MB vs 4,650 MB), and its memory per token is *flat* at
0.013–0.026 MB. The quadratic-memory story does not apply to a fused
implementation, so the asymptotic argument buys nothing at these lengths.

The honest summary is therefore narrow, and it is the one I am willing to defend:

> A from-scratch selective state-space model, in pure PyTorch on CPU, beats a
> Transformer whose attention is written with an explicit causal mask beyond
> roughly 16k–32k tokens on memory and marginally on time — and loses to the same
> Transformer when attention uses the fused kernel, at every length measured.

The interesting part is not the SSM. It is that **the same architecture wins or
loses depending on how the baseline is written**, and that both numbers are
needed to say anything true.

## What it took to get even that far

The first version of this scan was 20–40× slower than attention and 8× hungrier,
which is what a literal reading of the recurrence produces: it built `a` and `b`
at `(B, L, D, N)` for the whole sequence at once, so a linear-time algorithm cost
*more* memory than the quadratic one it was meant to beat.

The rewrite streams one chunk at a time and wraps each chunk in
`torch.utils.checkpoint`, so the backward pass recomputes chunks instead of
retaining them. Measured by the same method, on the same inputs, at `L = 4096`:

| path | peak rise in RSS |
|---|---|
| whole-sequence (materialising) | 131 MB |
| streaming (one chunk) | 6.6 MB |

a **20× reduction**, and the test that asserts it was mutation-checked — pointing
the streaming path back at the materialising implementation makes it fail. The
per-token memory of the model then falls to 0.036 MB and stays flat as length
grows, which is the property the architecture is supposed to have.

Two mistakes of my own are worth recording, because both produced results I
nearly published:

1. **The memory probe measured the wrong thing.** It subtracted `ru_maxrss`
   high-water marks. That value is reported out of `signal_struct`, which a
   forked child *inherits*: a child spawned by a large parent starts with the
   parent's peak already recorded, so its own allocations are invisible and the
   growth reads as zero. Writing `5` to `/proc/self/clear_refs` does not fix it
   — it lowers `mm->hiwater_rss` while `getrusage` reports the larger of that and
   the inherited `signal->maxrss`. Sampling *current* RSS during the call does
   fix it. The old probe reported the same constant for every configuration, and
   a constant prints as a table.
2. **The "fused baseline" run was not fused.** `run.py` did not pass
   `--causal-mode` down to the measurement subprocess, so a run labelled `sdpa`
   silently measured `mask`. That is where the crossover I first reported came
   from: it was a comparison against the slower baseline, mislabelled as the
   faster one. Fixing the flag reversed the conclusion.

Both were caught by controls, not by reading the code: the first by running the
probe where the parent was fat, the second by asking why one number moved by 2.8×
when only a label had changed.

## Reproducing this

```bash
uv venv && uv pip install --index-url https://download.pytorch.org/whl/cpu torch
uv pip install -e . pytest

python -m pytest tests/ -q                     # 181 correctness tests

python experiments/run.py --pairs 2 4 8 16 --steps 3000 --seeds 0 \
    --out results.json                         # main sweep    (~30 min, CPU)
python experiments/run.py --pairs 8 16 --steps 20000 --blocks attention \
    --out control-attention.json               # the control   (~5 min)

# length extrapolation: train at 2 pairs, evaluate frozen weights out to 5.7x.
# The reference models cost most of the runtime; --no-reference skips them, at
# the price of no longer being able to attribute the decay to anything.
python experiments/length_extrapolation.py --out length-extrapolation.json

# streamed memory: real RSS over a million tokens, with a positive control
python -u experiments/stream_memory.py --out stream-memory.json

# inner scan and chunk size across lengths (each child address-space capped)
python experiments/scan_inner.py --lengths 1024 2048 4096 8192 --chunks 64 256 \
    --out scan-inner-scaling.json

# both scaling baselines, same measurement method
python experiments/run.py --skip-sweep --causal-mode sdpa \
    --lengths 256 1024 4096 8192 16384 32768 --out scaling-final.json
python experiments/run.py --skip-sweep --causal-mode mask \
    --lengths 8192 16384 32768 --out scaling-mask.json

python experiments/render_readme.py --mqar results.json \
    --scaling scaling-final.json --scaling scaling-mask.json \
    --control control-attention.json --scan-inner scan-inner.json \
    --length-extrapolation length-extrapolation.json \
    --readme README.md

# the voice/affect front end: synthetic prosody conditions, seconds to run
python experiments/voice_affect.py --out voice-affect.json
python experiments/render_readme.py --voice voice-affect.json --readme README.md

# the trained classifier: 200 synthesised utterances, ~3 min on CPU
python -u experiments/emotion_classifier.py --out emotion-classifier.json
python experiments/render_readme.py --emotion emotion-classifier.json \
    --readme README.md

# the agent loop: 300 seeded tasks, six budgets, the controls and two sweeps (~2 s)
python experiments/agent_loop.py --out agent-loop.json
python experiments/render_readme.py --agent agent-loop.json --readme README.md

# the long-context store: distances to 1,024 steps, four controls, a byte ledger
python experiments/long_memory.py --out long-memory.json
python experiments/render_readme.py --memory long-memory.json --readme README.md

# the dashboard, from the committed results (deterministic, no clock)
python experiments/render_dashboard.py --voice voice-affect.json \
    --agent agent-loop.json --out dashboard.html
```

`experiments/run.py --help` lists the knobs; `--steps`, `--seeds`, `--pairs`,
`--queries` and `--blocks` are the ones that cost time.

The voice, agent and memory blocks render on their own (`--voice`, `--agent`,
`--memory`) rather than as part of the results command, so each can be
regenerated without touching the others. The results command takes
`--length-extrapolation` as well: that section is *inside* the results block, and
the renderer **refuses** to write the block without the file rather than silently
dropping a section it cannot produce. That refusal is the fix for a real bug
described in the limitations, and `tests/test_render_readme.py` holds it in place
by re-rendering every block from its JSON and comparing it with the README byte
for byte.

## Streaming: what it costs to keep reading

Everything above compares the two models on a fixed-length sequence, where both
see the whole input. Real use is a stream: the model must emit token `t` before
it has seen `t+1`, and the resource that decides what you can build is the state
carried between tokens.

`src/beyond_attention/streaming.py` implements that for both. Token-by-token
output is asserted to match the parallel forward pass for each model (max
absolute difference under `1e-4`), so this is the same model, not an
approximation of it. Then:

| context length | SSM state | attention KV cache |
|---:|---:|---:|
| 256 | 4,864 elts · 19 KiB | 65,536 elts · 256 KiB |
| 1,024 | 4,864 elts · 19 KiB | 262,144 elts · 1 MiB |
| 4,096 | 4,864 elts · 19 KiB | 1,048,576 elts · 4 MiB |
| 16,384 | 4,864 elts · 19 KiB | 4,194,304 elts · 16 MiB |
| 65,536 | 4,864 elts · 19 KiB | 16,777,216 elts · 64 MiB |
| 262,144 | 4,864 elts · 19 KiB | 67,108,864 elts · 256 MiB |
| 1,048,576 | 4,864 elts · 19 KiB | 268,435,456 elts · 1 GiB |

The SSM column is not approximately constant, it is *identically* constant: the
same 4,864 elements at 256 tokens and at 1,048,576. That is `2 layers x
(d_inner x d_state + (kernel-1) x d_inner)` with nothing length-dependent in it,
and a test asserts the count does not move.

The rows to 65,536 come from `experiments/stream_cost.py`. The last two come from
`experiments/long_context.py`, which streams a million tokens and re-checks at
every single step that the carried state has not moved. Across a **64x range of
lengths** the per-token cost varies by **1.14x** — flat, as a fixed-size
recurrence should be, and slow only because the loop is plain Python.

At a million tokens the difference in carried state is **55,200x** (19,456 bytes
against 1 GiB). It is also not a tuning gap: attention's cache is
`2 x n_heads x head_dim x length` per layer, and the `length` is structural. No
amount of kernel work removes it, because the model needs the keys and values it
has already seen. This is the concrete sense in which the two are different tools
rather than different speeds — a state-space model reads a stream in constant
memory, and an attention model, by construction, cannot.

### Measured, not just arithmetic

Every figure above is `numel * 4` — arithmetic on a tensor shape, which says
what the state *should* cost rather than what the process does.
`experiments/stream_memory.py` streams for real and samples `VmRSS` from
`/proc/self/status` throughout:

| | measured |
|---|---|
| SSM streamed over 1,048,576 tokens | RSS **243.1 MB, spread 0.00%** |
| positive control: same sampling loop, KV cache growing to 262,144 tokens | **+192.9 MB** |
| attention KV cache at 1,048,576 tokens | **1,024.0 MB** (1,024.0 MB arithmetic) |

The control is what makes the flat line mean anything. A flat line is only
evidence if the instrument can see growth, so the identical loop is run a second
time with the attention cache growing underneath it, and it rises by 193 MB.
Without that, "RSS did not move" is indistinguishable from "RSS was never
looked at".

Two things worth stating plainly rather than leaving to be inferred:

* **RSS is an upper bound, and that is why it is the right instrument.** The CPU
  allocator caches freed blocks instead of handing them back to the OS, so a loop
  that allocated and freed would plateau rather than return to its baseline.
  Reuse cannot hide an accumulation, which is exactly the property this claim
  needs. The measured cache column above lands on the arithmetic one to the
  megabyte, which is the cross-check that both are measuring the same thing.
* **The model's state is constant; the outputs are not.** `stream_step` discards
  each step's logits, which is what makes the loop bounded. `stream_sequence`
  appends them and returns a stacked `(B, L, vocab)` tensor — **132 MB** at a
  million tokens. That is the caller's choice rather than a property of the
  model, but a reader who conflated the two would be wrong in a direction that
  flatters the architecture, so both are measured.

### The other question: how long an input can each one read?

Constant carried state is a claim about memory. The parallel forward pass is a
different question, and on this CPU it has a hard wall, measured rather than
extrapolated:

| length | attention parallel forward | peak RSS |
|---:|---|---:|
| 4,096 | ok | 341 MB |
| 16,384 | ok | 1,567 MB |
| 65,536 | `RuntimeError: can't allocate memory` | — |

Peak memory tracks `4 x L^2` — a materialised float32 score matrix — to within
about 1.5x at 16,384 tokens. So the fused `scaled_dot_product_attention` call in
`model.py` is **not** taking a fused path here: `torch` 2.14 accepts
`SDPBackend.FLASH_ATTENTION` as a context manager on this machine without
raising, and then materialises the quadratic matrix anyway. That gap between what
the runtime advertises and what it does is why the measurement is reported and
the advertisement is not.

**An earlier version of this experiment took the machine down.** It answered the
question by trying, and at 131,072 tokens the attempt did not raise — memory
reached 15.6/16.0 GiB and swap 28.8/31.9 GiB, and `systemd-oomd` killed the
entire process scope: seventeen processes, no traceback, nothing to catch. That
is not a measurement, it is an outage. Attempts now run in a child with a hard
`RLIMIT_AS` cap, which turns an unkillable system event into an ordinary
catchable `RuntimeError` — the 65,536-token row above is that error, naming the
exact allocation it was refused. Where the wall sits depends on the machine and
on the cap; that the wall is **quadratic in length** does not.

Two honest caveats:

* **This is memory, not speed.** The streaming loop here is deliberately
  unoptimised, and the timings it prints are not a throughput result. The
  length-scaling section above remains the place to look for speed.
* **The attention *streaming* loop is still only measured to 4,096.** It is
  `O(L^2)`, so streaming it further is impractical on CPU. The cache size is a
  closed form, and `experiments/stream_cost.py` asserts that formula against
  measurement at every length where both exist. The prediction is checked, not
  assumed. (The parallel-forward wall above is a separate, measured result.)

Reproduce with:

```bash
python experiments/stream_cost.py --out stream-cost.json
python -u experiments/long_context.py --out long-context.json
```

## Voice: hearing *how* something is said

`src/beyond_attention/voice.py` is a front end for a different question from the
one the rest of this repository asks. The operator wanted the model to
understand **how** something was said — the feeling in a voice — rather than
what words were said. That is the direction this module works in: a raw waveform
goes in, framed prosodic features come out, and a seeded projection puts them in
the `(B, L, D)` layout `selective_scan` consumes.

### What this is, and what it is not

**It is** a working, tested measurement of prosody. F0 per frame by
autocorrelation over a bounded 60–400 Hz lag range with a voiced/unvoiced
decision; frame energy and its contour dynamics; zero-crossing rate; spectral
centroid, rolloff and flatness; and the descriptors affect work actually uses —
F0 mean and *spread*, energy mean and spread, voiced ratio, jitter, and a
speaking-rate proxy. Every number in the block below is generated from
`voice-affect.json` by `experiments/render_readme.py`.

**It is not** an emotion recogniser, and nothing in it is trained on anything.
Three claims that would be easy to make from the table below, and are all false:

* *"The descriptors identify emotion in speech."* They identify which of four
  **synthetic** conditions generated a signal — and the conditions were built to
  differ along exactly the axes the descriptors measure, with the labels coming
  from the generator rather than from a listener. No human voice was involved at
  any point, and there is no labelled affect corpus in this repository to
  validate against.
* *"The embedding carries affect."* `VoiceEncoder` is a fixed random projection
  with no bias and no nonlinearity — a test asserts that it is exactly the
  matrix product it documents — so it cannot represent anything the feature
  vector does not already contain. It is a shape adapter for the SSM, not a
  model, and it is deliberately incapable of adding information.
* *"Jitter and speaking rate are measured the way a phonetics tool measures
  them."* Jitter here is a frame-level successive-F0 difference restricted to
  adjacent voiced frames, not the cycle-to-cycle perturbation a Praat-style
  analysis reports, and it is bounded below by the frame hop. The rate proxy
  counts voiced *segments*; it approximates syllable rate only when the
  segmentation is right, and it cannot see unvoiced consonants at all. Both are
  documented as proxies in the module, and both are used as proxies here.

<!-- VOICE:BEGIN -->
**Synthetic utterances** — 64 utterances (16 per condition, 2.0 s each) at 16,000 Hz, seed 0. Classifier: leave-one-out nearest centroid, z-scored per fold, over 15 descriptors.

| condition | F0 target (Hz) | F0 mean (Hz) | F0 std (Hz) | energy std | energy std (voiced) | voiced runs/s | jitter (Hz) |
|---|---:|---:|---:|---:|---:|---:|---:|
| neutral | 150 | 151.7 | 1.68 | 0.1347 | 0.0138 | 2.02 | 0.307 |
| aroused | 225 | 225.3 | 21.38 | 0.1571 | 0.1151 | 5.05 | 4.476 |
| subdued | 110 | 111.5 | 0.91 | 0.1371 | 0.0094 | 1.01 | 0.299 |
| unsteady | 255 | 250.7 | 35.51 | 0.1305 | 0.0468 | 3.54 | 5.655 |

**Separability in descriptor space** — leave-one-out nearest centroid, chance 0.250

Accuracy **0.969** over 64 utterances (+13.3σ against chance). The closest pair of condition centroids is 1.50 apart against a mean within-condition spread of 1.09 — a ratio of **1.4x**.

| true \ predicted | neutral | aroused | subdued | unsteady |
|---|---:|---:|---:|---:|
| neutral | 14 | 0 | 0 | 2 |
| aroused | 0 | 16 | 0 | 0 |
| subdued | 0 | 0 | 16 | 0 |
| unsteady | 0 | 0 | 0 | 16 |

Each descriptor on its own, same classifier:

| descriptor alone | accuracy |
|---|---:|
| `energy_flux_mean` | 1.000 |
| `speaking_rate` | 1.000 |
| `f0_mean` | 0.984 |
| `f0_range` | 0.969 |
| `zcr_mean` | 0.969 |
| `f0_std` | 0.953 |
| `energy_std_voiced` | 0.953 |
| `voiced_ratio` | 0.906 |
| `flatness_mean` | 0.906 |
| `centroid_mean` | 0.859 |
| `jitter_relative` | 0.844 |
| `jitter` | 0.719 |
| `energy_std` | 0.625 |
| `energy_mean` | 0.406 |
| `energy_mean_voiced` | 0.391 |

| control | measured | what it rules out |
|---|---|---|
| labels shuffled, 200 rounds | mean 0.261, 95th pct 0.359, max 0.422 | that the accuracy is the classifier's rather than the descriptors' (chance 0.250) |
| two conditions, identical parameters | 0.562 (+0.7σ) | that anything other than the generator's parameters separates the conditions (chance 0.500) |
| white noise through the voiced decision | voiced ratio max 0.000, largest autocorrelation peak 0.140 | a voiced/unvoiced decision that always says yes (threshold 0.45) |
| F0 recovery against the generator | mean 1.07%, worst 1.67% | descriptors that do not track what the synthesiser was asked for |
| encoder separates two utterances | 0.1058 max difference | a degenerate all-zero projection |
| bridge output | `(1, 198, 32)` | a front end that never reaches the model's `(B, L, D)` layout |
<!-- VOICE:END -->

### What the numbers say, including the unflattering parts

* **The four conditions are separable, and not cleanly.** Leave-one-out nearest
  centroid gets most of the 64 utterances right against a chance of 0.250, but
  the closest pair of condition centroids is only about **1.4×** the mean
  within-condition spread apart, and the confusion is real: two of the sixteen
  "neutral" utterances sit closer to the "unsteady" centroid than to their own.
  An accuracy read without the margin would overstate how far apart these are.
* **The axis the design varied is not always the axis the obvious descriptor
  carries.** `energy_std` — the frame-RMS spread over *all* frames — recovers
  the condition 0.625 of the time on its own, because it mostly measures where
  the pauses are rather than how loud the speech is. Restricting the same
  statistic to voiced frames lifts that to 0.953. Both are reported; the second
  exists because the first was measured and found wanting.
* **The label-shuffle control is what makes the accuracy mean anything.** The
  same classifier on permuted labels averages 0.261 against a chance of 0.250,
  and the pair of conditions generated from *identical* parameters is classified
  at 0.562 — about +0.7σ, which is noise, and is the right answer for two groups
  that differ in nothing but the random stream.
* **The F0 estimator is checked against the generator, and its advertised
  range is wider than its working range.** Mean absolute error against the
  synthesised pitch is about 1%, and a 10 Hz sweep from 100 Hz to 380 Hz stays
  inside 1.6%. Below that it fails rather than degrading: at 70 Hz the estimate
  is 3.2% high and only 40% of frames are called voiced, and at 60 Hz the
  strongest peak inside the lag range is a short-lag artifact, so the frame is
  reported near the 400 Hz end of the band. A 25 ms window does not hold the two
  periods a 60 Hz fundamental needs, and the fix is a longer window rather than
  a different threshold. The tests accordingly assert a 2% tolerance at 150, 220
  and 300 Hz rather than across the advertised 60–400 Hz range.

None of this is a result about emotion. It is a result about whether a front end
measures the four prosodic axes it claims to, on signals where those axes are
known because they were set by hand.

The tests were mutation-checked the same way the scan's were, because a test
that passes is not evidence until something that should break it does. Ten
deliberate faults in `voice.py` — accepting every frame as voiced, removing the
energy gate, swapping the frame and hop, giving the encoder a bias, collapsing
the voiced-frame energy spread back onto all frames, counting voiced frames
instead of runs, letting jitter span the pauses, moving the rolloff threshold to
5% of the power, and weighting the spectral centroid by power instead of
magnitude — each fail a specific test, and seven of the ten fail exactly one.
The voicing fault is caught by the white-noise control, which is the reason that
control exists: it is the fault a pure-tone test cannot see.

### The trained classifier: what this is, and what it is not

`voice.py` said — in its module docstring and in this README — that no
classifier had been trained, and that a regression fitted to the descriptors
would be a hypothesis rather than a result. `src/beyond_attention/emotion.py`
closes that specific gap: five conditions built from the published acoustic
correlates of crying/sad-sobbing, excitement, anger, calm and fear; 26 features
per utterance, the 15 from `voice.py` plus 11 new voice-quality measurements
(harmonics-to-noise ratio, shimmer, tremor rate, F0 slope and terminal fall,
pause structure, onset sharpness, spectral spread); and a real PyTorch MLP with
an optimisation loop, a split **by utterance**, a fixed seed, and model
selection on validation only. The headline is held-out accuracy.

**The labels are the synthesiser's, so the classifier learns our acoustic model
of these emotions, not a listener's. It is not validated on speech.** That is
the single most important sentence here. Every "crying" utterance is a signal this repository generated
from a parameter tuple this repository chose, and its label is that tuple's
name. No human listener heard it, no annotator labelled it, and no real
recording is involved anywhere. Three claims that would be easy to make from
the block below, and are all false:

* *"It understands emotion."* It separates five clusters that were placed in a
  26-dimensional space by hand, along exactly the axes the features measure. It
  is not evidence that it recognises emotion in a voice, and it cannot be: the
  target is our own generator. The honest test of whether that is recognition
  or the generator is to hold a
  whole condition out, and it **fails**: trained on four conditions and pointed
  at the fifth, the model does not abstain and does not spread its answers — it
  puts 50–100% of the unseen condition onto one training class and is confident
  about it (assigned-class NLL 0.05–1.30 nats), naming the same class the
  untrained nearest-centroid rule names on three of the five. A model that
  understands emotion would have somewhere to put a condition it had never
  seen; this one has nowhere, because the target is our own generator.
* *"It can tell crying from excitement."* It separates the two signals this
  generator produces under those names, and the ablation says why that is not
  much of a claim: **19 of the 26 features separate the pair perfectly on their
  own**, removing any single feature changes the held-out accuracy by exactly
  **0.000**, and one feature — mean F0, with the conditions built 40 Hz apart
  and tight within-condition means — is enough to reach 1.000. The two
  conditions do share elevated F0 in the sense that both are high-pitched, but
  they were also built apart on twenty other axes at once, so what the
  classifier is reading is the separation the generator put there. There is no
  single feature "doing it", and naming one would be choosing the answer.
* *"This transfers to real speech."* Nothing here was tested on real speech.
  There is no affect corpus in this repository, none is downloaded, and no
  network access is used. Every condition is synthetic, and the only signals the
  extractor has ever seen are ones this repository's own synthesiser produced.
  The correct statement is not that it fails to transfer but that transfer is
  **untested**, in either direction, because there is no held-out human data
  anywhere in the path.

<!-- EMOTION:BEGIN -->
**Acoustically-grounded conditions** — 200 utterances (40 per condition, 2.0 s each) at 16,000 Hz, seed 0. Every parameter is a published acoustic correlate of the state, cited in `emotion.py`; the correlate column in the JSON carries the citation for each one.

| condition | F0 base (Hz) | F0 range (st) | tremor (Hz @ st) | jitter (st) | shimmer | breathiness | rate (syl/s) | duty | onset (ms) | harmonic α |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| afraid | 300 | 2.5 | 6.0 @ 2.0 | 0.05 | 0.16 | 0.30 | 5.0 | 0.65 | 50 | 1.2 |
| angry | 190 | 1.5 | 3.0 @ 1.5 | 0.03 | 0.06 | 0.15 | 5.0 | 0.70 | 5 | 0.6 |
| calm | 135 | 0.5 | 1.0 @ 0.4 | 0.02 | 0.02 | 0.08 | 2.2 | 0.62 | 40 | 1.5 |
| crying | 320 | 2.0 | 6.5 @ 2.5 | 0.06 | 0.35 | 0.55 | 3.0 | 0.45 | 60 | 1.6 |
| excited | 280 | 3.5 | 1.2 @ 3.0 | 0.02 | 0.04 | 0.10 | 5.5 | 0.75 | 10 | 1.0 |

What the extractor measures on those signals (means over the condition's utterances; the full 26-feature vector per condition is in the JSON):

| condition | F0 mean (Hz) | F0 std (Hz) | F0 range (Hz) | jitter (Hz) | shimmer | HNR (dB) | tremor est. (Hz) | final/initial F0 | pauses/s | onset sharpness | energy mean | energy std (voiced) | centroid (Hz) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| afraid | 301.2 | 15.3 | 52.0 | 4.53 | 0.098 | 8.23 | 5.99 | 1.026 | 1.43 | 0.205 | 0.115 | 0.098 | 1687 |
| angry | 189.0 | 6.3 | 23.3 | 0.92 | 0.065 | 15.26 | 3.03 | 0.980 | 0.35 | 0.362 | 0.532 | 0.202 | 1307 |
| calm | 135.3 | 1.4 | 5.9 | 0.46 | 0.042 | 26.53 | 0.95 | 0.996 | 1.52 | 0.230 | 0.166 | 0.058 | 730 |
| crying | 317.2 | 25.5 | 115.1 | 7.00 | 0.167 | 4.04 | 1.60 | 0.884 | 2.26 | 0.223 | 0.073 | 0.113 | 1370 |
| excited | 281.7 | 20.1 | 62.2 | 1.24 | 0.075 | 21.12 | 1.17 | 1.018 | 0.00 | 0.353 | 0.368 | 0.217 | 1389 |

**17 of 17 documented correlate checks hold.** Each is a test that the generated signal shows the acoustic profile its citation names — not that the profile means the emotion.

The checks: `crying: highest jitter`; `crying: highest shimmer`; `crying: lowest HNR`; `crying: lowest final/initial F0 ratio`; `crying: lower energy than angry`; `crying: more pauses than excited`; `excited: widest F0 range of the low-jitter conditions`; `excited: highest voiced-frame energy spread`; `excited: lower jitter than crying`; `angry: highest energy`; `angry: sharpest onsets`; `angry: higher centroid than calm`; `afraid: higher F0 than calm`; `afraid: more jitter than excited`; `afraid: softer onsets than angry`; `calm: lowest F0 spread`; `calm: lowest voiced-frame energy spread`

**Training** — 40 held-out utterances, split by utterance (train/validation/test 120/40/40), seed 0. Classifier: MLP 26->32->5, ReLU, full-batch Adam lr=0.01, weight_decay=0.0001, 1500 steps, snapshot chosen on validation accuracy.

Held-out accuracy **1.000** against chance 0.200 (+12.6σ over 40 utterances). Train 1.000, validation 1.000, snapshot at step 10. Baselines on the same split: majority 0.200, the previous increment's untrained nearest-centroid rule 1.000.

| true \ predicted | afraid | angry | calm | crying | excited | recall |
|---|---:|---:|---:|---:|---:|---:|
| afraid | 8 | 0 | 0 | 0 | 0 | 1.00 |
| angry | 0 | 8 | 0 | 0 | 0 | 1.00 |
| calm | 0 | 0 | 8 | 0 | 0 | 1.00 |
| crying | 0 | 0 | 0 | 8 | 0 | 1.00 |
| excited | 0 | 0 | 0 | 0 | 8 | 1.00 |

| control | measured | what it rules out |
|---|---|---|
| labels shuffled, 25 rounds, same recipe | mean 0.196 ± 0.119, 95th pct 0.350, max 0.400 | that the accuracy is the features' rather than the labels' (chance 0.200) |
| trained on Gaussian noise of the same shape | mean 0.190, 95th pct 0.220, max 0.225 | that 26 random columns carry the task (the positive control for real learning) |
| chance | 0.200 | a floor, not a result |
| majority class | 0.200 | a model that ignores its input |

**Ablations.** `without` removes one feature family and retrains; `only` keeps one family and discards the rest. The point of the second block is that a high accuracy without F0 is not evidence of a deeper representation: with six acoustic axes varied at once, several families are independently sufficient, so no one family is necessary.

| features | n | held-out accuracy |
|---|---:|---:|
| F0 level and contour removed | 21 | 1.000 |
| everything pitch-derived removed | 18 | 1.000 |
| only energy_dynamics | 6 | 1.000 |
| only f0_level_and_contour | 5 | 0.975 |
| only pitch_dynamics | 3 | 1.000 |
| only rhythm_and_pauses | 5 | 1.000 |
| only spectral_shape | 3 | 0.975 |
| only voice_quality | 4 | 0.975 |

Each feature alone, and each feature removed, on the five-class task:

| feature | alone | without | drop |
|---|---:|---:|---:|
| `voiced_ratio` | 1.000 | 1.000 | +0.000 |
| `hnr_db` | 1.000 | 1.000 | +0.000 |
| `f0_mean` | 0.975 | 1.000 | +0.000 |
| `f0_std` | 0.975 | 1.000 | +0.000 |
| `jitter` | 0.975 | 1.000 | +0.000 |
| `flatness_mean` | 0.975 | 1.000 | +0.000 |
| `hnr_db_std` | 0.975 | 1.000 | +0.000 |
| `f0_range` | 0.950 | 1.000 | +0.000 |
| `tremor_rate_hz` | 0.950 | 1.000 | +0.000 |
| `centroid_std` | 0.950 | 1.000 | +0.000 |
| `energy_mean` | 0.925 | 1.000 | +0.000 |
| `shimmer` | 0.925 | 1.000 | +0.000 |
| `pause_fraction` | 0.925 | 1.000 | +0.000 |
| `energy_flux_mean` | 0.900 | 1.000 | +0.000 |
| `centroid_mean` | 0.900 | 1.000 | +0.000 |
| `jitter_relative` | 0.850 | 1.000 | +0.000 |
| `speaking_rate` | 0.800 | 1.000 | +0.000 |
| `f0_slope_st_per_s` | 0.800 | 1.000 | +0.000 |
| `pause_rate` | 0.800 | 1.000 | +0.000 |
| `pause_mean_s` | 0.750 | 1.000 | +0.000 |
| `energy_std` | 0.725 | 1.000 | +0.000 |
| `energy_mean_voiced` | 0.725 | 1.000 | +0.000 |
| `f0_final_ratio` | 0.700 | 1.000 | +0.000 |
| `energy_std_voiced` | 0.650 | 1.000 | +0.000 |
| `onset_sharpness` | 0.500 | 1.000 | +0.000 |
| `zcr_mean` | 0.475 | 1.000 | +0.000 |

**Crying against excited** — 16 held-out utterances, a dedicated binary model on the same split rule, chance 0.500. Accuracy **1.000**.

| true \ predicted | crying | excited |
|---|---:|---:|
| crying | 8 | 0 |
| excited | 0 | 8 |

| feature | alone | without | drop |
|---|---:|---:|---:|
| `f0_mean` | 1.000 | 1.000 | +0.000 |
| `f0_range` | 1.000 | 1.000 | +0.000 |
| `jitter` | 1.000 | 1.000 | +0.000 |
| `jitter_relative` | 1.000 | 1.000 | +0.000 |
| `energy_mean` | 1.000 | 1.000 | +0.000 |
| `energy_std` | 1.000 | 1.000 | +0.000 |
| `energy_mean_voiced` | 1.000 | 1.000 | +0.000 |
| `energy_std_voiced` | 1.000 | 1.000 | +0.000 |
| `energy_flux_mean` | 1.000 | 1.000 | +0.000 |
| `voiced_ratio` | 1.000 | 1.000 | +0.000 |
| `speaking_rate` | 1.000 | 1.000 | +0.000 |
| `flatness_mean` | 1.000 | 1.000 | +0.000 |
| `hnr_db` | 1.000 | 1.000 | +0.000 |
| `hnr_db_std` | 1.000 | 1.000 | +0.000 |
| `tremor_rate_hz` | 1.000 | 1.000 | +0.000 |
| `f0_slope_st_per_s` | 1.000 | 1.000 | +0.000 |
| `pause_rate` | 1.000 | 1.000 | +0.000 |
| `pause_fraction` | 1.000 | 1.000 | +0.000 |
| `pause_mean_s` | 1.000 | 1.000 | +0.000 |
| `zcr_mean` | 0.938 | 1.000 | +0.000 |
| `shimmer` | 0.938 | 1.000 | +0.000 |
| `f0_final_ratio` | 0.938 | 1.000 | +0.000 |
| `centroid_std` | 0.938 | 1.000 | +0.000 |
| `f0_std` | 0.875 | 1.000 | +0.000 |
| `onset_sharpness` | 0.875 | 1.000 | +0.000 |
| `centroid_mean` | 0.750 | 1.000 | +0.000 |

The first row is the best single feature and it is **not** an answer to "which feature tells them apart": every `drop` in the table is +0.000, so no feature is load-bearing, and **19 of the 26 features** separate the pair perfectly on their own. The smallest set that reaches the accuracy, chosen on validation and scored on test, is **1 feature**: `f0_mean` (validation 1.000, test 1.000, stopping because `validation_perfect`).

| features kept | added | validation | test |
|---:|---|---:|---:|
| 1 | `f0_mean` | 1.000 | 1.000 |

**Cross-condition generalisation** — train on four conditions, hold the fifth out entirely. The held-out label is not in the trained label space, so accuracy is 0 by construction and an "accuracy" row here would be a restatement of that rather than a measurement. What is measured is where the model puts a condition it has never seen.

| held out | the model calls it | fraction | assigned-class NLL (nats) | distance to nearest training centroid (spreads) | nearest-centroid says | agrees |
|---|---|---:|---:|---:|---|---|
| afraid | crying | 0.50 | 1.30 | 6.04 | excited | no |
| angry | afraid | 0.68 | 1.09 | 4.10 | excited | no |
| calm | angry | 1.00 | 0.79 | 6.55 | angry | yes |
| crying | afraid | 1.00 | 0.05 | 9.73 | afraid | yes |
| excited | angry | 1.00 | 0.06 | 3.33 | angry | yes |

Mean dominant fraction **0.835** against 0.200 for any one training class: the model does not abstain and it does not spread its answers. It is confident in them, too — the mean cross-entropy of the class it *did* assign is 0.66 nats on the unseen conditions against 0.62 nats on the held-out rows of the conditions it was trained on, so it is not that it does not know; it is that it has no way to say so. The untrained nearest-centroid rule names the same class on 0.60 of the five held-out conditions, so the collapse is a property of the feature space rather than of the trained model.
<!-- EMOTION:END -->


### What the classifier's numbers say, including the unflattering parts

* **The headline accuracy is the least interesting number in the block.** It is
  1.000 held out, and it is 1.000 for a reason that is visible in the ablation
  table: only 2 of the 26 features reach 1.000 *alone*, but removing any one
  of them changes nothing at all, and each of the six feature families is
  sufficient by itself (0.975–1.000). The previous increment's untrained
  nearest-centroid rule also scores 1.000 on the same split, so on this task
  training is not what made the difference. The controls are what make the accuracy mean anything, and they
  behave: labels shuffled gives 0.196 against a chance of 0.200, Gaussian noise
  features give 0.190, and the identical loop learns a planted five-cluster task
  at better than 0.9 — so those two controls are not measuring a broken
  optimiser.
* **"A model that only knows loud and high is not recognising crying" — but
  this one does not have to be, and that is the finding.** Removing the entire
  F0 level/contour family leaves the held-out accuracy at 1.000, and removing
  everything pitch-derived — F0 level, spread, slope, terminal fall, jitter,
  relative jitter and tremor rate — leaves it at 1.000 as well. That is not
  evidence of a subtle representation. It is what happens when six acoustic
  axes are varied at once: rhythm, voice quality, energy dynamics and spectral
  shape each carry the task on their own, so no one family is necessary. The
  ablation that was meant to expose a shallow model instead exposed how easy
  the task was built to be.
* **The cross-condition check is the real result, and it is a negative.** Mean
  dominant assignment is 0.835 against 0.200 for any one training class.
  Held-out crying is called "afraid" 100% of the time at a distance of 9.7
  within-class spreads — farther from everything the model knows than the
  training classes are from each other — and the model is confident about it
  (0.05 nats). Held-out angry is split 68/32, the only case that is not a clean
  collapse. The untrained nearest-centroid rule names the same class on three
  of the five, so this is a property of the feature space rather than a
  pathology of the trained model: the conditions form five regions with no
  "other" region between them, and an unseen condition lands in whichever one
  is nearest.
* **One new feature does not work, and the block reports it rather than hiding
  it.** `tremor_rate_hz` recovers the generator's modulation rate on excited
  (1.17 against 1.2 Hz), angry (3.03 against 3.0), afraid (5.99 against 6.0)
  and calm (0.95 against 1.0), and returns 1.60 Hz against 6.5 Hz on crying,
  because the pitch track of a strongly breathy signal is noisy enough that
  low-frequency estimation error outweighs the tremor. It is kept, used by the
  classifier, and documented as valid only where the contour is periodic and
  the phonation clean.
* **The spectral centroid does not rank the conditions the way a naive reading
  of the citation would.** Banse & Scherer report a higher centroid for anger
  than for calm or neutrality, and the generated `angry` condition shows that
  (1,307 Hz against calm's 730); but fear's breathy high-frequency voice
  measures brighter still (1,687 Hz) and excitement (1,389 Hz) and crying
  (1,370 Hz) sit just below it. The centroid is driven by F0 level and by
  turbulence noise at least as much as by "tension", so it is reported as
  measured rather than bent to fit the claim.
* **Seventeen documented correlate checks are asserted, and all seventeen
  hold.** Each says the generated signal shows the acoustic profile its
  citation names — crying has the highest jitter and shimmer and the lowest HNR
  and the lowest terminal F0 ratio; excitement has the widest F0 range and the
  highest voiced-frame energy spread with low jitter; anger is the loudest with
  the sharpest onsets; calm has the lowest F0 and energy spread; fear has high
  F0, high jitter and soft onsets. They are a test, not a paragraph: a change
  that made a condition stop implementing its correlate would fail the suite.

## Agent: acting, not only reading

Everything above reads: a waveform or a token sequence goes in and a number comes
out. `src/beyond_attention/agent.py` is the other direction — a loop that chooses
a tool, runs it, reads the result and chooses again, with an explicit step budget
and a trace that replays.

The memory is the part worth describing exactly, because it is where the
temptation to overclaim is strongest. The trajectory's events — the task's
instructions and each observation — are embedded into the `(x, delta)` pair
`build_scan_terms` consumes, and streamed through the S6 recurrence the rest of
this repository implements. The resulting state *is* the agent's working memory:
an eight-wide register file whose dimensions are named (`carry`, `observations`,
`op`, `arg`, `arg2`, `miss`, `kind`, `instructions`) and which the policy decodes
into an opcode, an operand and the carried value. The `selective` family appends
key-addressed memory slots to those eight registers — `state_width` of them — so
the same recurrence is a register file whose width the task sets, and the
experiment sweeps that width rather than asserting it. `A` is chosen by hand so that a
write (`delta = 1`) replaces a register exactly and a hold (`delta = 0`) leaves it
bit-for-bit alone. A test runs the repository's own `selective_scan` on the same
input and requires it to reproduce this state exactly, so "the memory is the SSM's
recurrence" is a check rather than a claim.

### What this is, and what it is not

**It is** a working, tested agent loop: seven tools with strict schemas and typed
errors (`add`, `mul`, `sub`, `lookup`, `remember`, `fetch`, `finish`), a step
budget that is honoured exactly, a replayable trace of every action and result,
and a seeded task suite whose answers are analytic — computed in Python integers
by `evaluate_plan`, so correctness is a property of the task rather than of a
model. Every number in the block below is generated from
`agent-loop.json` by `experiments/render_readme.py`.

**It is not** a language model driving tools, and it is not general. Three claims
that would be easy to make from the table below, and are all false:

* *"This is an agent that generalises."* It runs one fixed, hand-written
  controller over a closed tool set — `add`, `mul`, `sub`, `lookup`, `remember`,
  `fetch`, `finish`, seven functions fixed at import time — and a closed task
  family: six plan shapes emitted by a seeded generator, plus the three recall
  shapes that use the external store. The "task text" is that
  instruction grammar delivered as structured events; there is no tokenizer, no
  parsing and no language understanding anywhere in the path. The loop cannot be
  asked for a task the generator does not generate, a tool the registry does not
  hold, or an instruction the grammar does not define, and 1.000 says only that
  the controller matches the shapes it was written against.
* *"The loop learns."* Nothing is trained. There is no gradient, no optimiser and
  no loss in this path: the recurrence's `A`, its read/write gate and the policy's
  branches are constants chosen by hand. A seed decides which tasks are generated
  and how the random control draws, and nothing else. The seven tools are pure
  functions of their arguments — no clock, no filesystem, no network — and a test
  reads the module's imports to keep it that way.
* *"The SSM is what makes it work."* On the five arithmetic families it is not.
  The no-memory control — the state wiped before every decision, with only the
  current instruction re-streamed — keeps every task whose answer is written in
  the task text and loses every task whose answer is an intermediate result, so
  *carrying the value* is necessary. It is not evidence that the recurrence is:
  the same controller with the carried value in one Python `int` solves exactly
  the same tasks, in the same number of steps, calling the same tools in the same
  order, and a variable would carry the value as well as the state does. The
  `selective` family is the one place that changes, and the change is narrow: one
  register cannot hold two keys at once, so the scalar scores 0.320 against the
  gated state's 1.000, and a state of the *same width* with a constant gate
  scores 0.200. That is a real difference, and what it supports is bounded below
  the table: the gate is hand-set in `embed`, the controller is hand-written, a
  Python dict in `evaluate_plan` computes the same answers with no model at all,
  and nothing anywhere in the path is trained.

<!-- AGENT:BEGIN -->
**Seeded task suite** — 300 tasks (50 per family), seed 0. Each task is a plan over bounded integers plus a bounded key/value table, and its answer is computed in Python integers by `evaluate_plan`, so correctness is a property of the task. Tools: `add`, `mul`, `sub`, `lookup`, `remember`, `fetch`, `finish`.

The memory is a selective-scan state 8 registers wide, one named register per dimension: `carry`, `observations`, `op`, `arg`, `arg2`, `miss`, `kind`, `instructions`.

| task family | loop steps needed | budget 1 | budget 2 | budget 3 | budget 4 | budget 5 | budget 6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| literal | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| one_op | 2 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| two_op | 3 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| branch | 4 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 |
| lookup | 5 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 |
| selective | 6 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |

**Controls at budget 6** — per-family solve rate, with the agent's own row for comparison. "Overall" averages the 6 families.

| condition | literal | one_op | two_op | branch | lookup | selective | overall |
|---|---:|---:|---:|---:|---:|---:|---:|
| the agent, memory = the SSM state | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | **1.000** |
| no memory: the state is wiped before each decision | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | **0.167** |
| scalar carry: the same controller, one Python int | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.320 | **0.887** |
| fixed decay: the same width, a constant gate | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.200 | **0.867** |
| random action: tools and arguments uniform | 0.001 | 0.001 | 0.003 | 0.003 | 0.006 | 0.003 | **0.003** |

The budget-1 slice of the agent is the one-step control: **0.167** solved over all 300 tasks. Only the `literal` family, whose answer is written in the task text, is reachable in a single decision — so the suite is not one call deep.

**Where the first version's residual bit.** Of the 50 `branch` tasks, 8 arrive at the `IFPOS` decision (step 2) carrying exactly 0 — the value a write residual turns positive. Those are exactly the tasks the `A = -50` register got wrong, and the last column below is the fix.

**Register write exactness** — what a register holds after being written 9 and then 0, for four choices of the decay. A residual above zero is not a rounding detail: `IFPOS` branches on the sign of this number.

| A | exp(A) | residual after overwriting | sign test holds |
|---:|---:|---:|---|
| -50 | 1.929e-22 | 1.736e-21 | no |
| -400 | 1.915e-174 | 1.724e-173 | no |
| -745 | 4.941e-324 | 4.447e-323 | no |
| -800 | 0.000e+00 | 0.000e+00 | yes |

**The published trace** — `add 3, then add 4, then multiply the result by 5, then report the key whose value equals the result, then report the result`, answer `7`, 5 steps, stop reason `finish`, solved `True`. "Reads" is the state the policy acted on, before the call.

| step | instruction | reads: op / arg / carry / observations | action | result |
|---:|---|---|---|---:|
| 0 | ADD 3 | 3 / 3 / 0 / 0 | `add(a=0, b=3)` | 3 |
| 1 | ADD 4 | 3 / 4 / 3 / 1 | `add(a=3, b=4)` | 7 |
| 2 | MUL 5 | 4 / 5 / 7 / 2 | `mul(a=7, b=5)` | 35 |
| 3 | KEY | 6 / 0 / 35 / 3 | `lookup(value=35)` | 7 |
| 4 | RET | 1 / 0 / 7 / 4 | `finish(answer=7)` | 7 |

### Selective memory: retention that depends on the input

**What this family asks.** 3 events carry a value tagged with a key, 2 carry a value tagged with nothing, and the events arrive in a shuffled order; then a query asks for the value of one of the keys, drawn uniformly. The answer is a dict lookup inside `evaluate_plan`, computed in Python integers — the target never consults the memory, which is what makes it ground truth rather than a restatement. What makes the family *selective* rather than merely multi-step is that retention depends on the event: a keyed event is written to the slot its key addresses and a distractor is written nowhere, so which events survive, and where they land, is a property of the input and not of the position. The memory is 4 slots addressed by `key % width`, appended to the same eight named registers the arithmetic families use and run through the same recurrence, so the state is 12 registers wide.

**Controls on the selective family, at budget 6** — the same controller with a different memory, on the same 50 tasks. The `fixed decay` and `scalar carry` rows are what decide what the architecture is worth.

| condition | selective | what the memory is |
|---|---:|---|
| the agent, memory = the gated SSM state | **1.000** | 4 slots, written by the key of the event |
| fixed decay: the same width, a constant gate | 0.200 | the same slots, every value-carrying event written to every one |
| scalar carry: one Python int | 0.320 | the last keyed value seen; distractors do not move it |
| no memory: the state is wiped before each decision | 0.000 | nothing survives a step |
| random action: tools and arguments uniform | 0.003 | the floor |

The scalar's failure is exact rather than statistical. It holds the last keyed value it saw, so it must be right precisely when the queried key is the one the last `PUT` named, and wrong otherwise: **16 of 50** tasks solved against **16** tasks whose query named the last store, with the predicted set matching the solved set on **50 of 50** tasks. A rate below 1.000 on its own would be consistent with a merely harder task; the correspondence is what says a second register is what was missing.

**State width** — the same tasks read by a memory of 0, 1, 2, 3, 4, 8 slots. The eight named registers are always present, so the state is that much wider again; "slots" is the column that matters. A one-slot state aliases every key onto slot 0, which is why its column is the scalar's.

| memory slots | total state width | agent | scalar | fixed decay | no memory |
|---:|---:|---:|---:|---:|---:|
| 0 | 8 | 0.000 | 0.320 | 0.000 | 0.000 |
| 1 | 9 | 0.320 | 0.320 | 0.200 | 0.000 |
| 2 | 10 | 0.700 | 0.320 | 0.200 | 0.000 |
| 3 | 11 | 0.800 | 0.320 | 0.200 | 0.000 |
| 4 | 12 | 1.000 | 0.320 | 0.200 | 0.000 |
| 8 | 16 | 1.000 | 0.320 | 0.200 | 0.000 |

The family starts to solve at **4 memory slots** on this key space of 4 keys, and the curve between zero and there is the honest capacity statement rather than a cliff: a narrow state does not fail, it aliases, and it is right exactly when no two keys in the stream collide.

**Distractor rate** — the same shape with more of the stream discarded, every point its own task set and its own budget. At zero distractors the fixed-decay state *is* the scalar (0.340 against 0.340, and the experiment asserts the per-task agreement). The distractors are what the gate is for.

| distractor share | agent | scalar | fixed decay | no memory |
|---:|---:|---:|---:|---:|
| 0.000 | 1.000 | 0.340 | 0.340 | 0.000 |
| 0.250 | 1.000 | 0.320 | 0.280 | 0.000 |
| 0.400 | 1.000 | 0.320 | 0.200 | 0.000 |
| 0.571 | 1.000 | 0.360 | 0.140 | 0.000 |
| 0.667 | 1.000 | 0.440 | 0.120 | 0.000 |
| 0.727 | 1.000 | 0.320 | 0.100 | 0.000 |

**The published selective trace** — `note that key 1 holds 40, then ignore 70, then note that key 3 holds 60, then note that key 2 holds 90, then report the value of key 3`, answer `60`, 5 steps, stop reason `finish`, solved `True`. The `slots` column is the whole memory at that step: three keyed values held at once, and the distractor in no slot.

| step | instruction | reads: op / arg / slots | action | result |
|---:|---|---|---|---:|
| 0 | PUT 1 | 8 / 1 / [0, 40, 0, 0] | `add(a=0, b=0)` | 0 |
| 1 | NOISE 70 | 9 / 70 / [0, 40, 0, 0] | `add(a=0, b=0)` | 0 |
| 2 | PUT 3 | 8 / 3 / [0, 40, 0, 60] | `add(a=0, b=0)` | 0 |
| 3 | PUT 2 | 8 / 2 / [0, 40, 90, 60] | `add(a=0, b=0)` | 0 |
| 4 | RECALL 3 | 10 / 3 / [0, 40, 90, 60] | `finish(answer=60)` | 60 |

Three claims this table makes easy, and that are false:

* *"The architecture is what makes the agent work."* On the 5 arithmetic families the scalar carry still reproduces the agent row for row, and the recurrence earns nothing there. On `selective` the gated state reaches **1.000** where the scalar reaches **0.320** and a fixed-decay state of the *same width* reaches **0.200** — so what is doing the work is the input-dependent gate, and the width alone is not enough. That gate is **hand-set**, in `embed`: it is not learned, it is not produced by a projection, and a Python dict in `evaluate_plan` computes the same answers with no model at all. What is established is narrower than the claim: *this* hand-designed gate, at *this* width, on *this* family, does something a scalar and a constant gate do not.

* *"The state is doing something a scalar cannot."* True on this family, and it rests on one design decision: the query key is drawn uniformly from the stored keys, so one register is insufficient on most tasks. It is not a general statement about state-space models — a scalar with a Python dict beside it would tie the agent exactly, and no task here requires more than the four slots the family declares.

* *"This generalises."* It does not. The family is closed and synthetic: a fixed grammar, a shuffled event stream, a bounded key space, and an answer computed by a dict. Nothing is trained — no gradient anywhere in this path — the controller is hand-written branching, and the gate is set by hand. The clearest evidence that the *selectivity* rather than the capacity is what this family measures is the fixed-decay row of the width table: it scores 0.200 at 1 slot and 0.200 at 8 slots, so widening it changes nothing, while the gated state at the same 8 slots reaches 1.000. The part of the model this repository has never learned is the part that decides what to write.
<!-- AGENT:END -->

### What the numbers say, including the unflattering parts

* **Carry-over is necessary, and that is the one thing these controls
  establish.** Wiping the state before each decision takes the five carry
  families from 1.000 to **0.000** at every budget, while the `literal` family —
  whose answer is written in the task text — stays at 1.000. The contrast is the
  measurement: a control that failed everything would only show the loop was
  broken, and one that passed everything would be evidence for nothing.
* **Where the state-space state earns it, and where it does not.** On the five
  arithmetic families `scalar carry` reproduces the agent's row exactly — 1.000,
  the same calls in the same order — so there the recurrent state is a faithful
  and tested way to hold the carried value and is *not* the source of the
  capability. On the `selective` family, where what must be retained depends on
  the input rather than on position, that changes: the gated state reaches
  **1.000** while a scalar carry manages **0.320** and a fixed, input-independent
  decay reaches **0.200**. The fixed-decay control is what makes that readable,
  because it rules out the boring explanation that the state merely has to be
  wide enough. Both halves are published as rows, since a control that flatters
  the increment and one that falsifies it are the same kind of evidence.
* **The budget curve is a ceiling, and mostly arithmetic.** Each family needs a
  known number of decisions (`literal` 1 through `lookup` 5, and `selective` 6 —
  it cannot answer before the whole event stream has arrived), so the staircase is
  what the step counts predict rather than a discovery. It is still worth
  publishing, because it shows the budget is enforced rather than nominal and
  says plainly that a family needing six decisions is unsolvable in five.
* **The first version of the memory had a bug the experiment caught, not the
  tests.** With `A = -50` a register write left 1.7e-21 of the previous value
  behind. That is not a rounding detail: `IFPOS` branches on the *sign* of the
  carry, 1.7e-21 is greater than zero, and 8 of the 50 `branch` tasks — exactly
  the ones arriving at the branch carrying 0 — took the wrong arm and failed. The
  fix was to make a write exact (a decay of `exp(-800)`, which underflows to
  `0.0`) rather than nearly exact, which is why nothing in the readout needs a
  tolerance and why the write-exactness table shows four values of the decay
  rather than asserting the chosen one.
* **Random tool choice is at the floor, and the floor is measured.** Uniform
  tools and arguments solve 0.001 to 0.006 per family, 0.003 overall — non-zero
  because a random agent may call `finish` with an answer that happens to be
  right, which is why the control is bounded rather than asserted to be zero, and
  why the argument span it drew from is recorded in the results file.
* **A one-step agent cannot solve the suite.** At a budget of one decision the
  agent scores **0.200**, and every task it solves is from `literal`. That is the
  task-difficulty control, and it is also the honest bound on the whole block:
  one family is trivial by construction, because it has to be for the no-memory
  contrast to have a surviving arm at all.

The trace in the block above is the mechanism in five rows. The first call uses a
carry of 0 because no observation has been written yet; each later call reads the
previous result out of the state — 3, then 7, then 35 — and the `KEY` step turns
35 into the key paired with it. Every argument after the first is a function of
the state the previous step produced, and the register column is printed next to
the action so that this is checkable rather than asserted.

## Long-context memory: remembering past the working state

Everything above measures memory at two extremes, and both are published. An
attention model's KV cache *grows with the context* — 1,024 MiB after a million
tokens, measured — and a state-space model's state does not: 19,456 bytes at
16,384 tokens and the same 19,456 bytes at 1,048,576. The selective family above
is the state's *internal* memory and it is bounded by its width: four slots hold
four keys, retention is a property of the event, and a wide-enough state answers
the question without anything external. The agent loop's working memory is the
same shape at a smaller size — eight named float64 registers, 64 bytes, holding
the last thing it was told.

So "it remembers everything" and "constant memory" are in direct tension, and
neither extreme is what an agent needs. `src/beyond_attention/memory.py` is the
third option: an **external**, keyed, in-process store the loop writes facts into
(`remember`) and reads them out of (`fetch`), at a **constant** byte cost, with
the state left at its eight registers. The read tool is named `fetch` rather than
`recall` on purpose: `RECALL` is the selective family's instruction and it reads
a *state slot*. Two memories with one name would be a bug waiting to happen.

### What this is, and what it is not

**It is** a working, tested episodic store and the task family that exercises it.
The store is a direct-mapped table — `slot = key % capacity`, each slot holding a
key, a value and a tag — allocated once as `numpy` arrays. The tasks are
generated, their answers are analytic, and `evaluate_plan` computes them with a
plain Python `dict`, deliberately *not* with the store, so a bug in the store
cannot make a failed retrieval look correct. Two instructions and two tools are
new (`REMEMBER`/`FETCH`, `remember`/`fetch`); the policy, the register file, the
loop and the trace are the ones from the section above.

**It is not** associative or semantic memory, and it is not learned. Retrieval is
integer equality on a key the *plan* supplies: nothing decides what is worth
remembering, there is no similarity search, no embedding, and no eviction policy
beyond `key % capacity`. Every number in the block below is generated from
`long-memory.json` by `experiments/render_readme.py`. Three claims that would be
easy to make from those tables, and are all false:

* *"It remembers everything."* The store is bounded and the boundedness is the
  point: eight slots hold eight facts, the ninth write evicts the first, and the
  byte curve is flat because the memory is a fixed array rather than a growing
  cache. What it buys over the working state is not capacity but
  **addressability** — a fact 1,024 steps back is returned exactly as one four
  steps back. The honest crossover is that the fixed state fails at distance
  **1**, this store fails at **9 facts**, and an unbounded store fails at neither
  and pays 17 bytes a write for it.
* *"More memory is strictly better."* The 64x wider state costs 4,096 bytes and
  fails in exactly the same place as the 64-byte one, because nothing addresses
  it; a single 17-byte slot passes every distance measured. Past the point where
  the key space fits, extra slots do help — linearly, 17 bytes a fact — and below
  it they buy nothing at all. This is the same conclusion the selective family
  reaches from the other side, and neither one is a claim about size alone.
* *"The state-space model gives it long memory."* The recurrence is the working
  memory, and it contributes exactly what it contributed before: it carries the
  last value and decodes the current instruction. The long-range result is a
  `numpy` array and a `%`. The control that settles it is the wide state — 64
  times the bytes, the same write rule, the same failure at distance 1 — and the
  trace below shows the call that carries the fact is `remember`, not the
  recurrence. The selective family's gate is a different capability, measured
  above; it is not this one, and it does not extend the distance.

<!-- MEMORY:BEGIN -->
**What each condition carries, and what it solves** — 62 recall tasks at distances 1, 4, 16, 64, 256, 1024 (a plan of `distance + 7` instructions), seed 0. Every cell is **solved/total**, not a rate: the long distances afford fewer tasks, and that is worth seeing. "Bytes carried" is the working state plus the store, measured from the live objects.

| condition | bytes carried | d=1 | d=4 | d=16 | d=64 | d=256 | d=1024 |
|---|---:|---:|---:|---:|---:|---:|---:|
| the store: 8 tagged slots | 200 | 16/16 | 16/16 | 16/16 | 8/8 | 4/4 | 2/2 |
| the store: 1 slot | 81 | 16/16 | 16/16 | 16/16 | 8/8 | 4/4 | 2/2 |
| an unbounded store | 81 | 16/16 | 16/16 | 16/16 | 8/8 | 4/4 | 2/2 |
| the working state alone (no store) | 64 | 0/16 | 0/16 | 0/16 | 0/8 | 0/4 | 0/2 |
| a 64x wider state, same write rule | 4,096 | 0/16 | 0/16 | 0/16 | 0/8 | 0/4 | 0/2 |
| the store, retrieval disabled | 200 | 0/16 | 0/16 | 0/16 | 0/8 | 0/4 | 0/2 |
| random live value (floor) | 200 | 16/16 | 16/16 | 16/16 | 8/8 | 4/4 | 2/2 |

Three rows need reading carefully. The **random live value** floor is not a floor at all in this table: with one fact written there is one live value, so drawing a live value at random *is* the answer. Its discriminating power appears only where several facts are live — the capacity table below — which is why it is reported in both places. The **unbounded store** costs less than the fixed store at every length here, because the single-fact task writes once: the unbounded alternative is cheap precisely when there is nothing to keep. And the **one-slot store** solves every distance in this table at 81 B, from which the only honest reading is that the *width* of the store is not what carries the fact — the key is. What one slot cannot do is hold two facts, and that is the capacity table.

**Bytes carried against length** — the store's figure is measured from a live `MemoryStore`, the unbounded one after that many writes (which is why it is a staircase: a growing `numpy` array doubles), and the transcript is arithmetic — one int64 per event, the least a replay needs. The run object retains every step whether or not anyone asks it to.

| episode length | working state | store (8 slots) | untagged store | unbounded store | its arithmetic floor | transcript (1 int64/event) | model KV cache |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 136 | 72 | 17 | 17 | 8 | 1,024 |
| 8 | 64 | 136 | 72 | 136 | 136 | 64 | 8,192 |
| 9 | 64 | 136 | 72 | 272 | 153 | 72 | 9,216 |
| 17 | 64 | 136 | 72 | 544 | 289 | 136 | 17,408 |
| 64 | 64 | 136 | 72 | 1,088 | 1,088 | 512 | 65,536 |
| 100 | 64 | 136 | 72 | 2,176 | 1,700 | 800 | 102,400 |
| 256 | 64 | 136 | 72 | 4,352 | 4,352 | 2,048 | 262,144 |
| 1,024 | 64 | 136 | 72 | 17,408 | 17,408 | 8,192 | 1,048,576 |
| 16,384 | 64 | 136 | 72 | 278,528 | 278,528 | 131,072 | 16,777,216 |
| 1,048,576 | 64 | 136 | 72 | 17,825,792 | 17,825,792 | 8,388,608 | 1,073,741,824 |

For scale: the model's own state is **19,456 B at every length** (`stream-memory.json`), and its attention cache is **1,073,741,824 B at 1,048,576 tokens** (`long-context.json`) — 1,024 B a token, quoted from the streaming measurements rather than recomputed here.

### The honest crossover

* **The working state starts failing at distance 1** — and so does the 4,096-byte state at exactly the same distance, because the write rule puts every observation in the same register. It is 64 B and it holds the last observation; nothing about the *width* of a state that is not addressed by key changes that.
* **A single 17-byte slot does the same job at every distance measured**, so this trade is not about the store's size: 81 B addressed by key beats 4,096 B that is not. The size starts to matter only when more than one fact is live, which is the capacity table below.
* **The store does not fail at any distance measured** (to 1024) — it is 136 B and constant. Its failure mode is capacity, not distance: 8 facts fit in the eight slots and the 9th evicts the first. Every extra fact costs 17 B, so holding F facts exactly costs 17×F bytes.
* **What the unbounded alternative costs.** The unbounded store is 17 B a write and passes the fixed store's 136 B at write 9; a full transcript at 8 B an event passes it at event 18. Below those lengths, keeping everything is *cheaper* than a bounded store — and above them the bounded store is exact only up to its capacity. At the model's scale a single attention token of KV cache is 1,024 B, so the agent's entire store is 1 token of it.

### Capacity: what eight slots hold, and what a cheaper tag costs

Keys are `1..F` against `key % 8`, so the collision is **designed** rather than drawn from a random key space: this is what a bounded store does at a known load, not the birthday-paradox rate a wider key space would give. The oldest fact is queried first because it is the one a direct-mapped store evicts first.

| condition | bytes carried | F=1 | F=2 | F=4 | F=8 | F=9 | F=16 | F=32 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| the store: 8 tagged slots | 200 | 16/16 | 16/16 | 16/16 | 16/16 | 8/16 | 8/16 | 8/16 |
| the store: 1 slot | 81 | 16/16 | 8/16 | 8/16 | 8/16 | 8/16 | 8/16 | 8/16 |
| an unbounded store | 608 | 16/16 | 16/16 | 16/16 | 16/16 | 16/16 | 16/16 | 16/16 |
| the working state alone (no store) | 64 | 0/16 | 0/16 | 0/16 | 0/16 | 0/16 | 0/16 | 0/16 |
| a 64x wider state, same write rule | 4,096 | 0/16 | 0/16 | 0/16 | 0/16 | 0/16 | 0/16 | 0/16 |
| the store, retrieval disabled | 200 | 0/16 | 0/16 | 0/16 | 0/16 | 0/16 | 0/16 | 0/16 |
| random live value (floor) | 200 | 16/16 | 9/16 | 2/16 | 1/16 | 1/16 | 0/16 | 0/16 |
| the store, untagged | 136 | 16/16 | 16/16 | 16/16 | 16/16 | 8/16 | 8/16 | 8/16 |

**Retrieval precision at F=32** — what the loop did with the answer. "Wrong fact" is a value stored under a *different* key: the store handed over another fact confidently, which is worse than a miss, and it is the failure the tag exists to prevent. Precision is over answered runs, so refusing to answer cannot raise it.

| condition | correct | wrong fact | stale | no answer | precision |
|---|---:|---:|---:|---:|---:|
| the store: 8 tagged slots | 8 | 0 | 0 | 0 | 0.500 |
| the store: 1 slot | 8 | 0 | 0 | 0 | 0.500 |
| an unbounded store | 16 | 0 | 0 | 0 | 1.000 |
| the working state alone (no store) | 0 | 0 | 0 | 0 | 0.000 |
| a 64x wider state, same write rule | 0 | 0 | 0 | 0 | 0.000 |
| the store, retrieval disabled | 0 | 0 | 0 | 0 | 0.000 |
| random live value (floor) | 0 | 16 | 0 | 0 | 0.000 |
| the store, untagged | 8 | 8 | 0 | 0 | 0.500 |

At F=9 the tagged store and the untagged one solve the same number (8/16 against 8/16) and fail differently. The tagged store **misses**: its retrievals find the slot occupied by another key, report a miss, and the loop answers its default zero — 8 of 16 runs end in a wrong number rather than another fact's. The untagged store returns **another fact's value** in 8 of 16, which is the same solve rate and a worse failure — at 72 B instead of 136 B. The tag costs 64 bytes in total, and it is the difference between a miss and a confident wrong answer.

### The stale-fact control

A fact written, overwritten, and queried — 16 tasks over distances 1, 4, 16, 64. The answer is the second value; a store that returns the first hands the loop something that *was* true, which a caller cannot tell from something that is.

| condition | solved | returned the new value | returned the superseded value | no answer |
|---|---:|---:|---:|---:|
| the store: 8 tagged slots | 16/16 | 16 | 0 | 0 |
| the store: 1 slot | 16/16 | 16 | 0 | 0 |
| an unbounded store | 16/16 | 16 | 0 | 0 |
| the working state alone (no store) | 0/16 | 0 | 0 | 0 |
| a 64x wider state, same write rule | 0/16 | 0 | 0 | 0 |
| the store, retrieval disabled | 0/16 | 0 | 0 | 0 |
| random live value (floor) | 16/16 | 16 | 0 | 0 |
| first-write-wins store | 0/16 | 0 | 16 | 0 |

**The published trace** — `add 3, then multiply the result by 4, then subtract 5, then report the key whose value equals the result, then remember the result under key 4, then add 5, then subtract 2, then fetch the value under key 4, then report the result`, answer `9`, table `[[2, 11], [5, 99], [9, 7]]`, fact stored under key `4`, 9 steps. "Reads" is the state the policy acted on, before the call.

| step | instruction | reads: op / arg / carry | action | result |
|---:|---|---|---|---:|
| 0 | ADD 3 | 3 / 3 / 0 | `add(a=0, b=3)` | 3 |
| 1 | MUL 4 | 4 / 4 / 3 | `mul(a=3, b=4)` | 12 |
| 2 | SUB 5 | 5 / 5 / 12 | `sub(a=12, b=5)` | 7 |
| 3 | KEY | 6 / 0 / 7 | `lookup(value=7)` | 9 |
| 4 | REMEMBER 4 | 11 / 4 / 9 | `remember(key=4, value=9)` | 9 |
| 5 | ADD 5 | 3 / 5 / 9 | `add(a=9, b=5)` | 14 |
| 6 | SUB 2 | 5 / 2 / 14 | `sub(a=14, b=2)` | 12 |
| 7 | FETCH 4 | 12 / 4 / 12 | `fetch(key=4)` | 9 |
| 8 | RET | 1 / 0 / 9 | `finish(answer=9)` | 9 |

The same task under the controls: `working_state` answers 0 (wrong_other). The working state answers with the last thing it was told — a wrong value, not an error — which is what makes the control a measurement of memory rather than of a broken loop.
<!-- MEMORY:END -->

### What the numbers say, including the unflattering parts

* **The distance is free and the capacity is not.** The store is 136 bytes at
  every distance from 1 to 1,024 and solves every task; the working state is 64
  bytes and solves none of them, at any distance, because the fact is behind it
  rather than in it. What ends the store's run is not length but the ninth live
  fact: eight slots hold eight facts exactly, and the ninth write takes the
  first one's slot.
* **The tag earns its 64 bytes, and the measurement is the failure it prevents.**
  At one fact past capacity the tagged store and the cheaper untagged one solve
  the *same number* and fail differently: the tagged store reports a miss, and
  the untagged store returns another fact's value. "It remembered" is only worth
  reporting next to which of those two happened, which is why the precision table
  counts correct, wrong-fact and no-answer separately.
* **Three controls that could have flattered the store are published failing.**
  The store with retrieval disabled pays every write and answers nothing; the
  working state with no store at all fails in exactly the same places; and the
  64x wider state fails identically to the 64-byte one. The last of those is the
  one that decides what the capability *is*: it is not the bytes, because 4,096
  of them buy nothing without a key, and it is not the recurrence, because the
  same recurrence in a wider state does no better.
* **The random-retrieval floor is only a floor where several facts are live.**
  With one fact written, drawing a live value at random *is* the answer, so the
  floor sits at 1.000 in the distance table and in the stale table and means
  nothing there. In the capacity table, where several facts are live, it falls
  from 16/16 at one fact to 1/16 at eight and 0/16 beyond — and the first version
  of it was silently reseeded once for the whole sweep, so every task drew the
  same slot and the "floor" was a constant that solved the newest query every
  time. A floor that is really a constant is worse than no floor.
* **The working state fails with a wrong number, not an exception.** With no
  store, `remember` is a typed `no_store` error and `fetch` is a miss; a failed
  observation writes the carry to zero (0 is a legitimate value, and the `miss`
  register is what distinguishes them), so the loop terminates with a valid,
  wrong answer. A control that crashed would show only that the loop was broken.
* **Two generator bugs were caught by the controls, not by reading the code.**
  Carries walked past the tools' argument bound, so the loop hit `out_of_range`
  in the middle of a plan and the measurement was of the tool schema rather than
  of memory; and a distractor chain that tried to *reject* out-of-range steps
  never terminated for a thousand-step walk, because a multiplicative step
  always exceeds the bound eventually. The fix for the first is a look-ahead
  bound in the generator and an invariant test that every step of every generated
  plan succeeds; the fix for the second is a sound upper bound carried along the
  chain rather than a redraw.

### Which of these numbers are not about a real agent

The store is a data structure and the family is synthetic. Specifically:

* **One retrieval design.** Direct-mapped, tagged, last-write-wins, with the key
  and the value both bounded integers. There is no comparison against a
  different eviction policy, a hash with collisions resolved by probing, a
  keyed-by-string store, or any approximate retrieval. The capacity curve is the
  curve of `key % 8`, not of "bounded memory" in general.
* **The plan decides, and nothing else does.** When to write and which key to
  write under come from the instruction stream, so this measures whether a store
  can be *used* by this loop, not whether an agent can decide what is worth
  keeping. Relevance, salience and forgetting are absent rather than handled.
* **The keys are small integers and the values are small integers.** The byte
  accounting is 8 + 8 + 1 bytes a slot because that is what the arrays are; a
  store keyed by text or holding vectors costs something else entirely, and
  nothing here says by how much.
* **"Unbounded" means a growing `numpy` array in this process**, not a database
  or a vector index. Its cost is allocation and over-allocation, and it says
  nothing about the query latency, index build time or durability a real
  long-term store would need.
* **The distances are steps of one synthetic plan.** A distance of 1,024 is
  1,024 instructions, not 1,024 turns of a real task, hours of wall clock, or a
  million tokens of history. The model-scale rows are quoted from the streaming
  measurements to give the bytes a scale, not because the two units are
  interchangeable.
* **The store and the selective memory are different mechanisms and only one of
  them is in the state.** The selective family's retention is a gate inside
  `embed`, measured against a constant-gate control; this store is an array
  outside the state, measured against a no-store control. Both are published,
  and neither result transfers to the other.

## Limitations, stated rather than discovered later

* **This is not a language model.** The task is synthetic, the vocabulary is 33
  tokens, and nothing here speaks to language-modelling quality.
* **Two layers and `d_model = 64`.** Far too small to separate "cannot" from
  "needs more capacity", so a low number means *this* model at *this* size did
  not learn it.
* **One seed in the sweep.** The spread is therefore unreported; the runner
  supports `--seeds` and the table grows a ± column when it is used.
* **One training task.** MQAR tests associative recall. Length extrapolation is
  now measured on it (see above) and neither architecture extrapolates — but
  that is one synthetic task at one model size, and the reference columns show
  the decay is mostly task difficulty rather than a generalisation failure. It
  is not a general statement about either architecture.
* **The extrapolation reference is one seed.** The extrapolated rows are three
  seeds with a spread column; the trained-at-that-length reference is a single
  seed, so a difference between them smaller than the spread is not a
  difference. At 16 pairs the two are within it.
* **Training the reference at 16 pairs is expensive on CPU.** The SSM's inner
  scan is a Python loop, and training at longer sequences costs real time — see
  "Which inner scan to use" for the measured scaling. That cost is why the
  reference is reported at the two lengths that bound the interpretation rather
  than at all four.
* **Streaming inference is measured, but on CPU with an unoptimised loop**, so
  its numbers are about state size, not speed. The state figures are now
  measured resident memory rather than arithmetic (see above), but still CPU
  resident memory.
* **CPU only.** No CUDA was available, so nothing reflects GPU throughput, where
  the memory-bandwidth contract is entirely different and where selective-scan
  kernels are designed to run. The measured memory figures are for CPU
  allocations.
* **The scaling numbers come from `scaling-final.json` (fused baseline) and
  `scaling-mask.json` (mask baseline)**, not from the sweep run, and both were
  taken after the memory probe was fixed. The earlier `scaling.json`,
  `scaling-fused.json`, `scaling-sdpa.json` and `scaling-crossover.json` are kept
  in the repository as the record of the two broken measurements described above
  — `scaling-sdpa.json` is the run that was labelled fused and measured mask.
* **Run-to-run variance is a few percent.** The SSM at 8,192 tokens measured
  8.106 s in one run and 7.835 s in another with identical settings, so
  differences below ~5% in these tables are noise, not signal.
* **The voice module is validated only on signals this repository synthesised.**
  There is no real speech anywhere in it: no corpus and no listener labels. The
  classifier now exists (`emotion.py`), and that is what makes this limitation
  sharper rather than weaker: its labels are the synthesiser's, so it learns our
  acoustic model of five emotions, not a listener's. Every accuracy in the voice
  and classifier blocks is a check that the descriptors recover the axes the
  generator was given. It says nothing about whether real prosody varies along
  the same axes, and it is not evidence that emotion is decodable from a voice.
* **The trained classifier does not generalise to a held-out condition.** Train
  on four conditions and point it at the fifth and it does not abstain: it puts
  50-100% of the unseen condition onto one training class, at a mean dominant
  fraction of 0.835 against 0.200 for any one class, and is confident about it.
  The untrained nearest-centroid rule names the same class on three of the five,
  so the collapse is a property of the feature space. This is the honest answer
  to "did it learn emotion or my generator", and it is the generator.
* **The five-class task is trivially separable, so the headline accuracy is not
  a result.** The conditions were built to differ along the axes the features
  measure, and they do — which is why the ablation table is flat rather than
  informative. Removing the entire F0 level/contour family, or everything
  pitch-derived, leaves the held-out accuracy at 1.000, because six acoustic
  axes were varied at once and rhythm, voice quality, energy dynamics and
  spectral shape each carry the task alone. No single feature family is necessary, and the
  per-feature ablation shows a drop of exactly 0.000 for all 26.
* **There is no held-out human data anywhere in the classifier path.** No affect
  corpus is present, none is downloaded, and no network access is used. "This
  transfers to real speech" is untested rather than disproved: the measurements
  cannot speak to it in either direction.
* **`tremor_rate_hz` is unreliable on breathy phonation.** It recovers the
  modulation rate to within 0.1 Hz on excited, angry, afraid and calm, and
  returns 1.60 Hz against a 6.5 Hz setting on crying, because the pitch track
  under low HNR is noisy enough that low-frequency estimation error outweighs
  the tremor. It is used by the classifier and documented as valid only where
  the contour is periodic.
* **The F0 estimator's accuracy is bounded by its window, and its advertised
  range is not its working range.** With the default 25 ms window and 16 kHz
  sampling, a 10 Hz sweep from 100 Hz to 380 Hz stays within 1.6% (worst 1.57%,
  at 110 Hz). At 70 Hz the error is 3.2% with only 40% of frames voiced, and at
  60 Hz the search locks onto a short-lag artifact and reports about 400 Hz —
  a 570% error. The 60–400 Hz band in the module is the *search* range, not a
  promise: the tests assert a 2% tolerance at 150, 220 and 300 Hz, and
  low-pitched audio needs a longer `window_ms`.
* **Jitter and speaking rate are frame-level proxies, not phonetics
  measurements.** Jitter is a successive-F0 difference between adjacent voiced
  frames and its resolution is bounded below by the 10 ms hop; the rate proxy
  counts voiced segments and cannot see unvoiced consonants at all. Both are
  named as proxies in every table above.
* **The README's generated regions were not all generated, and that is fixed.**
  The results block contained `### Does either model read longer than it
  trained?` written by hand, so re-running the documented results command
  *deleted* it: the renderer's guard checked for the markers its own tables
  produce, and a section it cannot produce is not one it notices losing.
  `experiments/render_readme.py` now renders that section from
  `length-extrapolation.json`, and the results render **fails** if the file is
  not supplied rather than writing a block without it.
  `tests/test_render_readme.py` compares every `BEGIN`/`END` region of this
  README with the renderer's output byte for byte, so hand-written content
  inside a generated region now fails a test instead of surviving until the next
  run deletes it. Everything outside those regions — the interpretation, the
  limitations, the corrections — is still written by hand, and is meant to be.
* **The agent loop is a closed grammar, not a language interface.** Its "task
  text" is a fixed instruction grammar delivered as structured events, and the
  controller is written against that grammar. There is no tokenizer and no
  natural-language input anywhere in the path, so no test in this repository
  could detect a failure to understand a sentence — nothing accepts one.
* **The tools are seven pure functions that cannot touch anything.** No network,
  no filesystem, no clock, and one call per decision, so the loop cannot compose
  or discover tools. "Open-ended tool use" is not a claim this harness can
  support, and the interesting failure modes of real tool use — irreversible
  actions, partial failure, retries — are absent rather than handled.
* **The suite is six generated plan shapes over bounded integers, plus the
  three recall shapes.** The six come from `task_suite` and the recall families
  from `recall_suite`, `capacity_suite` and `stale_suite`; their constraints (no
  answer is 0, table values are unique, operands are 2-9, every intermediate
  value inside the tools' argument bound) are what make the controls
  interpretable. Nothing here speaks to a task outside those shapes; the 1.000
  solve rate is a statement about a controller matching the shapes it was
  written for, and the README says so above the table as well as here.
* **The selective result is a hand-set gate on a closed, four-key family.**
  Which slot an event is written to is computed in `embed` from the event's own
  key; nothing learns it, and there is no comparison against a model that learned
  a gate. The key space is four, the query key is drawn uniformly from the stored
  keys — which is what makes one register insufficient — and a Python dict in
  `evaluate_plan` computes every answer without the memory at all. What the width
  and distractor sweeps establish is that *this* gate, once the state is wide
  enough, retains what a scalar and a constant decay cannot; they do not
  establish that a state-space model is what makes the loop work, and the five
  arithmetic families still show a scalar tying it exactly.

## References

* Gu & Dao, *Mamba: Linear-Time Sequence Modeling with Selective State Spaces* —
  [arXiv:2312.00752](https://arxiv.org/abs/2312.00752). The S6 recurrence
  implemented here.
* Dao & Gu, *Transformers are SSMs: Generalized Models and Efficient Algorithms
  Through Structured State Space Duality* —
  [arXiv:2405.21060](https://arxiv.org/abs/2405.21060). The associative-scan view
  used by `selective_scan_associative`.
* Arora et al., *Zoology: Measuring and Improving Recall in Efficient Language
  Models* — [arXiv:2312.04927](https://arxiv.org/abs/2312.04927). The
  associative-recall task, and the source of the expectation that efficient
  architectures trade recall for speed — an expectation the control above shows
  is about *training budget* at this scale.
* Olsson et al., *In-context Learning and Induction Heads* —
  [arXiv:2209.11895](https://arxiv.org/abs/2209.11895). Why the Transformer needs
  the extra steps.

## License

MIT.
