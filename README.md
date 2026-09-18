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

The "unseen size" column above is not a clean measure of that, and it took
building the experiment below to see why. `mqar_batch` sizes its vocabulary from
the pair count, so a model trained at 2 pairs meets keys 1-8 and values 9-16,
while a 16-pair batch hands it keys 1-16 and values 17-32 — token ids it has
never seen, in roles it has never seen them in. Every `0.000` in that column is
an evaluation at a *longer* length than training, which is precisely the case
the column was meant to measure. Those zeros are a vocabulary mismatch, not a
failure to generalise.

Holding the key space fixed so the vocabulary is identical at every length
changes the picture completely. Both models are trained at **2 pairs (6
tokens)** — the longest length at which *both* solve the task outright — then
evaluated with frozen weights at 4, 8 and 16 pairs (10, 18 and 34 tokens, up to
**5.7x**):

| pairs | tokens | x train | attention | ssm | chance | a model trained at that length: attention | ssm |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 6 | 1.0x | **1.000** | **1.000** | 1/16 | 1.000 | 1.000 |
| 4 | 10 | 1.7x | 0.793 ±0.007 | 0.541 ±0.007 | 1/16 | — | — |
| 8 | 18 | 3.0x | 0.459 ±0.005 | 0.297 ±0.010 | 1/16 | — | — |
| 16 | 34 | 5.7x | 0.240 ±0.009 | 0.176 ±0.014 | 1/16 | 0.205 | 0.197 |

Three seeds, spread shown, nothing retrained between rows.

What this says, including the parts that are unflattering:

* **Neither architecture extrapolates here.** Both are perfect at the length
  they trained on and both decay monotonically as it grows.
* **Attention decays more slowly than the SSM** at every step — 0.793 against
  0.541 at 1.7x, 0.240 against 0.176 at 5.7x. On this task the state-space model
  is the weaker of the two past its training length, which is the opposite of
  what the architecture's reputation would predict.
* **The decay is not an extrapolation failure.** At 16 pairs the extrapolated
  models score about what a model *trained from scratch at 16 pairs* reaches
  (0.240 against 0.205 for attention; 0.176 against 0.197 for the SSM). Reading
  down that column is reading how hard MQAR is at that length for a 2-layer,
  `d_model = 64` model — not how badly the model generalises. Without the
  reference columns this curve looks like a generalisation result and is not one.
* The SSM does fit *longer training lengths* better than attention: in the
  table above it reaches 1.000 at 4 and 8 pairs where attention reaches 0.413
  and 0.312. So "fits long sequences when trained on them" and "generalises to
  longer ones when trained short" are separate properties, and the two
  architectures sit on opposite sides of them.

One training length, one task, one model size. This measures MQAR at 2 pairs.

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

python -m pytest tests/ -q                     # 103 correctness tests

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
    --control control-attention.json --readme README.md
```

`experiments/run.py --help` lists the knobs; `--steps`, `--seeds`, `--pairs`,
`--queries` and `--blocks` are the ones that cost time.

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
