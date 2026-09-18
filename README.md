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

python -m pytest tests/ -q                     # 56 correctness tests

python experiments/run.py --pairs 2 4 8 16 --steps 3000 --seeds 0 \
    --out results.json                         # main sweep    (~30 min, CPU)
python experiments/run.py --pairs 8 16 --steps 20000 --blocks attention \
    --out control-attention.json               # the control   (~5 min)

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

## Limitations, stated rather than discovered later

* **This is not a language model.** The task is synthetic, the vocabulary is 33
  tokens, and nothing here speaks to language-modelling quality.
* **Two layers and `d_model = 64`.** Far too small to separate "cannot" from
  "needs more capacity", so a low number means *this* model at *this* size did
  not learn it.
* **One seed in the sweep.** The spread is therefore unreported; the runner
  supports `--seeds` and the table grows a ± column when it is used.
* **One task.** MQAR tests associative recall and says nothing about length
  extrapolation, streaming inference, or throughput on hardware with a real
  scan kernel.
* **CPU only.** No CUDA was available, so nothing reflects GPU throughput, where
  the memory-bandwidth contract is entirely different and where selective-scan
  kernels are designed to run.
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
