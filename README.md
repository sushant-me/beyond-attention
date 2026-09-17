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
* *"It is faster because it is linear."* Asymptotically yes; **as implemented
  here, no.** This implementation is 20–40× slower and ~8× hungrier per token
  than the Transformer at every length measured. The reason is not the
  architecture, and the section below explains it.
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

Activation memory is the growth in peak RSS over the post-import baseline, one fresh process per point, forward and backward pass.

| sequence length | model | forward+backward (s) | peak activation memory (MB) | MB per token |
|---|---|---|---|---|
| 128 | attention | 0.009 | 16.3 | 0.0318 |
| 128 | ssm | 0.265 | 112.7 | 0.2201 |
| 256 | attention | 0.014 | 24.4 | 0.0238 |
| 256 | ssm | 0.750 | 211.7 | 0.2067 |
| 512 | attention | 0.030 | 40.1 | 0.0196 |
| 512 | ssm | 1.902 | 409.5 | 0.1999 |
| 1024 | attention | 0.073 | 77.2 | 0.0189 |
| 1024 | ssm | 4.654 | 625.6 | 0.1527 |
| 2048 | attention | 0.241 | 159.8 | 0.0195 |
| 2048 | ssm | 9.110 | 1238.7 | 0.1512 |

```
parameter check at the sweep vocabulary: ssm=67,584 vs attention=67,968
main sweep wall time: 1814.6s
control wall time: 271.1s
```
<!-- RESULTS:END -->

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

## Why the linear-time model is not faster here

Attention is O(L²) in sequence length and this is O(L), so the textbook
expectation is that the state-space model pulls ahead at long sequences.
Measured on this implementation, the opposite happens at every length, and the
reasons are specific:

1. **The scan materialises its terms.** The chunked path builds `a` and `b` at
   shape `(B, L, D, N)` — with `d_inner = 128` and `N = 16` that is 2048 floats
   per token per layer before autograd bookkeeping, and autograd saves several
   such tensors for the backward pass. A production kernel never materialises
   them; it streams through the chunks holding only the running state. The memory
   advantage of a state-space model is a property of a *fused kernel*, not of the
   recurrence.
2. **The chunk loop is Python.** That is what makes it readable and what makes it
   slow.
3. **The baseline is already fused.** PyTorch's `nn.MultiheadAttention`
   dispatches to `scaled_dot_product_attention`, which is memory-efficient and
   does not materialise the `L × L` score matrix. So the usual "attention costs
   quadratic memory" comparison does not hold against this baseline: measured
   over 128 → 2048 tokens, attention's peak memory grows 16.3 → 159.8 MB, which
   is **linear**, while the SSM's grows 112.7 → 1238.7 MB. At 2048 tokens the
   state-space model uses 7.8× the memory and takes 38× the time.

So both architectural advantages are absent, for two different reasons: the SSM
is missing a kernel, and the attention it is being compared against already has
one. Closing the gap means writing a fused selective-scan implementation
(`mamba-ssm`'s CUDA kernel, or a chunked C++/TorchScript version) — not tuning
the model. Until that exists, any claim that this code is faster than attention
would be false, and the asymptotic argument alone would be misleading.

## Reproducing this

```bash
uv venv && uv pip install --index-url https://download.pytorch.org/whl/cpu torch
uv pip install -e . pytest

python -m pytest tests/ -q                     # 56 correctness tests

python experiments/run.py --pairs 2 4 8 16 --steps 3000 --seeds 0 \
    --out results.json                         # main sweep   (~30 min, CPU)
python experiments/run.py --skip-sweep --out scaling.json       # (~30 s)
python experiments/run.py --pairs 8 16 --steps 20000 --blocks attention \
    --out control-attention.json               # the control (~5 min)

python experiments/render_readme.py --mqar results.json \
    --scaling scaling.json --control control-attention.json --readme README.md
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
* **The scaling numbers come from `scaling.json`, not the sweep run**, because
  the first version of the memory measurement took its baseline *after* building
  the model and so reported the same constant for every configuration. That was
  a measurement that could not fail; it is fixed, and the fix is why the numbers
  above are the ones shown.

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
