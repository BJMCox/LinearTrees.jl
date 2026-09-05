# PILOT reference fixtures

Eight pinned datasets fit against the reference PILOT implementation, used by
`test/pilot_reference.jl` to check `fit_tree` against it. Python is
provenance only: it is not a runtime or test dependency of LinearTrees.jl.
The JSON files here are committed; the test suite reads only those.

## Reference

Clone: the `pilot` package from commit `ea769a0d55dd2babf4c57d31094758d7879f29ff`.

At that commit `pilot/__init__.py` does `from .cpilot import PILOT as CPILOT`,
but `pilot/cpilot.py` does not exist in the clone, so `import pilot` raises
`ModuleNotFoundError`. `make_fixtures.py` works around this by registering a
stub `pilot` package in `sys.modules` with the correct `__path__` and then
importing `pilot.Pilot` / `pilot.Tree` directly, skipping `__init__.py`.
`pilot/Tree.py` also imports `networkx` and `matplotlib.pyplot` at module
level, used only by an unused visualization helper (`construct_graph` /
`visualize_tree`); the script stubs those modules out if they are not
installed.

Versions used to generate the committed fixtures:

| | |
|---|---|
| Python | 3.12.14 |
| NumPy | 2.5.2 |
| Numba | 0.67.0 |
| scikit-learn | 1.9.0 |
| pandas | 3.0.5 |

## Constructor mapping

The brief that spawned this task named the reference's sample-count knob
`min_sample_fit`; that name does not exist in `pilot/Pilot.py`. The actual
`PILOT.__init__` parameter is `min_sample_split` ("the minimal number of
samples required to split an internal node"), which is what `fit_tree`'s
`min_fit` maps to. `max_depth`, `min_sample_leaf` -> `min_leaf`, and
`truncation_factor` match by name.

## Node format

`model.model_tree` is a `pilot.Tree.tree` object. Walking `.left`/`.right`
until `.node == "END"` gives, per split node:

- `.node`: `"con"`, `"lin"`, `"pcon"`, `"blin"`, `"plin"`, or `"pconc"`
  (categorical; unused here, no categorical columns in these fixtures) --
  see "Reference label defect" below, `pcon`/`blin` are swapped.
- `.pivot`: `(feature, value)`, feature 0-based.
- `.lm_l`, `.lm_r`: `[coef, intercept]` for the left/right child's linear
  piece (for `"con"`/`"lin"`, all information is in `.lm_l`; `.lm_r` is a
  `[0, 0]` placeholder).
- `.interval`: `[min, max]` of the split feature over the data reaching this
  node (used to clip predictions at that feature's training range).

Each JSON fixture's `nodes` array is this walk flattened (END leaves
dropped, label defect corrected -- see below), one entry per split node
with `kind`, `feature` (0-based), `threshold`, `lm_l`, `lm_r`, `range`.

## Reference label defect

`Pilot.py` line 21: `REGRESSION_NODES = ["con", "lin", "pcon", "blin",
"plin"]`. In `best_split`, the candidate coefficient/intercept rows are
filled in the order `blin` (row 0), `pcon` (row 1), `plin` (row 2) --
see the `"blin" in regression_nodes"` block (row 0) followed by the
`"pcon" in regression_nodes"` block (row 1) inside the pivot loop -- and
the winning row is labelled `regression_nodes[add_index + index_min]`
with `add_index = 2`. So row 0 (really `blin`) is labelled `"pcon"`, and
row 1 (really `pcon`) is labelled `"blin"`. `build_tree`, `predict`, and
`Tree._get_child_data` all key off `.node` identically for `pcon` and
`blin` (same split rule, same child partitioning), so the swap changes
nothing about the reference's fit or its predictions -- it is a label
defect only.

Evidence: before correcting for this, `hinge`'s root was labelled `"pcon"`
but carried `lm_r = [2.9916, -1.4855]` (a slope -- `pcon`'s right side
should be a plain intercept), and `step`'s root was labelled `"blin"`
with slope `0` on both sides (a `blin` fit degenerating to two flat
lines, i.e. actually a `pcon`).

`make_fixtures.py`'s `walk` corrects this with `LABEL_FIX = {"pcon":
"blin", "blin": "pcon"}` applied to the recorded `kind`, while still
passing the reference's own (mislabelled) node to `Tree._get_child_data`
for partitioning, since that function's behavior does not depend on which
of the two labels is attached. The dataset table below lists the
corrected (true) kinds.

## Feature indices

Reference: 0-based. Ours: 1-based `Int32`. The test adds 1 to `n.feature`
when comparing.

## Split-point convention

Both sides now agree: the split point is the largest observed value of
the split feature in the left partition (`x <= threshold` goes left), and
for `blin` this same point is the knot of the continuous hinge basis.
`src/scan.jl` originally used the midpoint between two consecutive
distinct sorted values instead; that was an implementation choice with
no basis in the PILOT spec (which only ever compares `x <= threshold` and
requires the reference's exact threshold to match), and it moved `blin`'s
knot off the true split point, changing its RSS and, for two of these
five fixtures, which kind won. Fixed in `src/scan.jl` (commit
"Split at the largest left value, matching PILOT"), so fixture
`threshold` values compare directly against `n.threshold` at `1e-8`, no
mapping needed.

For `"con"` and `"lin"` nodes, PILOT's `pivot` second element is left
over from whatever the `best_pivot` local carried from a previous
feature's loop iteration in `best_split` -- it is never assigned by the
`"con"`/`"lin"` branches themselves, so it is meaningless. This matches
LinearTrees.jl, where `LIN` nodes always store `threshold = NaN`. The
test does not compare thresholds for `"lin"` (and excludes `"con"` from
the split-node list entirely, since `CON` is always a leaf on both sides
-- see below).

## Tie rule

Ours: lowest feature index first, then lowest threshold. `src/fit.jl:257`
scans features in ascending order and only replaces `best` on a strictly
lower score (`c.score < best.score`), so the first (lowest-index) feature
keeps a tie; the parallel reduction at `src/fit.jl:282` restores that same
lowest-feature-index tie-break explicitly, so the result does not depend
on task completion order. Within a feature, `src/scan.jl`'s left-to-right
sweep keeps the same way: the first (lowest) threshold that achieves the
best score is never displaced by a later, merely-equal one.

The reference does not follow a fixed rule: `Pilot.py:209`'s
`random_sample` draws the cross-feature evaluation order from
`np.random.choice`, seeded by the legacy global NumPy RNG, so which
feature wins a cross-feature tie is RNG state, not a pinned rule.

No cross-feature tie occurs among the split nodes these fixtures compare.
Checked by refitting all five datasets with the legacy RNG seeded three
different ways (the fixture's own seed, and two large perturbations of
it) and confirming the recorded `lin`/`pcon`/`blin`/`plin` nodes and
predictions are bit-identical across all three -- if any two features had
tied at any split, permuting the evaluation order would have picked a
different one. The one thing that *does* move is which feature gets
attached to a `con` leaf: `con`'s loss never depends on the feature (it
only uses `y`), so every feature ties for it by construction, and the
recorded `con.pivot` feature is whichever one the RNG order visited first.
This is harmless here since the test excludes `con` from the comparison
entirely (see "CON nodes are leaves" below).

## Scope of the parity claim

The eight fixtures establish parity for continuous splits (plus, for
`categorical`, one categorical split), away from a few boundaries that
the reference and LinearTrees.jl handle differently. None of these bind
on any of the eight fixtures (smallest node size across all of them is
31 rows, `deep`'s, well clear of `min_fit = 10`); they would need to be
accounted for before extending the parity claim to other data.

- **A node with exactly `min_fit` rows.** Ours splits it: `src/fit.jl:322`
  stops only when `nw < st.min_fit` (strict). The reference stops it:
  `Pilot.py:638`'s `stop_criterion` allows further splitting only when
  `y.shape[0] > self.min_sample_split`, i.e. it also stops at exactly
  `min_sample_split` rows. A node landing on exactly 10 rows (`min_fit =
  10` here) would disagree on whether to split at all.
- **Numeric features with fewer than 5 unique values.** The reference
  allocates its `coef`/`intercept` arrays once per `best_split` call and
  only overwrites a row when `blin`/`plin` are eligible at a pivot; a
  feature with under 5 unique values never becomes eligible, so its row
  keeps stale values from whichever feature was scored just before it and
  can still be scored (and win) against the current feature's moments.
  Exact parity is not defined on such a feature. Every non-categorical
  column across all eight fixtures is continuous `Uniform`, so this
  never triggers.
- **Singularity guards and RSS floors differ in kind.** The reference
  rejects a `blin` fit on an absolute `det(XtX) > 0.001`; ours uses a
  scale-invariant guard (`1e-12 * sxx * sw * suu`, `fit_blin`, `src/accumulate.jl`). The
  reference floors RSS at `1e-8` for split nodes only, leaving `con`/`lin`
  unfloored; ours floors every kind at `dmin = eps * max(Σ h z², n)`.
  Neither difference binds here (smallest eligible determinant is order
  1e2, residual RSS is order 0.75), but on small-`n` or small-scale data
  one guard could accept a fit the other rejects.
- **`max_lin_chain = 10` vs. the reference's `max_model_depth = 100`.**
  Ours caps a chain of `lin` nodes at 10; the reference's much larger cap
  effectively never binds. `linear` uses six `lin` fits in a row, four
  short of ours, so this fixture doesn't exercise the cap either -- a
  noisier linear design that needed more than 10 chained `lin` fits would
  diverge.

## CON nodes are leaves

In the reference, a `"con"` node's only child is `"END"`
(`Pilot.py`'s `build_tree`, the `best_node == "con"` branch). In
LinearTrees.jl, `CON` nodes are always created with `feature = 0`, so
`LinearTrees.isleaf` is `true` for them. Both sides therefore exclude `CON`
from the split-node comparison list.

## Datasets

The first five: `n = 300`, `p = 3`, `X ~ Uniform(-2, 2)` via
`numpy.random.default_rng(seed)`, rounded to 6 decimals, `seed` = the
dataset's position in the list below (1-5). Fit with `max_depth = 6`,
`min_sample_split = 10`, `min_sample_leaf = 5`, `truncation_factor = 3`.

| name | seed | y | true split kinds (root to leaves, END dropped) |
|---|---|---|---|
| linear | 1 | `1.5*X0 - 0.7*X1 + 0.3*X2 + N(0, 0.05)` | lin, lin, lin, lin, lin, lin, con |
| piecewise | 2 | `where(X0<0, 2*X0, 0.3*X0) + 0.2*X1 + N(0, 0.05)` | blin, lin, lin, con, lin, con |
| hinge | 3 | `max(X0-0.5, 0)*3 + N(0, 0.02)` | blin, con, con |
| interaction | 4 | `1.2*X0 + 0.8*X1 + 0.6*X0*X1 + N(0, 0.05)` | lin, lin, lin, con |
| step | 5 | `where(X0<0, -1, 1) + N(0, 0.02)` | pcon, con, con |

Three more fixtures, added to exercise a `plin` root, a binding
`max_depth`, and a categorical column -- none of which the first five
touch. Each has its own shape/seed, listed with it; all still use
`min_sample_split = 10`, `min_sample_leaf = 5`, `truncation_factor = 3`,
and `max_depth = 6` unless noted.

| name | seed | n, p | y | fit override | true split kinds |
|---|---|---|---|---|---|
| twoslope | 6 | 400, 3 | `where(X0<0.5, 3*X0, 4-2*X0) + N(0, 0.05)`, X0 ~ U(0,1), X1,X2 ~ U(-2,2) noise | -- | plin, lin, con, con |
| deep | 7 | 600, 2 | a 9-level step in X0 (cuts at -1.6..1.6 by 0.4) plus a 2-level step in X1, X ~ U(-2,2), + N(0, 0.05) | `max_depth = 3` | lin, pcon, plin, plin, con, con, plin, con, plin |
| categorical | 8 | 400, 2 | level means `[-2, 1, -2, 3]` for X0 in `{1,2,3,4}`, `+ 0.3*X1 + N(0, 0.05)`, X1 ~ U(-2,2) | `categorical = [1]` | pconc, lin, con, pconc, lin, con, lin, con |

**twoslope**: the discontinuity at `X0 = 0.5` (left slope 3, right slope
-2, a jump of 1.5) is steep enough on its own that the reference already
picks `plin` at the root with no extra steepening needed -- the brief's
fallback ("steepen until `plin` wins") never triggered.

**deep**: fit twice while designing this fixture, once with `max_depth =
12` (reaches `tree_depth = 9`) and once with `max_depth = 3` (stops at
exactly `tree_depth = 3`, the committed fixture) -- confirming the cap
actually changes the tree (30 nodes down to 9), not just failing to
matter. `lin` and `con` never count toward `tree_depth` on either side
(`Pilot.py:690-691`'s `best_node in ["con", "lin"]` decrement; likewise
`src/fit.jl`'s `LIN` branch does not increment `depth`), so the root
`lin` here is depth 0 and the cap only starts counting at its child.

**categorical**: PILOT's reference supports a categorical split (node
kind `"pconc"`, `Pilot.py`'s categorical branch of `best_split`, guarded
by a `categorical` array of 0-based column indices passed to `.fit`).
LinearTrees.jl has no separate `PCONC` kind: a categorical split is a
`PCON` node with `catwords > 0` (`src/node.jl`), so the fixture's
reference `"pconc"` kind maps to our `PCON` plus `iscategorical(node)`.
Column 0 (reference) / column 1 (ours, 1-based) is declared categorical
with codes `1..4`, chosen to match `fit_tree`'s codes convention
directly -- no remapping needed on that axis, only the kind-name mapping
above.

*Left-set direction is not ambiguous here.* Both sides sort the column's
levels by ascending mean response and cut the sorted order at the
lowest-scoring point, keeping the low-rank prefix on the left
(`Pilot.py:446,477`'s `mean_idx = argsort(mean_vec)` then
`pivot_c = possible_p[:i+1]`; `src/fit.jl`'s `scan_categorical` sorts
`present` the same way and takes `rank[lc] <= cand.threshold`). Given
that agreement, `test/pilot_reference.jl` compares the reference's
`pivot_c` (as a `Set` of 1-based codes) directly against ours, no `||`
fallback for the complementary partition (contrast
`test/categorical.jl`'s `lefts == ... || lefts == ...`, needed there only
because that test has no external oracle fixing a canonical direction).

`node.pivot` for a `"pconc"` node carries a meaningless second element
(`best_pivot` never updated in the categorical branch of `best_split`,
so it keeps its `-1.0` initializer, like `"lin"`'s placeholder pivot) --
the real split lives in `pivot_c` alone. Its `range`/`interval` is worse
than meaningless: `best_split` never assigns `interval` in that branch
either, so it keeps the function's own `[-inf, inf]` initializer, and
`json.dump`'s default `allow_nan=True` writes that as bare `Infinity` /
`-Infinity` tokens -- not valid strict JSON, which broke JSON3.jl's
parser (`ArgumentError: invalid JSON ... InvalidChar`) the first time
this fixture was generated. `make_fixtures.py`'s `walk` now records
`range: null` for any `"pconc"` node instead of trusting `node.interval`.

Each JSON file also records the `sha256` of its own `X.tobytes()` /
`y.tobytes()` (post-rounding) internally, for provenance; regenerating
with the same seeds reproduces those exactly. The table below is a
different thing: the SHA-256 of each *committed file's bytes as a whole*
(`nodes`, `pred`, and everything else included, not just the `X`/`y`
arrays), so a change to any part of a fixture is caught even where the
internal `X`/`y` hash would not move.

| file | sha256 |
|---|---|
| `linear.json` | `a8807ef9803fc480e6fd0f5166320f0b015f74dda54161a7ff3341042c9dfdd4` |
| `piecewise.json` | `dcb6a802f23b45e3da066a5d9235a4e20378c314b22a4231be6d04ea9df4f9d6` |
| `hinge.json` | `8db93f143ba483cd785d50ee7a0e6e0060076255d6af65c6ceeceb2d37e73cb8` |
| `interaction.json` | `392e87bbfb06c9c18839bfe74a32138a66b47eb98f6d39dd7e3f85a2411cbc97` |
| `step.json` | `126c5e0b779cadf694dfd601294df46d9e76a44072e61ce36fa97f41cdb64053` |
| `twoslope.json` | `4dec319e96871e749d45c4d7261e2ae6cdc3d6a88eb947e678b4b19406393b30` |
| `deep.json` | `967f42b6551df8f7649d8f4c16a4171da2e4b217d7537c8b168f104c704546d1` |
| `categorical.json` | `2959f4910e72eea835828d42dcf2dfd0b83e398c8b5ef09970ddc33971f8a09d` |

## Test outcome

All eight fixtures pass exact structural and prediction parity
(`predict` at `1e-8`, and every non-leaf node's kind, 1-based feature,
threshold, and `(coef, intercept)` on each side at `1e-8`, within the
"Scope of the parity claim" above, plus an exact left-level-set check
for `categorical`'s two `pconc`/`PCON` nodes). `linear` through `step`
fit with `max_depth = 6, min_leaf = 5, min_fit = 10, truncation_factor =
3`; `deep` overrides `max_depth = 3`; `categorical` adds `categorical =
[1]`. `test/pilot_reference.jl` holds these per-fixture overrides in
`FIXTURE_KW`.
