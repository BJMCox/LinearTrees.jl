# PILOT reference fixtures

Five pinned datasets fit against the reference PILOT implementation, used by
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

## CON nodes are leaves

In the reference, a `"con"` node's only child is `"END"`
(`Pilot.py`'s `build_tree`, the `best_node == "con"` branch). In
LinearTrees.jl, `CON` nodes are always created with `feature = 0`, so
`LinearTrees.isleaf` is `true` for them. Both sides therefore exclude `CON`
from the split-node comparison list.

## Datasets

All five: `n = 300`, `p = 3`, `X ~ Uniform(-2, 2)` via
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

`sha256` of `X.tobytes()` / `y.tobytes()` (post-rounding) is recorded in
each JSON file for provenance; regenerating with the same seeds reproduces
these exactly.

## Test outcome

Full suite: 456 passed, 0 failed, 0 errored. All five fixtures pass exact
structural and prediction parity (`fit_tree(...; max_depth = 6, min_leaf =
5, min_fit = 10, truncation_factor = 3)` against `predict` at `1e-8`, and
every non-leaf node's kind, 1-based feature, and threshold at `1e-8`).
