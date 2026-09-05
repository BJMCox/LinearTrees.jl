"""Generate the pinned PILOT reference fixtures under test/fixtures/pilot/*.json.

This script is provenance only. It is not a runtime or test dependency of
LinearTrees.jl: the JSON files it writes are committed, and the Julia test
suite reads only those files.

To regenerate:
    1. Clone https://github.com/STAN-UAntwerp/PILOT (or its current home) and
       check out commit ea769a0d55dd2babf4c57d31094758d7879f29ff.
    2. Set PILOT_REF_PATH to the clone root (the directory that contains the
       `pilot/` package, i.e. `pilot/Pilot.py`), or edit PILOT_REF_PATH below.
    3. Run this file with a Python that has numpy, numba, pandas and
       scikit-learn installed (versions pinned in README.md). networkx and
       matplotlib are optional stand-ins are provided below since Tree.py
       only needs them for an unused visualization helper.
"""

import hashlib
import importlib
import json
import os
import subprocess
import sys
import types

import numpy as np

PILOT_REF_PATH = os.environ.get("PILOT_REF_PATH", "/path/to/pilot-ref")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_pilot(pilot_ref_path):
    """Import pilot.Pilot / pilot.Tree without running the package's
    __init__.py, which at the pinned commit imports a `cpilot` submodule
    that is not present in this clone."""
    if pilot_ref_path not in sys.path:
        sys.path.insert(0, pilot_ref_path)

    pkg = types.ModuleType("pilot")
    pkg.__path__ = [os.path.join(pilot_ref_path, "pilot")]
    sys.modules["pilot"] = pkg

    # Tree.py imports networkx/matplotlib only for an unused visualization
    # helper (construct_graph/visualize_tree); stub them out if absent.
    for name in ("networkx", "matplotlib", "matplotlib.pyplot", "networkx.drawing.nx_pydot"):
        try:
            importlib.import_module(name)
        except ImportError:
            sys.modules[name] = types.ModuleType(name)
    if not hasattr(sys.modules["networkx.drawing.nx_pydot"], "graphviz_layout"):
        sys.modules["networkx.drawing.nx_pydot"].graphviz_layout = lambda *a, **k: {}
    if not hasattr(sys.modules["matplotlib.pyplot"], "get_fignums"):
        sys.modules["matplotlib.pyplot"].get_fignums = lambda: []

    Tree = importlib.import_module("pilot.Tree")
    Pilot = importlib.import_module("pilot.Pilot")
    return Pilot, Tree


# Pilot.py REGRESSION_NODES = ["con", "lin", "pcon", "blin", "plin"] (line 21), but
# best_split fills its coefficient rows in the order blin (row 0), pcon (row 1), plin
# (row 2) and labels the winner `regression_nodes[2 + index_min]` (line 427) -- row 0
# (really blin) reads off REGRESSION_NODES[2] = "pcon", and row 1 (really pcon) reads
# off REGRESSION_NODES[3] = "blin". build_tree/predict/_get_child_data all key off
# node.node identically for pcon and blin, so this swap changes nothing about the fit
# or the predictions -- it is a label defect only. See README "Reference label defect".
LABEL_FIX = {"pcon": "blin", "blin": "pcon"}


def walk(node, data, Tree, out):
    """Flatten the fitted tree into a list of split nodes (skip END leaves)."""
    if node is None or node.node == "END":
        return
    entry = {"kind": LABEL_FIX.get(node.node, node.node)}
    if node.pivot is not None:
        entry["feature"] = int(node.pivot[0])
        entry["threshold"] = float(node.pivot[1])
    else:
        entry["feature"] = None
        entry["threshold"] = None
    entry["lm_l"] = None if node.lm_l is None else [float(v) for v in node.lm_l]
    entry["lm_r"] = None if node.lm_r is None else [float(v) for v in node.lm_r]
    # best_split never assigns `interval` in its categorical branch, so a
    # "pconc" node keeps the function's initial [-inf, inf] default -- not
    # meaningful (there is no numeric range for a categorical split), and
    # not valid strict JSON (Infinity/-Infinity aren't JSON tokens). Record
    # null there instead, like `pivot_c` below.
    entry["range"] = None if node.interval is None or node.node == "pconc" else [float(v) for v in node.interval]
    # Only "pconc" (categorical) nodes carry a meaningful pivot_c (left-level
    # set); omit the key entirely for every other kind, so the five original
    # fixtures -- none of which have a categorical column -- regenerate
    # byte-identical to before this field existed.
    if node.node == "pconc":
        entry["pivot_c"] = [int(v) for v in node.pivot_c]
    out.append(entry)
    left_data, right_data = Tree._get_child_data(data, node)
    walk(node.left, left_data, Tree, out)
    walk(node.right, right_data, Tree, out)


def linear(X, rng):
    return 1.5 * X[:, 0] - 0.7 * X[:, 1] + 0.3 * X[:, 2] + rng.normal(scale=0.05, size=X.shape[0])


def piecewise(X, rng):
    return (
        np.where(X[:, 0] < 0, 2.0 * X[:, 0], 0.3 * X[:, 0])
        + 0.2 * X[:, 1]
        + rng.normal(scale=0.05, size=X.shape[0])
    )


def hinge(X, rng):
    return np.maximum(X[:, 0] - 0.5, 0) * 3.0 + rng.normal(scale=0.02, size=X.shape[0])


def interaction(X, rng):
    return (
        1.2 * X[:, 0]
        + 0.8 * X[:, 1]
        + 0.6 * X[:, 0] * X[:, 1]
        + rng.normal(scale=0.05, size=X.shape[0])
    )


def step(X, rng):
    return np.where(X[:, 0] < 0, -1.0, 1.0) + rng.normal(scale=0.02, size=X.shape[0])


DESIGNS = [
    ("linear", 1, linear),
    ("piecewise", 2, piecewise),
    ("hinge", 3, hinge),
    ("interaction", 4, interaction),
    ("step", 5, step),
]

N, P = 300, 3
MAX_DEPTH, MIN_SAMPLE_LEAF, MIN_SAMPLE_SPLIT, TRUNCATION_FACTOR = 6, 5, 10, 3


# --- Extra fixtures (stream D): each owns its own shape, y-generator, and
# fit-kwarg overrides, since none of them share DESIGNS' fixed (N, P, uniform
# -2..2) shape. Each `gen` returns (X, y), already rounded to 6 decimals.


def gen_twoslope(seed):
    """Root should pick 'plin': a two-slope line with a discontinuity at 0.5,
    plus two uninformative noise features."""
    rng = np.random.default_rng(seed)
    n = 400
    x1 = rng.uniform(0, 1, size=n)
    noise = rng.uniform(-2, 2, size=(n, 2))
    X = np.c_[x1, noise].round(6)
    y = np.where(X[:, 0] < 0.5, 3 * X[:, 0], 4 - 2 * X[:, 0])
    y = (y + rng.normal(scale=0.05, size=n)).round(6)
    return X, y


def gen_deep(seed):
    """Many step levels on feature 0 plus a coarser step on feature 1, so the
    unconstrained reference tree grows to depth 9 -- fit with max_depth=3 so
    the cap actually binds and truncates it to depth 3."""
    rng = np.random.default_rng(seed)
    n = 600
    X = rng.uniform(-2, 2, size=(n, 2)).round(6)
    cuts0 = np.array([-1.6, -1.2, -0.8, -0.4, 0.0, 0.4, 0.8, 1.2, 1.6])
    vals0 = np.array([-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    y = vals0[np.searchsorted(cuts0, X[:, 0], side="right")]
    y = y + np.where(X[:, 1] < 0, -0.5, 0.5)
    y = (y + rng.normal(scale=0.05, size=n)).round(6)
    return X, y


def gen_categorical(seed):
    """One categorical column (4 levels, coded 1..4 to match
    LinearTrees.jl's 1-based codes directly, no remapping needed) driving a
    step, plus one continuous column."""
    rng = np.random.default_rng(seed)
    n = 400
    lvl = rng.integers(1, 5, size=n)  # levels 1..4
    xc = rng.uniform(-2, 2, size=n).round(6)
    level_means = np.array([np.nan, -2.0, 1.0, -2.0, 3.0])  # index by level; 1,3 low, 2,4 high
    y = level_means[lvl] + 0.3 * xc
    y = (y + rng.normal(scale=0.05, size=n)).round(6)
    X = np.c_[lvl.astype(float), xc]
    return X, y


EXTRA_DESIGNS = [
    dict(name="twoslope", seed=6, gen=gen_twoslope, fit_kwargs={}, categorical=np.array([-1])),
    dict(name="deep", seed=7, gen=gen_deep, fit_kwargs={"max_depth": 3}, categorical=np.array([-1])),
    dict(
        name="categorical",
        seed=8,
        gen=gen_categorical,
        fit_kwargs={},
        categorical=np.array([0], dtype=np.int64),
    ),
]


def git_head(path):
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=path, text=True
        ).strip()
    except Exception:
        return None


def main():
    Pilot, Tree = load_pilot(PILOT_REF_PATH)

    for name, seed, design in DESIGNS:
        rng = np.random.default_rng(seed)
        X = rng.uniform(-2, 2, size=(N, P)).round(6)
        y = design(X, rng).round(6)

        # Seed the legacy global numpy RNG that PILOT's internal
        # random_sample (feature-order shuffling) draws from, so a
        # regeneration is bit-for-bit reproducible even though it does not
        # affect these fixtures' outcome (no exact split ties occur).
        np.random.seed(seed)
        model = Pilot.PILOT(
            max_depth=MAX_DEPTH,
            min_sample_split=MIN_SAMPLE_SPLIT,
            min_sample_leaf=MIN_SAMPLE_LEAF,
            truncation_factor=TRUNCATION_FACTOR,
        )
        model.fit(X, y)

        nodes = []
        walk(model.model_tree, X, Tree, nodes)
        pred = model.predict(X)

        fixture = {
            "params": {
                "max_depth": MAX_DEPTH,
                "min_leaf": MIN_SAMPLE_LEAF,
                "min_fit": MIN_SAMPLE_SPLIT,
                "truncation_factor": TRUNCATION_FACTOR,
            },
            "seed": seed,
            "sha256": {
                "X": hashlib.sha256(X.tobytes()).hexdigest(),
                "y": hashlib.sha256(y.tobytes()).hexdigest(),
            },
            "X": X.tolist(),
            "y": y.tolist(),
            "nodes": nodes,
            "pred": pred.tolist(),
        }

        out_path = os.path.join(OUT_DIR, f"{name}.json")
        with open(out_path, "w") as f:
            json.dump(fixture, f)
        print(f"wrote {out_path} ({len(nodes)} split nodes: {[n['kind'] for n in nodes]})")

    for spec in EXTRA_DESIGNS:
        name, seed, gen = spec["name"], spec["seed"], spec["gen"]
        X, y = gen(seed)

        # Same reproducibility note as the DESIGNS loop above: seeds the
        # legacy global NumPy RNG that random_sample draws from.
        np.random.seed(seed)
        params = dict(
            max_depth=MAX_DEPTH,
            min_sample_split=MIN_SAMPLE_SPLIT,
            min_sample_leaf=MIN_SAMPLE_LEAF,
            truncation_factor=TRUNCATION_FACTOR,
        )
        params.update(spec["fit_kwargs"])
        model = Pilot.PILOT(**params)
        model.fit(X, y, categorical=spec["categorical"])

        nodes = []
        walk(model.model_tree, X, Tree, nodes)
        pred = model.predict(X)

        fixture = {
            "params": {
                "max_depth": params["max_depth"],
                "min_leaf": params["min_sample_leaf"],
                "min_fit": params["min_sample_split"],
                "truncation_factor": params["truncation_factor"],
            },
            "seed": seed,
            "sha256": {
                "X": hashlib.sha256(X.tobytes()).hexdigest(),
                "y": hashlib.sha256(y.tobytes()).hexdigest(),
            },
            "X": X.tolist(),
            "y": y.tolist(),
            "nodes": nodes,
            "pred": pred.tolist(),
        }

        out_path = os.path.join(OUT_DIR, f"{name}.json")
        with open(out_path, "w") as f:
            json.dump(fixture, f)
        print(f"wrote {out_path} ({len(nodes)} split nodes: {[n['kind'] for n in nodes]})")

    versions = {
        "python": sys.version,
        "numpy": np.__version__,
        "numba": importlib.import_module("numba").__version__,
        "sklearn": importlib.import_module("sklearn").__version__,
        "pandas": importlib.import_module("pandas").__version__,
        "pilot_commit": git_head(PILOT_REF_PATH),
    }
    print("versions:", json.dumps(versions, indent=2))


if __name__ == "__main__":
    main()
