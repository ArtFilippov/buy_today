# Local third-party type declarations

These declarations refine the installed libraries' actual APIs. They are used
only by the type checker; Python continues to import the installed libraries.

## Configuration

Set `"stubPath": "typings"` in the root `pyrightconfig.json`. No new runtime or
development dependency, editable install, or path dependency is needed. Retain
the installed `scikit-learn-stubs==0.0.3` and the existing NumPy/SciPy/pandas stubs:
modules outside this overlay still resolve through those providers.

Before the root configuration is updated, run:

```bash
.venv/bin/pyright --pythonpath .venv/bin/python -p typings/checks/project.json src tests
```

Pyright also uses `typings` as its default stub path; explicitly configuring it
records the intended dependency rather than relying on that default.

## Scope and precision

| Library | Verified installed version | Refinements |
|---|---|---|
| joblib | 1.6.0 | `dump` overloads distinguish path → `list[str]` from stream → `None`; `load` supports the documented mmap/byte-order options. |
| nbformat | 5.11.1 | Notebook/cell/output fields, typed v4 constructor options, file/string I/O, validation and the real `Struct.copy` return type. |
| threadpoolctl | 3.7.0 | Limit inputs, restoring context managers, typed information records and signature-preserving `.wrap` decorators. |
| scikit-learn | 1.9.1 | Structural estimator cloning, container cloning, `safe=False` deep copies, parameter dictionaries, fluent setters, fitted-state validation and silhouette metrics. |
| Matplotlib | 3.11.2 | Existing complete declarations with refined property kwargs and path/file types for the plotting methods used in this project, including `pyplot.rc_context`. |

`Any` is limited to genuinely dynamic contracts: pickle reconstruction, arbitrary
JSON/extension fields, serialization state, forwarding JSON options, and
metric-dependent arguments. Estimator parameter values are `object`, not `Any`.
Matplotlib's extensible property kwargs are `object`; named arguments and concrete
artist return types retain their upstream types. Unknown output members are not
made into `Any` modules or catch-all class attributes.

Notebook field annotations describe normalized v4 Python notebooks: cell source
and stream output text are strings. `NotebookNode` remains an extensible JSON
mapping; arbitrary metadata attributes are necessarily dynamic. Attributes that
do not belong to a particular node can still raise `AttributeError`, just as at
runtime. Notebook constructor kwargs are checked against their documented node
fields rather than accepting arbitrary property names.

## Why the sklearn overlay includes unchanged modules

A sparse `stubPath` overlay is **not** a PEP 561 package merge. Pyright resolves
relative imports from the physical importing file. Experimentally, overriding
only `sklearn/base.pyi` leaves installed `StandardScaler` and `TruncatedSVD`
inheriting a second, unpatched `BaseEstimator`; they then fail assignment to the
public `BaseEstimator`. Re-exported metrics similarly bypass leaf overrides.

`build_overlays.py` computes the smallest transitive **relative-importer closure**
of the changed sklearn modules and retains those upstream declarations. Relative
imports in copied files are made absolute so omitted dependencies fall back to
the installed providers. This preserves a single estimator identity without
editing site-packages or requiring a replacement stub distribution. Unchanged
declarations are retained in full, rather than replacing unrelated APIs with
empty stubs. This preserves the existing provider's coverage; it is not a claim
to have upgraded every API in the older provider to sklearn 1.9.

Matplotlib's affected `.pyi` modules and re-export facade are likewise retained
in full. Its inline-typed `pyplot.py` is converted to declarations by replacing
function bodies with `...` and removing runtime docstring decorators; signatures,
overloads, imports and module declarations remain. Only the reviewed signatures
are refined. This avoids hiding the rest of pyplot behind a one-function stub.

The positive contract fixture checks concrete return types, clone preservation,
nominal compatibility of unrelated estimators, and fallback imports from SciPy,
sklearn preprocessing/manifold/decomposition, and Matplotlib's ticker/pyplot.

## Regeneration and verification

Generated declarations are tracked so ordinary checking does not run a generator.
Their source paths, versions and SHA-256 digests are in `upstream.json`.

```bash
# Verify the copied declarations still match the reviewed installed versions:
.venv/bin/python typings/build_overlays.py --check

# After reviewing a dependency upgrade and updating the generator's pins:
.venv/bin/python typings/build_overlays.py --write

# Positive and negative static contracts, plus real library behavior:
.venv/bin/python -m unittest discover -s typings/checks -p 'test_*.py' -v
```

The negative fixture must produce an error on **every** marked line and no errors
elsewhere. It exercises invalid paths, mmap modes, notebook fields, thread limits,
decorator arguments, metrics and plotting arguments/return types. It uses no type
ignores or diagnostic baselines. Runtime tests verify serialization, clone/unfitted
semantics, notebook round trips, thread-limit restoration, and a rendered PNG.

The generator pins Matplotlib 3.11.2 and scikit-learn-stubs 0.0.3. It refuses other
versions so API and import-graph changes require review. Keep a dependency upgrade
and its regenerated declarations together. Do not run a formatter over generated
files: their small, reproducible deltas from upstream are intentional.

## Sources and attribution

Primary evidence is the installed library implementations and their declarations:
`joblib/numpy_pickle.py`, `threadpoolctl.py`, `nbformat/{__init__,notebooknode,_struct,validator}.py`,
`nbformat/v4/nbbase.py`, `sklearn/base.py`, `sklearn/utils/validation.py`,
`sklearn/metrics/cluster/_unsupervised.py`, and Matplotlib's shipped `.pyi`/`pyplot.py`.

- [Pyright import resolution](https://github.com/microsoft/pyright/blob/1.1.414/docs/import-resolution.md)
- [scikit-learn-stubs upstream](https://github.com/hoel-bagard/scikit-learn-stubs)
- [Matplotlib typing guidance](https://matplotlib.org/stable/api/typing_api.html)

Copied Matplotlib declarations: Copyright (c) 2012– Matplotlib Development Team;
All Rights Reserved. Changes are the reviewed annotations/import resolution
described above. Its license is in `licenses/matplotlib.txt`.
The scikit-learn-stubs MIT license is in `licenses/scikit-learn-stubs.txt`;
upstream per-module author/license comments are retained.
