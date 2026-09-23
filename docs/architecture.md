# Reviewed architecture

`tach.toml` defines the production module graph. Its `exact = true` declarations
are reviewed dependencies, not the output of dependency synchronization. Cycles
are forbidden, type-checking imports are checked, unowned source modules are
forbidden, and cross-layer imports require explicit dependencies. Named public
interfaces include the helper functions imported by generated report notebooks.

## Layers and contracts

This application is primarily scientific-data adapters. Computing a dataframe
without doing file I/O does not make a NumPy/Pandas implementation Domain code.

| Layer | Modules and responsibilities |
| --- | --- |
| Domain | Empty package root; artifact paths and preparation statistics; generic estimator contracts; generic EDA, clustering, generation and ranking records/protocols; sampling/evaluation invariants and frozen pipeline configuration. |
| Infrastructure | NumPy/Pandas/sklearn estimators and calculations; table validation; CSV/joblib/JSON persistence; notebook execution and rendering; plot rendering; progress logging; preparation, reference updates and report summaries. |
| Bootstrap | CLI entry points and parsers that validate registered model configurations; pipeline execution; temporal model/distance composition and default training strategy; ranking model registry, default-ranker selection and benchmark execution; clustering/ranking compatibility facades that export these operations. |

There is currently no separate application service depending only on injected
ports, or separate Interface layer. The five-layer vocabulary is retained for
future modules. Every present Infrastructure dependency targets Infrastructure
or Domain; every Domain dependency targets Domain. Exact explicit dependencies
also prevent the lateral layer directions forbidden by Eryx's matrix.

Important reviewed distinctions:

- `buy_today.__init__` contains package documentation only. It has no imports or
  dependency on subsystem facades, so importing standalone tooling does not load
  the numerical stack through the root.
- `schema.py` is Infrastructure: `read_dataset` performs CSV I/O and restores
  concrete Pandas dtypes. Its schema constants are shared between adapters.
- `progress.py` is Infrastructure because it emits logging events and measures
  execution time.
- `bundled_data.py` remains a standalone, standard-library-only Infrastructure
  module executable directly during the Docker build.
- Clustering's `models/temporal.py` composes a model and a distance. The actual
  `TemporalClustering` numerical implementation is in `temporal_estimator.py`.
  Reports share `labels.check_labels`, independently of training composition.
- Ranking's `storage.py` selects the default ranker and preserves compatibility
  exports. `snapshots.py` performs persistence/evaluation and trains an explicitly
  supplied estimator; inference and benchmark persistence import this adapter
  directly.
- `pipeline_config.py` imports pure drift/sampling parameters rather than the
  EDA report facade or history-generation implementation.
- Generic records are defined without NumPy/Pandas imports. Adapters specialize
  them with actual dataframe/array types. Existing constructor exports are kept;
  public records used in runtime instance checks remain classes. Generic defaults
  retain existing unsubscripted record annotations; numerical production
  functions specify their concrete table types.
- Estimator contracts retain the fluent `fit(X, y)` lifecycle and delegate model
  state updates to `_fit`. The same lifecycle serves temporal, random and SVD
  implementations and the stateless distance. sklearn remains the implementation
  of parameter inspection, cloning, tags and metadata routing.

## Native framework configuration

The consumer's native settings in `pyproject.toml` are synchronized to the generated
test configuration by `uv run --locked --group dev python eryx/apply.py --json`.

- `ignored-parents` adds exactly `sklearn.base.BaseEstimator` to the existing
  defaults. As documented in `eryx/PROFILE.md`, framework ancestry and its
  ancestors are not project inheritance. `max-parents` remains **3** and project
  protocol/mixin ancestry still counts.
- `good-names` preserves Pylint's `i`, `j`, `k`, `ex`, `Run`, `_` defaults and adds
  `X`, `Y`, the public sklearn fit/pairwise keyword names.
- `extension-pkg-allow-list = ["numpy.random.mtrand"]` enables inspection of the
  actual C extension providing `RandomState`.
- clean-arch's `allowed_prefixes = ["numbers"]` recognizes the standard-library
  numeric ABC module as pure. `Integral` and `Real` let Domain validation accept
  registered numeric scalars without importing NumPy; these ABCs perform no I/O.

## External runtime dependencies

The report runner uses `ipykernel.kernelspec.make_ipkernel_cmd` to build its real
kernel command with the selected Python executable. This replaces a handwritten
`-m ipykernel_launcher` command and makes the runtime dependency explicit.

Tach 0.35.1 checks `[project].dependencies` against imports in Python source. Its
native external settings provide `exclude` and import/distribution `rename`;
there is no declaration of dependencies consumed only through CLI commands or
notebooks. In particular, `[tool.tach.external]` cannot add such usage metadata.
See the pinned implementation:

- [Dependency extraction](https://github.com/gauge-sh/tach/blob/v0.35.1/src/external/parsing.rs)
- [External checking](https://github.com/gauge-sh/tach/blob/v0.35.1/src/commands/check/check_external.rs)

`jupyterlab` is interactive tooling and is installed through the development group.
Ordinary runtime
dependencies continue to be checked for genuinely unused declarations.

## Framework contract compatibility

The initial integration exposed 44 false `W9201` findings on inherited sklearn
metadata-routing methods for `RandomRanker`, `SVDRanker`, `TemporalClustering`
and `TimestampDistance`.

The pinned architecture checker walks all methods of each Infrastructure class,
including inherited methods. sklearn's `_MetadataRequester` declares ten
`set_*_request` methods under `if TYPE_CHECKING`, with an explicit comment that
these declarations never execute. Its runtime instead installs only request
methods appropriate to each estimator. All ten declarations are reported for
each estimator, plus the real inherited `get_metadata_routing` method.

Runtime inspection of these four estimators finds `get_metadata_routing` on all
four and `set_predict_request` on the two rankers. The other **38** reported
method/class combinations do not exist at runtime. The six real inherited
methods belong to sklearn's integration interface, rather than the narrow
application prediction/training contracts. The Eryx compatibility patch ignores
proven typing-only declarations and recognizes unchanged inherited parameter and
metadata-routing plumbing from the verified sklearn 1.9.1 implementation.
Project-defined methods, overrides and unrelated framework methods retain their
Domain contract checks. Standalone real-Pylint regression fixtures cover both
valid framework inheritance and genuine violations. See
`eryx/COMPATIBILITY.md` for the exact provenance and version guards.
