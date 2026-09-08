# Development

Run the package tests from the repository root:

```sh
julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

Build the documentation and run its doctests:

```sh
julia --project=docs -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'
julia --project=docs docs/make.jl
```

Generated HTML is in `docs/build`. Serve that directory with a local HTTP server
to inspect it. CI retains the same build as a `documentation` artifact. After the
tests and documentation pass on a non-pull-request build, GitHub Pages publishes
the versioned Documenter site, including the development build at
<https://bjmcox.github.io/LinearTrees.jl/dev/>.

CI checks Julia 1.10 serially, current Julia with four threads on
Linux/macOS/Windows, and Julia prereleases with four threads on Linux. The current
Julia/Linux job retains `lcov.info` as a `coverage` artifact and uploads it to
Codecov using GitHub OIDC. Coverage measures executed lines; it does not replace
tests of numerical or statistical behavior.

Before submitting a change, keep public behavior documented, add focused tests for
new behavior or regressions, and run the relevant package and documentation checks.
Do not commit generated documentation in `docs/build`.
