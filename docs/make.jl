using LinearTrees
using Documenter

DocMeta.setdocmeta!(LinearTrees, :DocTestSetup, :(using LinearTrees); recursive=true)

makedocs(;
    root=@__DIR__,
    repo=Documenter.Remotes.GitHub("BJMCox", "LinearTrees.jl"),
    modules=[LinearTrees],
    authors="Benjamin Cox <bcox@mpp.mpg.de>",
    sitename="LinearTrees.jl",
    checkdocs=:exports,
    doctest=true,
    warnonly=false,
    format=Documenter.HTML(;
        canonical="https://bjmcox.github.io/LinearTrees.jl/dev/",
        edit_link="main",
        prettyurls=true,
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Getting started" => "guide.md",
        "Manual" => [
            "Tree fitting" => "trees.md",
            "Boosting" => "boosting.md",
            "Loss functions" => "losses.md",
            "Interpretation" => "interpretation.md",
            "Interfaces and persistence" => "interfaces.md",
            "Performance" => "performance.md",
        ],
        "API reference" => "api.md",
    ],
)

deploydocs(;
    repo="github.com/BJMCox/LinearTrees.jl",
    devbranch="main",
)
