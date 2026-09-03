using LinearTrees
using Documenter

DocMeta.setdocmeta!(LinearTrees, :DocTestSetup, :(using LinearTrees); recursive=true)

makedocs(;
    modules=[LinearTrees],
    authors="Ben Cox <bcox@mpp.mpg.de>",
    sitename="LinearTrees.jl",
    format=Documenter.HTML(;
        canonical="https://BJMCox.github.io/LinearTrees.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/BJMCox/LinearTrees.jl",
    devbranch="main",
)
