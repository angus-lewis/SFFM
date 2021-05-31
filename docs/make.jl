using Documenter, SFFM

makedocs(
    modules = [SFFM],
    format = Documenter.HTML(; prettyurls = get(ENV, "CI", nothing) == "true"),
    authors = "angus-lewis",
    sitename = "SFFM.jl",
    pages = Any["index.md"]
    # strict = true,
    # clean = true,
    # checkdocs = :exports,
)

deploydocs(
    repo = "github.com/angus-lewis/SFFM.jl.git",
    push_preview = true
)
