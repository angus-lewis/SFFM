include("../src/SFFM.jl")
using LinearAlgebra, Plots, JLD2, StatsBase

## define the model(s)
include("testModel1.jl")

## section 4.2: Ψ paths

## load simulated Ψ paths
@load pwd()*"/tests/dump/sims.jld2" sims

## mesh
Δ = 0.4
Nodes = collect(approxBounds[1, 1]:Δ:approxBounds[1, 2])

NBases = 1
Basis = "lagrange"
mesh = SFFM.DGMesh(
    model,
    Nodes = Nodes,
    NBases = NBases,
    Basis = Basis,
)

## turn sims into a cdf
simprobs = SFFM.Sims2Dist(model=model,mesh=mesh,sims=sims,type="cumulative")

