include("./SFFM.jl")
using LinearAlgebra, Plots

## define the model(s)
include("./exampleModelSpec.jl")

simBounds = [0 Inf; -Inf Inf] # bounds for simulation only
simModel = SFFM.MakeModel(T = T, C = C, r = r, Bounds = simBounds)

approxBounds = [0 16; -Inf Inf] # bounds for approximation only
approxModel = SFFM.MakeModel(T = T, C = C, r = r, Bounds = approxBounds)
## simulate
innerNSim = 10^2
NSim = 10^2*innerNSim

using SharedArrays, Distributed
nprocs() < 5 && addprocs(5-nprocs())
@everywhere include("./SFFM.jl")
@everywhere include("./exampleModelSpec.jl")

simsOuter = SharedArray(zeros(NSim,5))
@time @sync @distributed for n in 1:(NSim÷innerNSim)
    IC = (
        φ = 3 .* ones(Int, innerNSim),
        X = 5 .* ones(innerNSim),
        Y = zeros(innerNSim)
    )
    simsInner = SFFM.SimSFFM(
        Model = simModel,
        StoppingTime = SFFM.FirstExitY(u = 0, v = Inf),
        InitCondition = IC,
    )
    simsOuter[1+(n-1)*innerNSim:n*innerNSim,:] =
        [simsInner.t simsInner.φ simsInner.X simsInner.Y simsInner.n]
end

sims = (
    t = simsOuter[:,1],
    φ = simsOuter[:,2],
    X = simsOuter[:,3],
    Y = simsOuter[:,4],
    n = simsOuter[:,5],
)
# Profile.clear()
# @profiler SFFM.SimSFFM(
#     Model = simModel,
#     StoppingTime = SFFM.FirstExitY(u = 0, v = Inf),
#     InitCondition = IC,
# )
## mesh
Δ = 0.4
Nodes = collect(approxBounds[1, 1]:Δ:approxBounds[1, 2])

NBases = 1
Basis = "lagrange"
Mesh = SFFM.MakeMesh(
    Model = approxModel,
    Nodes = Nodes,
    NBases = NBases,
    Basis=Basis
)

## DG
All = SFFM.MakeAll(Model = approxModel, Mesh = Mesh, approxType = "projection")
Ψ = SFFM.PsiFun(D=All.D)
initpm = [
    zeros(sum(approxModel.C.<=0)) # LHS point mass
    zeros(sum(approxModel.C.>=0)) # RHS point mass
]
initprobs = zeros(Float64,1,Mesh.NIntervals,approxModel.NPhases)
initprobs[1,13,3] = 1
initdist = (
    pm = initpm,
    distribution = initprobs,
    x = Matrix(Mesh.CellNodes[1,:]'),
    type = "probability"
)
x0 = SFFM.Dist2Coeffs(Model = approxModel, Mesh = Mesh, Distn = initdist)
x0 = x0[[Mesh.Fil["p+"]; Mesh.Fil["+"]; Mesh.Fil["q+"]]]'

z = zeros(Float64,Mesh.NIntervals*approxModel.NPhases+sum(approxModel.C.<=0)+sum(approxModel.C.>=0))
z[[Mesh.Fil["p-"]; Mesh.Fil["-"]; Mesh.Fil["q-"]]] = x0*Ψ

DGProbs = SFFM.Coeffs2Dist(
    Model = approxModel,
    Mesh = Mesh,
    Coeffs = z,
    type="probability",
    )
## plots
simprobs = SFFM.Sims2Dist(Model=simModel,Mesh=Mesh,sims=simsInner,type="probability")
cumprobs = cumsum(simprobs.distribution,dims=2)
cumprobs[:,:,3] = cumprobs[:,:,3] .+ simprobs.pm[1]
cumprobs[:,:,4] = cumprobs[:,:,4] .+ simprobs.pm[2]
cumprobs = repeat(cumprobs,2,1,1)
cumprobs[1,:,1] = [0; cumprobs[1,1:end-1,1]]
cumprobs[1,:,2] = [0; cumprobs[1,1:end-1,2]]
cumprobs[1,:,3] = [simprobs.pm[1]; cumprobs[1,1:end-1,3]]
cumprobs[1,:,4] = [simprobs.pm[2]; cumprobs[1,1:end-1,4]]
x = [
    Mesh.CellNodes .- Δ/2;
    Mesh.CellNodes .+ Δ/2;
]
cumprobs = (
    pm = simprobs.pm,
    distribution = cumprobs,
    x = x,
    type = "density",
)


p = SFFM.PlotSFM(Model=approxModel,Mesh=Mesh,Dist=DGProbs)

p1 = SFFM.PlotSFM(Model=approxModel,Mesh=Mesh,Dist=cumprobs)

p2 = plot(
    cumprobs.x,
    cumprobs.distribution[:,:,4],
    color = 1,
    legend = false,
    xlims = (0,1.6),
)

p3 = plot(
    cumprobs.x,
    cumprobs.distribution[:,:,2],
    color = 1,
    legend = false,
    xlims = (0,1.6),
)
