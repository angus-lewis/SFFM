include("./SFFM.jl")
using LinearAlgebra, Plots

## define the model(s)
include("./exampleModelDef.jl")

simBounds = [0 Inf; -Inf Inf] # bounds for simulation only
simModel = SFFM.MakeModel(T = T, C = C, r = r, Bounds = simBounds)

approxBounds = [0 16; -Inf Inf] # bounds for approximation only
approxModel = SFFM.MakeModel(T = T, C = C, r = r, Bounds = approxBounds)

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

## simulate
innerNSim = 10^2
NSim = 10^3*innerNSim

using SharedArrays, Distributed
nprocs() < 5 && addprocs(5-nprocs())
@everywhere include("./SFFM.jl")
@everywhere include("./exampleModelDef.jl")

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

simprobs = SFFM.Sims2Dist(Model=simModel,Mesh=Mesh,sims=sims,type="probability")
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
println(sum(x0))
z = zeros(Float64,Mesh.NIntervals*approxModel.NPhases+sum(approxModel.C.<=0)+sum(approxModel.C.>=0))
z[[Mesh.Fil["p-"]; Mesh.Fil["-"]; Mesh.Fil["q-"]]] = x0*Ψ
println(sum(z))
DGProbs = SFFM.Coeffs2Dist(
    Model = approxModel,
    Mesh = Mesh,
    Coeffs = z,
    type="probability",
)

DGcumprobs = cumsum(DGProbs.distribution,dims=2)
DGcumprobs[:,:,3] = DGcumprobs[:,:,3] .+ DGProbs.pm[1]
DGcumprobs[:,:,4] = DGcumprobs[:,:,4] .+ DGProbs.pm[2]
DGcumprobs = repeat(DGcumprobs,2,1,1)
DGcumprobs[1,:,1] = [0; DGcumprobs[1,1:end-1,1]]
DGcumprobs[1,:,2] = [0; DGcumprobs[1,1:end-1,2]]
DGcumprobs[1,:,3] = [DGProbs.pm[1]; DGcumprobs[1,1:end-1,3]]
DGcumprobs[1,:,4] = [DGProbs.pm[2]; DGcumprobs[1,1:end-1,4]]
x = [
    Mesh.CellNodes .- Δ/2;
    Mesh.CellNodes .+ Δ/2;
]
DGcumprobs = (
    pm = DGProbs.pm,
    distribution = DGcumprobs,
    x = x,
    type = "density",
)
## plots

p1 = SFFM.PlotSFM(Model=approxModel,Mesh=Mesh,Dist=DGcumprobs)
p1 = SFFM.PlotSFM!(p1;Model=approxModel,Mesh=Mesh,Dist=cumprobs,color=2)

p2 = plot(
    cumprobs.x[:],
    cumprobs.distribution[:,:,4][:],
    label = "Sim",
    color = 1,
    xlims = (0,1.6),
    legend = :topleft,
)
p2 = plot!(
    DGcumprobs.x[:],
    DGcumprobs.distribution[:,:,4][:],
    label = "DG",
    color = 2,
    xlims = (0,1.6),
)

p3 = plot(
    DGcumprobs.x[:],
    DGcumprobs.distribution[:,:,2][:],
    label = "Sim",
    color = 1,
    xlims = (0,1.6),
    legend = :topleft,
)
p3 = plot!(
    cumprobs.x[:],
    cumprobs.distribution[:,:,2][:],
    label = "DG",
    color = 2,
    xlims = (0,1.6),
)
