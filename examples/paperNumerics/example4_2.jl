include("../../src/SFFM.jl")
using LinearAlgebra, Plots

## define the model(s)
include("exampleModelDef.jl")

## section 4.2: Ψ paths

## simulate Ψ paths
innerNSim = 10^2
NSim = 10^3*innerNSim

# we can do this in parallell
using SharedArrays, Distributed
nprocs() < 5 && addprocs(5-nprocs())
@everywhere include("src/SFFM.jl")
@everywhere include("examples/exampleModelDef.jl")

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

## turn sims into a cdf
simprobs = SFFM.Sims2Dist(Model=simModel,Mesh=Mesh,sims=sims,type="cumulative")

## plot simulations
p2 = plot(
    simprobs.x[:],
    simprobs.distribution[:,:,2][:],
    label = "Sim",
    color = 1,
    xlims = (0,2),
    legend = :bottomright,
    title = "Phase 2",
)

p4 = plot(
    simprobs.x[:],
    simprobs.distribution[:,:,4][:],
    label = "Sim",
    color = 1,
    xlims = (0,2),
    legend = :bottomright,
    title = "Phase 4",
)

## DG
# construct matrices
All = SFFM.MakeAll(Model = approxModel, Mesh = Mesh, approxType = "projection")
Ψ = SFFM.PsiFun(D=All.D)

# construct initial condition
initpm = [
    zeros(sum(approxModel.C.<=0)) # LHS point mass
    zeros(sum(approxModel.C.>=0)) # RHS point mass
]
initprobs = zeros(Float64,1,Mesh.NIntervals,approxModel.NPhases)
initprobs[1,convert(Int,ceil(5/Δ)),3] = 1 # a point mass of 1 on the cell ceil(5/Δ))
initdist = (
    pm = initpm,
    distribution = initprobs,
    x = Matrix(Mesh.CellNodes[1,:]'),
    type = "probability"
) # convert to a distribution object so we can apply Dist2Coeffs
# convert to Coeffs α in the DG context
x0 = SFFM.Dist2Coeffs(Model = approxModel, Mesh = Mesh, Distn = initdist)
# the initial condition on Ψ is restricted to + states so find the + states
plusIdx = [
    Mesh.Fil["p+"];
    repeat(Mesh.Fil["+"]', Mesh.NBases, 1)[:];
    Mesh.Fil["q+"];
]
# get the elements of x0 in + states only
x0 = x0[plusIdx]'
# check that it is equal to 1 (or at least close)
println(sum(x0))

# compute x-distribution at the time when Y returns to 0
w = x0*Ψ
# this can occur in - states only, so find the - states
minusIdx = [
    Mesh.Fil["p-"];
    repeat(Mesh.Fil["-"]', Mesh.NBases, 1)[:];
    Mesh.Fil["q-"];
]
# then map to the whole state space for plotting
z = zeros(
    Float64,
    Mesh.NBases*Mesh.NIntervals * approxModel.NPhases +
        sum(approxModel.C.<=0) + sum(approxModel.C.>=0)
)
z[minusIdx] = w
# check that it is equal to 1
println(sum(z))

# convert to a distribution object for plotting
DGProbs = SFFM.Coeffs2Dist(
    Model = approxModel,
    Mesh = Mesh,
    Coeffs = z,
    type="cumulative",
)

# plot them
p2 = plot!(p2,
    DGProbs.x[:],
    DGProbs.distribution[:,:,2][:],
    label = "DG",
    color = 2,
    xlims = (0,2),
)

p4 = plot!(p4,
    DGProbs.x[:],
    DGProbs.distribution[:,:,4][:],
    label = "DG",
    color = 2,
    xlims = (0,2),
)
