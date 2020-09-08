include("src/SFFM.jl")
using LinearAlgebra, Plots, JLD2

include("examples/paperNumerics/exampleModelDef.jl")

@load pwd()*"/examples/paperNumerics/dump/sims.jld2" sims

simprobs = SFFM.Sims2Dist(Model=simModel,Mesh=Mesh,sims=sims,type="cumulative")

p2 = plot(
    simprobs.x,
    simprobs.distribution[:,:,2],
    label = :false,
    color = :red,
    xlims = (0,2),
    legend = :bottomright,
    title = "Phase 10",
    seriestype = :line,
    linestyle = :dot,
    markershape = :x,
    markersize = 4,
    xlabel = "x",
    ylabel = "Cumulative probability",
    windowsize = (600,400),
    grid = false,
    tickfontsize = 10,
    guidefontsize = 12,
    titlefontsize = 18,
    legendfontsize = 10,
)

Δ = 0.4
Nodes = collect(approxBounds[1, 1]:Δ:approxBounds[1, 2])

NBases = 2
Basis = "legendre"
Mesh = SFFM.MakeMesh(
    Model = approxModel,
    Nodes = Nodes,
    NBases = NBases,
    Basis = Basis,
)

All = SFFM.MakeAll(Model = approxModel, Mesh = Mesh, approxType = "projection")
Matrices = SFFM.MakeMatrices2(Model=approxModel,Mesh=Mesh)
MatricesR = SFFM.MakeMatricesR(Model=approxModel,Mesh=Mesh)
Dr = SFFM.MakeDR(
    Matrices=Matrices,
    MatricesR=MatricesR,
    Model=approxModel,
    Mesh=Mesh,
    B=All.B,
)

Ψ = SFFM.PsiFun(D=Dr.DDict)
# construct initial condition
theNodes = Mesh.CellNodes[:,convert(Int,ceil(5/Δ))]
basisValues = zeros(length(theNodes))
for n in 1:length(theNodes)
    basisValues[n] = prod(5.0.-theNodes[[1:n-1;n+1:end]])./prod(theNodes[n].-theNodes[[1:n-1;n+1:end]])
end
basisValues = basisValues'*V.V
initpm = [
    zeros(sum(approxModel.C.<=0)) # LHS point mass
    zeros(sum(approxModel.C.>=0)) # RHS point mass
]
initprobs = zeros(Float64,Mesh.NBases,Mesh.NIntervals,approxModel.NPhases)
initprobs[:,convert(Int,ceil(5/Δ)),3] = basisValues'*sqrt(2)/0.4#*All.Matrices.Local.V.V*All.Matrices.Local.V.V'.*2/Δ
# initprobs[1,convert(Int,ceil(5/Δ)),3] = sqrt(2)
initdist = (
    pm = initpm,
    distribution = initprobs,
    x = Mesh.CellNodes,
    type = "density"
) # convert to a distribution object so we can apply Dist2Coeffs
# convert to Coeffs α in the DG context
x0 = [0;0;initdist.distribution[:];0;0]' # SFFM.Dist2Coeffs(Model = approxModel, Mesh = Mesh, Distn = initdist) # #
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
    DGProbs.x,
    DGProbs.distribution[:,:,2],
    label = :false,#"DG: N_k = "*string(NBases),
    color = :blue,
    xlims = (0,2),
    seriestype = :line,
    linestyle = :dash,
)

SFFM.PlotSFM(Model=approxModel,Mesh=Mesh,Dist=DGProbs)
for sp in 1:4
    plot!(subplot=sp, xlims=(0,2))
end
display(plot!())
plot(z)










## simpler set up to start
## Model 1
T = [-1.0 1.0; 1.0 -1.0]
C = [2.0; -1.0]
r = (
    r = function (x)
        [abs.((x .<= 1)-(x .> 1)) abs.(((x .> 1))-2*(x .<= 1).*(x .> 0)-(x.==0))]#[ones(size(x)) ones(size(x))]#
    end,
    R = function (x)
        [x.*abs.(((x .<= 1)-(x .> 1))) x.*abs.((((x .> 1))-2*(x .<= 1).*(x .> 0)-(x.==0)))]#[x x]
    end,
)

## Model 3
T = [-1.0 1.0; 1.0 -1.0]
C = [2.0; -1.0]
r = (
    r = function (x)
        [cos.(x).+1.05 -1.25*(x.>-1)]
    end,
    R = function (x)
        [sin.(x).+1.05.*x -1.25*(x.>-1).*x]
    end,
)

Bounds = [0 30; -Inf Inf]
Model = SFFM.MakeModel(T = T, C = C, r = r, Bounds = Bounds)


## Define mesh

Δ = 1
Nodes = collect(Bounds[1,1]:Δ:Bounds[1,2])
NBases = 5
Basis = "legendre"
Mesh = SFFM.MakeMesh(Model = Model, Nodes = Nodes, NBases = NBases, Basis=Basis)

## matrices
All = SFFM.MakeAll(Model = Model, Mesh = Mesh, approxType = "projection")
Matrices = All.Matrices
MatricesR = SFFM.MakeMatricesR(Model=Model,Mesh=Mesh)
Dr = SFFM.MakeDR(
    Matrices=Matrices,
    MatricesR=MatricesR,
    Model=Model,
    Mesh=Mesh,
    B=All.B,
)
## sims fro gound truth
x₀ = 4*π
NSim = 50000
IC = (φ = ones(Int, NSim), X = x₀ .* ones(NSim), Y = zeros(NSim))
y = 10
# @time sims =
    # SFFM.SimSFFM(Model = Model, StoppingTime = SFFM.FirstExitY(u = -Inf, v = y), InitCondition = IC)

## DG stationary dist
# construct initial condition
theNodes = Mesh.CellNodes[:,convert(Int,ceil(x₀/Δ))]
basisValues = zeros(length(theNodes))
for n in 1:length(theNodes)
    basisValues[n] = prod(x₀.-theNodes[[1:n-1;n+1:end]])./prod(theNodes[n].-theNodes[[1:n-1;n+1:end]])
end
initpm = [
    zeros(sum(Model.C.<=0)) # LHS point mass
    zeros(sum(Model.C.>=0)) # RHS point mass
]
initprobs = zeros(Float64,Mesh.NBases,Mesh.NIntervals,Model.NPhases)
initprobs[:,convert(Int,ceil(x₀/Δ)),1] = basisValues'*All.Matrices.Local.V.V*All.Matrices.Local.V.V'.*2/Δ
initdist = (
    pm = initpm,
    distribution = initprobs,
    x = Mesh.CellNodes,
    type = "density"
)
x0 = SFFM.Dist2Coeffs(Model = Model, Mesh = Mesh, Distn = initdist)
# h = 0.0001
# @time DRyvals = SFFM.EulerDG(D = Dr.DR(s = 0), y = y, x0 = x0, h = h)
# @time vikramyvals = SFFM.EulerDG(D = vikramD, y = y, x0 = x0, h = h)

## plotting
# simdensity = SFFM.Sims2Dist(Model=Model,Mesh=Mesh,sims=sims,type="probability")
# DRdensity = SFFM.Coeffs2Dist(Model=Model,Mesh=Mesh,Coeffs=DRyvals,type="probability")

# vikramdensity = SFFM.Coeffs2Dist(Model=Model,Mesh=Mesh,Coeffs=vikramyvals,type="density")
# println("DR error: ",SFFM.starSeminorm(d1 = DRdensity, d2 = simdensity))

## plots
# plot solutions
# densities
# p = SFFM.PlotSFM(Model=Model,Mesh=Mesh,Dist=DRdensity,color=:blue,label="DR")#,marker=:rtriangle)

# plot sims
# p = SFFM.PlotSFM!(p;Model=Model,Mesh=Mesh,Dist=simdensity,color=:black,label="sim")#,marker=:ltriangle)

# display(p)
## probabilities
# simdensity = SFFM.Sims2Dist(Model=Model,Mesh=Mesh,sims=sims,type="density")
# DRdensity = SFFM.Coeffs2Dist(Model=Model,Mesh=Mesh,Coeffs=DRyvals,type="density")

## plots
# plot solutions
# densities
# p = SFFM.PlotSFM(Model=Model,Mesh=Mesh,Dist=DRdensity,color=:blue,label="DR")#,marker=:rtriangle)

# plot sims
# p = SFFM.PlotSFM!(p;Model=Model,Mesh=Mesh,Dist=simdensity,color=:black,label="sim")#,marker=:ltriangle)

# display(p)

## Ψ
Ψr = SFFM.PsiFun(D=Dr.DDict)
Ψ = SFFM.PsiFun(D=All.D)

@time sims =
    SFFM.SimSFFM(Model = Model, StoppingTime = SFFM.FirstExitY(u = 0, v = Inf), InitCondition = IC)

simdensity = SFFM.Sims2Dist(Model=Model,Mesh=Mesh,sims=sims,type="density")

plusIdx = [
    Mesh.Fil["p+"];
    repeat(Mesh.Fil["+"]', Mesh.NBases, 1)[:];
    Mesh.Fil["q+"];
]
# get the elements of x0 in + states only
x0 = x0[plusIdx]'

wr = x0*Ψr
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
    Mesh.NBases*Mesh.NIntervals * Model.NPhases +
        sum(Model.C.<=0) + sum(Model.C.>=0)
)
z[minusIdx] = w
zr = zeros(
    Float64,
    Mesh.NBases*Mesh.NIntervals * Model.NPhases +
        sum(Model.C.<=0) + sum(Model.C.>=0)
)
zr[minusIdx] = wr
# check that it is equal to 1
println(sum(z))

DRdensity = SFFM.Coeffs2Dist(Model=Model,Mesh=Mesh,Coeffs=zr,type="density")
Ddensity = SFFM.Coeffs2Dist(Model=Model,Mesh=Mesh,Coeffs=z,type="density")

SFFM.PlotSFM(Model=Model,Mesh=Mesh,Dist=simdensity)
SFFM.PlotSFM!(p;Model=Model,Mesh=Mesh,Dist=DRdensity,color=:red)
SFFM.PlotSFM!(p;Model=Model,Mesh=Mesh,Dist=Ddensity,color=:black)

plot(DRdensity.x[:], DRdensity.distribution[:,:,2][:])
plot!(Ddensity.x[:], Ddensity.distribution[:,:,2][:])
plot!(simdensity.x[:], simdensity.distribution[:,:,2][:])
