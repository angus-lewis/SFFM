include("../src/SFFM.jl")
using LinearAlgebra, Plots, JLD2

include("../examples/paperNumerics/exampleModelDef.jl")

@load pwd()*"/examples/paperNumerics/dump/sims.jld2" sims

Δ = 0.4
Nodes = collect(approxBounds[1, 1]:Δ:approxBounds[1, 2])

NBases = 1
Basis = "lagrange"
Mesh = SFFM.MakeMesh(
    Model = approxModel,
    Nodes = Nodes,
    NBases = NBases,
    Basis = Basis,
)

simprobs = SFFM.Sims2Dist(Model=simModel,Mesh=Mesh,sims=sims,type="density")

p2 = plot(
    simprobs.x,
    simprobs.distribution[:,:,4],
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
Ψbase = SFFM.PsiFun(D=All.D)

# construct initial condition
theNodes = Mesh.CellNodes[:,convert(Int,ceil(5/Δ))]
basisValues = zeros(length(theNodes))
for n in 1:length(theNodes)
    basisValues[n] = prod(5.0.-theNodes[[1:n-1;n+1:end]])./prod(theNodes[n].-theNodes[[1:n-1;n+1:end]])
end
V = SFFM.vandermonde(NBases=NBases)
if Mesh.Basis=="lagrange"
    basisValues = basisValues'*V.V*V.V'*2/Δ
else
    basisValues = basisValues./V.w
    basisValues = basisValues'*V.inv'*2/Δ
end
initpm = [
    zeros(sum(Model.C.<=0)) # LHS point mass
    zeros(sum(Model.C.>=0)) # RHS point mass
]
initprobs = zeros(Float64,Mesh.NBases,Mesh.NIntervals,approxModel.NPhases)
initprobs[:,convert(Int,ceil(5/Δ)),3] = basisValues #*All.Matrices.Local.V.V*All.Matrices.Local.V.V'.*2/Δ
# initprobs[1,convert(Int,ceil(5/Δ)),3] = sqrt(2)
initdist = (
    pm = initpm,
    distribution = initprobs,
    x = Mesh.CellNodes,
    type = "cumulative"
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
wbase = x0*Ψbase

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
zbase = copy(z)
z[minusIdx] = w
zbase[minusIdx] = wbase
# check that it is equal to 1
println(sum(z))
println(sum(zbase))

# convert to a distribution object for plotting
DGProbs = SFFM.Coeffs2Dist(
    Model = approxModel,
    Mesh = Mesh,
    Coeffs = z,
    type="density",
)
DGProbsBase = SFFM.Coeffs2Dist(
    Model = approxModel,
    Mesh = Mesh,
    Coeffs = zbase,
    type="density",
)
display(DGProbsBase.pm)
display(DGProbs.pm)
# plot them
if Basis=="lagrange"
    p2 = plot!(p2,
        DGProbs.x,
        [1;1]*reshape(z[3:end-2], Mesh.NBases, Mesh.NIntervals, approxModel.NPhases)[:,:,4], #DGProbs.distribution[:,:,2],
        label = :false,#"DG: N_k = "*string(NBases),
        color = :blue,
        xlims = (0,2),
        seriestype = :line,
        linestyle = :dash,
    )
    p2 = plot!(p2,
        DGProbsBase.x,
        [1;1]*reshape(zbase[3:end-2], Mesh.NBases, Mesh.NIntervals, approxModel.NPhases)[:,:,4], #DGProbsBase.distribution[:,:,2],
        label = :false,#"DG: N_k = "*string(NBases),
        color = :black,
        xlims = (0,2),
        seriestype = :line,
        linestyle = :dash,
    )
else
    p2 = plot!(p2,
        DGProbs.x,
        DGProbs.distribution[:,:,4],
        label = :false,#"DG: N_k = "*string(NBases),
        color = :blue,
        xlims = (0,2),
        seriestype = :line,
        linestyle = :dash,
    )
    p2 = plot!(p2,
        DGProbsBase.x,
        DGProbsBase.distribution[:,:,4],
        label = :false,#"DG: N_k = "*string(NBases),
        color = :black,
        xlims = (0,2),
        seriestype = :line,
        linestyle = :dash,
    )
end

p = SFFM.PlotSFM(Model=approxModel,Mesh=Mesh,Dist=DGProbs)
SFFM.PlotSFM!(p;Model=approxModel,Mesh=Mesh,Dist=simprobs,color=2)
for sp in 1:4
    plot!(subplot=sp, xlims=(0,2))
end
display(plot!())
plot(z)










## simpler set up to start
## Model 1
T = [-1.0 1.0; 1.0 -1.0]
C = [1.0; -1.0]
r = (
    r = function (x)
        [abs.((x .<= 1)-(x .> 1)) abs.(((x .> 1))-2*(x .<= 1).*(x .> 0)-(x.==0))]#[ones(size(x)) ones(size(x))]#
    end,
    R = function (x)
        [x.*abs.(((x .<= 1)-(x .> 1))) x.*abs.((((x .> 1))-2*(x .<= 1).*(x .> 0)-(x.==0)))]#[x x]
    end,
)

r = (
    r = function (x)
        [2*(x.>-1).*(x.<5) -2*(x.>-1)]#[ones(size(x)) ones(size(x))]#
    end,
    R = function (x)
        [2*(x.>-1).*(x.<5).*x.+(x.>=5).*5 -2*(x.>-1).*x]
    end,
)

## Model 3
T = [-1.0 1.0; 1.0 -1.0]
C = [1.0; -1.01]
r = (
    r = function (x)
        [(cos.(x).+1.05) -4*(x.>-1)]
    end,
    R = function (x)
        [(sin.(x).+1.05.*x) -4*(x.>-1).*x]
    end,
)

Bounds = [0 15; -Inf Inf]
Model = SFFM.MakeModel(T = T, C = C, r = r, Bounds = Bounds)


## Define mesh

Δ = 4
Nodes = collect(Bounds[1,1]:Δ:Bounds[1,2])
NBases = 1
Basis = "lagrange"
Mesh = SFFM.MakeMesh(Model = Model, Nodes = Nodes, NBases = NBases, Basis=Basis)

## matrices
All = SFFM.MakeAll(Model = Model, Mesh = Mesh, approxType = "projection")
Matrices = All.Matrices
MatricesR = SFFM.MakeMatricesR(Model=Model,Mesh=Mesh)
Matrices2 = SFFM.MakeMatrices(Model=Model,Mesh=Mesh,probTransform=false)
B = SFFM.MakeB(Model=Model,Mesh=Mesh,Matrices=Matrices,probTransform=false)
Dr = SFFM.MakeDR(
    Matrices=Matrices,
    MatricesR=MatricesR,
    Model=Model,
    Mesh=Mesh,
    B=B,
)
## sims for gound truth
x₀ = 0.01;#4*π
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
V = SFFM.vandermonde(NBases=NBases)
if Mesh.Basis=="lagrange"
    basisValues = basisValues'*V.V*V.V'*2/Δ
else
    basisValues = basisValues./V.w
    basisValues = basisValues'*V.inv'*2
end
initpm = [
    zeros(sum(Model.C.<=0)) # LHS point mass
    zeros(sum(Model.C.>=0)) # RHS point mass
]
initprobs = zeros(Float64,Mesh.NBases,Mesh.NIntervals,Model.NPhases)
initprobs[:,convert(Int,ceil(x₀/Δ)),1] = basisValues #*All.Matrices.Local.V.V*All.Matrices.Local.V.V'.*2/Δ
# initprobs[1,convert(Int,ceil(5/Δ)),3] = sqrt(2)
initdist = (
    pm = initpm,
    distribution = initprobs,
    x = Mesh.CellNodes,
    type = "density"
)
x0 = [0;initdist.distribution[:];0]'
# x0 = SFFM.Dist2Coeffs(Model = Model, Mesh = Mesh, Distn = initdist)
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
simBounds = [0 Inf; -Inf Inf]
simModel = SFFM.MakeModel(T = T, C = C, r = r, Bounds = simBounds)

@time sims =
    SFFM.SimSFFM(Model = simModel, StoppingTime = SFFM.FirstExitY(u = 0, v = Inf), InitCondition = IC)

simdensity = SFFM.Sims2Dist(Model=simModel,Mesh=Mesh,sims=sims,type="density")

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

# p = SFFM.PlotSFM(Model=Model,Mesh=Mesh,Dist=simdensity)
# p = SFFM.PlotSFM!(p;Model=Model,Mesh=Mesh,Dist=DRdensity,color=:red)
# p = SFFM.PlotSFM!(p;Model=Model,Mesh=Mesh,Dist=Ddensity,color=:black)

simprobs = SFFM.Sims2Dist(Model=Model,Mesh=Mesh,sims=sims,type="probability")
DRprobs = SFFM.Coeffs2Dist(Model=Model,Mesh=Mesh,Coeffs=zr,type="probability")
Dprobs = SFFM.Coeffs2Dist(Model=Model,Mesh=Mesh,Coeffs=z,type="probability")
eR = SFFM.starSeminorm(d1 = DRprobs, d2 = simprobs)
e = SFFM.starSeminorm(d1 = Dprobs, d2 = simprobs)
ediff = SFFM.starSeminorm(d1 = DRprobs, d2 = Dprobs)
plot(DRdensity.x[:], DRdensity.distribution[:,:,2][:],label="DR")
plot!(Ddensity.x[:], Ddensity.distribution[:,:,2][:],label="D")
plot!(simdensity.x[:], simdensity.distribution[:,:,2][:],label="Sim")
