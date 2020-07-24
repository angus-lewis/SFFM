include("./SFFM.jl")
using Plots, LinearAlgebra, KernelDensity, StatsBase

## Define the model
T = [-2.0 1.0 1.0; 1.0 -2.0 1.0; 1.0 1.0 -2.0]
C = [1.0; -2.0; 0.0]
r = (
    r = function (x)
        [(1.1 .+ sin.(π*x)) (sqrt.(x .* (x .> 0)) .+ 1) ((x .> 0) .* x .+ 1)]
    end, # r = function (x); [1.0.+0.01*x 1.0.+0.01*x 1*ones(size(x))]; end,
    R = function (x)
        [(1.1 .* x .- cos.(π*x)./π) ((x .* (x .> 0)) .^ (3 / 2) .* 2 / 3 .+ 1 * x) (
            (x .> 0) .* x .^ 2 / 2.0 .+ 1 * x
        )]
    end, # R = function (x); [1*x.+0.01.*x.^2.0./2 1*x.+0.01.*x.^2.0./2 1*x]; end
) # [1*(x.<=2.0).-2.0*(x.>1.0) -2.0*(x.<=2.0).+(x.>2.0)] # [1.0 -2.0].*ones(size(x))#

# r = (
#     r = function (x)
#         [ones(size(x)) ones(size(x)) ones(size(x))]
#     end, # r = function (x); [1.0.+0.01*x 1.0.+0.01*x 1*ones(size(x))]; end,
#     R = function (x)
#         [x x x]
#     end, # R = function (x); [1*x.+0.01.*x.^2.0./2 1*x.+0.01.*x.^2.0./2 1*x]; end
# )

Bounds = [-10 10; -Inf Inf]
Model = SFFM.MakeModel(T = T, C = C, r = r, Bounds = Bounds)

# in out Y-level
y = 10

## Simulate the model
NSim = 50000
IC = (φ = ones(Int, NSim), X = zeros(NSim), Y = zeros(NSim))
# IC = (φ = 2 .*ones(Int, NSim), X = -10*ones(NSim), Y = zeros(NSim))
# IC = (
#     φ = sum(rand(NSim) .< [1 / 3 2 / 3 1], dims = 2),
#     X = 20.0 .* rand(NSim) .- 10,
#     Y = zeros(NSim),
# )
sims =
    SFFM.SimSFFM(Model = Model, StoppingTime = SFFM.InOutYLevel(y = y), InitCondition = IC)

## Define the mesh
Δ = 0.25
Nodes = collect(Bounds[1, 1]:Δ:Bounds[1, 2])
# Fil = Dict{String,BitArray{1}}(
#     "1+" => trues(length(Nodes) - 1),
#     "2+" => trues(length(Nodes) - 1),
#     "3+" => trues(length(Nodes) - 1),
#     "p2+" => trues(1),
#     "p3+" => trues(1),
#     "q1+" => trues(1),
#     "q3+" => trues(1),
# )
NBases = 10
Basis = "legendre"
Mesh = SFFM.MakeMesh(Model = Model, Nodes = Nodes, NBases = NBases, Basis=Basis)

## Construct all DG operators
All = SFFM.MakeAll(Model = Model, Mesh = Mesh)
Matrices = All.Matrices
MatricesR = All.MatricesR
B = All.B
R = All.R
D = All.D
DR = All.DR

## initial condition
initpm = [
    zeros(sum(Model.C.<=0)) # LHS point mass
    zeros(sum(Model.C.>=0)) # RHS point mass
]
initprobs = zeros(Float64,1,Mesh.NIntervals,Model.NPhases)
initprobs[1,(1+Mesh.NIntervals)÷2+1,1] = 1
initdist = (pm = initpm, distribution = initprobs, x = Matrix(Mesh.CellNodes[1,:]'+Mesh.Δ'/2), type = "probability")
x0 = SFFM.Dist2Coeffs(Model = Model, Mesh = Mesh, Distn = initdist)
p = SFFM.PlotSFM(Model=Model,Mesh=Mesh,Dist=initdist)

## approximations to exp(Dy)
h = 0.0001
@time yvals = SFFM.EulerDG(D = D["++"](s = 0), y = y, x0 = x0, h = h)

# convert to densities
# densities
simdensity = SFFM.Sims2Dist(Model=Model,Mesh=Mesh,sims=sims,type="density")
density = SFFM.Coeffs2Dist(Model=Model,Mesh=Mesh,Coeffs=yvals,type="density")

# convert to probabilities
# get estimates of probabilities
simprobs = SFFM.Sims2Dist(Model=Model,Mesh=Mesh,sims=sims,type="probability")
probs = SFFM.Coeffs2Dist(Model=Model,Mesh=Mesh,Coeffs=yvals,type="probability")

## plots
# plot solutions
# densities
p = SFFM.PlotSFM(Model=Model,Mesh=Mesh,Dist=density)
# plot sims
p = SFFM.PlotSFM!(p;Model=Model,Mesh=Mesh,Dist=simdensity,color=4)

# probabilities
p = SFFM.PlotSFM(Model=Model,Mesh=Mesh,Dist=probs)
# plot sims
p = SFFM.PlotSFM!(p;Model=Model,Mesh=Mesh,Dist=simprobs,color=2)

## other analysis
# compute errors
derrpm = density.pm-simdensity.pm
derrdensity = density.distribution - simdensity.distribution
derrdist = (pm=derrpm,distribution=derrdensity,x=density.x,type="density")

# plot
p = SFFM.PlotSFM(Model=Model,Mesh=Mesh,Dist=derrdist)

# display point mass data
pmdata = [
    ["." (Nodes[1] * ones(sum(Model.C .<= 0)))' (Nodes[end] * ones(sum(Model.C .>= 0)))']
    "sim" simprobs.pm'
    "pm" probs.pm'
]
display(pmdata)

# probabilities
# compute errors
perrpm = probs.pm-simprobs.pm
perrdensity = probs.distribution - simprobs.distribution
perrdist = (pm=perrpm,distribution=perrdensity,x=probs.x,type="probability")

p = SFFM.PlotSFM!(p;Model=Model,Mesh=Mesh,Dist=perrdist)
