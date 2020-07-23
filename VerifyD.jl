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
        [(1.1 .* x .- cos.(π*x)./4) ((x .* (x .> 0)) .^ (3 / 2) .* 2 / 3 .+ 1 * x) (
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
NSim = 30000
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
Δ = 2
Nodes = collect(Bounds[1, 1]:Δ:Bounds[1, 2])
Fil = Dict{String,BitArray{1}}(
    "1+" => trues(length(Nodes) - 1),
    "2+" => trues(length(Nodes) - 1),
    "3+" => trues(length(Nodes) - 1),
    "p2+" => trues(1),
    "p3+" => trues(1),
    "q1+" => trues(1),
    "q3+" => trues(1),
)
NBases = 4
Basis = "legendre"
Mesh = SFFM.MakeMesh(Model = Model, Nodes = Nodes, NBases = NBases, Fil = Fil, Basis=Basis)

## Construct all DG operators
All = SFFM.MakeAll(Model = Model, Mesh = Mesh)
Matrices = All.Matrices
MatricesR = All.MatricesR
B = All.B
R = All.R
D = All.D
DR = All.DR
MyD = SFFM.MakeMyD(Model = Model, Mesh = Mesh, B = B, V = Matrices.Local.V)

## initial condition
if Basis == "legendre"
    x0 = Matrix( # = a(t) for the legendre basis
        [
            zeros(sum(Model.C.<=0)) # LHS point mass
            zeros(Mesh.NBases * Mesh.NIntervals * 1 ÷ 2) # phase 1
            1.0./Mesh.Δ[1]*sqrt(2) # phase 2
            zeros(NBases - 1) # phase 2
            zeros(Mesh.NBases * Mesh.NIntervals * 1 ÷ 2 - NBases)
            zeros(Mesh.TotalNBases * 2)
            zeros(sum(Model.C.>=0)) # RHS point mass
        ]',
    )
elseif Basis == "lagrange"
    x0 = Matrix( # = α(t) for the legendre basis
        [
            zeros(sum(Model.C.<=0)) # LHS point mass
            zeros(Mesh.NBases * Mesh.NIntervals * 1 ÷ 2) # phase 1
            1.0./Mesh.Δ[1]*diagm(Matrices.Local.V.w)*Mesh.Δ[1]/2*ones(NBases) # phase 2
            zeros(Mesh.NBases * Mesh.NIntervals * 1 ÷ 2 - NBases)
            zeros(Mesh.TotalNBases * 2)
            zeros(sum(Model.C.>=0)) # RHS point mass
        ]',
    )
end
# x0 = Matrix(
#     [
#         zeros(sum(Model.C .<= 0)) # LHS point mass
#         repeat([1; zeros(NBases - 1)], Model.NPhases * Mesh.NIntervals, 1) ./
#         (Model.NPhases * Mesh.NIntervals)
#         zeros(sum(Model.C .>= 0)) # RHS point mass
#     ]',
# )

## DG approximations to exp(Dy)
h = 0.001
if Basis == "legendre"
    idx = [
        1:sum(Model.C .<= 0)
        sum(Model.C .<= 0).+1:NBases:length(x0).-sum(Model.C .>= 0)
        length(x0).-sum(Model.C .>= 0).+1:length(x0)
    ]
    #  yvalsR = SFFM.EulerDG(D = DR.DDict["++"](s = 0), y = y, x0 = x0, h = h)[idx]
    yvals = SFFM.EulerDG(D = D["++"](s = 0), y = y, x0 = x0, h = h)
    MyDyvals = SFFM.EulerDG(D = MyD.D(s = 0), y = y, x0 = x0, h = h)
elseif Basis == "lagrange"
    # yvalsR = SFFM.EulerDG(D = DR.DDict["++"](s = 0), y = y, x0 = x0, h = h)
    yvals = SFFM.EulerDG(D = D["++"](s = 0), y = y, x0 = x0, h = h)
    MyDyvals = SFFM.EulerDG(D = MyD.D(s = 0), y = y, x0 = x0, h = h)
end

## analysis and plots

# plot solutions
# densities
p = SFFM.PlotSFM(Model=Model,Mesh=Mesh,Coeffs=yvals)
p = SFFM.PlotSFM!(p;Model=Model,Mesh=Mesh,Coeffs=MyDyvals,color=2)
# plot sims
p = SFFM.PlotSFMSim!(p;Model=Model,Mesh=Mesh,sims=sims,type="density")

# probabilities
p = SFFM.PlotSFM(Model=Model,Mesh=Mesh,Coeffs=yvals,type="probability")
p = SFFM.PlotSFM!(p;Model=Model,Mesh=Mesh,Coeffs=MyDyvals,color=2,type="probability")
# plot sims
p = SFFM.PlotSFMSim!(p;Model=Model,Mesh=Mesh,sims=sims,type="probability")

## other analysis
# point masses
simprobs = SFFM.Sims2Dist(Model=Model,Mesh=Mesh,sims=sims,type="probability")
probs = SFFM.Coeffs2Dist(Model=Model,Mesh=Mesh,Coeffs=yvals,type="probability")
MyDprobs = SFFM.Coeffs2Dist(Model=Model,Mesh=Mesh,Coeffs=MyDyvals,type="probability")

pmdata = [
    ["." (Nodes[1] * ones(sum(Model.C .<= 0)))' (Nodes[end] * ones(sum(Model.C .>= 0)))']
    "sim" simprobs.pm'
    "pm" probs.pm'
    "MyDpm" MyDprobs.pm'
]
display(pmdata)

# probabilities
simprobs = SFFM.Sims2Dist(Model=Model,Mesh=Mesh,sims=sims,type="probability")
probs = SFFM.Coeffs2Dist(Model=Model,Mesh=Mesh,Coeffs=yvals,type="probability")
#MyDprobs = SFFM.Coeffs2Dist(Model=Model,Mesh=Mesh,Coeffs=MyDyvals,type="probability")

errpm = probs.pm-simprobs.pm
errdensity = probs.distribution - simprobs.distribution
errdist = (pm=errpm,distribution=errdensity,x=x,type="probability")
errcoeffs = SFFM.Dist2Coeffs(Model=Model, Mesh=Mesh, Distn=errdist)

p = SFFM.PlotSFM(Model=Model,Mesh=Mesh,Coeffs=errcoeffs,type=errdist.type)

# densities
simprobs = SFFM.Sims2Dist(Model=Model,Mesh=Mesh,sims=sims,type="density")
probs = SFFM.Coeffs2Dist(Model=Model,Mesh=Mesh,Coeffs=yvals,type="density")
#MyDprobs = SFFM.Coeffs2Dist(Model=Model,Mesh=Mesh,Coeffs=MyDyvals,type="density")

errpm = probs.pm-simprobs.pm
errdensity = probs.distribution - simprobs.distribution
errdist = (pm=errpm,distribution=errdensity,x=x,type="density")
errcoeffs = SFFM.Dist2Coeffs(Model=Model, Mesh=Mesh, Distn=errdist)

p = SFFM.PlotSFM!(p;Model=Model,Mesh=Mesh,Coeffs=errcoeffs,type="density")

# MyDerr = SFFM.Dist2Coeffs(Model=Model, Distn = (pm = MyDpm, yvals = H - MyDdistn, xvals=xvals))
ys = SFFM.Dist2Coeffs(Model=Model, Mesh=Mesh, Distn=probs)
p = SFFM.PlotSFM!(p;Model=Model,Mesh=Mesh,Coeffs=ys,type="density")
