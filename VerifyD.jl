include("./SFFM.jl")
using Plots, LinearAlgebra, KernelDensity, StatsBase

## Define the model
T = [-2.0 1.0 1.0; 1.0 -2.0 1.0; 1.0 1.0 -2.0]
C = [1.0; -2.0; 0.0]
r = (
    r = function (x)
        [(1.1 .+ sin.(4*x)) (sqrt.(x .* (x .> 0)) .+ 1) ((x .> 0) .* x .+ 1)]
    end, # r = function (x); [1.0.+0.01*x 1.0.+0.01*x 1*ones(size(x))]; end,
    R = function (x)
        [(1.1 .* x .- cos.(4*x)./4) ((x .* (x .> 0)) .^ (3 / 2) .* 2 / 3 .+ 1 * x) (
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
y = 20

## Simulate the model
NSim = 100000
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
Δ = 5
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
NBases = 3
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
    # yvalsR = SFFM.EulerDG(D = DR.DDict["++"](s = 0), y = y, x0 = x0, h = h)[idx]
    yvals = SFFM.EulerDG(D = D["++"](s = 0), y = y, x0 = x0, h = h)
    yvals = [yvals[1:sum(Model.C.<=0)]; (Matrices.Local.V.V*reshape(yvals[sum(Model.C.<=0)+1:end-sum(Model.C.>=0)],NBases,length(yvals[sum(Model.C.<=0)+1:end-sum(Model.C.>=0)])÷NBases))[:]; yvals[end-sum(Model.C.>=0)+1:end]]
    # MyDyvals = SFFM.EulerDG(D = MyD.D(s = 0), y = y, x0 = x0, h = h)
    # MyDyvals = [MyDyvals[1:sum(Model.C.<=0)]; (diagm(Matrices.Local.V.w)*Matrices.Local.V.V*reshape(MyDyvals[3:end-2],NBases,length(MyDyvals[3:end-2])÷NBases))[:]; MyDyvals[end-sum(Model.C.>=0)+1:end]]
elseif Basis == "lagrange"
    # yvalsR = SFFM.EulerDG(D = DR.DDict["++"](s = 0), y = y, x0 = x0, h = h)
    # yvalsR = [yvalsR[1:sum(Model.C.<=0)]; sum(reshape(yvalsR[3:end-2],NBases,length(yvalsR[3:end-2])÷NBases),dims=1)'; yvalsR[end-sum(Model.C.>=0)+1:end]]
    yvals = SFFM.EulerDG(D = D["++"](s = 0), y = y, x0 = x0, h = h)
    yvals = [yvals[1:sum(Model.C.<=0)]; yvals[sum(Model.C.<=0)+1:end-sum(Model.C.>=0)].*repeat(1.0./Matrices.Local.V.w,Mesh.NIntervals*Model.NPhases).*(repeat(2.0./Mesh.Δ,1,Mesh.NBases*Model.NPhases)'[:]); yvals[end-sum(Model.C.>=0)+1:end]]
    # MyDyvals = SFFM.EulerDG(D = MyD.D(s = 0), y = y, x0 = x0, h = h)
end
# yvalsleg = diagm(Matrices.Local.V.w)*Matrices.Local.V.V*reshape(yvals[3:end-2],NBases,length(yvals[3:end-2])÷NBases)/sqrt(2)

## analysis and plots

# plot solutions
#p = plot(legend = false, layout = (3, 1))
Y = zeros(Mesh.TotalNBases, Model.NPhases)
YR = zeros(Mesh.TotalNBases, Model.NPhases)
MyY = zeros(Mesh.TotalNBases, Model.NPhases)
let cum = 0
    for i = 1:Model.NPhases
        idx =
            findall(repeat(.!Fil[string(i)*"0"],1,NBases)'[:]) .- cum .+ (i - 1) * Mesh.TotalNBases .+
            sum(Model.C .<= 0)
        cum = cum + sum(Fil[string(i)*"0"])*NBases
        p = plot!(
            Mesh.CellNodes[:, .!Fil[string(i)*"0"]][:],
            yvals[idx],
            label = "D",
            subplot = i,
        )
        Y[:, i] = yvals[idx]
        # p = plot!(
        #     (
        #         Mesh.CellNodes[1, .!Fil[string(i)*"0"]][:] +
        #         Mesh.CellNodes[end, .!Fil[string(i)*"0"]][:]
        #     ) / 2,
        #     yvalsR[idx],
        #     label = "DR",
        #     subplot = i,
        # )
        # YR[:, i] = yvalsR[idx]
        # p = plot!(
        #     Mesh.CellNodes[:, .!Fil[string(i)*"0"]][:],
        #     MyDyvals[idx],
        #     label = "MyD",
        #     subplot = i,
        # )
        # MyY[:, i] = MyDyvals[idx]
    end
end
p = plot!(subplot = 1, legend = :topright)
pmdata = [
    [Nodes[1] * ones(sum(Model.C .<= 0)); Nodes[end] * ones(sum(Model.C .>= 0))]'
    yvals[[1:2;end-1:end]]'
    # yvalsR[[1:2;end-1:end]]'
    # MyDyvals[[1:2;end-1:end]]'
    [(sum(repeat(sims.X,1,Model.NPhases).*(sims.φ.==[1 2 3]).==Nodes[1],dims=1)./NSim)[Model.C.<=0];
    (sum(repeat(sims.X,1,Model.NPhases).*(sims.φ.==[1 2 3]).==Nodes[end],dims=1)./NSim)[Model.C.<=0]]'
]
SFFM.MyPrint([".";"pm";"pmR";"pmMyD";"sim"])
SFFM.MyPrint(pmdata)

display(p)

# plot sims
H = zeros(length(Nodes) - 1, Model.NPhases)
for whichφ = 1:Model.NPhases
    #pltvals = kde(sims.X[sims.φ.==whichφ])
    #p = histogram!(sims.X[sims.φ.==whichφ],bins=Nodes,normalize=:probability,alpha=0.2)
    # plot!(
    #     range(
    #         minimum(sims.X[sims.φ.==whichφ]),
    #         maximum(sims.X[sims.φ.==whichφ]),length=100
    #     ),
    #     z->pdf(pltvals,z)*sum(sims.φ.==whichφ)/length(sims.φ),
    #     label = "φ="*string(i)*" - sim"
    # )
    h = fit(
        Histogram,
        sims.X[(sims.φ.==whichφ) .& (sims.X.!=Nodes[1]) .& (sims.X.!=Nodes[end])],
        Nodes,
    )
    h = h.weights ./ sum(h.weights) * sum(sims.φ .== whichφ) / length(sims.φ)
    H[:, whichφ] = h
    #p = plot!(
    #    Nodes[1:end-1] + diff(Nodes) / 2,
    #    h,
    #    label = "hist" * string(whichφ),
    #)
    p = bar!(
        (Nodes[1:end-1] + Nodes[2:end]) / 2,
        h,
        alpha = 0.2,
        bar_width = Mesh.Δ,
        label = "sims",
        subplot = whichφ,
    )
end
display(p)

# display errors
err = H - Y
errR = H - YR
MyDerr = H - MyY
plot(
    Nodes[1:end-1] + diff(Nodes) / 2,
    err,
    label = "err",
    legend = :topleft,
    layout = (3, 1),
)
plot!(Nodes[1:end-1] + diff(Nodes) / 2, errR, label = "errR")
plot!(Nodes[1:end-1] + diff(Nodes) / 2, MyDerr, label = "MyDerr")
display(sum(abs.(err) ))
display(sum(abs.(errR)))
display(sum(abs.(MyDerr)))
display(abs.(err))
display(abs.(errR))
