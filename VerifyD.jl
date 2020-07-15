include("./SFFM.jl")
using Plots, LinearAlgebra, KernelDensity, StatsBase

## Define the model
T = [-2.0 1.0 1.0; 1.0 -2.0 1.0; 1.0 1.0 -2]
C = [1.0; -2.0; 0]
r = (
    r = function (x)
        [(1.1 .+ sin.(x)) (sqrt.(x .* (x .> 0)) .+ 1) ((x .> 0) .* x .+ 1)]
    end, # r = function (x); [1.0.+0.01*x 1.0.+0.01*x 1*ones(size(x))]; end,
    R = function (x)
        [(1.1 .* x .- cos.(x)) ((x .* (x .> 0)) .^ (3 / 2) .* 2 / 3 .+ 1 * x) (
            (x .> 0) .* x .^ 2 / 2.0 .+ 1 * x
        )]
    end, # R = function (x); [1*x.+0.01.*x.^2.0./2 1*x.+0.01.*x.^2.0./2 1*x]; end
) # [1*(x.<=2.0).-2.0*(x.>1.0) -2.0*(x.<=2.0).+(x.>2.0)] # [1.0 -2.0].*ones(size(x))#
Bounds = [-10 10; -Inf Inf]
Model = SFFM.MakeModel(T = T, C = C, r = r, Bounds = Bounds)

# in out Y-level
y = 10

## Simulate the model
NSim = 40000
# IC = (φ = ones(Int, NSim), X = zeros(NSim), Y = zeros(NSim))
# IC = (φ = 2 .*ones(Int, NSim), X = -10*ones(NSim), Y = zeros(NSim))
IC = (
    φ = sum(rand(NSim) .< [1 / 3 2 / 3 1], dims = 2),
    X = 20.0 .* rand(NSim) .- 10,
    Y = zeros(NSim),
)
sims =
    SFFM.SimSFFM(Model = Model, StoppingTime = SFFM.InOutYLevel(y = y), InitCondition = IC)

## Define the mesh
Δ = 0.25
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
Mesh = SFFM.MakeMesh(Model = Model, Nodes = Nodes, NBases = NBases, Fil = Fil)

## Construct all DG operators
All = SFFM.MakeAll(Model = Model, Mesh = Mesh)
Matrices = All.Matrices
MatricesR = All.MatricesR
B = All.B
R = All.R
D = All.D
DR = All.DR

## initial condition
# x0 = Matrix(
#     [
#         zeros(sum(Model.C.<=0)) # LHS point mass
#         zeros(Mesh.NBases * Mesh.NIntervals * 1 ÷ 2) # phase 1
#         Mesh.Δ[1] # phase 2
#         zeros(NBases - 1) # phase 2
#         zeros(Mesh.NBases * Mesh.NIntervals * 1 ÷ 2 - NBases)
#         zeros(Mesh.TotalNBases * 2)
#         zeros(sum(Model.C.>=0)) # RHS point mass
#     ]',
# )
x0 = Matrix(
    [
        zeros(sum(Model.C .<= 0)) # LHS point mass
        repeat([1; zeros(NBases - 1)], Model.NPhases * Mesh.NIntervals, 1) ./
        (Model.NPhases * Mesh.NIntervals)
        zeros(sum(Model.C .>= 0)) # RHS point mass
    ]',
)

## DG approximations to exp(Dy)
yvalsR =
    SFFM.EulerDG(D = DR.DDict["++"](s = 0), y = y, x0 = x0, h = 0.0001)[[
        1:sum(Model.C .<= 0)
        sum(Model.C .<= 0).+1:NBases:end.-sum(Model.C .>= 0)
        end.-sum(Model.C .>= 0).+1:end
    ]]
yvals =
    SFFM.EulerDG(D = D["++"](s = 0), y = y, x0 = x0, h = 0.0001)[[
        1:sum(Model.C .<= 0)
        sum(Model.C .<= 0).+1:NBases:end.-sum(Model.C .>= 0)
        end.-sum(Model.C .>= 0).+1:end
    ]]

## analysis and plots

# plot solutions
p = plot(legend = false, layout = (3, 1))
Y = zeros(length(Nodes) - 1, Model.NPhases)
YR = zeros(length(Nodes) - 1, Model.NPhases)
let cum = 0
    for i = 1:Model.NPhases
        idx =
            findall(.!Fil[string(i)*"0"]) .- cum .+ (i - 1) * Mesh.NIntervals .+
            sum(Model.C .<= 0)
        cum = cum + sum(Fil[string(i)*"0"])
        p = plot!(
            (
                Mesh.CellNodes[1, .!Fil[string(i)*"0"]][:] +
                Mesh.CellNodes[end, .!Fil[string(i)*"0"]][:]
            ) / 2,
            yvals[idx],
            label = "φ=" * string(i) * " - D",
            subplot = i,
        )
        Y[:, i] = yvals[idx]
        p = plot!(
            (
                Mesh.CellNodes[1, .!Fil[string(i)*"0"]][:] +
                Mesh.CellNodes[end, .!Fil[string(i)*"0"]][:]
            ) / 2,
            yvalsR[idx],
            label = "φ=" * string(i) * " - DR",
            subplot = i,
        )
        YR[:, i] = yvalsR[idx]
    end
end
p = plot!(subplot = 1, legend = :topright)
pmdata = [
    [Nodes[1] * ones(sum(Model.C .<= 0)); Nodes[end] * ones(sum(Model.C .>= 0))]'
    yvals[[.!Mesh.Fil["p0"]; falses(Model.NPhases * Mesh.NIntervals); .!Mesh.Fil["q0"]]]'
    yvalsR[[.!Mesh.Fil["p0"]; falses(Model.NPhases * Mesh.NIntervals); .!Mesh.Fil["q0"]]]'
    [(sum(repeat(sims.X,1,Model.NPhases).*(sims.φ.==[1 2 3]).==Nodes[1],dims=1)./NSim)[Model.C.<=0];
    (sum(repeat(sims.X,1,Model.NPhases).*(sims.φ.==[1 2 3]).==Nodes[end],dims=1)./NSim)[Model.C.<=0]]'
]
SFFM.MyPrint([".";"pm";"pmR";"sim"])
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
plot(
    Nodes[1:end-1] + diff(Nodes) / 2,
    err,
    label = "err",
    legend = :topleft,
    layout = (3, 1),
)
plot!(Nodes[1:end-1] + diff(Nodes) / 2, errR, label = "errR")
display(sum(abs.(err) ))
display(sum(abs.(errR)))
display(abs.(err))
display(abs.(errR))
