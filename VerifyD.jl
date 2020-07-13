push!(LOAD_PATH,"/Users/a1627293/Documents/SFFM")
using Plots, LinearAlgebra, KernelDensity, StatsBase, SFFM

## Define the model
T = [-2.0 1.0 1.0; 1.0 -2.0 1.0; 1.0 1.0 -2]
C = [1.0; -2.0; 0]
r = (
    r = function (x)
        [1.0 .+ sin.(x) 2 * (x .> 0) .* x .^ 2.0 .+ 1 (x .> 0) .* x .+ 1]
    end, # r = function (x); [1.0.+0.01*x 1.0.+0.01*x 1*ones(size(x))]; end,
    R = function (x)
        [x .- cos.(x) 2 * (x .> 0) .* x .^ 3 / 3.0 .+ 1 * x (x .> 0) .* x .^ 2 / 2.0 .+
                                                            1 * x]
    end, # R = function (x); [1*x.+0.01.*x.^2.0./2 1*x.+0.01.*x.^2.0./2 1*x]; end
) # [1*(x.<=2.0).-2.0*(x.>1.0) -2.0*(x.<=2.0).+(x.>2.0)] # [1.0 -2.0].*ones(size(x))#
Bounds = [-Inf Inf; -Inf Inf]
Model = SFFM.MakeModel(T = T, C = C, r = r, Bounds = Bounds)

# in out Y-level
y = 10

## Simulate the model
NSim = 40000
IC = (φ = ones(Int, NSim), X = zeros(NSim), Y = zeros(NSim))
sims = SFFM.SimSFFM(
    Model = Model,
    StoppingTime = SFFM.InOutYLevel(y = y),
    InitCondition = IC,
)

## Define the mesh
Nodes = collect(-10.0:0.5:10.0)
Fil = Dict{String,BitArray{1}}(
    "1+" => trues(length(Nodes) - 1),
    "2+" => trues(length(Nodes) - 1),
    "3+" => trues(length(Nodes) - 1),
)
NBases = 2
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
x0 = Matrix(
    [
        zeros(Mesh.NBases * Mesh.NIntervals * 1 ÷ 2)
        Mesh.Δ[1]
        zeros(NBases - 1)
        zeros(Mesh.NBases * Mesh.NIntervals * 1 ÷ 2 - NBases)
        zeros(Mesh.TotalNBases * 2)
    ]',
)

## DG approximations to exp(Dy)
yvalsR =
    SFFM.EulerDG(D = DR.DDict["++"](s = 0), y = y, x0 = x0, h = 0.0001)[1:NBases:end] ./
    Mesh.Δ[1]
yvals =
    SFFM.EulerDG(D = D["++"](s = 0), y = y, x0 = x0, h = 0.0001)[1:NBases:end] ./
    Mesh.Δ[1]

## analysis and plots

# plot solutions
p = plot(legend = :topleft, layout = (3, 1))
Y = zeros(length(Nodes) - 1, Model.NPhases)
YR = zeros(length(Nodes) - 1, Model.NPhases)
let cum = 0
    for i = 1:Model.NPhases
        idx = findall(.!Fil[string(i)*"0"]) .- cum .+ (i - 1) * Mesh.NIntervals
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
    h = fit(Histogram, sims.X[sims.φ.==whichφ], Nodes)
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
display(sum(abs.(err) * Mesh.Δ[1]))
display(sum(abs.(errR) * Mesh.Δ[1]))
display(abs.(err))
display(abs.(errR))
