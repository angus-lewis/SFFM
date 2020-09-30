include("../src/SFFM.jl")
using LinearAlgebra, Plots, JLD2

include("../examples/paperNumerics/exampleModelDef.jl")

@load pwd()*"/examples/paperNumerics/dump/sims.jld2" sims

Δ = 0.4
Nodes = collect(approxBounds[1, 1]:Δ:approxBounds[1, 2])

NBases = 4
Basis = "lagrange"
Mesh = SFFM.MakeMesh(
    Model = approxModel,
    Nodes = Nodes,
    NBases = NBases,
    Basis = Basis,
)

simprobs = SFFM.Sims2Dist(Model=simModel,Mesh=Mesh,sims=sims,type="probability")

p2 = plot(
    Mesh.CellNodes[[1;end],:],# [1;1]*simprobs.x',
    [1;1]*simprobs.distribution[:,:,4],
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
B = SFFM.MakeB(Model=approxModel,Mesh=Mesh,Matrices=Matrices,probTransform=false)
Dr = SFFM.MakeDR(
    Matrices=Matrices,
    MatricesR=MatricesR,
    Model=approxModel,
    Mesh=Mesh,
    B=B,
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
    zeros(sum(approxModel.C.<=0)) # LHS point mass
    zeros(sum(approxModel.C.>=0)) # RHS point mass
]
initprobs = zeros(Float64,Mesh.NBases,Mesh.NIntervals,approxModel.NPhases)
initprobs[:,convert(Int,ceil(5/Δ)),3] = basisValues #*All.Matrices.Local.V.V*All.Matrices.Local.V.V'.*2/Δ
initprobsBase = zeros(Float64,Mesh.NBases,Mesh.NIntervals,approxModel.NPhases)
initprobsBase[:,convert(Int,ceil(5/Δ)),3] = basisValues.*V.w'/2*Δ
# initprobs[1,convert(Int,ceil(5/Δ)),3] = sqrt(2)
initdist = (
    pm = initpm,
    distribution = initprobs,
    x = Mesh.CellNodes,
    type = "cumulative"
) # convert to a distribution object so we can apply Dist2Coeffs
# convert to Coeffs α in the DG context
x0 = [0;0;initdist.distribution[:];0;0]' # SFFM.Dist2Coeffs(Model = approxModel, Mesh = Mesh, Distn = initdist) # #
x0Base = [0;0;initprobsBase[:];0;0]' # SFFM.Dist2Coeffs(Model = approxModel, Mesh = Mesh, Distn = initdist) # #
# the initial condition on Ψ is restricted to + states so find the + states
plusIdx = [
    Mesh.Fil["p+"];
    repeat(Mesh.Fil["+"]', Mesh.NBases, 1)[:];
    Mesh.Fil["q+"];
]
# get the elements of x0 in + states only
x0 = x0[plusIdx]'
x0Base = x0Base[plusIdx]'
# check that it is equal to 1 (or at least close)
println(sum(x0))

# compute x-distribution at the time when Y returns to 0
w = x0*Ψ

wbase = x0Base*Ψbase

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
    type="probability",
)
DGProbsBase = SFFM.Coeffs2Dist(
    Model = approxModel,
    Mesh = Mesh,
    Coeffs = zbase,
    type="probability",
)
display(DGProbsBase.pm)
display(DGProbs.pm)
display(simprobs.pm)
plot!(p2, DGProbsBase.x[:], DGProbsBase.distribution[:,:,4][:],label = :false,#"DG: N_k = "*string(NBases),
    color = :blue,
    xlims = (0,2),
    seriestype = :scatter
    )
1
# plot them
if Basis=="lagrange"
    if Mesh.NBases==1
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
            linestyle = :dot,
        )
    else
        p2 = plot!(p2,
            Mesh.CellNodes[[1;end],:],
            [1;1]*0.5*Δ*V.w'*reshape(z[3:end-2], Mesh.NBases, Mesh.NIntervals, approxModel.NPhases)[:,:,4], #DGProbs.distribution[:,:,2],
            label = :false,#"DG: N_k = "*string(NBases),
            color = :blue,
            xlims = (0,2),
            seriestype = :line,
            linestyle = :dash,
        )
        p2 = plot!(p2,
            Mesh.CellNodes[[1;end],:],
            [1;1]*ones(1,NBases)*reshape(zbase[3:end-2], Mesh.NBases, Mesh.NIntervals, approxModel.NPhases)[:,:,4], #DGProbs.distribution[:,:,2],
            label = :false,#"DG: N_k = "*string(NBases),
            color = :black,
            xlims = (0,2),
            seriestype = :line,
            linestyle = :dot,
        )
    end
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
        linestyle = :dot,
    )
end

# p = SFFM.PlotSFM(Model=approxModel,Mesh=Mesh,Dist=DGProbs)
# SFFM.PlotSFM!(p;Model=approxModel,Mesh=Mesh,Dist=simprobs,color=2)
# for sp in 1:4
#     plot!(subplot=sp, xlims=(0,2))
# end
# display(plot!())
# plot(z)










## simpler set up to start
## Model 1
T = [-1.0 1.0; 1.0 -1.0]
C = [1.0; -1.2]
# r = (
#     r = function (x)
#         [abs.((x .<= 1)-(x .> 1)) abs.(((x .> 1))-2*(x .<= 1).*(x .> 0)-(x.==0))]#[ones(size(x)) ones(size(x))]#
#     end,
#     R = function (x)
#         [x.*abs.(((x .<= 1)-(x .> 1))) x.*abs.((((x .> 1))-2*(x .<= 1).*(x .> 0)-(x.==0)))]#[x x]
#     end,
# )

# r = (
#     r = function (x)
#         [1.5*(x.>-1) -2*(x.>-1)]#[ones(size(x)) ones(size(x))]#
#     end,
#     R = function (x)
#         [1.5*(x.>-1).*x -2*(x.>-1).*x]
#     end,
# )

## Model 3
# T = [-1.0 1.0; 1.0 -1.0]*4
# C = [1.0; -2.0]
# r = (
#     r = function (x)
#         [-2*(cos.(x).+1.05) 1*(x.>-1)]
#     end,
#     R = function (x)
#         [-2*(sin.(x).+1.05.*x) 1*(x.>-1).*x]
#     end,
# )

T = [-1.0 1.0; 1.0 -1.0]*5
C = [1.0; -5]
r = (
    r = function (x)
        [(cos.(x).+1.05)./(2.05) -3*(x.>1).*(x.>0).-(x.<=1)]
    end,
    R = function (x)
        [(sin.(x).+1.05.*x)./(2.05) -(x.<=1).*x.-3*(x.>1).*x.-1]
    end,
)

Bounds = [0 10; -Inf Inf]
Model = SFFM.MakeModel(T = T, C = C, r = r, Bounds = Bounds)


## Define mesh

Δ = 2.5
Nodes = collect(Bounds[1,1]:Δ:Bounds[1,2])
NBases = 3
Basis = "lagrange"
Mesh = SFFM.MakeMesh(Model = Model, Nodes = Nodes, NBases = NBases, Basis=Basis)

## matrices
All = SFFM.MakeAll(Model = Model, Mesh = Mesh, approxType = "projection")
Matrices = All.Matrices
MatricesR = SFFM.MakeMatricesR(Model=Model,Mesh=Mesh)
Matrices2 = SFFM.MakeMatrices(Model=Model,Mesh=Mesh,probTransform=false)
B = SFFM.MakeB(Model=Model,Mesh=Mesh,Matrices=Matrices2,probTransform=false)
Dr = SFFM.MakeDR(
    Matrices=Matrices2,
    MatricesR=MatricesR,
    Model=Model,
    Mesh=Mesh,
    B=B,
)
## sims for gound truth
x₀ = 0.21
NSim = 50000
IC = (φ = ones(Int, NSim), X = x₀ .* ones(NSim), Y = zeros(NSim))
y = 10
Ψr = SFFM.PsiFun(D=Dr.DDict,s=0)
Ψ = SFFM.PsiFun(D=All.D,s=0)
simBounds = [0 Inf; -Inf Inf]
simModel = SFFM.MakeModel(T = T, C = C, r = r, Bounds = simBounds)

@time sims =
    SFFM.SimSFFM(Model = simModel, StoppingTime = SFFM.FirstExitY(u = 0, v = Inf), InitCondition = IC)

simdensity = SFFM.Sims2Dist(Model=simModel,Mesh=Mesh,sims=sims,type="density")
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
    basisValues = basisValues'*V.inv'*2/Δ
end
initpm = [
    zeros(sum(Model.C.<=0)) # LHS point mass
    zeros(sum(Model.C.>=0)) # RHS point mass
]
initprobs = zeros(Float64,Mesh.NBases,Mesh.NIntervals,Model.NPhases)
initprobs[:,convert(Int,max(1,ceil(x₀/Δ))),1] = basisValues #*All.Matrices.Local.V.V*All.Matrices.Local.V.V'.*2/Δ
initprobsBase = zeros(Float64,Mesh.NBases,Mesh.NIntervals,Model.NPhases)
initprobsBase[:,convert(Int,max(1,ceil(x₀/Δ))),1] = basisValues.*V.w'/2*Δ
# initprobs[1,convert(Int,ceil(5/Δ)),3] = sqrt(2)
initdist = (
    pm = initpm,
    distribution = initprobs,
    x = Mesh.CellNodes,
    type = "cumulative"
) # convert to a distribution object so we can apply Dist2Coeffs
# convert to Coeffs α in the DG context
x0 = [0;initdist.distribution[:];0]' # SFFM.Dist2Coeffs(Model = approxModel, Mesh = Mesh, Distn = initdist) # #
x0Base = [0;initprobsBase[:];0]' # SFFM.Dist2Coeffs(Model = approxModel, Mesh = Mesh, Distn = initdist) # #
# the initial condition on Ψ is restricted to + states so find the + states
plusIdx = [
    Mesh.Fil["p+"];
    repeat(Mesh.Fil["+"]', Mesh.NBases, 1)[:];
    Mesh.Fil["q+"];
]
# get the elements of x0 in + states only
x0 = x0[plusIdx]'
x0Base = x0Base[plusIdx]'
# check that it is equal to 1 (or at least close)
println(sum(x0))

# compute x-distribution at the time when Y returns to 0
w = x0*Ψr

wbase = x0Base*Ψ

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
zbase = copy(z)
z[minusIdx] = w
zbase[minusIdx] = wbase
# check that it is equal to 1
println(sum(z))
println(sum(zbase))

DRdensity = SFFM.Coeffs2Dist(Model=Model,Mesh=Mesh,Coeffs=z,type="density")
Ddensity = SFFM.Coeffs2Dist(Model=Model,Mesh=Mesh,Coeffs=zbase,type="density")

# p = SFFM.PlotSFM(Model=Model,Mesh=Mesh,Dist=simdensity)
# p = SFFM.PlotSFM!(p;Model=Model,Mesh=Mesh,Dist=DRdensity,color=:red)
# p = SFFM.PlotSFM!(p;Model=Model,Mesh=Mesh,Dist=Ddensity,color=:black)

simprobs = SFFM.Sims2Dist(Model=Model,Mesh=Mesh,sims=sims,type="probability")
DRprobs = SFFM.Coeffs2Dist(Model=Model,Mesh=Mesh,Coeffs=z,type="probability")
Dprobs = SFFM.Coeffs2Dist(Model=Model,Mesh=Mesh,Coeffs=zbase,type="probability")

display(simprobs.pm)
display(DRprobs.pm)
display(Dprobs.pm)
eR = SFFM.starSeminorm(d1 = DRprobs, d2 = simprobs)
e = SFFM.starSeminorm(d1 = Dprobs, d2 = simprobs)
ediff = SFFM.starSeminorm(d1 = DRprobs, d2 = Dprobs)
# plot(DRprobs.x[:], DRprobs.distribution[:,:,2][:],label="DR")
xvals = [Mesh.Nodes[1:end-1]'; Mesh.Nodes[2:end]']
plot(xvals, [1;1]*Dprobs.distribution[:,:,2],label=false, color=:black,
    seriestype = :line,
)
plot!(xvals, [1;1]*simprobs.distribution[:,:,2],label=false, color=:red)

if Basis=="lagrange"
    if Mesh.NBases==1
        p2 = plot!(
            xvals,
            [1;1]*0.5*Δ*2*reshape(z[2:end-1], Mesh.NBases, Mesh.NIntervals, Model.NPhases)[:,:,2], #DGProbs.distribution[:,:,2],
            label = :false,#"DG: N_k = "*string(NBases),
            color = :blue,
            xlims = (0,10),
            seriestype = :line,
            linestyle = :dash,
        )
    else
        p2 = plot!(
            xvals,
            [1;1]*0.5*Δ*V.w'*reshape(z[2:end-1], Mesh.NBases, Mesh.NIntervals, Model.NPhases)[:,:,2], #DGProbs.distribution[:,:,2],
            label = :false,#"DG: N_k = "*string(NBases),
            color = :blue,
            xlims = (0,10),
            seriestype = :line,
            linestyle = :dash,
        )
    end
else
    plot!(xvals, [1;1]*DRprobs.distribution[:,:,2],label=false, color=:blue)
    # p2 = plot!(
    #     DGProbs.x,
    #     DGProbs.distribution[:,:,2],
    #     label = :false,#"DG: N_k = "*string(NBases),
    #     color = :blue,
    #     xlims = (0,10),
    #     seriestype = :line,
    #     linestyle = :dash,
    # )
end

plot!([0],[0], color = :blue, label = "DR")
plot!([0],[0], color = :black,label = "D")
plot!([0],[0], color = :red,label = "SIM")
