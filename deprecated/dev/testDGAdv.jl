include("../src/SFFM.jl")
using LinearAlgebra, Plots, JLD2, SparseArrays

include("../examples/paperNumerics/exampleModelDef.jl")

@load pwd()*"/examples/paperNumerics/dump/sims.jld2" sims

Δ = 1.6
Nodes = collect(approxBounds[1, 1]:Δ:approxBounds[1, 2]) # collect(approxBounds[1, 1]:Δ:approxBounds[1, 2])
for NBases in 1:5
# NBases = 2
Basis = "lagrange"
mesh = SFFM.MakeMesh(
    model = approxModel,
    Nodes = Nodes,
    NBases = NBases,
    Basis = Basis,
)

simprobs = SFFM.Sims2Dist(model=simModel,mesh=mesh,sims=sims,type="probability")
whichphase = 4
p2 = plot!(
    mesh.CellNodes[[1;end],:],# [1;1]*simprobs.x',
    [1;1]*simprobs.distribution[:,:,whichphase],
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

All = SFFM.MakeAll(model = approxModel, mesh = mesh, approxType = "projection")
Matrices = SFFM.MakeMatrices(model=approxModel,mesh=mesh,probTransform=false)
MatricesR = SFFM.MakeMatricesR(model=approxModel,mesh=mesh)
B = SFFM.MakeB(model=approxModel,mesh=mesh,Matrices=Matrices,probTransform=false)
Dr = SFFM.MakeDR(
    Matrices=Matrices,
    MatricesR=MatricesR,
    model=approxModel,
    mesh=mesh,
    B=B,
)

Ψ = SFFM.PsiFun(D=Dr.DDict)
Ψbase = SFFM.PsiFun(D=All.D)

# construct initial condition
theNodes = mesh.CellNodes[:,convert(Int,ceil(5/Δ))]
basisValues = zeros(length(theNodes))
for n in 1:length(theNodes)
    basisValues[n] = prod(5.0.-theNodes[[1:n-1;n+1:end]])./prod(theNodes[n].-theNodes[[1:n-1;n+1:end]])
end
V = SFFM.vandermonde(NBases=NBases)
if mesh.Basis=="lagrange"
    basisValues = basisValues'*V.V*V.V'*2/Δ
else
    basisValues = basisValues./V.w
    basisValues = basisValues'*V.inv'*2/Δ
end
initpm = [
    zeros(sum(approxModel.C.<=0)) # LHS point mass
    zeros(sum(approxModel.C.>=0)) # RHS point mass
]
initprobs = zeros(Float64,mesh.NBases,mesh.NIntervals,approxModel.NPhases)
initprobs[:,convert(Int,ceil(5/Δ)),3] = basisValues #*All.Matrices.Local.V.V*All.Matrices.Local.V.V'.*2/Δ
initprobsBase = zeros(Float64,mesh.NBases,mesh.NIntervals,approxModel.NPhases)
initprobsBase[:,convert(Int,ceil(5/Δ)),3] = basisValues.*V.w'/2*Δ
# initprobs[1,convert(Int,ceil(5/Δ)),3] = sqrt(2)
initdist = (
    pm = initpm,
    distribution = initprobs,
    x = mesh.CellNodes,
    type = "cumulative"
) # convert to a distribution object so we can apply Dist2Coeffs
# convert to Coeffs α in the DG context
x0 = [0;0;initdist.distribution[:];0;0]' # SFFM.Dist2Coeffs(model = approxModel, mesh = mesh, Distn = initdist) # #
x0Base = [0;0;initprobsBase[:];0;0]' # SFFM.Dist2Coeffs(model = approxModel, mesh = mesh, Distn = initdist) # #
# the initial condition on Ψ is restricted to + states so find the + states
plusIdx = [
    mesh.Fil["p+"];
    repeat(mesh.Fil["+"]', mesh.NBases, 1)[:];
    mesh.Fil["q+"];
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
    mesh.Fil["p-"];
    repeat(mesh.Fil["-"]', mesh.NBases, 1)[:];
    mesh.Fil["q-"];
]
# then map to the whole state space for plotting
z = zeros(
    Float64,
    mesh.NBases*mesh.NIntervals * approxModel.NPhases +
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
    model = approxModel,
    mesh = mesh,
    Coeffs = z,
    type="probability",
)
DGProbsBase = SFFM.Coeffs2Dist(
    model = approxModel,
    mesh = mesh,
    Coeffs = zbase,
    type="probability",
)
display(DGProbsBase.pm)
display(DGProbs.pm)
display(simprobs.pm)
plot!(p2, DGProbsBase.x[:], DGProbsBase.distribution[:,:,whichphase][:],label = :false,#"DG: N_k = "*string(NBases),
    color = :blue,
    xlims = (0,2),
    seriestype = :scatter
    )
1
# plot them
if Basis=="lagrange"
    if mesh.NBases==1
        p2 = plot!(p2,
            DGProbs.x,
            [1;1]*reshape(z[3:end-2], mesh.NBases, mesh.NIntervals, approxModel.NPhases)[:,:,whichphase], #DGProbs.distribution[:,:,whichphase],
            label = :false,#"DG: N_k = "*string(NBases),
            color = :blue,
            xlims = (0,2),
            seriestype = :line,
            linestyle = :dash,
        )
        p2 = plot!(p2,
            DGProbsBase.x,
            [1;1]*reshape(zbase[3:end-2], mesh.NBases, mesh.NIntervals, approxModel.NPhases)[:,:,whichphase], #DGProbsBase.distribution[:,:,whichphase],
            label = :false,#"DG: N_k = "*string(NBases),
            color = :black,
            xlims = (0,2),
            seriestype = :line,
            linestyle = :dot,
        )
    else
        p2 = plot!(p2,
            mesh.CellNodes[[1;end],:],
            [1;1]*0.5*Δ*V.w'*reshape(z[3:end-2], mesh.NBases, mesh.NIntervals, approxModel.NPhases)[:,:,whichphase], #DGProbs.distribution[:,:,whichphase],
            label = :false,#"DG: N_k = "*string(NBases),
            color = :blue,
            xlims = (0,2),
            seriestype = :line,
            linestyle = :dash,
        )
        p2 = plot!(p2,
            mesh.CellNodes[[1;end],:],
            [1;1]*ones(1,NBases)*reshape(zbase[3:end-2], mesh.NBases, mesh.NIntervals, approxModel.NPhases)[:,:,whichphase], #DGProbs.distribution[:,:,whichphase],
            label = :false,#"DG: N_k = "*string(NBases),
            color = :black,
            xlims = (0,2),
            seriestype = :line,
            linestyle = :dot,
        )
    end
else
    p2 = plot!(p2,
        [mesh.Nodes[1:end-1]';mesh.Nodes[2:end]'],
        [1;1]*DGProbs.distribution[:,:,whichphase],
        label = :false,#"DG: N_k = "*string(NBases),
        color = :blue,
        xlims = (0,2),
        seriestype = :line,
        linestyle = :dash,
    )
    p2 = plot!(p2,
        [mesh.Nodes[1:end-1]';mesh.Nodes[2:end]'],
        [1;1]*DGProbsBase.distribution[:,:,whichphase],
        label = :false,#"DG: N_k = "*string(NBases),
        color = :black,
        xlims = (0,2),
        seriestype = :line,
        linestyle = :dot,
    )
end
d = zeros(1, mesh.NIntervals, approxModel.NPhases)
for i in 1:approxModel.NPhases
    d[:,:,i] = 0.5*Δ*V.w'*reshape(z[3:end-2], mesh.NBases, mesh.NIntervals, approxModel.NPhases)[:,:,i]
end
DGProbs = (pm = DGProbs.pm,
    distribution = d,
    x = DGProbs.x,
    type = "probability")
eR = SFFM.starSeminorm(d1 = DGProbs, d2 = simprobs)
e = SFFM.starSeminorm(d1 = DGProbsBase, d2 = simprobs)
ediff = SFFM.starSeminorm(d1 = DGProbsBase, d2 = DGProbs)
println("DR: ",eR)
println("D: ",e)
println("diff: ",ediff)
end
1
# p = SFFM.PlotSFM(model=approxModel,mesh=mesh,Dist=DGProbs)
# SFFM.PlotSFM!(p;model=approxModel,mesh=mesh,Dist=simprobs,color=2)
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
let
T = [-1.0 1.0; 1.0 -1.0]*0.5
C = [1.0; -2]
r = (
    r = function (x)
        [(cos.(x).+1.05)./(2.05) -3*(x.>-1)]
    end,
    R = function (x)
        [(sin.(x).+1.05.*x)./(2.05) -3*(x.>-1).*x]
    end,
)

Bounds = [0 10; -Inf Inf]
model = SFFM.Model(T = T, C = C, r = r, Bounds = Bounds)


## Define mesh

Δ = 1
Nodes = collect(Bounds[1,1]:Δ:Bounds[1,2])
# NBases = 3
for NBases = 1:5
Basis = "lagrange"
mesh = SFFM.MakeMesh(model = model, Nodes = Nodes, NBases = NBases, Basis=Basis)

## matrices
All = SFFM.MakeAll(model = model, mesh = mesh, approxType = "projection")
Matrices = All.Matrices
MatricesR = SFFM.MakeMatricesR(model=model,mesh=mesh)
Matrices2 = SFFM.MakeMatrices2(model=model,mesh=mesh)#,probTransform=false)
B = SFFM.MakeB(model=model,mesh=mesh,Matrices=Matrices2,probTransform=false)
Dr = SFFM.MakeDR(
    Matrices=Matrices2,
    MatricesR=MatricesR,
    model=model,
    mesh=mesh,
    B=B,
)
## sims for gound truth
x₀ = 1.21
NSim = 50000
IC = (φ = ones(Int, NSim), X = x₀ .* ones(NSim), Y = zeros(NSim))
y = 10
Ψr = SFFM.PsiFun(D=Dr.DDict,s=0)
Ψ = SFFM.PsiFun(D=All.D,s=0)
simBounds = [0 Inf; -Inf Inf]
simModel = SFFM.Model(T = T, C = C, r = r, Bounds = simBounds)

@time sims =
    SFFM.SimSFFM(model = simModel, StoppingTime = SFFM.FirstExitY(u = 0, v = Inf), InitCondition = IC)

simdensity = SFFM.Sims2Dist(model=simModel,mesh=mesh,sims=sims,type="density")
# @time sims =
    # SFFM.SimSFFM(model = model, StoppingTime = SFFM.FirstExitY(u = -Inf, v = y), InitCondition = IC)

## DG stationary dist
# construct initial condition
theNodes = mesh.CellNodes[:,convert(Int,ceil(x₀/Δ))]
basisValues = zeros(length(theNodes))
for n in 1:length(theNodes)
    basisValues[n] = prod(x₀.-theNodes[[1:n-1;n+1:end]])./prod(theNodes[n].-theNodes[[1:n-1;n+1:end]])
end
V = SFFM.vandermonde(NBases=NBases)
if mesh.Basis=="lagrange"
    basisValues = basisValues'*V.V*V.V'*2/Δ
else
    basisValues = basisValues./V.w
    basisValues = basisValues'*V.inv'*2/Δ
end
initpm = [
    zeros(sum(model.C.<=0)) # LHS point mass
    zeros(sum(model.C.>=0)) # RHS point mass
]
initprobs = zeros(Float64,mesh.NBases,mesh.NIntervals,model.NPhases)
initprobs[:,convert(Int,max(1,ceil(x₀/Δ))),1] = basisValues #*All.Matrices.Local.V.V*All.Matrices.Local.V.V'.*2/Δ
initprobsBase = zeros(Float64,mesh.NBases,mesh.NIntervals,model.NPhases)
initprobsBase[:,convert(Int,max(1,ceil(x₀/Δ))),1] = basisValues.*V.w'/2*Δ
# initprobs[1,convert(Int,ceil(5/Δ)),3] = sqrt(2)
initdist = (
    pm = initpm,
    distribution = initprobs,
    x = mesh.CellNodes,
    type = "cumulative"
) # convert to a distribution object so we can apply Dist2Coeffs
# convert to Coeffs α in the DG context
x0 = [0;initdist.distribution[:];0]' # SFFM.Dist2Coeffs(model = approxModel, mesh = mesh, Distn = initdist) # #
x0Base = [0;initprobsBase[:];0]' # SFFM.Dist2Coeffs(model = approxModel, mesh = mesh, Distn = initdist) # #
# the initial condition on Ψ is restricted to + states so find the + states
plusIdx = [
    mesh.Fil["p+"];
    repeat(mesh.Fil["+"]', mesh.NBases, 1)[:];
    mesh.Fil["q+"];
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
    mesh.Fil["p-"];
    repeat(mesh.Fil["-"]', mesh.NBases, 1)[:];
    mesh.Fil["q-"];
]
# then map to the whole state space for plotting
z = zeros(
    Float64,
    mesh.NBases*mesh.NIntervals * model.NPhases +
        sum(model.C.<=0) + sum(model.C.>=0)
)
zbase = copy(z)
z[minusIdx] = w
zbase[minusIdx] = wbase
# check that it is equal to 1
println(sum(z))
println(sum(zbase))

DRdensity = SFFM.Coeffs2Dist(model=model,mesh=mesh,Coeffs=z,type="density")
Ddensity = SFFM.Coeffs2Dist(model=model,mesh=mesh,Coeffs=zbase,type="density")

# p = SFFM.PlotSFM(model=model,mesh=mesh,Dist=simdensity)
# p = SFFM.PlotSFM!(p;model=model,mesh=mesh,Dist=DRdensity,color=:red)
# p = SFFM.PlotSFM!(p;model=model,mesh=mesh,Dist=Ddensity,color=:black)

simprobs = SFFM.Sims2Dist(model=model,mesh=mesh,sims=sims,type="probability")
DRprobs = SFFM.Coeffs2Dist(model=model,mesh=mesh,Coeffs=z,type="probability")
Dprobs = SFFM.Coeffs2Dist(model=model,mesh=mesh,Coeffs=zbase,type="probability")

display(simprobs.pm)
display(DRprobs.pm)
display(Dprobs.pm)
d = zeros(1, mesh.NIntervals, model.NPhases)
for i in 1:model.NPhases
    d[:,:,i] = 0.5*Δ*V.w'*reshape(z[2:end-1], mesh.NBases, mesh.NIntervals, model.NPhases)[:,:,i]
end
DGProbs = (pm = DRprobs.pm,
    distribution = d,
    x = DRprobs.x,
    type = "probability")
eR = SFFM.starSeminorm(d1 = DGProbs, d2 = simprobs)
e = SFFM.starSeminorm(d1 = Dprobs, d2 = simprobs)
ediff = SFFM.starSeminorm(d1 = DGProbs, d2 = Dprobs)
# plot(DRprobs.x[:], DRprobs.distribution[:,:,2][:],label="DR")
xvals = [mesh.Nodes[1:end-1]'; mesh.Nodes[2:end]']
plot(xvals, [1;1]*Dprobs.distribution[:,:,2],label=false, color=:black,
    seriestype = :line,
)
plot!(xvals, [1;1]*simprobs.distribution[:,:,2],label=false, color=:red)

if Basis=="lagrange"
    if mesh.NBases==1
        p2 = plot!(
            xvals,
            [1;1]*0.5*Δ*2*reshape(z[2:end-1], mesh.NBases, mesh.NIntervals, model.NPhases)[:,:,2], #DGProbs.distribution[:,:,2],
            label = :false,#"DG: N_k = "*string(NBases),
            color = :blue,
            xlims = (0,10),
            seriestype = :line,
            linestyle = :dash,
        )
    else
        p2 = plot!(
            xvals,
            [1;1]*0.5*Δ*V.w'*reshape(z[2:end-1], mesh.NBases, mesh.NIntervals, model.NPhases)[:,:,2], #DGProbs.distribution[:,:,2],
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

println("DR: ", eR)
println("D: ", e)
println("diff: ", ediff)

plot!([0],[0], color = :blue, label = "DR")
plot!([0],[0], color = :black,label = "D")
plot!([0],[0], color = :red,label = "SIM")
end

end
