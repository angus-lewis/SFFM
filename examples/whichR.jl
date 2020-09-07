# Toy model from paper
# include("../src/SFFM.jl")
# using LinearAlgebra, Plots

## Define 1st model
T = [-1.0 1.0; 1.0 -1.0]
C = [1.0; -1.0]
r = (
    r = function (x)
        [abs.((x .<= 1)-(x .> 1)) abs.(((x .> 1))-2*(x .<= 1).*(x .> 0)-(x.==0))]
    end,
    R = function (x)
        [x.*abs.(((x .<= 1)-(x .> 1))) x.*abs.((((x .> 1))-2*(x .<= 1).*(x .> 0)-(x.==0)))]
    end,
)

Bounds = [0 30; -Inf Inf]
Model = SFFM.MakeModel(T = T, C = C, r = r, Bounds = Bounds)

## Define mesh
Δ = 5
Nodes = collect(Bounds[1,1]:Δ:Bounds[1,2])
NBases = 3
Basis = "lagrange"
Mesh = SFFM.MakeMesh(Model = Model, Nodes = Nodes, NBases = NBases, Basis=Basis)

## Make matrices
All = SFFM.MakeAll(Model = Model, Mesh = Mesh, approxType = "projection")
proj = All
interp = SFFM.MakeAll(Model=Model,Mesh=Mesh,approxType="interpolation")

vikramD = copy(interp.D["++"]())
vikramD[sum(Model.C.<=0)+1:end-sum(Model.C.>=0),:] =
    repeat(All.Matrices.Local.V.w, Mesh.NIntervals*Model.NPhases, 1) .* vikramD[sum(Model.C.<=0)+1:end-sum(Model.C.>=0),:]

## sims fro gound truth
x₀ = 15
NSim = 50000
IC = (φ = ones(Int, NSim), X = x₀ .* ones(NSim), Y = zeros(NSim))
y = 10
@time sims =
    SFFM.SimSFFM(Model = Model, StoppingTime = SFFM.FirstExitY(u = -Inf, v = y), InitCondition = IC)

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
h = 0.0001
@time projyvals = SFFM.EulerDG(D = proj.D["++"](s = 0), y = y, x0 = x0, h = h)
@time interpyvals = SFFM.EulerDG(D = interp.D["++"](s = 0), y = y, x0 = x0, h = h)
# @time vikramyvals = SFFM.EulerDG(D = vikramD, y = y, x0 = x0, h = h)

## plotting
simdensity = SFFM.Sims2Dist(Model=Model,Mesh=Mesh,sims=sims,type="probability")
projdensity = SFFM.Coeffs2Dist(Model=Model,Mesh=Mesh,Coeffs=projyvals,type="probability")
interpdensity = SFFM.Coeffs2Dist(Model=Model,Mesh=Mesh,Coeffs=interpyvals,type="probability")
# vikramdensity = SFFM.Coeffs2Dist(Model=Model,Mesh=Mesh,Coeffs=vikramyvals,type="density")
println("proj error: ",SFFM.starSeminorm(d1 = projdensity, d2 = simdensity))
println("interp error: ",SFFM.starSeminorm(d1 = interpdensity, d2 = simdensity))

## plots
# plot solutions
# densities
p = SFFM.PlotSFM(Model=Model,Mesh=Mesh,Dist=projdensity,color=:blue,label="proj")#,marker=:rtriangle)
p = SFFM.PlotSFM!(p;Model=Model,Mesh=Mesh,Dist=interpdensity,color=:red,label="interp")#,marker=:x)
# p = SFFM.PlotSFM!(p;Model=Model,Mesh=Mesh,Dist=vikramdensity,color=3,label="vikram")
# plot sims
p = SFFM.PlotSFM!(p;Model=Model,Mesh=Mesh,Dist=simdensity,color=:black,label="sim")#,marker=:ltriangle)

display(p)
## probabilities
simdensity = SFFM.Sims2Dist(Model=Model,Mesh=Mesh,sims=sims,type="density")
projdensity = SFFM.Coeffs2Dist(Model=Model,Mesh=Mesh,Coeffs=projyvals,type="density")
interpdensity = SFFM.Coeffs2Dist(Model=Model,Mesh=Mesh,Coeffs=interpyvals,type="density")
# vikramdensity = SFFM.Coeffs2Dist(Model=Model,Mesh=Mesh,Coeffs=vikramyvals,type="density")

## plots
# plot solutions
# densities
p = SFFM.PlotSFM(Model=Model,Mesh=Mesh,Dist=projdensity,color=:blue,label="proj")#,marker=:rtriangle)
p = SFFM.PlotSFM!(p;Model=Model,Mesh=Mesh,Dist=interpdensity,color=:red,label="interp")#,marker=:x)
# p = SFFM.PlotSFM!(p;Model=Model,Mesh=Mesh,Dist=vikramdensity,color=3,label="vikram")
# plot sims
p = SFFM.PlotSFM!(p;Model=Model,Mesh=Mesh,Dist=simdensity,color=:black,label="sim")#,marker=:ltriangle)
savefig(p,"/Users/a1627293/Dropbox/PhD/NotesMaster/whichR/m1.png")
display(p)
## end 1st model


########################
########################
########################


## Define 2nd model
T = [-1.0 1.0; 1.0 -1.0]
C = [1.0; -1.0]
r = (
    r = function (x)
        [0.1.*x.+0.05 (x.>-1)]
    end,
    R = function (x)
        [0.1.*x.^2/2.0.+0.05.*x (x.>-1).*x]
    end,
)

Bounds = [0 30; -Inf Inf]
Model = SFFM.MakeModel(T = T, C = C, r = r, Bounds = Bounds)

## Define mesh
# Δ = 3
Nodes = collect(Bounds[1,1]:Δ:Bounds[1,2])
# NBases = 3
Basis = "lagrange"
Mesh = SFFM.MakeMesh(Model = Model, Nodes = Nodes, NBases = NBases, Basis=Basis)

## Make matrices
All = SFFM.MakeAll(Model = Model, Mesh = Mesh, approxType = "projection")
proj = All
interp = SFFM.MakeAll(Model=Model,Mesh=Mesh,approxType="interpolation")

vikramD = copy(interp.D["++"]())
vikramD[sum(Model.C.<=0)+1:end-sum(Model.C.>=0),:] =
    repeat(All.Matrices.Local.V.w, Mesh.NIntervals*Model.NPhases, 1) .* vikramD[sum(Model.C.<=0)+1:end-sum(Model.C.>=0),:]

## sims fro gound truth
x₀ = 15
NSim = 50000
IC = (φ = ones(Int, NSim), X = x₀ .* ones(NSim), Y = zeros(NSim))
y = 10
@time sims =
    SFFM.SimSFFM(Model = Model, StoppingTime = SFFM.FirstExitY(u = -Inf, v = y), InitCondition = IC)

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
h = 0.0001
@time projyvals = SFFM.EulerDG(D = proj.D["++"](s = 0), y = y, x0 = x0, h = h)
@time interpyvals = SFFM.EulerDG(D = interp.D["++"](s = 0), y = y, x0 = x0, h = h)
# @time vikramyvals = SFFM.EulerDG(D = vikramD, y = y, x0 = x0, h = h)

## plotting
simdensity = SFFM.Sims2Dist(Model=Model,Mesh=Mesh,sims=sims,type="probability")
projdensity = SFFM.Coeffs2Dist(Model=Model,Mesh=Mesh,Coeffs=projyvals,type="probability")
interpdensity = SFFM.Coeffs2Dist(Model=Model,Mesh=Mesh,Coeffs=interpyvals,type="probability")
# vikramdensity = SFFM.Coeffs2Dist(Model=Model,Mesh=Mesh,Coeffs=vikramyvals,type="density")
println("proj error: ",SFFM.starSeminorm(d1 = projdensity, d2 = simdensity))
println("interp error: ",SFFM.starSeminorm(d1 = interpdensity, d2 = simdensity))
## plots
# plot solutions
# densities
# plot sims
p = SFFM.PlotSFM(Model=Model,Mesh=Mesh,Dist=simdensity,color=:black,label="sim")#,marker=:ltriangle)
p = SFFM.PlotSFM!(p;Model=Model,Mesh=Mesh,Dist=projdensity,color=:blue,label="proj")#,marker=:rtriangle)
p = SFFM.PlotSFM!(p;Model=Model,Mesh=Mesh,Dist=interpdensity,color=:red,label="interp")#,marker=:x)

display(p)
## probabilities
simdensity = SFFM.Sims2Dist(Model=Model,Mesh=Mesh,sims=sims,type="density")
projdensity = SFFM.Coeffs2Dist(Model=Model,Mesh=Mesh,Coeffs=projyvals,type="density")
interpdensity = SFFM.Coeffs2Dist(Model=Model,Mesh=Mesh,Coeffs=interpyvals,type="density")
# vikramdensity = SFFM.Coeffs2Dist(Model=Model,Mesh=Mesh,Coeffs=vikramyvals,type="density")

## plots
# plot solutions
# densities
p = SFFM.PlotSFM(Model=Model,Mesh=Mesh,Dist=projdensity,color=:blue,label="proj")#,marker=:rtriangle)
p = SFFM.PlotSFM!(p;Model=Model,Mesh=Mesh,Dist=interpdensity,color=:red,label="interp")#,marker=:x)
# p = SFFM.PlotSFM!(p;Model=Model,Mesh=Mesh,Dist=vikramdensity,color=3,label="vikram")
# plot sims
p = SFFM.PlotSFM!(p;Model=Model,Mesh=Mesh,Dist=simdensity,color=:black,label="sim")#,marker=:ltriangle)
savefig(p,"/Users/a1627293/Dropbox/PhD/NotesMaster/whichR/m2.png")
display(p)
## end 2nd model

########################
########################
########################


## Define 3rd model
T = [-1.0 1.0; 1.0 -1.0]
C = [1.0; -1.0]
r = (
    r = function (x)
        [cos.(x).+1.05 x.>-1]
    end,
    R = function (x)
        [sin.(x).+1.05.*x (x.>-1).*x]
    end,
)

Bounds = [0 10*π; -Inf Inf]
Model = SFFM.MakeModel(T = T, C = C, r = r, Bounds = Bounds)

## Define mesh
# Δ = 3
Nodes = collect(Bounds[1,1]:Δ:Bounds[1,2])
# NBases = 3
Basis = "lagrange"
Mesh = SFFM.MakeMesh(Model = Model, Nodes = Nodes, NBases = NBases, Basis=Basis)

## Make matrices
All = SFFM.MakeAll(Model = Model, Mesh = Mesh, approxType = "projection")
proj = All
interp = SFFM.MakeAll(Model=Model,Mesh=Mesh,approxType="interpolation")

vikramD = copy(interp.D["++"]())
vikramD[sum(Model.C.<=0)+1:end-sum(Model.C.>=0),:] =
    repeat(All.Matrices.Local.V.w, Mesh.NIntervals*Model.NPhases, 1) .* vikramD[sum(Model.C.<=0)+1:end-sum(Model.C.>=0),:]

## sims fro gound truth
x₀ = 15
NSim = 50000
IC = (φ = ones(Int, NSim), X = x₀ .* ones(NSim), Y = zeros(NSim))
y = 10
@time sims =
    SFFM.SimSFFM(Model = Model, StoppingTime = SFFM.FirstExitY(u = -Inf, v = y), InitCondition = IC)

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
h = 0.0001
@time projyvals = SFFM.EulerDG(D = proj.D["++"](s = 0), y = y, x0 = x0, h = h)
@time interpyvals = SFFM.EulerDG(D = interp.D["++"](s = 0), y = y, x0 = x0, h = h)
# @time vikramyvals = SFFM.EulerDG(D = vikramD, y = y, x0 = x0, h = h)

## plotting
simdensity = SFFM.Sims2Dist(Model=Model,Mesh=Mesh,sims=sims,type="probability")
projdensity = SFFM.Coeffs2Dist(Model=Model,Mesh=Mesh,Coeffs=projyvals,type="probability")
interpdensity = SFFM.Coeffs2Dist(Model=Model,Mesh=Mesh,Coeffs=interpyvals,type="probability")
# vikramdensity = SFFM.Coeffs2Dist(Model=Model,Mesh=Mesh,Coeffs=vikramyvals,type="density")
println("proj error: ",SFFM.starSeminorm(d1 = projdensity, d2 = simdensity))
println("interp error: ",SFFM.starSeminorm(d1 = interpdensity, d2 = simdensity))

## plots
# plot solutions
# densities
p = SFFM.PlotSFM(Model=Model,Mesh=Mesh,Dist=projdensity,color=:blue,label="proj")#,marker=:rtriangle)
p = SFFM.PlotSFM!(p;Model=Model,Mesh=Mesh,Dist=interpdensity,color=:red,label="interp")#,marker=:x)
# p = SFFM.PlotSFM!(p;Model=Model,Mesh=Mesh,Dist=vikramdensity,color=3,label="vikram")
# plot sims
p = SFFM.PlotSFM!(p;Model=Model,Mesh=Mesh,Dist=simdensity,color=:black,label="sim")#,marker=:ltriangle)

display(p)
## probabilities
simdensity = SFFM.Sims2Dist(Model=Model,Mesh=Mesh,sims=sims,type="density")
projdensity = SFFM.Coeffs2Dist(Model=Model,Mesh=Mesh,Coeffs=projyvals,type="density")
interpdensity = SFFM.Coeffs2Dist(Model=Model,Mesh=Mesh,Coeffs=interpyvals,type="density")
# vikramdensity = SFFM.Coeffs2Dist(Model=Model,Mesh=Mesh,Coeffs=vikramyvals,type="density")

## plots
# plot solutions
# densities
p = SFFM.PlotSFM(Model=Model,Mesh=Mesh,Dist=projdensity,color=:blue,label="proj")#,marker=:rtriangle)
p = SFFM.PlotSFM!(p;Model=Model,Mesh=Mesh,Dist=interpdensity,color=:red,label="interp")#,marker=:x)
# p = SFFM.PlotSFM!(p;Model=Model,Mesh=Mesh,Dist=vikramdensity,color=3,label="vikram")
# plot sims
p = SFFM.PlotSFM!(p;Model=Model,Mesh=Mesh,Dist=simdensity,color=:black,label="sim")#,marker=:ltriangle)
savefig(p,"/Users/a1627293/Dropbox/PhD/NotesMaster/whichR/m3.png")
display(p)
