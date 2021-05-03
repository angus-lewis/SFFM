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
model = SFFM.Model( T, C, r, Bounds = Bounds)

## Define mesh
Δ = 5
Nodes = collect(Bounds[1,1]:Δ:Bounds[1,2])
nBases = 3
Basis = "lagrange"
mesh = SFFM.DGMesh(model, Nodes, nBases, Basis=Basis)

## Make matrices
All = SFFM.MakeAll( model, mesh, approxType = "projection")
proj = All
interp = SFFM.MakeAll(model,mesh,approxType="interpolation")

vikramD = copy(interp.D["++"]())
vikramD[sum(model.C.<=0)+1:end-sum(model.C.>=0),:] =
    repeat(All.Matrices.Local.V.w, SFFM.NIntervals(mesh)*SFFM.NPhases(model), 1) .* vikramD[sum(model.C.<=0)+1:end-sum(model.C.>=0),:]

## sims fro gound truth
x₀ = 15
NSim = 50000
IC = (φ = ones(Int, NSim), X = x₀ .* ones(NSim), Y = zeros(NSim))
y = 10
@time sims =
    SFFM.SimSFFM( model, SFFM.FirstExitY(-Inf, y), IC)

## DG stationary dist
# construct initial condition
theNodes = SFFM.CellNodes(mesh)[:,convert(Int,ceil(x₀/Δ))]
basisValues = zeros(length(theNodes))
for n in 1:length(theNodes)
    basisValues[n] = prod(x₀.-theNodes[[1:n-1;n+1:end]])./prod(theNodes[n].-theNodes[[1:n-1;n+1:end]])
end
initpm = [
    zeros(sum(model.C.<=0)) # LHS point mass
    zeros(sum(model.C.>=0)) # RHS point mass
]
initprobs = zeros(Float64,SFFM.NBases(mesh),SFFM.NIntervals(mesh),SFFM.NPhases(model))
initprobs[:,convert(Int,ceil(x₀/Δ)),1] = basisValues'*All.Matrices.Local.V.V*All.Matrices.Local.V.V'.*2/Δ
initdist = (
    pm = initpm,
    distribution = initprobs,
    x = SFFM.CellNodes(mesh),
    type = "density"
)
x0 = SFFM.Dist2Coeffs( model, mesh, initdist)
h = 0.0001
@time projyvals = SFFM.EulerDG( proj.D["++"](s = 0), y, x0, h = h)
@time interpyvals = SFFM.EulerDG( interp.D["++"](s = 0), y, x0, h = h)

## plotting
simdensity = SFFM.Sims2Dist(model,mesh,sims,type="probability")
projdensity = SFFM.Coeffs2Dist(model,mesh,projyvals,type="probability")
interpdensity = SFFM.Coeffs2Dist(model,mesh,interpyvals,type="probability")
println("proj error: ",SFFM.starSeminorm( projdensity, simdensity))
println("interp error: ",SFFM.starSeminorm( interpdensity, simdensity))

## plots
# plot solutions
# densities
p = SFFM.PlotSFM(model; mesh = mesh,dist = projdensity,color=:blue,label="proj")#,marker=:rtriangle)
p = SFFM.PlotSFM!(p,model,mesh,interpdensity,color=:red,label="interp")#,marker=:x)
# plot sims
p = SFFM.PlotSFM!(p,model,mesh,simdensity,color=:black,label="sim")#,marker=:ltriangle)

display(p)
## probabilities
simdensity = SFFM.Sims2Dist(model,mesh,sims,type="density")
projdensity = SFFM.Coeffs2Dist(model,mesh,projyvals,type="density")
interpdensity = SFFM.Coeffs2Dist(model,mesh,interpyvals,type="density")
# vikramdensity = SFFM.Coeffs2Dist(model=model,mesh=mesh,Coeffs=vikramyvals,type="ensity")

## plots
# plot solutions
# densities
p = SFFM.PlotSFM(model,mesh=mesh,dist=projdensity,color=:blue,label="proj")#,marker=:rtriangle)
p = SFFM.PlotSFM!(p,model,mesh,interpdensity,color=:red,label="interp")#,marker=:x)
# plot sims
p = SFFM.PlotSFM!(p,model,mesh,simdensity,color=:black,label="sim")#,marker=:ltriangle)
# savefig(p,"/Users/a1627293/Dropbox/PhD/NotesMaster/whichR/m1.png")
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
model = SFFM.Model( T, C, r, Bounds = Bounds)

## Define mesh
# Δ = 3
Nodes = collect(Bounds[1,1]:Δ:Bounds[1,2])
# nBases = 3
Basis = "lagrange"
mesh = SFFM.DGMesh(model, Nodes, nBases, Basis=Basis)

## Make matrices
All = SFFM.MakeAll( model, mesh, approxType = "projection")
proj = All
interp = SFFM.MakeAll( model, mesh, approxType="interpolation")

vikramD = copy(interp.D["++"]())
vikramD[sum(model.C.<=0)+1:end-sum(model.C.>=0),:] =
    repeat(All.Matrices.Local.V.w, SFFM.NIntervals(mesh)*SFFM.NPhases(model), 1) .* vikramD[sum(model.C.<=0)+1:end-sum(model.C.>=0),:]

## sims fro gound truth
x₀ = 15
NSim = 50000
IC = (φ = ones(Int, NSim), X = x₀ .* ones(NSim), Y = zeros(NSim))
y = 10
@time sims =
    SFFM.SimSFFM( model, SFFM.FirstExitY(-Inf, y), IC)

## DG stationary dist
# construct initial condition
theNodes = SFFM.CellNodes(mesh)[:,convert(Int,ceil(x₀/Δ))]
basisValues = zeros(length(theNodes))
for n in 1:length(theNodes)
    basisValues[n] = prod(x₀.-theNodes[[1:n-1;n+1:end]])./prod(theNodes[n].-theNodes[[1:n-1;n+1:end]])
end
initpm = [
    zeros(sum(model.C.<=0)) # LHS point mass
    zeros(sum(model.C.>=0)) # RHS point mass
]
initprobs = zeros(Float64,SFFM.NBases(mesh),SFFM.NIntervals(mesh),SFFM.NPhases(model))
initprobs[:,convert(Int,ceil(x₀/Δ)),1] = basisValues'*All.Matrices.Local.V.V*All.Matrices.Local.V.V'.*2/Δ
initdist = (
    pm = initpm,
    distribution = initprobs,
    x = SFFM.CellNodes(mesh),
    type = "density"
)
x0 = SFFM.Dist2Coeffs( model, mesh, initdist)
h = 0.0001
@time projyvals = SFFM.EulerDG(proj.D["++"](s = 0), y, x0, h = h)
@time interpyvals = SFFM.EulerDG(interp.D["++"](s = 0), y, x0, h = h)


## plotting
simdensity = SFFM.Sims2Dist( model, mesh, sims,type="probability")
projdensity = SFFM.Coeffs2Dist( model, mesh, projyvals,type="probability")
interpdensity = SFFM.Coeffs2Dist( model, mesh, interpyvals,type="probability")

println("proj error: ",SFFM.starSeminorm( projdensity, simdensity))
println("interp error: ",SFFM.starSeminorm(interpdensity, simdensity))
## plots
# plot solutions
# densities
# plot sims
p = SFFM.PlotSFM( model, mesh = mesh, dist = simdensity, color=:black,label="sim")#,marker=:ltriangle)
p = SFFM.PlotSFM!(p, model, mesh, projdensity, color=:blue,label="proj")#,marker=:rtriangle)
p = SFFM.PlotSFM!(p, model, mesh, interpdensity, color=:red,label="interp")#,marker=:x)

display(p)
## probabilities
simdensity = SFFM.Sims2Dist( model, mesh, sims,type="density")
projdensity = SFFM.Coeffs2Dist( model, mesh, projyvals,type="density")
interpdensity = SFFM.Coeffs2Dist( model, mesh, interpyvals,type="density")

## plots
# plot solutions
# densities
p = SFFM.PlotSFM( model, mesh = mesh, dist = projdensity, color=:blue,label="proj")#,marker=:rtriangle)
p = SFFM.PlotSFM!(p, model, mesh, interpdensity, color=:red,label="interp")#,marker=:x)

# plot sims
p = SFFM.PlotSFM!(p, model, mesh, simdensity, color=:black,label="sim")#,marker=:ltriangle)
# savefig(p,"/Users/a1627293/Dropbox/PhD/NotesMaster/whichR/m2.png")
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
model = SFFM.Model( T, C, r, Bounds = Bounds)

## Define mesh
# Δ = 3
Nodes = collect(Bounds[1,1]:Δ:Bounds[1,2])
# nBases = 3
Basis = "lagrange"
mesh = SFFM.DGMesh(model, Nodes, nBases, Basis=Basis)

## Make matrices
All = SFFM.MakeAll( model, mesh, approxType = "projection")
proj = All
interp = SFFM.MakeAll( model, mesh, approxType="interpolation")

vikramD = copy(interp.D["++"]())
vikramD[sum(model.C.<=0)+1:end-sum(model.C.>=0),:] =
    repeat(All.Matrices.Local.V.w, SFFM.NIntervals(mesh)*SFFM.NPhases(model), 1) .* vikramD[sum(model.C.<=0)+1:end-sum(model.C.>=0),:]

## sims fro gound truth
x₀ = 15
NSim = 50000
IC = (φ = ones(Int, NSim), X = x₀ .* ones(NSim), Y = zeros(NSim))
y = 10
@time sims =
    SFFM.SimSFFM( model, SFFM.FirstExitY( -Inf, y), IC)

## DG stationary dist
# construct initial condition
theNodes = SFFM.CellNodes(mesh)[:,convert(Int,ceil(x₀/Δ))]
basisValues = zeros(length(theNodes))
for n in 1:length(theNodes)
    basisValues[n] = prod(x₀.-theNodes[[1:n-1;n+1:end]])./prod(theNodes[n].-theNodes[[1:n-1;n+1:end]])
end
initpm = [
    zeros(sum(model.C.<=0)) # LHS point mass
    zeros(sum(model.C.>=0)) # RHS point mass
]
initprobs = zeros(Float64,SFFM.NBases(mesh),SFFM.NIntervals(mesh),SFFM.NPhases(model))
initprobs[:,convert(Int,ceil(x₀/Δ)),1] = basisValues'*All.Matrices.Local.V.V*All.Matrices.Local.V.V'.*2/Δ
initdist = (
    pm = initpm,
    distribution = initprobs,
    x = SFFM.CellNodes(mesh),
    type = "density"
)
x0 = SFFM.Dist2Coeffs( model, mesh, initdist)
h = 0.0001
@time projyvals = SFFM.EulerDG( proj.D["++"](s = 0), y, x0, h = h)
@time interpyvals = SFFM.EulerDG( interp.D["++"](s = 0), y, x0, h = h)

## plotting
simdensity = SFFM.Sims2Dist( model, mesh, sims, type="probability")
projdensity = SFFM.Coeffs2Dist( model, mesh, projyvals, type="probability")
interpdensity = SFFM.Coeffs2Dist( model, mesh, interpyvals, type="probability")
println("proj error: ",SFFM.starSeminorm( projdensity, simdensity))
println("interp error: ",SFFM.starSeminorm( interpdensity, simdensity))

## plots
# plot solutions
# densities
p = SFFM.PlotSFM( model, mesh = mesh, dist = projdensity, color=:blue,label="proj")#,marker=:rtriangle)
p = SFFM.PlotSFM!(p, model, mesh, interpdensity, color=:red,label="interp")#,marker=:x)
# plot sims
p = SFFM.PlotSFM!(p, model, mesh, simdensity, color=:black,label="sim")#,marker=:ltriangle)

display(p)
## probabilities
simdensity = SFFM.Sims2Dist( model, mesh, sims, type="density")
projdensity = SFFM.Coeffs2Dist( model, mesh, projyvals, type="density")
interpdensity = SFFM.Coeffs2Dist( model, mesh, interpyvals, type="density")

## plots
# plot solutions
# densities
p = SFFM.PlotSFM( model, mesh = mesh, dist = projdensity, color=:blue,label="proj")#,marker=:rtriangle)
p = SFFM.PlotSFM!(p, model, mesh, interpdensity, color=:red,label="interp")#,marker=:x)
# plot sims
p = SFFM.PlotSFM!(p, model, mesh, simdensity, color=:black,label="sim")#,marker=:ltriangle)
# savefig(p,"/Users/a1627293/Dropbox/PhD/NotesMaster/whichR/m3.png")
display(p)
