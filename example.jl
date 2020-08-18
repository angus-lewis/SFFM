include("./SFFM.jl")
using LinearAlgebra, Plots

## Define the model
γ₁, β₁, λ₁, θ₁, κ = 11.0, 1.0, 12.48, 1.6, 2.6
γ₂, β₂, λ₂, θ₂, xstar = 22.0, 1.0, 16.25, 1.0, 1.6

T = [
    -(γ₁ + γ₂) γ₂ γ₁ 0;
    β₂ -(γ₁ + β₂) 0 γ₁;
    β₁ 0 -(γ₂ + β₁) γ₂;
    0 β₁ β₂ -(β₂ + β₂);
    ]
C = [λ₁ - θ₁; λ₁ - θ₁; -θ₁; -θ₁]

r₁₁(x) = (λ₂ - κ) * (x .== 0) +
    (λ₂ - θ₂) * ((x .> 0) .& (x .< xstar)) +
    λ₂ * (x .> xstar)
r₁₀(x) = -κ * (x .== 0) +
    -θ₂ * ((x .> 0) .& (x .< xstar))
r₀₁(x) = (λ₂ - κ) * (x .== 0) +
    (λ₂ - θ₂) * ((x .> 0) .& (x .< xstar)) +
    λ₂ * (x .> xstar)
r₀₀(x) = -κ * (x .== 0) +
    -θ₂ * ((x .> 0) .& (x .< xstar))

R₁₁(x) = x .* (λ₂ - θ₂) .* ((x .> 0) .& (x .< xstar)) .+
    ((x .- xstar) .* λ₂ .+ xstar .* (λ₂ - θ₂)) .* (x .>= xstar)
R₁₀(x) = x .* -θ₂ * ((x .> 0) .& (x .< xstar)) + xstar .* -θ₂ * (x .>= xstar)
R₀₁(x) = x .* (λ₂ - θ₂) .* ((x .> 0) .& (x .< xstar)) .+
    ((x .- xstar) .* λ₂ .+ xstar .* (λ₂ - θ₂)) .* (x .>= xstar)
R₀₀(x) = x .* -θ₂ * ((x .> 0) .& (x .< xstar)) + xstar .* -θ₂ * (x .>= xstar)

r = (
    r = function (x)
        [r₁₁(x) r₁₀(x) r₀₁(x) r₀₀(x)]
    end,
    R = function (x)
        [R₁₁(x) R₁₀(x) R₀₁(x) R₀₀(x)]
    end,
)

simBounds = [0 Inf; -Inf Inf] # bounds for simulation only
simModel = SFFM.MakeModel(T = T, C = C, r = r, Bounds = simBounds)

approxBounds = [0 16; -Inf Inf] # bounds for approximation only
approxModel = SFFM.MakeModel(T = T, C = C, r = r, Bounds = approxBounds)
## simulate
NSim = 10^5
IC = (φ = 3 .* ones(Int, NSim), X = 5 .* ones(NSim), Y = zeros(NSim))
sims = SFFM.SimSFFM(
    Model = simModel,
    StoppingTime = SFFM.FirstExitY(u = 0, v = Inf),
    InitCondition = IC,
)

## mesh
Δ = 0.4
Nodes = collect(approxBounds[1, 1]:Δ:approxBounds[1, 2])

NBases = 1
Basis = "lagrange"
Mesh = SFFM.MakeMesh(
    Model = approxModel,
    Nodes = Nodes,
    NBases = NBases,
    Basis=Basis
)

## plots
simprobs = SFFM.Sims2Dist(Model=simModel,Mesh=Mesh,sims=sims,type="probability")
cumprobs = cumsum(simprobs.distribution,dims=2)
cumprobs[:,:,3] = cumprobs[:,:,3] .+ simprobs.pm[1]
cumprobs[:,:,4] = cumprobs[:,:,4] .+ simprobs.pm[2]
cumprobs = repeat(cumprobs,2,1,1)
cumprobs[1,:,1] = [0; cumprobs[1,1:end-1,1]]
cumprobs[1,:,2] = [0; cumprobs[1,1:end-1,2]]
cumprobs[1,:,3] = [simprobs.pm[1]; cumprobs[1,1:end-1,3]]
cumprobs[1,:,4] = [simprobs.pm[2]; cumprobs[1,1:end-1,4]]
x = [
    Mesh.CellNodes .- Δ/2;
    Mesh.CellNodes .+ Δ/2;
]
cumprobs = (
    pm = simprobs.pm,
    distribution = cumprobs,
    x = x,
    type = "density",
)
p1 = SFFM.PlotSFM(Model=approxModel,Mesh=Mesh,Dist=cumprobs)

p2 = plot(
    cumprobs.x,
    cumprobs.distribution[:,:,4],
    color = 1,
    legend = false,
    xlims = (0,1.6),
)

p3 = plot(
    cumprobs.x,
    cumprobs.distribution[:,:,2],
    color = 1,
    legend = false,
    xlims = (0,1.6),
)
