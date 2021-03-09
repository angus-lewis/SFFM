## Define the model
## parameters
γ₁, β₁, λ₁, θ₁, κ = 11.0, 1.0, 12.48, 1.6, 2.6
γ₂, β₂, λ₂, θ₂, xstar = 22.0, 1.0, 16.25, 1.0, 1.6

## state space is {11,10,01,00}
T = [
    -(γ₁ + γ₂) γ₂ γ₁ 0;
    β₂ -(γ₁ + β₂) 0 γ₁;
    β₁ 0 -(γ₂ + β₁) γ₂;
    0 β₁ β₂ -(β₂ + β₂);
    ]

C = [λ₁ - θ₁; λ₁ - θ₁; -θ₁; -θ₁]

r₁₁(x) = (λ₂ - κ) * (x .== 0) +
    (λ₂ - θ₂) * ((x .> 0) .& (x .<= xstar)) +
    λ₂ * (x .> xstar)
r₁₀(x) = -κ * (x .== 0) +
    -θ₂ * ((x .> 0) .& (x .<= xstar))
r₀₁(x) = (λ₂ - κ) * (x .== 0) +
    (λ₂ - θ₂) * ((x .> 0) .& (x .<= xstar)) +
    λ₂ * (x .> xstar)
r₀₀(x) = -κ * (x .== 0) +
    -θ₂ * ((x .> 0) .& (x .<= xstar))

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

## make models
simBounds = [0 Inf; -Inf Inf] # bounds for simulation only
simModel = SFFM.Model(T = T, C = C, r = r, Bounds = simBounds)
println("created simModel with no upper bound")
println("")

approxBounds = [0 48; -Inf Inf] # bounds for approximation only
approxModel = SFFM.Model(T = T, C = C, r = r, Bounds = approxBounds)
println("Created approxModel with upper bound at x=", approxModel.Bounds[1,end])
println("")
