## Define the model
# This model was taken from
# /Users/angus2/Documents/Vikram/Fluid_Fluid/notebooks/My First Fluid Fluid Model.ipynb


γ₁, β₁, λ₁, θ₁, κ = 11.0, 1.0, 12.48, 1.6, 2.6
γ₂, β₂, λ₂, θ₂, xstar = 22.0, 1.0, 16.25, 1.0, 1.6

T = [
    -(γ₁ + γ₂) γ₂ γ₁ 0;
    β₂ -(γ₁ + β₂) 0 γ₁;
    β₁ 0 -(γ₂ + β₁) γ₂;
    0 β₁ β₂ -(β₂ + β₂);
    ]

C = [λ₁ - θ₁; λ₁ - θ₁; -θ₁; -θ₁]

r₁₁(x) = (λ₂ - κ).*ones(size(x))
r₁₀(x) = -κ * (x .== 0) +
    -θ₂ * ((x .> 0) .& (x .< xstar))
r₀₁(x) = (λ₂ - κ).*ones(size(x))
r₀₀(x) = -κ * (x .== 0) +
    -θ₂ * ((x .> 0) .& (x .< xstar))


R₁₁(x) = x .* (λ₂ - κ)
R₁₀(x) = x .* -θ₂ * ((x .> 0) .& (x .< xstar)) + xstar .* -θ₂ * (x .>= xstar)
R₀₁(x) = x .* (λ₂ - κ)
R₀₀(x) = x .* -θ₂ * ((x .> 0) .& (x .< xstar)) + xstar .* -θ₂ * (x .>= xstar)

r = (
    r = function (x)
        [r₁₁(x) r₁₀(x) r₀₁(x) r₀₀(x)]
    end,
    R = function (x)
        [R₁₁(x) R₁₀(x) R₀₁(x) R₀₀(x)]
    end,
)
