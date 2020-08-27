## Define the model

T = [
    -1.0 1.0;
    2.0 -2.0;
    ]

C = [1.0; -3.0]

r₁(x) = 1.0 .* ones(Float64,size(x))
r₂(x) = -4.0 .* ones(Float64,size(x))

R₁(x) = x .* r₁(x)
R₂(x) = x .* r₂(x)

r = (
    r = function (x)
        [r₁(x) r₂(x)]
    end,
    R = function (x)
        [R₁(x) R₂(x)]
    end,
)
