## Define the model
T = [
    -2 1 1;
    0.5 -1 0.5; 
    1 1 -2;
    ]

C = [1; -1; 0]

r₁(x) = ones(size(x))
r₂(x) = -ones(size(x))

R₁(x) = x
R₂(x) = -x

r = (
    r = function (x)
        [r₁(x) r₂(x) r₂(x)]
    end,
    R = function (x)
        [R₁(x) R₂(x) R₂(x)]
    end,
)

## make models
simBounds = [0 Inf; -Inf Inf] # bounds for simulation only
simModel = SFFM.Model(T, C, r, Bounds = simBounds)

approxBounds = [0 48; -Inf Inf] # bounds for approximation only
model = SFFM.Model(T, C, r, Bounds = approxBounds)
