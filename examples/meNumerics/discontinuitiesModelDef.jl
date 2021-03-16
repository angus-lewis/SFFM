## define a model
T = [-0.1 0.1;
    1 -1]
C = [1; -11]

r₁(x) = - (x.>1) + (x.<=1)
r₂(x) = x.*0
R₁(x) = (x.>1).*(x.-1) - (x.<=1).*x

r = (
    r = function (x)
        [r₁(x) r₂(x)]
    end,
    R = function (x)
        [R₁(x) r₂(x)]
    end
)

bounds = [0 6; -Inf Inf]
model = SFFM.Model(T = T, C = C, r = r, Bounds = bounds)
simBounds = [0 Inf; -Inf Inf]
simModel = SFFM.Model(T = T, C = C, r = r, Bounds = simBounds)