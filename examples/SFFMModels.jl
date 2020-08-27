SimpleInOutModel = begin
    T = [-2.0 1.0 1.0; 1.0 -2.0 1.0; 1.0 1.0 -2]
    C = [1.0; -2.0; 0]
    r = (
    r = function (x)
            [1.0 .+ sin.(x) 2 * (x .> 0) .* x .^ 2.0 .+ 1 (x .> 0) .* x .+ 1]
        end, # r = function (x); [1.0.+0.01*x 1.0.+0.01*x 1*ones(size(x))]; end,
        R = function (x)
            [x .- cos.(x) 2 * (x .> 0) .* x .^ 3 / 3.0 .+ 1 * x (x .> 0) .* x .^ 2 / 2.0 .+
                                                                1 * x]
        end,
    )
    Bounds = [-Inf Inf; -Inf Inf]
    SFFM.MakeModel(T = T, C = C, r = r, Bounds = Bounds)
end
