include("../../src/SFFM.jl")
using LinearAlgebra, Plots, JLD2
#
# ## define the model(s)
include("exampleModelDef.jl")

## simulate Ψ paths for comparison
innerNSim = 10^2
NSim = 10^1 * innerNSim

# # we can do this in parallell
using SharedArrays, Distributed
nprocs() < 5 && addprocs(5 - nprocs())
@everywhere include("src/SFFM.jl")
@everywhere include("examples/paperNumerics/exampleModelDef.jl")
simsOuter = SharedArray(zeros(NSim, 5))

# using Profile
# Profile.clear()
# @profiler @time for n = 1:(NSim÷innerNSim)
#     IC = (φ = 3 .* ones(Int, innerNSim), X = 5 .* ones(innerNSim), Y = zeros(innerNSim))
#     simsInner = SFFM.SimSFFM(
#         model = simModel,
#         StoppingTime = SFFM.FirstExitY(u = 0, v = Inf),
#         InitCondition = IC,
#     )
#     simsOuter[1+(n-1)*innerNSim:n*innerNSim, :] =
#         [simsInner.t simsInner.φ simsInner.X simsInner.Y simsInner.n]
# end

@time @sync @distributed for n = 1:(NSim÷innerNSim)
    IC = (φ = 3 .* ones(Int, innerNSim), X = 5 .* ones(innerNSim), Y = zeros(innerNSim))
    simsInner = SFFM.SimSFFM(
        model = simModel,
        StoppingTime = SFFM.FirstExitY(u = 0, v = Inf),
        InitCondition = IC,
    )
    simsOuter[1+(n-1)*innerNSim:n*innerNSim, :] =
        [simsInner.t simsInner.φ simsInner.X simsInner.Y simsInner.n]
end

sims = (
    t = simsOuter[:, 1],
    φ = simsOuter[:, 2],
    X = simsOuter[:, 3],
    Y = simsOuter[:, 4],
    n = simsOuter[:, 5],
)

# @save pwd()*"/examples/paperNumerics/dump/sims.jld2" sims
