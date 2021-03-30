include(pwd()*"/src/SFFM.jl")
using LinearAlgebra, Plots, JLD2
#
# ## define the model(s)
include(pwd()*"/examples/meNumerics/discontinuitiesModelDef.jl")

## simulate Ψ paths for comparison
innerNSim = 10^4
NSim = 10^2 * innerNSim

# # we can do this in parallell
using SharedArrays, Distributed
nprocs() < 5 && addprocs(5 - nprocs())
@everywhere include(pwd()*"/src/SFFM.jl")
@everywhere include(pwd()*"/examples/meNumerics/discontinuitiesModelDef.jl")
simsOuter_Psi = SharedArray(zeros(NSim, 5))
simsOuter_1 = SharedArray(zeros(NSim, 5))

@time @sync @distributed for n = 1:(NSim÷innerNSim)
    IC = (φ = 1 .* ones(Int, innerNSim), X = zeros(innerNSim), Y = zeros(innerNSim))
    simsInner_Psi = SFFM.SimSFFM(
        model,
        SFFM.FirstExitY(0, Inf),
        IC,
    )
    simsOuter_Psi[1+(n-1)*innerNSim:n*innerNSim, :] =
        [simsInner_Psi.t simsInner_Psi.φ simsInner_Psi.X simsInner_Psi.Y simsInner_Psi.n]
    simsInner_1 = SFFM.SimSFFM(
        model,
        SFFM.FixedTime(1.2),
        IC,
    )
    simsOuter_1[1+(n-1)*innerNSim:n*innerNSim, :] =
        [simsInner_1.t simsInner_1.φ simsInner_1.X simsInner_1.Y simsInner_1.n]
end

sims_Psi = (
    t = simsOuter_Psi[:, 1],
    φ = simsOuter_Psi[:, 2],
    X = simsOuter_Psi[:, 3],
    Y = simsOuter_Psi[:, 4],
    n = simsOuter_Psi[:, 5],
)

sims_1 = (
    t = simsOuter_1[:, 1],
    φ = simsOuter_1[:, 2],
    X = simsOuter_1[:, 3],
    Y = simsOuter_1[:, 4],
    n = simsOuter_1[:, 5],
)

@save pwd()*"/examples/meNumerics/discontinuitiesModelSims.jld2" sims_Psi sims_1 model
interrupt()