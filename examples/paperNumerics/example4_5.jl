include("../../src/SFFM.jl")
using LinearAlgebra, Plots

## define the model(s)
include("exampleModelDef.jl")

## analytic X distribution for comparison
# construction
Ψₓ = SFFM.PsiFunX(Model=approxModel)
ξₓ = SFFM.MakeXiX(Model=approxModel, Ψ=Ψₓ)
pₓ, πₓ, Πₓ, Kₓ = SFFM.StationaryDistributionX(Model=approxModel, Ψ=Ψₓ, ξ=ξₓ)

# ## simulate Ψ paths for comparison
# innerNSim = 10^2
# NSim = 10^3*innerNSim
#
# # we can do this in parallell
# using SharedArrays, Distributed
# nprocs() < 5 && addprocs(5-nprocs())
# @everywhere include("src/SFFM.jl")
# @everywhere include("examples/exampleModelDef.jl")
#
# simsOuter = SharedArray(zeros(NSim,5))
# @time @sync @distributed for n in 1:(NSim÷innerNSim)
#     IC = (
#         φ = 3 .* ones(Int, innerNSim),
#         X = 5 .* ones(innerNSim),
#         Y = zeros(innerNSim)
#     )
#     simsInner = SFFM.SimSFFM(
#         Model = simModel,
#         StoppingTime = SFFM.FirstExitY(u = 0, v = Inf),
#         InitCondition = IC,
#     )
#     simsOuter[1+(n-1)*innerNSim:n*innerNSim,:] =
#         [simsInner.t simsInner.φ simsInner.X simsInner.Y simsInner.n]
# end
#
# sims = (
#     t = simsOuter[:,1],
#     φ = simsOuter[:,2],
#     X = simsOuter[:,3],
#     Y = simsOuter[:,4],
#     n = simsOuter[:,5],
# )

## section 4.5: error for approximation of stationary distribution of X
Δs = [1.6;0.8;0.4;0.2;0.1]#;0.05]
NBasesRange = [1;2;3;4]#;8]
πnorms = zeros(length(Δs),length(NBasesRange))
Ψnorms = zeros(length(Δs),length(NBasesRange))
Basis = "lagrange"
for d in 1:length(Δs), n in 1:length(NBasesRange)
    if d + 2*n <= 10
        # define the mesh for each iteration
        NBases = NBasesRange[n]
        Δ = Δs[d]
        println("Mesh details; Δ = ", Δ, " NBases = ",NBases)
        Nodes = collect(approxBounds[1, 1]:Δ:approxBounds[1, 2])
        Mesh = SFFM.MakeMesh(
            Model = approxModel,
            Nodes = Nodes,
            NBases = NBases,
            Basis=Basis,
        )

        ## stationary distirbution stuff
        # evaluate the analytic result given the mesh
        analyticX = (
            pm = [pₓ[:];0;0],
            distribution = Πₓ(Matrix(Mesh.Nodes[2:end]')) - Πₓ(Matrix(Mesh.Nodes[1:end-1]')),
            x = Mesh.Nodes[1:end-1] + Mesh.Δ/2,
            type = "probability"
        )

        # compute the marginal via DG
        All = SFFM.MakeAll(Model = approxModel, Mesh = Mesh, approxType = "projection")
        Ψ = SFFM.PsiFun(D=All.D)

        # the distribution of X when Y first returns to 0
        ξ = SFFM.MakeXi(B=All.B.BDict, Ψ = Ψ)

        marginalX, p, K = SFFM.MakeLimitDistMatrices(;
            B = All.B.BDict,
            D = All.D,
            R = All.R.RDict,
            Ψ = Ψ,
            ξ = ξ,
            Mesh = Mesh,
        )

        # convert marginalX to a distribution for analysis
        DGStationaryDist = SFFM.Coeffs2Dist(
            Model = approxModel,
            Mesh = Mesh,
            Coeffs = marginalX,
            type="probability",
        )

        # save them
        πnorms[d,n] = SFFM.starSeminorm(d1 = DGStationaryDist, d2 = analyticX)

        # diff = (pm = DGStationaryDist.pm - analyticX.pm,
        #     distribution = DGStationaryDist.distribution - analyticX.distribution,
        #     x = DGStationaryDist.x,
        #     type = DGStationaryDist.type)

        # ## now do Ψ
        # ## turn sims into a cdf
        # simprobs = SFFM.Sims2Dist(Model=simModel,Mesh=Mesh,sims=sims,type="probability")
        #
        # # do DG for Ψ
        # initpm = [
        #     zeros(sum(approxModel.C.<=0)) # LHS point mass
        #     zeros(sum(approxModel.C.>=0)) # RHS point mass
        # ]
        # initprobs = zeros(Float64,1,Mesh.NIntervals,approxModel.NPhases)
        # initprobs[1,convert(Int,ceil(5/Δ)),3] = 1 # a point mass of 1 on the cell ceil(5/Δ))
        # initdist = (
        #     pm = initpm,
        #     distribution = initprobs,
        #     x = Matrix(Mesh.CellNodes[1,:]'),
        #     type = "probability"
        # ) # convert to a distribution object so we can apply Dist2Coeffs
        # # convert to Coeffs α in the DG context
        # x0 = SFFM.Dist2Coeffs(Model = approxModel, Mesh = Mesh, Distn = initdist)
        # # the initial condition on Ψ is restricted to + states so find the + states
        # plusIdx = [
        #     Mesh.Fil["p+"];
        #     repeat(Mesh.Fil["+"]', Mesh.NBases, 1)[:];
        #     Mesh.Fil["q+"];
        # ]
        # # get the elements of x0 in + states only
        # x0 = x0[plusIdx]'
        # # check that it is equal to 1 (or at least close)
        # println(sum(x0))
        #
        # # compute x-distribution at the time when Y returns to 0
        # w = x0*Ψ
        # # this can occur in - states only, so find the - states
        # minusIdx = [
        #     Mesh.Fil["p-"];
        #     repeat(Mesh.Fil["-"]', Mesh.NBases, 1)[:];
        #     Mesh.Fil["q-"];
        # ]
        # # then map to the whole state space for analysis
        # z = zeros(
        #     Float64,
        #     Mesh.NBases*Mesh.NIntervals * approxModel.NPhases +
        #         sum(approxModel.C.<=0) + sum(approxModel.C.>=0)
        # )
        # z[minusIdx] = w
        # # check that it is equal to 1
        # println(sum(z))
        #
        # # convert to a distribution object for plotting
        # DGΨProbs = SFFM.Coeffs2Dist(
        #     Model = approxModel,
        #     Mesh = Mesh,
        #     Coeffs = z,
        #     type="probability",
        # )
        #
        # Ψnorms[d,n] = SFFM.starSeminorm(d1 = DGΨDist, d2 = simprobs)
    else
        πnorms[d,n] = NaN
    end
end

let p = plot()
    for n in 1:length(NBasesRange)
        p = plot!(
            Δs,
            πnorms[:,n],
            xaxis=:log,
            yaxis=:log,
            xlabel="Δ",
            ylabel="Error",
            label="NBases = "*string(NBasesRange[n]),
            legend=:outertopright,
            seriestype=:line,
            markershape=:diamond,
        )
    end
    display(p)
end

m = (log.(πnorms[1:end-1,:]).-log.(πnorms[2:end,:]))./(log.(Δs[1:end-1])-log.(Δs[2:end]))
n = (log.(πnorms[:,1:end-1]).-log.(πnorms[:,2:end]))./(log.(NBasesRange[1:end-1])-log.(NBasesRange[2:end]))'
