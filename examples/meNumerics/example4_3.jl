include("../../src/SFFM.jl")
using LinearAlgebra, Plots

## define the model(s)
include("exampleModelDef.jl")

## section 4.3: the marginal stationary distribution of X
## mesh
Δ = 0.4
Nodes = collect(approxBounds[1, 1]:Δ:approxBounds[1, 2])
NBases = 3
Basis = "lagrange"
mesh = SFFM.MakeMesh(
    model = approxModel,
    Nodes = Nodes,
    NBases = NBases,
    Basis = Basis,
)
let 
    ## analytic version for comparison
    # construction
    Ψₓ = SFFM.PsiFunX(model=approxModel)
    ξₓ = SFFM.MakeXiX(model=approxModel, Ψ=Ψₓ)
    pₓ, πₓ, Πₓ, Kₓ = SFFM.StationaryDistributionX(model=approxModel, Ψ=Ψₓ, ξ=ξₓ)

    c = 0
    for NBases in 1:2:3
        mesh = SFFM.MakeMesh(
            model = approxModel,
            Nodes = Nodes,
            NBases = NBases,
            Basis=Basis,
        )

        # evaluate the distribution
        analyticX = (
            pm = [pₓ[:];0;0],
            distribution = Πₓ(mesh.Nodes[2:end]) - Πₓ(mesh.Nodes[1:end-1]),
            x = (mesh.Nodes[1:end-1]+mesh.Nodes[2:end])./2,
            type = "probability"
        )

        # plot it
        q = SFFM.PlotSFM(
            model = approxModel,
            mesh = mesh,
            Dist = analyticX,
            color = :red,
            label = "Analytic",
            marker = :x,
            seriestype = :line,
            jitter = 0.5,
        )

        # now DG it
        c = c+1

        # compute the marginal via DG
        All = SFFM.MakeAll(model = approxModel, mesh = mesh, approxType = "interpolation")
        Ψ = SFFM.PsiFun(D=All.D)

        # construct FRAP matrices
        me = SFFM.MakeME(SFFM.CMEParams[NBases], mean = mesh.Δ[1])
        B = SFFM.MakeBFRAP(model=approxModel,mesh=mesh,me=me)
        D = SFFM.MakeD(R=All.R,B=B,model=approxModel,mesh=mesh)
        Ψme = SFFM.PsiFun(D=D)

        # the distribution of X when Y first returns to 0
        ξ = SFFM.MakeXi(B=All.B.BDict, Ψ = Ψ)
        ξme = SFFM.MakeXi(B=B.BDict, Ψ = Ψme)

        marginalX, p, K = SFFM.MakeLimitDistMatrices(;
            B = All.B.BDict,
            D = All.D,
            R = All.R.RDict,
            Ψ = Ψ,
            ξ = ξ,
            mesh = mesh,
        )
        marginalXme, pme, Kme = SFFM.MakeLimitDistMatrices(;
            B = B.BDict,
            D = D,
            R = All.R.RDict,
            Ψ = Ψme,
            ξ = ξme,
            mesh = mesh,
        )
        # convert marginalX to a distribution for plotting
        Dist = SFFM.Coeffs2Dist(
            model = approxModel,
            mesh = mesh,
            Coeffs = marginalX,
            type="probability",
        )
        Distme = SFFM.Coeffs2Dist(
            model = approxModel,
            mesh = mesh,
            Coeffs = marginalXme,
            type = "probability",
            probTransform = true,
        )
        # plot it
        q = SFFM.PlotSFM!(q;model=approxModel,mesh=mesh,
            Dist = Dist,
            color = :green,
            label = "DG: N_k = "*string(NBases),
            seriestype = :line,
            jitter = 0.5,
        )
        q = SFFM.PlotSFM!(q;model=approxModel,mesh=mesh,
            Dist = Distme,
            color = :blue,
            label = "ME: N_k = "*string(NBases),
            seriestype = :line,
            jitter = 0.5,
        )

        # ## DG eigenvalue problem for πₓ
        # Q = copy(All.B.B)
        # Q[:,1] .= 1
        #
        # b = zeros(1,size(Q,1))
        # b[1] = 1
        # # w solves wQ = 0 s.t. sum(w) = 1
        # w = b/Q
        #
        # # convert w to a distribution
        # eigDist = SFFM.Coeffs2Dist(
        #     model = approxModel,
        #     mesh = mesh,
        #     Coeffs = w,
        #     type="density",
        # )
        #
        # # plot it
        # q = SFFM.PlotSFM!(q;
        #     model = approxModel,
        #     mesh = mesh,
        #     Dist = eigDist,
        #     color = NBases+10,
        #     label = "DG eigen: "*string(NBases),
        #     marker = :none,
        # )
        q = plot!(windowsize=(600,600))
        titles = ["Phase 11" "Phase 10" "Phase 01" "Phase 00"]
        for sp in 1:4
            q = plot!(
                subplot = sp,
                xlims = (-0.5,5),
                title = titles[sp],
                xlabel = "x",
                ylabel = "Density / Probability",
            )
        end
        display(q)
    end
    # savefig(pwd()*"/examples/paperNumerics/dump/marginalStationaryDistX.png")
end
