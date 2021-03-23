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
    Ψₓ = SFFM.PsiFunX( approxModel)
    ξₓ = SFFM.MakeXiX( approxModel, Ψₓ)
    pₓ, πₓ, Πₓ, Kₓ = SFFM.StationaryDistributionX( approxModel, Ψₓ, ξₓ)

    c = 0
    for NBases in 1:2:3
        mesh = SFFM.MakeMesh(
            approxModel,
            Nodes,
            NBases,
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
            approxModel,
            mesh,
            analyticX,
            color = :red,
            label = "Analytic",
            marker = :x,
            seriestype = :line,
            jitter = 0.5,
        )

        # now DG it
        c = c+1

        # compute the marginal via DG
        All = SFFM.MakeAll( approxModel, mesh, approxType = "interpolation")
        Ψ = SFFM.PsiFun(All.D)

        # construct FRAP matrices
        me = SFFM.MakeME(SFFM.CMEParams[NBases], mean = mesh.Δ[1])
        B = SFFM.MakeBFRAP( approxModel, mesh, me)
        D = SFFM.MakeD( All.R, B, approxModel, mesh)
        Ψme = SFFM.PsiFun( D)

        # the distribution of X when Y first returns to 0
        ξ = SFFM.MakeXi( All.B.BDict, Ψ)
        ξme = SFFM.MakeXi( B.BDict, Ψme)

        marginalX, p, K = SFFM.MakeLimitDistMatrices(;
            All.B.BDict,
            All.D,
            All.R.RDict,
            Ψ,
            ξ,
            mesh,
        )
        marginalXme, pme, Kme = SFFM.MakeLimitDistMatrices(;
            B.BDict,
            D,
            All.R.RDict,
            Ψme,
            ξme,
            mesh,
        )
        # convert marginalX to a distribution for plotting
        Dist = SFFM.Coeffs2Dist(
            approxModel,
            mesh,
            marginalX,
            type="probability",
        )
        Distme = SFFM.Coeffs2Dist(
            approxModel,
            mesh,
            marginalXme,
            type = "probability",
            probTransform = true,
        )
        # plot it
        q = SFFM.PlotSFM!(
            q,
            approxModel,
            mesh,
            Dist,
            color = :green,
            label = "DG: N_k = "*string(NBases),
            seriestype = :line,
            jitter = 0.5,
        )
        q = SFFM.PlotSFM!(
            q,
            approxModel,
            mesh,
            Dist = Distme,
            color = :blue,
            label = "ME: N_k = "*string(NBases),
            seriestype = :line,
            jitter = 0.5,
        )

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
