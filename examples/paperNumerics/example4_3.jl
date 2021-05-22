include("../../src/SFFM.jl")
using LinearAlgebra, Plots

## define the model(s)
include("exampleModelDef.jl")

## section 4.3: the marginal stationary distribution of X
## mesh
Δtemp = 0.4
Nodes = collect(approxBounds[1, 1]:Δtemp:approxBounds[1, 2])
nBases = 2
Basis = "lagrange"
mesh = SFFM.DGMesh(
    approxModel,
    Nodes,
    nBases,
    Basis = Basis,
)
let q = SFFM.plot(approxModel)
    ## analytic version for comparison
    # construction
    Ψₓ = SFFM.PsiFunX(approxModel)
    ξₓ = SFFM.MakeXiX(approxModel, Ψₓ)
    pₓ, πₓ, Πₓ, Kₓ = SFFM.StationaryDistributionX( approxModel, Ψₓ, ξₓ)

    # evaluate the distribution
    analyticX = SFFM.SFFMDensity(
        [pₓ[:];0;0],
        πₓ(SFFM.CellNodes(mesh)),
        SFFM.CellNodes(mesh),
    )

    # plot it
    q = SFFM.plot!(q,
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
    c = 0
    colours = [:green;:blue]
    for nBases in 1:2
        c = c+1
        mesh = SFFM.DGMesh(
            approxModel,
            Nodes,
            nBases,
            Basis=Basis,
        )

        # compute the marginal via DG
        All = SFFM.MakeAll( approxModel, mesh, approxType = "projection")
        Ψ = SFFM.PsiFun(All.D)

        # the distribution of X when Y first returns to 0
        ξ = SFFM.MakeXi(All.B.BDict, Ψ)

        marginalX, p, K = SFFM.MakeLimitDistMatrices(
            All.B.BDict,
            All.D,
            All.R.RDict,
            Ψ,
            ξ,
            mesh,
            approxModel,
        )
        # convert marginalX to a distribution for plotting
        Dist = SFFM.Coeffs2Dist(
            approxModel,
            mesh,
            marginalX,
            SFFM.SFFMDensity,
        )
        # plot it
        q = SFFM.plot!(
            q,
            approxModel,
            mesh,
            Dist,
            color = colours[c],
            label = "DG: N_k = "*string(nBases),
            seriestype = :line,
            jitter = 0.5,
        )

    end
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
    # savefig(pwd()*"/examples/paperNumerics/dump/marginalStationaryDistX.png")
end
