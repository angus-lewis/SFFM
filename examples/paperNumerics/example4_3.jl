include("../../src/SFFM.jl")
using LinearAlgebra, Plots

## define the model(s)
include("exampleModelDef.jl")

## section 4.3: the marginal stationary distribution of X
## mesh
Δ = 0.4
Nodes = collect(approxBounds[1, 1]:Δ:approxBounds[1, 2])
NBases = 2
Basis = "lagrange"
mesh = SFFM.DGMesh(
    approxModel,
    Nodes = Nodes,
    NBases = NBases,
    Basis = Basis,
)
let q = SFFM.PlotSFM(model = approxModel)
    ## analytic version for comparison
    # construction
    Ψₓ = SFFM.PsiFunX(model=approxModel)
    ξₓ = SFFM.MakeXiX(model=approxModel, Ψ=Ψₓ)
    pₓ, πₓ, Πₓ, Kₓ = SFFM.StationaryDistributionX(model=approxModel, Ψ=Ψₓ, ξ=ξₓ)

    # evaluate the distribution
    analyticX = (
        pm = [pₓ[:];0;0],
        distribution = πₓ(mesh.CellNodes),
        x = mesh.CellNodes,
        type = "density"
    )

    # plot it
    q = SFFM.PlotSFM!(q;
        model=approxModel,
        mesh=mesh,
        Dist = analyticX,
        color = :red,
        label = "Analytic",
        marker = :x,
        seriestype = :line,
        jitter = 0.5,
    )

    # now DG it
    c = 0
    colours = [:green;:blue]
    for NBases in 1:2
        c = c+1
        mesh = SFFM.DGMesh(
            approxModel,
            Nodes = Nodes,
            NBases = NBases,
            Basis=Basis,
        )

        # compute the marginal via DG
        All = SFFM.MakeAll(model = approxModel, mesh = mesh, approxType = "projection")
        Ψ = SFFM.PsiFun(D=All.D)

        # the distribution of X when Y first returns to 0
        ξ = SFFM.MakeXi(B=All.B.BDict, Ψ = Ψ)

        marginalX, p, K = SFFM.MakeLimitDistMatrices(;
            B = All.B.BDict,
            D = All.D,
            R = All.R.RDict,
            Ψ = Ψ,
            ξ = ξ,
            mesh = mesh,
        )
        # convert marginalX to a distribution for plotting
        Dist = SFFM.Coeffs2Dist(
            model = approxModel,
            mesh = mesh,
            Coeffs = marginalX,
            type="density",
        )
        # plot it
        q = SFFM.PlotSFM!(q;model=approxModel,mesh=mesh,
            Dist = Dist,
            color = colours[c],
            label = "DG: N_k = "*string(NBases),
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
