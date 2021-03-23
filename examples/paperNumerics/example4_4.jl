# include("../../src/SFFM.jl")
# # using LinearAlgebra, Plots
# #
# # ## define the model(s)
# include("exampleModelDef.jl")

## section 4.4: the sensitivity of the stationary distribution of X to rates r
Tfun(γ₂) = [
    -(γ₁ + γ₂) γ₂ γ₁ 0;
    β₂ -(γ₁ + β₂) 0 γ₁;
    β₁ 0 -(γ₂ + β₁) γ₂;
    0 β₁ β₂ -(β₂ + β₂);
    ]
let
    c = 0
    colours = [:green;:blue;:red]
    shapes = [:diamond,:x,:+]
    styles = [:solid,:dash,:dot]
    q = plot(layout = (1,2))
    for sp in 1:2
        q = plot!(windowsize = (600,250), subplot = sp)
    end
    for γ₂ in [11;16;22]
        c = c+1
        Ttemp = Tfun(γ₂)
        tempModel = SFFM.Model( Ttemp, C, r, Bounds = approxModel.Bounds)
        println("created tempModel with upper bound x=", tempModel.Bounds[1,end])
        ## mesh
        Δ = 0.4
        Nodes = collect(approxBounds[1, 1]:Δ:approxBounds[1, 2])

        Basis = "lagrange"
        NBases = 2
        mesh = SFFM.DGMesh(
            tempModel,
            Nodes,
            NBases,
            Basis=Basis,
        )

        # compute the marginal via DG
        All = SFFM.MakeAll( tempModel, mesh, approxType = "projection")
        Ψ = SFFM.PsiFun( All.D)

        # the distribution of X when Y first returns to 0
        ξ = SFFM.MakeXi( All.B.BDict, Ψ)

        marginalX, p, K = SFFM.MakeLimitDistMatrices(
            All.B.BDict,
            All.D,
            All.R.RDict,
            Ψ,
            ξ,
            mesh,
            tempModel,
        )
        println("For γ₂ = ",
            γ₂, ", χ⁰ = ",
            sum(p),
            ", χ¹ = ",
            sum(marginalX)-sum(p),
            " and total prob is ",
            sum(marginalX),
            ".",
        )
        tempDist = SFFM.Coeffs2Dist(
            tempModel,
            mesh,
            marginalX,
            type="density",
        )
        temp = zeros(mesh.NBases,mesh.NIntervals,2)
        temp[:,:,1] = tempDist.distribution[:,:,1]+tempDist.distribution[:,:,2]
        temp[:,:,2] = tempDist.distribution[:,:,3]+tempDist.distribution[:,:,4]
        q = plot!(
            tempDist.x,
            temp[:,:,1],
            subplot = 1,
            legend = false,
            color = colours[c],
            seriestype = :line,
            linestyle = styles[c],
            markershape = shapes[c],
            xlabel = "x",
        )
        q = scatter!(
            [-0.2 + (c-1)*0.2],
            [sum(tempDist.pm)],
            subplot = 2,
            color = colours[c],
            label = :none,
        )
        q = plot!(
            tempDist.x[:,1],
            temp[:,1,2],
            subplot = 2,
            color = colours[c],
            label = "γ₂: "*string(γ₂),
            linestyle = styles[c],
            markershape = shapes[c],
        )
        q = plot!(
            tempDist.x[:,2:end],
            temp[:,2:end,2],
            subplot = 2,
            color = colours[c],
            label = :none,
            markershape = shapes[c],
            linestyle = styles[c],
            xlabel = "x",
        )

        println("")
    end
    titles = ["Phases 11 + 10" "Phases 01 + 00"]
    for sp in 1:2
        q = plot!(
            subplot = sp,
            xlims = (-0.5,8),
            title = titles[sp],
            ylabel = "Density / Probability",
            grid = false,
        )
    end
    display(q)
    # savefig(pwd()*"/examples/paperNumerics/dump/sensitivityMarginalStationaryDistX.png")
end
