include("../../src/SFFM.jl")
using LinearAlgebra, Plots, JLD2

## define the model(s)
include("exampleModelDef.jl")

## section 4.2: Ψ paths

## load simulated Ψ paths
@load pwd()*"/examples/paperNumerics/dump/sims.jld2" sims

## mesh
Δ = 0.4
Nodes = collect(approxBounds[1, 1]:Δ:approxBounds[1, 2])

NBases = 1
Basis = "lagrange"
Mesh = SFFM.MakeMesh(
    Model = approxModel,
    Nodes = Nodes,
    NBases = NBases,
    Basis = Basis,
)

## turn sims into a cdf
simprobs = SFFM.Sims2Dist(Model=simModel,Mesh=Mesh,sims=sims,type="cumulative")

## plot simulations
let
    p2 = plot(
        simprobs.x[:],
        simprobs.distribution[:,:,2][:],
        label = "Sim",
        color = :red,
        xlims = (0,2),
        legend = :bottomright,
        title = "Phase 10",
        seriestype = :line,
        linestyle = :dot,
        markershape = :x,
        markersize = 4,
        xlabel = "x",
        ylabel = "Cumulative probability",
        windowsize = (600,400),
        grid = false,
        tickfontsize = 10,
        guidefontsize = 12,
        titlefontsize = 18,
        legendfontsize = 10,
    )

    p4 = plot(
        simprobs.x[:],
        simprobs.distribution[:,:,4][:],
        label = "Sim",
        color = :red,
        xlims = (0,2),
        legend = :bottomright,
        title = "Phase 00",
        seriestype = :line,
        linestyle = :dot,
        markershape = :x,
        markersize = 4,
        xlabel = "x",
        ylabel = "Cumulative probability",
        windowsize = (600,400),
        grid = false,
        tickfontsize = 10,
        guidefontsize = 12,
        titlefontsize = 18,
        legendfontsize = 10,
    )

    colours = [:blue;:black]
    ## DG
    c = 0
    styles = [:solid,:dash]
    for NBases in 1:2
        c = c+1
        Mesh = SFFM.MakeMesh(
            Model = approxModel,
            Nodes = Nodes,
            NBases = NBases,
            Basis = Basis,
        )
        # construct matrices
        All = SFFM.MakeAll(Model = approxModel, Mesh = Mesh, approxType = "projection")
        Ψ = SFFM.PsiFun(D=All.D)

        # construct initial condition
        theNodes = Mesh.CellNodes[:,convert(Int,ceil(5/Δ))]
        basisValues = zeros(length(theNodes))
        for n in 1:length(theNodes)
            basisValues[n] = prod(5.0.-theNodes[[1:n-1;n+1:end]])./prod(theNodes[n].-theNodes[[1:n-1;n+1:end]])
        end
        initpm = [
            zeros(sum(approxModel.C.<=0)) # LHS point mass
            zeros(sum(approxModel.C.>=0)) # RHS point mass
        ]
        initprobs = zeros(Float64,Mesh.NBases,Mesh.NIntervals,approxModel.NPhases)
        initprobs[:,convert(Int,ceil(5/Δ)),3] = basisValues'*All.Matrices.Local.V.V*All.Matrices.Local.V.V'.*2/Δ
        initdist = (
            pm = initpm,
            distribution = initprobs,
            x = Mesh.CellNodes,
            type = "density"
        ) # convert to a distribution object so we can apply Dist2Coeffs
        # convert to Coeffs α in the DG context
        x0 = SFFM.Dist2Coeffs(Model = approxModel, Mesh = Mesh, Distn = initdist)
        # the initial condition on Ψ is restricted to + states so find the + states
        plusIdx = [
            Mesh.Fil["p+"];
            repeat(Mesh.Fil["+"]', Mesh.NBases, 1)[:];
            Mesh.Fil["q+"];
        ]
        # get the elements of x0 in + states only
        x0 = x0[plusIdx]'
        # check that it is equal to 1 (or at least close)
        println(sum(x0))

        # compute x-distribution at the time when Y returns to 0
        w = x0*Ψ
        # this can occur in - states only, so find the - states
        minusIdx = [
            Mesh.Fil["p-"];
            repeat(Mesh.Fil["-"]', Mesh.NBases, 1)[:];
            Mesh.Fil["q-"];
        ]
        # then map to the whole state space for plotting
        z = zeros(
            Float64,
            Mesh.NBases*Mesh.NIntervals * approxModel.NPhases +
                sum(approxModel.C.<=0) + sum(approxModel.C.>=0)
        )
        z[minusIdx] = w
        # check that it is equal to 1
        println(sum(z))

        # convert to a distribution object for plotting
        DGProbs = SFFM.Coeffs2Dist(
            Model = approxModel,
            Mesh = Mesh,
            Coeffs = z,
            type="cumulative",
        )

        # plot them
        p2 = plot!(p2,
            DGProbs.x[:],
            DGProbs.distribution[:,:,2][:],
            label = "DG: N_k = "*string(NBases),
            color = colours[NBases],
            xlims = (0,2),
            seriestype = :line,
            linestyle = styles[c],
        )

        p4 = plot!(p4,
            DGProbs.x[:],
            DGProbs.distribution[:,:,4][:],
            label = "DG:  N_k = "*string(NBases),
            color = colours[NBases],
            xlims = (0,2),
            seriestype = :line,
            linestyle = styles[c],
        )
    end

    display(p2)
    display(p4)
    savefig(p2, pwd()*"/examples/paperNumerics/dump/psiPhase10.png")
    savefig(p4, pwd()*"/examples/paperNumerics/dump/psiPhase00.png")
end
