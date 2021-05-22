include("../../src/SFFM.jl")
using LinearAlgebra, Plots, JLD2, StatsBase

## define the model(s)
include("exampleModelDef.jl")

## section 4.2: Ψ paths

## load simulated Ψ paths
@load pwd()*"/examples/paperNumerics/dump/sims.jld2" sims

## mesh
Δtemp = 0.4
Nodes = collect(approxBounds[1, 1]:Δtemp:approxBounds[1, 2])

nBases = 1
Basis = "lagrange"
mesh = SFFM.DGMesh(
    approxModel,
    Nodes,
    nBases,
    Basis = Basis,
)

## turn sims into a cdf
simprobs = SFFM.Sims2Dist(simModel,mesh,sims,SFFM.SFFMCDF)

## bootstrap to get CI
function bootFun(sims; nBoot = 10)
    l = length(sims.t)
    samplesBoot = Array{Float64,3}(undef,nBoot,6,2)
    for n in 1:nBoot
        sampleIdx = sample(1:l,l)
        tempData = (
            φ = sims.φ[sampleIdx],
            X = sims.X[sampleIdx],
        )
        tempDist = SFFM.Sims2Dist(simModel,mesh,tempData,SFFM.SFFMCDF).distribution[1,1:6,[2;4]]
        samplesBoot[n,:,:] = tempDist
    end
    ql = zeros(6,2)
    qu = zeros(6,2)
    for xpos in 1:6
        for phase in 1:2
            ql[xpos,phase] = quantile(samplesBoot[:,xpos,phase],0.025)
            qu[xpos,phase] = quantile(samplesBoot[:,xpos,phase],0.975)
        end
    end
    return (ql, qu)
end
@time ql, qu = bootFun(sims)



## plot simulations
let
    p2 = plot()
    p4 = plot()

    colours = [:black, :false ,:blue]
    shapes = [:star5, :x, :star7, :+]
    ## DG
    c = -1
    styles = [:dashdot,:dash]
    for nBases in [1;5]
        c = c+2
        mesh = SFFM.DGMesh(
            approxModel,
            Nodes,
            nBases,
            Basis = Basis,
        )

        frapmesh = SFFM.FRAPMesh(
            approxModel,
            Nodes,
            nBases,
        )
        # construct DG matrices
        All = SFFM.MakeAll(approxModel, mesh, approxType = "interpolation")
        Ψ = SFFM.PsiFun(All.D)

        # construct FRAP matrices
        me = SFFM.MakeME(SFFM.CMEParams[nBases], mean = SFFM.Δ(mesh)[1])
        B = SFFM.MakeBFRAP( approxModel, frapmesh, me)
        D = SFFM.MakeD( frapmesh, B, All.R)
        Ψme = SFFM.PsiFun( D)
        
        # construct initial condition
        theNodes = SFFM.CellNodes(mesh)[:,convert(Int,ceil(5/Δtemp))]
        basisValues = zeros(length(theNodes))
        for n in 1:length(theNodes)
            basisValues[n] = prod(5.0.-theNodes[[1:n-1;n+1:end]])./prod(theNodes[n].-theNodes[[1:n-1;n+1:end]])
        end
        initpm = [
            zeros(sum(approxModel.C.<=0)) # LHS point mass
            zeros(sum(approxModel.C.>=0)) # RHS point mass
        ]
        initprobs = zeros(Float64,SFFM.NBases(mesh),SFFM.NIntervals(mesh),SFFM.NPhases(approxModel))
        initprobs[:,convert(Int,ceil(5/Δtemp)),3] = basisValues'*All.Matrices.Local.V.V*All.Matrices.Local.V.V'.*2/Δtemp
        initdist = SFFM.SFFMDensity(
            initpm,
            initprobs,
            SFFM.CellNodes(mesh),
        ) # convert to a distribution object so we can apply Dist2Coeffs
        # convert to Coeffs α in the DG context
        x0 = SFFM.Dist2Coeffs( approxModel, mesh, initdist)
        # the initial condition on Ψ is restricted to + states so find the + states
        plusIdx = [
            mesh.Fil["p+"];
            repeat(mesh.Fil["+"]', SFFM.NBases(mesh), 1)[:];
            mesh.Fil["q+"];
        ]
        # get the elements of x0 in + states only
        x0 = x0[plusIdx]'
        # check that it is equal to 1 (or at least close)
        println(sum(x0))

        # compute x-distribution at the time when Y returns to 0
        w = x0*Ψ

        wme = x0*Ψme
        # this can occur in - states only, so find the - states
        minusIdx = [
            mesh.Fil["p-"];
            repeat(mesh.Fil["-"]', SFFM.NBases(mesh), 1)[:];
            mesh.Fil["q-"];
        ]
        # then map to the whole state space for plotting
        z = zeros(
            Float64,
            SFFM.NBases(mesh)*SFFM.NIntervals(mesh) * SFFM.NPhases(approxModel) +
                sum(approxModel.C.<=0) + sum(approxModel.C.>=0)
        )
        zme = zeros(
            Float64,
            SFFM.NBases(mesh)*SFFM.NIntervals(mesh) * SFFM.NPhases(approxModel) +
                sum(approxModel.C.<=0) + sum(approxModel.C.>=0)
        )
        z[minusIdx] = w
        zme[minusIdx] = wme
        # check that it is equal to 1
        println(sum(z))
        println(sum(zme))

        # convert to a distribution object for plotting
        DGProbs = SFFM.Coeffs2Dist(
            approxModel,
            mesh,
            z,
            SFFM.SFFMCDF,
        )
        
        meProbs = SFFM.Coeffs2Dist(
            approxModel,
            mesh,
            zme,
            SFFM.SFFMCDF,
        )

        # plot them
        p2 = plot!(p2,
            DGProbs.x[:],
            DGProbs.distribution[:,:,2][:],
            label = "DG: N_k = "*string(nBases),
            color = colours[c],
            xlims = (-0.1,2.1),
            seriestype = :scatter,
            # linestyle = styles[c],
            markershape = shapes[c],
            markercolor = colours[c],
            markerstrokecolor = colours[c],
            markersize=6,
        )
        p2 = plot!(p2,
            meProbs.x[:],
            meProbs.distribution[:,:,2][:],
            label = "me: N_k = "*string(nBases),
            color = colours[c],
            xlims = (-0.1,2.1),
            seriestype = :scatter,
            # linestyle = styles[c],
            markershape = shapes[c+1],
            markercolor = colours[c],
            markerstrokecolor = colours[c],
            markersize=6,
        )

        p4 = plot!(p4,
            DGProbs.x[:],
            DGProbs.distribution[:,:,4][:],
            label = "DG:  N_k = "*string(nBases),
            color = colours[c],
            xlims = (-0.1,2.1),
            seriestype = :scatter,
            # linestyle = styles[c],
            markershape = shapes[c],
            markercolor = colours[c],
            markerstrokecolor = colours[c],
            markersize=6,
        )
        p4 = plot!(p4,
            meProbs.x[:],
            meProbs.distribution[:,:,4][:],
            label = "me:  N_k = "*string(nBases),
            color = colours[c],
            xlims = (-0.1,2.1),
            seriestype = :scatter,
            # linestyle = styles[c],
            markershape = shapes[c+1],
            markercolor = colours[c],
            markerstrokecolor = colours[c],
            markersize=6,
        )
    end
    p2 = plot!(p2,
        [1;1]*simprobs.x[1,1:6]',
        [ql[:,1]';qu[:,1]'],
        color=:red,
        marker=:hline,
        markersize=12,
        seriestype=:line,
        label=:false,
        xlabel = "x",
        ylabel = "Cumulative probability",
        windowsize = (600,400),
        grid = false,
        tickfontsize = 12,
        guidefontsize = 14,
        titlefontsize = 20,
        legendfontsize = 12,
        title = "Phase 10",
    )
    p2 = plot!(p2,
        [1;1]*simprobs.x[1,1]',
        [ql[1,1]';qu[1,1]'],
        color=:red,
        marker=:hline,
        markersize=12,
        seriestype=:scatter,
        label="Simulation",
        legend = :bottomright,
    )

    p4 = plot!(p4,
        [1;1]*simprobs.x[1,1:6]',
        [ql[:,2]';qu[:,2]'],
        color=:red,
        marker=:hline,
        markersize=12,
        seriestype=:line,
        label=:false,
        xlabel = "x",
        ylabel = "Cumulative probability",
        windowsize = (600,400),
        grid = false,
        tickfontsize = 12,
        guidefontsize = 14,
        titlefontsize = 20,
        legendfontsize = 12,
        legend = :bottomright,
        title = "Phase 00",
    )
    p4 = plot!(p4,
        [1;1]*simprobs.x[1,1]',
        [ql[1,2]';qu[1,2]'],
        color=:red,
        marker=:hline,
        markersize=12,
        seriestype=:scatter,
        label="Simulation",
        legend = :bottomright,
    )

    display(p2)
    display(p4)
    # savefig(p2, pwd()*"/examples/paperNumerics/dump/psiPhase10.png")
    # savefig(p4, pwd()*"/examples/paperNumerics/dump/psiPhase00.png")
end
