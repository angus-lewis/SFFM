include("../../src/SFFM.jl")
using LinearAlgebra, Plots, JLD2, StatsBase

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
mesh = SFFM.MakeMesh(
    model = approxModel,
    Nodes = Nodes,
    NBases = NBases,
    Basis = Basis,
)

## turn sims into a cdf
simprobs = SFFM.Sims2Dist(model=simModel,mesh=mesh,sims=sims,type="cumulative")

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
        tempDist = SFFM.Sims2Dist(model=simModel,mesh=mesh,sims=tempData,type="cumulative").distribution[1,1:6,[2;4]]
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
    shapes = [:star5, :rtriangle, :star7, :ltriangle]
    ## DG
    c = 0
    styles = [:dashdot,:dash]
    for NBases in 1:2:3
        c = c+1
        mesh = SFFM.MakeMesh(
            model = approxModel,
            Nodes = Nodes,
            NBases = NBases,
            Basis = Basis,
        )
        # construct DG matrices
        All = SFFM.MakeAll(model = approxModel, mesh = mesh, approxType = "interpolation")
        Ψ = SFFM.PsiFun(D=All.D)

        # construct FRAP matrices
        me = SFFM.MakeME(SFFM.CMEParams[NBases])
        B = SFFM.MakeBFRAP(model=approxModel,mesh=mesh,me=me)
        D = SFFM.MakeD(R=All.R,B=B,model=approxModel,mesh=mesh)
        Ψme = SFFM.PsiFun(D=D)
        
        # construct initial condition
        theNodes = mesh.CellNodes[:,convert(Int,ceil(5/Δ))]
        basisValues = zeros(length(theNodes))
        for n in 1:length(theNodes)
            basisValues[n] = prod(5.0.-theNodes[[1:n-1;n+1:end]])./prod(theNodes[n].-theNodes[[1:n-1;n+1:end]])
        end
        initpm = [
            zeros(sum(approxModel.C.<=0)) # LHS point mass
            zeros(sum(approxModel.C.>=0)) # RHS point mass
        ]
        initprobs = zeros(Float64,mesh.NBases,mesh.NIntervals,approxModel.NPhases)
        initprobs[:,convert(Int,ceil(5/Δ)),3] = basisValues'*All.Matrices.Local.V.V*All.Matrices.Local.V.V'.*2/Δ
        initdist = (
            pm = initpm,
            distribution = initprobs,
            x = mesh.CellNodes,
            type = "density"
        ) # convert to a distribution object so we can apply Dist2Coeffs
        # convert to Coeffs α in the DG context
        x0 = SFFM.Dist2Coeffs(model = approxModel, mesh = mesh, Distn = initdist)
        # the initial condition on Ψ is restricted to + states so find the + states
        plusIdx = [
            mesh.Fil["p+"];
            repeat(mesh.Fil["+"]', mesh.NBases, 1)[:];
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
            repeat(mesh.Fil["-"]', mesh.NBases, 1)[:];
            mesh.Fil["q-"];
        ]
        # then map to the whole state space for plotting
        z = zeros(
            Float64,
            mesh.NBases*mesh.NIntervals * approxModel.NPhases +
                sum(approxModel.C.<=0) + sum(approxModel.C.>=0)
        )
        zme = zeros(
            Float64,
            mesh.NBases*mesh.NIntervals * approxModel.NPhases +
                sum(approxModel.C.<=0) + sum(approxModel.C.>=0)
        )
        z[minusIdx] = w
        zme[minusIdx] = wme
        # check that it is equal to 1
        println(sum(z))
        println(sum(zme))

        # convert to a distribution object for plotting
        DGProbs = SFFM.Coeffs2Dist(
            model = approxModel,
            mesh = mesh,
            Coeffs = z,
            type="cumulative",
        )
        plot(zme)
        display(plot!(z))
        meProbs = SFFM.Coeffs2Dist(
            model = approxModel,
            mesh = mesh,
            Coeffs = zme,
            type="cumulative",
        )

        # plot them
        p2 = plot!(p2,
            DGProbs.x[:],
            DGProbs.distribution[:,:,2][:],
            label = "DG: N_k = "*string(NBases),
            color = colours[NBases],
            xlims = (-0.1,2.1),
            seriestype = :scatter,
            # linestyle = styles[c],
            markershape = shapes[NBases],
            markercolor = colours[NBases],
            markerstrokecolor = colours[NBases],
            markersize=6,
        )
        p2 = plot!(p2,
            meProbs.x[:],
            meProbs.distribution[:,:,2][:],
            label = "me: N_k = "*string(NBases),
            color = colours[NBases],
            xlims = (-0.1,2.1),
            seriestype = :scatter,
            # linestyle = styles[c],
            markershape = shapes[NBases+1],
            markercolor = colours[NBases],
            markerstrokecolor = colours[NBases],
            markersize=6,
        )

        p4 = plot!(p4,
            DGProbs.x[:],
            DGProbs.distribution[:,:,4][:],
            label = "DG:  N_k = "*string(NBases),
            color = colours[NBases],
            xlims = (-0.1,2.1),
            seriestype = :scatter,
            # linestyle = styles[c],
            markershape = shapes[NBases],
            markercolor = colours[NBases],
            markerstrokecolor = colours[NBases],
            markersize=6,
        )
        p4 = plot!(p4,
            meProbs.x[:],
            meProbs.distribution[:,:,4][:],
            label = "me:  N_k = "*string(NBases),
            color = colours[NBases],
            xlims = (-0.1,2.1),
            seriestype = :scatter,
            # linestyle = styles[c],
            markershape = shapes[NBases+1],
            markercolor = colours[NBases],
            markerstrokecolor = colours[NBases],
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
