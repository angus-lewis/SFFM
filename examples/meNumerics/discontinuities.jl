using Plots, JLD2
include(pwd()*"/src/SFFM.jl")
include(pwd()*"/examples/meNumerics/discontinuitiesModelDef.jl")

@load pwd()*"/examples/meNumerics/discontinuitiesModelSims.jld2" sims_Psi sims_1

## mesh set up
orders = [1;5;11;15;21;25]
errors_1 = []
errors_Psi = []
for order in orders
# order = 3
    println("order = "*string(order))
    Δ = 1 # the grid size; must have kΔ = 1 for some k due to discontinuity in r at 1
    nodes = collect(0:Δ:bounds[1,2])
    dgmesh = SFFM.DGMesh(
        model, 
        nodes, 
        order,
        Basis = "lagrange",
    )
    frapmesh = SFFM.FRAPMesh(
        model, 
        nodes, 
        order,
    )

    simmesh = SFFM.SimMesh(
        model, 
        nodes, 
        1,
    )

    # simulated distributions 
    simprobs_Psi = SFFM.Sims2Dist(
        model, 
        simmesh, 
        sims_Psi, 
        type = "probability"
    )
    simdensity_Psi = SFFM.Sims2Dist(
        model, 
        simmesh, 
        sims_Psi, 
        type = "density"
    )
    simprobs_1 = SFFM.Sims2Dist(
        model,
        simmesh, 
        sims_1, 
        type = "probability"
    )
    simdensity_1 = SFFM.Sims2Dist(
        model,
        simmesh, 
        sims_1, 
        type = "density"
    )

    # DG
    B_DG = SFFM.MakeB(
        model, 
        dgmesh, 
    )

    #ME
    me = SFFM.MakeME(SFFM.CMEParams[order], mean = Δ)
    B_ME = SFFM.MakeBFRAP(model, frapmesh, me)
    # Erlang (this is the erlang which is equivalent to DG)
    erlang = SFFM.MakeErlang(order, mean = Δ)
    B_Erlang = SFFM.MakeBFRAP(model, frapmesh, erlang)
    
    # construct initial condition
    point = 0+eps()
    pointIdx = convert(Int,ceil(point/Δ))
    begin
        V = SFFM.vandermonde(order)
        theNodes = dgmesh.CellNodes[:,pointIdx]
        basisValues = zeros(length(theNodes))
        for n in 1:length(theNodes)
            basisValues[n] = prod(point.-theNodes[[1:n-1;n+1:end]])./prod(theNodes[n].-theNodes[[1:n-1;n+1:end]])
        end
        initpm = [
            zeros(sum(model.C.<=0)) # LHS point mass
            zeros(sum(model.C.>=0)) # RHS point mass
        ]
        initprobs = zeros(Float64,dgmesh.NBases,dgmesh.NIntervals,model.NPhases)
        initprobs[:,pointIdx,1] = basisValues'*V.V*V.V'.*2/Δ
        initdist = (
            pm = initpm,
            distribution = initprobs,
            x = dgmesh.CellNodes,
            type = "density"
        ) # convert to a distribution object so we can apply Dist2Coeffs
        # convert to Coeffs α in the DG context
        x0_DG = SFFM.Dist2Coeffs(model, dgmesh, initdist)
    end
    begin
        initpm = [
            zeros(sum(model.C.<=0)) # LHS point mass
            zeros(sum(model.C.>=0)) # RHS point mass
        ]
        initprobs = zeros(Float64,frapmesh.NBases,frapmesh.NIntervals,model.NPhases)
        initprobs[:,pointIdx,1] = me.a
        initdist = (
            pm = initpm,
            distribution = initprobs,
            x = frapmesh.CellNodes,
            type = "density"
        ) # convert to a distribution object so we can apply Dist2Coeffs
        # convert to Coeffs α in the DG context
        x0_ME = SFFM.Dist2Coeffs( model, frapmesh, initdist, probTransform = false)
    end
    begin
        initpm = [
            zeros(sum(model.C.<=0)) # LHS point mass
            zeros(sum(model.C.>=0)) # RHS point mass
        ]
        initprobs = zeros(Float64,frapmesh.NBases,frapmesh.NIntervals,model.NPhases)
        initprobs[1,pointIdx,1] = 1
        initdist = (
            pm = initpm,
            distribution = initprobs,
            x = frapmesh.CellNodes,
            type = "density"
        ) # convert to a distribution object so we can apply Dist2Coeffs
        # convert to Coeffs α in the DG context
        x0_Erlang = SFFM.Dist2Coeffs( model, frapmesh, initdist, probTransform = false)
    end

    euler(B,x0) = SFFM.EulerDG( B, 1.2, x0, h = 0.0001) 
    x1_DG = euler(B_DG.B, x0_DG)
    x1_ME = euler(B_ME.B, x0_ME)
    x1_Erlang = euler(B_Erlang.B, x0_Erlang)

    x1_DG = SFFM.Coeffs2Dist(
        model,
        dgmesh,
        x1_DG,
        type = "probability",
    )
    x1_ME = SFFM.Coeffs2Dist(
        model,
        frapmesh,
        x1_ME,
        type = "probability",
    )
    x1_Erlang = SFFM.Coeffs2Dist(
        model,
        frapmesh,
        x1_Erlang,
        type = "probability",
    )

    errVec_1 = (
        SFFM.starSeminorm(x1_DG, simprobs_1),
        SFFM.starSeminorm(x1_ME, simprobs_1),
        SFFM.starSeminorm(x1_Erlang, simprobs_1),
    )
    push!(errors_1, errVec_1)

    p = SFFM.PlotSFM(model, mesh = dgmesh, dist = x1_DG,
        color = 1, label = "DG")
    p = SFFM.PlotSFM(model, mesh = frapmesh, dist = x1_ME, 
        color = 2, label = "ME")
    SFFM.PlotSFM!(p, model, frapmesh, x1_Erlang, 
        color = 3, label = "Erlang")
    SFFM.PlotSFM!(p, model, simmesh, simdensity_1, 
        color = 4, label = "Sim")
    p = plot!(title = "approx dist at t=1; order = "*string(order), subplot = 1)
    display(p)

    # the initial condition on Ψ is restricted to + states so find the + states
    plusIdx = [
        dgmesh.Fil["p+"];
        repeat(dgmesh.Fil["+"]', dgmesh.NBases, 1)[:];
        dgmesh.Fil["q+"];
    ]
    # get the elements of x0_DG in + states only
    x0_Psi_DG = x0_DG[plusIdx]'
    x0_Psi_ME = x0_ME[plusIdx]'
    x0_Psi_Erlang = x0_Erlang[plusIdx]'
    # check that it is equal to 1 (or at least close)
    # println(sum(x0_Psi_DG))
    # println(sum(x0_Psi_ME))
    # println(sum(x0_Psi_Erlang))

    ## Psi paths 
    R = SFFM.MakeR( model, dgmesh, approxType = "interpolation")

    ΨFun(B) = SFFM.MakeD( dgmesh, B, R) |> D -> SFFM.PsiFun(D)
    Ψ_DG = ΨFun(B_DG)
    Ψ_ME = ΨFun(B_ME)
    Ψ_Erlang = ΨFun(B_Erlang)

    w_DG = x0_Psi_DG*Ψ_DG
    w_ME = x0_Psi_ME*Ψ_ME
    w_Erlang = x0_Psi_Erlang*Ψ_Erlang

    minusIdx = [
        dgmesh.Fil["p-"];
        repeat(dgmesh.Fil["-"]', dgmesh.NBases, 1)[:];
        dgmesh.Fil["q-"];
    ]
    z_DG = zeros(
        Float64,
        dgmesh.NBases*dgmesh.NIntervals * model.NPhases +
            sum(model.C.<=0) + sum(model.C.>=0)
    )
    z_ME = zeros(
        Float64,
        dgmesh.NBases*dgmesh.NIntervals * model.NPhases +
            sum(model.C.<=0) + sum(model.C.>=0)
    )

    z_Erlang = zeros(
        Float64,
        dgmesh.NBases*dgmesh.NIntervals * model.NPhases +
            sum(model.C.<=0) + sum(model.C.>=0)
    )
    z_DG[minusIdx] = w_DG
    z_ME[minusIdx] = w_ME
    z_Erlang[minusIdx] = w_Erlang

    returnDist_DG =  SFFM.Coeffs2Dist(
        model,
        frapmesh,
        z_DG,
        type = "probability",
    ) 
    returnDist_ME =  SFFM.Coeffs2Dist(
        model,
        frapmesh,
        z_ME,
        type = "probability",
    ) 
    returnDist_Erlang =  SFFM.Coeffs2Dist(
        model,
        frapmesh,
        z_Erlang,
        type = "probability",
    ) 

    errVec_Psi = (
        SFFM.starSeminorm(returnDist_DG, simprobs_Psi),
        SFFM.starSeminorm(returnDist_ME, simprobs_Psi),
        SFFM.starSeminorm(returnDist_Erlang, simprobs_Psi),
    )
    push!(errors_Psi, errVec_Psi)

    # p = SFFM.PlotSFM( model, dgmesh, returnDist_DG,
    #     color = 1, label = "DG")
    p = SFFM.PlotSFM( model, mesh = dgmesh, dist = returnDist_ME, 
        color = 2, label = "ME")
    p = SFFM.PlotSFM!(p, model, frapmesh, returnDist_Erlang, 
        color = 3, label = "Erlang")
    p = SFFM.PlotSFM!(p, model, simmesh, simdensity_Psi, 
        color = 4, label = "Sim")    
    p = plot!(title = "approx Ψ; order = "*string(order), subplot = 1)
    display(p)
end

