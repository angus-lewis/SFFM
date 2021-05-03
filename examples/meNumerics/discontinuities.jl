# using Plots, JLD2
include(pwd()*"/src/SFFM.jl")
include(pwd()*"/examples/meNumerics/discontinuitiesModelDef.jl")

@load pwd()*"/examples/meNumerics/discontinuitiesModelSims.jld2" sims_Psi sims_1

## mesh set up
orders = [1;3;5;7;11;13;15;21]
errors_1 = []
errors_Psi = []
errors_Pi = []
Δtemp = 1/2 # the grid size; must have kΔ = 1 for some k due to discontinuity in r at 1
nodes = collect(0:Δtemp:bounds[1,2])
for order in orders
# order = 5
    println("order = "*string(order))
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
    fvmesh = SFFM.FVMesh(
        model, 
        nodes, 
    )

    # simulated distributions 
    simprobs_Psi = SFFM.Sims2Dist(
        model, 
        fvmesh, 
        sims_Psi, 
        type = "probability"
    )
    simdensity_Psi = SFFM.Sims2Dist(
        model, 
        fvmesh, 
        sims_Psi, 
        type = "density"
    )
    simprobs_1 = SFFM.Sims2Dist(
        model,
        fvmesh, 
        sims_1, 
        type = "probability"
    )
    simdensity_1 = SFFM.Sims2Dist(
        model,
        fvmesh, 
        sims_1, 
        type = "density"
    )

    # DG
    B_DG = SFFM.MakeB(model, dgmesh)
    #ME
    me = SFFM.MakeME(SFFM.CMEParams[order], mean = Δtemp)
    B_ME = SFFM.MakeBFRAP(model, frapmesh, me)
    # Erlang (this is the erlang which is equivalent to DG)
    erlang = SFFM.MakeErlang(order, mean = Δtemp)
    B_Erlang = SFFM.MakeBFRAP(model, frapmesh, erlang)
    # meph (this is the erlang treated as an ME)
    meph = SFFM.ME(erlang.a, erlang.S, erlang.s; D = SFFM.erlangDParams[string(order)])
    B_MEPH = SFFM.MakeBFRAP(model, frapmesh, meph)
    # FVM
    B_FV = SFFM.MakeBFV(model, fvmesh, order)
    
    # construct initial condition
    point = 0+eps()
    pointIdx = convert(Int,ceil(point/Δtemp))
    begin
        V = SFFM.vandermonde(order)
        theNodes = SFFM.CellNodes(dgmesh)[:,pointIdx]
        basisValues = zeros(length(theNodes))
        for n in 1:length(theNodes)
            basisValues[n] = prod(point.-theNodes[[1:n-1;n+1:end]])./prod(theNodes[n].-theNodes[[1:n-1;n+1:end]])
        end
        initpm = [
            zeros(sum(model.C.<=0)) # LHS point mass
            zeros(sum(model.C.>=0)) # RHS point mass
        ]
        initprobs = zeros(Float64,SFFM.NBases(dgmesh),SFFM.NIntervals(dgmesh),SFFM.NPhases(model))
        initprobs[:,pointIdx,1] = basisValues'*V.V*V.V'.*2/Δtemp
        initdist = (
            pm = initpm,
            distribution = initprobs,
            x = SFFM.CellNodes(dgmesh),
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
        initprobs = zeros(Float64,SFFM.NBases(frapmesh),SFFM.NIntervals(frapmesh),SFFM.NPhases(model))
        initprobs[:,pointIdx,1] = me.a
        initdist = (
            pm = initpm,
            distribution = initprobs,
            x = SFFM.CellNodes(frapmesh),
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
        initprobs = zeros(Float64,SFFM.NBases(frapmesh),SFFM.NIntervals(frapmesh),SFFM.NPhases(model))
        initprobs[1,pointIdx,1] = 1
        initdist = (
            pm = initpm,
            distribution = initprobs,
            x = SFFM.CellNodes(frapmesh),
            type = "density"
        ) # convert to a distribution object so we can apply Dist2Coeffs
        # convert to Coeffs α in the DG context
        x0_Erlang = SFFM.Dist2Coeffs( model, frapmesh, initdist, probTransform = false)
    end
    begin
        initpm = [
            zeros(sum(model.C.<=0)) # LHS point mass
            zeros(sum(model.C.>=0)) # RHS point mass
        ]
        initprobs = zeros(Float64,SFFM.NBases(fvmesh),SFFM.NIntervals(fvmesh),SFFM.NPhases(model))
        initprobs[1,pointIdx,1] = 1
        initdist = (
            pm = initpm,
            distribution = initprobs,
            x = SFFM.CellNodes(fvmesh),
            type = "density"
        ) # convert to a distribution object so we can apply Dist2Coeffs
        # convert to Coeffs α in the DG context
        x0_FV = SFFM.Dist2Coeffs( model, fvmesh, initdist, probTransform = false)
    end

    euler(B,x0) = SFFM.EulerDG( B, 1.2, x0, h = 0.0001) 
    x1_DG = euler(B_DG.B, x0_DG)
    x1_ME = euler(B_ME.B, x0_ME)
    x1_Erlang = euler(B_Erlang.B, x0_Erlang)
    x1_MEPH = euler(B_MEPH.B, x0_Erlang)
    x1_FV = euler(B_FV.B, x0_FV)

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
    x1_MEPH = SFFM.Coeffs2Dist(
        model,
        frapmesh,
        x1_MEPH,
        type = "probability",
    )
    x1_FV = SFFM.Coeffs2Dist(
        model,
        fvmesh,
        x1_FV,
        type = "probability",
    )

    errVec_1 = (
        SFFM.starSeminorm(x1_DG, simprobs_1),
        SFFM.starSeminorm(x1_ME, simprobs_1),
        SFFM.starSeminorm(x1_Erlang, simprobs_1),
        SFFM.starSeminorm(x1_MEPH, simprobs_1),
        SFFM.starSeminorm(x1_FV, simprobs_1),
    )
    push!(errors_1, errVec_1)

    # p = SFFM.PlotSFM(model, mesh = dgmesh, dist = x1_DG,
    #     color = 1, label = "DG")
    # p = SFFM.PlotSFM!(p, model, frapmesh, x1_ME, 
    #     color = 2, label = "ME")
    # SFFM.PlotSFM!(p, model, frapmesh, x1_Erlang, 
    #     color = 3, label = "Erlang")
    # SFFM.PlotSFM!(p, model, frapmesh, simprobs_1, 
    #     color = 4, label = "Sim")
    # SFFM.PlotSFM!(p, model, frapmesh, x1_MEPH, 
    #     color = 5, label = "ME-PH")
    # SFFM.PlotSFM!(p, model, fvmesh, x1_FV, 
    #     color = 7, label = "FV")
    # p = plot!(title = "approx dist at t=1.2; order = "*string(order), subplot = 1)
    # display(p)

    # the initial condition on Ψ is restricted to + states so find the + states
    plusIdx = [
        dgmesh.Fil["p+"];
        repeat(dgmesh.Fil["+"]', SFFM.NBases(dgmesh), 1)[:];
        dgmesh.Fil["q+"];
    ]
    plusIdxFV = [
        fvmesh.Fil["p+"];
        repeat(fvmesh.Fil["+"]', SFFM.NBases(fvmesh), 1)[:];
        fvmesh.Fil["q+"];
    ]
    # get the elements of x0_DG in + states only
    x0_Psi_DG = x0_DG[plusIdx]'
    x0_Psi_ME = x0_ME[plusIdx]'
    x0_Psi_Erlang = x0_Erlang[plusIdx]'
    x0_Psi_FV = x0_FV[plusIdxFV]'
    # check that it is equal to 1 (or at least close)
    # println(sum(x0_Psi_DG))
    # println(sum(x0_Psi_ME))
    # println(sum(x0_Psi_Erlang))

    ## Psi paths 
    R = SFFM.MakeR( model, dgmesh, approxType = "interpolation")
    R_FV = SFFM.MakeR( model, fvmesh, approxType = "interpolation")

    ΨFun(B) = SFFM.MakeD(dgmesh, B, R) |> D -> SFFM.PsiFun(D)
    Ψ_DG = ΨFun(B_DG)
    Ψ_ME = ΨFun(B_ME)
    Ψ_Erlang = ΨFun(B_Erlang)
    Ψ_MEPH = ΨFun(B_MEPH)
    Ψ_FV = SFFM.MakeD(fvmesh, B_FV, R_FV) |> D -> SFFM.PsiFun(D)

    w_DG = x0_Psi_DG*Ψ_DG
    w_ME = x0_Psi_ME*Ψ_ME
    w_Erlang = x0_Psi_Erlang*Ψ_Erlang
    w_MEPH = x0_Psi_Erlang*Ψ_MEPH
    w_FV = x0_Psi_FV*Ψ_FV

    minusIdx = [
        dgmesh.Fil["p-"];
        repeat(dgmesh.Fil["-"]', SFFM.NBases(dgmesh), 1)[:];
        dgmesh.Fil["q-"];
    ]
    minusIdxFV = [
        fvmesh.Fil["p-"];
        repeat(fvmesh.Fil["-"]', SFFM.NBases(fvmesh), 1)[:];
        fvmesh.Fil["q-"];
    ]
    z_DG = zeros(
        Float64,
        SFFM.NBases(dgmesh)*SFFM.NIntervals(dgmesh) * SFFM.NPhases(model) +
            sum(model.C.<=0) + sum(model.C.>=0)
    )
    z_ME = zeros(
        Float64,
        SFFM.NBases(dgmesh)*SFFM.NIntervals(dgmesh) * SFFM.NPhases(model) +
            sum(model.C.<=0) + sum(model.C.>=0)
    )
    z_Erlang = zeros(
        Float64,
        SFFM.NBases(dgmesh)*SFFM.NIntervals(dgmesh) * SFFM.NPhases(model) +
            sum(model.C.<=0) + sum(model.C.>=0)
    )
    z_MEPH = zeros(
        Float64,
        SFFM.NBases(dgmesh)*SFFM.NIntervals(dgmesh) * SFFM.NPhases(model) +
            sum(model.C.<=0) + sum(model.C.>=0)
    )
    z_FV = zeros(
        Float64,
        SFFM.NBases(fvmesh)*SFFM.NIntervals(fvmesh) * SFFM.NPhases(model) +
            sum(model.C.<=0) + sum(model.C.>=0)
    )

    z_DG[minusIdx] = w_DG
    z_ME[minusIdx] = w_ME
    z_Erlang[minusIdx] = w_Erlang
    z_MEPH[minusIdx] = w_MEPH
    z_FV[minusIdxFV] = w_FV

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
    returnDist_MEPH =  SFFM.Coeffs2Dist(
        model,
        frapmesh,
        z_MEPH,
        type = "probability",
    )
    returnDist_FV =  SFFM.Coeffs2Dist(
        model,
        fvmesh,
        z_FV,
        type = "probability",
    ) 

    errVec_Psi = (
        SFFM.starSeminorm(returnDist_DG, simprobs_Psi),
        SFFM.starSeminorm(returnDist_ME, simprobs_Psi),
        SFFM.starSeminorm(returnDist_Erlang, simprobs_Psi),
        SFFM.starSeminorm(returnDist_MEPH, simprobs_Psi),
        SFFM.starSeminorm(returnDist_FV, simprobs_Psi),
    )
    push!(errors_Psi, errVec_Psi)

    # p = SFFM.PlotSFM(model, mesh = dgmesh, dist = returnDist_DG,
    #     color = 1, label = "DG")
    # p = SFFM.PlotSFM!(p, model, dgmesh, returnDist_ME, 
    #     color = 2, label = "ME")
    # p = SFFM.PlotSFM!(p, model, frapmesh, returnDist_Erlang, 
    #     color = 3, label = "Erlang")
    # p = SFFM.PlotSFM!(p, model, frapmesh, simprobs_Psi, 
    #     color = 4, label = "Sim")   
    # p = SFFM.PlotSFM!(p, model, frapmesh, returnDist_MEPH, 
    #     color = 5, label = "MEPH") 
    # p = SFFM.PlotSFM!(p, model, fvmesh, returnDist_FV, 
    #     color = 7, label = "FV") 
    # p = plot!(title = "approx Ψ; order = "*string(order), subplot = 1)
    # display(p)
end

fvmesh = SFFM.FVMesh(
    model, 
    collect(model.Bounds[1,1]:0.1:model.Bounds[1,2]), 
)

simdensity_Psi = SFFM.Sims2Dist(
    model, 
    fvmesh, 
    sims_Psi, 
    type = "probability"
)
q = SFFM.PlotSFM(model, mesh = fvmesh, dist = simdensity_Psi)
display(q)

q = plot(xlabel = "order", ylabel = "log10 error", title = "error for Psi")
methodNames = ["DG";"ME";"Er";"Er as ME"; "FV"]
shapes = [:circle, :utriangle, :x, :+, :square]
for whichOrder in 2:length(orders)
# order = 3
    for whichMethod in 1:length(methodNames)
        plot!(
            q, 
            [orders[whichOrder-1];orders[whichOrder]], 
            [
                log10(errors_Psi[whichOrder-1][whichMethod]);
                log10(errors_Psi[whichOrder][whichMethod])
            ], 
            label = whichOrder == 2 && methodNames[whichMethod],
            colour = whichMethod,
            markershape = shapes[whichMethod],
        )
    end
end
display(q)

simdensity_1 = SFFM.Sims2Dist(
    model, 
    fvmesh, 
    sims_1, 
    type = "probability"
)
q = SFFM.PlotSFM(model, mesh = fvmesh, dist = simdensity_1)
display(q)

q = plot(
    xlabel = "order", 
    ylabel = "log10 error", 
    title = "error for t=1.2", 
    legend = :bottomleft
)
for whichOrder in 2:length(orders)
# order = 3
    for whichMethod in 1:length(methodNames)
        plot!(
            q, 
            [orders[whichOrder-1];orders[whichOrder]], 
            [
                log10(errors_1[whichOrder-1][whichMethod]);
                log10(errors_1[whichOrder][whichMethod])
            ], 
            label = whichOrder == 2 && methodNames[whichMethod],
            colour = whichMethod,
            markershape = shapes[whichMethod],
        )
    end
end
display(q)