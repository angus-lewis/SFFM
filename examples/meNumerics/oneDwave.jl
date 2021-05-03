include(pwd()*"/src/SFFM.jl")

## define a model
T = [0.0]
C = [1]

rfun(x) = x.*0
Rfun(x) = r(x)

r = (
    r = function (x)
        rfun(x)
    end,
    R = function (x)
        Rfun(x)
    end
)

bounds = [0 12; -Inf Inf]
model = SFFM.Model( T, C, r, Bounds = bounds)

orders = [1;3;5;7;11;13;15;21]
errors_1 = []

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
        collect(0:Δtemp/order:bounds[1,2]), 
    )
    simmesh = SFFM.FVMesh(
        model, 
        nodes, 
    )

    trueprobs = zeros(Float64,1,SFFM.NIntervals(frapmesh),SFFM.NPhases(model))
    truepos = convert(Int,ceil((1.2+eps())/Δtemp))
    trueprobs[truepos] = 1
    groundtruth = SFFM.SFFMDistribution(
        [0],
        trueprobs,
        SFFM.CellNodes(frapmesh),
        "probability",
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
    B_FV = SFFM.MakeBFV(model, fvmesh, 3)

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
        initdist = SFFM.SFFMDistribution(
            initpm,
            initprobs,
            SFFM.CellNodes(dgmesh),
            "density",
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
        initdist = SFFM.SFFMDistribution(
            initpm,
            initprobs,
            SFFM.CellNodes(frapmesh),
            "density",
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
        initdist = SFFM.SFFMDistribution(
            initpm,
            initprobs,
            SFFM.CellNodes(frapmesh),
            "density",
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
        initdist = SFFM.SFFMDistribution(
            initpm,
            initprobs,
            SFFM.CellNodes(fvmesh),
            "density",
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
        frapmesh,
        x1_FV,
        type = "probability",
    )

    errVec_1 = (
        SFFM.starSeminorm(x1_DG, groundtruth),
        SFFM.starSeminorm(x1_ME, groundtruth),
        SFFM.starSeminorm(x1_Erlang, groundtruth),
        SFFM.starSeminorm(x1_MEPH, groundtruth),
        SFFM.starSeminorm(x1_FV, groundtruth),
    )
    p = SFFM.PlotSFM(model, frapmesh, x1_ME,
        color = 1, label = "DG")
    display(p)

    p = SFFM.PlotSFM(model, dgmesh, x1_DG,
        color = 1, label = "DG", alpha = 0.25)
    p = SFFM.PlotSFM!(p, model, frapmesh, x1_ME, 
        color = 2, label = "ME", alpha = 0.25)
    # SFFM.PlotSFM!(p, model, frapmesh, x1_Erlang, 
    #     color = 3, label = "Erlang", alpha = 0.25)
    # SFFM.PlotSFM!(p, model, frapmesh, x1_MEPH, 
    #     color = 5, label = "ME-PH", alpha = 0.25)
    SFFM.PlotSFM!(p, model, frapmesh, x1_FV, 
        color = 7, label = "FV", alpha = 0.25)
    p = plot!(title = "approx dist at t=1.2; order = "*string(order), subplot = 1)
    display(p)

    push!(errors_1, errVec_1)
end

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