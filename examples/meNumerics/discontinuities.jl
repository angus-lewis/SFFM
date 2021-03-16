using Plots, JLD2
include(pwd()*"/src/SFFM.jl")
include(pwd()*"/examples/meNumerics/discontinuitiesModelDef.jl")

@load pwd()*"/examples/meNumerics/discontinuitiesModelSims.jld2" sims_Psi sims_1

## mesh set up
orders = [1;3;5;11;15]
errors_1 = []
errors_Psi = []
# for order in orders
# order = 3
    println("order = "*string(order))
    Δ = 0.5 # the grid size; must have kΔ = 1 for some k due to discontinuity in r at 1
    nodes = collect(0:Δ:bounds[1,2])
    mesh = SFFM.MakeMesh(
        model = model, 
        Nodes = nodes, 
        NBases = order,
        Basis = "lagrange",
    )

    simMesh = SFFM.MakeMesh(
        model = model, 
        Nodes = nodes, 
        NBases = order,
        Basis = "lagrange",
    )

    # simulated distributions 
    simprobs_Psi = SFFM.Sims2Dist(
        model = model, 
        mesh = simMesh, 
        sims = sims_Psi, 
        type = "probability"
    )
    simprobs_1 = SFFM.Sims2Dist(
        model = model,
        mesh = simMesh, 
        sims = sims_1, 
        type = "probability"
    )

    # DG
    M = SFFM.MakeMatrices(
        model = model, 
        mesh = mesh,
    )
    B_DG = SFFM.MakeB(
        model = model, 
        mesh = mesh, 
        Matrices = M, 
    )

    #ME
    me = SFFM.MakeME(SFFM.CMEParams[order], mean = Δ)
    B_ME = SFFM.MakeBFRAP(model = model, mesh = mesh, me = me)
    # Erlang (this is the erlang which is equivalent to DG)
    erlang = SFFM.MakeErlang(order, mean = Δ)
    B_Erlang = SFFM.MakeBFRAP(model = model, mesh = mesh, me = erlang)

    # construct initial condition
    point = 0+eps()
    pointIdx = convert(Int,ceil(point/Δ))
    begin
        theNodes = mesh.CellNodes[:,pointIdx]
        basisValues = zeros(length(theNodes))
        for n in 1:length(theNodes)
            basisValues[n] = prod(point.-theNodes[[1:n-1;n+1:end]])./prod(theNodes[n].-theNodes[[1:n-1;n+1:end]])
        end
        initpm = [
            zeros(sum(model.C.<=0)) # LHS point mass
            zeros(sum(model.C.>=0)) # RHS point mass
        ]
        initprobs = zeros(Float64,mesh.NBases,mesh.NIntervals,model.NPhases)
        initprobs[:,pointIdx,1] = basisValues'*M.Local.V.V*M.Local.V.V'.*2/Δ
        initdist = (
            pm = initpm,
            distribution = initprobs,
            x = mesh.CellNodes,
            type = "density"
        ) # convert to a distribution object so we can apply Dist2Coeffs
        # convert to Coeffs α in the DG context
        x0_DG = SFFM.Dist2Coeffs(model = model, mesh = mesh, Distn = initdist)
    end
    begin
        initpm = [
            zeros(sum(model.C.<=0)) # LHS point mass
            zeros(sum(model.C.>=0)) # RHS point mass
        ]
        initprobs = zeros(Float64,mesh.NBases,mesh.NIntervals,model.NPhases)
        initprobs[:,pointIdx,1] = me.a
        initdist = (
            pm = initpm,
            distribution = initprobs,
            x = mesh.CellNodes,
            type = "density"
        ) # convert to a distribution object so we can apply Dist2Coeffs
        # convert to Coeffs α in the DG context
        x0_ME = SFFM.Dist2Coeffs(model = model, mesh = mesh, Distn = initdist, probTransform = false)
    end
    begin
        initpm = [
            zeros(sum(model.C.<=0)) # LHS point mass
            zeros(sum(model.C.>=0)) # RHS point mass
        ]
        initprobs = zeros(Float64,mesh.NBases,mesh.NIntervals,model.NPhases)
        initprobs[1,pointIdx,1] = 1
        initdist = (
            pm = initpm,
            distribution = initprobs,
            x = mesh.CellNodes,
            type = "density"
        ) # convert to a distribution object so we can apply Dist2Coeffs
        # convert to Coeffs α in the DG context
        x0_Erlang = SFFM.Dist2Coeffs(model = model, mesh = mesh, Distn = initdist, probTransform = false)
    end

    toDist(x) = SFFM.Coeffs2Dist(
        model = model,
        mesh = mesh,
        Coeffs = x,
        type="probability",
    )

    euler(B,x0) = SFFM.EulerDG(D = B, y = 1, x0 = x0, h = 0.001) |> toDist
    x1_DG = euler(B_DG.B, x0_DG)
    x1_ME = euler(B_ME.B, x0_ME)
    x1_Erlang = euler(B_ME.B, x0_Erlang)

    errVec_1 = (
        SFFM.starSeminorm(d1 = x1_DG, d2 = simprobs_1),
        SFFM.starSeminorm(d1 = x1_ME, d2 = simprobs_1),
        SFFM.starSeminorm(d1 = x1_Erlang, d2 = simprobs_1),
    )
    push!(errors_1, errVec_1)

    p = SFFM.PlotSFM(model = model, mesh = mesh, Dist = x1_DG,
        color = 1, label = "DG")
    p = SFFM.PlotSFM!(p;model = model, mesh = mesh, Dist = x1_ME, 
        color = 2, label = "ME")
    SFFM.PlotSFM!(p;model = model, mesh = mesh, Dist = x1_Erlang, 
        color = 3, label = "Erlang")
    SFFM.PlotSFM!(p;model = model, mesh = mesh, Dist = simprobs_1, 
        color = 4, label = "Sim")
    p = plot!(title = "approx dist at t=1; order = "*string(order), subplot = 1)
    display(p)

    # the initial condition on Ψ is restricted to + states so find the + states
    plusIdx = [
        mesh.Fil["p+"];
        repeat(mesh.Fil["+"]', mesh.NBases, 1)[:];
        mesh.Fil["q+"];
    ]
    # get the elements of x0_DG in + states only
    x0_Psi_DG = x0_DG[plusIdx]'
    x0_Psi_ME = x0_ME[plusIdx]'
    x0_Psi_Erlang = x0_Erlang[plusIdx]'
    # check that it is equal to 1 (or at least close)
    println(sum(x0_Psi_DG))
    println(sum(x0_Psi_ME))
    println(sum(x0_Psi_Erlang))

    ## Psi paths 
    R = SFFM.MakeR(model = model, mesh = mesh, approxType = "interpolation")

    ΨFun(B) = SFFM.MakeD(R = R, B = B, model = model, mesh = mesh) |> D -> SFFM.PsiFun(D = D)
    Ψ_DG = ΨFun(B_DG)
    Ψ_ME = ΨFun(B_ME)
    Ψ_Erlang = ΨFun(B_Erlang)

    w_DG = x0_Psi_DG*Ψ_DG
    w_ME = x0_Psi_ME*Ψ_ME
    w_Erlang = x0_Psi_Erlang*Ψ_Erlang

    minusIdx = [
        mesh.Fil["p-"];
        repeat(mesh.Fil["-"]', mesh.NBases, 1)[:];
        mesh.Fil["q-"];
    ]
    z_DG = zeros(
        Float64,
        mesh.NBases*mesh.NIntervals * model.NPhases +
            sum(model.C.<=0) + sum(model.C.>=0)
    )
    z_ME = zeros(
        Float64,
        mesh.NBases*mesh.NIntervals * model.NPhases +
            sum(model.C.<=0) + sum(model.C.>=0)
    )

    z_Erlang = zeros(
        Float64,
        mesh.NBases*mesh.NIntervals * model.NPhases +
            sum(model.C.<=0) + sum(model.C.>=0)
    )
    z_DG[minusIdx] = w_DG
    z_ME[minusIdx] = w_ME
    z_Erlang[minusIdx] = w_Erlang

    returnDist_DG = toDist(z_DG)
    returnDist_ME = toDist(z_ME)
    returnDist_Erlang = toDist(z_Erlang)

    errVec_Psi = (
        SFFM.starSeminorm(d1 = returnDist_DG, d2 = simprobs_Psi),
        SFFM.starSeminorm(d1 = returnDist_ME, d2 = simprobs_Psi),
        SFFM.starSeminorm(d1 = returnDist_Erlang, d2 = simprobs_Psi),
    )
    push!(errors_Psi, errVec_Psi)

    p = SFFM.PlotSFM(model = model, mesh = mesh, Dist = returnDist_DG,
        color = 1, label = "DG")
    p = SFFM.PlotSFM!(p;model = model, mesh = mesh, Dist = returnDist_ME, 
        color = 2, label = "ME")
    p = SFFM.PlotSFM!(p;model = model, mesh = mesh, Dist = returnDist_Erlang, 
        color = 3, label = "Erlang")
    p = SFFM.PlotSFM!(p;model = model, mesh = mesh, Dist = simprobs_Psi, 
        color = 4, label = "Sim")    
    p = plot!(title = "approx Ψ; order = "*string(order), subplot = 1)

    display(p)
    display(order)
# end

# aDist = SFFM.Coeffs2Dist(
#         model = model,
#         mesh = mesh,
#         Coeffs = z_DG,
#         type="density",
#     )
# SFFM.PlotSFM(model = model, mesh = mesh, Dist = aDist,
# color = 1, label = "DG")