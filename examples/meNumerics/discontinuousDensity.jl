using Plots 
include("/Users/a1627293/Documents/SFFM/src/SFFM.jl")

## define a model
T = [-0.05 0.05;
    1 -1]
C = [1; -24]

r₁(x) = - (x.>1) + (x.<=1)
r₂(x) = x.*0
R₁(x) = (x.>1).*(x.-1) - (x.<=1).*x

r = (
    r = function (x)
        [r₁(x) r₂(x)]
    end,
    R = function (x)
        [R₁(x) r₂(x)]
    end
)

bounds = [0 6; -Inf Inf]
model = SFFM.Model(T = T, C = C, r = r, Bounds = bounds)

## mesh set up
for order in [1;3;5]
# order = 3
    display(order)
    Δ = 0.5 # the grid size; must have kΔ = 1 for some k due to discontinuity in r at 1
    nodes = collect(0:Δ:bounds[1,2])
    mesh = SFFM.MakeMesh(
        model = model, 
        Nodes = nodes, 
        NBases = order,
        Basis = "lagrange",
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

    toDist(x) = SFFM.Coeffs2Dist(
        model = model,
        mesh = mesh,
        Coeffs = x,
        type="probability",
    )

    euler(B,x0) = SFFM.EulerDG(D = B, y = 1, x0 = x0, h = 0.001) |> toDist
    x1_DG = euler(B_DG.B, x0_DG)
    x1_ME = euler(B_ME.B, x0_ME)

    p = SFFM.PlotSFM(model = model, mesh = mesh, Dist = x1_DG,
        color = 1, label = "DG")
    p = SFFM.PlotSFM!(p;model = model, mesh = mesh, Dist = x1_ME, 
        color = 2, label = "ME")
    p = plot!(title = "approx dist at t=1; order = "*string(order))
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
    # check that it is equal to 1 (or at least close)
    println(sum(x0_Psi_DG))
    println(sum(x0_Psi_ME))

    ## Psi paths 
    R = SFFM.MakeR(model = model, mesh = mesh, approxType = "interpolation")

    ΨFun(B) = SFFM.MakeD(R = R, B = B, model = model, mesh = mesh) |> D -> SFFM.PsiFun(D = D)
    Ψ_DG = ΨFun(B_DG)
    Ψ_ME = ΨFun(B_ME)

    w_DG = x0_Psi_DG*Ψ_DG
    w_ME = x0_Psi_ME*Ψ_ME

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
    z_DG[minusIdx] = w_DG
    z_ME[minusIdx] = w_ME

    returnDist_DG = toDist(z_DG)
    returnDist_ME = toDist(z_ME)

    p = SFFM.PlotSFM(model = model, mesh = mesh, Dist = returnDist_DG,
        color = 1, label = "DG")
    p = SFFM.PlotSFM!(p;model = model, mesh = mesh, Dist = returnDist_ME, 
        color = 2, label = "ME")
    p = plot!(title = "approx Ψ; order = "*string(order))

    display(p)
    display(order)
end

# aDist = SFFM.Coeffs2Dist(
#         model = model,
#         mesh = mesh,
#         Coeffs = z_DG,
#         type="density",
#     )
# SFFM.PlotSFM(model = model, mesh = mesh, Dist = aDist,
# color = 1, label = "DG")