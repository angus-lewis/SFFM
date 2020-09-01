# include("../../src/SFFM.jl")
# using LinearAlgebra, Plots
#
# ## define the model(s)
# include("exampleModelDef.jl")

## section 4.4: the sensitivity of the stationary distribution of X to rates r
Tfun(γ₂) = [
    -(γ₁ + γ₂) γ₂ γ₁ 0;
    β₂ -(γ₁ + β₂) 0 γ₁;
    β₁ 0 -(γ₂ + β₁) γ₂;
    0 β₁ β₂ -(β₂ + β₂);
    ]
for γ₂ in [11;16;22]
    Ttemp = Tfun(γ₂)
    tempModel = SFFM.MakeModel(T = Ttemp, C = C, r = r, Bounds = approxModel.Bounds)
    println("created tempModel with upper bound x=", tempModel.Bounds[1,end])
    ## mesh
    Δ = 0.4
    Nodes = collect(approxBounds[1, 1]:Δ:approxBounds[1, 2])

    Basis = "lagrange"
    NBases = 2
    Mesh = SFFM.MakeMesh(
        Model = tempModel,
        Nodes = Nodes,
        NBases = NBases,
        Basis=Basis,
    )

    # compute the marginal via DG
    All = SFFM.MakeAll(Model = tempModel, Mesh = Mesh, approxType = "projection")
    Ψ = SFFM.PsiFun(D=All.D)

    # the distribution of X when Y first returns to 0
    ξ = SFFM.MakeXi(B=All.B.BDict, Ψ = Ψ)

    marginalX, p, K = SFFM.MakeLimitDistMatrices(;
        B = All.B.BDict,
        D = All.D,
        R = All.R.RDict,
        Ψ = Ψ,
        ξ = ξ,
        Mesh = Mesh,
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
    println("")
end
