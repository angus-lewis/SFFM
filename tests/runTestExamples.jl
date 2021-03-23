include("../src/SFFM.jl")
using LinearAlgebra, Plots

## define the model(s)
@testset "$modelfile" for modelfile in ["testModel1.jl"; "testModel2.jl"; "testModel3.jl"]
    include(modelfile)

    ## section 4.3: the marginal stationary distribution of X
    ## mesh
    Δ = 0.4
    Nodes = collect(approxBounds[1, 1]:Δ:approxBounds[1, 2])
    NBases = 2
    Basis = "lagrange"
    mesh = SFFM.DGMesh(
        model,
        Nodes,
        NBases,
        Basis = Basis,
    )
    Ψₓ = SFFM.PsiFunX(model=model)
    ξₓ = SFFM.MakeXiX(model=model, Ψ=Ψₓ)

    pₓ, πₓ, Πₓ, Kₓ = SFFM.StationaryDistributionX(model=model, Ψ=Ψₓ, ξ=ξₓ)

    analyticX = (
            pm = [pₓ[:];0;0],
            distribution = πₓ(mesh.CellNodes),
            x = mesh.CellNodes,
            type = "density"
        )

    @test sum(Πₓ(40))-1 < 1e-6

    @testset "DG witn NBases $NBases" for NBases in 1:2
            mesh = SFFM.DGMesh(
                model,
                Nodes,
                NBases,
                Basis = Basis,
            )

            # compute the marginal via DG
            All = SFFM.MakeAll(model = model, mesh = mesh, approxType = "projection")
            @test isapprox(sum(All.B.B,dims=2), zeros(size(All.B.B,1)), atol = sqrt(eps()))
            @test -sum(All.B.BDict["++"],dims=2) ≈ sum(All.B.BDict["+-"],dims=2) + sum(All.B.BDict["+0"],dims=2)
            @test -sum(All.B.BDict["--"],dims=2) ≈ sum(All.B.BDict["-+"],dims=2) + sum(All.B.BDict["-0"],dims=2)
            @test -sum(All.B.BDict["00"],dims=2) ≈ sum(All.B.BDict["0-"],dims=2) + sum(All.B.BDict["0+"],dims=2)
            @test -sum(All.D["++"](s=0),dims=2) ≈ sum(All.D["+-"](s=0),dims=2)
            @test sum(All.D["-+"](s=0),dims=2) ≈ -sum(All.D["--"](s=0),dims=2)

            Ψ = SFFM.PsiFun(D=All.D)
            @test sum(Ψ,dims=2) ≈ ones(size(Ψ,1))

            # the distribution of X when Y first returns to 0
            ξ = SFFM.MakeXi(B=All.B.BDict, Ψ = Ψ)
            @test sum(ξ) ≈ 1

            marginalX, p, K = SFFM.MakeLimitDistMatrices(;
                B = All.B.BDict,
                D = All.D,
                R = All.R.RDict,
                Ψ = Ψ,
                ξ = ξ,
                mesh = mesh,
            )
            
            # convert marginalX to a distribution for plotting
            Dist = SFFM.Coeffs2Dist(
                    model = model,
                    mesh = mesh,
                    Coeffs = marginalX,
                    type="probability",
                )
            @test sum(Dist.pm) + sum(Dist.distribution) ≈ 1
    end
end