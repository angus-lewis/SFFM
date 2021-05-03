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
    Ψₓ = SFFM.PsiFunX(model)
    ξₓ = SFFM.MakeXiX(model, Ψₓ)

    pₓ, πₓ, Πₓ, Kₓ = SFFM.StationaryDistributionX(model, Ψₓ, ξₓ)

    analyticX = (
            pm = [pₓ[:];0;0],
            distribution = πₓ(SFFM.CellNodes(mesh)),
            x = SFFM.CellNodes(mesh),
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
            All = SFFM.MakeAll( model, mesh, approxType = "projection")
            @test isapprox(sum(All.B.B,dims=2), zeros(size(All.B.B,1)), atol = sqrt(eps()))
            @test -sum(All.B.BDict["++"],dims=2) ≈ sum(All.B.BDict["+-"],dims=2) + sum(All.B.BDict["+0"],dims=2)
            @test -sum(All.B.BDict["--"],dims=2) ≈ sum(All.B.BDict["-+"],dims=2) + sum(All.B.BDict["-0"],dims=2)
            @test -sum(All.B.BDict["00"],dims=2) ≈ sum(All.B.BDict["0-"],dims=2) + sum(All.B.BDict["0+"],dims=2)
            @test -sum(All.D["++"](s=0),dims=2) ≈ sum(All.D["+-"](s=0),dims=2)
            @test sum(All.D["-+"](s=0),dims=2) ≈ -sum(All.D["--"](s=0),dims=2)

            Ψ = SFFM.PsiFun(All.D)
            @test sum(Ψ,dims=2) ≈ ones(size(Ψ,1))

            # the distribution of X when Y first returns to 0
            ξ = SFFM.MakeXi(All.B.BDict, Ψ)
            @test sum(ξ) ≈ 1

            marginalX, p, K = SFFM.MakeLimitDistMatrices(
                All.B.BDict,
                All.D,
                All.R.RDict,
                Ψ,
                ξ,
                mesh,
                model,
            )
            
            # convert marginalX to a distribution for plotting
            Dist = SFFM.Coeffs2Dist(
                    model,
                    mesh,
                    marginalX,
                    type="probability",
                )
            @test sum(Dist.pm) + sum(Dist.distribution) ≈ 1
    end
end