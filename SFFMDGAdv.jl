function MakeBlockDiagonalMatrixR(;
    Model::NamedTuple{(:T, :C, :r, :IsBounded, :Bounds, :NPhases)},
    Mesh::NamedTuple{
        (:NBases, :CellNodes, :Fil, :Δ, :NIntervals, :MeshArray, :Nodes, :TotalNBases),
    },
    Blocks,
    Factors::Array,
)
    # MakeBlockDiagonalMatrix makes a matrix from diagonal block elements
    # inputs:
    # Model - A Model tuple from MakeModel
    # Mesh - A tuple from MakeMesh
    # Blocks - Mesh.NBases×Mesh.NBases Array{Float64}, blocks to put along the
    #           diagonal
    # Factors - Mesh.NIntervals×1 Array, factors which multiply blocks
    # output:
    # BlockMatrix - Mesh.TotalNBases×Mesh.TotalNBases Array{Float64,2}, the
    #             block matrix

    BlockMatrix = zeros(Float64, Mesh.TotalNBases, Mesh.TotalNBases, Model.NPhases)
    for i = 1:Mesh.NIntervals, j = 1:Model.NPhases
        idx = (1:Mesh.NBases) .+ (i - 1) * Mesh.NBases
        BlockMatrix[idx, idx, j] = Blocks(Mesh.CellNodes[:, i], j) * Factors[i]
    end
    return (BlockMatrix = BlockMatrix)
end

function MakeFluxMatrixR(;
    Mesh::NamedTuple{
        (:NBases, :CellNodes, :Fil, :Δ, :NIntervals, :MeshArray, :Nodes, :TotalNBases),
    },
    Model::NamedTuple{(:T, :C, :r, :IsBounded, :Bounds, :NPhases)},
    Phi,
)
    # MakeFluxMatrix creates the global block tridiagonal flux matrix for the
    # lagrange basis
    # inputs:
    # Mesh - a Mesh tuple from MakeMesh
    # Model - a Model tuple from MakeModel
    # outputs:
    # F - TotalNBases×TotalNBases×NPhases Array{Float64,3}, global flux matrix

    ## Create the blocks
    PosDiagBlock = -Phi[end, :] * Phi[end, :]'
    NegDiagBlock = Phi[1, :] * Phi[1, :]'
    UpDiagBlock = Phi[end, :] * Phi[1, :]'
    LowDiagBlock = -Phi[1, :] * Phi[end, :]'

    ## Construct global block diagonal matrix
    F = zeros(Float64, Mesh.TotalNBases, Mesh.TotalNBases, Model.NPhases)
    for i = 1:Model.NPhases
        for k = 1:Mesh.NIntervals
            idx = (1:Mesh.NBases) .+ (k - 1) * Mesh.NBases
            if Model.C[i] > 0
                xright = Mesh.CellNodes[end, k]
                R = 1.0 ./ abs(Model.r.r(xright)[i])
                F[idx, idx, i] = PosDiagBlock * R
            elseif Model.C[i] < 0
                xleft = Mesh.CellNodes[1, k]
                R = 1.0 ./ abs(Model.r.r(xleft)[i])
                F[idx, idx, i] = NegDiagBlock * R
            end # end if C[i]
            if k > 1
                idxup = (1:Mesh.NBases) .+ (k - 2) * Mesh.NBases
                if Model.C[i] > 0
                    xright = Mesh.CellNodes[end, k-1]
                    R = 1.0 ./ abs(Model.r.r(xright)[i])
                    η = (Mesh.Δ[k] / Mesh.NBases) / (Mesh.Δ[k-1] / Mesh.NBases)
                    F[idxup, idx, i] = UpDiagBlock * η * R
                elseif Model.C[i] < 0
                    xleft = Mesh.CellNodes[1, k]
                    R = 1.0 ./ abs(Model.r.r(xleft)[i])
                    η = (Mesh.Δ[k-1] / Mesh.NBases) / (Mesh.Δ[k] / Mesh.NBases)
                    F[idx, idxup, i] = LowDiagBlock * η * R
                end # end if C[i]
            end # end if k>1
        end # for k in ...
    end # end for i in NPhases

    # ## Check if bounded and make sure no mass can leave
    # if Model.IsBounded
    #     for i = 1:Model.NPhases
    #         if Model.C[i] < 0
    #             idx = 1:Mesh.NBases
    #             F[idx, idx, i] .= 0
    #         elseif Model.C[i] > 0
    #             idx = (1:Mesh.NBases) .+ (Mesh.NIntervals - 1) * Mesh.NBases
    #             F[idx, idx, i] .= 0
    #         end # end if C[i]
    #     end # end for i ...
    # end # end if IsBounded

    return (F = F)
end

function MakeMatricesR(;
    Model::NamedTuple{(:T, :C, :r, :IsBounded, :Bounds, :NPhases)},
    Mesh::NamedTuple{
        (:NBases, :CellNodes, :Fil, :Δ, :NIntervals, :MeshArray, :Nodes, :TotalNBases),
    },
    Basis::String = "legendre",
)
    # Creates the Local and global mass, stiffness and flux
    # matrices.
    # inputs:
    # Model - A model tuple from MakeModel
    # Mesh - A mesh tuple from MakeMesh
    # Basis - A string specifying whether to use the lagrange or legendre basis
    #         representations
    # outputs: A tuple of tuples with fields Local and Global
    # Global - A tuple with fields
    #   .G - TotalNBases×TotalNBases Array{Float64}, global stiffness matrix
    #   .M - TotalNBases×TotalNBases Array{Float64}, global mass matrix
    #   .MInv - the inverse of Global.M
    #   .F - TotalNBases×TotalNBases×NPhases Array{Float64,3} global flux matrix
    #   .Q - TotalNBases×TotalNBases×NPhases Array{Float64}, global DG flux
    #         operator
    # Local - A tuple with fields
    #   .G - NBases×NBases Array{Float64}, Local stiffness matrix
    #   .M - NBases×NBases Array{Float64}, Local mass matrix
    #   .MInv - the inverse of Local.M
    #   .V - tuple used to make M, G, Minv, as output from SFFM.vandermonde

    ## Construct blocks
    V = vandermonde(NBases = Mesh.NBases)
    if Basis == "legendre"
        MLocal = function (x::Array{Float64}, i::Int)
            # Numerical integration of ϕᵢ(x)|r(x)|ϕⱼ(x) over Dk with Gauss-Lobatto
            # quadrature
            # Inputs:
            #   - x a vector of Gauss-Lobatto points on Dk
            #   - i a phase
            V.V' * LinearAlgebra.diagm(0 => V.w ./ abs.(Model.r.r(x)[:, i])) * V.V
        end
        GLocal = function (x::Array{Float64}, i::Int)
            # Numerical integration of ϕᵢ(x)|r(x)|ϕⱼ'(x) over Dk with Gauss-Lobatto
            # quadrature
            # Inputs:
            #   - x a vector of Gauss-Lobatto points on Dk
            #   - i a phase
            V.V' * LinearAlgebra.diagm(0 => V.w ./ abs.(Model.r.r(x)[:, i])) * V.D
        end
        MInvLocal = function (x::Array{Float64}, i::Int)
            MLocal(x, i)^-1
        end
        Phi = V.V[[1; end], :]

    elseif Basis == "lagrange"
        MLocal = function (x::Array{Float64}, i::Int)
            # Numerical integration of ϕᵢ(x)|r(x)|ϕⱼ(x) over Dk with Gauss-Lobatto
            # quadrature
            # Inputs:
            #   - x a vector of Gauss-Lobatto points on Dk
            #   - i a phase
            LinearAlgebra.diagm(V.w ./ abs.(Model.r.r(x)[:, i]))
        end
        GLocal = function (x::Array{Float64}, i::Int)
            # Numerical integration of ϕᵢ(x)|r(x)|ϕⱼ'(x) over Dk with Gauss-Lobatto
            # quadrature
            # Inputs:
            #   - x a vector of Gauss-Lobatto points on Dk
            #   - i a phase
            V.inv' * V.inv * MLocal(x, i) * V.inv' * V.D
        end
        MInvLocal = function (x::Array{Float64}, i::Int)
            MLocal(x, i)^-1
        end
        Phi = (V.inv*V.V)[[1; end], :]
    end

    ## Assemble into block diagonal matrices
    G = SFFM.MakeBlockDiagonalMatrixR(
        Model = Model,
        Mesh = Mesh,
        Blocks = GLocal,
        Factors = ones(Mesh.NIntervals),
    )
    M = SFFM.MakeBlockDiagonalMatrixR(
        Model = Model,
        Mesh = Mesh,
        Blocks = MLocal,
        Factors = Mesh.Δ * 0.5,
    )
    MInv = SFFM.MakeBlockDiagonalMatrixR(
        Model = Model,
        Mesh = Mesh,
        Blocks = MInvLocal,
        Factors = 2.0 ./ Mesh.Δ,
    )

    F = SFFM.MakeFluxMatrixR(Mesh = Mesh, Model = Model, Phi = Phi)

    ## Assemble the DG drift operator
    Q = zeros(Float64, Mesh.TotalNBases, Mesh.TotalNBases, length(Model.C))
    for i = 1:Model.NPhases
        Q[:, :, i] = Model.C[i] * (G[:, :, i] + F[:, :, i]) * MInv[:, :, i]
    end

    Local = (G = GLocal, M = MLocal, MInv = MInvLocal, V = V, Phi = Phi)
    Global = (G = G, M = M, MInv = MInv, F = F, Q = Q)
    println("Matrices.Fields with Fields (.Local, .Global)")
    println("Matrices.Local.Fields with Fields (.G, .M, .MInv, .V)")
    println("Matrices.Global.Fields with Fields (.G, .M, .MInv, F, .Q)")
    return (Local = Local, Global = Global)
end

function MakeDR(;
    Matrices,
    MatricesR,
    Model::NamedTuple{(:T, :C, :r, :IsBounded, :Bounds, :NPhases)},
    Mesh::NamedTuple{
        (:NBases, :CellNodes, :Fil, :Δ, :NIntervals, :MeshArray, :Nodes, :TotalNBases),
    },
    B,
)
    N₊ = sum(Model.C .>= 0)
    N₋ = sum(Model.C .<= 0)

    BigN = Model.NPhases * Mesh.TotalNBases + N₊ + N₋
    MR = zeros(Float64, Model.NPhases * Mesh.TotalNBases, Model.NPhases * Mesh.TotalNBases)
    Minv =
        zeros(Float64, Model.NPhases * Mesh.TotalNBases, Model.NPhases * Mesh.TotalNBases)
    FGR = zeros(Float64, Model.NPhases * Mesh.TotalNBases, Model.NPhases * Mesh.TotalNBases)
    for i = 1:Model.NPhases
        idx = ((i-1)*Mesh.TotalNBases+1:i*Mesh.TotalNBases)
        MR[idx, idx] = MatricesR.Global.M[:, :, i]
        Minv[idx, idx] = Matrices.Global.M^-1
        FGR[idx, idx] =
            (MatricesR.Global.F[:, :, i] + MatricesR.Global.G[:, :, i]) *
            Model.C[i] *
            Minv[idx, idx]
    end

    # Interior behaviour
    T = kron(Model.T, LinearAlgebra.I(Mesh.TotalNBases))
    BR = zeros(Float64, BigN, BigN)
    BR[(N₋+1):(end-N₊), (N₋+1):(end-N₊)] = MR * T * Minv + FGR

    # Boundary behaviour
    # Lower boundary
    # At boundary
    BR[1:N₋, 1:N₋] = (1.0./abs.(Model.r.r(Mesh.CellNodes[1]))'.*Model.T)[Model.C.<=0, Model.C.<=0]
    # Out of boundary
    idxup = ((1:Mesh.NBases).+Mesh.TotalNBases*(findall(Model.C .> 0) .- 1)')[:] .+ N₋
    BR[1:N₋, idxup] = kron(
        (1.0./abs.(Model.r.r(Mesh.CellNodes[1]))'.*Model.T)[Model.C.<=0, Model.C.>0],
        Matrices.Local.Phi[1, :]' * Matrices.Local.MInv,
    )
    # Into boundary
    idxdown = ((1:Mesh.NBases).+Mesh.TotalNBases*(findall(Model.C .<= 0) .- 1)')[:] .+ N₋
    BR[idxdown, 1:N₋] = LinearAlgebra.kron(
        LinearAlgebra.diagm(
            0 => Model.C[Model.C.<=0] ./ abs.(Model.r.r(Mesh.CellNodes[1]))[Model.C.<=0],
        ),
        -Matrices.Local.Phi[1, :] * 2 / Mesh.Δ[1],
    )

    # Upper boundary
    # At boundary
    BR[(end-N₊+1):end, (end-N₊+1):end] =
        (1.0./abs.(Model.r.r(Mesh.CellNodes[end]))'.*Model.T)[Model.C.>=0, Model.C.>=0]
    # Out of boundary
    idxdown =
        ((1:Mesh.NBases).+Mesh.TotalNBases*(findall(Model.C .< 0) .- 1)')[:] .+
        (N₋ + Mesh.TotalNBases - Mesh.NBases)
    BR[(end-N₊+1):end, idxdown] = kron(
        (1.0./abs.(Model.r.r(Mesh.CellNodes[end]))'.*Model.T)[Model.C.>=0, Model.C.<0],
        Matrices.Local.Phi[end, :]' * Matrices.Local.MInv,
    )
    # Into boundary
    idxup =
        ((1:Mesh.NBases).+Mesh.TotalNBases*(findall(Model.C .>= 0) .- 1)')[:] .+
        (N₋ + Mesh.TotalNBases - Mesh.NBases)
    BR[idxup, (end-N₊+1):end] = LinearAlgebra.kron(
        LinearAlgebra.diagm(
            0 => Model.C[Model.C.>=0] ./ abs.(Model.r.r(Mesh.CellNodes[end]))[Model.C.<=0],
        ),
        Matrices.Local.Phi[end, :] * 2 / Mesh.Δ[end],
    )

    idx0 = [Mesh.Fil["p0"]; repeat(Mesh.Fil["0"]', Mesh.NBases, 1)[:]; Mesh.Fil["q0"]]
    bullet = [
        (Mesh.Fil["p+"] .| Mesh.Fil["p-"])
        repeat((Mesh.Fil["+"] .| Mesh.Fil["-"])', Mesh.NBases, 1)[:]
        (Mesh.Fil["q+"] .| Mesh.Fil["q-"])
    ]
    MR = [
        LinearAlgebra.I(N₋) zeros(Float64, N₋, BigN - N₋)
        zeros(Float64, BigN - N₊ - N₋, N₋) MR zeros(Float64, BigN - N₊ - N₋, N₊)
        zeros(Float64, N₊, BigN - N₊) LinearAlgebra.I(N₊)
    ]
    Minv = [
        LinearAlgebra.I(N₋) zeros(Float64, N₋, BigN - N₋)
        zeros(Float64, BigN - N₊ - N₋, N₋) Minv zeros(Float64, BigN - N₊ - N₋, N₊)
        zeros(Float64, N₊, BigN - N₊) LinearAlgebra.I(N₊)
    ]

    BR[idx0, :] = B.B[idx0, :]

    DR = function (;s=0)
        BR[bullet, bullet] -
        MR[bullet, bullet] * s * LinearAlgebra.I(sum(bullet)) * Minv[bullet, bullet] +
        BR[bullet, idx0] *
        (BR[idx0, idx0] - s * LinearAlgebra.I(sum(idx0)))^-1 *
        BR[idx0, bullet]
    end
    # D = function (s)
    #     MR[bullet, bullet] *
    #     (T[bullet, bullet] - s * LinearAlgebra.I(sum(bullet))) *
    #     Minv[bullet, bullet] +
    #     FGR[bullet, bullet] +
    #     MR[bullet, bullet] *
    #     T[bullet, idx0] *
    #     (B.BDict["00"] - s * LinearAlgebra.I(sum(idx0)))^-1 *
    #     T[idx0, bullet]
    # end

    DDict = Dict{String,Any}()
    for ℓ in ["+", "-"], m in ["+", "-"]
        FlBases = Mesh.Fil[ℓ][.!Mesh.Fil["0"]]
        FmBases = Mesh.Fil[m][.!Mesh.Fil["0"]]
        FlBases = [
            Mesh.Fil["p"*ℓ][.!Mesh.Fil["p0"]]
            repeat(FlBases', Mesh.NBases, 1)[:]
            Mesh.Fil["q"*ℓ][.!Mesh.Fil["q0"]]
        ]
        FmBases = [
            Mesh.Fil["p"*m][.!Mesh.Fil["p0"]]
            repeat(FmBases', Mesh.NBases, 1)[:]
            Mesh.Fil["q"*m][.!Mesh.Fil["q0"]]
        ]
        DDict[ℓ*m] = function (; s = 0)#::Array{Float64}
            DR(s=s)[FlBases, FmBases]
        end # end function
    end # end for ℓ, m ...
    return (DDict = DDict, DR = DR)
end
