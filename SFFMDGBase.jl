function MakeMesh(;
    Model::NamedTuple{(:T, :C, :r, :IsBounded, :Bounds, :NPhases)},
    Nodes::Array{Float64,1},
    NBases::Int,
    Fil::Dict{String,BitArray{1}},
    Basis::String = "legendre",
)
    # MakeMesh constructs the Mass and Stiffness matrices
    # Model - a MakeModel object
    # Nodes - (K+1)×1 Array{Float64}, specifying the edges of the cells
    # NBases - Int, specifying the number of bases within each cell
    #          (same for all cells)
    # Fil - Dict{String,BitArray{1}}, A dictionary of the sets Fᵢᵐ, they keys
    #        are Strings specifying i and m, i.e. "2+", the values are BitArrays
    #        of boolean values which specify which cells of the stencil
    #        correspond to Fᵢᵐ
    #
    # output is a MakeMesh tupe with fields: .NBases, CellNodes, .Fil,
    #        .Δ, .NIntervals, .MeshArray, .Nodes, .TotalNBases
    # .NBases - Int the number of bases in each cell
    # .CellNodes - NBases×NIntervals Array{Float64}
    # .Fil - same as input
    # .Δ - NIntervals×1 Array{Float64}, the width of the cells
    # .NIntervals - Int, the number of intervals
    # .MeshArray - 2×NIntervals Array{Float64}, end points of each cell, 1st row
    #               LHS edges, 2nd row RHS edges
    # .Nodes - as input
    # .TotalNBases - Int, the total number of bases in the mesh

    ## Stencil specification
    NIntervals = length(Nodes) - 1 # the number of intervals
    Δ = (Nodes[2:end] - Nodes[1:end-1]) # interval width
    CellNodes = zeros(Float64, NBases, NIntervals)
    if NBases > 1
        z = Jacobi.zglj(NBases, 0, 0) # the LGL nodes
    elseif NBases == 1
        z = 0.0
    end
    for i = 1:NIntervals
        # Map the LGL nodes on [-1,1] to each cell
        CellNodes[:, i] .= (Nodes[i+1] + Nodes[i]) / 2 .+ (Nodes[i+1] - Nodes[i]) / 2 * z
    end
    MeshArray = zeros(NIntervals, 2)
    MeshArray[:, 1] = Nodes[1:end-1] # Left-hand end points of each interval
    MeshArray[:, 2] = Nodes[2:end] # Right-hand edges
    TotalNBases = NBases * NIntervals # the total number of bases in the stencil

    ## Construct the sets Fᵐ = ⋃ᵢ Fᵢᵐ, global index for sets of type m
    CurrKeys = keys(Fil)
    for ℓ in ["+", "-", "0"], i = 1:Model.NPhases
        if !in(string(i) * ℓ, CurrKeys)
            Fil[string(i)*ℓ] = falses(NIntervals)
        end
        if !in("p" * string(i) * ℓ, CurrKeys) && Model.C[i] <= 0
            Fil["p"*string(i)*ℓ] = falses(1)
        end
        if !in("p" * string(i) * ℓ, CurrKeys) && Model.C[i] > 0
            Fil["p"*string(i)*ℓ] = falses(0)
        end
        if !in("q" * string(i) * ℓ, CurrKeys) && Model.C[i] >= 0
            Fil["q"*string(i)*ℓ] = falses(1)
        end
        if !in("q" * string(i) * ℓ, CurrKeys) && Model.C[i] < 0
            Fil["q"*string(i)*ℓ] = falses(0)
        end
    end
    for ℓ in ["+", "-", "0"]
        Fil[ℓ] = falses(NIntervals * Model.NPhases)
        Fil["p"*ℓ] = trues(0)
        Fil["q"*ℓ] = trues(0)
        for i = 1:Model.NPhases
            idx = findall(Fil[string(i)*ℓ]) .+ (i - 1) * NIntervals
            Fil[string(ℓ)][idx] .= true
            Fil["p"*ℓ] = [Fil["p"*ℓ]; Fil["p"*string(i)*ℓ]]
            Fil["q"*ℓ] = [Fil["q"*ℓ]; Fil["q"*string(i)*ℓ]]
        end
    end

    println("Mesh.Field with Fields (.NBases, .CellNodes, .Fil, .Δ,
              .NIntervals, .MeshArray, .Nodes, .TotalNBases)")
    return (
        NBases = NBases,
        CellNodes = CellNodes,
        Fil = Fil,
        Δ = Δ,
        NIntervals = NIntervals,
        MeshArray = MeshArray,
        Nodes = Nodes,
        TotalNBases = TotalNBases,
        Basis = Basis,
    )
end

function vandermonde(; NBases::Int)
    # requires Jacobi package Pkg.add("Jacobi")
    # construct a generalised vandermonde matrix
    # NBases is the degree of the basis
    # outputs: V.V, is a vandermonde matrix, of lagendre
    #               polynomials evaluated at G-L points
    #          V.inv, its inverse
    #          V.D, the derivative of the bases

    if NBases > 1
        z = Jacobi.zglj(NBases, 0, 0) # the LGL nodes
    elseif NBases == 1
        z = 0.0
    end
    V = zeros(Float64, NBases, NBases)
    DV = zeros(Float64, NBases, NBases)
    if NBases > 1
        for j = 1:NBases
            # compute the polynomials at gauss-labotto quadrature points
            V[:, j] = Jacobi.legendre.(z, j - 1) .* sqrt((2 * (j - 1) + 1) / 2)
            DV[:, j] = Jacobi.dlegendre.(z, j - 1) .* sqrt((2 * (j - 1) + 1) / 2)
        end
        # Compute the Gauss-Lobatto weights for numerical quadrature
        w =
            2.0 ./ (
                NBases *
                (NBases - 1) *
                Jacobi.legendre.(Jacobi.zglj(NBases, 0, 0), NBases - 1) .^ 2
            )
    elseif NBases == 1
        V .= 1
        DV .= 0
        w = 1
    end
    return (V = V, inv = inv(V), D = DV, w = w)
end

function MakeBlockDiagonalMatrix(;
    Mesh::NamedTuple{
        (
            :NBases,
            :CellNodes,
            :Fil,
            :Δ,
            :NIntervals,
            :MeshArray,
            :Nodes,
            :TotalNBases,
            :Basis,
        ),
    },
    Blocks::Array{Float64,2},
    Factors::Array,
)
    # MakeBlockDiagonalMatrix makes a matrix from diagonal block elements
    # inputs:
    # Mesh - A tuple from MakeMesh
    # Blocks - Mesh.NBases×Mesh.NBases Array{Float64}, blocks to put along the
    #           diagonal
    # Factors - Mesh.NIntervals×1 Array, factors which multiply blocks
    # output:
    # BlockMatrix - Mesh.TotalNBases×Mesh.TotalNBases Array{Float64,2}, the
    #             block matrix

    BlockMatrix = zeros(Float64, Mesh.TotalNBases, Mesh.TotalNBases)
    for i = 1:Mesh.NIntervals
        idx = (1:Mesh.NBases) .+ (i - 1) * Mesh.NBases
        BlockMatrix[idx, idx] = Blocks * Factors[i]
    end
    return (BlockMatrix = BlockMatrix)
end

function MakeFluxMatrix(;
    Mesh::NamedTuple{
        (
            :NBases,
            :CellNodes,
            :Fil,
            :Δ,
            :NIntervals,
            :MeshArray,
            :Nodes,
            :TotalNBases,
            :Basis,
        ),
    },
    Model::NamedTuple{(:T, :C, :r, :IsBounded, :Bounds, :NPhases)},
    Phi,
    Dw,
)
    # MakeFluxMatrix creates the global block tridiagonal flux matrix for the
    # lagrange basis
    # inputs:
    # Mesh - a Mesh tuple from MakeMesh
    # Model - a Model tuple from MakeModel
    # outputs:
    # F - TotalNBases×TotalNBases×NPhases Array{Float64,3}, global flux matrix

    ## Create the blocks
    PosDiagBlock = -Dw.DwInv * Phi[end, :] * Phi[end, :]' * Dw.Dw
    NegDiagBlock = Dw.DwInv * Phi[1, :] * Phi[1, :]' * Dw.Dw
    UpDiagBlock = Dw.DwInv * Phi[end, :] * Phi[1, :]' * Dw.Dw
    LowDiagBlock = -Dw.DwInv * Phi[1, :] * Phi[end, :]' * Dw.Dw

    ## Construct global block diagonal matrix
    F = zeros(Float64, Mesh.TotalNBases, Mesh.TotalNBases, Model.NPhases)
    for i = 1:Model.NPhases
        for k = 1:Mesh.NIntervals
            idx = (1:Mesh.NBases) .+ (k - 1) * Mesh.NBases
            if Model.C[i] > 0
                F[idx, idx, i] = PosDiagBlock
            elseif Model.C[i] < 0
                F[idx, idx, i] = NegDiagBlock
            end # end if C[i]
            if k > 1
                idxup = (1:Mesh.NBases) .+ (k - 2) * Mesh.NBases
                if Model.C[i] > 0
                    if Mesh.Basis == "legendre"
                        η = 1
                    elseif Mesh.Basis == "lagrange"
                        η = Mesh.Δ[k] / Mesh.Δ[k-1]
                    end
                    F[idxup, idx, i] = UpDiagBlock * η
                elseif Model.C[i] < 0
                    if Mesh.Basis == "legendre"
                        η = 1
                    elseif Mesh.Basis == "lagrange"
                        η = Mesh.Δ[k-1] / Mesh.Δ[k]
                    end
                    F[idx, idxup, i] = LowDiagBlock * η
                end # end if C[i]
            end # end if k>1
        end # for k in ...
    end # end for i in NPhases

    return (F = F)
end

function MakeMatrices(;
    Model::NamedTuple{(:T, :C, :r, :IsBounded, :Bounds, :NPhases)},
    Mesh::NamedTuple{
        (
            :NBases,
            :CellNodes,
            :Fil,
            :Δ,
            :NIntervals,
            :MeshArray,
            :Nodes,
            :TotalNBases,
            :Basis,
        ),
    },
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
    if Mesh.Basis == "legendre"
        Dw = (
            DwInv = LinearAlgebra.diagm(0 => ones(Float64, Mesh.NBases)),
            Dw = LinearAlgebra.diagm(0 => ones(Float64, Mesh.NBases)),
        )
        MLocal = Matrix{Float64}(LinearAlgebra.I(Mesh.NBases))
        GLocal = V.inv * V.D
        MInvLocal = Matrix{Float64}(LinearAlgebra.I(Mesh.NBases))
        Phi = V.V[[1; end], :]
    elseif Mesh.Basis == "lagrange"
        Dw = (
            DwInv = LinearAlgebra.diagm(0 => 1.0 ./ V.w),
            Dw = LinearAlgebra.diagm(0 => V.w),
        )
        MLocal = Dw.DwInv * V.inv' * V.inv * Dw.Dw
        GLocal = Dw.DwInv * V.inv' * V.inv * (V.D * V.inv) * Dw.Dw
        MInvLocal = Dw.DwInv * V.V * V.V' * Dw.Dw
        Phi = (V.inv*V.V)[[1; end], :]
    end

    ## Assemble into block diagonal matrices
    G = SFFM.MakeBlockDiagonalMatrix(
        Mesh = Mesh,
        Blocks = GLocal,
        Factors = ones(Mesh.NIntervals),
    )
    M = SFFM.MakeBlockDiagonalMatrix(Mesh = Mesh, Blocks = MLocal, Factors = Mesh.Δ * 0.5)
    MInv = SFFM.MakeBlockDiagonalMatrix(
        Mesh = Mesh,
        Blocks = MInvLocal,
        Factors = 2.0 ./ Mesh.Δ,
    )
    F = SFFM.MakeFluxMatrix(Mesh = Mesh, Model = Model, Phi = Phi, Dw = Dw)

    ## Assemble the DG drift operator
    Q = zeros(Float64, Mesh.TotalNBases, Mesh.TotalNBases, length(Model.C))
    for i = 1:Model.NPhases
        Q[:, :, i] = Model.C[i] * (G + F[:, :, i]) * MInv
    end

    Local = (G = GLocal, M = MLocal, MInv = MInvLocal, V = V, Phi = Phi, Dw = Dw)
    Global = (G = G, M = M, MInv = MInv, F = F, Q = Q)
    println("Matrices.Fields with Fields (.Local, .Global)")
    println("Matrices.Local.Fields with Fields (.G, .M, .MInv, .V, .Phi , .Dw)")
    println("Matrices.Global.Fields with Fields (.G, .M, .MInv, F, .Q)")
    return (Local = Local, Global = Global)
end

function MakeB(;
    Model::NamedTuple{(:T, :C, :r, :IsBounded, :Bounds, :NPhases)},
    Mesh::NamedTuple{
        (
            :NBases,
            :CellNodes,
            :Fil,
            :Δ,
            :NIntervals,
            :MeshArray,
            :Nodes,
            :TotalNBases,
            :Basis,
        ),
    },
    Matrices,
)
    # MakeB, makes the DG approximation to B, the transition operator +
    # shift operator.
    # inputs:
    # Model - A model tuple from MakeModel
    # Mesh - A Mesh tuple from MakeMesh
    # Matrices - A Matrices tuple from MakeMatrices
    # output:
    # A tuple with fields .BDict, .B, .QBDidx
    # .BDict - Dict{String,Array{Float64,2}}, a dictionary storing Bᵢⱼˡᵐ with
    #          keys string(i,j,ℓ,m), and values Bᵢⱼˡᵐ,
    #          i.e. .BDict["12+-"] = B₁₂⁺⁻
    # .B - Model.NPhases*Mesh.TotalNBases×Model.NPhases*Mesh.TotalNBases
    #       Array{Float64,2}, the global approximation to B
    # .QBDidx - Model.NPhases*Mesh.TotalNBases×1 Int, vector of integers such
    #           such that .B[.QBDidx,.QBDidx] puts all the blocks relating to
    #           cell k next to each other

    ## MakeB
    N₊ = sum(Model.C .>= 0)
    N₋ = sum(Model.C .<= 0)
    B = zeros(
        Float64,
        Model.NPhases * Mesh.TotalNBases + N₋ + N₊,
        Model.NPhases * Mesh.TotalNBases + N₋ + N₊,
    )
    Id = Matrix(LinearAlgebra.I, Mesh.TotalNBases, Mesh.TotalNBases)
    for i = 1:Model.NPhases
        idx = ((i-1)*Mesh.TotalNBases+1:i*Mesh.TotalNBases) .+ N₋
        B[idx, idx] = Matrices.Global.Q[:, :, i]
    end

    # interior behaviour
    B[(N₋+1):(end-N₊), (N₋+1):(end-N₊)] =
        B[(N₋+1):(end-N₊), (N₋+1):(end-N₊)] + LinearAlgebra.kron(Model.T, Id)
    # Boundary behaviour
    if Mesh.Basis == "legendre"
        η = Mesh.Δ[[1; end]] ./ 2
    elseif Mesh.Basis == "lagrange"
        η = [1; 1]
    end
    # Lower boundary
    # At boundary
    B[1:N₋, 1:N₋] = Model.T[Model.C.<=0, Model.C.<=0]
    # Out of boundary
    idxup = ((1:Mesh.NBases).+Mesh.TotalNBases*(findall(Model.C .> 0) .- 1)')[:] .+ N₋
    B[1:N₋, idxup] = kron(
        Model.T[Model.C.<=0, Model.C.>0],
        Matrices.Local.Phi[1, :]' * Matrices.Local.Dw.Dw * Matrices.Local.MInv ./ η[1],
    )
    # Into boundary
    idxdown = ((1:Mesh.NBases).+Mesh.TotalNBases*(findall(Model.C .<= 0) .- 1)')[:] .+ N₋
    B[idxdown, 1:N₋] = LinearAlgebra.kron(
        LinearAlgebra.diagm(0 => Model.C[Model.C.<=0]),
        -Matrices.Local.Dw.DwInv * 2.0 ./ Mesh.Δ[1] * Matrices.Local.Phi[1, :] * η[1],
    )

    # Upper boundary
    # At boundary
    B[(end-N₊+1):end, (end-N₊+1):end] = Model.T[Model.C.>=0, Model.C.>=0]
    # Out of boundary
    idxdown =
        ((1:Mesh.NBases).+Mesh.TotalNBases*(findall(Model.C .< 0) .- 1)')[:] .+
        (N₋ + Mesh.TotalNBases - Mesh.NBases)
    B[(end-N₊+1):end, idxdown] = kron(
        Model.T[Model.C.>=0, Model.C.<0],
        Matrices.Local.Phi[end, :]' * Matrices.Local.Dw.Dw * Matrices.Local.MInv ./ η[1],
    )
    # Into boundary
    idxup =
        ((1:Mesh.NBases).+Mesh.TotalNBases*(findall(Model.C .>= 0) .- 1)')[:] .+
        (N₋ + Mesh.TotalNBases - Mesh.NBases)
    B[idxup, (end-N₊+1):end] = LinearAlgebra.kron(
        LinearAlgebra.diagm(0 => Model.C[Model.C.>=0]),
        Matrices.Local.Dw.DwInv * 2.0 ./ Mesh.Δ[end] * Matrices.Local.Phi[end, :] * η[end],
    )

    ## Make a Dictionary so that the blocks of B are easy to access
    BDict = Dict{String,Array{Float64,2}}()
    pfalses = falses(N₋)
    qfalses = falses(N₊)
    ppositions = cumsum(Model.C .<= 0)
    qpositions = cumsum(Model.C .>= 0)
    for ℓ in ["+", "-", "0"], m in ["+", "-", "0"]
        for i = 1:Model.NPhases, j = 1:Model.NPhases
            FilBases = repeat(Mesh.Fil[string(i, ℓ)]', Mesh.NBases, 1)[:]
            pitemp = pfalses
            qitemp = qfalses
            pjtemp = pfalses
            qjtemp = qfalses
            if Model.C[i] <= 0
                pitemp[ppositions[i]] = Mesh.Fil["p"*string(i)*ℓ][1]
            end
            if Model.C[j] <= 0
                pjtemp[ppositions[j]] = Mesh.Fil["p"*string(j)*m][1]
            end
            if Model.C[i] >= 0
                qitemp[qpositions[i]] = Mesh.Fil["q"*string(i)*ℓ][1]
            end
            if Model.C[j] >= 0
                qjtemp[qpositions[j]] = Mesh.Fil["q"*string(j)*m][1]
            end
            i_idx = [
                pitemp
                falses((i - 1) * Mesh.TotalNBases)
                FilBases
                falses(Model.NPhases * Mesh.TotalNBases - i * Mesh.TotalNBases)
                qitemp
            ]
            FjmBases = repeat(Mesh.Fil[string(j, m)]', Mesh.NBases, 1)[:]
            j_idx = [
                pjtemp
                falses((j - 1) * Mesh.TotalNBases)
                FjmBases
                falses(Model.NPhases * Mesh.TotalNBases - j * Mesh.TotalNBases)
                qjtemp
            ]
            BDict[string(i, j, ℓ, m)] = B[i_idx, j_idx]
        end
        FlBases =
            [Mesh.Fil["p"*ℓ]; repeat(Mesh.Fil[ℓ]', Mesh.NBases, 1)[:]; Mesh.Fil["q"*ℓ]]
        FmBases =
            [Mesh.Fil["p"*m]; repeat(Mesh.Fil[m]', Mesh.NBases, 1)[:]; Mesh.Fil["q"*m]]
        BDict[ℓ*m] = B[FlBases, FmBases]
    end

    ## Make QBD index
    c = 0
    QBDidx = zeros(Int, Model.NPhases * Mesh.TotalNBases)
    for k = 1:Mesh.NIntervals, i = 1:Model.NPhases, n = 1:Mesh.NBases
        c += 1
        QBDidx[c] = (i - 1) * Mesh.TotalNBases + (k - 1) * Mesh.NBases + n
    end

    println("B.Fields with Fields (.BDict, .B, .QBDidx)")
    return (BDict = BDict, B = B, QBDidx = QBDidx)
end

function MakeR(;
    Model::NamedTuple{(:T, :C, :r, :IsBounded, :Bounds, :NPhases)},
    Mesh::NamedTuple{
        (
            :NBases,
            :CellNodes,
            :Fil,
            :Δ,
            :NIntervals,
            :MeshArray,
            :Nodes,
            :TotalNBases,
            :Basis,
        ),
    },
    V,
)
    # interpolant approximation to r(x)
    EvalPoints = Mesh.CellNodes
    EvalPoints[1, :] .+= sqrt(eps()) # LH edges + eps
    EvalPoints[end, :] .+= -sqrt(eps()) # RH edges - eps
    EvalR = 1.0 ./ abs.(Model.r.r(EvalPoints[:]))

    N₋ = sum(Model.C .<= 0)
    N₊ = sum(Model.C .>= 0)

    R = zeros(
        N₋ + N₊ + Mesh.TotalNBases * Model.NPhases,
        N₋ + N₊ + Mesh.TotalNBases * Model.NPhases,
    )
    R[1:N₋, 1:N₋] = LinearAlgebra.I(N₋)
    R[(end-N₊+1):end, (end-N₊+1):end] = LinearAlgebra.I(N₊)

    for n = 1:(Mesh.NIntervals*Model.NPhases)
        if Mesh.Basis == "legendre"
            temp =
                V.V' *
                LinearAlgebra.diagm(EvalR[Mesh.NBases*(n-1).+(1:Mesh.NBases)]) *
                V.inv'
        elseif Mesh.Basis == "lagrange"
            temp = LinearAlgebra.diagm(EvalR[Mesh.NBases*(n-1).+(1:Mesh.NBases)])
        end
        R[Mesh.NBases*(n-1).+(1:Mesh.NBases).+N₋, Mesh.NBases*(n-1).+(1:Mesh.NBases).+N₋] =
            temp
    end

    RDict = Dict{String,Array{Float64,2}}()
    pfalses = falses(N₋)
    qfalses = falses(N₊)
    ppositions = cumsum(Model.C .<= 0)
    qpositions = cumsum(Model.C .>= 0)
    for ℓ in ["+", "-"]
        for i = 1:Model.NPhases
            FilBases = repeat(Mesh.Fil[string(i, ℓ)]', Mesh.NBases, 1)[:]
            pitemp = pfalses
            qitemp = qfalses
            if Model.C[i] <= 0
                pitemp[ppositions[i]] = Mesh.Fil["p"*string(i)*ℓ][1]
            end
            if Model.C[i] >= 0
                qitemp[qpositions[i]] = Mesh.Fil["q"*string(i)*ℓ][1]
            end
            i_idx = [
                pitemp
                falses((i - 1) * Mesh.TotalNBases)
                FilBases
                falses(Model.NPhases * Mesh.TotalNBases - i * Mesh.TotalNBases)
                qitemp
            ]
            RDict[string(i, ℓ)] = R[i_idx, i_idx]
        end
        FlBases =
            [Mesh.Fil["p"*ℓ]; repeat(Mesh.Fil[ℓ]', Mesh.NBases, 1)[:]; Mesh.Fil["q"*ℓ]]
        RDict[ℓ] = R[FlBases, FlBases]
    end

    # N₋ = sum(Model.C .<= 0)
    # N₊ = sum(Model.C .>= 0)
    #
    # R = zeros(Float64, Model.NPhases * Mesh.TotalNBases + N₋ + N₊, Model.NPhases * Mesh.TotalNBases + N₋ + N₊)
    # R[1:N₋,1:N₋] = LinearAlgebra.diagm(1.0 ./ abs.(Model.r.r(Mesh.CellNodes[1])[Model.C.<=0]))
    # R[(end-N₊+1):end,(end-N₊+1):end] = LinearAlgebra.diagm(1.0 ./ abs.(Model.r.r(Mesh.CellNodes[end])[Model.C.>=0]))
    # for i = 1:Model.NPhases
    #     if Model.C[i] < 0
    #         p = 1.0 ./ abs.(Model.r.r(Mesh.CellNodes[1])[i])
    #         q = Float64[]
    #     elseif Model.C[i] > 0
    #         p = Float64[]
    #         q = 1.0 ./ abs.(Model.r.r(Mesh.CellNodes[end])[i])
    #     elseif Model.C[i] == 0
    #         p = 1.0 ./ abs.(Model.r.r(Mesh.CellNodes[1])[i])
    #         q = 1.0 ./ abs.(Model.r.r(Mesh.CellNodes[end])[i])
    #     end
    #     temp = 1.0 ./ EvalR[:, i]
    #     RDict[string(i)] = LinearAlgebra.diagm([p; temp; q])
    #     R[((i-1)*Mesh.TotalNBases+1:i*Mesh.TotalNBases).+N₋,((i-1)*Mesh.TotalNBases+1:i*Mesh.TotalNBases).+N₋] = LinearAlgebra.diagm(temp)
    #     for ℓ in ["+", "-"]
    #         FilBases = repeat(Mesh.Fil[string(i)*ℓ]', Mesh.NBases, 1)[:]
    #         RDict[string(i)*ℓ] = LinearAlgebra.diagm([
    #             p .* Mesh.Fil["p"*string(i)*ℓ]
    #             temp[FilBases]
    #             q .* Mesh.Fil["q"*string(i)*ℓ]
    #         ])
    #     end
    # end
    #
    # for ℓ in ["+", "-"]
    #     FlBases =
    #         [Mesh.Fil["p"*ℓ]; repeat(Mesh.Fil[ℓ]', Mesh.NBases, 1)[:]; Mesh.Fil["q"*ℓ]]
    #     RDict[ℓ] = R[FlBases,FlBases]
    # end
    println("R.Fields with Fields (.RDict, .R)")
    return (RDict = RDict, R = R)
end

function MakeMyD(;
    Model::NamedTuple{(:T, :C, :r, :IsBounded, :Bounds, :NPhases)},
    Mesh::NamedTuple{
        (
            :NBases,
            :CellNodes,
            :Fil,
            :Δ,
            :NIntervals,
            :MeshArray,
            :Nodes,
            :TotalNBases,
            :Basis,
        ),
    },
    B,
    V,
)

    if Mesh.Basis == "legendre"
        MRLocal = function (x::Array{Float64}, i::Int)
            # Numerical integration of ϕᵢ(x)|r(x)|ϕⱼ(x) over Dk with Gauss-Lobatto
            # quadrature
            # Inputs:
            #   - x a vector of Gauss-Lobatto points on Dk
            #   - i a phase
            V.V' * LinearAlgebra.diagm(V.w ./ abs.(Model.r.r(x)[:, i])) * V.V
        end
    elseif Mesh.Basis == "lagrange"
        MRLocal = function (x::Array{Float64}, i::Int)
            # Numerical integration of ϕᵢ(x)|r(x)|ϕⱼ(x) over Dk with Gauss-Lobatto
            # quadrature
            # Inputs:
            #   - x a vector of Gauss-Lobatto points on Dk
            #   - i a phase
            LinearAlgebra.diagm(1.0 ./ abs.(Model.r.r(x)[:, i]))
        end
    end
    MyR = zeros(
        Float64,
        Mesh.TotalNBases * Model.NPhases + sum(Model.C .<= 0) + sum(Model.C .>= 0),
        Mesh.TotalNBases * Model.NPhases + sum(Model.C .<= 0) + sum(Model.C .>= 0),
    )
    for i = 1:Model.NPhases, k = 1:Mesh.NIntervals
        idx =
            sum(Model.C .<= 0) .+ (1:Mesh.NBases) .+ (Mesh.NBases .* (k - 1)) .+
            (Mesh.TotalNBases .* (i - 1))
        MyR[idx, idx] = MRLocal(Mesh.CellNodes[:, k], i)
    end
    MyR[1:sum(Model.C .<= 0), 1:sum(Model.C .<= 0)] =
        LinearAlgebra.diagm(1.0 ./ abs.(Model.r.r(Mesh.Nodes[1])[Model.C.<=0]))
    MyR[end-sum(Model.C .>= 0).+1:end, end-sum(Model.C .>= 0).+1:end] =
        LinearAlgebra.diagm(1.0 ./ abs.(Model.r.r(Mesh.Nodes[end])[Model.C.>=0]))
    idx0 = [Mesh.Fil["p0"]; repeat(Mesh.Fil["0"]', Mesh.NBases, 1)[:]; Mesh.Fil["q0"]]
    MyD = function (; s = 0)
        MyR[.!idx0, .!idx0] * (
            B.B[.!idx0, .!idx0] - LinearAlgebra.I(sum(.!idx0)) * s +
            B.B[.!idx0, idx0] *
            (LinearAlgebra.I(sum(idx0)) * s - B.B[idx0, idx0])^-1 *
            B.B[idx0, .!idx0]
        )
    end

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
            MyD(s = s)[FlBases, FmBases]
        end # end function
    end # end for ℓ, m ...

    return (D = MyD, DDict = DDict)
end

function MakeD(;
    R,
    B,
    Model::NamedTuple{(:T, :C, :r, :IsBounded, :Bounds, :NPhases)},
    Mesh::NamedTuple{
        (
            :NBases,
            :CellNodes,
            :Fil,
            :Δ,
            :NIntervals,
            :MeshArray,
            :Nodes,
            :TotalNBases,
            :Basis,
        ),
    },
)
    RDict = R.RDict
    BDict = B.BDict
    DDict = Dict{String,Any}()
    for ℓ in ["+", "-"], m in ["+", "-"]
        Idℓ = LinearAlgebra.I(
            sum(Mesh.Fil["p"*ℓ]) + sum(Mesh.Fil[ℓ]) * Mesh.NBases + sum(Mesh.Fil["q"*ℓ]),
        )
        if any(Mesh.Fil["p0"]) || any(Mesh.Fil["0"]) || any(Mesh.Fil["q0"]) # in("0", Model.Signs)
            Id0 = LinearAlgebra.I(
                sum(Mesh.Fil["p0"]) +
                sum(Mesh.Fil["0"]) * Mesh.NBases +
                sum(Mesh.Fil["q0"]),
            )
            DDict[ℓ*m] = function (; s = 0)#::Array{Float64}
                return if (ℓ == m)
                    RDict[ℓ] * (
                        BDict[ℓ*m] - s * Idℓ +
                        BDict[ℓ*"0"] * inv(s * Id0 - BDict["00"]) * BDict["0"*m]
                    )
                else
                    RDict[ℓ] * (
                        BDict[ℓ*m] +
                        BDict[ℓ*"0"] * inv(s * Id0 - BDict["00"]) * BDict["0"*m]
                    )
                end
            end # end function
        else
            DDict[ℓ*m] = function (; s = 0)#::Array{Float64}
                return if (ℓ == m)
                    RDict[ℓ] * (BDict[ℓ*m] - s * Idℓ)
                else
                    RDict[ℓ] * BDict[ℓ*m]
                end
            end # end function
        end # end if ...
    end # end for ℓ ...
    return (DDict = DDict)
end

function PsiFun(; s = 0, D, MaxIters = 1000, err = 1e-8)
    #
    exitflag = ""

    EvalD = Dict{String,Array{Float64}}("+-" => D["+-"](s = s))
    Dimensions = size(EvalD["+-"])
    for ℓ in ["++", "--", "-+"]
        EvalD[ℓ] = D[ℓ](s = s)
    end
    Psi = zeros(Float64, Dimensions)
    A = EvalD["++"]
    B = EvalD["--"]
    C = EvalD["+-"]
    OldPsi = Psi
    flag = 1
    for n = 1:MaxIters
        Psi = LinearAlgebra.sylvester(A, B, C)
        if maximum(abs.(OldPsi - Psi)) < err
            flag = 0
            exitflag = string(
                "Reached err tolerance in ",
                n,
                " iterations with error ",
                string(maximum(abs.(OldPsi - Psi))),
            )
            break
        elseif any(isnan.(Psi))
            flag = 0
            exitflag = string("Produced NaNs at iteration ", n)
            break
        end
        OldPsi = Psi
        A = EvalD["++"] + Psi * EvalD["-+"]
        B = EvalD["--"] + EvalD["-+"] * Psi
        C = EvalD["+-"] - Psi * EvalD["-+"] * Psi
    end
    if flag == 1
        exitflag = string(
            "Reached Max Iters ",
            MaxIters,
            " with error ",
            string(maximum(abs.(OldPsi - Psi))),
        )
    end
    display(exitflag)
    return Psi
end

function EulerDG(; D::Array{<:Real,2}, y::Real, x0::Array{<:Real}, h::Float64 = 0.0001)
    x = x0
    for t = h:h:y
        dx = h * x * D
        x = x + dx
    end
    return x
end

function Coeffs2Distn(;
    Model::NamedTuple{(:T, :C, :r, :IsBounded, :Bounds, :NPhases)},
    Mesh::NamedTuple{
        (
            :NBases,
            :CellNodes,
            :Fil,
            :Δ,
            :NIntervals,
            :MeshArray,
            :Nodes,
            :TotalNBases,
            :Basis,
        ),
    },
    Coeffs,
    type::String = "probability"
)
    V = SFFM.vandermonde(NBases = Mesh.NBases)
    N₋ = sum(Model.C .<= 0)
    N₊ = sum(Model.C .>= 0)
    if type == "density"
        xvals = Mesh.CellNodes
        if Mesh.Basis == "legendre"
            yvals =
                V.V * reshape(Coeffs[N₋+1:end-N₊], Mesh.NBases, Mesh.NIntervals, Model.NPhases)
            pm = [Coeffs[1:N₊]; Coeffs[end-N₊+1:end]]
        elseif Mesh.Basis == "lagrange"
            yvals =
                Coeffs[N₋+1:end-N₊] .* repeat(1.0 ./ V.w, Mesh.NIntervals * Model.NPhases) .*
                (repeat(2.0 ./ Mesh.Δ, 1, Mesh.NBases * Model.NPhases)'[:])
            yvals = reshape(yvals, Mesh.NBases, Mesh.NIntervals, Model.NPhases)
            pm = [Coeffs[1:N₋]; Coeffs[end-N₊+1:end]]
        end
    elseif type == "probability"
        xvals = Mesh.CellNodes[1, :] + (Mesh.Δ ./ 2)
        if Mesh.Basis == "legendre"
            yvals = reshape(Coeffs[N₋+1:Mesh.NBases:end-N₊], 1, Mesh.NIntervals, Model.NPhases)
            pm = [Coeffs[1:N₊]; Coeffs[end-N₊+1:end]]
        elseif Mesh.Basis == "lagrange"
            yvals = sum(
                reshape(Coeffs[N₋+1:end-N₊], Mesh.NBases, Mesh.NIntervals, Model.NPhases),
                dims = 1,
            )
            pm = [Coeffs[1:N₋]; Coeffs[end-N₊+1:end]]
        end
    end
    return (pm, yvals, xvals)
end

function Distn2Coeffs(; Model, Distn::NamedTuple{(:pm, :yvals, :xvals)})
    coeffs = [
        Distn.pm[1:sum(Model.C .<= 0)]
        V.inv*Distn.yvals[:]
        Distn.pm[sum(Model.C .<= 0)+1:end]
    ]
    return coeffs
end
