function MakeMesh(;
    Model::NamedTuple{(:T, :C, :r, :Bounds, :NPhases, :SDict, :TDict)},
    Nodes::Array{Float64,1},
    NBases::Int,
    Fil::Dict{String,BitArray{1}}=Dict{String,BitArray{1}}(),
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
    #        .Δ, .NIntervals, .Nodes, .TotalNBases
    # .NBases - Int the number of bases in each cell
    # .CellNodes - NBases×NIntervals Array{Float64}
    # .Fil - same as input
    # .Δ - NIntervals×1 Array{Float64}, the width of the cells
    # .NIntervals - Int, the number of intervals
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
    TotalNBases = NBases * NIntervals # the total number of bases in the stencil

    ## Construct the sets Fᵐ = ⋃ᵢ Fᵢᵐ, global index for sets of type m
    if isempty(Fil)
        idxPlus = Model.r.r(Nodes[1:end-1].+Δ[:]/2).>0
        idxZero = Model.r.r(Nodes[1:end-1].+Δ[:]/2).==0
        idxMinus = Model.r.r(Nodes[1:end-1].+Δ[:]/2).<0
        for i in 1:Model.NPhases
            Fil[string(i)*"+"] = idxPlus[:,i]
            Fil[string(i)*"0"] = idxZero[:,i]
            Fil[string(i)*"-"] = idxMinus[:,i]
            if Model.C[i] .<= 0
                Fil["p"*string(i)*"+"] = [Model.r.r(Model.Bounds[1,1])[i]].>0
                Fil["p"*string(i)*"0"] = [Model.r.r(Model.Bounds[1,1])[i]].==0
                Fil["p"*string(i)*"-"] = [Model.r.r(Model.Bounds[1,1])[i]].<0
            end
            if Model.C[i] .>= 0
                Fil["q"*string(i)*"+"] = [Model.r.r(Model.Bounds[1,end])[i]].>0
                Fil["q"*string(i)*"0"] = [Model.r.r(Model.Bounds[1,end])[i]].==0
                Fil["q"*string(i)*"-"] = [Model.r.r(Model.Bounds[1,end])[i]].<0
            end
        end
    end
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
              .NIntervals, .Nodes, .TotalNBases)")
    return (
        NBases = NBases,
        CellNodes = CellNodes,
        Fil = Fil,
        Δ = Δ,
        NIntervals = NIntervals,
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
        V .= [1/sqrt(2)]
        DV .= [0]
        w = [2]
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

    BlockMatrix = SparseArrays.spzeros(Float64, Mesh.TotalNBases, Mesh.TotalNBases)
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
            :Nodes,
            :TotalNBases,
            :Basis,
        ),
    },
    Model::NamedTuple{(:T, :C, :r, :Bounds, :NPhases, :SDict, :TDict)},
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
    F = Array{SparseArrays.SparseMatrixCSC{Float64,Int64},1}(undef,Model.NPhases)
    for i = 1:Model.NPhases
        F[i] = SparseArrays.spzeros(Float64, Mesh.TotalNBases, Mesh.TotalNBases)
        for k = 1:Mesh.NIntervals
            idx = (1:Mesh.NBases) .+ (k - 1) * Mesh.NBases
            if Model.C[i] > 0
                F[i][idx, idx] = PosDiagBlock
            elseif Model.C[i] < 0
                F[i][idx, idx] = NegDiagBlock
            end # end if C[i]
            if k > 1
                idxup = (1:Mesh.NBases) .+ (k - 2) * Mesh.NBases
                if Model.C[i] > 0
                    if Mesh.Basis == "legendre"
                        η = 1
                    elseif Mesh.Basis == "lagrange"
                        η = Mesh.Δ[k] / Mesh.Δ[k-1]
                    end
                    F[i][idxup, idx] = UpDiagBlock * η
                elseif Model.C[i] < 0
                    if Mesh.Basis == "legendre"
                        η = 1
                    elseif Mesh.Basis == "lagrange"
                        η = Mesh.Δ[k-1] / Mesh.Δ[k]
                    end
                    F[i][idx, idxup] = LowDiagBlock * η
                end # end if C[i]
            end # end if k>1
        end # for k in ...
    end # end for i in NPhases

    return (F = F)
end

function MakeMatrices(;
    Model::NamedTuple{(:T, :C, :r, :Bounds, :NPhases, :SDict, :TDict)},
    Mesh::NamedTuple{
        (
            :NBases,
            :CellNodes,
            :Fil,
            :Δ,
            :NIntervals,
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
    Q = Array{SparseArrays.SparseMatrixCSC{Float64,Int64},1}(undef,Model.NPhases)
    for i = 1:Model.NPhases
        Q[i] = Model.C[i] * (G + F[i]) * MInv
    end

    Local = (G = GLocal, M = MLocal, MInv = MInvLocal, V = V, Phi = Phi, Dw = Dw)
    Global = (G = G, M = M, MInv = MInv, F = F, Q = Q)
    println("Matrices.Fields with Fields (.Local, .Global)")
    println("Matrices.Local.Fields with Fields (.G, .M, .MInv, .V, .Phi , .Dw)")
    println("Matrices.Global.Fields with Fields (.G, .M, .MInv, F, .Q)")
    return (Local = Local, Global = Global)
end

function MakeB(;
    Model::NamedTuple{(:T, :C, :r, :Bounds, :NPhases, :SDict, :TDict)},
    Mesh::NamedTuple{
        (
            :NBases,
            :CellNodes,
            :Fil,
            :Δ,
            :NIntervals,
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
    B = SparseArrays.spzeros(
        Float64,
        Model.NPhases * Mesh.TotalNBases + N₋ + N₊,
        Model.NPhases * Mesh.TotalNBases + N₋ + N₊,
    )
    Id = SparseArrays.sparse(LinearAlgebra.I, Mesh.TotalNBases, Mesh.TotalNBases)
    for i = 1:Model.NPhases
        idx = ((i-1)*Mesh.TotalNBases+1:i*Mesh.TotalNBases) .+ N₋
        B[idx, idx] = Matrices.Global.Q[i]
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
        Matrices.Local.Phi[1, :]' * Matrices.Local.MInv * Matrices.Local.Dw.Dw ./ η[1],
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
        Matrices.Local.Phi[end, :]' * Matrices.Local.MInv * Matrices.Local.Dw.Dw  ./ η[1],
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
    BDict = Dict{String,SparseArrays.SparseMatrixCSC{Float64,Int64}}()
    ppositions = cumsum(Model.C .<= 0)
    qpositions = cumsum(Model.C .>= 0)
    for ℓ in ["+", "-", "0"], m in ["+", "-", "0"]
        for i = 1:Model.NPhases, j = 1:Model.NPhases
            FilBases = repeat(Mesh.Fil[string(i, ℓ)]', Mesh.NBases, 1)[:]
            pitemp = falses(N₋)
            qitemp = falses(N₊)
            pjtemp = falses(N₋)
            qjtemp = falses(N₊)
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
    c = N₋
    QBDidx = zeros(Int, Model.NPhases * Mesh.TotalNBases + N₊ + N₋)
    for k = 1:Mesh.NIntervals, i = 1:Model.NPhases, n = 1:Mesh.NBases
        c += 1
        QBDidx[c] = (i - 1) * Mesh.TotalNBases + (k - 1) * Mesh.NBases + n + N₋
    end
    QBDidx[1:N₋] = 1:N₋
    QBDidx[(end-N₊+1):end] = (Model.NPhases * Mesh.TotalNBases + N₋) .+ (1:N₊)

    println("B.Fields with Fields (.BDict, .B, .QBDidx)")
    return (BDict = BDict, B = B, QBDidx = QBDidx)
end

function MakeR(;
    Model::NamedTuple{(:T, :C, :r, :Bounds, :NPhases, :SDict, :TDict)},
    Mesh::NamedTuple{
        (
            :NBases,
            :CellNodes,
            :Fil,
            :Δ,
            :NIntervals,
            :Nodes,
            :TotalNBases,
            :Basis,
        ),
    },
    approxType::String = "interpolation",
)
    V = SFFM.vandermonde(NBases=Mesh.NBases)
    # interpolant approximation to r(x)
    EvalPoints = Mesh.CellNodes
    EvalPoints[1, :] .+= sqrt(eps()) # LH edges + eps
    EvalPoints[end, :] .+= -sqrt(eps()) # RH edges - eps
    EvalR = 1.0 ./ Model.r.a(EvalPoints[:])

    N₋ = sum(Model.C .<= 0)
    N₊ = sum(Model.C .>= 0)

    R = SparseArrays.spzeros(
        Float64,
        N₋ + N₊ + Mesh.TotalNBases * Model.NPhases,
        N₋ + N₊ + Mesh.TotalNBases * Model.NPhases,
    )
    R[1:N₋, 1:N₋] = (1.0 ./ Model.r.a(Model.Bounds[1,1])[Model.C .<= 0]).*LinearAlgebra.I(N₋)
    R[(end-N₊+1):end, (end-N₊+1):end] =  (1.0 ./ Model.r.a(Model.Bounds[1,end])[Model.C .>= 0]).* LinearAlgebra.I(N₊)

    for n = 1:(Mesh.NIntervals*Model.NPhases)
        if Mesh.Basis == "legendre"
            if approxType == "interpolation"
                leftM = V.V'
                rightM = V.inv'
            elseif approxType == "projection"
                leftM = V.V' * LinearAlgebra.diagm(V.w)
                rightM = V.V
            end
            temp = leftM*LinearAlgebra.diagm(EvalR[Mesh.NBases*(n-1).+(1:Mesh.NBases)])*rightM
        elseif Mesh.Basis == "lagrange"
            if approxType == "interpolation"
                temp = LinearAlgebra.diagm(EvalR[Mesh.NBases*(n-1).+(1:Mesh.NBases)])
            elseif approxType == "projection"
                temp = LinearAlgebra.diagm(EvalR[Mesh.NBases*(n-1).+(1:Mesh.NBases)])*V.V*V.V'*LinearAlgebra.diagm(V.w)
            end
        end
        R[Mesh.NBases*(n-1).+(1:Mesh.NBases).+N₋, Mesh.NBases*(n-1).+(1:Mesh.NBases).+N₋] =
            temp
    end

    # RFun = function (ind::String=":")
    #     if ind == ":"
    #         return R[:,:]
    #     else
    #         FlBases = Mesh.Fil[ind]
    #         FlBases = [
    #             Mesh.Fil["p"*ind]
    #             repeat(FlBases', Mesh.NBases, 1)[:]
    #             Mesh.Fil["q"*ind]
    #         ]
    #         return R[FlBases,FlBases]
    #     end
    # end
    RDict = Dict{String,SparseArrays.SparseMatrixCSC{Float64,Int64}}()
    ppositions = cumsum(Model.C .<= 0)
    qpositions = cumsum(Model.C .>= 0)
    for ℓ in ["+", "-"]
        for i = 1:Model.NPhases
            FilBases = repeat(Mesh.Fil[string(i, ℓ)]', Mesh.NBases, 1)[:]
            pitemp = falses(N₋)
            qitemp = falses(N₊)
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

    # println("R, function with arguments (string) +, -, 0, : (default = :)")
    return (R=R, RDict=RDict)
end


function MakeD(;
    R,
    B,
    Model::NamedTuple{(:T, :C, :r, :Bounds, :NPhases, :SDict, :TDict)},
    Mesh::NamedTuple{
        (
            :NBases,
            :CellNodes,
            :Fil,
            :Δ,
            :NIntervals,
            :Nodes,
            :TotalNBases,
            :Basis,
        ),
    },
)
    DDict = Dict{String,Any}()
    for ℓ in ["+", "-"], m in ["+", "-"]
        nℓ = sum(Mesh.Fil["p"*ℓ]) + sum(Mesh.Fil[ℓ]) * Mesh.NBases + sum(Mesh.Fil["q"*ℓ])
        Idℓ = SparseArrays.sparse(LinearAlgebra.I,nℓ,nℓ)
        if any(Mesh.Fil["p0"]) || any(Mesh.Fil["0"]) || any(Mesh.Fil["q0"]) # in("0", Model.Signs)
            n0 = sum(Mesh.Fil["p0"]) +
                sum(Mesh.Fil["0"]) * Mesh.NBases +
                sum(Mesh.Fil["q0"])
            Id0 = SparseArrays.sparse(LinearAlgebra.I,n0,n0)
            DDict[ℓ*m] = function (; s = 0)#::Array{Float64}
                return if (ℓ == m)
                    R.RDict[ℓ] * (
                        B.BDict[ℓ*m] - s * Idℓ +
                        B.BDict[ℓ*"0"] * inv(Matrix(s * Id0 - B.BDict["00"])) * B.BDict["0"*m]
                    )
                else
                    R.RDict[ℓ] * (
                        B.BDict[ℓ*m] +
                        B.BDict[ℓ*"0"] * inv(Matrix(s * Id0 - B.BDict["00"])) * B.BDict["0"*m]
                    )
                end
            end # end function
        else
            DDict[ℓ*m] = function (; s = 0)#::Array{Float64}
                return if (ℓ == m)
                    R.RDict[ℓ] * (B.BDict[ℓ*m] - s * Idℓ)
                else
                    R.RDict[ℓ] * B.BDict[ℓ*m]
                end
            end # end function
        end # end if ...
    end # end for ℓ ...
    return (DDict = DDict)
end

function PsiFun(; s = 0, D, MaxIters = 1000, err = 1e-8)
    #
    exitflag = ""

    EvalD = Dict{String,SparseArrays.SparseMatrixCSC{Float64,Int64}}("+-" => D["+-"](s = s))
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
        Psi = LinearAlgebra.sylvester(Matrix(A), Matrix(B), Matrix(C))
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

function EulerDG(;
    D::Union{Array{<:Real,2},SparseArrays.SparseMatrixCSC{Float64,Int64}},
    y::Real,
    x0::Array{<:Real},
    h::Float64 = 0.0001,
)
    x = x0
    for t = h:h:y
        dx = h * x * D
        x = x + dx
    end
    return x
end

function Coeffs2Dist(;
    Model::NamedTuple{(:T, :C, :r, :Bounds, :NPhases, :SDict, :TDict)},
    Mesh::NamedTuple{
        (
            :NBases,
            :CellNodes,
            :Fil,
            :Δ,
            :NIntervals,
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
            yvals = reshape(Coeffs[N₋+1:end-N₊], Mesh.NBases, Mesh.NIntervals, Model.NPhases)
            for i in 1:Model.NPhases
                yvals[:,:,i] = V.V * yvals[:,:,i]
            end
            pm = [Coeffs[1:N₊]; Coeffs[end-N₊+1:end]]
        elseif Mesh.Basis == "lagrange"
            yvals =
                Coeffs[N₋+1:end-N₊] .* repeat(1.0 ./ V.w, Mesh.NIntervals * Model.NPhases) .*
                (repeat(2.0 ./ Mesh.Δ, 1, Mesh.NBases * Model.NPhases)'[:])
            yvals = reshape(yvals, Mesh.NBases, Mesh.NIntervals, Model.NPhases)
            pm = [Coeffs[1:N₋]; Coeffs[end-N₊+1:end]]
        end
        if Mesh.NBases == 1
            yvals = [1;1].*yvals
            xvals = [Mesh.CellNodes-Mesh.Δ'/2;Mesh.CellNodes+Mesh.Δ'/2]
        end
    elseif type == "probability"
        if Mesh.NBases > 1
            xvals = Mesh.CellNodes[1, :] + (Mesh.Δ ./ 2)
        else
            xvals = Mesh.CellNodes
        end
        if Mesh.Basis == "legendre"
            yvals = (reshape(Coeffs[N₋+1:Mesh.NBases:end-N₊], 1, Mesh.NIntervals, Model.NPhases).*Mesh.Δ')./sqrt(2)
            pm = [Coeffs[1:N₊]; Coeffs[end-N₊+1:end]]
        elseif Mesh.Basis == "lagrange"
            yvals = sum(
                reshape(Coeffs[N₋+1:end-N₊], Mesh.NBases, Mesh.NIntervals, Model.NPhases),
                dims = 1,
            )
            pm = [Coeffs[1:N₋]; Coeffs[end-N₊+1:end]]
        end
    elseif type == "cumulative"
        if Mesh.NBases > 1
            xvals = Mesh.CellNodes[[1;end], :]
        else
            xvals = [Mesh.CellNodes-Mesh.Δ'/2;Mesh.CellNodes+Mesh.Δ'/2]
        end
        if Mesh.Basis == "legendre"
            tempDist = (reshape(Coeffs[N₋+1:Mesh.NBases:end-N₊], 1, Mesh.NIntervals, Model.NPhases).*Mesh.Δ')./sqrt(2)
            pm = [Coeffs[1:N₊]; Coeffs[end-N₊+1:end]]
        elseif Mesh.Basis == "lagrange"
            tempDist = sum(
                reshape(Coeffs[N₋+1:end-N₊], Mesh.NBases, Mesh.NIntervals, Model.NPhases),
                dims = 1,
            )
            pm = [Coeffs[1:N₋]; Coeffs[end-N₊+1:end]]
        end
        tempDist = cumsum(tempDist,dims=2)
        temppm = zeros(Float64,1,2,Model.NPhases)
        temppm[:,1,Model.C.<=0] = pm[1:N₋]
        temppm[:,2,Model.C.>=0] = pm[N₊+1:end]
        yvals = zeros(Float64,2,Mesh.NIntervals,Model.NPhases)
        yvals[1,2:end,:] = tempDist[1,1:end-1,:]
        yvals[2,:,:] = tempDist
        yvals = yvals .+ reshape(temppm[1,1,:],1,1,Model.NPhases)
        pm[N₋+1:end] = pm[N₋+1:end] + yvals[end,end,Model.C.>=0]
    end
    return (pm=pm, distribution=yvals, x=xvals, type=type)
end

function Dist2Coeffs(;
    Model::NamedTuple{(:T, :C, :r, :Bounds, :NPhases, :SDict, :TDict)},
    Mesh::NamedTuple{
        (
            :NBases,
            :CellNodes,
            :Fil,
            :Δ,
            :NIntervals,
            :Nodes,
            :TotalNBases,
            :Basis,
        ),
    },
    Distn::NamedTuple{(:pm, :distribution, :x, :type)},
)
    # Distn.type == "probability" is an 1 × NIntervals × NPhases array where
    # each element represents a (cell,phase) pair. i.e. each element is
    # P(X(0) ∈ D_k, φ(0) = i )
    V = SFFM.vandermonde(NBases = Mesh.NBases)
    theDistribution =
        zeros(Float64, Mesh.NBases, Mesh.NIntervals, Model.NPhases)
    if Mesh.Basis == "legendre"
        if Distn.type == "probability"
            theDistribution[1, :, :] = Distn.distribution./Mesh.Δ'.*sqrt(2)
        elseif Distn.type == "density"
            theDistribution = Distn.distribution
            for i = 1:Model.NPhases
                theDistribution[:, :, i] = V.inv * theDistribution[:, :, i]
            end
        end
        coeffs = [
            Distn.pm[1:sum(Model.C .<= 0)]
            theDistribution[:]
            Distn.pm[sum(Model.C .<= 0)+1:end]
        ]
    elseif Mesh.Basis == "lagrange"
        theDistribution .= Distn.distribution
        if Distn.type == "probability"
            theDistribution = (V.w .* theDistribution / 2)[:]
        elseif Distn.type == "density"
            theDistribution = ((V.w .* theDistribution).*(Mesh.Δ / 2)')[:]
        end
        coeffs = [
            Distn.pm[1:sum(Model.C .<= 0)]
            theDistribution
            Distn.pm[sum(Model.C .<= 0)+1:end]
        ]
    end
    coeffs = Matrix(coeffs[:]')
    return coeffs
end
