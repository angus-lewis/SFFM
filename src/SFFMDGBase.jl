"""
Constructs a Mesh object (a tuple with fields which describe the DG mesh).

    MakeMesh(;
        Model::NamedTuple{(:T, :C, :r, :Bounds, :NPhases, :SDict, :TDict)},
        Nodes::Array{Float64,1},
        NBases::Int,
        Fil::Dict{String,BitArray{1}}=Dict{String,BitArray{1}}(),
        Basis::String = "legendre",
    )

# Arguments
- `Model`: a MakeModel object
- `Nodes::Array{Float64,1}`: (K+1)×1 array, specifying the edges of the cells
- `NBases::Int`: specifying the number of bases within each cell (same for all
    cells)
- `Fil::Dict{String,BitArray{1}}`: (optional) A dictionary of the sets Fᵢᵐ, they
    keys are Strings specifying i and m, i.e. `"2+"`, the values are BitArrays of
    boolean values which specify which cells of the stencil correspond to Fᵢᵐ. If no
    value specified then `Fil` is generated automatically evaluating ``r_i(x)`` at
    the modpoint of each cell.
- `Basis::String`: a string specifying whether to use the `"lagrange"` basis or
    the `"legendre"` basis

# Output
- a MakeMesh tuple with keys:
    - `:NBases`: the number of bases in each cell
    - `:CellNodes`: Array of nodal points (cell edges + GLL points)
    - `:Fil`: As described in the arguments
    - `:Δ`:A vector of mesh widths, Δ[k] = x_{k+1} - x_k
    - `:NIntervals`: The number of cells
    - `:Nodes`: the cell edges
    - `:TotalNBases`: `NIntervals*NBases`
    - `:Basis`: a string specifying whether the

# Examples
TBC
"""
function MakeMesh(;
    Model::NamedTuple{(:T, :C, :r, :Bounds, :NPhases, :SDict, :TDict)},
    Nodes::Array{Float64,1},
    NBases::Int,
    Fil::Dict{String,BitArray{1}}=Dict{String,BitArray{1}}(),
    Basis::String = "legendre",
)
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

    out = (
        NBases = NBases,
        CellNodes = CellNodes,
        Fil = Fil,
        Δ = Δ,
        NIntervals = NIntervals,
        Nodes = Nodes,
        TotalNBases = TotalNBases,
        Basis = Basis,
    )
    println("UPDATE: Mesh object created with fields ",keys(out))
    return out
end

"""
Construct a generalised vandermonde matrix.

    `vandermonde(; NBases::Int)`

Note: requires Jacobi package Pkg.add("Jacobi")

# Arguments
- `NBases::Int`: the degree of the basis

# Output
- a tuple with keys
    - `:V::Array{Float64,2}`: where `:V[:,i]` contains the values of the `i`th
        legendre polynomial evaluate at the GLL nodes.
    - `:inv`: the inverse of :V
    - `:D::Array{Float64,2}`: where `V.D[:,i]` contains the values of the derivative
        of the `i`th legendre polynomial evaluate at the GLL nodes.
"""
function vandermonde(; NBases::Int)
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

"""
Constructs a block diagonal matrix from blocks

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

# Aguments
- `Mesh`: A tuple from MakeMesh
- `Blocks::Array{Float64,2}`: a `Mesh.NBases×Mesh.NBases` block to put along the
        diagonal
- `Factors::Array{<:Real,1}`: a `Mesh.NIntervals×1` vector of factors which multiply blocks

# Output
- `BlockMatrix::Array{Float64,2}`: `Mesh.TotalNBases×Mesh.TotalNBases` the
        block matrix
"""
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
    BlockMatrix = SparseArrays.spzeros(Float64, Mesh.TotalNBases, Mesh.TotalNBases)
    for i = 1:Mesh.NIntervals
        idx = (1:Mesh.NBases) .+ (i - 1) * Mesh.NBases
        BlockMatrix[idx, idx] = Blocks * Factors[i]
    end
    return (BlockMatrix = BlockMatrix)
end

"""
Constructs the flux matrices for DG

    MakeFluxMatrix(;
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
        probTransform::Bool=true,
    )

# Arguments
- `Mesh`: a Mesh tuple from MakeMesh
- `Phi::Array{Float64,2}`: where `Phi[1,:]` and `Phi[1,:]` are the basis
    function evaluated at the left-hand and right-hand edge of a cell,
    respectively
- `Dw::Array{Float64,2}`: a diagonal matrix function weights
- `probTransform::Bool=true`: an (optional) specification for the lagrange basis
    to specify whether transform to probability coefficients.

# Output
- `F::Dict{String, SparseArrays.SparseMatrixCSC{Float64,Int64},1}`: a dictionary
    with keys `"+"` and `"-"` and values which are `TotalNBases×TotalNBases`
    flux matrices for `Model.C[i]>0` and `Model.C[i]<0`, respectively.
"""
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
    Phi,
    Dw,
    probTransform::Bool=true,
)
    ## Create the blocks
    PosDiagBlock = -Dw.DwInv * Phi[end, :] * Phi[end, :]' * Dw.Dw
    NegDiagBlock = Dw.DwInv * Phi[1, :] * Phi[1, :]' * Dw.Dw
    UpDiagBlock = Dw.DwInv * Phi[end, :] * Phi[1, :]' * Dw.Dw
    LowDiagBlock = -Dw.DwInv * Phi[1, :] * Phi[end, :]' * Dw.Dw

    ## Construct global block diagonal matrix
    F = Dict{String,SparseArrays.SparseMatrixCSC{Float64,Int64}}()
    for i = ["+","-"]
        F[i] = SparseArrays.spzeros(Float64, Mesh.TotalNBases, Mesh.TotalNBases)
        for k = 1:Mesh.NIntervals
            idx = (1:Mesh.NBases) .+ (k - 1) * Mesh.NBases
            if i=="+"
                F[i][idx, idx] = PosDiagBlock
            elseif i=="-"
                F[i][idx, idx] = NegDiagBlock
            end # end if C[i]
            if k > 1
                idxup = (1:Mesh.NBases) .+ (k - 2) * Mesh.NBases
                if i=="+"
                    # the legendre basis works in density world so there are no etas
                    if Mesh.Basis == "legendre"
                        η = 1
                    elseif Mesh.Basis == "lagrange"
                        if !probTransform
                            η = 1
                        else
                            η = Mesh.Δ[k] / Mesh.Δ[k-1]
                        end
                    end
                    F[i][idxup, idx] = UpDiagBlock * η
                elseif i=="-"
                    if Mesh.Basis == "legendre"
                        η = 1
                    elseif Mesh.Basis == "lagrange"
                        if !probTransform
                            η = 1
                        else
                            η = Mesh.Δ[k-1] / Mesh.Δ[k]
                        end
                    end
                    F[i][idx, idxup] = LowDiagBlock * η
                end # end if C[i]
            end # end if k>1
        end # for k in ...
    end # for i in ...

    return (F = F)
end

"""
Creates the Local and global mass, stiffness and flux matrices.

    MakeMatrices(;
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
        probTransform::Bool=true,
    )

# Arguments
- `Model`: A model tuple from MakeModel
- `Mesh`: A mesh tuple from MakeMesh
- `probTransform::Bool=true`: an (optional) specification for the lagrange basis
    to specify whether transform to probability coefficients.

# Output
- A tuple of tuples
    - `:Global`: a tuple with fields
      - `:G::SparseArrays.SparseMatrixCSC{Float64,Int64}`:
        `TotalNBases×TotalNBases`, global stiffness matrix
      - `:M::SparseArrays.SparseMatrixCSC{Float64,Int64}`:
        `TotalNBases×TotalNBases`, global mass matrix
      - `:MInv::SparseArrays.SparseMatrixCSC{Float64,Int64}`: the inverse of
        `Global.M`
      - `:F::Dict{String,SparseArrays.SparseMatrixCSC{Float64,Int64}}`:
            `TotalNBases×TotalNBases` global flux matrix
      - `:Q::Array{SparseArrays.SparseMatrixCSC{Float64,Int64},1}`:
            a `NPhases` length array containing `TotalNBases×TotalNBases`
            dimensional global DG drift operator
    - `:Local`: a tuple with fields
      - `:G::Array{Float64,2}`: `NBases×NBases` Local stiffness matrix
      - `:M::Array{Float64,2}`: `NBases×NBases` Local mass matrix
      - `:MInv::Array{Float64,2}`: the inverse of `Local.M`
      - `:V::NamedTuple`: as output from SFFM.vandermonde
 """
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
    probTransform::Bool=true,
)
    ## Construct local blocks
    V = vandermonde(NBases = Mesh.NBases)
    if Mesh.Basis == "legendre"
        Dw = (
            DwInv = LinearAlgebra.diagm(0 => ones(Float64, Mesh.NBases)),
            Dw = LinearAlgebra.diagm(0 => ones(Float64, Mesh.NBases)),
        ) # function weights are not available for legendre basis as this is
        # in density land
        MLocal = Matrix{Float64}(LinearAlgebra.I(Mesh.NBases))
        GLocal = V.inv * V.D
        MInvLocal = Matrix{Float64}(LinearAlgebra.I(Mesh.NBases))
        Phi = V.V[[1; end], :]
    elseif Mesh.Basis == "lagrange"
        if !probTransform
            Dw = (
                DwInv = LinearAlgebra.I,
                Dw = LinearAlgebra.I,
            )# function weights so that we can work in probability land as
            # opposed to density land
        else
            Dw = (
                DwInv = LinearAlgebra.diagm(0 => 1.0 ./ V.w),
                Dw = LinearAlgebra.diagm(0 => V.w),
            )# function weights so that we can work in probability land as
            # opposed to density land
        end
        MLocal = Dw.DwInv * V.inv' * V.inv * Dw.Dw
        GLocal = Dw.DwInv * V.inv' * V.inv * (V.D * V.inv) * Dw.Dw
        MInvLocal = Dw.DwInv * V.V * V.V' * Dw.Dw
        Phi = (V.inv*V.V)[[1; end], :]
    end

    ## Assemble into global block diagonal matrices
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
    F = SFFM.MakeFluxMatrix(Mesh = Mesh, Phi = Phi, Dw = Dw, probTransform = probTransform)

    ## Assemble the DG drift operator
    Q = Array{SparseArrays.SparseMatrixCSC{Float64,Int64},1}(undef,Model.NPhases)
    for i = 1:Model.NPhases
        if Model.C[i] > 0
            Q[i] = Model.C[i] * (G + F["+"]) * MInv
        elseif Model.C[i] < 0
            Q[i] = Model.C[i] * (G + F["-"]) * MInv
        end
    end

    Local = (G = GLocal, M = MLocal, MInv = MInvLocal, V = V, Phi = Phi, Dw = Dw)
    Global = (G = G, M = M, MInv = MInv, F = F, Q = Q)
    out = (Local = Local, Global = Global)
    println("UPDATE: Matrices object created with keys ", keys(out))
    println("UPDATE:    Matrices[:",keys(out)[1],"] object created with keys ", keys(out[1]))
    println("UPDATE:    Matrices[:",keys(out)[2],"] object created with keys ", keys(out[2]))
    return out
end

"""
Creates the DG approximation to the generator B.

    MakeB(;
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
        Matrices::NamedTuple,
        probTransform::Bool=true,
    )

# Arguments
- `Model`: A model tuple from `MakeModel`
- `Mesh`: A Mesh tuple from `MakeMesh`
- `Matrices`: A Matrices tuple from `MakeMatrices`
- `probTransform::Bool=true`: an (optional) specification for the lagrange basis
    to specify whether transform to probability coefficients.

# Output
- A tuple with fields .BDict, .B, .QBDidx
    - `:BDict::Dict{String,Array{Float64,2}}`: a dictionary storing Bᵢⱼˡᵐ with
        keys string(i,j,ℓ,m), and values Bᵢⱼˡᵐ, i.e. .BDict["12+-"] = B₁₂⁺⁻
    - `:B::SparseArrays.SparseMatrixCSC{Float64,Int64}`:
        `Model.NPhases*Mesh.TotalNBases×Model.NPhases*Mesh.TotalNBases`, the
        global approximation to B
    - `:QBDidx::Array{Int64,1}`: `Model.NPhases*Mesh.TotalNBases×1` vector of
        integers such such that `:B[QBDidx,QBDidx]` puts all the blocks relating
        to cell k next to each other
"""
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
    Matrices::NamedTuple,
    probTransform::Bool=true,
)
    ## Make B on the interior of the space
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
    B[(N₋+1):(end-N₊), (N₋+1):(end-N₊)] =
        B[(N₋+1):(end-N₊), (N₋+1):(end-N₊)] + LinearAlgebra.kron(Model.T, Id)

    # Boundary behaviour
    if Mesh.Basis == "legendre"
        η = Mesh.Δ[[1; end]] ./ 2 # this is the inverse of the η=Mesh.Δ/2 bit
        # below there are no η's for the legendre basis
    elseif Mesh.Basis == "lagrange"
        if !probTransform
            η = Mesh.Δ[[1; end]] ./ 2
        else
            η = [1; 1]
        end
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
        Matrices.Local.Phi[end, :]' * Matrices.Local.Dw.Dw * Matrices.Local.MInv  ./ η[end],
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
        # note: below we need to use repeat(Mesh.Fil[ℓ]', Mesh.NBases, 1)[:] to
        # expand the index Mesh.Fil[ℓ] from cells to all basis function
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

    out = (BDict = BDict, B = B, QBDidx = QBDidx)
    println("UPDATE: B object created with keys ", keys(out))
    return out
end

"""
# Construct the DG approximation to the operator R.

    MakeR(;
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
        approxType::String = "projection",
    )

# Arguments
- `Model`: a model object from MakeModel
- `Mesh`: a mesh object from MakeMesh
- `approxType::String`: (optional) either "interpolation" or
    "projection" (default).

# Output
- a tuple with keys
    - `:R::SparseArrays.SparseMatrixCSC{Float64,Int64}`: an approximation to R
        for the whole space. If ``rᵢ(x)=0`` on any cell, the corresponding
        elements of R are zero.
    - `:RDict::Dict{String,SparseArrays.SparseMatrixCSC{Float64,Int64}}`: a
        disctionary containing sub-blocks of R. Keys are of the form
        `"PhaseSign"` or just `"Sign"`. i.e. `"1-"` cells in ``Fᵢ⁻``, and
        `"-"` for cells in ``∪ᵢFᵢ⁻``.
"""
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
    approxType::String = "projection",
)
    V = SFFM.vandermonde(NBases=Mesh.NBases)

    EvalPoints = copy(Mesh.CellNodes)
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
    # at the boundaries
    R[1:N₋, 1:N₋] = (1.0 ./ Model.r.a(Model.Bounds[1,1])[Model.C .<= 0]).*LinearAlgebra.I(N₋)
    R[(end-N₊+1):end, (end-N₊+1):end] =  (1.0 ./ Model.r.a(Model.Bounds[1,end])[Model.C .>= 0]).* LinearAlgebra.I(N₊)

    # on the interior
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
                # the first term, LinearAlgebra.diagm(EvalR[Mesh.NBases*(n-1).+(1:Mesh.NBases)])
                # is the quadrature approximation of M^r. The quadrature weights to not
                # appear since they cancel when we transform to integral/probability
                # representation. The second term V.V*V.V' is Minv. The last term
                # LinearAlgebra.diagm(V.w)is a result of the conversion to probability
                # / integral representation.
                temp = LinearAlgebra.diagm(EvalR[Mesh.NBases*(n-1).+(1:Mesh.NBases)])*V.V*V.V'*LinearAlgebra.diagm(V.w)
            end
        end
        R[Mesh.NBases*(n-1).+(1:Mesh.NBases).+N₋, Mesh.NBases*(n-1).+(1:Mesh.NBases).+N₋] = temp
    end

    # construc the dictionary
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

    out = (R=R, RDict=RDict)
    println("UPDATE: R object created with keys ", keys(out))
    return out
end

"""
Construct the operator D(s).

    MakeD(;
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

# Arguments
- `R`: a tuple as constructed by MakeR
- `B`: a tuple as constructed by MakeB
- `Model`: a model object as constructed by MakeModel
- `Mesh`: a mesh object as constructed by MakeMesh

# Output
- `DDict::Dict{String,Function(s::Real)}`: a dictionary of functions. Keys are
  of the for `"ℓm"` where `ℓ,m∈{+,-}`. Values are functions with one argument.
  Usage is along the lines of `D["+-"](s=1)`.
"""
function MakeD(;
    R::NamedTuple{(:R, :RDict)},
    B::NamedTuple{(:BDict, :B, :QBDidx)},
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
            DDict[ℓ*m] = function (; s::Real = 0)
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
            DDict[ℓ*m] = function (; s::Real = 0)
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

"""
Construct and evaluate ``Ψ(s)``.

Uses newtons method to solve the Ricatti equation
``D⁺⁻(s) + Ψ(s)D⁻⁺(s)Ψ(s) + Ψ(s)D⁻⁻(s) + D⁺⁺(s)Ψ(s) = 0.``

    PsiFun(; s = 0, D, MaxIters = 1000, err = 1e-8)

# Arguments
- `s::Real`: a value to evaluate the LST at
- `D`: a `Dict{String,Function(s::Real)}` as output from MakeD
- `MaxIters::Int`: the maximum number of iterations of newtons method
- `err::Float64`: an error tolerance for terminating newtons method. Terminates
    when `max(Ψ_{n} - Ψ{n-1}) .< eps`.

# Output
- `Ψ(s)::Array{Float64,2}`: a matrix approxiamtion to ``Ψ(s)``.
"""
function PsiFun(; s::Real = 0, D, MaxIters::Int = 1000, err::Float64 = 1e-8)
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
    println("UPDATE: Iterations for Ψ exited with flag: ", exitflag)
    return Psi
end

"""
Uses Eulers method to integrate the matrix DE ``f'(x) = f(x)D`` to
approxiamte ``f(y)``.

    EulerDG(;
        D::Union{Array{<:Real,2},SparseArrays.SparseMatrixCSC{Float64,Int64}},
        y::Real,
        x0::Array{<:Real},
        h::Float64 = 0.0001,
    )

# Arguments
- `D::Union{Array{<:Real,2},SparseArrays.SparseMatrixCSC{Float64,Int64}}`:
    the matrix ``D`` in ``f'(x) = f(x)D``.
- `y::Real`: the value where we want to evaluate ``f(y)``.
- `x0::Array{<:Real}`: a row-vector initial condition.
- `h::Float64`: a stepsize for theEuler scheme.

# Output
- `f(y)::Array`: a row-vector approximation to ``f(y)``
"""
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

"""
Convert from a vector of coefficients for the DG system to a distribution.

    Coeffs2Dist(;
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

# Arguments
- `Model`: a model object as output from MakeModel
- `Mesh`: a mesh object as output from MakeMesh
- `Coeffs::Array`: a vector of coefficients from the DG method
- `type::String`: an (optional) declaration of what type of distribution you
    want to convert to. Options are `"probability"` to return the probabilities
    ``P(X(t)∈ D_k, φ(t) = i)`` where ``D_k``is the kth cell, `"cumulative"` to
    return the CDF evaluated at cell edges, or `"density"` to return an
    approximation to the density ar at the Mesh.CellNodes.

# Output
- a tuple with keys
(pm=pm, distribution=yvals, x=xvals, type=type)
    - `pm::Array{Float64}`: a vector containing the point masses, the first
        `sum(Model.C.<=0)` entries are the left hand point masses and the last
        `sum(Model.C.>=0)` are the right-hand point masses.
    - `distribution::Array{Float64,3}`:
        - if `type="cumulative"` returns a `2×NIntervals×NPhases` array
            containing the CDF evaluated at the cell edges as contained in
            `x` below. i.e. `distribution[1,:,i]` returns the cdf at the
            left-hand edges of the cells in phase `i` and `distribution[2,:,i]`
            at the right hand edges.
        - if `type="probability"` returns a `1×NIntervals×NPhases` array
            containing the probabilities ``P(X(t)∈ D_k, φ(t) = i)`` where ``D_k``
            is the kth cell.
        - if `type="density"` returns a `NBases×NIntervals×NPhases` array
            containing the density function evaluated at the cell nodes as
            contained in `x` below.
    - `x::Array{Float64,2}`:
        - if `type="cumulative"` returns a `2×NIntervals×NPhases` array
            containing the cell edges as contained. i.e. `x[1,:]`
            returns the left-hand edges of the cells and `x[2,:]` at the
            right-hand edges.
        - if `type="probability"` returns a `1×NIntervals×NPhases` array
            containing the cell centers.
        - if `type="density"` returns a `NBases×NIntervals×NPhases` array
            containing the cell nodes.
    - `type`: as input in arguments.
"""
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
    Coeffs::Array,
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

    out = (pm=pm, distribution=yvals, x=xvals, type=type)
    println("UPDATE: distribution object created with keys ", keys(out))
    return out
end

"""
Converts a distribution as output from `Coeffs2Dist()` to a vector of DG
coefficients.

    Dist2Coeffs(;
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

# Arguments
- `Model`: a model object as output from MakeModel
- `Mesh`: a model object as output from MakeMesh
- `Distn::NamedTuple{(:pm, :distribution, :x, :type)}`: a distribution object
    i.e. a `NamedTuple` with fields
    - `pm::Array{Float64}`: a vector containing the point masses, the first
        `sum(Model.C.<=0)` entries are the left hand point masses and the last
        `sum(Model.C.>=0)` are the right-hand point masses.
    - `distribution::Array{Float64,3}`:
        - if `type="probability"` is a `1×NIntervals×NPhases` array containing
            the probabilities ``P(X(t)∈ D_k, φ(t) = i)`` where ``D_k``
            is the kth cell.
        - if `type="density"` is a `NBases×NIntervals×NPhases` array containing
            either the density function evaluated at the cell nodes which are in
            `x` below, or, the inner product of the density function against the
            lagrange polynomials.
    - `x::Array{Float64,2}`:
        - if `type="probability"` is a `1×NIntervals×NPhases` array
            containing the cell centers.
        - if `type="density"` is a `NBases×NIntervals×NPhases` array
            containing the cell nodes at which the denisty is evaluated.
    - `type::String`: either `"probability"` or `"density"`. `:cumulative` is
        not possible.
"""
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
    V = SFFM.vandermonde(NBases = Mesh.NBases)
    theDistribution =
        zeros(Float64, Mesh.NBases, Mesh.NIntervals, Model.NPhases)
    if Mesh.Basis == "legendre"
        if Distn.type == "probability"
            # for the legendre basis the first basis function is ϕ(x)=Δ√2 and
            # all other basis functions are orthogonal to this. Hence, we map
            # the cell probabilities to the first basis function only.
            theDistribution[1, :, :] = Distn.distribution./Mesh.Δ'.*sqrt(2)
        elseif Distn.type == "density"
            # if given density coefficients in lagrange form
            theDistribution = Distn.distribution
            for i = 1:Model.NPhases
                theDistribution[:, :, i] = V.inv * theDistribution[:, :, i]
            end
        end
        # also put the point masses on the ends
        coeffs = [
            Distn.pm[1:sum(Model.C .<= 0)]
            theDistribution[:]
            Distn.pm[sum(Model.C .<= 0)+1:end]
        ]
    elseif Mesh.Basis == "lagrange"
        theDistribution .= Distn.distribution
        if Distn.type == "probability"
            # convert to probability coefficients by multiplying by the
            # weights in V.w/2
            theDistribution = (V.w .* theDistribution / 2)[:]
        elseif Distn.type == "density"
            # convert to probability coefficients by multiplying by the
            # weights in V.w/2 and cell widths Δ
            theDistribution = ((V.w .* theDistribution).*(Mesh.Δ / 2)')[:]
        end
        # also put the point masses on the ends
        coeffs = [
            Distn.pm[1:sum(Model.C .<= 0)]
            theDistribution
            Distn.pm[sum(Model.C .<= 0)+1:end]
        ]
    end
    coeffs = Matrix(coeffs[:]')
    return coeffs
end

function starSeminorm(;
    d1::NamedTuple{(:pm, :distribution, :x, :type)},
    d2::NamedTuple{(:pm, :distribution, :x, :type)},
    )
    if d1.type!="probability" || d2.type!="probability"
        throw(ArgumentError("distributions need to be of type probability"))
    end
    return sum(abs.(d1.pm-d2.pm)) + sum(abs.(d1.distribution-d2.distribution))
end
