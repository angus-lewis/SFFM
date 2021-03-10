"""
Constructs a Mesh object (a tuple with fields which describe the DG mesh).

    MakeMesh(;
        model::Model,
        Nodes::Array{Float64,1},
        NBases::Int,
        Fil::Dict{String,BitArray{1}}=Dict{String,BitArray{1}}(),
        Basis::String = "legendre",
    )

# Arguments
- `model`: a Model object
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
- a Mesh object with fieldnames:
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
    model::Model,
    Nodes::Array{<:Real,1},
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
    CellNodes[1,:] .+= sqrt(eps())
    if NBases>1
        CellNodes[end,:] .-= sqrt(eps())
    end
    TotalNBases = NBases * NIntervals # the total number of bases in the stencil

    ## Construct the sets Fᵐ = ⋃ᵢ Fᵢᵐ, global index for sets of type m
    if isempty(Fil)
        idxPlus = model.r.r(Nodes[1:end-1].+Δ[:]/2).>0
        idxZero = model.r.r(Nodes[1:end-1].+Δ[:]/2).==0
        idxMinus = model.r.r(Nodes[1:end-1].+Δ[:]/2).<0
        for i in 1:model.NPhases
            Fil[string(i)*"+"] = idxPlus[:,i]
            Fil[string(i)*"0"] = idxZero[:,i]
            Fil[string(i)*"-"] = idxMinus[:,i]
            if model.C[i] .<= 0
                Fil["p"*string(i)*"+"] = [model.r.r(model.Bounds[1,1])[i]].>0
                Fil["p"*string(i)*"0"] = [model.r.r(model.Bounds[1,1])[i]].==0
                Fil["p"*string(i)*"-"] = [model.r.r(model.Bounds[1,1])[i]].<0
            end
            if model.C[i] .>= 0
                Fil["q"*string(i)*"+"] = [model.r.r(model.Bounds[1,end])[i]].>0
                Fil["q"*string(i)*"0"] = [model.r.r(model.Bounds[1,end])[i]].==0
                Fil["q"*string(i)*"-"] = [model.r.r(model.Bounds[1,end])[i]].<0
            end
        end
    end
    CurrKeys = keys(Fil)
    for ℓ in ["+", "-", "0"], i = 1:model.NPhases
        if !in(string(i) * ℓ, CurrKeys)
            Fil[string(i)*ℓ] = falses(NIntervals)
        end
        if !in("p" * string(i) * ℓ, CurrKeys) && model.C[i] <= 0
            Fil["p"*string(i)*ℓ] = falses(1)
        end
        if !in("p" * string(i) * ℓ, CurrKeys) && model.C[i] > 0
            Fil["p"*string(i)*ℓ] = falses(0)
        end
        if !in("q" * string(i) * ℓ, CurrKeys) && model.C[i] >= 0
            Fil["q"*string(i)*ℓ] = falses(1)
        end
        if !in("q" * string(i) * ℓ, CurrKeys) && model.C[i] < 0
            Fil["q"*string(i)*ℓ] = falses(0)
        end
    end
    for ℓ in ["+", "-", "0"]
        Fil[ℓ] = falses(NIntervals * model.NPhases)
        Fil["p"*ℓ] = trues(0)
        Fil["q"*ℓ] = trues(0)
        for i = 1:model.NPhases
            idx = findall(Fil[string(i)*ℓ]) .+ (i - 1) * NIntervals
            Fil[string(ℓ)][idx] .= true
            Fil["p"*ℓ] = [Fil["p"*ℓ]; Fil["p"*string(i)*ℓ]]
            Fil["q"*ℓ] = [Fil["q"*ℓ]; Fil["q"*string(i)*ℓ]]
        end
    end

    mesh = SFFM.Mesh(
        NBases,
        CellNodes,
        Fil,
        Δ,
        NIntervals,
        Nodes,
        TotalNBases,
        Basis,
    )
    println("UPDATE: Mesh object created with fields ", fieldnames(SFFM.Mesh))
    return mesh
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

    MakeBlockDiagonalMatrix(;
        mesh::Mesh,
        Blocks::Array{Float64,2},
        Factors::Array,
    )

# Aguments
- `mesh`: A Mesh object
- `Blocks::Array{Float64,2}`: a `mesh.NBases×mesh.NBases` block to put along the
        diagonal
- `Factors::Array{<:Real,1}`: a `mesh.NIntervals×1` vector of factors which multiply blocks

# Output
- `BlockMatrix::Array{Float64,2}`: `mesh.TotalNBases×mesh.TotalNBases` the
        block matrix
"""
function MakeBlockDiagonalMatrix(;
    mesh::Mesh,
    Blocks::Array{Float64,2},
    Factors::Array,
)
    BlockMatrix = SparseArrays.spzeros(Float64, mesh.TotalNBases, mesh.TotalNBases)
    for i = 1:mesh.NIntervals
        idx = (1:mesh.NBases) .+ (i - 1) * mesh.NBases
        BlockMatrix[idx, idx] = Blocks * Factors[i]
    end
    return (BlockMatrix = BlockMatrix)
end

"""
Constructs the flux matrices for DG

    MakeFluxMatrix(;
        mesh::Mesh,
        model::Model,
        Phi,
        Dw,
        probTransform::Bool=true,
    )

# Arguments
- `mesh`: a Mesh object
- `Phi::Array{Float64,2}`: where `Phi[1,:]` and `Phi[1,:]` are the basis
    function evaluated at the left-hand and right-hand edge of a cell,
    respectively
- `Dw::Array{Float64,2}`: a diagonal matrix function weights
- `probTransform::Bool=true`: an (optional) specification for the lagrange basis
    to specify whether transform to probability coefficients.

# Output
- `F::Dict{String, SparseArrays.SparseMatrixCSC{Float64,Int64},1}`: a dictionary
    with keys `"+"` and `"-"` and values which are `TotalNBases×TotalNBases`
    flux matrices for `model.C[i]>0` and `model.C[i]<0`, respectively.
"""
function MakeFluxMatrix(;
    mesh::Mesh,
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
        F[i] = SparseArrays.spzeros(Float64, mesh.TotalNBases, mesh.TotalNBases)
        for k = 1:mesh.NIntervals
            idx = (1:mesh.NBases) .+ (k - 1) * mesh.NBases
            if i=="+"
                F[i][idx, idx] = PosDiagBlock
            elseif i=="-"
                F[i][idx, idx] = NegDiagBlock
            end # end if C[i]
            if k > 1
                idxup = (1:mesh.NBases) .+ (k - 2) * mesh.NBases
                if i=="+"
                    # the legendre basis works in density world so there are no etas
                    if mesh.Basis == "legendre"
                        η = 1
                    elseif mesh.Basis == "lagrange"
                        if !probTransform
                            η = 1
                        else
                            η = mesh.Δ[k] / mesh.Δ[k-1]
                        end
                    end
                    F[i][idxup, idx] = UpDiagBlock * η
                elseif i=="-"
                    if mesh.Basis == "legendre"
                        η = 1
                    elseif mesh.Basis == "lagrange"
                        if !probTransform
                            η = 1
                        else
                            η = mesh.Δ[k-1] / mesh.Δ[k]
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
Creates the Local and global mass, stiffness and flux matrices to compute `B`.

    MakeMatrices(;
        model::Model,
        mesh::Mesh,
        probTransform::Bool=true,
    )

# Arguments
- `model`: A Model object
- `mesh`: A Mesh object
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
    model::Model,
    mesh::Mesh,
    probTransform::Bool=true,
)
    ## Construct local blocks
    V = vandermonde(NBases = mesh.NBases)
    if mesh.Basis == "legendre"
        Dw = (
            DwInv = LinearAlgebra.diagm(0 => ones(Float64, mesh.NBases)),
            Dw = LinearAlgebra.diagm(0 => ones(Float64, mesh.NBases)),
        ) # function weights are not available for legendre basis as this is
        # in density land
        MLocal = Matrix{Float64}(LinearAlgebra.I(mesh.NBases))
        GLocal = V.inv * V.D
        MInvLocal = Matrix{Float64}(LinearAlgebra.I(mesh.NBases))
        Phi = V.V[[1; end], :]
    elseif mesh.Basis == "lagrange"
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
        mesh = mesh,
        Blocks = GLocal,
        Factors = ones(mesh.NIntervals),
    )
    M = SFFM.MakeBlockDiagonalMatrix(mesh = mesh, Blocks = MLocal, Factors = mesh.Δ * 0.5)
    MInv = SFFM.MakeBlockDiagonalMatrix(
        mesh = mesh,
        Blocks = MInvLocal,
        Factors = 2.0 ./ mesh.Δ,
    )
    F = SFFM.MakeFluxMatrix(mesh = mesh, Phi = Phi, Dw = Dw, probTransform = probTransform)

    ## Assemble the DG drift operator
    Q = Array{SparseArrays.SparseMatrixCSC{Float64,Int64},1}(undef,model.NPhases)
    for i = 1:model.NPhases
        if model.C[i] > 0
            Q[i] = model.C[i] * (G + F["+"]) * MInv
        elseif model.C[i] < 0
            Q[i] = model.C[i] * (G + F["-"]) * MInv
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
Creates the DG approximation to the generator `B`.

    MakeB(;
        model::Model,
        mesh::Mesh,
        Matrices::NamedTuple,
        probTransform::Bool=true,
    )

# Arguments
- `model`: A Model object
- `mesh`: A Mesh object
- `Matrices`: A Matrices tuple from `MakeMatrices`
- `probTransform::Bool=true`: an (optional) specification for the lagrange basis
    to specify whether transform to probability coefficients.

# Output
- A tuple with fields `:BDict, :B, :QBDidx`
    - `:BDict::Dict{String,Array{Float64,2}}`: a dictionary storing Bᵢⱼˡᵐ with
        keys string(i,j,ℓ,m), and values Bᵢⱼˡᵐ, i.e. `B.BDict["12+-"]` = B₁₂⁺⁻
    - `:B::SparseArrays.SparseMatrixCSC{Float64,Int64}`:
        `model.NPhases*mesh.TotalNBases×model.NPhases*mesh.TotalNBases`, the
        global approximation to `B`
    - `:QBDidx::Array{Int64,1}`: `model.NPhases*mesh.TotalNBases×1` vector of
        integers such such that `:B[QBDidx,QBDidx]` puts all the blocks relating
        to cell `k` next to each other
"""
function MakeB(;
    model::Model,
    mesh::Mesh,
    Matrices::NamedTuple,
    probTransform::Bool=true,
)
    ## Make B on the interior of the space
    N₊ = sum(model.C .>= 0)
    N₋ = sum(model.C .<= 0)
    B = SparseArrays.spzeros(
        Float64,
        model.NPhases * mesh.TotalNBases + N₋ + N₊,
        model.NPhases * mesh.TotalNBases + N₋ + N₊,
    )
    Id = SparseArrays.I(mesh.TotalNBases)
    for i = 1:model.NPhases
        idx = ((i-1)*mesh.TotalNBases+1:i*mesh.TotalNBases) .+ N₋
        B[idx, idx] = Matrices.Global.Q[i]
    end
    B[(N₋+1):(end-N₊), (N₋+1):(end-N₊)] =
        B[(N₋+1):(end-N₊), (N₋+1):(end-N₊)] + LinearAlgebra.kron(model.T, Id)

    # Boundary behaviour
    if mesh.Basis == "legendre"
        η = mesh.Δ[[1; end]] ./ 2 # this is the inverse of the η=mesh.Δ/2 bit
        # below there are no η's for the legendre basis
    elseif mesh.Basis == "lagrange"
        if !probTransform
            η = mesh.Δ[[1; end]] ./ 2
        else
            η = [1; 1]
        end
    end

    # Lower boundary
    # At boundary
    B[1:N₋, 1:N₋] = model.T[model.C.<=0, model.C.<=0]
    # Out of boundary
    idxup = ((1:mesh.NBases).+mesh.TotalNBases*(findall(model.C .> 0) .- 1)')[:] .+ N₋
    B[1:N₋, idxup] = kron(
        model.T[model.C.<=0, model.C.>0],
        Matrices.Local.Phi[1, :]' * Matrices.Local.Dw.Dw * Matrices.Local.MInv ./ η[1],
    )
    # Into boundary
    idxdown = ((1:mesh.NBases).+mesh.TotalNBases*(findall(model.C .<= 0) .- 1)')[:] .+ N₋
    B[idxdown, 1:N₋] = LinearAlgebra.kron(
        LinearAlgebra.diagm(0 => model.C[model.C.<=0]),
        -Matrices.Local.Dw.DwInv * 2.0 ./ mesh.Δ[1] * Matrices.Local.Phi[1, :] * η[1],
    )

    # Upper boundary
    # At boundary
    B[(end-N₊+1):end, (end-N₊+1):end] = model.T[model.C.>=0, model.C.>=0]
    # Out of boundary
    idxdown =
        ((1:mesh.NBases).+mesh.TotalNBases*(findall(model.C .< 0) .- 1)')[:] .+
        (N₋ + mesh.TotalNBases - mesh.NBases)
    B[(end-N₊+1):end, idxdown] = kron(
        model.T[model.C.>=0, model.C.<0],
        Matrices.Local.Phi[end, :]' * Matrices.Local.Dw.Dw * Matrices.Local.MInv  ./ η[end],
    )
    # Into boundary
    idxup =
        ((1:mesh.NBases).+mesh.TotalNBases*(findall(model.C .>= 0) .- 1)')[:] .+
        (N₋ + mesh.TotalNBases - mesh.NBases)
    B[idxup, (end-N₊+1):end] = LinearAlgebra.kron(
        LinearAlgebra.diagm(0 => model.C[model.C.>=0]),
        Matrices.Local.Dw.DwInv * 2.0 ./ mesh.Δ[end] * Matrices.Local.Phi[end, :] * η[end],
    )

    ## Make a Dictionary so that the blocks of B are easy to access
    BDict = Dict{String,SparseArrays.SparseMatrixCSC{Float64,Int64}}()
    ppositions = cumsum(model.C .<= 0)
    qpositions = cumsum(model.C .>= 0)
    for ℓ in ["+", "-", "0"], m in ["+", "-", "0"]
        for i = 1:model.NPhases, j = 1:model.NPhases
            FilBases = repeat(mesh.Fil[string(i, ℓ)]', mesh.NBases, 1)[:]
            pitemp = falses(N₋)
            qitemp = falses(N₊)
            pjtemp = falses(N₋)
            qjtemp = falses(N₊)
            if model.C[i] <= 0
                pitemp[ppositions[i]] = mesh.Fil["p"*string(i)*ℓ][1]
            end
            if model.C[j] <= 0
                pjtemp[ppositions[j]] = mesh.Fil["p"*string(j)*m][1]
            end
            if model.C[i] >= 0
                qitemp[qpositions[i]] = mesh.Fil["q"*string(i)*ℓ][1]
            end
            if model.C[j] >= 0
                qjtemp[qpositions[j]] = mesh.Fil["q"*string(j)*m][1]
            end
            i_idx = [
                pitemp
                falses((i - 1) * mesh.TotalNBases)
                FilBases
                falses(model.NPhases * mesh.TotalNBases - i * mesh.TotalNBases)
                qitemp
            ]
            FjmBases = repeat(mesh.Fil[string(j, m)]', mesh.NBases, 1)[:]
            j_idx = [
                pjtemp
                falses((j - 1) * mesh.TotalNBases)
                FjmBases
                falses(model.NPhases * mesh.TotalNBases - j * mesh.TotalNBases)
                qjtemp
            ]
            BDict[string(i, j, ℓ, m)] = B[i_idx, j_idx]
        end
        # note: below we need to use repeat(mesh.Fil[ℓ]', mesh.NBases, 1)[:] to
        # expand the index mesh.Fil[ℓ] from cells to all basis function
        FlBases =
            [mesh.Fil["p"*ℓ]; repeat(mesh.Fil[ℓ]', mesh.NBases, 1)[:]; mesh.Fil["q"*ℓ]]
        FmBases =
            [mesh.Fil["p"*m]; repeat(mesh.Fil[m]', mesh.NBases, 1)[:]; mesh.Fil["q"*m]]
        BDict[ℓ*m] = B[FlBases, FmBases]
    end

    ## Make QBD index
    c = N₋
    QBDidx = zeros(Int, model.NPhases * mesh.TotalNBases + N₊ + N₋)
    for k = 1:mesh.NIntervals, i = 1:model.NPhases, n = 1:mesh.NBases
        c += 1
        QBDidx[c] = (i - 1) * mesh.TotalNBases + (k - 1) * mesh.NBases + n + N₋
    end
    QBDidx[1:N₋] = 1:N₋
    QBDidx[(end-N₊+1):end] = (model.NPhases * mesh.TotalNBases + N₋) .+ (1:N₊)

    out = (BDict = BDict, B = B, QBDidx = QBDidx)
    println("UPDATE: B object created with keys ", keys(out))
    return out
end

"""
# Construct the DG approximation to the operator `R`.

    MakeR(;
        model::Model,
        mesh::Mesh,
        approxType::String = "projection",
        probTransform::Bool = true,
    )

# Arguments
- `Model`: a Model object
- `Mmesh`: a Mesh object
- `approxType::String`: (optional) either "interpolation" or
    "projection" (default).
- `probTransform::Bool=true`: an (optional) specification for the lagrange basis
    to specify whether transform to probability coefficients.

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
    model::Model,
    mesh::Mesh,
    approxType::String = "projection",
    probTransform::Bool = true,
)
    V = SFFM.vandermonde(NBases=mesh.NBases)

    EvalR = 1.0 ./ model.r.a(mesh.CellNodes[:])

    N₋ = sum(model.C .<= 0)
    N₊ = sum(model.C .>= 0)

    R = SparseArrays.spzeros(
        Float64,
        N₋ + N₊ + mesh.TotalNBases * model.NPhases,
        N₋ + N₊ + mesh.TotalNBases * model.NPhases,
    )
    # at the boundaries
    R[1:N₋, 1:N₋] = (1.0 ./ model.r.a(model.Bounds[1,1])[model.C .<= 0]).*LinearAlgebra.I(N₋)
    R[(end-N₊+1):end, (end-N₊+1):end] =  (1.0 ./ model.r.a(model.Bounds[1,end])[model.C .>= 0]).* LinearAlgebra.I(N₊)

    # on the interior
    for n = 1:(mesh.NIntervals*model.NPhases)
        if mesh.Basis == "legendre"
            if approxType == "interpolation"
                leftM = V.V'
                rightM = V.inv'
            elseif approxType == "projection"
                leftM = V.V' * LinearAlgebra.diagm(V.w)
                rightM = V.V
            end
            temp = leftM*LinearAlgebra.diagm(EvalR[mesh.NBases*(n-1).+(1:mesh.NBases)])*rightM
        elseif mesh.Basis == "lagrange"
            if approxType == "interpolation"
                temp = LinearAlgebra.diagm(EvalR[mesh.NBases*(n-1).+(1:mesh.NBases)])
            elseif approxType == "projection"
                # the first term, LinearAlgebra.diagm(EvalR[mesh.NBases*(n-1).+(1:mesh.NBases)])
                # is the quadrature approximation of M^r. The quadrature weights to not
                # appear since they cancel when we transform to integral/probability
                # representation. The second term V.V*V.V' is Minv. The last term
                # LinearAlgebra.diagm(V.w)is a result of the conversion to probability
                # / integral representation.
                if probTransform
                    temp = LinearAlgebra.diagm(EvalR[mesh.NBases*(n-1).+(1:mesh.NBases)])*V.V*V.V'*LinearAlgebra.diagm(V.w)
                elseif !probTransform
                    temp = LinearAlgebra.diagm(V.w)*LinearAlgebra.diagm(EvalR[mesh.NBases*(n-1).+(1:mesh.NBases)])*V.V*V.V'
                end
            end
        end
        R[mesh.NBases*(n-1).+(1:mesh.NBases).+N₋, mesh.NBases*(n-1).+(1:mesh.NBases).+N₋] = temp
    end

    # construc the dictionary
    RDict = Dict{String,SparseArrays.SparseMatrixCSC{Float64,Int64}}()
    ppositions = cumsum(model.C .<= 0)
    qpositions = cumsum(model.C .>= 0)
    for ℓ in ["+", "-"]
        for i = 1:model.NPhases
            FilBases = repeat(mesh.Fil[string(i, ℓ)]', mesh.NBases, 1)[:]
            pitemp = falses(N₋)
            qitemp = falses(N₊)
            if model.C[i] <= 0
                pitemp[ppositions[i]] = mesh.Fil["p"*string(i)*ℓ][1]
            end
            if model.C[i] >= 0
                qitemp[qpositions[i]] = mesh.Fil["q"*string(i)*ℓ][1]
            end
            i_idx = [
                pitemp
                falses((i - 1) * mesh.TotalNBases)
                FilBases
                falses(model.NPhases * mesh.TotalNBases - i * mesh.TotalNBases)
                qitemp
            ]
            RDict[string(i, ℓ)] = R[i_idx, i_idx]
        end
        FlBases =
            [mesh.Fil["p"*ℓ]; repeat(mesh.Fil[ℓ]', mesh.NBases, 1)[:]; mesh.Fil["q"*ℓ]]
        RDict[ℓ] = R[FlBases, FlBases]
    end

    out = (R=R, RDict=RDict)
    println("UPDATE: R object created with keys ", keys(out))
    return out
end

"""
Construct the operator `D(s)` from `B, R`.

    MakeD(;
        R,
        B,
        model::Model,
        mesh::Mesh,
    )

# Arguments
- `R`: a tuple as constructed by MakeR
- `B`: a tuple as constructed by MakeB
- `Model`: a Model object
- `mesh`: a Mesh object

# Output
- `DDict::Dict{String,Function(s::Real)}`: a dictionary of functions. Keys are
  of the for `"ℓm"` where `ℓ,m∈{+,-}`. Values are functions with one argument.
  Usage is along the lines of `D["+-"](s=1)`.
"""
function MakeD(;
    R::NamedTuple{(:R, :RDict)},
    B::NamedTuple{(:BDict, :B, :QBDidx)},
    model::Model,
    mesh::Mesh,
)
    DDict = Dict{String,Any}()
    for ℓ in ["+", "-"], m in ["+", "-"]
        nℓ = sum(mesh.Fil["p"*ℓ]) + sum(mesh.Fil[ℓ]) * mesh.NBases + sum(mesh.Fil["q"*ℓ])
        Idℓ = SparseArrays.sparse(LinearAlgebra.I,nℓ,nℓ)
        if any(mesh.Fil["p0"]) || any(mesh.Fil["0"]) || any(mesh.Fil["q0"]) # in("0", model.Signs)
            n0 = sum(mesh.Fil["p0"]) +
                sum(mesh.Fil["0"]) * mesh.NBases +
                sum(mesh.Fil["q0"])
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
    println("UPDATE: D(s) operator created with keys ", keys(DDict))
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
    println("UPDATE: Iterations for Ψ(s=", s,") exited with flag: ", exitflag)
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
        model::Model,
        mesh::Mesh,
        Coeffs,
        type::String = "probability",
        probTransform::Bool = true,
    )

# Arguments
- `Model`: a Model object
- `mesh`: a Mesh object as output from MakeMesh
- `Coeffs::Array`: a vector of coefficients from the DG method
- `type::String`: an (optional) declaration of what type of distribution you
    want to convert to. Options are `"probability"` to return the probabilities
    ``P(X(t)∈ D_k, φ(t) = i)`` where ``D_k``is the kth cell, `"cumulative"` to
    return the CDF evaluated at cell edges, or `"density"` to return an
    approximation to the density ar at the mesh.CellNodes.
- `probTransform::Bool` a boolean value specifying whether to transform to a
    probabilistic interpretation or not. Valid only for lagrange basis.

# Output
- a tuple with keys
(pm=pm, distribution=yvals, x=xvals, type=type)
    - `pm::Array{Float64}`: a vector containing the point masses, the first
        `sum(model.C.<=0)` entries are the left hand point masses and the last
        `sum(model.C.>=0)` are the right-hand point masses.
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
    model::Model,
    mesh::Mesh,
    Coeffs::Array,
    type::String = "probability",
    probTransform::Bool = true,
)
    V = SFFM.vandermonde(NBases = mesh.NBases)
    N₋ = sum(model.C .<= 0)
    N₊ = sum(model.C .>= 0)
    if !probTransform
        temp = reshape(Coeffs[N₋+1:end-N₊], mesh.NBases, mesh.NIntervals, model.NPhases)
        temp = V.w.*temp.*(mesh.Δ./2.0)'
        Coeffs = [Coeffs[1:N₋]; temp[:]; Coeffs[end-N₊+1:end]]
    end
    if type == "density"
        xvals = mesh.CellNodes
        if mesh.Basis == "legendre"
            yvals = reshape(Coeffs[N₋+1:end-N₊], mesh.NBases, mesh.NIntervals, model.NPhases)
            for i in 1:model.NPhases
                yvals[:,:,i] = V.V * yvals[:,:,i]
            end
            pm = [Coeffs[1:N₋]; Coeffs[end-N₊+1:end]]
        elseif mesh.Basis == "lagrange"
            yvals =
                Coeffs[N₋+1:end-N₊] .* repeat(1.0 ./ V.w, mesh.NIntervals * model.NPhases) .*
                (repeat(2.0 ./ mesh.Δ, 1, mesh.NBases * model.NPhases)'[:])
            yvals = reshape(yvals, mesh.NBases, mesh.NIntervals, model.NPhases)
            pm = [Coeffs[1:N₋]; Coeffs[end-N₊+1:end]]
        end
        if mesh.NBases == 1
            yvals = [1;1].*yvals
            xvals = [mesh.CellNodes-mesh.Δ'/2;mesh.CellNodes+mesh.Δ'/2]
        end
    elseif type == "probability"
        if mesh.NBases > 1
            xvals = mesh.CellNodes[1, :] + (mesh.Δ ./ 2)
        else
            xvals = mesh.CellNodes
        end
        if mesh.Basis == "legendre"
            yvals = (reshape(Coeffs[N₋+1:mesh.NBases:end-N₊], 1, mesh.NIntervals, model.NPhases).*mesh.Δ')./sqrt(2)
            pm = [Coeffs[1:N₋]; Coeffs[end-N₊+1:end]]
        elseif mesh.Basis == "lagrange"
            yvals = sum(
                reshape(Coeffs[N₋+1:end-N₊], mesh.NBases, mesh.NIntervals, model.NPhases),
                dims = 1,
            )
            pm = [Coeffs[1:N₋]; Coeffs[end-N₊+1:end]]
        end
    elseif type == "cumulative"
        if mesh.NBases > 1
            xvals = mesh.CellNodes[[1;end], :]
        else
            xvals = [mesh.CellNodes-mesh.Δ'/2;mesh.CellNodes+mesh.Δ'/2]
        end
        if mesh.Basis == "legendre"
            tempDist = (reshape(Coeffs[N₋+1:mesh.NBases:end-N₊], 1, mesh.NIntervals, model.NPhases).*mesh.Δ')./sqrt(2)
            pm = [Coeffs[1:N₋]; Coeffs[end-N₊+1:end]]
        elseif mesh.Basis == "lagrange"
            tempDist = sum(
                reshape(Coeffs[N₋+1:end-N₊], mesh.NBases, mesh.NIntervals, model.NPhases),
                dims = 1,
            )
            pm = [Coeffs[1:N₋]; Coeffs[end-N₊+1:end]]
        end
        tempDist = cumsum(tempDist,dims=2)
        temppm = zeros(Float64,1,2,model.NPhases)
        temppm[:,1,model.C.<=0] = pm[1:N₋]
        temppm[:,2,model.C.>=0] = pm[N₊+1:end]
        yvals = zeros(Float64,2,mesh.NIntervals,model.NPhases)
        yvals[1,2:end,:] = tempDist[1,1:end-1,:]
        yvals[2,:,:] = tempDist
        yvals = yvals .+ reshape(temppm[1,1,:],1,1,model.NPhases)
        pm[N₋+1:end] = pm[N₋+1:end] + yvals[end,end,model.C.>=0]
    end

    out = (pm=pm, distribution=yvals, x=xvals, type=type)
    println("UPDATE: distribution object created with keys ", keys(out))
    return out
end

"""
Converts a distribution as output from `Coeffs2Dist()` to a vector of DG
coefficients.

    Dist2Coeffs(;
        model::Model,
        mesh::Mesh,
        Distn::NamedTuple{(:pm, :distribution, :x, :type)},
        probTransform::Bool = true,
    )

# Arguments
- `Model`: a Model object
- `mesh`: a Mesh object as output from MakeMesh
- `Distn::NamedTuple{(:pm, :distribution, :x, :type)}`: a distribution object
    i.e. a `NamedTuple` with fields
    - `pm::Array{Float64}`: a vector containing the point masses, the first
        `sum(model.C.<=0)` entries are the left hand point masses and the last
        `sum(model.C.>=0)` are the right-hand point masses.
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
- `probTransform::Bool` a boolean value specifying whether to transform to a
    probabilistic interpretation or not. Valid only for lagrange basis.

# Output
- `coeffs` a row vector of coefficient values of length
    `TotalNBases*NPhases + N₋ + N₊` ordered according to LH point masses, RH
    point masses, interior basis functions according to basis function, cell,
    phase. Used to premultiply operators such as B from `MakeB()`
"""
function Dist2Coeffs(;
    model::Model,
    mesh::Mesh,
    Distn::NamedTuple{(:pm, :distribution, :x, :type)},
    probTransform::Bool = true,
)
    V = SFFM.vandermonde(NBases = mesh.NBases)
    theDistribution =
        zeros(Float64, mesh.NBases, mesh.NIntervals, model.NPhases)
    if mesh.Basis == "legendre"
        if Distn.type == "probability"
            # for the legendre basis the first basis function is ϕ(x)=Δ√2 and
            # all other basis functions are orthogonal to this. Hence, we map
            # the cell probabilities to the first basis function only.
            theDistribution[1, :, :] = Distn.distribution./mesh.Δ'.*sqrt(2)
        elseif Distn.type == "density"
            # if given density coefficients in lagrange form
            theDistribution = Distn.distribution
            for i = 1:model.NPhases
                theDistribution[:, :, i] = V.inv * theDistribution[:, :, i]
            end
        end
        # also put the point masses on the ends
        coeffs = [
            Distn.pm[1:sum(model.C .<= 0)]
            theDistribution[:]
            Distn.pm[sum(model.C .<= 0)+1:end]
        ]
    elseif mesh.Basis == "lagrange"
        theDistribution .= Distn.distribution
        if !probTransform
            theDistribution = (1.0./V.w) .* theDistribution .* (2.0./mesh.Δ')
        end
        if Distn.type == "probability"
            # convert to probability coefficients by multiplying by the
            # weights in V.w/2
            theDistribution = (V.w .* theDistribution / 2)[:]
        elseif Distn.type == "density"
            # convert to probability coefficients by multiplying by the
            # weights in V.w/2 and cell widths Δ
            theDistribution = ((V.w .* theDistribution).*(mesh.Δ / 2)')[:]
        end
        # also put the point masses on the ends
        coeffs = [
            Distn.pm[1:sum(model.C .<= 0)]
            theDistribution
            Distn.pm[sum(model.C .<= 0)+1:end]
        ]
    end
    coeffs = Matrix(coeffs[:]')
    return coeffs
end

"""
Computes the error between distributions.

    starSeminorm(;
        d1::NamedTuple{(:pm, :distribution, :x, :type)},
        d2::NamedTuple{(:pm, :distribution, :x, :type)},
        )

# Arguments
- `d1`: a distribution object as output from `Coeffs2Dist` with
    `type="probability"``
- `d2`: a distribution object as output from `Coeffs2Dist` with
    `type="probability"``
"""
function starSeminorm(;
    d1::NamedTuple{(:pm, :distribution, :x, :type)},
    d2::NamedTuple{(:pm, :distribution, :x, :type)},
    )
    if d1.type!="probability" || d2.type!="probability"
        throw(ArgumentError("distributions need to be of type probability"))
    end
    return sum(abs.(d1.pm-d2.pm)) + sum(abs.(d1.distribution-d2.distribution))
end
