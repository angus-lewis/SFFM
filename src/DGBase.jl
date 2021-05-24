"""
Constructs a DGMesh composite type, a subtype of the abstract type Mesh.

    DGMesh(
        model::SFFM.Model,
        Nodes::Array{Float64,1},
        NBases::Int;
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

#
A blank initialiser for a DGMesh.

    DGMesh()

Used for initialising a blank plot only. There is no reason to call this, ever. 
"""
struct DGMesh <: Mesh 
    Nodes::Array{<:Real,1}
    NBases::Int
    Fil::Dict{String,BitArray{1}}
    Basis::String
    function DGMesh(
        model::SFFM.Model,
        Nodes::Array{<:Real,1},
        NBases::Int;
        Fil::Dict{String,BitArray{1}}=Dict{String,BitArray{1}}(),
        Basis::String = "lagrange",
    )
        if isempty(Fil)
            Fil = MakeFil(model, Nodes)
        end

        mesh = new(Nodes, NBases, Fil, Basis)

        println("UPDATE: DGMesh object created with fields ", fieldnames(SFFM.DGMesh))
        return mesh
    end
    function DGMesh()
        new(
            [0.0],
            0,
            Dict{String,BitArray{1}}(),
            "",
        )
    end
end 

"""

    MakeFil(
        model::SFFM.Model,
        Nodes::Array{<:Real,1},
        )

Construct dict with entries indexing which cells belong to Fᵢᵐ. 
"""
function MakeFil(
    model::SFFM.Model,
    Nodes::Array{<:Real,1},
    )
    meshNIntervals = length(Nodes) - 1
    Δtemp = Nodes[2:end] - Nodes[1:end-1]

    Fil = Dict{String,BitArray{1}}()
    
    ## Construct the sets Fᵐ = ⋃ᵢ Fᵢᵐ, global index for sets of type m
    idxPlus = model.r.r(Nodes[1:end-1].+Δtemp[:]/2).>0
    idxZero = model.r.r(Nodes[1:end-1].+Δtemp[:]/2).==0
    idxMinus = model.r.r(Nodes[1:end-1].+Δtemp[:]/2).<0
    for i in 1:NPhases(model)
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
    currKeys = keys(Fil)
    for ℓ in ["+", "-", "0"], i = 1:NPhases(model)
        if !in(string(i) * ℓ, currKeys)
            Fil[string(i)*ℓ] = falses(meshNIntervals)
        end
        if !in("p" * string(i) * ℓ, currKeys) && model.C[i] <= 0
            Fil["p"*string(i)*ℓ] = falses(1)
        end
        if !in("p" * string(i) * ℓ, currKeys) && model.C[i] > 0
            Fil["p"*string(i)*ℓ] = falses(0)
        end
        if !in("q" * string(i) * ℓ, currKeys) && model.C[i] >= 0
            Fil["q"*string(i)*ℓ] = falses(1)
        end
        if !in("q" * string(i) * ℓ, currKeys) && model.C[i] < 0
            Fil["q"*string(i)*ℓ] = falses(0)
        end
    end
    for ℓ in ["+", "-", "0"]
        Fil[ℓ] = falses(meshNIntervals * NPhases(model))
        Fil["p"*ℓ] = trues(0)
        Fil["q"*ℓ] = trues(0)
        for i = 1:NPhases(model)
            idx = findall(Fil[string(i)*ℓ]) .+ (i - 1) * meshNIntervals
            Fil[string(ℓ)][idx] .= true
            Fil["p"*ℓ] = [Fil["p"*ℓ]; Fil["p"*string(i)*ℓ]]
            Fil["q"*ℓ] = [Fil["q"*ℓ]; Fil["q"*string(i)*ℓ]]
        end
    end
    return Fil
end

"""

    NBases(mesh::DGMesh)

Number of bases in a cell
"""
NBases(mesh::DGMesh) = mesh.NBases

"""

    NIntervals(mesh::Mesh)

Total number of cells for a mesh
"""
NIntervals(mesh::Mesh) = length(mesh.Nodes) - 1


"""

    Δ(mesh::Mesh)

The width of each cell
"""
Δ(mesh::Mesh) = mesh.Nodes[2:end] - mesh.Nodes[1:end-1]

"""

    TotalNBases(mesh::DGMesh)

Total number of bases in the stencil
"""
TotalNBases(mesh::Mesh) = NBases(mesh) * NIntervals(mesh)

"""

    CellNodes(mesh::DGMesh)

The positions of the GLJ nodes within each cell
"""
function CellNodes(mesh::DGMesh)
    meshNBases = NBases(mesh)
    meshNIntervals = NIntervals(mesh)
    cellNodes = zeros(Float64, NBases(mesh), NIntervals(mesh))
    if meshNBases > 1
        z = Jacobi.zglj(meshNBases, 0, 0) # the LGL nodes
    elseif meshNBases == 1
        z = 0.0
    end
    for i = 1:meshNIntervals
        # Map the LGL nodes on [-1,1] to each cell
        cellNodes[:, i] .= (mesh.Nodes[i+1] + mesh.Nodes[i]) / 2 .+ (mesh.Nodes[i+1] - mesh.Nodes[i]) / 2 * z
    end
    cellNodes[1,:] .+= sqrt(eps())
    if meshNBases>1
        cellNodes[end,:] .-= sqrt(eps())
    end
    return cellNodes
end

"""

    Basis(mesh::DGMesh)

Returns mesh.Basis; either "lagrange" or "legendre"
"""
Basis(mesh::DGMesh) = mesh.Basis


"""
Construct a generalised vandermonde matrix.

    vandermonde( nBases::Int)

Note: requires Jacobi package Pkg.add("Jacobi")

# Arguments
- `nBases::Int`: the degree of the basis

# Output
- a tuple with keys
    - `:V::Array{Float64,2}`: where `:V[:,i]` contains the values of the `i`th
        legendre polynomial evaluate at the GLL nodes.
    - `:inv`: the inverse of :V
    - `:D::Array{Float64,2}`: where `V.D[:,i]` contains the values of the derivative
        of the `i`th legendre polynomial evaluate at the GLL nodes.
"""
function vandermonde(nBases::Int)
    if nBases > 1
        z = Jacobi.zglj(nBases, 0, 0) # the LGL nodes
    elseif nBases == 1
        z = 0.0
    end
    V = zeros(Float64, nBases, nBases)
    DV = zeros(Float64, nBases, nBases)
    if nBases > 1
        for j = 1:nBases
            # compute the polynomials at gauss-labotto quadrature points
            V[:, j] = Jacobi.legendre.(z, j - 1) .* sqrt((2 * (j - 1) + 1) / 2)
            DV[:, j] = Jacobi.dlegendre.(z, j - 1) .* sqrt((2 * (j - 1) + 1) / 2)
        end
        # Compute the Gauss-Lobatto weights for numerical quadrature
        w =
            2.0 ./ (
                nBases *
                (nBases - 1) *
                Jacobi.legendre.(Jacobi.zglj(nBases, 0, 0), nBases - 1) .^ 2
            )
    elseif nBases == 1
        V .= [1/sqrt(2)]
        DV .= [0]
        w = [2]
    end
    return (V = V, inv = inv(V), D = DV, w = w)
end

"""
Constructs a block diagonal matrix from blocks

    MakeBlockDiagonalMatrix(
        mesh::DGMesh,
        Blocks::Array{Float64,2},
        Factors::Array,
    )

# Aguments
- `mesh`: A Mesh object
- `Blocks::Array{Float64,2}`: a `NBases(mesh)×NBases(mesh)` block to put along the
        diagonal
- `Factors::Array{<:Real,1}`: a `NIntervals(mesh)×1` vector of factors which multiply blocks

# Output
- `BlockMatrix::Array{Float64,2}`: `TotalNBases(mesh)×TotalNBases(mesh)` the
        block matrix
"""
function MakeBlockDiagonalMatrix(
    mesh::DGMesh,
    Blocks::Array{Float64,2},
    Factors::Array,
)
    BlockMatrix = kron(SparseArrays.spdiagm(0=>Factors), Blocks) # SparseArrays.spzeros(Float64, TotalNBases(mesh), TotalNBases(mesh))
    # for i = 1:NIntervals(mesh)
    #     idx = (1:NBases(mesh)) .+ (i - 1) * NBases(mesh)
    #     BlockMatrix[idx, idx] = Blocks * Factors[i]
    # end
    return (BlockMatrix = BlockMatrix)
end

"""
Constructs the flux matrices for DG

    MakeFluxMatrix(
        mesh::DGMesh,
        model::SFFM.Model,
        Phi,
        Dw;
        probTransform::Bool=true,
    )

# Arguments
- `mesh`: a Mesh object
- `Phi::Array{Float64,2}`: where `Phi[1,:]` and `Phi[1,:]` are the basis
    function evaluated at the left-hand and right-hand edge of a cell,
    respectively
- `Dw::Array{Float64,2}`: a diagonal matrix function weights

# Output
- `F::Dict{String, SparseArrays.SparseMatrixCSC{Float64,Int64},1}`: a dictionary
    with keys `"+"` and `"-"` and values which are `TotalNBases×TotalNBases`
    flux matrices for `model.C[i]>0` and `model.C[i]<0`, respectively.
"""
function MakeFluxMatrix(
    mesh::DGMesh,
    Phi,
    Dw;
    probTransform::Bool=true,
)
    ## Create the blocks
    PosDiagBlock = -Dw.DwInv * Phi[end, :] * Phi[end, :]' * Dw.Dw
    NegDiagBlock = Dw.DwInv * Phi[1, :] * Phi[1, :]' * Dw.Dw
    UpDiagBlock = Dw.DwInv * Phi[end, :] * Phi[1, :]' * Dw.Dw
    LowDiagBlock = -Dw.DwInv * Phi[1, :] * Phi[end, :]' * Dw.Dw

    ## Construct global block diagonal matrix
    if Basis(mesh) == "legendre"
        η = ones(NIntervals(mesh)-1)
    elseif Basis(mesh) == "lagrange"
        if probTransform
            η = Δ(mesh)[2:end] ./ Δ(mesh)[1:end-1]
        else 
            η = ones(NIntervals(mesh)-1)
        end
    end
    
    F = Dict{String,SparseArrays.SparseMatrixCSC{Float64,Int64}}()
    ncells = NIntervals(mesh)
    F["+"] = kron( SparseArrays.spdiagm(0=>ones(ncells)), PosDiagBlock) + kron( SparseArrays.spdiagm(1=>η), UpDiagBlock)
    F["-"] = kron( SparseArrays.spdiagm(0=>ones(ncells)), NegDiagBlock) + kron( SparseArrays.spdiagm(-1=> (1 ./η)), LowDiagBlock)
    return (F = F)
end

"""
Creates the Local and global mass, stiffness and flux matrices to compute `B`.

    MakeMatrices(
        model::SFFM.Model,
        mesh::DGMesh;
        probTransform::Bool=true,
    )

# Arguments
- `model`: A Model object
- `mesh`: A Mesh object

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
function MakeMatrices(
    model::SFFM.Model,
    mesh::DGMesh;
    probTransform::Bool=true,
)
    ## Construct local blocks
    V = vandermonde(NBases(mesh))
    if Basis(mesh) == "legendre"
        Dw = (
            DwInv = LinearAlgebra.diagm(0 => ones(Float64, NBases(mesh))),
            Dw = LinearAlgebra.diagm(0 => ones(Float64, NBases(mesh))),
        ) # function weights are not available for legendre basis as this is
        # in density land
        MLocal = Matrix{Float64}(LinearAlgebra.I(NBases(mesh)))
        GLocal = V.inv * V.D
        MInvLocal = Matrix{Float64}(LinearAlgebra.I(NBases(mesh)))
        Phi = V.V[[1; end], :]
    elseif Basis(mesh) == "lagrange"
        if probTransform
            Dw = (
                DwInv = LinearAlgebra.diagm(0 => 1.0 ./ V.w),
                Dw = LinearAlgebra.diagm(0 => V.w),
            )# function weights so that we can work in probability land as
            # opposed to density land
        else
            Dw = (
                DwInv = LinearAlgebra.I,
                Dw = LinearAlgebra.I,
            )
        end

        MLocal = Dw.DwInv * V.inv' * V.inv * Dw.Dw
        GLocal = Dw.DwInv * V.inv' * V.inv * (V.D * V.inv) * Dw.Dw
        MInvLocal = Dw.DwInv * V.V * V.V' * Dw.Dw
        Phi = (V.inv*V.V)[[1; end], :]
    end

    ## Assemble into global block diagonal matrices
    G = SFFM.MakeBlockDiagonalMatrix(
        mesh,
        GLocal,
        ones(NIntervals(mesh)),
    )
    M = SFFM.MakeBlockDiagonalMatrix(mesh, MLocal, Δ(mesh) * 0.5)
    MInv = SFFM.MakeBlockDiagonalMatrix(
        mesh,
        MInvLocal,
        2.0 ./ Δ(mesh),
    )
    F = SFFM.MakeFluxMatrix(mesh, Phi, Dw, probTransform = probTransform)

    ## Assemble the DG drift operator
    up = model.C.*(model.C .> 0)
    down = model.C.*(model.C .< 0)
    Q = kron( SparseArrays.spdiagm(0=>up), (G + F["+"]) * MInv) + kron( SparseArrays.spdiagm(0=>down), (G + F["-"]) * MInv) 
    # Q = Array{SparseArrays.SparseMatrixCSC{Float64,Int64},1}(undef,NPhases(model))
    # for i = 1:NPhases(model)
    #     if model.C[i] > 0
    #         Q[i] = model.C[i] * (G + F["+"]) * MInv
    #     elseif model.C[i] < 0
    #         Q[i] = model.C[i] * (G + F["-"]) * MInv
    #     else
    #         Q[i] = SparseArrays.spzeros(size(G,1),size(G,2))
    #     end
    # end

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

    MakeB(
        model::SFFM.Model,
        mesh::DGMesh,
        Matrices::NamedTuple;
        probTransform::Bool=true,
    )

# Arguments
- `model`: A Model object
- `mesh`: A Mesh object
- `Matrices`: A Matrices tuple from `MakeMatrices`

# Output
- A tuple with fields `:BDict, :B, :QBDidx`
    - `:BDict::Dict{String,Array{Float64,2}}`: a dictionary storing Bᵢⱼˡᵐ with
        keys string(i,j,ℓ,m), and values Bᵢⱼˡᵐ, i.e. `B.BDict["12+-"]` = B₁₂⁺⁻
    - `:B::SparseArrays.SparseMatrixCSC{Float64,Int64}`:
        `NPhases(model)*TotalNBases(mesh)×NPhases(model)*TotalNBases(mesh)`, the
        global approximation to `B`
    - `:QBDidx::Array{Int64,1}`: `NPhases(model)*TotalNBases(mesh)×1` vector of
        integers such such that `:B[QBDidx,QBDidx]` puts all the blocks relating
        to cell `k` next to each other
"""
function MakeB(
    model::SFFM.Model,
    mesh::DGMesh,
    Matrices::NamedTuple;
    probTransform::Bool=true,
)
    ## Make B on the interior of the space
    N₊ = sum(model.C .>= 0)
    N₋ = sum(model.C .<= 0)
    # B = SparseArrays.spzeros(
    #     Float64,
    #     NPhases(model) * TotalNBases(mesh) + N₋ + N₊,
    #     NPhases(model) * TotalNBases(mesh) + N₋ + N₊,
    # )
    Id = SparseArrays.I(TotalNBases(mesh))
    # for i = 1:NPhases(model)
    #     idx = ((i-1)*TotalNBases(mesh)+1:i*TotalNBases(mesh)) .+ N₋
    #     B[idx, idx] = Matrices.Global.Q[i]
    # end
    # B[(N₋+1):(end-N₊), (N₋+1):(end-N₊)] +=
    #     B[(N₋+1):(end-N₊), (N₋+1):(end-N₊)] + LinearAlgebra.kron(model.T, Id) 
    B = LinearAlgebra.kron(model.T, Id) + Matrices.Global.Q

    # Boundary behaviour
    if Basis(mesh) == "legendre"
        η = Δ(mesh)[[1; end]] ./ 2 # this is the inverse of the η=Δ(mesh)/2 bit
        # below there are no η's for the legendre basis
    elseif Basis(mesh) == "lagrange"
        if probTransform
            η = [1; 1]
        else
            η = Δ(mesh)[[1; end]] ./ 2
        end
    end
    
    # Lower boundary
    tmp = Matrices.Local.Phi[1, :]' * Matrices.Local.Dw.Dw * Matrices.Local.MInv ./ η[1]
    top = [model.T[model.C.<=0, model.C.<=0] kron( kron([1 SparseArrays.spzeros(1,NIntervals(mesh)-1)], model.T[model.C.<=0, :].*(model.C.>0)'),tmp) SparseArrays.spzeros(sum(model.C.<=0), sum(model.C.>=0))]
    # At boundary
    # B[1:N₋, 1:N₋] = model.T[model.C.<=0, model.C.<=0]
    # Out of boundary
    # idxup = ((1:NBases(mesh)).+TotalNBases(mesh)*(findall(model.C .> 0) .- 1)')[:] .+ N₋
    # B[1:N₋, idxup] = kron(
    #     model.T[model.C.<=0, model.C.>0],
    #     Matrices.Local.Phi[1, :]' * Matrices.Local.Dw.Dw * Matrices.Local.MInv ./ η[1],
    # )
    # Into boundary
    lft = SparseArrays.spzeros(TotalNBases(mesh).*NPhases(model), N₋)
    idxdown = ((1:NBases(mesh)).+TotalNBases(mesh)*(findall(model.C .<= 0) .- 1)')[:]
    # B[idxdown, 1:N₋] 
    lft[idxdown,:] = LinearAlgebra.kron(
        LinearAlgebra.diagm(0 => model.C[model.C.<=0]),
        -Matrices.Local.Dw.DwInv * 2.0 ./ Δ(mesh)[1] * Matrices.Local.Phi[1, :] * η[1],
    )

    # Upper boundary
    tmp = Matrices.Local.Phi[end, :]' * Matrices.Local.Dw.Dw * Matrices.Local.MInv  ./ η[end]
    btm = [SparseArrays.spzeros(sum(model.C.>=0), sum(model.C.<=0)) kron( kron([SparseArrays.spzeros(1,NIntervals(mesh)-1) 1], model.T[model.C.>=0, :].*(model.C.<0)'), tmp) model.T[model.C.>=0, model.C.>=0]]
    # At boundary
    # B[(end-N₊+1):end, (end-N₊+1):end] = model.T[model.C.>=0, model.C.>=0]
    # # Out of boundary
    # idxdown =
    #     ((1:NBases(mesh)).+TotalNBases(mesh)*(findall(model.C .< 0) .- 1)')[:] .+
    #     (N₋ + TotalNBases(mesh) - NBases(mesh))
    # B[(end-N₊+1):end, idxdown] = kron(
    #     model.T[model.C.>=0, model.C.<0],
    #     Matrices.Local.Phi[end, :]' * Matrices.Local.Dw.Dw * Matrices.Local.MInv  ./ η[end],
    # )
    # Into boundary
    rght = SparseArrays.spzeros(TotalNBases(mesh).*NPhases(model), N₊)
    # LinearAlgebra.kron(
    #     LinearAlgebra.diagm(0 => model.C[model.C.>=0]),
    #     Matrices.Local.Dw.DwInv * 2.0 ./ Δ(mesh)[end] * Matrices.Local.Phi[end, :] * η[end],
    # )
    idxup =
        ((1:NBases(mesh)).+TotalNBases(mesh)*(findall(model.C .>= 0) .- 1)')[:] .+
        (TotalNBases(mesh) - NBases(mesh))
    # B[idxup, (end-N₊+1):end] = 
    rght[idxup,:] = LinearAlgebra.kron(
        LinearAlgebra.diagm(0 => model.C[model.C.>=0]),
        Matrices.Local.Dw.DwInv * 2.0 ./ Δ(mesh)[end] * Matrices.Local.Phi[end, :] * η[end],
    )
    
    B = vcat(top,hcat(lft, B, rght),btm)

    BDict = MakeDict(B, model, mesh)
    ## Make QBD index
    c = N₋
    QBDidx = zeros(Int, NPhases(model) * TotalNBases(mesh) + N₊ + N₋)
    for k = 1:NIntervals(mesh), i = 1:NPhases(model), n = 1:NBases(mesh)
        c += 1
        QBDidx[c] = (i - 1) * TotalNBases(mesh) + (k - 1) * NBases(mesh) + n + N₋
    end
    QBDidx[1:N₋] = 1:N₋
    QBDidx[(end-N₊+1):end] = (NPhases(model) * TotalNBases(mesh) + N₋) .+ (1:N₊)

    out = (BDict = BDict, B = B, QBDidx = QBDidx)
    println("UPDATE: B object created with keys ", keys(out))
    return out
end

"""
Creates the DG approximation to the generator `B`.

    MakeB(
        model::SFFM.Model,
        mesh::DGMesh;
        probTransform::Bool=true,
    )

# Arguments
- `model`: A Model object
- `mesh`: A Mesh object

# Output
- A tuple with fields `:BDict, :B, :QBDidx`
    - `:BDict::Dict{String,Array{Float64,2}}`: a dictionary storing Bᵢⱼˡᵐ with
        keys string(i,j,ℓ,m), and values Bᵢⱼˡᵐ, i.e. `B.BDict["12+-"]` = B₁₂⁺⁻
    - `:B::SparseArrays.SparseMatrixCSC{Float64,Int64}`:
        `NPhases(model)*TotalNBases(mesh)×NPhases(model)*TotalNBases(mesh)`, the
        global approximation to `B`
    - `:QBDidx::Array{Int64,1}`: `NPhases(model)*TotalNBases(mesh)×1` vector of
        integers such such that `:B[QBDidx,QBDidx]` puts all the blocks relating
        to cell `k` next to each other
"""
function MakeB(
    model::SFFM.Model,
    mesh::DGMesh;
    probTransform::Bool=true,
)
    M = SFFM.MakeMatrices(
        model,
        mesh;
        probTransform=probTransform,
    )
    B = SFFM.MakeB(model, mesh, M; probTransform=probTransform)
    println("UPDATE: B object created with keys ", keys(B))
    return B
end

"""
Uses Eulers method to integrate the matrix DE ``f'(x) = f(x)D`` to
approxiamte ``f(y)``.

    EulerDG(
        D::Union{Array{<:Real,2},SparseArrays.SparseMatrixCSC{Float64,Int64}},
        y::Real,
        x0::Array{<:Real};
        h::Float64 = 0.0001,
    )

# Arguments
- `D::Union{Array{<:Real,2},SparseArrays.SparseMatrixCSC{Float64,Int64}}`:
    the matrix ``D`` in the system of ODEs ``f'(x) = f(x)D``.
- `y::Real`: the value where we want to evaluate ``f(y)``.
- `x0::Array{<:Real}`: a row-vector initial condition.
- `h::Float64`: a stepsize for theEuler scheme.

# Output
- `f(y)::Array`: a row-vector approximation to ``f(y)``
"""
function EulerDG(
    D::Union{Array{<:Real,2},SparseArrays.SparseMatrixCSC{Float64,Int64}},
    y::Real,
    x0::Array{<:Real};
    h::Float64 = 0.0001,
)
    x = x0
    for t = h:h:y
        dx = h * x * D
        x = x + dx
    end
    return x
end


