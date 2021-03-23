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
    NBases::Int
    CellNodes::Array{<:Real,2}
    Fil::Dict{String,BitArray{1}}
    Δ::Array{Float64,1}
    NIntervals::Int
    Nodes::Array{Float64,1}
    TotalNBases::Int
    Basis::String
    function DGMesh(
        model::SFFM.Model,
        Nodes::Array{<:Real,1},
        NBases::Int;
        Fil::Dict{String,BitArray{1}}=Dict{String,BitArray{1}}(),
        Basis::String = "lagrange",
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

        mesh = new(
            NBases,
            CellNodes,
            Fil,
            Δ,
            NIntervals,
            Nodes,
            TotalNBases,
            Basis,
        )
        println("UPDATE: DGMesh object created with fields ", fieldnames(SFFM.DGMesh))
        return mesh
    end
    function DGMesh()
        new(
            0,
            Array{Real,2}(undef,0,0),
            Dict{String,BitArray{1}}(),
            Array{Float64,1}(undef,0),
            0,
            Array{Float64,1}(undef,0),
            0,
            ""
        )
    end
end 

"""
Construct a generalised vandermonde matrix.

    vandermonde( NBases::Int)

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
function vandermonde(NBases::Int)
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

    MakeBlockDiagonalMatrix(
        mesh::DGMesh,
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
function MakeBlockDiagonalMatrix(
    mesh::DGMesh,
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
- `probTransform::Bool=true`: an (optional) specification for the lagrange basis
    to specify whether transform to probability coefficients.

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

    MakeMatrices(
        model::SFFM.Model,
        mesh::DGMesh;
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
function MakeMatrices(
    model::SFFM.Model,
    mesh::DGMesh;
    probTransform::Bool=true,
)
    ## Construct local blocks
    V = vandermonde(mesh.NBases)
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
        mesh,
        GLocal,
        ones(mesh.NIntervals),
    )
    M = SFFM.MakeBlockDiagonalMatrix(mesh, MLocal, mesh.Δ * 0.5)
    MInv = SFFM.MakeBlockDiagonalMatrix(
        mesh,
        MInvLocal,
        2.0 ./ mesh.Δ,
    )
    F = SFFM.MakeFluxMatrix(mesh, Phi, Dw, probTransform = probTransform)

    ## Assemble the DG drift operator
    Q = Array{SparseArrays.SparseMatrixCSC{Float64,Int64},1}(undef,model.NPhases)
    for i = 1:model.NPhases
        if model.C[i] > 0
            Q[i] = model.C[i] * (G + F["+"]) * MInv
        elseif model.C[i] < 0
            Q[i] = model.C[i] * (G + F["-"]) * MInv
        else
            Q[i] = SparseArrays.spzeros(size(G,1),size(G,2))
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
function MakeB(
    model::SFFM.Model,
    mesh::DGMesh,
    Matrices::NamedTuple;
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

    BDict = MakeDict(B, model, mesh)
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
Creates the DG approximation to the generator `B`.

    MakeB(
        model::SFFM.Model,
        mesh::DGMesh;
        probTransform::Bool=true,
    )

# Arguments
- `model`: A Model object
- `mesh`: A Mesh object
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
function MakeB(
    model::SFFM.Model,
    mesh::DGMesh;
    probTransform::Bool=true,
)
    M = SFFM.MakeMatrices(
        model,
        mesh;
        probTransform= probTransform,
    )
    B = SFFM.MakeB(model, mesh, M, probTransform = probTransform)
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


