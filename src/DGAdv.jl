"""
Constructs a block diagonal matrix from blocks

    MakeBlockDiagonalMatrixR(
        model::SFFM.Model,
        mesh::DGMesh,
        Blocks,
        Factors::Array,
    )

# Aguments
- `model`: A Model object
- `mesh`: A DGMesh object from `DGMesh()`
- `Blocks(x::Array{Float64}, i::Int)`: a function wich returns a
    `NBases(mesh)×NBases(mesh)` array to put along the diagonal. The input
    argument `x` is a column vector of corresponding to the nodes in each cell,
    i.e. `SFFM.CellNodes(mesh)[:,n]`. `i` denotes the phase.
- `Factors::Array{<:Real,1}`: a `NIntervals(mesh)×1` vector of factors which multiply blocks

# Output
- `BlockMatrix::Array{Float64,2}`: `TotalNBases(mesh)×TotalNBases(mesh)` the
        block matrix
"""
function MakeBlockDiagonalMatrixR(
    model::SFFM.Model,
    mesh::DGMesh,
    Blocks,
    Factors::Array,
)
    BlockMatrix = Array{SparseArrays.SparseMatrixCSC{Float64,Int64},1}(undef,NPhases(model))
    for i in 1:NPhases(model)
        BlockMatrix[i] = SparseArrays.spzeros(Float64, TotalNBases(mesh), TotalNBases(mesh))
    end
    for i = 1:NIntervals(mesh), j = 1:NPhases(model)
        idx = (1:NBases(mesh)) .+ (i - 1) * NBases(mesh)
        BlockMatrix[j][idx, idx] = Blocks(SFFM.CellNodes(mesh)[:, i], j) * Factors[i]
    end
    return (BlockMatrix = BlockMatrix)
end

"""
Constructs the flux matrices for DG

    MakeFluxMatrixR(
        mesh::DGMesh,
        model::SFFM.Model,
        Phi,
    )

# Arguments
- `mesh`: a Mesh object from MakeMesh
- `model`: A Model object
- `Phi::Array{Float64,2}`: where `Phi[1,:]` and `Phi[1,:]` are the basis
    function evaluated at the left-hand and right-hand edge of a cell,
    respectively

# Output
- `F::Array{SparseArrays.SparseMatrixCSC{Float64,Int64},1}(undef,NPhases(model))`:
    an array with index `i ∈ 1:NPhases(model)`, of sparse arrays which are
    `TotalNBases×TotalNBases` flux matrices for phase `i`.
"""
function MakeFluxMatrixR(
    mesh::DGMesh,
    model::SFFM.Model,
    Phi,
)
    ## Create the blocks
    PosDiagBlock = -Phi[end, :] * Phi[end, :]'
    NegDiagBlock = Phi[1, :] * Phi[1, :]'
    UpDiagBlock = Phi[end, :] * Phi[1, :]'
    LowDiagBlock = -Phi[1, :] * Phi[end, :]'

    ## Construct global block diagonal matrix
    F = Array{SparseArrays.SparseMatrixCSC{Float64,Int64},1}(undef,NPhases(model))
    for i = 1:NPhases(model)
        F[i] = SparseArrays.spzeros(Float64, TotalNBases(mesh), TotalNBases(mesh))
        for k = 1:NIntervals(mesh)
            idx = (1:NBases(mesh)) .+ (k - 1) * NBases(mesh)
            if model.C[i] > 0
                xright = SFFM.CellNodes(mesh)[end, k]
                R = 1.0 ./ model.r.a(xright)[i]
                F[i][idx, idx] = PosDiagBlock * R
            elseif model.C[i] < 0
                xleft = SFFM.CellNodes(mesh)[1, k]
                R = 1.0 ./ model.r.a(xleft)[i]
                F[i][idx, idx] = NegDiagBlock * R
            end # end if C[i]
            if k > 1
                idxup = (1:NBases(mesh)) .+ (k - 2) * NBases(mesh)
                if model.C[i] > 0
                    xright = SFFM.CellNodes(mesh)[end, k-1]
                    R = 1.0 ./ model.r.a(xright)[i]
                    η = (Δ(mesh)[k] / NBases(mesh)) / (Δ(mesh)[k-1] / NBases(mesh))
                    F[i][idxup, idx] = UpDiagBlock * η * R
                elseif model.C[i] < 0
                    xleft = SFFM.CellNodes(mesh)[1, k]
                    R = 1.0 ./ model.r.a(xleft)[i]
                    η = (Δ(mesh)[k-1] / NBases(mesh)) / (Δ(mesh)[k] / NBases(mesh))
                    F[i][idx, idxup] = LowDiagBlock * η * R
                end # end if C[i]
            end # end if k>1
        end # for k in ...
    end # end for i in NPhases

    return (F = F)
end

"""
Creates the Local and global mass, stiffness and flux matrices to compute `D(s)`
directly.

    MakeMatricesR(
        model::SFFM.Model,
        mesh::DGMesh,
    )

# Arguments
- `model`: A Model object
- `mesh`: A mesh object

# Output
- A tuple of tuples
    - `:Global`: a tuple with fields
      - `:G::Array{SparseArrays.SparseMatrixCSC{Float64,Int64},1}`:
            a `NPhases` length array containing `TotalNBases×TotalNBases`
            dimensional global stiffness matrices
      - `:M::Array{SparseArrays.SparseMatrixCSC{Float64,Int64},1}`:
            a `NPhases` length array containing `TotalNBases×TotalNBases`
            dimensional global mass matrices
      - `:MInv::Array{SparseArrays.SparseMatrixCSC{Float64,Int64},1}`:
            a `NPhases` length array containing `TotalNBases×TotalNBases`
            dimensional matrices, the inverse of `Global.M`
      - `:F::Array{SparseArrays.SparseMatrixCSC{Float64,Int64},1}`:
            a `NPhases` length array containing `TotalNBases×TotalNBases`
            dimensional global flux matrices
      - `:Q::Array{SparseArrays.SparseMatrixCSC{Float64,Int64},1}`:
            a `NPhases` length array containing `TotalNBases×TotalNBases`
            dimensional global DG drift operator
    - `:Local`: a tuple with fields
      - `:G::Array{Float64,2}`: `NBases×NBases` Local stiffness matrix
      - `:M::Array{Float64,2}`: `NBases×NBases` Local mass matrix
      - `:MInv::Array{Float64,2}`: the inverse of `Local.M`
      - `:V::NamedTuple`: as output from SFFM.vandermonde
 """
function MakeMatricesR(
    model::SFFM.Model,
    mesh::DGMesh,
)
    ## Construct blocks
    V = vandermonde(NBases(mesh))
    if Basis(mesh) == "legendre"
        MLocal = function (x::Array{Float64}, i::Int)
            # Numerical integration of ϕᵢ(x)|r(x)|ϕⱼ(x) over Dk with Gauss-Lobatto
            # quadrature
            # Inputs:
            #   - x a vector of Gauss-Lobatto points on Dk
            #   - i a phase
            V.V' * LinearAlgebra.diagm(V.w ./ model.r.a(x)[:, i]) * V.V
        end
        GLocal = function (x::Array{Float64}, i::Int)
            # Numerical integration of ϕᵢ(x)|r(x)|ϕⱼ'(x) over Dk with Gauss-Lobatto
            # quadrature
            # Inputs:
            #   - x a vector of Gauss-Lobatto points on Dk
            #   - i a phase
            V.V' * LinearAlgebra.diagm(V.w ./ model.r.a(x)[:, i]) * V.D
        end
        MInvLocal = function (x::Array{Float64}, i::Int)
            MLocal(x, i)^-1
        end
        Phi = V.V[[1; end], :]

    elseif Basis(mesh) == "lagrange"
        MLocal = function (x::Array{Float64}, i::Int)
            # Numerical integration of ϕᵢ(x)|r(x)|ϕⱼ(x) over Dk with Gauss-Lobatto
            # quadrature
            # Inputs:
            #   - x a vector of Gauss-Lobatto points on Dk
            #   - i a phase
            LinearAlgebra.diagm(V.w ./ model.r.a(x)[:, i])
        end
        GLocal = function (x::Array{Float64}, i::Int)
            # Numerical integration of ϕᵢ(x)|r(x)|ϕⱼ'(x) over Dk with Gauss-Lobatto
            # quadrature
            # Inputs:
            #   - x a vector of Gauss-Lobatto points on Dk
            #   - i a phase
            MLocal(x, i) * V.D * V.inv
        end
        MInvLocal = function (x::Array{Float64}, i::Int)
            MLocal(x, i)^-1
        end
        Phi = (V.inv*V.V)[[1; end], :]
    end

    ## Assemble into block diagonal matrices
    G = SFFM.MakeBlockDiagonalMatrixR(
        model,
        mesh,
        GLocal,
        ones(NIntervals(mesh)),
    )
    M = SFFM.MakeBlockDiagonalMatrixR(
        model,
        mesh,
        MLocal,
        Δ(mesh) * 0.5,
    )
    MInv = SFFM.MakeBlockDiagonalMatrixR(
        model,
        mesh,
        MInvLocal,
        2.0 ./ Δ(mesh),
    )

    F = SFFM.MakeFluxMatrixR(mesh, model, Phi)

    ## Assemble the DG drift operator
    Q = Array{SparseArrays.SparseMatrixCSC{Float64,Int64},1}(undef,NPhases(model))
    for i = 1:NPhases(model)
        Q[i] = model.C[i] * (G[i] + F[i])
    end

    Local = (G = GLocal, M = MLocal, MInv = MInvLocal, V = V, Phi = Phi)
    Global = (G = G, M = M, MInv = MInv, F = F, Q = Q)
    out = (Local = Local, Global = Global)
    println("UPDATE: MatricesR object created with keys ", keys(out))
    println("UPDATE:    MatricesR[:",keys(out)[1],"] object created with keys ", keys(out[1]))
    println("UPDATE:    MatricesR[:",keys(out)[2],"] object created with keys ", keys(out[2]))
    return out
end

"""
Construct the operator `D(s)` directly.

    MakeDR(
        Matrices,
        MatricesR,
        model::SFFM.Model,
        mesh::DGMesh,
        B,
    )

# Arguments
- `Matrices`: a tuple as output from
    `Matrices = SFFM.MakeMatrices(model=model,mesh=mesh,probTransform=false)`
    note, must have `probTransform=false`.
- `MatricesR`: a tuple as output from
    `MatricesR = SFFM.MakeMatricesR(model=approxModel,mesh=mesh)`
- `model`: a Model object 
- `mesh`: a Mesh object as constructed by `MakeMesh`

# Output
- a tuple with keys `:DDict` and `:DR`
    - `DDict::Dict{String,Function(s::Real)}`: a dictionary of functions. Keys are
      of the for `"ℓm"` where `ℓ,m∈{+,-}`. Values are functions with one argument.
      Usage is along the lines of `D["+-"](s=1)`.
    - `DR`: a function which evaluates `D(s)`,
      Usage is along the lines of `DR(s=1)`. Returns an array structured as
      `D = [D["++"](s=1) D["+-"](s=1); D["-+"](s=1) D["--"](s=1)]`.
"""
function MakeDR(
    Matrices,
    MatricesR,
    model::SFFM.Model,
    mesh::DGMesh,
    B,
)
    N₊ = sum(model.C .>= 0)
    N₋ = sum(model.C .<= 0)

    BigN = NPhases(model) * TotalNBases(mesh) + N₊ + N₋
    MR = SparseArrays.spzeros(Float64, NPhases(model) * TotalNBases(mesh), NPhases(model) * TotalNBases(mesh))
    Minv =
        SparseArrays.spzeros(Float64, NPhases(model) * TotalNBases(mesh), NPhases(model) * TotalNBases(mesh))
    FGR = SparseArrays.spzeros(Float64, NPhases(model) * TotalNBases(mesh), NPhases(model) * TotalNBases(mesh))
    for i = 1:NPhases(model)
        idx = ((i-1)*TotalNBases(mesh)+1:i*TotalNBases(mesh))
        MR[idx, idx] = MatricesR.Global.M[i]
        Minv[idx, idx] = Matrices.Global.MInv
        FGR[idx, idx] = MatricesR.Global.Q[i] * Matrices.Global.MInv
    end

    # Interior behaviour
    T = SparseArrays.kron(model.T, SparseArrays.sparse(LinearAlgebra.I,TotalNBases(mesh),TotalNBases(mesh)))
    BR = SparseArrays.spzeros(Float64, BigN, BigN)
    BR[(N₋+1):(end-N₊), (N₋+1):(end-N₊)] = MR * T * Minv + FGR

    # Boundary behaviour
    # Lower boundary
    # At boundary
    BR[1:N₋, 1:N₋] = (1.0./ model.r.a(mesh.Nodes[1])'.*model.T)[model.C.<=0, model.C.<=0]
    # Out of boundary
    idxup = ((1:NBases(mesh)).+TotalNBases(mesh)*(findall(model.C .> 0) .- 1)')[:] .+ N₋
    BR[1:N₋, idxup] = kron(
        (1.0./model.r.a(mesh.Nodes[1])'.*model.T)[model.C.<=0, model.C.>0],
        Matrices.Local.Phi[1, :]' * Matrices.Local.MInv * 2 ./ Δ(mesh)[1],
    )
    # Into boundary
    idxdown = ((1:NBases(mesh)).+TotalNBases(mesh)*(findall(model.C .<= 0) .- 1)')[:] .+ N₋
    BR[idxdown, 1:N₋] = LinearAlgebra.kron(
        LinearAlgebra.diagm(
            model.C[model.C.<=0] ./ model.r.a(mesh.Nodes[1].+sqrt(eps()))[model.C.<=0],
        ),
        -Matrices.Local.Phi[1, :],
    )

    # Upper boundary
    # At boundary
    BR[(end-N₊+1):end, (end-N₊+1):end] =
        (1.0./model.r.a(mesh.Nodes[end])'.*model.T)[model.C.>=0, model.C.>=0]
    # Out of boundary
    idxdown =
        ((1:NBases(mesh)).+TotalNBases(mesh)*(findall(model.C .< 0) .- 1)')[:] .+
        (N₋ + TotalNBases(mesh) - NBases(mesh))
    BR[(end-N₊+1):end, idxdown] = kron(
        (1.0./model.r.a(mesh.Nodes[end])'.*model.T)[model.C.>=0, model.C.<0],
        Matrices.Local.Phi[end, :]' * Matrices.Local.MInv * 2 ./ Δ(mesh)[end],
    )
    # Into boundary
    idxup =
        ((1:NBases(mesh)).+TotalNBases(mesh)*(findall(model.C .>= 0) .- 1)')[:] .+
        (N₋ + TotalNBases(mesh) - NBases(mesh))
    BR[idxup, (end-N₊+1):end] = LinearAlgebra.kron(
        LinearAlgebra.diagm(
            model.C[model.C.>=0] ./ model.r.a(mesh.Nodes[end].-sqrt(eps()))[model.C.>=0],
        ),
        Matrices.Local.Phi[end, :],
    )

    idx0 = [mesh.Fil["p0"]; repeat(mesh.Fil["0"]', NBases(mesh), 1)[:]; mesh.Fil["q0"]]
    bullet = [
        (mesh.Fil["p+"] .| mesh.Fil["p-"])
        repeat((mesh.Fil["+"] .| mesh.Fil["-"])', NBases(mesh), 1)[:]
        (mesh.Fil["q+"] .| mesh.Fil["q-"])
    ]
    MR = [
        LinearAlgebra.I(N₋) SparseArrays.spzeros(Float64, N₋, BigN - N₋)
        SparseArrays.spzeros(Float64, BigN - N₊ - N₋, N₋) MR SparseArrays.spzeros(Float64, BigN - N₊ - N₋, N₊)
        SparseArrays.spzeros(Float64, N₊, BigN - N₊) LinearAlgebra.I(N₊)
    ]
    Minv = [
        LinearAlgebra.I(N₋) SparseArrays.spzeros(Float64, N₋, BigN - N₋)
        SparseArrays.spzeros(Float64, BigN - N₊ - N₋, N₋) Minv SparseArrays.spzeros(Float64, BigN - N₊ - N₋, N₊)
        SparseArrays.spzeros(Float64, N₊, BigN - N₊) LinearAlgebra.I(N₊)
    ]

    # BR[idx0, :] = B.B[idx0, :]
    DR = function (;s=0)
        BR[bullet, bullet] -
        MR[bullet, bullet] * s * SparseArrays.sparse(LinearAlgebra.I,sum(bullet),sum(bullet)) * Minv[bullet, bullet] +
        BR[bullet, idx0] *
        Matrix(s * SparseArrays.sparse(LinearAlgebra.I,sum(idx0),sum(idx0)) - B.B[idx0, idx0])^-1 *
        B.B[idx0, bullet]
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
        FlBases = mesh.Fil[ℓ][.!mesh.Fil["0"]]
        FmBases = mesh.Fil[m][.!mesh.Fil["0"]]
        FlBases = [
            mesh.Fil["p"*ℓ][.!mesh.Fil["p0"]]
            repeat(FlBases', NBases(mesh), 1)[:]
            mesh.Fil["q"*ℓ][.!mesh.Fil["q0"]]
        ]
        FmBases = [
            mesh.Fil["p"*m][.!mesh.Fil["p0"]]
            repeat(FmBases', NBases(mesh), 1)[:]
            mesh.Fil["q"*m][.!mesh.Fil["q0"]]
        ]
        DDict[ℓ*m] = function (; s = 0)#::Array{Float64}
            DR(s=s)[FlBases, FmBases]
        end # end function
    end # end for ℓ, m ...
    out = (DDict = DDict, DR = DR)
    println("UPDATE: D(s) operator created with keys ", keys(out))
    println("UPDATE:    DDict[key](s) operator created with keys ", keys(DDict))
    println("UPDATE:    DR(s) operator created (a matrix)")
    return out
end
