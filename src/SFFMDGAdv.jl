# function MakeMatrices2(;
#     model::SFFM.Model,
#     mesh::NamedTuple{
#         (
#             :NBases,
#             :CellNodes,
#             :Fil,
#             :Δ,
#             :NIntervals,
#             :Nodes,
#             :TotalNBases,
#             :Basis,
#         ),
#     },
# )
#     ## Construct local blocks
#     V = vandermonde(NBases = Mesh.NBases)
#     if Mesh.Basis == "legendre"
#         MLocal = Matrix{Float64}(LinearAlgebra.I(Mesh.NBases))
#         GLocal = V.inv * V.D
#         MInvLocal = Matrix{Float64}(LinearAlgebra.I(Mesh.NBases))
#         Phi = V.V[[1; end], :]
#     elseif Mesh.Basis == "lagrange"
#         MLocal = V.inv' * V.inv
#         GLocal = V.inv' * V.inv * (V.D * V.inv)
#         MInvLocal = V.V * V.V'
#         Phi = (V.inv*V.V)[[1; end], :]
#     end
#     Dw = (
#         DwInv = LinearAlgebra.I,
#         Dw = LinearAlgebra.I,
#     )
#
#     ## Assemble into global block diagonal matrices
#     G = SFFM.MakeBlockDiagonalMatrix(
#         Mesh = Mesh,
#         Blocks = GLocal,
#         Factors = ones(Mesh.NIntervals),
#     )
#     M = SFFM.MakeBlockDiagonalMatrix(Mesh = Mesh, Blocks = MLocal, Factors = Mesh.Δ * 0.5)
#     MInv = SFFM.MakeBlockDiagonalMatrix(
#         Mesh = Mesh,
#         Blocks = MInvLocal,
#         Factors = 2 ./ Mesh.Δ,
#     )
#     F = SFFM.MakeFluxMatrix(
#         Mesh = Mesh,
#         Phi = Phi,
#         Dw = (Dw=LinearAlgebra.I,DwInv=LinearAlgebra.I),
#         probTransform = false,
#     )
#
#     ## Assemble the DG drift operator
#     Q = Array{SparseArrays.SparseMatrixCSC{Float64,Int64},1}(undef,model.NPhases)
#     for i = 1:model.NPhases
#         if model.C[i] > 0
#             Q[i] = model.C[i] * (G + F["+"]) * MInv
#         elseif model.C[i] < 0
#             Q[i] = model.C[i] * (G + F["-"]) * MInv
#         end
#     end
#
#     Local = (G = GLocal, M = MLocal, MInv = MInvLocal, V = V, Phi = Phi, Dw = Dw)
#     Global = (G = G, M = M, MInv = MInv, F = F, Q = Q)
#     out = (Local = Local, Global = Global)
#     # println("UPDATE: Matrices object created with keys ", keys(out))
#     # println("UPDATE:    Matrices[:",keys(out)[1],"] object created with keys ", keys(out[1]))
#     # println("UPDATE:    Matrices[:",keys(out)[2],"] object created with keys ", keys(out[2]))
#     return out
# end
#
"""
Constructs a block diagonal matrix from blocks

    MakeBlockDiagonalMatrixR(;
        model::SFFM.Model,
        mesh::DGMesh,
        Blocks,
        Factors::Array,
    )

# Aguments
- `model`: A Model object
- `mesh`: A DGMesh object from `DGMesh()`
- `Blocks(x::Array{Float64}, i::Int)`: a function wich returns a
    `mesh.NBases×mesh.NBases` array to put along the diagonal. The input
    argument `x` is a column vector of corresponding to the nodes in each cell,
    i.e. `mesh.CellNodes[:,n]`. `i` denotes the phase.
- `Factors::Array{<:Real,1}`: a `mesh.NIntervals×1` vector of factors which multiply blocks

# Output
- `BlockMatrix::Array{Float64,2}`: `mesh.TotalNBases×mesh.TotalNBases` the
        block matrix
"""
function MakeBlockDiagonalMatrixR(;
    model::SFFM.Model,
    mesh::DGMesh,
    Blocks,
    Factors::Array,
)
    BlockMatrix = Array{SparseArrays.SparseMatrixCSC{Float64,Int64},1}(undef,model.NPhases)
    for i in 1:model.NPhases
        BlockMatrix[i] = SparseArrays.spzeros(Float64, mesh.TotalNBases, mesh.TotalNBases)
    end
    for i = 1:mesh.NIntervals, j = 1:model.NPhases
        idx = (1:mesh.NBases) .+ (i - 1) * mesh.NBases
        BlockMatrix[j][idx, idx] = Blocks(mesh.CellNodes[:, i], j) * Factors[i]
    end
    return (BlockMatrix = BlockMatrix)
end

"""
Constructs the flux matrices for DG

    MakeFluxMatrixR(;
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
- `F::Array{SparseArrays.SparseMatrixCSC{Float64,Int64},1}(undef,model.NPhases)`:
    an array with index `i ∈ 1:model.NPhases`, of sparse arrays which are
    `TotalNBases×TotalNBases` flux matrices for phase `i`.
"""
function MakeFluxMatrixR(;
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
    F = Array{SparseArrays.SparseMatrixCSC{Float64,Int64},1}(undef,model.NPhases)
    for i = 1:model.NPhases
        F[i] = SparseArrays.spzeros(Float64, mesh.TotalNBases, mesh.TotalNBases)
        for k = 1:mesh.NIntervals
            idx = (1:mesh.NBases) .+ (k - 1) * mesh.NBases
            if model.C[i] > 0
                xright = mesh.CellNodes[end, k]
                R = 1.0 ./ model.r.a(xright)[i]
                F[i][idx, idx] = PosDiagBlock * R
            elseif model.C[i] < 0
                xleft = mesh.CellNodes[1, k]
                R = 1.0 ./ model.r.a(xleft)[i]
                F[i][idx, idx] = NegDiagBlock * R
            end # end if C[i]
            if k > 1
                idxup = (1:mesh.NBases) .+ (k - 2) * mesh.NBases
                if model.C[i] > 0
                    xright = mesh.CellNodes[end, k-1]
                    R = 1.0 ./ model.r.a(xright)[i]
                    η = (mesh.Δ[k] / mesh.NBases) / (mesh.Δ[k-1] / mesh.NBases)
                    F[i][idxup, idx] = UpDiagBlock * η * R
                elseif model.C[i] < 0
                    xleft = mesh.CellNodes[1, k]
                    R = 1.0 ./ model.r.a(xleft)[i]
                    η = (mesh.Δ[k-1] / mesh.NBases) / (mesh.Δ[k] / mesh.NBases)
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

    MakeMatricesR(;
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
function MakeMatricesR(;
    model::SFFM.Model,
    mesh::DGMesh,
)
    ## Construct blocks
    V = vandermonde(NBases = mesh.NBases)
    if mesh.Basis == "legendre"
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

    elseif mesh.Basis == "lagrange"
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
        model = model,
        mesh = mesh,
        Blocks = GLocal,
        Factors = ones(mesh.NIntervals),
    )
    M = SFFM.MakeBlockDiagonalMatrixR(
        model = model,
        mesh = mesh,
        Blocks = MLocal,
        Factors = mesh.Δ * 0.5,
    )
    MInv = SFFM.MakeBlockDiagonalMatrixR(
        model = model,
        mesh = mesh,
        Blocks = MInvLocal,
        Factors = 2.0 ./ mesh.Δ,
    )

    F = SFFM.MakeFluxMatrixR(mesh = mesh, model = model, Phi = Phi)

    ## Assemble the DG drift operator
    Q = Array{SparseArrays.SparseMatrixCSC{Float64,Int64},1}(undef,model.NPhases)
    for i = 1:model.NPhases
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

    MakeDR(;
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
function MakeDR(;
    Matrices,
    MatricesR,
    model::SFFM.Model,
    mesh::DGMesh,
    B,
)
    N₊ = sum(model.C .>= 0)
    N₋ = sum(model.C .<= 0)

    BigN = model.NPhases * mesh.TotalNBases + N₊ + N₋
    MR = SparseArrays.spzeros(Float64, model.NPhases * mesh.TotalNBases, model.NPhases * mesh.TotalNBases)
    Minv =
        SparseArrays.spzeros(Float64, model.NPhases * mesh.TotalNBases, model.NPhases * mesh.TotalNBases)
    FGR = SparseArrays.spzeros(Float64, model.NPhases * mesh.TotalNBases, model.NPhases * mesh.TotalNBases)
    for i = 1:model.NPhases
        idx = ((i-1)*mesh.TotalNBases+1:i*mesh.TotalNBases)
        MR[idx, idx] = MatricesR.Global.M[i]
        Minv[idx, idx] = Matrices.Global.MInv
        FGR[idx, idx] = MatricesR.Global.Q[i] * Matrices.Global.MInv
    end

    # Interior behaviour
    T = SparseArrays.kron(model.T, SparseArrays.sparse(LinearAlgebra.I,mesh.TotalNBases,mesh.TotalNBases))
    BR = SparseArrays.spzeros(Float64, BigN, BigN)
    BR[(N₋+1):(end-N₊), (N₋+1):(end-N₊)] = MR * T * Minv + FGR

    # Boundary behaviour
    # Lower boundary
    # At boundary
    BR[1:N₋, 1:N₋] = (1.0./ model.r.a(mesh.Nodes[1])'.*model.T)[model.C.<=0, model.C.<=0]
    # Out of boundary
    idxup = ((1:mesh.NBases).+mesh.TotalNBases*(findall(model.C .> 0) .- 1)')[:] .+ N₋
    BR[1:N₋, idxup] = kron(
        (1.0./model.r.a(mesh.Nodes[1])'.*model.T)[model.C.<=0, model.C.>0],
        Matrices.Local.Phi[1, :]' * Matrices.Local.MInv * 2 ./ mesh.Δ[1],
    )
    # Into boundary
    idxdown = ((1:mesh.NBases).+mesh.TotalNBases*(findall(model.C .<= 0) .- 1)')[:] .+ N₋
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
        ((1:mesh.NBases).+mesh.TotalNBases*(findall(model.C .< 0) .- 1)')[:] .+
        (N₋ + mesh.TotalNBases - mesh.NBases)
    BR[(end-N₊+1):end, idxdown] = kron(
        (1.0./model.r.a(mesh.Nodes[end])'.*model.T)[model.C.>=0, model.C.<0],
        Matrices.Local.Phi[end, :]' * Matrices.Local.MInv * 2 ./ mesh.Δ[end],
    )
    # Into boundary
    idxup =
        ((1:mesh.NBases).+mesh.TotalNBases*(findall(model.C .>= 0) .- 1)')[:] .+
        (N₋ + mesh.TotalNBases - mesh.NBases)
    BR[idxup, (end-N₊+1):end] = LinearAlgebra.kron(
        LinearAlgebra.diagm(
            model.C[model.C.>=0] ./ model.r.a(mesh.Nodes[end].-sqrt(eps()))[model.C.>=0],
        ),
        Matrices.Local.Phi[end, :],
    )

    idx0 = [mesh.Fil["p0"]; repeat(mesh.Fil["0"]', mesh.NBases, 1)[:]; mesh.Fil["q0"]]
    bullet = [
        (mesh.Fil["p+"] .| mesh.Fil["p-"])
        repeat((mesh.Fil["+"] .| mesh.Fil["-"])', mesh.NBases, 1)[:]
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
            repeat(FlBases', mesh.NBases, 1)[:]
            mesh.Fil["q"*ℓ][.!mesh.Fil["q0"]]
        ]
        FmBases = [
            mesh.Fil["p"*m][.!mesh.Fil["p0"]]
            repeat(FmBases', mesh.NBases, 1)[:]
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
