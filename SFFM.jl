module SFFM
import Jacobi, Plots, LinearAlgebra
using Plots: Animation, frame, gif

# Fil = Dict{String,BitArray{1}}("1+" => Bool[1, 1, 0, 0, 0],
#                                "2+" => Bool[0, 0, 1, 1, 1],
#                                "2-" => Bool[1, 1, 0, 0, 0],
#                                "1-" => Bool[0, 0, 1, 1, 1])

function MyPrint(Obj)
  show(stdout, "text/plain", Obj)
end

function MakeModel(;
  T::Array{Float64},
  C::Array{Float64,1},
  r,
  Signs::Array{String,1} = ["+"; "-"; "0"],
  Bounds::Array{<:Number,2} = [-Inf Inf; -Inf Inf],
)
  # Make a 'Model' object which carries all the info we need to
  # know about the SFFM.
  # T - n×n Array{Float64}, a generator matrix of φ(t)
  # C - n×1 Array{Float64}, rates of the first fluid
  # Signs - n×1 Array{String}, the m∈{"+","-","0"} where Fᵢᵐ≂̸∅
  # IsBounded - Bool, whether the first fluid is bounded or not
  # r - array of rates for the second fluid,
  #     functions r(x) = [r₁(x) r₂(x) ... r_n(x)], where x is a column vector
  #
  # output is a NamedTuple with fields
  #                         .T, .C, .r, .Signs, .IsBounded, .NPhases, .NSigns

  NPhases = length(C)
  NSigns = length(Signs)
  println("Model.Field with Fields (.T, .C, .r, .Signs, .IsBounded, .NPhases,
            .NSigns)")
  IsBounded = true
  return (
    T = T,
    C = C,
    r = r,
    Signs = Signs,
    IsBounded = IsBounded,
    Bounds = Bounds,
    NPhases = NPhases,
    NSigns = NSigns,
  )
end

function MakeMesh(;
  Model,
  Nodes::Array{Float64,1},
  NBases::Int,
  Fil::Dict{String,BitArray{1}},
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
    CellNodes[:, i] .=
      (Nodes[i+1] + Nodes[i]) / 2 .+ (Nodes[i+1] - Nodes[i]) / 2 * z
  end
  MeshArray = zeros(NIntervals, 2)
  MeshArray[:, 1] = Nodes[1:end-1] # Left-hand end points of each interval
  MeshArray[:, 2] = Nodes[2:end] # Right-hand edges
  TotalNBases = NBases * NIntervals # the total number of bases in the stencil

  ## Construct the sets Fᵐ = ⋃ᵢ Fᵢᵐ, global index for sets of type m
  for ℓ in Model.Signs
    Fil[string(ℓ)] = falses(NIntervals * Model.NPhases)
    for i = 1:Model.NPhases
      idx = findall(Fil[string(i, ℓ)]) .+ (i - 1) * NIntervals
      Fil[string(ℓ)][idx] .= true
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
  elseif NBases == 1
    V .= 1
    DV .= 0
  end
  # Compute the Gauss-Lobatto weights for numerical quadrature
  w =
    2.0 ./ (
      NBases *
      (NBases - 1) *
      Jacobi.legendre.(Jacobi.zglj(NBases, 0, 0), NBases - 1) .^ 2
    )
  return (V = V, inv = inv(V), D = DV, w = w)
end

function MakeBlockDiagonalMatrix(;
  Mesh,
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

function MakeBlockDiagonalMatrixR(; Model, Mesh, Blocks, Factors::Array)
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

  BlockMatrix =
    zeros(Float64, Mesh.TotalNBases, Mesh.TotalNBases, Model.NPhases)
  for i = 1:Mesh.NIntervals, j = 1:Model.NPhases
    idx = (1:Mesh.NBases) .+ (i - 1) * Mesh.NBases
    BlockMatrix[idx, idx, j] = Blocks(Mesh.CellNodes[:, i], j) * Factors[i]
  end
  return (BlockMatrix = BlockMatrix)
end

function PlotVt(; Nodes, Vt, C, YMAX = 1, PointMass = true, labels = [])
  if isempty(labels)
    labels = string.(C)
  end
  Interval = [Nodes[1]; Nodes[end]]
  NNodes = length(Nodes)
  ΔEnds = [Nodes[2] - Nodes[1]; Nodes[end] - Nodes[end-1]]
  FivePercent = (Interval[2] - Interval[1]) * 0.025 .* [-1; 1]
  Plt = Plots.plot(xlims = Interval + FivePercent, ylims = (0, YMAX))
  for i = 1:length(C)
    if PointMass
      if C[i] > 0
        CtsIdx = [1; 1:NNodes-1]
        xIdx = [1; 2; 2:NNodes-1]
        PMIdx = NNodes
        ΔIdx = 2
      elseif C[i] < 0
        CtsIdx = [2:NNodes; NNodes]
        xIdx = [2:NNodes-1; NNodes - 1; NNodes]
        PMIdx = 1
        ΔIdx = 1
      else
        CtsIdx = 2:NNodes-1
        xIdx = CtsIdx
        PMIdx = [1; NNodes]
        ΔIdx = [1; 2]
      end # end if C[i]
      Plt = Plots.plot!(
        Nodes[xIdx],
        Vt[CtsIdx, i],
        label = string("Phase ", i, " rate: ", C[i], " and ", labels[i]),
        color = i,
      )
      Plt = Plots.scatter!(
        [Nodes[PMIdx]],
        [Vt[PMIdx, i] .* ΔEnds[ΔIdx]],
        marker = :hexagon,
        label = string("mass"),
        color = i,
      )
    else
      Plt = Plots.plot!(
        Nodes,
        Vt[:, i],
        label = string("Phase ", i, " rate: ", labels[i]),
        color = i,
      )
    end # end if PointMass
  end # end for i
  return (Plt = Plt)
end # end PlotVt

function MakeFluxMatrix(; Mesh, Model, Phi)
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
        F[idx, idx, i] = PosDiagBlock
      elseif Model.C[i] < 0
        F[idx, idx, i] = NegDiagBlock
      end # end if C[i]
      if k > 1
        idxup = (1:Mesh.NBases) .+ (k - 2) * Mesh.NBases
        if Model.C[i] > 0
          η = (Mesh.Δ[k] / Mesh.NBases) / (Mesh.Δ[k-1] / Mesh.NBases)
          F[idxup, idx, i] = UpDiagBlock * η
        elseif Model.C[i] < 0
          η = (Mesh.Δ[k-1] / Mesh.NBases) / (Mesh.Δ[k] / Mesh.NBases)
          F[idx, idxup, i] = LowDiagBlock * η
        end # end if C[i]
      end # end if k>1
    end # for k in ...
  end # end for i in NPhases

  ## Check if bounded and make sure no mass can leave
  if Model.IsBounded
    for i = 1:Model.NPhases
      if Model.C[i] < 0
        idx = 1:Mesh.NBases
        F[idx, idx, i] .= 0
      elseif Model.C[i] > 0
        idx = (1:Mesh.NBases) .+ (Mesh.NIntervals - 1) * Mesh.NBases
        F[idx, idx, i] .= 0
      end # end if C[i]
    end # end for i ...
  end # end if IsBounded

  return (F = F)
end

function MakeFluxMatrixR(; Mesh, Model, Phi)
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

  ## Check if bounded and make sure no mass can leave
  if Model.IsBounded
    for i = 1:Model.NPhases
      if Model.C[i] < 0
        idx = 1:Mesh.NBases
        F[idx, idx, i] .= 0
      elseif Model.C[i] > 0
        idx = (1:Mesh.NBases) .+ (Mesh.NIntervals - 1) * Mesh.NBases
        F[idx, idx, i] .= 0
      end # end if C[i]
    end # end for i ...
  end # end if IsBounded

  return (F = F)
end

function MakeMatrices(; Model, Mesh, Basis::String = "legendre")
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
    MLocal = Matrix{Float64}(LinearAlgebra.I(Mesh.NBases))
    GLocal = V.inv * V.D
    MInvLocal = Matrix{Float64}(LinearAlgebra.I(Mesh.NBases))
    Phi = V.V[[1; end], :]
  elseif Basis == "lagrange"
    MLocal = V.inv' * V.inv
    GLocal = MLocal * (V.D * V.inv)
    MInvLocal = V.V * V.V'
    Phi = (V.inv*V.V)[[1; end], :]
  end

  ## Assemble into block diagonal matrices
  G = SFFM.MakeBlockDiagonalMatrix(
    Mesh = Mesh,
    Blocks = GLocal,
    Factors = ones(Mesh.NIntervals),
  )
  M = SFFM.MakeBlockDiagonalMatrix(
    Mesh = Mesh,
    Blocks = MLocal,
    Factors = Mesh.Δ * 0.5,
  )
  MInv = SFFM.MakeBlockDiagonalMatrix(
    Mesh = Mesh,
    Blocks = MInvLocal,
    Factors = 2.0 ./ Mesh.Δ,
  )

  F = SFFM.MakeFluxMatrix(Mesh = Mesh, Model = Model, Phi = Phi)

  ## Assemble the DG drift operator
  Q = zeros(Float64, Mesh.TotalNBases, Mesh.TotalNBases, length(Model.C))
  for i = 1:Model.NPhases
    Q[:, :, i] = Model.C[i] * (G + F[:, :, i]) * MInv
  end

  Local = (G = GLocal, M = MLocal, MInv = MInvLocal, V = V)
  Global = (G = G, M = M, MInv = MInv, F = F, Q = Q)
  println("Matrices.Fields with Fields (.Local, .Global)")
  println("Matrices.Local.Fields with Fields (.G, .M, .MInv, .V)")
  println("Matrices.Global.Fields with Fields (.G, .M, .MInv, F, .Q)")
  return (Local = Local, Global = Global)
end

function MakeMatricesR(; Model, Mesh, Basis::String = "legendre")
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

  Local = (G = GLocal, M = MLocal, MInv = MInvLocal, V = V)
  Global = (G = G, M = M, MInv = MInv, F = F, Q = Q)
  println("Matrices.Fields with Fields (.Local, .Global)")
  println("Matrices.Local.Fields with Fields (.G, .M, .MInv, .V)")
  println("Matrices.Global.Fields with Fields (.G, .M, .MInv, F, .Q)")
  return (Local = Local, Global = Global)
end

function MakeB(; Model, Mesh, Matrices)
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
  B = zeros(
    Float64,
    Model.NPhases * Mesh.TotalNBases,
    Model.NPhases * Mesh.TotalNBases,
  )
  Id = Matrix(LinearAlgebra.I, Mesh.TotalNBases, Mesh.TotalNBases)
  for i = 1:Model.NPhases
    idx = (i-1)*Mesh.TotalNBases+1:i*Mesh.TotalNBases
    B[idx, idx] = Matrices.Global.Q[:, :, i]
  end

  B = B + LinearAlgebra.kron(Model.T, Id)

  ## Make a Dictionary so that the blocks of B are easy to access
  BDict = Dict{String,Array{Float64,2}}()
  for ℓ in Model.Signs, m in Model.Signs
    for i = 1:Model.NPhases, j = 1:Model.NPhases
      FilBases = repeat(Mesh.Fil[string(i, ℓ)]', Mesh.NBases, 1)[:]
      i_idx = [
        falses((i - 1) * Mesh.TotalNBases)
        FilBases
        falses(Model.NPhases * Mesh.TotalNBases - i * Mesh.TotalNBases)
      ]
      FjmBases = repeat(Mesh.Fil[string(j, m)]', Mesh.NBases, 1)[:]
      j_idx = [
        falses((j - 1) * Mesh.TotalNBases)
        FjmBases
        falses(Model.NPhases * Mesh.TotalNBases - j * Mesh.TotalNBases)
      ]
      BDict[string(i, j, ℓ, m)] = B[i_idx, j_idx]
    end
    FlBases = repeat(Mesh.Fil[string(ℓ)]', Mesh.NBases, 1)[:]
    FmBases = repeat(Mesh.Fil[string(m)]', Mesh.NBases, 1)[:]
    BDict[string(ℓ, m)] = B[FlBases, FmBases]
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

function SFFMGIF(;
  a0,
  Nodes,
  B,
  Times,
  C,
  PointMass = true,
  YMAX = 1,
  labels = [],
)
  gifplt = Plots.@gif for n = 1:length(Times)
    Vt = a0 * exp(B * Times[n])
    Vt = reshape(Vt, (length(Nodes), length(C)))
    SFFM.PlotVt(
      Nodes = Nodes',
      Vt = Vt,
      C = C,
      PointMass = PointMass,
      YMAX = YMAX,
      labels = labels,
    )
  end
  return (gitplt = gifplt)
end

function MakeR(; Model, Mesh)
  # interpolant approximation to r(x)
  EvalPoints = Mesh.CellNodes
  EvalPoints[1, :] .+= sqrt(eps()) # LH edges + eps
  EvalPoints[end, :] .+= -sqrt(eps()) # RH edges - eps
  EvalR = abs.(Model.r.r(EvalPoints[:]))
  RDict = Dict{String,Array{Float64,1}}()
  for i = 1:Model.NPhases
    RDict[string(i)] = 1.0 ./ EvalR[:, i]
  end

  R = zeros(Float64, Model.NPhases * Mesh.TotalNBases)
  for i = 1:Model.NPhases
    Ri = RDict[string(i)]
    R[(i-1)*Mesh.TotalNBases+1:i*Mesh.TotalNBases] = Ri
    for ℓ in Model.Signs
      FilBases = repeat(Mesh.Fil[string(i, ℓ)]', Mesh.NBases, 1)[:]
      RDict[string(i, ℓ)] = Ri[FilBases]
    end
  end
  for ℓ in Model.Signs
    FlBases = repeat(Mesh.Fil[string(ℓ)]', Mesh.NBases, 1)[:]
    RDict[ℓ] = R[FlBases]
  end
  println("R.Fields with Fields (.RDict, .R)")
  return (RDict = RDict, R = R)
end

function MakeD(; R, B, Model, Mesh)
  RDict = R.RDict
  BDict = B.BDict
  DDict = Dict{String,Any}()
  for ℓ in ["+", "-"], m in ["+", "-"]
    Idℓ = LinearAlgebra.I(sum(Mesh.Fil[ℓ]) * Mesh.NBases)
    if in("0", Model.Signs)
      Id0 = LinearAlgebra.I(sum(Mesh.Fil["0"]) * Mesh.NBases)
      DDict[ℓ*m] = function (; s = 0)#::Array{Float64}
        return if (ℓ == m)
          RDict[ℓ] .* (
            BDict[ℓ*m] - s * Idℓ +
            BDict[ℓ*"0"] * inv(s * Id0 - BDict["00"]) * BDict["0"*m]
          )
        else
          RDict[ℓ] .* (
            BDict[ℓ*m] +
            BDict[ℓ*"0"] * inv(s * Id0 - BDict["00"]) * BDict["0"*m]
          )
        end
      end # end function
    else
      DDict[ℓ*m] = function (; s = 0)#::Array{Float64}
        return if (ℓ == m)
          RDict[ℓ] .* (BDict[ℓ*m] - s * Idℓ)
        else
          RDict[ℓ] .* BDict[ℓ*m]
        end
      end # end function
    end # end if ...
  end # end for ℓ ...
  return (DDict = DDict)
end

function MakeDR(; Matrices, MatricesR, Model, Mesh, R, B)
  MR = zeros(
    Float64,
    Model.NPhases * Mesh.TotalNBases,
    Model.NPhases * Mesh.TotalNBases,
  )
  Minv = zeros(
    Float64,
    Model.NPhases * Mesh.TotalNBases,
    Model.NPhases * Mesh.TotalNBases,
  )
  FGR = zeros(
    Float64,
    Model.NPhases * Mesh.TotalNBases,
    Model.NPhases * Mesh.TotalNBases,
  )
  for i = 1:Model.NPhases
    idx = (i-1)*Mesh.TotalNBases+1:i*Mesh.TotalNBases
    MR[idx, idx] = MatricesR.Global.M[:, :, i]
    Minv[idx, idx] = Matrices.Global.M^-1
    FGR[idx, idx] =
      (MatricesR.Global.F[:, :, i] + MatricesR.Global.G[:, :, i]) *
      Model.C[i] *
      Minv[idx, idx]
  end
  bullet = repeat(Mesh.Fil["+"] .| Mesh.Fil["-"], Mesh.NBases, 1)[:]
  idx0 = repeat(Mesh.Fil["0"], Mesh.NBases, 1)[:]
  T = kron(Model.T, LinearAlgebra.I(Mesh.TotalNBases))
  D(s) =
    MR[bullet, bullet] *
    (T[bullet, bullet] - s * LinearAlgebra.I(sum(bullet))) *
    Minv[bullet, bullet] +
    FGR[bullet, bullet] +
    MR[bullet, bullet] *
    T[bullet, idx0] *
    (B.BDict["00"] - s * LinearAlgebra.I(sum(idx0)))^-1 *
    T[idx0, bullet]

  DDict = Dict{String,Any}()
  for ℓ in ["+", "-"], m in ["+", "-"]
    FlBases = Mesh.Fil[string(ℓ)][.!Mesh.Fil["0"]]
    FmBases = Mesh.Fil[string(m)][.!Mesh.Fil["0"]]
    FlBases = repeat(FlBases', Mesh.NBases, 1)[:]
    FmBases = repeat(FmBases', Mesh.NBases, 1)[:]
    DDict[ℓ*m] = function (; s = 0)#::Array{Float64}
      D(s)[FlBases, FmBases]
    end # end function
  end # end for ℓ, m ...
  return (D = D, DDict = DDict)
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

function SimSFM(; Model, StoppingTime::Function, InitCondition)
  # Simulates a SFM defined by Model until the StoppingTime has occured,
  # given the InitialConditions on (φ(0),X(0))
  # Model - A Model object as output from MakeModel
  # StoppingTime - A function which takes four arguments e.g.
  #   StoppingTime(;t,φ,X,n) where
  #   t is the current time, φ the current phase, X, the current level, n
  #   the number of transition to time t, and returns a tuple (Ind,SFM),
  #   where Ind is a boolen specifying whether the stopping time, τ, has
  #   occured or not, and SFM is a tuple (τ,φ(τ),X(τ),n).
  # InitCondition - M×2 Array with rows [φ(0) X(0)], M is the number of sims
  #   to be performed, e.g.~to simulate ten SFMs starting from
  #   [φ(0)=1 X(0)=0] then we set InitCondition = repeat([1 0], 10,1).

  # the transition matrix of the jump chain
  d = LinearAlgebra.diag(Model.T)
  P = (Model.T - LinearAlgebra.diagm(0 => d)) ./ -d
  CumP = cumsum(P, dims = 2)
  Λ = LinearAlgebra.diag(Model.T)

  M = size(InitCondition, 1)
  tSims = Array{Float64,1}(undef, M)
  φSims = Array{Float64,1}(undef, M)
  XSims = Array{Float64,1}(undef, M)
  nSims = Array{Float64,1}(undef, M)

  for m = 1:M
    SFM0 = (t = 0.0, φ = InitCondition[m, 1], X = InitCondition[m, 2], n = 0)
    while 1 == 1
      S = log(rand()) / Λ[SFM0.φ] # generate exp random variable
      t = SFM0.t + S
      X = UpdateXt(Model = Model, SFM0 = SFM0, S = S)
      φ = findfirst(rand() .< CumP[SFM0.φ, :])
      n = SFM0.n + 1
      SFM = (t = t, φ = φ, X = X, n = n)
      τ = StoppingTime(Model, SFM, SFM0)
      if τ.Ind
        (tSims[m], φSims[m], XSims[m], nSims[m]) = τ.SFM
        break
      end
      SFM0 = SFM
    end
  end
  return (t = tSims, φ = φSims, X = XSims, n = nSims)
end

function SimSFFM(; Model, StoppingTime::Function, InitCondition)
  d = LinearAlgebra.diag(Model.T)
  P = (Model.T - LinearAlgebra.diagm(0 => d)) ./ -d
  CumP = cumsum(P, dims = 2)
  Λ = LinearAlgebra.diag(Model.T)

  M = size(InitCondition, 1)
  tSims = Array{Float64,1}(undef, M)
  φSims = Array{Float64,1}(undef, M)
  XSims = Array{Float64,1}(undef, M)
  YSims = Array{Float64,1}(undef, M)
  nSims = Array{Float64,1}(undef, M)

  for m = 1:M
    SFFM0 = (
      t = 0.0,
      φ = InitCondition[m, 1],
      X = InitCondition[m, 2],
      Y = InitCondition[m, 3],
      n = 0,
    )
    while 1 == 1
      S = log(rand()) / Λ[SFFM0.φ]
      t = SFFM0.t + S
      X = UpdateXt(Model = Model, SFM0 = SFFM0, S = S)
      Y = UpdateYt(Model = Model, SFFM0 = SFFM0, S = S)
      φ = findfirst(rand() .< CumP[SFFM0.φ, :])
      n = SFFM0.n + 1
      SFFM = (t = t, φ = φ, X = X, Y = Y, n = n)
      τ = StoppingTime(Model, SFFM, SFFM0)
      if τ.Ind
        (tSims[m], φSims[m], XSims[m], YSims[m], nSims[m]) = τ.SFFM
        break
      end
      SFFM0 = SFFM
    end
  end
  return (t = tSims, φ = φSims, X = XSims, Y = YSims, n = nSims)
end

function UpdateXt(; Model, SFM0::NamedTuple, S::Real)
  # given the last position of a SFM, SFM0, a time step of size s, find the
  # position of X at time t
  X = min(
    max(SFM0.X + Model.C[SFM0.φ] * S, Model.Bounds[1, 1]),
    Model.Bounds[1, 2],
  )
  return X
end

function UpdateYt(; Model, SFFM0::NamedTuple, S::Real)
  # given the last position of a SFFM, SFFM0, a time step of size s, find the
  # position of Y at time t
  if Model.C[SFFM0.φ] == 0
    Y = SFFM0.Y + S * Model.r.r(SFFM0.X)[SFFM0.φ]
  else
    ind = (SFFM0.X.==Model.Bounds[1, :])[:]
    X = UpdateXt(Model = Model, SFM0 = SFFM0, S = S)
    if any(ind)
      # time at which Xt hits u or v
      t0 = (Model.Bounds[1, ind][1] - SFFM0.X) / Model.C[SFFM0.φ]
      Y =
        SFFM0.Y +
        (Model.r.R(X)[SFFM0.φ] - Model.r.R(SFFM0.X)[SFFM0.φ]) /
        Model.C[SFFM0.φ] +
        Model.r.r(Model.Bounds[1, ind][1])[SFFM0.φ] * (S - t0)
    else
      Y =
        SFFM0.Y +
        (Model.r.R(X)[SFFM0.φ] - Model.r.R(SFFM0.X)[SFFM0.φ]) / Model.C[SFFM0.φ]
    end
  end
  return Y
end

function FixedTime(; T::Real)
  # Defines a simple stopping time, 1(t>T).
  # SFM method
  function FixedTimeFun(
    Model,
    SFM::NamedTuple{(:t, :φ, :X, :n)},
    SFM0::NamedTuple{(:t, :φ, :X, :n)},
  )
    Ind = SFM.t > T
    if Ind
      s = T - SFM0.t
      X = UpdateXt(Model = Model, SFM0 = SFM0, S = s)
      SFM = (t = T, φ = SFM0.φ, X = X, n = SFM0.n)
    end
    return (Ind = Ind, SFM = SFM)
  end
  # SFFM METHOD
  function FixedTimeFun(
    Model,
    SFFM::NamedTuple{(:t, :φ, :X, :Y, :n)},
    SFFM0::NamedTuple{(:t, :φ, :X, :Y, :n)},
  )
    Ind = SFFM.t > T
    if Ind
      s = T - SFFM0.t
      X = UpdateXt(Model = Model, SFM0 = SFFM0, S = s)
      Y = UpdateYt(Model = Model, SFFM0 = SFFM0, S = s)
      SFFM = (T, SFFM0.φ, X, Y, SFFM0.n)
    end
    return (Ind = Ind, SFFM = SFFM)
  end
  return FixedTimeFun
end

function NJumps(; N::Int)
  # Defines a simple stopping time, 1(n>N), where n is the number of jumps of φ.
  # SFM method
  function NJumpsFun(
    Model,
    SFM::NamedTuple{(:t, :φ, :X, :n)},
    SFM0::NamedTuple{(:t, :φ, :X, :n)},
  )
    Ind = SFM.n >= N
    return (Ind = Ind, SFM = SFM)
  end
  # SFFM method
  function NJumpsFun(
    Model,
    SFFM::NamedTuple{(:t, :φ, :X, :Y, :n)},
    SFFM0::NamedTuple{(:t, :φ, :X, :Y, :n)},
  )
    Ind = SFFM.n >= N
    return (Ind = Ind, SFFM = SFFM)
  end
  return NJumpsFun
end

function FirstExitX(; u::Real, v::Real)
  # Defines a first exit stopping time rule for the interval [u,v]
  # Inputs:
  #   u,v - scalars
  # Outputs:
  #   FirstExitFun(Model,t::Float64,φ,X,n::Int), a function with inputs;
  #   t is the current time, φ the current phase, X, the current level, n
  #   the number of transition to time t, and returns a tuple (Ind,SFM),
  #   where Ind is a boolen specifying whether the stopping time, τ, has
  #   occured or not, and SFM is a tuple (τ,φ(τ),X(τ),n).

  # SFM Method
  function FirstExitXFun(
    Model,
    SFM::NamedTuple{(:t, :φ, :X, :n)},
    SFM0::NamedTuple{(:t, :φ, :X, :n)},
  )
    Ind = SFM.X > v || SFM.X < u
    if Ind
      if SFM.X > v
        X = v
      else
        X = u
      end
      s = (X - SFM0.X) / Model.C[SFM0.φ] # can't exit with c = 0
      t = SFM0.t + s
      SFM = (t = t, φ = SFM0.φ, X = X, n = SFM0.n)
    end
    return (Ind = Ind, SFM = SFM)
  end
  # SFFM Method
  function FirstExitXFun(
    Model,
    SFFM::NamedTuple{(:t, :φ, :X, :Y, :n)},
    SFFM0::NamedTuple{(:t, :φ, :X, :Y, :n)},
  )
    Ind = SFFM.X > v || SFFM.X < u
    if Ind
      if SFFM.X > v
        X = v
      else
        X = u
      end
      s = (X - SFFM0.X) / Model.C[SFFM0.φ] # can't exit with c = 0.
      t = SFFM0.t + s
      Y = UpdateYt(Model = Model, SFFM0 = SFFM0, S = s)
      SFFM = (t, SFFM0.φ, X, Y, SFFM0.n)
    end
    return (Ind = Ind, SFFM = SFFM)
  end
  return FirstExitXFun
end

function InOutYLevel(; y::Real)
  # Defines a first exit stopping time rule for the Y-in-out fluid hitting y
  # Inputs:
  #   y - scalars
  # Outputs:
  #   FirstExitFun(Model,t::Float64,φ,X,n::Int), a function with inputs;
  #   t is the current time, φ the current phase, X, the current level, n
  #   the number of transition to time t, and returns a tuple (Ind,SFM),
  #   where Ind is a boolen specifying whether the stopping time, τ, has
  #   occured or not, and SFM is a tuple (τ,φ(τ),X(τ),n).

  # SFFM Method
  function InOutYLevelFun(
    Model,
    SFFM::NamedTuple{(:t, :φ, :X, :Y, :n)},
    SFFM0::NamedTuple{(:t, :φ, :X, :Y, :n)},
  )
    Ind = SFFM.Y > y && SFFM0.Y <= y
    if Ind
      YFun(t) = UpdateYt(Model = Model, SFFM0 = SFFM0, S = t) - y
      S = SFFM.t - SFFM0.t
      tstar = fzero(f = YFun, a = 0, b = S)
      X = UpdateXt(Model = Model, SFM0 = SFFM0, S = tstar)
      t = SFFM0.t + tstar
      Y = y
      SFFM = (t, SFFM0.φ, X, Y, SFFM0.n)
    end
    return (Ind = Ind, SFFM = SFFM)
  end
  return InOutYLevelFun
end

function fzero(; f::Function, a::Real, b::Real, err::Float64 = 1e-14)
  # finds zeros of f using the bisection method
  c = a + (b - a) / 2
  while a < c < b
    fc = f(c)
    if abs(fc) < err
      break
    end
    if f(a) * fc < 0
      a, b = a, c
    else
      a, b = c, b
    end
    c = a + (b - a) / 2
  end
  return c
end

end # end module
