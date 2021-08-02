module SFFM
import Base.*, Base.size
import Jacobi, LinearAlgebra, SparseArrays
import Plots, StatsBase, KernelDensity

"""
Construct a SFFM model object.

    Model(
        T::Array{Float64,2},
        C::Array{Float64,1},
        r::NamedTuple{(:r, :R)};
        Bounds::Array{<:Real,2} = [-Inf Inf; -Inf Inf],
    )

# Arguments
- `T::Array{Float64,2}`: generator matrix for the CTMC ``φ(t)``
- `C::Array{Float64,1}`: vector of rates ``d/dt X(t)=C[i]`` for ``i=φ(t)``.
- `r::NamedTuple{(:r, :R)}`: rates for the second fluid.
    - `:r(x::Array{Real})`, a function  which takes arrays of x-values and
        returns a row vector of values for each x-value. i.e. `:r([0;1])`
        returns a `2×NPhases` array where the first row contains all the
        ``rᵢ(0)`` and row 2 all the ``rᵢ(1)`` values.
    - `:R(x::Array{Real})`: has the same structure/behaviour as ``:r`` but
        returns the integral of ``:r``. i.e. `Rᵢ(x)=∫ˣrᵢ(y)dy`.
- `Bounds::Array{<:Real,2}`: contains the bounds for the model. The first row
    are the L and R bounds for ``X(t)`` and the second row the bounds for
    ``Y(t)`` (although the bounds for ``Y(t)`` don't actually do anything yet).

# Outputs
- a model object which is a tuple with fields
    - `:T`: as input
    - `:C`: as input
    - `:r`: a named tuple with fields `(:r, :R, :a)`, `:r` and `:R` are as input
        and `:a = abs.(:r)` returns the absolute values of the rates.
    - `Bounds`: as input
)
"""
struct Model 
    T::Array{<:Real}
    C::Array{<:Real,1}
    r::NamedTuple{(:r, :R, :a)}
    Bounds::Array{<:Real}
end 
# Convenience constructors
function Model(
    T::Array{<:Real},
    C::Array{<:Real,1},
    r::NamedTuple{(:r, :R)};
    Bounds::Array{<:Real,2} = [-Inf Inf; -Inf Inf],
    v::Bool = false,
)
    a(x) = abs.(r.r(x))
    r = (r = r.r, R = r.R, a = a)

    v && println("UPDATE: Model object created with fields ", fieldnames(SFFM.Model))
    return Model(
        T,
        C,
        r,
        Bounds,
    )
end
function Model()
    Model(
        [0],
        [0],
        (r=0, R=0, a=0),
        [0],
    )
end

"""

    NPhases(model::Model)

the number of states in the state space
"""
NPhases(model::Model) = length(model.C)

"""

    modelDicts(model::Model) 

input: a Model object

outputs:
     - SDict: a dictionary with keys `"+","-","0","bullet"`
    and corresponding values `findall(model.C .> 0)`, `findall(model.C .< 0)`,
    `findall(model.C .== 0)`, `findall(model.C .!= 0)`, respectively.

     - TDict: a dictionary of submatrices of `T` with keys
    `"ℓm"` with ``ℓ,m∈{+,-,0,bullet}`` and corresponding values
    `model.T[S[ℓ],S[m]]`.
"""
function modelDicts(model::Model) 
    nPhases = NPhases(model)
    SDict = Dict{String,Array}("S" => 1:nPhases)
    SDict["+"] = findall(model.C .> 0)
    SDict["-"] = findall(model.C .< 0)
    SDict["0"] = findall(model.C .== 0)
    SDict["bullet"] = findall(model.C .!= 0)

    TDict = Dict{String,Array}("T" => model.T)
    for ℓ in ["+" "-" "0" "bullet"], m in ["+" "-" "0" "bullet"]
        TDict[ℓ*m] = model.T[SDict[ℓ], SDict[m]]
    end

    return SDict, TDict
end

"""

    TDict(model::Model) 


"""
function TDict(model::Model) 
    
end


"""

    Mesh 

Abstract type representing a mesh for a numerical scheme. 
"""
abstract type Mesh end 

include("METools.jl")

function MakeB(
    model::SFFM.Model,
    mesh::Mesh;
    probTransform::Bool=true,
    v::Bool = false,
)
    throw(DomainError("Unknown mesh type"))
end
function MakeB(model::SFFM.Model, mesh::SFFM.Mesh, order::Int)
    throw(DomainError("Unknown mesh type"))
end
function MakeB(model::Model, mesh::Mesh, me::ME)
    throw(DomainError("Unknown mesh type"))
end

function MakeDict(
    B::Union{Array{<:Real,2},SparseArrays.SparseMatrixCSC{<:Real,Int64}},
    model::Model, 
    mesh::Mesh;
    zero::Bool=true,
    )

    ## Make a Dictionary so that the blocks of B are easy to access
    N₋ = sum(model.C.<=0)
    N₊ = sum(model.C.<=0)

    BDict = Dict{String,SparseArrays.SparseMatrixCSC{Float64,Int64}}()
    if zero
        ppositions = cumsum(model.C .<= 0)
        qpositions = cumsum(model.C .>= 0)
        for ℓ in ["+", "-", "0"], m in ["+", "-", "0"]
            for i = 1:NPhases(model), j = 1:NPhases(model)
                FilBases = repeat(mesh.Fil[string(i, ℓ)]', NBases(mesh), 1)[:]
                pitemp = falses(N₋)
                qitemp = falses(N₊)
                pjtemp = falses(N₋)
                qjtemp = falses(N₊)
                if model.C[i] <= 0
                    if length(pitemp) > 0 
                        pitemp[ppositions[i]] = mesh.Fil["p"*string(i)*ℓ][1]
                    end
                end
                if model.C[j] <= 0
                    if length(pjtemp) > 0
                        pjtemp[ppositions[j]] = mesh.Fil["p"*string(j)*m][1]
                    end
                end
                if model.C[i] >= 0
                    if length(qitemp) > 0
                        qitemp[qpositions[i]] = mesh.Fil["q"*string(i)*ℓ][1]
                    end
                end
                if model.C[j] >= 0
                    if length(qjtemp) > 0
                        qjtemp[qpositions[j]] = mesh.Fil["q"*string(j)*m][1]
                    end
                end
                i_idx = [
                    pitemp
                    falses((i - 1) * TotalNBases(mesh))
                    FilBases
                    falses(NPhases(model) * TotalNBases(mesh) - i * TotalNBases(mesh))
                    qitemp
                ]
                FjmBases = repeat(mesh.Fil[string(j, m)]', NBases(mesh), 1)[:]
                j_idx = [
                    pjtemp
                    falses((j - 1) * TotalNBases(mesh))
                    FjmBases
                    falses(NPhases(model) * TotalNBases(mesh) - j * TotalNBases(mesh))
                    qjtemp
                ]
                BDict[string(i, j, ℓ, m)] = B[i_idx, j_idx]
            end
            # below we need to use repeat(mesh.Fil[ℓ]', NBases(mesh), 1)[:] to
            # expand the index mesh.Fil[ℓ] from cells to all basis function
            FlBases =
                [mesh.Fil["p"*ℓ]; repeat(mesh.Fil[ℓ]', NBases(mesh), 1)[:]; mesh.Fil["q"*ℓ]]
            FmBases =
                [mesh.Fil["p"*m]; repeat(mesh.Fil[m]', NBases(mesh), 1)[:]; mesh.Fil["q"*m]]
            BDict[ℓ*m] = B[FlBases, FmBases]
        end
    else
        ppositions = cumsum(model.C .<= 0)
        qpositions = cumsum(model.C .>= 0)
        for ℓ in ["+", "-"]
            for i = 1:NPhases(model)
                FilBases = repeat(mesh.Fil[string(i, ℓ)]', NBases(mesh), 1)[:]
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
                    falses((i - 1) * TotalNBases(mesh))
                    FilBases
                    falses(NPhases(model) * TotalNBases(mesh) - i * TotalNBases(mesh))
                    qitemp
                ]
                BDict[string(i, ℓ)] = B[i_idx, i_idx]
            end
            FlBases =
                [mesh.Fil["p"*ℓ]; repeat(mesh.Fil[ℓ]', NBases(mesh), 1)[:]; mesh.Fil["q"*ℓ]]
            BDict[ℓ] = B[FlBases, FlBases]
        end
    end
    return BDict
end

abstract type Generator end 

issquare(A::AbstractArray{<:Any,2}) = size(A,1)==size(A,2)

struct LazyB <: AbstractArray{Real,2}
    blocks::Tuple{Array{Float64,2},Array{Float64,2},Array{Float64,2},Array{Float64,2}}
    boundary_flux::NamedTuple{(:in, :out),Tuple{Array{Float64,1},Array{Float64,1}}}
    T::Array{<:Real,2}
    C::Array{<:Real,1}
    Δ::Array{<:Real,1}
    D::Union{Array{Float64,2},LinearAlgebra.Diagonal{Bool,Array{Bool,1}}}
    pmidx::Union{Array{Bool,2},BitArray{2}}
    function LazyB(
        blocks::Tuple{Array{Float64,2},Array{Float64,2},Array{Float64,2},Array{Float64,2}},
        boundary_flux::NamedTuple{(:in, :out),Tuple{Array{Float64,1},Array{Float64,1}}},
        T::Array{<:Real,2},
        C::Array{<:Real,1},
        Δ::Array{<:Real,1},
        D::Union{Array{Float64,2},LinearAlgebra.Diagonal{Bool,Array{Bool,1}}},
        pmidx::Union{Array{Bool,2},BitArray{2}},
    )
        s = size(blocks[1])
        for b in 1:4
            !issquare(blocks[1]) && throw(DomainError("blocks must be square"))
            !(s == size(blocks[b])) && throw(DomainError("blocks must be the same size"))
        end
        !issquare(T) && throw(DomainError("T must be square"))
        !issquare(D) && throw(DomainError("D must be square"))
        !(s == size(D)) && throw(DomainError("blocks must be the same size as D"))
        !issquare(pmidx) && throw(DomainError("pmidx must be square"))
        !(size(T) == size(pmidx)) && throw(DomainError("pmidx must be the same size as T"))
        !(length(C) == size(T,1)) && throw(DomainError("C must be the same length as T"))
        
        return new(blocks,boundary_flux,T,C,Δ,D,pmidx)
    end
    # size_blocks::Int64
    # size_T::Int64
end
function LazyB(
    blocks::Tuple{Array{Float64,2},Array{Float64,2},Array{Float64,2}},
    boundary_flux::NamedTuple{(:in, :out),Tuple{Array{Float64,1},Array{Float64,1}}},
    T::Array{<:Real,2},
    C::Array{<:Real,1},
    Δ::Array{<:Real,1},
    D::Union{Array{Float64,2},LinearAlgebra.Diagonal{Bool,Array{Bool,1}}},
    pmidx::Union{Array{Bool,2},BitArray{2}},
)
    blocks = (blocks[1],blocks[2],blocks[2],blocks[3])
    return LazyB(
        blocks,
        boundary_flux,
        T,
        C,
        Δ,
        D,
        pmidx,
    )
end

function size(B::LazyB)
    sz = size(B.T,1)*size(B.blocks[1],1)*length(B.Δ) + sum(B.C.<=0) + sum(B.C.>=0)
    return (sz,sz)
end

function *(u::Array{<:Real,2}, B::LazyB)
    sz_u_1 = size(u,1)
    sz_u_2 = size(u,2)
    sz_B_1 = size(B,1)
    sz_B_2 = size(B,2)
    !(sz_u_2 == sz_B_1) && throw(DomainError("Dimension mismatch, u*B, length(u) must be size(B,1)"))
    N₋ = sum(B.C.<=0)
    N₊ = sum(B.C.>=0)
    v = zeros(sz_u_1,sz_B_2)
    size_delta = length(B.Δ)
    size_blocks = size(B.blocks[1],1)
    size_T = size(B.T,1)
    for row in 1:sz_u_1
        # v[row,]
        # boundaries
        # at lower
        v[row,1:N₋] += u[row,1:N₋]'*B.T[B.C.<=0,B.C.<=0]
        # in to lower 
        idxdown = N₋ .+ ((1:size_blocks).+size_blocks*size_delta*(findall(B.C .<= 0) .- 1)')[:]
        v[row,1:N₋] += u[row,idxdown]'*LinearAlgebra.kron(
            LinearAlgebra.diagm(0 => abs.(B.C[B.C.<=0])),
            B.boundary_flux.in/B.Δ[1],
        )
        # out of lower 
        idxup = N₋ .+ (size_blocks*size_delta*(findall(B.C .> 0).-1)' .+ (1:size_blocks))[:]
        v[row,idxup] = u[row,1:N₋]'*kron(B.T[B.C.<=0,B.C.>0],B.boundary_flux.out')

        # at upper
        v[row,end-N₊+1:end] += u[row,end-N₊+1:end]'*B.T[B.C.>=0,B.C.>=0]
        # in to upper
        idxup = N₋ .+ ((1:size_blocks).+size_blocks*size_delta*(findall(B.C .>= 0) .- 1)')[:] .+
            (size_blocks*size_delta - size_blocks)
        v[row,end-N₊+1:end] += u[row,idxup]'*LinearAlgebra.kron(
            LinearAlgebra.diagm(0 => B.C[B.C.>=0]),
            B.boundary_flux.in/B.Δ[end],
        )
        # out of upper 
        idxdown = N₋ .+ (size_blocks*size_delta*(findall(B.C .< 0).-1)' .+ (1:size_blocks))[:] .+
            (size_blocks*size_delta - size_blocks)
        v[row,idxdown] = u[row,1:N₋]'*kron(B.T[B.C.<=0,B.C.>0],B.boundary_flux.out')

        # innards
        for i in 1:size_T, j in 1:size_T
            if i == j 
                # mult on diagonal
                for k in 1:size_delta
                    k_idx = (i-1)*size_blocks*size_delta .+ (k-1)*size_blocks .+ (1:size_blocks) .+ N₋
                    for ℓ in 1:size_delta
                        if (k == ℓ+1) && (B.C[i] > 0)
                            ℓ_idx = k_idx .- size_blocks 
                            v[row,k_idx] += B.C[i]*(u[row,ℓ_idx]'*B.blocks[4])'/B.Δ[ℓ]
                        elseif k == ℓ
                            v[row,k_idx] += (u[row,k_idx]'*(abs(B.C[i])*B.blocks[2 + (B.C[i].<0)]/B.Δ[ℓ] + B.T[i,j]*LinearAlgebra.I))'
                        elseif (k == ℓ-1) && (B.C[i] < 0)
                            ℓ_idx = k_idx .+ size_blocks 
                            v[row,k_idx] += abs(B.C[i])*(u[row,ℓ_idx]'*B.blocks[1])'/B.Δ[ℓ]
                        end
                    end
                end
            elseif B.pmidx[i,j]
                # changes from S₊ to S₋ etc.
                for k in 1:size_delta
                    for ℓ in 1:size_delta
                        if k == ℓ
                            i_idx = (i-1)*size_blocks*size_delta .+ (k-1)*size_blocks .+ (1:size_blocks) .+ N₋
                            j_idx = (j-1)*size_blocks*size_delta .+ (k-1)*size_blocks .+ (1:size_blocks) .+ N₋
                            v[row,j_idx] += (u[row,i_idx]'*(B.T[i,j]*B.D))'
                        end
                    end
                end
            else
                i_idx = (i-1)*size_blocks*size_delta .+ (1:size_blocks*size_delta) .+ N₋
                j_idx = (j-1)*size_blocks*size_delta .+ (1:size_blocks*size_delta) .+ N₋
                v[row,j_idx] += (u[row,i_idx]'*B.T[i,j])'
            end
        end
    end
    return v
end

struct Lazy_Generator <: Generator 
    BDict::Dict{String,AbstractArray{Float64,2}}
    B::LazyB
    QBDidx::Array{Int64,1}
end

struct Full_Generator <: Generator 
    BDict::Dict{String,AbstractArray{Float64,2}}
    B::Union{AbstractArray{Float64,Int64},SparseArrays.SparseMatrixCSC{Float64,Int64}}
    QBDidx::Array{Int64,1}
end


include("DGBase.jl")
include("Operators.jl")
include("FVM.jl")
include("FRAPApproximation.jl")
include("Distributions.jl")
include("SimulateSFFM.jl")
include("SFM.jl")
include("Plots.jl")

function MyPrint(Obj)
    show(stdout, "text/plain", Obj)
end

"""
Construct all the DG operators.

    MakeAll(
        model::SFFM.Model,
        mesh::DGMesh;
        approxType::String = "projection"
    )

# Arguments
- `model`: a model object as output from Model
- `mesh`: a Mesh object
- `approxType::String`: (optional) argument specifying how to approximate R (in
    `MakeR()`)


# Output
- a tuple with keys
    - `Matrices`: see `MakeMatrices`
    - `MatricesR`: see `MakeMatricesR`
    - `B`: see `MakeB`
    - `D`: see `MakeD`
    - `DR`: see `MakeDR`
"""
function MakeAll(
    model::SFFM.Model,
    mesh::DGMesh;
    approxType::String = "projection"
)

    Matrices = MakeMatrices(model, mesh)
    
    B = MakeB(model, mesh, Matrices)
    R = MakeR(model, mesh, approxType = approxType)
    D = MakeD(mesh, B, R)
    return (
        Matrices = Matrices,
        B = B,
        R = R,
        D = D,
    )
end

end
