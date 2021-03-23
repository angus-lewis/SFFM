module SFFM
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
    - `NPhases::Int`: the number of states in the state space
    - `SDict::Dict{String,Array}`: a dictionary with keys `"+","-","0","bullet"`
        and corresponding values `findall(C .> 0)`, `findall(C .< 0)`,
        `findall(C .== 0)`, `findall(C .!= 0)`, respectively.
    - `TDict::Dict{String,Array}`: a dictionary of submatrices of `T` with keys
        `"ℓm"` with ``ℓ,m∈{+,-,0,bullet}`` and corresponding values
        `T[S[ℓ],S[m]]`.
)
"""
struct Model 
    T::Array{<:Real}
    C::Array{<:Real,1}
    r::NamedTuple{(:r, :R, :a)}
    Bounds::Array{<:Real}
    NPhases::Int
    SDict::Dict{String,Array}
    TDict::Dict{String,Array}

    function Model(
        T::Array{<:Real},
        C::Array{<:Real,1},
        r::NamedTuple{(:r, :R)};
        Bounds::Array{<:Real,2} = [-Inf Inf; -Inf Inf],
    )
        a(x) = abs.(r.r(x))
        r = (r = r.r, R = r.R, a = a)
        NPhases = length(C)
    
        SDict = Dict{String,Array}("S" => 1:NPhases)
        SDict["+"] = findall(C .> 0)
        SDict["-"] = findall(C .< 0)
        SDict["0"] = findall(C .== 0)
        SDict["bullet"] = findall(C .!= 0)
    
        TDict = Dict{String,Array}("T" => T)
        for ℓ in ["+" "-" "0" "bullet"], m in ["+" "-" "0" "bullet"]
            TDict[ℓ*m] = T[SDict[ℓ], SDict[m]]
        end
    
        println("UPDATE: Model object created with fields ", fieldnames(SFFM.Model))
        return new(
            T,
            C,
            r,
            Bounds,
            NPhases,
            SDict,
            TDict,
        )
    end
    function Model()
        new(
            [0],
            [0],
            (r=0, R=0, a=0),
            [0],
            0,
            Dict{String,Array}(),
            Dict{String,Array}(),
        )
    end
end 

"""

    Mesh 

Abstract type representing a mesh for a numerical scheme. 
"""
abstract type Mesh end 

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
            # below we need to use repeat(mesh.Fil[ℓ]', mesh.NBases, 1)[:] to
            # expand the index mesh.Fil[ℓ] from cells to all basis function
            FlBases =
                [mesh.Fil["p"*ℓ]; repeat(mesh.Fil[ℓ]', mesh.NBases, 1)[:]; mesh.Fil["q"*ℓ]]
            FmBases =
                [mesh.Fil["p"*m]; repeat(mesh.Fil[m]', mesh.NBases, 1)[:]; mesh.Fil["q"*m]]
            BDict[ℓ*m] = B[FlBases, FmBases]
        end
    else
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
                BDict[string(i, ℓ)] = B[i_idx, i_idx]
            end
            FlBases =
                [mesh.Fil["p"*ℓ]; repeat(mesh.Fil[ℓ]', mesh.NBases, 1)[:]; mesh.Fil["q"*ℓ]]
            BDict[ℓ] = B[FlBases, FlBases]
        end
    end
    return BDict
end

include("Plots.jl")
include("SimulateSFFM.jl")
include("DGBase.jl")
include("DGAdv.jl")
include("Operators.jl")
include("Distributions.jl")
include("SFM.jl")
include("METools.jl")
include("FRAPApproximation.jl")

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
