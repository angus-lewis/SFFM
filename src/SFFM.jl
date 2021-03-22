module SFFM
import Jacobi, LinearAlgebra, SparseArrays
import Plots, StatsBase, KernelDensity

"""
Construct a SFFM model object.

    Model(;
        T::Array{Float64,2},
        C::Array{Float64,1},
        r::NamedTuple{(:r, :R)},
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
    T::Array{<:Real,2}
    C::Array{<:Real,1}
    r::NamedTuple{(:r, :R, :a)}
    Bounds::Array{<:Real,2}
    NPhases::Int
    SDict::Dict{String,Array}
    TDict::Dict{String,Array}

    function Model(;
        T::Array{<:Real,2},
        C::Array{<:Real,1},
        r::NamedTuple{(:r, :R)},
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
end 

"""

    Mesh 

Abstract type representing a mesh for a numerical scheme. 
"""
abstract type Mesh end 

struct FVMesh <: Mesh 
    NBases::Int
    CellNodes::Array{<:Real,2}
    Fil::Dict{String,BitArray{1}}
    Δ::Array{Float64,1}
    NIntervals::Int
    Nodes::Array{Float64,1}
    Basis::String
    function FVMesh(
        model::SFFM.Model;
        Nodes::Array{Float64,1},
        Fil::Dict{String,BitArray{1}}=Dict{String,BitArray{1}}(),
    )
        NBases = 1
        NIntervals = length(Nodes) - 1 # the number of intervals
        Δ = (Nodes[2:end] - Nodes[1:end-1]) # interval width
        CellNodes = zeros(Float64, NBases, NIntervals)
        for i = 1:NIntervals
            CellNodes[:, i] .= (Nodes[i+1] + Nodes[i]) / 2 
        end
        TotalNBases = NBases * NIntervals 

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
        new(NBases, CellNodes, Fil, Δ, NIntervals, Nodes, "Constant")
    end
end 

include("SFFMPlots.jl")
include("SimulateSFFM.jl")
include("SFFMDGBase.jl")
include("SFFMDGAdv.jl")
include("SFFMDGpi.jl")
include("SFM.jl")
include("METools.jl")
include("FRAPApproximation.jl")

function MyPrint(Obj)
    show(stdout, "text/plain", Obj)
end

"""
Construct all the DG operators.

    MakeAll(;
        model::SFFM.Model,
        mesh::DGMesh,
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
function MakeAll(;
    model::SFFM.Model,
    mesh::DGMesh,
    approxType::String = "projection"
)

    Matrices = MakeMatrices(model = model, mesh = mesh)
    # MatricesR = MakeMatricesR(model = model, mesh = mesh)
    B = MakeB(model = model, mesh = mesh, Matrices = Matrices)
    R = MakeR(model = model, mesh = mesh, approxType = approxType)
    D = MakeD(model = model, mesh = mesh, R = R, B = B)
    # DR = MakeDR(
    #     Matrices = Matrices,
    #     MatricesR = MatricesR,
    #     model = model,
    #     mesh = mesh,
    #     B = B,
    # )
    return (
        Matrices = Matrices,
        # MatricesR = MatricesR,
        B = B,
        R = R,
        D = D,
        # DR = DR
    )
end

end
