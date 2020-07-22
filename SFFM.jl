module SFFM
import Jacobi, LinearAlgebra

include("SFFMPlots.jl")
include("SimulateSFFM.jl")
include("SFFMDGBase.jl")
include("SFFMDGAdv.jl")

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
    r::NamedTuple{(:r, :R)},
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
    #                         .T, .C, .r, .IsBounded, .NPhases

    NPhases = length(C)
    println("Model.Field with Fields (.T, .C, .r, .IsBounded, .NPhases)")
    IsBounded = true
    return (T = T, C = C, r = r, IsBounded = IsBounded, Bounds = Bounds, NPhases = NPhases)
end

function MakeAll(;
    Model::NamedTuple{(:T, :C, :r, :IsBounded, :Bounds, :NPhases)},
    Mesh::NamedTuple{
        (:NBases, :CellNodes, :Fil, :Δ, :NIntervals, :MeshArray, :Nodes, :TotalNBases, :Basis),
    },
)

    Matrices = MakeMatrices(Model = Model, Mesh = Mesh)
    MatricesR = MakeMatricesR(Model = Model, Mesh = Mesh)
    B = MakeB(Model = Model, Mesh = Mesh, Matrices = Matrices)
    R = MakeR(Model = Model, Mesh = Mesh, V = Matrices.Local.V)
    D = MakeD(Model = Model, Mesh = Mesh, R = R, B = B)
    DR = MakeDR(
        Matrices = Matrices,
        MatricesR = MatricesR,
        Model = Model,
        Mesh = Mesh,
        B = B,
    )
    return (Matrices = Matrices, MatricesR = MatricesR, B = B, R = R, D = D, DR = DR)
end

end # end module
