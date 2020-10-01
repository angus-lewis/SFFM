include("../src/SFFM.jl")
using LinearAlgebra, Plots, JLD2, SparseArrays

T = [-1.0 1.0 0; 0.5 -1.0 0.5; 0.5 0.5 -1]
C = [2.0; -2.0; -1.0]

r = (
    r = function (x)
        [-2*(x.>-1) 2*(x.>-1) -1*(x.>-1)] # [ones(size(x)) ones(size(x))]#
    end,
    R = function (x)
        [-2*(x.>-1).*x 2*(x.>-1).*x -1*(x.>-1).*x] # [x x]
    end,
)

Bounds = [0 6; -Inf Inf]
Model = SFFM.MakeModel(T = T, C = C, r = r, Bounds = Bounds)

Δ = 2
Nodes = collect(Bounds[1,1]:Δ:Bounds[1,2])
NBases = 3
Basis = "lagrange"
Mesh = SFFM.MakeMesh(Model = Model, Nodes = Nodes, NBases = NBases, Basis=Basis)

V = SFFM.vandermonde(NBases=NBases)
Matrices2 = SFFM.MakeMatrices2(Model=Model, Mesh=Mesh)
MatricesR = SFFM.MakeMatricesR(Model=Model, Mesh=Mesh)
MR = SparseArrays.spzeros(Float64, Model.NPhases * Mesh.TotalNBases, Model.NPhases * Mesh.TotalNBases)
Minv =
    SparseArrays.spzeros(Float64, Model.NPhases * Mesh.TotalNBases, Model.NPhases * Mesh.TotalNBases)
FGR = SparseArrays.spzeros(Float64, Model.NPhases * Mesh.TotalNBases, Model.NPhases * Mesh.TotalNBases)
for i in 1:Model.NPhases
    idx = (1:Mesh.TotalNBases) .+ (i-1)*Mesh.TotalNBases
    MR[idx,idx] = MatricesR.Global.M[i]
    Minv[idx,idx] = Matrices2.Global.MInv
    FGR[idx,idx] = Model.C[i]*(MatricesR.Global.F[i]+MatricesR.Global.G[i])
end

T = SparseArrays.kron(Model.T, SparseArrays.sparse(LinearAlgebra.I,Mesh.TotalNBases,Mesh.TotalNBases))
DR = MR * T * Minv + FGR * Minv




All = SFFM.MakeAll(Model = Model, Mesh = Mesh, approxType = "interpolation")
Matrices = SFFM.MakeMatrices(Model=Model,Mesh=Mesh,probTransform=false)
MatricesR = SFFM.MakeMatricesR(Model=Model,Mesh=Mesh)
B = SFFM.MakeB(Model=Model,Mesh=Mesh,Matrices=Matrices,probTransform=false)
Dr = SFFM.MakeDR(
    Matrices=Matrices,
    MatricesR=MatricesR,
    Model=Model,
    Mesh=Mesh,
    B=B,
)

Db = All.R.R[3:end-1,3:end-1]*B.B[3:end-1,3:end-1]
