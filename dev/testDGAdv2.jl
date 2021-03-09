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
model = SFFM.Model(T = T, C = C, r = r, Bounds = Bounds)

Δ = 2
Nodes = collect(Bounds[1,1]:Δ:Bounds[1,2])
NBases = 3
Basis = "lagrange"
mesh = SFFM.MakeMesh(model = model, Nodes = Nodes, NBases = NBases, Basis=Basis)

V = SFFM.vandermonde(NBases=NBases)
Matrices2 = SFFM.MakeMatrices2(model=model, mesh=mesh)
MatricesR = SFFM.MakeMatricesR(model=model, mesh=mesh)
MR = SparseArrays.spzeros(Float64, model.NPhases * mesh.TotalNBases, model.NPhases * mesh.TotalNBases)
Minv =
    SparseArrays.spzeros(Float64, model.NPhases * mesh.TotalNBases, model.NPhases * mesh.TotalNBases)
FGR = SparseArrays.spzeros(Float64, model.NPhases * mesh.TotalNBases, model.NPhases * mesh.TotalNBases)
for i in 1:model.NPhases
    idx = (1:mesh.TotalNBases) .+ (i-1)*mesh.TotalNBases
    MR[idx,idx] = MatricesR.Global.M[i]
    Minv[idx,idx] = Matrices2.Global.MInv
    FGR[idx,idx] = model.C[i]*(MatricesR.Global.F[i]+MatricesR.Global.G[i])
end

T = SparseArrays.kron(model.T, SparseArrays.sparse(LinearAlgebra.I,mesh.TotalNBases,mesh.TotalNBases))
DR = MR * T * Minv + FGR * Minv




All = SFFM.MakeAll(model = model, mesh = mesh, approxType = "interpolation")
Matrices = SFFM.MakeMatrices(model=model,mesh=mesh,probTransform=false)
MatricesR = SFFM.MakeMatricesR(model=model,mesh=mesh)
B = SFFM.MakeB(model=model,mesh=mesh,Matrices=Matrices,probTransform=false)
Dr = SFFM.MakeDR(
    Matrices=Matrices,
    MatricesR=MatricesR,
    model=model,
    mesh=mesh,
    B=B,
)

Db = All.R.R[3:end-1,3:end-1]*B.B[3:end-1,3:end-1]
