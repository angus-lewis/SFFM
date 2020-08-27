# Toy model from paper
include("../src/SFFM.jl")
using LinearAlgebra

## Define the model
T = [-1.0 1.0; 1.0 -1.0]
C = [1.0; -2.0]
r = (
    r = function (x)
        [(x .< 1)-(x .>= 1) (x .>= 1)-(x .< 1)]
    end,
    R = function (x)
        [(x .< 1).*x-(x .>= 1).*x (x .>= 1).*x-(x .< 1).*x]
    end,
)

Bounds = [0 1.8; -Inf Inf]
Model = SFFM.MakeModel(T = T, C = C, r = r, Bounds = Bounds)

## Define mesh
Nodes = [0;1;1.8]
NBases = 2
Basis = "lagrange"
Mesh = SFFM.MakeMesh(Model = Model, Nodes = Nodes, NBases = NBases, Basis=Basis)

## Make matrices
All = SFFM.MakeAll(Model = Model, Mesh = Mesh, approxType = "projection")
display(Matrix(All.B.B[All.B.QBDidx,All.B.QBDidx]))

display(Matrix(All.B.BDict["11++"]))
display(Matrix(All.B.BDict["11+-"]))
display(Matrix(All.B.BDict["11-+"]))
display(Matrix(All.B.BDict["11--"]))

display(Matrix(All.B.BDict["12++"]))
display(Matrix(All.B.BDict["12+-"]))
display(Matrix(All.B.BDict["12-+"]))
display(Matrix(All.B.BDict["12--"]))

display(Matrix(All.B.BDict["21++"]))
display(Matrix(All.B.BDict["21+-"]))
display(Matrix(All.B.BDict["21-+"]))
display(Matrix(All.B.BDict["21--"]))

display(Matrix(All.B.BDict["22++"]))
display(Matrix(All.B.BDict["22+-"]))
display(Matrix(All.B.BDict["22-+"]))
display(Matrix(All.B.BDict["22--"]))
