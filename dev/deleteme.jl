# Toy model from paper
include("../src/SFFM.jl")
using LinearAlgebra

## Define the model
T = [-1.0 1.0; 1.0 -1.0]
C = [1.0; -2.0]
r = (
    r = function (x)
        [(x .<= 1)-(x .> 1) ((x .> 1))-2*(x .<= 1).*(x .> 0)-(x.==0)]
    end,
    R = function (x)
        [x.*((x .<= 1)-(x .> 1)) x.*(((x .> 1))-2*(x .<= 1).*(x .> 0)-(x.==0))]
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
All = SFFM.MakeAll(Model=Model,Mesh=Mesh)
M = SFFM.MakeMatrices2(Model = Model, Mesh = Mesh)

for t in 0:0.05:8
    temp = [1 0]*(exp([-1 1; -1 -1]./2*t) - exp([-1 1; 1 -1]./2*t))
    if sum(temp)<0
        display(temp)
        display(t)
    end
end
