# Toy model from paper
include("../../src/SFFM.jl")
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
model = SFFM.Model(T, C, r, Bounds = Bounds)

## Define mesh
Nodes = [0;1;1.8]
nBases = 3
Basis = "lagrange"
mesh = SFFM.DGMesh(model, Nodes, nBases, Basis=Basis)

## Make matrices
All = SFFM.MakeAll( model, mesh)
M = SFFM.MakeMatrices( model, mesh)
B = SFFM.MakeB( model, mesh, M)
R = SFFM.MakeR( model, mesh)

println("")
println("The operator B looks like: ")
display(Matrix(B.B[B.QBDidx,B.QBDidx]))

println("")
println("The subarrays of B look like this:")
println("for 11++")
display(Matrix(B.BDict["11++"]))
println("")
println("for 11+-")
display(Matrix(B.BDict["11+-"]))
println("")
println("for 11-+")
display(Matrix(B.BDict["11-+"]))
println("")
println("for 11--")
display(Matrix(B.BDict["11--"]))

println("")
println("for 12++")
display(Matrix(B.BDict["12++"]))
println("")
println("for 12+-")
display(Matrix(B.BDict["12+-"]))
println("")
println("for 12-+")
display(Matrix(B.BDict["12-+"]))
println("")
println("for 12--")
display(Matrix(B.BDict["12--"]))

println("")
println("for 21++")
display(Matrix(B.BDict["21++"]))
println("")
println("for 21+-")
display(Matrix(B.BDict["21+-"]))
println("")
println("for 21-+")
display(Matrix(B.BDict["21-+"]))
println("")
println("for 21--")
display(Matrix(B.BDict["21--"]))

println("")
println("for 22++")
display(Matrix(B.BDict["22++"]))
println("")
println("for 22+-")
display(Matrix(B.BDict["22+-"]))
println("")
println("for 22-+")
display(Matrix(B.BDict["22-+"]))
println("")
println("for 22--")
display(Matrix(B.BDict["22--"]))

plusIdx = B.QBDidx[[mesh.Fil["p+"];repeat(mesh.Fil["+"]',2)[:];mesh.Fil["q+"]]]
minusIdx = B.QBDidx[[mesh.Fil["p-"];repeat(mesh.Fil["-"]',2)[:];mesh.Fil["q-"]]]
println("")
println("for ++")
display(Matrix(B.B)[plusIdx,plusIdx])
println("")
println("for +-")
display(Matrix(B.B)[plusIdx,minusIdx])
println("")
println("for -+")
display(Matrix(B.B)[minusIdx,plusIdx])
println("")
println("for --")
display(Matrix(B.B)[minusIdx,minusIdx])

println("The operator R⁺ is ")
display(Matrix(R.RDict["+"]))
println("")
println("The operator R⁻ is ")
display(Matrix(R.RDict["-"]))
