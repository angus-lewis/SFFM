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
model = SFFM.Model(T = T, C = C, r = r, Bounds = Bounds)

## Define mesh
Nodes = [0;1;1.8]
NBases = 2
Basis = "lagrange"
mesh = SFFM.MakeMesh(model = model, Nodes = Nodes, NBases = NBases, Basis=Basis)

## Make matrices
All = SFFM.MakeAll(model=model,mesh=mesh)
M = SFFM.MakeMatrices(model = model, mesh = mesh, probTransform=false)
B = SFFM.MakeB(model=model,mesh=mesh,Matrices=M, probTransform=false)
R = SFFM.MakeR(model=model,mesh=mesh)

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
