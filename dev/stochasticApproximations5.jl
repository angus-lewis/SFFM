include("../src/SFFM.jl")
T =[-2.0 1.0 1; 1.0 -2.0 1; 4 3 -7]# [-2.0 2.0; 1.0 -1.0]#
C = [3.0; -2.0; 0]
fn(x) = [ones(size(x)) -ones(size(x)) -ones(size(x))]
model = SFFM.Model(;T=T,C=C,r=(r=fn,R=fn),Bounds=[0 10;-Inf Inf])

order = 1
mesh = SFFM.MakeMesh(model=model,Nodes = collect(0:5:10),NBases=order,Basis="lagrange")

me = SFFM.MakeME(SFFM.CMEParams[order])
erlang = SFFM.MakeErlang(order)

me = SFFM.MakeME(SFFM.CMEParams[order], mean=mesh.Δ[1])
B = SFFM.MakeBFRAP(model=model,mesh=mesh,me=me)
R = SFFM.MakeR(model = model, mesh = mesh, approxType = "interpolation", probTransform = false)
D = SFFM.MakeD(R=R,B=B,model=model,mesh=mesh)
Ψme = SFFM.PsiFun(D=D)

mat = SFFM.MakeMatrices(model=model,mesh=mesh)
B2 = SFFM.MakeB(model=model,mesh=mesh,Matrices=mat)
All = SFFM.MakeAll(model = model, mesh = mesh, approxType = "interpolation")
Ψ = SFFM.PsiFun(D=All.D)