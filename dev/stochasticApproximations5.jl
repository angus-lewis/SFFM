include("../src/SFFM.jl")
T =[-2.0 1.0 1; 1.0 -2.0 1; 4 3 -7]# [-2.0 2.0; 1.0 -1.0]#
C = [1.0; -2.0; 0]
fn(x) = [ones(size(x)) -ones(size(x)) -ones(size(x))]
model = SFFM.Model(;T=T,C=C,r=(r=fn,R=fn),Bounds=[0 10;-Inf Inf])

order = 3
mesh = SFFM.MakeMesh(model=model,Nodes = collect(0:2:10),NBases=order)

me = SFFM.MakeME(SFFM.CMEParams[order])
erlang = SFFM.MakeErlang(order)

me = SFFM.MakeME(SFFM.CMEParams[order])
B = SFFM.MakeBFRAP(model=model,mesh=mesh,me=me)
R = SFFM.MakeR(model = model, mesh = mesh, approxType = "interpolation")
D = SFFM.MakeD(R=R,B=B,model=model,mesh=mesh)
Ψme = SFFM.PsiFun(D=D)

mat = SFFM.MakeMatrices(model=model,mesh=mesh)
B2 = SFFM.MakeB(model=model,mesh=mesh,Matrices=mat)