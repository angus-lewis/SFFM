include("../src/SFFM.jl")
T =[-2.0 2.0 0; 1.0 -2.0 1; 4 3 -7]# [-2.0 2.0; 1.0 -1.0]#
C = [1.0; 2.0; 0]
fn(x) = [ones(size(x)) ones(size(x)) ones(size(x))]
model = SFFM.Model(;T=T,C=C,r=(r=fn,R=fn),Bounds=[0 10;-Inf Inf])

order = 3

me = SFFM.MakeME(SFFM.CMEParams[order])
erlang = SFFM.MakeErlang(order)