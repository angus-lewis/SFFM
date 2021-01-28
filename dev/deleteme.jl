using LinearAlgebra, Plots, JSON, Jacobi
include("METools.jl")
using SymPy

order = 5
p = plot(layout = (2,1))
for order = [1,3,7,15,23]
    μ = 1
    ME = MakeME(CMEParams[order], mean = μ)
    MC = MakeErlang(order, mean = μ)
    D = CMEParams[order]["D"]

    MEproperties = renewalProperties(ME)

    MEproperties.density(1)
    MEproperties.mean(1)
    MEproperties.ExpectedOrbit(1)

    MCproperties = renewalProperties(MC)

    MCproperties.density(1)
    MCproperties.mean(1)
    MCproperties.ExpectedOrbit(1)

    t = 0:0.05:4
    plot!(p,t,MEproperties.density.(t),subplot = 1, legend = :outertopright,
        label = order, color = order)
    plot!(p,t,MEproperties.mean.(t),subplot = 2, legend = :outertopright,
        label = order, colour = order)
    # plot!(p,t,MCproperties.density.(t),subplot = 1, label = false,
    #     color = order, linestyle = :dash)
    # plot!(p,t,MCproperties.mean.(t),subplot = 2, label = false,
    #     colour = order, linestyle = :dash)
end
display(p)

E = eigen(ME.Q)
E.values
x = real(E.values[2])
y = abs.(imag(E.values[1]))

C=[0 1 0; 0 0 1; (x^3+x*y^2) -(3*x^2+y^2) 3*x]

b = -C[end,:]'
bfun(s) = b*(s.^(0:2))+s^3

M = eigen(C).vectors[:,[3;1;2]]

V = eigen(ME.Q).vectors[:,[2;1;3]]

W = M*V^-1

s = Sym("s")
L(s) = ME.α*inv(s*I-ME.Q)*ME.q

p = [L(0)*bfun(0);L(1)*bfun(1);L(2)*bfun(2)]
P = [1 0 0; 1 1 1; 1 2 4]

a = (P\p)'

a*exp(C*2)*[0;0;1]

Cu = [0 1 0 0; 0 0 1 0; 0 0 0 1; 0 -(b-a)]
display(simplify(([b 1]*inv(s*I(4)-Cu)*[0;0;0;1])[1]))
U(t) = [b 1]*exp(Cu*t)*[0;0;0;1]

t = 0:0.05:4

plot!(t,U.(t),subplot = 2)
