using Plots, SFFM

include("../../src/SFFM.jl")

cme_21 = SFFM.MakeME(SFFM.CMEParams[21])

f = SFFM.pdf(cme_21)
F(x) = 1 - SFFM.cdf(cme_21)(x)

# x = 0:0.05:1.5
plot(x->f(x), 0, 1.5, label = "α exp(Sz) s")
plot!(x->f(x.+0.3)./F.(0.3), 0, 1.5, label = "α exp(S(0.3+z)) s/α exp(S 0.3) e")
plot!(x->f(x.+0.6)./F.(0.6), 0, 1.5, label = "α exp(S(0.6+z)) s/α exp(S 0.6) e")
plot!(
    xlabel = "z",
    ylabel = "Density",
    legend = :outertop,
)

savefig("examples/meNumerics/ME_residual_life_density.pdf")