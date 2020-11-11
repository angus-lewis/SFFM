using LinearAlgebra, Plots, JSON
include("../src/SFFM.jl")
include("METools.jl")

# define SFM
T = [-2.0 2.0; 1.0 -1.0]
C = [1.0; -2.0]
fn(x) = [ones(size(x)) ones(size(x))]
Model = SFFM.MakeModel(;T=T,C=C,r=(r=fn,R=fn),Bounds=[0 10;-Inf Inf])
N₋ = sum(C.<=0)
N₊ = sum(C.>=0)
NPhases = length(C)

Δ = 1
NBases = 7
order = NBases

Erlang = MakeErlang(order,mean=Δ)

Coxian = MakeSomeCoxian(order,mean=Δ)

Coxianbkwd = reversal(Coxian)


t = 0:0.01:4
orbitCoxian = orbit(t,Coxian)
orbitCoxianbkwd = orbit(t,Coxianbkwd)
plot(t,orbitCoxian,label = "fwd")
plot!(t,orbitCoxianbkwd,label = "bkwd")

plot(orbitCoxian[:,2],orbitCoxian[:,3],label = "fwd",legend=:outerright)
plot!(orbitCoxianbkwd[:,2],orbitCoxianbkwd[:,3],label = "bkwd")

αME, QME, ~ = MakeME(CMEParams[NBases],mean=Δ)#MakeSomeCoxian(order,mean=Δ)#
CME = (α = αME, Q = QME, q = sum(QME,dims=2), π = αME*QME^(-1)./sum(αME*QME^(-1)))
CMEr = reversal(CME)
πME = (α = CME.π, Q = QME, π = CME.π)
πMEr = (α = CMEr.π, Q = CMEr.Q, π = CME.π)
idx = [1; [3:2:NBases 2:2:NBases]'[:]]
CMEt = (α = αME[idx]', Q = QME')
D = jumpMatrixD(CME)
MEj = (α = αME*D, Q=CME.Q)

orbitME = orbit(t,CME)
orbitMEr = orbit(t,CMEr)
orbitMED = orbit(t,CME)*D^-1
orbitMEj = orbit(t,MEj,norm=2)
# orbitMEQ = (orbit(t,CME)*-CME.Q^-1)./sum(orbit(t,πME)*-CME.Q^-1,dims=2)
orbitπME = orbit(t,πME)
orbitπMEQ = (orbit(t,πME)*-CME.Q)./sum(orbit(t,πME)*-CME.Q,dims=2)
orbitπMED = orbit(t,πME)*D^-1
orbitπMEr = orbit(t,πMEr)
orbitCMEt = orbit(t,CMEt)
densityME = density(t,CME)
densityMEj = density(t,MEj)
densityπME = density(t,πME)
densityCMEt = density(t,CMEt)
intensityME = intensity(t,CME)
intensityπME = intensity(t,πME)
intensityCMEt = intensity(t,CMEt)
plot(t,orbitME,label = "fwd")
plot(t,orbitMED,label = "D")
plot(t,orbitπME,label = "π")
plot(t,orbitCMEt,label = "tr")
plot(orbitME[:,4],orbitME[:,5],label = "ME fwd orbit",legend=:outerright)
plot!(orbitMED[:,4],orbitMED[:,5],label = "MED fwd",legend=:outerright)
# plot!(orbitπMEQ[:,4],orbitπMEQ[:,5],label = "MEQ fwd",legend=:outerright)
# plot!(orbitMEQ[:,4],orbitMEQ[:,5],label = "MEQ fwd",legend=:outerright)
plot!(orbitMEr[:,4],orbitMEr[:,5],label = "MEr",legend=:outerright)
# plot!(orbitMEj[:,4],orbitMEj[:,5],label = "MEj",legend=:outerright)
plot!(orbitπME[:,4],orbitπME[:,5],label = "π orbit",legend=:outerright)
# plot!(orbitπMEr[:,4],orbitπMEr[:,5],label = "πr orbit",legend=:outerright)
plot!(orbitπMED[:,4],orbitπMED[:,5],label = "πD orbit",legend=:outerright)
plot!(orbitCMEt[:,4],orbitCMEt[:,5],label = "tr orbit",legend=:outerbottomright)
scatter!([CME.α[4]],[CME.α[5]],label = :false,series_annotations = ["     α"])
scatter!([(CME.α*D^-1)[4]],[(CME.α*D^-1)[5]],label = :false,series_annotations = ["     α"])
# scatter!([CMEt.α[4]],[CMEt.α[5]],label = :false,series_annotations = ["     α̃"])
scatter!([orbit(1,CME)[4]],[orbit(1,CME)[5]],label = :false, series_annotations = ["       t=1"])
scatter!([(orbit(0.1,CME))[4]],[(orbit(0.1,CME))[5]],label = :false, series_annotations = ["           t=0.1"])
scatter!([(orbit(1,CME)*D^-1)[4]],[(orbit(1,CME)*D^-1)[5]],label = :false, series_annotations = ["                A(t)D, t=1"])
scatter!([orbit(1,CMEt)[4]],[orbit(1,CMEt)[5]],label = :false, series_annotations = ["       t̃=1"])
# τ = 0.5
# scatter!([orbit(τ,CME)[2]],[orbit(τ,CME)[3]],label = :false, series_annotations = ["         t="*string(τ)])
# # scatter!([orbit(1,πME)[2]],[orbit(1,πME)[3]],label = :false, series_annotations = ["        t=1"])
# # scatter!([orbit(2.2885,πME)[2]],[orbit(2.2885,πME)[3]],label = :false, series_annotations = ["                                πexp(Qt)Q, t=2.2885"])
# tstar = orbit_zero(CME,orbit(1,CMEt))
# val = orbit(τ,CME)*exp(CME.Q*tstar)./sum(exp(CME.Q*tstar)[1,1])
# scatter!([val[2]],[val[3]],label = :false,
#         series_annotations = ["                         A(t)exp(Q) t=0.1"])
# Atplus = val./sum(val)
# scatter!([Atplus[2]],[Atplus[3]],label = :false,
#         series_annotations = ["                         A(t)exp(Q) t=0.1"])
#
# scatter!([orbit(2,CME)[2]],[orbit(2,CME)[3]],label = :false, series_annotations = ["       t=2"])
# # scatter!([orbit(0.1,CMEr)[2]],[orbit(0.1,CMEr)[3]],label = :false, series_annotations = ["          t̃=0.1"])
# # scatter!([orbit(2.05,CME)[2]],[orbit(2.05,CME)[3]], label = "t=2")
# # scatter!([CME.π[2]],[CME.π[3]],label = :false, series_annotations = ["      π"])

plot(t,intensityME,label = "ME")
plot!(t,intensityπME,label = "πME")
plot!(t,intensityCMEt,label = "CMEt")
#
plot(t,densityME,label = "ME")
plot!(t,densityπME,label = "πME")
plot!(t,densityCMEt,label = "CMEt")
plot!(t,densityMEj,label = "MEj")
# CCDF(t) = sum(CME.α*exp(CME.Q*t))
# plot!(t,CCDF.(t),label = "πME")
#
# S = Coxian.Q
# s = -sum(Coxian.Q,dims=2)
# α = Coxian.α
# m₁ = α*(-S)^-1
# π₁ = m₁./sum(m₁)
# m₂ = -π₁*S^-1
# ρ₁ = sum(m₁)
# α̂₁ = [s'*diagm(m₂[:]) zeros(1,order)]
# Ŝ₁ = [diagm(m₂[:])^-1*S'*diagm(m₂[:]) ρ₁^-1*diagm(m₂[:])^-1*diagm(m₁[:]);
#         zeros(order,order) diagm(m₁[:])^-1*S'*diagm(m₁[:])]
# Spreadbkwd = (α = α̂₁, Q = Ŝ₁)
#
# vecρ₁ = sum(-S^-1,dims=2)
# α₁ = [ρ₁^-1*α*diagm(vecρ₁[:]) zeros(1,order)]
# S₁ = [diagm(vecρ₁[:])^-1*S*diagm(vecρ₁[:]) diagm(vecρ₁[:])^-1;
#         zeros(order,order) S]
# Spread = (α = α₁, Q = S₁)
#
#
# plot(t,density(t,Coxian))
# plot!(t,density(t,Spreadbkwd))
# plot!(t,density(t,Spread))
#
# MEα₁ = [CME.α*S^-1/(sum(CME.α*S^-1)) 0*CME.α]
# MES₁ = [CME.Q -CME.Q; 0*CME.Q CME.Q]
# MEs₁ = sum(-MES₁,dims=1)
