# using LinearAlgebra, Plots, JSON
# include("../src/SFFM.jl")

tempCMEParams = Dict()
open("dev/iltcme.json", "r") do f
    global tempCMEParams
    tempCMEParams=JSON.parse(f)  # parse and transform data
end
CMEParams = Dict()
for n in keys(tempCMEParams)
    CMEParams[2*tempCMEParams[n]["n"]+1] = tempCMEParams[n]
end
CMEParams[1] = Dict(
  "n"       => 0,
  "c"       => 1,
  "b"       => Any[],
  "mu2"     => [],
  "a"       => Any[],
  "omega"   => 0,
  "phi"     => [],
  "mu1"     => [],
  "cv2"     => [],
  "optim"   => "full",
  "lognorm" => [],
)

function MakeME(params; mean = 1)
    N = 2*params["n"]+1
    α = zeros(1,N)
    α[1] = params["c"]
    a = params["a"]
    b = params["b"]
    ω =  params["omega"]
    for k in 1:params["n"]
        kω = k*ω
        α[2*k] = (1/2)*( a[k]*(1+kω) - b[k]*(1-kω) )/(1+kω^2)
        α[2*k+1] = (1/2)*( a[k]*(1-kω) + b[k]*(1+kω) )/(1+kω^2)
    end
    α = α./sum(α)
    A = zeros(N,N)
    A[1,1] = -1
    for k in 1:params["n"]
        kω = k*ω
        idx = 2*k:(2*k+1)
        A[idx,idx] = [-1 -kω; kω -1]
    end
    A = A.*sum(-α*A^-1)./mean
    a = -sum(A,dims=2)
    return (α,A,a)
end

# define SFM
T = [-2.0 2.0; 1.0 -1.0]
C = [1.0; -2.0]
fn(x) = [ones(size(x)) ones(size(x))]
Model = SFFM.MakeModel(;T=T,C=C,r=(r=fn,R=fn),Bounds=[0 10;-Inf Inf])
N₋ = sum(C.<=0)
N₊ = sum(C.>=0)
NPhases = length(C)

Δ = 1
NBases = 3
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

αME, QME, ~ = MakeME(CMEParams[NBases],mean=Δ)
CME = (α = αME, Q = QME, π = αME*QME^(-1)./sum(αME*QME^(-1)))
πME = (α = CME.π, Q = QME, π = CME.π)
idx = [1; [3:2:NBases 2:2:NBases]'[:]]
CMEt = (α = αME[idx]', Q = QME')

orbitME = orbit(t,CME)
orbitπME = orbit(t,πME)
orbitCMEt = orbit(t,CMEt)
densityME = density(t,CME)
densityπME = density(t,πME)
densityCMEt = density(t,CMEt)
intensityME = intensity(t,CME)
intensityπME = intensity(t,πME)
intensityCMEt = intensity(t,CMEt)
plot(t,orbitME,label = "fwd")
plot(t,orbitπME,label = "π")
plot(t,orbitCMEt,label = "tr")
plot(orbitME[:,2],orbitME[:,3],label = "ME fwd orbit",legend=:outerright)
plot!(orbitπME[:,2],orbitπME[:,3],label = "π orbit",legend=:outerright)
plot!(orbitCMEt[:,2],orbitCMEt[:,3],label = "tr orbit",legend=:outerright)
scatter!([CME.α[2]],[CME.α[3]],label = :false,series_annotations = ["     α"])
scatter!([CMEt.α[2]],[CMEt.α[3]],label = :false,series_annotations = ["     α̃"])
scatter!([orbit(1,CME)[2]],[orbit(1,CME)[3]],label = :false, series_annotations = ["       t=1"])
scatter!([orbit(1,CMEt)[2]],[orbit(1,CMEt)[3]],label = :false, series_annotations = ["       t̃=1"])
# scatter!([orbit(2.05,CME)[2]],[orbit(2.05,CME)[3]], label = "t=2")
scatter!([CME.π[2]],[CME.π[3]],label = :false, series_annotations = ["      π"])

plot(t,intensityME,label = "ME")
plot!(t,intensityπME,label = "πME")
plot!(t,intensityCMEt,label = "CMEt")

plot(t,densityME,label = "ME")
plot!(t,densityπME,label = "πME")
plot!(t,densityCMEt,label = "CMEt")
CCDF(t) = sum(CME.α*exp(CME.Q*t))
plot!(t,CCDF.(t),label = "πME")

S = Coxian.Q
s = -sum(Coxian.Q,dims=2)
α = Coxian.α
m₁ = α*(-S)^-1
π₁ = m₁./sum(m₁)
m₂ = -π₁*S^-1
ρ₁ = sum(m₁)
α̂₁ = [s'*diagm(m₂[:]) zeros(1,order)]
Ŝ₁ = [diagm(m₂[:])^-1*S'*diagm(m₂[:]) ρ₁^-1*diagm(m₂[:])^-1*diagm(m₁[:]);
        zeros(order,order) diagm(m₁[:])^-1*S'*diagm(m₁[:])]
Spreadbkwd = (α = α̂₁, Q = Ŝ₁)

vecρ₁ = sum(-S^-1,dims=2)
α₁ = [ρ₁^-1*α*diagm(vecρ₁[:]) zeros(1,order)]
S₁ = [diagm(vecρ₁[:])^-1*S*diagm(vecρ₁[:]) diagm(vecρ₁[:])^-1;
        zeros(order,order) S]
Spread = (α = α₁, Q = S₁)


plot(t,density(t,Coxian))
plot!(t,density(t,Spreadbkwd))
plot!(t,density(t,Spread))

MEα₁ = [CME.α*S^-1/(sum(CME.α*S^-1)) 0*CME.α]
MES₁ = [CME.Q -CME.Q; 0*CME.Q CME.Q]
MEs₁ = sum(-MES₁,dims=1)
