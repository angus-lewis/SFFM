include(pwd()*"/src/SFFM.jl")
using Plots

## define 1D wave equation model 
T = [0.0]
C = [1]

rfun(x) = x.*0
Rfun(x) = r(x)

r = (
    r = function (x)
        rfun(x)
    end,
    R = function (x)
        Rfun(x)
    end
)

bounds = [0 12; -Inf Inf]
model = SFFM.Model( T, C, r, Bounds = bounds)

## approximation stuff
order = 15 # must be odd 
Δ = 1
me = SFFM.MakeME(SFFM.CMEParams[order], mean = Δ)
nodes = collect(0:Δ:12)
mesh = SFFM.DGMesh(model, nodes, order)

## Initial condition 
λ = -0.25# me.S[1,1]
f(x::Real) = sin(3*pi*x)#-λ*exp(λ*x)/(1-exp(λ*10))#1/12 #

## approximate integral of A(x)f(x)dx on [xₖ, xₖ₊₁]
nevals = 10000
params = SFFM.CMEParams[order]

α = zeros(order)
α[1] = params["c"]
a = params["a"]
b = params["b"]
ω =  params["omega"]
for k in 1:params["n"]
    kω = k*ω
    α[2*k] = (1/2)*( a[k]*(1+kω) - b[k]*(1-kω) )/(1+kω^2)
    α[2*k+1] = (1/2)*( a[k]*(1-kω) + b[k]*(1+kω) )/(1+kω^2)
end

A = zeros(Float64,SFFM.NIntervals(mesh),order)
for m in 1:SFFM.NIntervals(mesh)
    aInt = zeros(Float64,order)

    orbit_LHS = α
    orbit_RHS = zeros(order)
    x = mesh.Nodes[m]
    h = Δ/nevals
    for n in 2:nevals
        orbit_RHS[1] = α[1]
        x += h 
        for k in 1:params["n"]
            kωx = k*ω*((n-1)*h).*-me.S[1,1]
            idx = 2*k
            idx2 = idx+1
            temp_cos = cos(kωx)
            temp_sin = sin(kωx)
            orbit_RHS[idx] = α[idx]*temp_cos + α[idx2]*temp_sin
            orbit_RHS[idx2] = -α[idx]*temp_sin + α[idx2]*temp_cos
        end
        orbit_RHS = orbit_RHS./sum(orbit_RHS)
        orbit = (orbit_LHS+orbit_RHS)./2
        
        aInt += orbit.*h.*(f(x)+f(x-h))/2

        orbit_LHS = copy(orbit_RHS)
    end
    A[m,:] = aInt
end

xvec = collect(mesh.Nodes[1]:0.05:mesh.Nodes[end])
p = plot(xvec,f.(xvec),label = "f")

xvec = collect(0:0.05:Δ)
for k in 1:SFFM.NIntervals(mesh)
    mek = SFFM.ME(Matrix(A[k,:]'),me.S,me.s,D = me.D)
    thePDF = SFFM.pdf(mek,xvec)
    thePDFplusΔ = SFFM.pdf(mek,xvec.+Δ)[end:-1:1]
    plot!(mesh.Nodes[k+1] .- xvec, thePDF+thePDFplusΔ, label = "A")
end
p