include(pwd()*"/src/SFFM.jl")
include(pwd()*"/examples/meNumerics/discontinuitiesModelDef.jl")

Δ = 1 
nodes = collect(0:Δ:bounds[1,2])
mesh = SFFM.MakeMesh(
    model = model, 
    Nodes = nodes, 
    NBases = 1,
    Basis = "lagrange",
)

order = 4
nodes = mesh.Nodes[1:order]
evalPt = (nodes[3]+nodes[2])/2
polyCoefs = zeros(order)
for n in 1:order
    notn = [1:n-1;n+1:order]
    polyCoefs[n] = prod(evalPt.-nodes[notn])./prod(nodes[n].-nodes[notn])
end
