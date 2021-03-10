using JLD2, JSON
include("../../src/SFFM.jl")
@load pwd()*"/dev/CMEParamsData/CMEParams1.jld2" tempDict
CMEParams = tempDict

erlangDParams = Dict()

function integrateD(evals,order)
    # evals is an integer specifying how many points to eval the function at
    # order is the order of the erlang distribution
    
    erlang = SFFM.MakeErlang(order)
    λ = -erlang.S[1,1]
    
    edges = range(0,1+20/order,length=evals+1) # points at which to evaluate the fn
    h = (4/order)/(evals)

    orbit_LHS = erlang.a'
    orbit_RHS = zeros(order)
    v_RHS = zeros(order)
    v_RHS[1] = 1
    v_LHS = ones(order)
    D = zeros(order,order)
    fct = factorial.(big.(0:order-1))
    poly(x) = ((λ*x).^(0:order-1))./fct
    for t in edges[2:end]
        vec = poly(t)
        orbit_RHS = vec./sum(vec)
        v_RHS = exp(-λ*t).*(cumsum(vec)[end:-1:1])
        orbit = (orbit_LHS+orbit_RHS)./2
        
        v = v_LHS - v_RHS

        Dᵢ = v*orbit'
        D += Dᵢ

        orbit_LHS = copy(orbit_RHS)
        v_LHS = copy(v_RHS)
    end
    return D
end

# numerical approximation of D
k = 100_000_000
T=[]
N=[]
for n in keys(CMEParams)
    evals = Int(ceil(k/(n/3)))# number of function evals
    display(n)
    display(evals)
    erlangDParams[n] = integrateD(evals,n)
    push!(N,n)
end

open("dev/erlangParamsData/erlangDParams.json","w") do f
    JSON.print(f, erlangDParams)
end

@save "dev/erlangParamsData/erlangDParams.jld2" erlangDParams

# let
#     CMEKeys = sort(collect(keys(CMEParams)))
#     a = 0
#     filecounter = 1
#     tempDict = Dict()
#     for key in CMEKeys
#         a += key^2*8
#         tempDict[key] = CMEParams[key]
#         if a > 2e7
#             open(pwd()*"/dev/erlangParamsData/erlangDParams"*string(filecounter)*".json","w") do f
#                 JSON.print(f, tempDict)
#             end
#             @save pwd()*"/dev/erlangParamsData/erlangDParams"*string(filecounter)*".jld2" tempDict
#             tempDict = Dict()
#             a = 0
#             filecounter += 1
#         end
#     end
# end