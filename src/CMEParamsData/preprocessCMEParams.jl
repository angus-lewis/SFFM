using JSON, LinearAlgebra, JLD2, BenchmarkTools, GLM, Distributed
println("nthreads ", Threads.nthreads())
println("nprocs ", nprocs())
println("nworkers ", nworkers())
# read the data from json
tempCMEParams = Dict()
open("iltcme.json", "r") do f
    global tempCMEParams
    tempCMEParams=JSON.parse(f)  # parse and transform data
end

# put the parameters in CMEParams with keys corresponding the ME order
CMEParams = Dict()
for n in keys(tempCMEParams)
    if 2*tempCMEParams[n]["n"]+1 < 51
        if !in(2*tempCMEParams[n]["n"]+1,keys(CMEParams)) # if no already in dict add it
            CMEParams[2*tempCMEParams[n]["n"]+1] = tempCMEParams[n]
        elseif tempCMEParams[n]["cv2"]<CMEParams[2*tempCMEParams[n]["n"]+1]["cv2"]
            # if its already in there, add only if it has the smallest CV
            CMEParams[2*tempCMEParams[n]["n"]+1] = tempCMEParams[n]
        end
    end
end

# a function to numerically approximate D
function integrateD(
    evals::Int64,
    p_n::Int64,
    p_a::Array{Float64,1},
    p_b::Array{Float64,1},
    p_c::Float64,
    p_omega::Float64,
)#,params::Dict{String,Any})
    # evals is an integer specifying how many points to eval the function at
    # params is a CMEParams dictionary entry, i.e. CMEParams[3]
    N = 2*p_n+1 # ME order

    α = zeros(BigFloat,N)
    α[1] = BigFloat(p_c)
    a = BigFloat.(p_a)
    b = BigFloat.(p_b)
    ω =  BigFloat.(p_omega)
    for k in 1:p_n
        kω = k*ω
        α[2*k] = (1/2)*( a[k]*(1+kω) - b[k]*(1-kω) )/(1+kω^2)
        α[2*k+1] = (1/2)*( a[k]*(1-kω) + b[k]*(1+kω) )/(1+kω^2)
    end

    period = 2*π/ω # the orbit repeats after this time
    edges = BigFloat.(range(0,period,length=evals+1)) # points at which to evaluate the fn
    h = BigFloat(period/(evals))

    orbit_LHS = α
    orbit_RHS = zeros(BigFloat,N)
    v_RHS = zeros(BigFloat,N)
    v_RHS[1] = 1
    v_LHS = 0.5.*ones(BigFloat,N)
    D = zeros(BigFloat,N,N)
    
    idxVec_1 = 2 .* (1:p_n)
    idxVec_2 = idxVec_1 .+ 1
    for t in edges[2:end]
        orbit_RHS[1] = α[1]
        ωt = ω*t
        for k in 1:p_n
            kωt = k*ωt
            temp_cos = cos(kωt)
            temp_sin = sin(kωt)
            orbit_RHS[idxVec_1[k]] = α[idxVec_1[k]]*temp_cos + α[idxVec_2[k]]*temp_sin
            orbit_RHS[idxVec_2[k]] = -α[idxVec_1[k]]*temp_sin + α[idxVec_2[k]]*temp_cos
            v_RHS[idxVec_1[k]] = temp_cos - temp_sin
            v_RHS[idxVec_2[k]] = temp_sin + temp_cos
        end
        orbit_RHS = orbit_RHS./(2*sum(orbit_RHS))
        orbit = orbit_LHS+orbit_RHS

        v_RHS = exp(-t)*v_RHS
        v = v_LHS - v_RHS

        Dᵢ = v*orbit'
        D += Dᵢ

        orbit_LHS = copy(orbit_RHS)
        v_LHS = copy(v_RHS)
    end
    D = (1/(1-exp(-period)))*D
    return D
end

# 19.0   2.23564
# 21.0   2.68764
# 25.0   3.72726
# 30.0   5.28104
# 40.0   9.23611
# 50.0  14.3212

# numerical approximation of D
# T=Float64[]
# N=Float64[]
# # for n in [3,5,7,9,11,13]#keys(CMEParams)
# #     println(n)
# #     # display(evals)

# #     t = @benchmark integrateD(
# #         k,
# #         CMEParams[$n]["n"],
# #         Float64.(CMEParams[$n]["a"]),
# #         Float64.(CMEParams[$n]["b"]),
# #         CMEParams[$n]["c"],
# #         CMEParams[$n]["omega"],
# #     )
# #     # CMEParams[n]["D"], t = @timed integrateD(
# #     #     k,
# #     #     CMEParams[n]["n"],
# #     #     Float64.(CMEParams[n]["a"]),
# #     #     Float64.(CMEParams[n]["b"]),
# #     #     CMEParams[n]["c"],
# #     #     CMEParams[n]["omega"],
# #     # )
# #     println(mean(t).time)
# #     push!(N,n)
# #     push!(T,mean(t).time)
# # end

# # T = T.*10^-9
# # X = Array{Float64,2}(undef,6,2)
# # X[:,1] .= 1
# # X[:,2] .= N

# # lm1 = lm(X.^2,T)
# # newX = [1 19^2; 1 21^2; 1 25^2; 1 30^2; 1 40^2; 1 50.0^2]

# # println()
# # display(hcat( sqrt.(newX[:,2]), predict(lm1,newX)*1_000_000_000/k/60/60/24))

# numerical approximation of D
n = 2*(parse(Int,ARGS[1])+1) + 1 
if n < 20 
    k = 1_000_000_000
else
    k = 100_000_000
end
# for n in keys(CMEParams)
println("n=",n)
println("k=",k)
D, t = @timed integrateD(
    k,
    CMEParams[n]["n"],
    Float64.(CMEParams[n]["a"]),
    Float64.(CMEParams[n]["b"]),
    CMEParams[n]["c"],
    CMEParams[n]["omega"],
)
println("t=",t)
# end

vals = Dict("D"=>D, "t"=>t, "k"=>k, "n"=>n)

open("CMEParams_D_"*string(n)*".json","w") do f
    JSON.print(f, vals)
end

@save "CMEParams_D_"*string(n)*".jld2" vals

# let
#     CMEKeys = sort(collect(keys(CMEParams)))
#     a = 0
#     filecounter = 1
#     tempDict = Dict()
#     for key in CMEKeys
#         a += key^2*8
#         tempDict[key] = CMEParams[key]
#         if a > 2e7
#             open(pwd()*"/dev/CMEParamsData/CMEParams"*string(filecounter)*".json","w") do f
#                 JSON.print(f, tempDict)
#             end
#             @save pwd()*"/dev/CMEParamsData/CMEParams"*string(filecounter)*".jld2" tempDict
#             tempDict = Dict()
#             a = 0
#             filecounter += 1
#         end
#     end
# end
