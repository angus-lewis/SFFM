using JLD2

@load "/Users/a1627293/Documents/SFFM/src/CMEParamsData/CMEParams.jld2" CMEParams

CMEParams[1]["D"] = [1.0]
for i in 3:2:49
    @load pwd()*"/src/CMEParamsData/CMEParams_D_"*string(i)*".jld2" vals
    # @load pwd()*"/dev/CMEParamsData/CMEParams"*string(i)*".jld2" tempDict
    CMEParams[i]["D"] = convert.(Float64,vals["D"])
    CMEParams[i]["D_Evals"] = vals["k"]
    CMEParams[i]["D_time"] = vals["t"]
    # merge!(CMEParams,tempDict)
end

# open("dev/CMEParams.json","w") do f
#     JSON.print(f, CMEParams)
# end

# @save "dev/CMEParams_1_49.jld2" CMEParams
