CMEParams = Dict()
for i in 1:26
    @load pwd()*"/dev/CMEParamsData/CMEParams"*string(i)*".jld2" tempDict
    merge!(CMEParams,tempDict)
end

open("dev/CMEParams.json","w") do f
    JSON.print(f, CMEParams)
end

@save "dev/CMEParams.jld2" CMEParams
