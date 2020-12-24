using JSON, LinearAlgebra, JLD2

tempCMEParams = Dict()
open("dev/CMEParamsData/iltcme.json", "r") do f
    global tempCMEParams
    tempCMEParams=JSON.parse(f)  # parse and transform data
end
CMEParams = Dict()
for n in keys(tempCMEParams)
    if !in(2*tempCMEParams[n]["n"]+1,keys(CMEParams))
        CMEParams[2*tempCMEParams[n]["n"]+1] = tempCMEParams[n]
    elseif tempCMEParams[n]["cv2"]<CMEParams[2*tempCMEParams[n]["n"]+1]["cv2"]
        CMEParams[2*tempCMEParams[n]["n"]+1] = tempCMEParams[n]
    end
end

k = 100_000_000
T=[]
N=[]
for n in keys(CMEParams)
    evals = Int(ceil(k/(n/3)))#Int(ceil((k^1.5*n/3)^-1.5))#
    display(n)
    display(evals)
    CMEParams[n]["D"], t = @timed integrateD(evals,CMEParams[n])
    CMEParams[n]["intDevals"] = evals
    display(t)
    push!(N,n)
    push!(T,t)
end

scatter(N,T)

CMEParams[1] = Dict(
  "n"       => 0,
  "c"       => 1,
  "b"       => Any[],
  "mu2"     => 1,
  "a"       => Any[],
  "omega"   => 1,
  "phi"     => [],
  "mu1"     => 1,
  "D"       => [1.0],
  "cv2"     => 1,
  "optim"   => "full",
  "lognorm" => [],
)

open("dev/CMEParams.json","w") do f
    JSON.print(f, CMEParams)
end

@save "dev/CMEParams.jld2" CMEParams

let
    CMEKeys = sort(collect(keys(CMEParams)))
    a = 0
    filecounter = 1
    tempDict = Dict()
    for key in CMEKeys
        a += key^2*8
        tempDict[key] = CMEParams[key]
        if a > 2e7
            open(pwd()*"/dev/CMEParamsData/CMEParams"*string(filecounter)*".json","w") do f
                JSON.print(f, tempDict)
            end
            @save pwd()*"/dev/CMEParamsData/CMEParams"*string(filecounter)*".jld2" tempDict
            tempDict = Dict()
            a = 0
            filecounter += 1
        end
    end
end
