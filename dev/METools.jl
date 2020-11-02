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
    Q = zeros(N,N)
    Q[1,1] = -1
    for k in 1:params["n"]
        kω = k*ω
        idx = 2*k:(2*k+1)
        Q[idx,idx] = [-1 -kω; kω -1]
    end
    Q = Q.*sum(-α*Q^-1)./mean
    q = -sum(Q,dims=2)
    return (α,Q,q)
end

function MakeErlang(order; mean = 1)
    α = zeros(1,order) # inital distribution
    α[1] = 1
    λ = order/mean
    Q = zeros(order,order)
    Q = Q + diagm(0=>repeat(-[λ],order), 1=>repeat([λ],order-1))
    q = -sum(Q,dims=2)
    return (α = α, Q = Q, q = q)
end

function MakeSomeCoxian(order; mean = 1)
    α = zeros(1,order) # inital distribution
    α[1] = 1
    λ = order/mean
    p = 0.95#1-0.5/NBases
    d = [repeat(-[0.5*λ],order÷2);repeat(-[2*λ],order-(order÷2))]
    Q = diagm(0=>d, 1=>-p*d[1:end-1])
    Q = Q.*sum(-α*Q^-1)./mean
    q = -sum(Q,dims=2)
    return (α = α, Q = Q, q = q)
end

function MakeSomePH(order; mean = 1)
    # α = collect(1:order)' # inital distribution
    # α = α./sum(α)
    # Q = repeat(1:order,1,order)
    # Q = Q - diagm(diag(Q))
    # Q = Q - diagm(sum(Q,dims=2)[:]) - diagm(order:-1:1)
    # Q = Q.*sum(-α*Q^-1)./mean
    # q = -sum(Q,dims=2)

    # α = collect(1:order)' # inital distribution
    # α = α./sum(α)
    # Q = repeat(1:order,1,order)
    # Q = Q - diagm(diag(Q))
    # Q = Q - diagm(sum(Q,dims=2)[:]) - diagm(1:order)
    # Q = Q.*sum(-α*Q^-1)./mean
    # q = -sum(Q,dims=2)

    # α = collect(1:order)' # inital distribution
    # α = α./sum(α)
    # Q = repeat(1:order,1,order)
    # Q = Q - diagm(diag(Q))
    # Q = Q - diagm(sum(Q,dims=2)[:]) - diagm(ones(order))
    # Q = Q.*sum(-α*Q^-1)./mean
    # q = -sum(Q,dims=2)

    # α = ones(order)' # inital distribution
    # α = α./sum(α)
    # Q = repeat(1:order,1,order)
    # Q = Q - diagm(diag(Q))
    # Q = Q - diagm(sum(Q,dims=2)[:]) - diagm(order:-1:1)
    # Q = Q.*sum(-α*Q^-1)./mean
    # q = -sum(Q,dims=2)

    # α = zeros(order)' # inital distribution
    # α[1] = 1
    # Q = repeat(1:order,1,order)
    # Q = Q - diagm(diag(Q))
    # Q = Q - diagm(sum(Q,dims=2)[:]) - diagm([zeros(order-1);1])
    # Q = Q.*sum(-α*Q^-1)./mean
    # q = -sum(Q,dims=2)

    α = zeros(order)' # inital distribution
    α[1] = 1
    Q = repeat(ones(order),1,order)
    Q = Q - diagm(diag(Q))
    Q = Q - diagm(sum(Q,dims=2)[:]) - diagm([zeros(order-1);1])
    Q = Q.*sum(-α*Q^-1)./mean
    q = -sum(Q,dims=2)
    return (α = α, Q = Q, q = q)
end

function reversal(PH)
    α, Q = PH
    πPH = -α*Q^-1
    μ = sum(πPH)
    πPH = πPH./μ
    P = diagm(πPH[:])
    if any(abs.(πPH).<1e-4)
        display(" ")
        display("!!!!!!!!")
        display("!!!!!!!!")
        display("!!!!!!!!")
        display(" ")
    end
    q = -sum(Q,dims=2)
    α = q'*P*μ
    Q = P^-1*Q'*P
    Q = Q.*sum(-α*Q^-1)./μ
    return (α = α, Q = Q, q = q, π = πPH)
end

function MakeGlobalApprox(;NCells = 3,up, down,T,C,bkwd=false)
    αup,Qup = up
    αdown,Qdown = down
    N₋ = sum(C.<=0)
    N₊ = sum(C.>=0)
    NPhases = length(C)
    NBases = length(αup)
    qup = -sum(Qup,dims=2)
    qdown = -sum(Qdown,dims=2)
    Q = zeros(NCells*NBases*NPhases,NCells*NBases*NPhases)
    for n in 1:NCells
        for i in 1:NPhases
            idx = (1:NBases) .+ (n-1)*(NBases*NPhases) .+ (i-1)*NBases
            if C[i]>0
                Q[idx,idx] = Qup*abs(C[i])
                if n<NCells
                    Q[idx,idx .+ NBases*NPhases] = qup*αup*abs(C[i])
                end
            elseif C[i]<0
                Q[idx,idx] = Qdown*abs(C[i])
                if n>1
                    Q[idx,idx .- NBases*NPhases] = qdown*αdown*abs(C[i])
                end
            end
        end
    end
    T₋₋ = T[C.<=0,C.<=0]
    T₊₋ = T[C.>=0,:].*((C.<0)')
    T₋₊ = T[C.<=0,:].*((C.>0)')
    T₊₊ = T[C.>=0,C.>=0]

    inLower = [kron(diagm(abs.(C).*(C.<0)),qdown)[:,C.<0]; zeros((NCells-1)*NPhases*NBases,N₋)]
    outLower = [kron(1,kron(T₋₊,αup)) zeros(N₋,N₊+(NCells-1)*NPhases*NBases)]
    inUpper = [zeros((NCells-1)*NPhases*NBases,N₊);kron(diagm(abs.(C).*(C.>0)),qup)[:,C.>0]]
    outUpper = [zeros(N₊,N₋+(NCells-1)*NPhases*NBases) kron(1,kron(T₊₋,αdown))]
    # display(inLower)
    # display(outLower)
    # display(inUpper)
    # display(outUpper)
    # display(NCells)
    # display(NPhases)
    # display(NBases)
    # display(N₊)
    # display(N₋)
    # display(kron(I(NCells),kron(T,I(NBases)))+Q)
    if bkwd
        # idx = [1; [3:2:NBases 2:2:NBases]'[:]]
        Tdiag = diagm(diag(T))
        Toff = T-diagm(diag(T))
        # πME = -αup*Qup^-1
        # μ = sum(πME)
        # πME = πME./μ
        I2 = I(NBases)[:,idx]#diagm(πME[:])#repeat(πME,length(πME),1)#
        B = [
            T₋₋ outLower;
            inLower kron(I(NCells),kron(Tdiag,I(NBases)))+kron(I(NCells),kron(Toff,I2))+Q inUpper;
            outUpper T₊₊;
        ]
    else
        B = [
            T₋₋ outLower;
            inLower kron(I(NCells),kron(T,I(NBases)))+Q inUpper;
            outUpper T₊₊;
        ]
    end
    # display(Q)
    return Q, B
end

orbit(t,ME) = begin
    orbits = zeros(length(t),NBases)
    for i in 1:length(t)
        num = ME.α*exp(ME.Q*t[i])
        denom = sum(num)
        orbits[i,:] = num./denom
    end
    return orbits
end
density(t,ME) = begin
    pdf = zeros(length(t),1)
    for i in 1:length(t)
        num = ME.α*exp(ME.Q*t[i])*sum(-ME.Q,dims=2)
        pdf[i] = sum(num)
    end
    return pdf
end
intensity(t,ME) = begin
    intensity = zeros(length(t),1)
    for i in 1:length(t)
        num = ME.α*exp(ME.Q*t[i])
        denom = sum(num)
        intensity[i] = sum(num*sum(-ME.Q,dims=2))/denom
    end
    return intensity
end
