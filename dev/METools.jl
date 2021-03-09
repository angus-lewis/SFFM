using JLD2, LinearAlgebra

# CMEParams = Dict()
# open("dev/CMEParams.json", "r") do f
#     global CMEParams
#     CMEParams=JSON.parse(f)  # parse and transform data
# end

@load pwd()*"/dev/CMEParamsData/CMEParams1.jld2" tempDict
CMEParams = tempDict

"""
ME constructor method
    
    ME(
        a::Array{<:Real,2},
        S::Array{<:Real,2},
        s::Array{<:Real,1},
    )

Inputs: 
 - `a` a 1 by p Array of reals
 - `S` a p by p Array of reals
 - `s` a p by 1 Array of reals
 Throws an error if the dimensions are inconsistent.
"""
struct ME 
    a::Array{<:Real,2}
    S::Array{<:Real,2}
    s::Array{<:Real,1}
    function ME(a,S,s) 
        s1 = size(a,1)
        s2 = size(a,2)
        s3 = size(S,1)
        s4 = size(S,2)
        s5 = size(s,1)
        s6 = size(s,2)
        test = (s1!=1) || (s6!=1) || any(([s2;s3;s4].-s5).!=0)
        if test
            error("Dimensions of ME representation not consistent")
        else
            return new(a,S,s)
        end
    end
end

"""

"""
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
    return (α = α, Q = Q, q = q)
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
    p = 1#1-0.5/NBases
    d = [repeat(-[0.5*λ],order÷2);repeat(-[2*λ],order-(order÷2))]
    Q = diagm(0=>d, 1=>-p*d[1:end-1])
    Q = Q.*sum(-α*Q^-1)./mean
    q = -sum(Q,dims=2)
    return (α = α, Q = Q, q = q)
end

function MakeSomePH(order; mean = 1)
    α = collect(1:order)' # inital distribution
    α = α./sum(α)
    Q = repeat(1:order,1,order)
    Q = Q - diagm(diag(Q))
    Q = Q - diagm(sum(Q,dims=2)[:]) - diagm(order:-1:1)
    Q = Q.*sum(-α*Q^-1)./mean
    q = -sum(Q,dims=2)

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

    # α = zeros(order)' # inital distribution
    # α[1] = 1
    # Q = repeat(ones(order),1,order)
    # Q = Q - diagm(diag(Q))
    # Q = Q - diagm(sum(Q,dims=2)[:]) - diagm([zeros(order-1);1])
    # Q = Q.*sum(-α*Q^-1)./mean
    # q = -sum(Q,dims=2)
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
    q = -sum(Q,dims=2)
    return (α = α, Q = Q, q = q, π = πPH)
end

function jumpMatrixD(PH)
    α, Q = PH
    πPH = -α*Q^-1
    μ = sum(πPH)
    πPH = πPH./μ
    U = zeros(size(Q))
    V = zeros(size(Q))
    for c in 1:size(Q,2)
        U[:,c] = Q^(c-1)*ones(size(πPH'))
        V[:,c] = Q'^(c-1)*πPH'
    end
    D = V*U^-1
    # display(D*Q*D^-1)
    D = diagm(πPH[:])^-1*D
    # display(D*Q*D^-1)
    # display(reversal(PH).Q)
    return (D)
end

function MakeGlobalApprox(;NCells = 3,up, down,T,C,bkwd=false,D=[],plusI = false)
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
        T₊₋ = T.*((C.>0)*(C.<0)')
        T₋₊ = T.*((C.<0)*(C.>0)')
        Tchange = T₊₋ + T₋₊
        Tnochange = T-Tchange
        # πME = -αup*Qup^-1
        # μ = sum(πME)
        # πME = πME./μ
        I2 = I(NBases)#[:,idx]#diagm(πME[:])#repeat(πME,length(πME),1)#
        D = zeros(size(Q))
        for c in 1:NCells, i in 1:length(C), j in 1:length(C)
            idxi = ((i-1)*NBases) .+ (1:NBases) .+ ((c-1)*NBases*NPhases)
            idxj = ((j-1)*NBases) .+ (1:NBases) .+ ((c-1)*NBases*NPhases)
            if i!=j
                if C[i]>0 && C[j]<0
                    # display(Qup+T[i,i]*I(NBases)*plusI)
                    # display(Qup)
                    πtemp = -αup*(abs(C[i])*Qup-T[i,i]*I(NBases)*plusI)^-1
                    πtemp = πtemp./sum(πtemp)
                    D[idxi,idxj] = T[i,j].*repeat(πtemp,NBases,1)
                elseif C[i]<0 && C[j]>0
                    πtemp = -αdown*(abs(C[i])*Qdown-T[i,i]*I(NBases)*plusI)^-1
                    πtemp = πtemp./sum(πtemp)
                    D[idxi,idxj] = T[i,j].*repeat(πtemp,NBases,1)
                else
                    D[idxi,idxj] = T[i,j].*I(NBases)
                end
            else
                D[idxi,idxj] = T[i,i].*I(NBases)
            end
        end
        B = [
            T₋₋ outLower;
            inLower D+Q inUpper;
            outUpper T₊₊;
        ]
        # B = [
        #     T₋₋ outLower;
        #     inLower (#kron(I(NCells),kron(Tdiag,I(NBases)))
        #         +kron(I(NCells),kron(Tnochange,I2))
        #         +kron(I(NCells),kron(T₋₊,D^-1))
        #         +kron(I(NCells),kron(T₊₋,D))
        #         +Q) inUpper;
        #     outUpper T₊₊;
        # ]
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

orbit(t,ME; norm = 1) = begin
    orbits = zeros(length(t),length(ME.α))
    for i in 1:length(t)
        num = ME.α*exp(ME.Q*t[i])
        if norm == 1
            denom = sum(num)
        elseif norm == 2
            denom = exp(ME.Q[1,1]*t[i])#sum(num)
        else
            denom = 1
        end
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

function orbit_zero(ME, target)
    err = 10
    a = 1
    b = 1.3
    c = (b+a)/2
    while err>sqrt(eps())
        Aa = orbit(a,ME)[1]
        Ab = orbit(b,ME)[1]
        Ac = orbit(c,ME)[1]
        if (sign(sum(Aa - target[1])) * sign(sum(Ac - target[1]))) < 0
            b = c
        else
            a = c
        end
        err = sum(abs.(Ac - target[1]))
        c = (a+b)/2
    end
    return c
end

function renewalProperties(ME)
    density(t) = begin
        Q = ME.Q
        q = ME.q
        α = ME.α
        e = ones(size(q))
        (α*exp((Q+q*α)*t)*q)[1]
    end
    mean(t) = begin
        Q = ME.Q
        q = ME.q
        α = ME.α
        e = ones(size(q))
        temp1 = α*-Q^-1
        temp2 = temp1*e
        temp3 = temp1./temp2
        temp4 = Q + (q*α)
        ((t./temp2) - α*(I - exp(temp4*t))*(temp4 + e*temp3)^-1*q)[1]
    end
    ExpectedOrbit(t) = begin
        Q = ME.Q
        q = ME.q
        α = ME.α
        e = ones(size(q))
        (α*exp((Q + q*α)*t))[1]
    end
    return (density=density,mean=mean,ExpectedOrbit=ExpectedOrbit)
end
