import JLD2, LinearAlgebra, JSON

erlangDParams = Dict()
JLD2.@load pwd()*"/dev/erlangParamsData/erlangDParams.jld2" erlangDParams

JLD2.@load pwd()*"/dev/CMEParamsData/CMEParams1.jld2" tempDict
CMEParams = tempDict

"""
ME constructor method
    
    ME(
        a::Array{<:Real,2},
        S::Array{<:Real,2},
        s::Array{<:Real,1},
        D::Array{<:Real}=[0],
    )

Inputs: 
 - `a` a 1 by p Array of reals
 - `S` a p by p Array of reals
 - `s` a p by 1 Array of reals
 - `D` an optional argument, if empty then me.D is the identity, 
    else is a p by p matrix
 Throws an error if the dimensions are inconsistent.
"""
struct ME 
    a::Union{Array{<:Real,1},Array{<:Real,2}}
    S::Union{Array{<:Real,1},Array{<:Real,2}}
    s::Union{Array{<:Real,1},Array{<:Real,2}}
    D::Union{Array{<:Real,1},Array{<:Real,2}}
    f::Function
    F::Function
    function ME(
        a::Union{Array{<:Real,1},Array{<:Real,2}},
        S::Union{Array{<:Real,1},Array{<:Real,2}},
        s::Union{Array{<:Real,1},Array{<:Real,2}};
        D::Union{Array{<:Real,1},Array{<:Real,2}}=[0],
    )
    
        if D==[0]
            D = Array{Float64}(LinearAlgebra.I(size(S,1)))
        end
        s1 = size(a,1)
        s2 = size(a,2)
        s3 = size(S,1)
        s4 = size(S,2)
        s5 = size(s,1)
        s6 = size(s,2)
        s7 = size(D,1)
        s8 = size(D,2)
        test = (s1!=1) || (s6!=1) || any(([s2;s3;s4;s7;s8].-s5).!=0)
        if test
            error("Dimensions of ME representation not consistent")
        else
            f(x) = (a*exp(S*x)*s)[1]
            F(x) = 1-sum(a*exp(S*x))
            return new(a,S,s,D,f,F)
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
    display(params["D"])
    return SFFM.ME(α,Q,q;D=params["D"])
end

function MakeErlang(order; mean = 1)
    α = zeros(1,order) # inital distribution
    α[1] = 1
    λ = order/mean
    Q = zeros(order,order)
    Q = Q + LinearAlgebra.diagm(0=>repeat(-[λ],order), 1=>repeat([λ],order-1))
    q = -sum(Q,dims=2)
    D = Matrix(LinearAlgebra.I(order))[end:-1:1,:]
    return SFFM.ME(α,Q,q;D=D)
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
    return SFFM.ME(α,Q,q)
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
    return SFFM.ME(α,Q,q)
end

function reversal(me)
    α, Q = me.a, me.S
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

orbit(t,me::SFFM.ME; norm = 1) = begin
    orbits = zeros(length(t),length(me.a))
    for i in 1:length(t)
        num = me.a*exp(me.S*t[i])
        if norm == 1
            denom = sum(num)
        elseif norm == 2
            denom = exp(me.S[1,1]*t[i])#sum(num)
        else
            denom = 1
        end
        orbits[i,:] = num./denom
    end
    return orbits
end

function renewalProperties(me::SFFM.ME)
    density(t) = begin
        Q = me.S
        q = me.s
        α = me.a
        e = ones(size(q))
        (α*exp((Q+q*α)*t)*q)[1]
    end
    mean(t) = begin
        Q = me.S
        q = me.s
        α = me.a
        e = ones(size(q))
        temp1 = α*-Q^-1
        temp2 = temp1*e
        temp3 = temp1./temp2
        temp4 = Q + (q*α)
        ((t./temp2) - α*(I - exp(temp4*t))*(temp4 + e*temp3)^-1*q)[1]
    end
    ExpectedOrbit(t) = begin
        Q = me.S
        q = me.s
        α = me.a
        e = ones(size(q))
        (α*exp((Q + q*α)*t))[1]
    end
    return (density=density,mean=mean,ExpectedOrbit=ExpectedOrbit)
end
