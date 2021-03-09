# using LinearAlgebra, Plots, JSON
# include("../src/SFFM.jl")
include("METools.jl")
T = [-2.0 2.0; 1.0 -1.0]#[-2.0 2.0 0; 1.0 -2.0 1; 1 1 -2]
C = [1.0; -2.0]#; -1]
fn(x) = [ones(size(x)) ones(size(x))]# ones(size(x))]
model = SFFM.Model(;T=T,C=C,r=(r=fn,R=fn),Bounds=[0 10;-Inf Inf])
N₋ = sum(C.<=0)
N₊ = sum(C.>=0)
NPhases = length(C)

t = 4.0
τ = SFFM.FixedTime(T=t)
NSim = 100_000
sims = SFFM.SimSFM(model=model,StoppingTime=τ,InitCondition=(φ=2*ones(Int,NSim),X=zeros(NSim)))

function MakeGlobalApprox(;NCells = 3,up, down,T,C,bkwd=false,D=[],plusI = false)
    αup,Qup,qup = up
    αdown,Qdown,qdown = down
    N₋ = sum(C.<=0)
    N₊ = sum(C.>=0)
    NPhases = length(C)
    NBases = length(αup)
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

    Tdiag = diagm(diag(T))
    Toff = T-diagm(diag(T))

    G = kron(I(NCells),kron(Tdiag,I(NPhases*NBases÷2))) +
        kron(I(NCells),kron(kron(Toff,I(2)[end:-1:1,:]),I(NBases÷2)))
        display(size(G))
        display(size(Q))
    B = [
        T₋₋ outLower;
        inLower G+Q inUpper;
        outUpper T₊₊;
    ]

    return Q, B
end

let
    globalerrME = []
    globalerrDG = []
    BasesVec = 1:2:21
    for NBases in BasesVec
        Δ = 1
        # NBases = 2
        Nodes = collect(0:Δ:10)
        NCells = length(Nodes)-1
        Erlang = MakeErlang(NBases,mean=Δ)#MakeME(CMEParams[NBases], mean = Δ)#
        AugQ = [Erlang.Q 0*Erlang.Q;
            0*Erlang.Q Erlang.Q']
        # AugQ = [Erlang.Q 0*Erlang.Q;
        #     0*Erlang.Q (Erlang.Q-diagm(diag(Erlang.Q)))']
        # AugQ = [Erlang.Q 0*Erlang.Q;
        #     0*Erlang.Q -Erlang.Q]
        # AugQ[NBases+1,NBases+1] = 0
        rev = reversal(Erlang)
        AugQ = [Erlang.Q 0*Erlang.Q;
            0*Erlang.Q rev.Q+rev.q*rev.α]
        # Augα = [Erlang.α 0*Erlang.α*exp(Erlang.Q*Δ)./sum(Erlang.α*exp(Erlang.Q*Δ))]
        Augα = [Erlang.α rev.α]
        Augmented = (
            α = Augα,
            Q = AugQ,
            q = [-sum(Erlang.Q,dims=2); zeros(size(Erlang.Q,1))]
            )

        Q, B = MakeGlobalApprox(
            NCells = NCells,
            up = Augmented,
            down = Augmented,
            T = T,
            C = C,
        )
        if NBases<6
            display(Erlang.Q)
            display(B)
        end
        # display(Q)

        DGMesh = SFFM.MakeMesh(model=model,NBases=1,Nodes=collect(Nodes[1]:Δ/NBases:Nodes[end]),Basis="lagrange")
        All = SFFM.MakeAll(model=model,mesh=DGMesh)

        initDist = zeros(1,size(All.B.B,1))
        initDist[1] = 1

        temp = initDist*exp(Matrix(All.B.B)*t)#SFFM.EulerDG(D=All.B.B,y=t,x0=initDist)#
        DGdist_t = SFFM.Coeffs2Dist(model=model,mesh=DGMesh,Coeffs=temp,type="probability")

        initDist = zeros(1,size(B,1))
        initDist[1] = 1
        dist_t = initDist*exp(B*t)#SFFM.EulerDG(D=B,y=t,x0=initDist)#
        display(dist_t)
        display(DGdist_t.distribution)
        pm_t = dist_t[[1:N₋;(end-N₊+1):end]]
        manipulate(dist_t) = begin
            temp = zeros(NCells*NPhases)
            for n in 1:NBases
                temp += dist_t[(N₋+n):(2*NBases):(end-N₊)]
            end
            dist_t = reshape(temp,NPhases,NCells)
            return dist_t
        end
        dist_t = manipulate(dist_t)

        mesh = SFFM.MakeMesh(model=model,NBases=NBases,Nodes=Nodes,Basis="lagrange")
        simDist = SFFM.Sims2Dist(model=model,mesh=mesh,sims=sims,type="probability")

        localerrME = sum(abs.(pm_t-simDist.pm))
        localerrDG = sum(abs.(DGdist_t.pm-simDist.pm))

        q = plot(layout = (2,1))
        for n in 1:NCells
            x = simDist.x[n]
            for i in 1:NPhases
                if n==1
                    label1 = "DG"
                    label2 = "ME"
                else
                    label1 = false
                    label2 = false
                end
                yvalsME = [sum(dist_t[i,n])]
                yvalsDG = [sum(DGdist_t.distribution[:,(1:NBases).+NBases*(n-1),i])]
                scatter!([x],yvalsDG,label=label1,subplot=i,color=:blue,markershape=:rtriangle)
                scatter!([x],yvalsME,label=label2,subplot=i,color=:black,markershape=:ltriangle)
                localerrME += abs(sum(yvalsME-simDist.distribution[:,n,i]))
                localerrDG += abs(sum(yvalsDG-simDist.distribution[:,n,i]))
            end
        end
        display(q)
        push!(globalerrME,localerrME)
        push!(globalerrDG,localerrDG)
        # for i in 1:NPhases
        #     scatter!(simDist.x,simDist.distribution[:,:,i][:],subplot=i,label="sim")
        # end
        # display(plot!())
    end
    p = plot(BasesVec,log.(globalerrME),label=false,linestyle=:dash,colour=1)
    plot!(p,BasesVec,log.(globalerrDG),label=false,linestyle=:solid,colour=3)
    display(p)
end
