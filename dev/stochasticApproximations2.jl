# using LinearAlgebra, Plots, JSON
# include("../src/SFFM.jl")
include("METools.jl")

include("stochasticApproximations.jl")
p = plot!()

# # define SFM
# T = [-2.0 2.0; 1.0 -1.0]#[-2.0 2.0 0; 1.0 -2.0 1; 1 1 -2]
# C = [1.0; -2.0]#; -1]
# fn(x) = [ones(size(x)) ones(size(x))]# ones(size(x))]
# Model = SFFM.MakeModel(;T=T,C=C,r=(r=fn,R=fn),Bounds=[0 10;-Inf Inf])
# N₋ = sum(C.<=0)
# N₊ = sum(C.>=0)
# NPhases = length(C)
#
# t = 4.0
# τ = SFFM.FixedTime(T=t)
# NSim = 100_000
# sims = SFFM.SimSFM(Model=Model,StoppingTime=τ,InitCondition=(φ=2*ones(Int,NSim),X=zeros(NSim)))

function MakeGlobalApprox(;NCells = 3,up, down,T,C,bkwd=false,D=[],NCounters=2)
    αup,Qup = up
    αdown,Qdown = down
    N₋ = sum(C.<=0)
    N₊ = sum(C.>=0)
    NPhases = length(C)
    NBases = length(αup)
    qup = -sum(Qup,dims=2)
    qdown = -sum(Qdown,dims=2)
    Q = zeros(NCells*NBases*NPhases*NCounters,NCells*NBases*NPhases*NCounters)
    for n in 1:NCells
        for i in 1:NPhases
            for c in 1:NCounters
                tempmat = zeros(NBases,NBases)
                for nn in 1:(c-1)
                    phase = mod(1+i+nn,2)+1
                    tempmat += T[phase,phase].*I(NBases)
                end
                idx = (1:NBases) .+ (n-1)*(NBases*NPhases*NCounters) .+ (i-1)*NBases .+ (c-1)*NBases*NPhases
                if C[i]>0
                    Q[idx,idx] = (Qup+tempmat)*abs(C[i])
                    if n<NCells
                        qup = -sum((Qup+tempmat),dims=2)
                        Q[idx,(1:NBases) .+ (n)*NBases*NPhases*NCounters .+ NBases*(i-1)] = qup*αup*abs(C[i])
                    end
                elseif C[i]<0
                    Q[idx,idx] = (Qdown+tempmat)*abs(C[i])
                    if n>1
                        qdown = -sum((Qdown+tempmat),dims=2)
                        Q[idx,(n-2)*NBases*NPhases*NCounters .+ (1:NBases) .+ NBases*(i-1)] = qdown*αdown*abs(C[i])
                    end
                end
            end
        end
    end
    T₋₋ = T[C.<=0,C.<=0]
    T₊₋ = T[C.>=0,:].*((C.<0)')
    T₋₊ = T[C.<=0,:].*((C.>0)')
    T₊₊ = T[C.>=0,C.>=0]
    negidx = kron(
        ones(NCells*NCounters),
        kron((C.<0)[:],ones(NBases))
        )
    inLower = -sum(Q,dims=2).*negidx
    outLower = [kron(1,kron(T₋₊,αup)) zeros(N₋,N₊+(NCells-1)*NPhases*NBases*NCounters+(NCounters-1)*NPhases*NBases)]
    inUpper = -sum(Q,dims=2).*(1 .- negidx)
    outUpper = [zeros(N₊,N₋+(NCells-1)*NPhases*NBases*NCounters) kron(1,kron(T₊₋,αdown)) zeros(N₊,(NCounters-1)*NPhases*NBases)]
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
    for c in 1:NCells, i in 1:length(C), j in 1:length(C), n in 1:(NCounters)
        if n < NCounters
            idxi = (1:NBases) .+ (i-1)*NBases .+ (n-1)*NBases*NPhases .+ (c-1)*NBases*NPhases*NCounters
            idxj = (1:NBases) .+ (j-1)*NBases .+ (n)*NBases*NPhases .+ (c-1)*NBases*NPhases*NCounters
        else
            idxi = (1:NBases) .+ (i-1)*NBases .+ (n-1)*NBases*NPhases .+ (c-1)*NBases*NPhases*NCounters
            idxj = (1:NBases) .+ (j-1)*NBases .+ (n-1)*NBases*NPhases .+ (c-1)*NBases*NPhases*NCounters
        end
        # idxi = (n-1)*NBases .+ ((i-1)*NBases*NCounters) .+ (1:NBases) .+ ((c-1)*NBases*NPhases)
        # idxj = (n-1)*NBases .+ ((j-1)*NBases*NCounters) .+ (1:NBases) .+ ((c-1)*NBases*NPhases)
        if i!=j
            tempmat = zeros(NBases,NBases)
            for nn in 1:(n-1)
                phase = mod(1+i+nn,2)+1
                tempmat += T[phase,phase].*I(NBases)
            end
            if C[i]>0 && C[j]<0
                # display(Qup+T[i,i]*I(NBases)*plusI)
                # display(Qup)
                if n < NCounters
                    πtemp = -αup*(abs(C[i])*Qup+tempmat)^-1
                    πtemp = πtemp./sum(πtemp)
                    D[idxi,idxj] = T[i,j].*repeat(πtemp,NBases,1)
                else
                    πtemp = αup
                    D[idxi,idxj] = T[i,j].*repeat(πtemp,NBases,1)
                end
            elseif C[i]<0 && C[j]>0
                if n < NCounters
                    πtemp = -αdown*(abs(C[i])*Qdown+tempmat)^-1
                    πtemp = πtemp./sum(πtemp)
                    D[idxi,idxj] = T[i,j].*repeat(πtemp,NBases,1)
                else
                    πtemp = αdown
                    D[idxi,idxj] = T[i,j].*repeat(πtemp,NBases,1)
                end
            else
                D[idxi,idxj] = T[i,j].*I(NBases)
            end
        else
                if n < NCounters
                    D[idxi,idxj.-NBases*NPhases] = T[i,i].*I(NBases)
                else
                    D[idxi,idxj] = T[i,i].*I(NBases)
                end
        end
    end
    B = [
        T₋₋ outLower;
        inLower D+Q inUpper;
        outUpper T₊₊;
    ]
    return Q, B
end

for NCounters in 2:2
    globalerrME = []
    globalerrDG = []
    for NBases in 1:2:21
        # NBases = 2
        Nodes = collect(0:Δ:10)
        NCells = length(Nodes)-1
        Erlang = MakeME(CMEParams[NBases], mean = Δ)#MakeErlang(NBases,mean=Δ)#
        display(Erlang.Q)
        Q, B = MakeGlobalApprox(
            NCells = NCells,
            up = Erlang,
            down = Erlang,
            T = T,
            C = C,
            NCounters = NCounters,
        )
        display(B)
        display(Q)

        DGMesh = SFFM.MakeMesh(Model=Model,NBases=1,Nodes=collect(Nodes[1]:Δ/NBases:Nodes[end]),Basis="lagrange")
        All = SFFM.MakeAll(Model=Model,Mesh=DGMesh)

        initDist = zeros(1,size(All.B.B,1))
        initDist[1] = 1

        temp = initDist*exp(Matrix(All.B.B)*t)#SFFM.EulerDG(D=All.B.B,y=t,x0=initDist)#
        DGdist_t = SFFM.Coeffs2Dist(Model=Model,Mesh=DGMesh,Coeffs=temp,type="probability")

        initDist = zeros(1,size(B,1))
        initDist[1] = 1
        dist_t = initDist*exp(B*t)#SFFM.EulerDG(D=B,y=t,x0=initDist)#
        pm_t = dist_t[[1:N₋;(end-N₊+1):end]]
        dist_t = sum(reshape(dist_t[N₋+1:end-N₊],NBases,NCounters*NPhases*NCells),dims=1)
        dist_t = sum(reshape(dist_t,NPhases,NCounters,NCells),dims=2)
        Mesh = SFFM.MakeMesh(Model=Model,NBases=NBases,Nodes=Nodes,Basis="lagrange")
        simDist = SFFM.Sims2Dist(Model=Model,Mesh=Mesh,sims=sims,type="probability")

        localerrME = sum(abs.(pm_t-simDist.pm))
        localerrDG = sum(abs.(DGdist_t.pm-simDist.pm))

        # q = plot(layout = (2,1))
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
                yvalsME = [sum(dist_t[i,1,n])]
                yvalsDG = [sum(DGdist_t.distribution[:,(1:NBases).+NBases*(n-1),i])]
                # scatter!([x],yvalsDG,label=label1,subplot=i,color=:blue,markershape=:rtriangle)
                # scatter!([x],yvalsME,label=label2,subplot=i,color=:black,markershape=:ltriangle)
                localerrME += abs(sum(yvalsME-simDist.distribution[:,n,i]))
                localerrDG += abs(sum(yvalsDG-simDist.distribution[:,n,i]))
            end
        end
        # display(q)
        push!(globalerrME,localerrME)
        push!(globalerrDG,localerrDG)
        # for i in 1:NPhases
        #     scatter!(simDist.x,simDist.distribution[:,:,i][:],subplot=i,label="sim")
        # end
        # display(plot!())
    end
    plot!(p,1:2:21,log.(globalerrME),label=false,linestyle=:dash,colour=1+NCounters)
    plot!(p,1:2:21,log.(globalerrDG),label=false,linestyle=:solid,colour=3)
    display(p)
end
