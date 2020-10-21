# using LinearAlgebra, Plots, JSON
# include("../src/SFFM.jl")

CMEParams = Dict()
open("dev/iltcme.json", "r") do f
    global CMEParams
    CMEParams=JSON.parse(f)  # parse and transform data
end

function MakeME(;params, mean = 1)
    n = params["n"]
    if mod(n,2)==0
        display("[!!!ERROR!!!] n must be odd")
    end
    α = zeros(1,n)
    α[1] = params["c"]
    a = params["a"]
    b = params["b"]
    ω =  params["omega"]
    for k in 2:n
        kω = k*ω
        if mod(k,2)==0
            α[k] = (1/2)*(a[k]*(1+kω)-b[k]*(1-kω))/(1+kω^2)
        else
            α[k] = (1/2)*(a[k]*(1-kω)+b[k]*(1+kω))/(1+kω^2)
        end
    end
    α = α./sum(α)
    A = zeros(n,n)
    A[1,1] = -1
    for k in 2:2:n
        kω = k*ω
        idx = k:(k+1)
        A[idx,idx] = [-1 -kω; kω -1]
    end
    A = A.*sum(-α*A^-1)./mean
    a = -sum(A,dims=2)
    return (α,A,a)
end

# define SFM
T = [-2.0 2.0; 1.0 -1.0]
C = [1.0; -2.0]
fn(x) = [ones(size(x)) ones(size(x))]
Model = SFFM.MakeModel(;T=T,C=C,r=(r=fn,R=fn),Bounds=[0 10;-Inf Inf])
N₋ = sum(C.<=0)
N₊ = sum(C.>=0)
NPhases = length(C)

# construct global approximation
function MakeGlobalApprox(;NCells = 4,αup,Qup,αdown,Qdown,T,C)
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
    T₊₋ = T[C.>=0,C.<0]
    T₋₊ = T[C.<=0,C.>0]
    T₊₊ = T[C.>=0,C.>=0]

    inLower = [kron(diagm(abs.(C).*(C.<0)),qdown)[:,C.<0]; zeros((NCells-1)*NPhases*NBases,N₋)]
    outLower = [kron((C.>0)',kron(T₋₊,αup)) zeros(N₋,N₊+(NCells-1)*NPhases*NBases)]
    inUpper = [zeros((NCells-1)*NPhases*NBases,N₊);kron(diagm(abs.(C).*(C.>0)),qdown)[:,C.>0]]
    outUpper = [zeros(N₊,N₋+(NCells-1)*NPhases*NBases) kron((C.<0)',kron(T₊₋,αdown))]
    B = [
        T₋₋ outLower;
        inLower kron(I(NCells),kron(T,I(NBases)))+Q inUpper;
        outUpper T₊₊;
    ]
    return Q, B
end

τ = SFFM.FixedTime(T=t)
NSim = 100_000
sims = SFFM.SimSFM(Model=Model,StoppingTime=τ,InitCondition=(φ=2*ones(Int,NSim),X=zeros(NSim)))

let
    vecNBases = [1,3,5,7,11,15,21,29]
    vecΔ = [5 2.5 1.25 1.25/2 1.25/4]
    errfwdPH = vecNBases
    errfwdME = vecNBases
    errDG = vecNBases
    plot()
    c=0
    for Δ in vecΔ
        c+=1
        Nodes = collect(0:Δ:10)
        globalerrfwdPH = []
        globalerrfwdME = []
        globalerrDG = []
        for NBases in vecNBases
            Mesh = SFFM.MakeMesh(Model=Model,NBases=NBases,Nodes=Nodes,Basis="lagrange")
            simDist = SFFM.Sims2Dist(Model=Model,Mesh=Mesh,sims=sims,type="probability")

            # define generator for up approximation
            αup = zeros(1,NBases) # inital distribution
            αup[1] = 1
            λ = NBases/Δ
            Qup = zeros(NBases,NBases)
            Qup = Qup + diagm(0=>repeat(-[λ],NBases), 1=>repeat([λ],NBases-1))

            # define generator for down approximaton
            αdown = zeros(1,NBases) # inital distribution
            αdown[1] = 1
            Qdown = Qup

            NCells = length(Nodes)-1
            Q, B = MakeGlobalApprox(;
                NCells = NCells,
                αup = αup,
                Qup = Qup,
                αdown = αdown,
                Qdown = Qdown,
                T = T,
                C = C,
            )

            αupME, QupME, ~ = MakeME(params=CMEParams[NBases],mean=Δ)
            αdownME, QdownME, ~ = MakeME(params=CMEParams[NBases],mean=Δ)
            QME, BME = MakeGlobalApprox(;
                NCells = NCells,
                αup = αupME,
                Qup = QupME,
                αdown = αdownME,
                Qdown = QdownME,
                T = T,
                C = C,
            )

            DGMesh = SFFM.MakeMesh(Model=Model,NBases=1,Nodes=collect(Nodes[1]:Δ/NBases:Nodes[end]),Basis="lagrange")
            All = SFFM.MakeAll(Model=Model,Mesh=DGMesh)

            initDist = zeros(1,size(B,1))
            initDist[1] = 1

            t = 4

            temp = initDist*exp(Matrix(All.B.B)*t)
            DGdist_t = SFFM.Coeffs2Dist(Model=Model,Mesh=DGMesh,Coeffs=temp,type="probability")

            dist_t = initDist*exp(B*t)
            pm_t = dist_t[[1:N₋;(end-N₊+1):end]]
            dist_t = reshape(dist_t[N₋+1:end-N₊],NBases,NPhases,NCells)

            distME_t = initDist*exp(BME*t)
            pmME_t = distME_t[[1:N₋;(end-N₊+1):end]]
            distME_t = reshape(distME_t[N₋+1:end-N₊],NBases,NPhases,NCells)

            plot(layout = (NPhases,1))
            localerrfwdPH = sum(abs.(pm_t-simDist.pm))
            localerrfwdME = sum(abs.(pmME_t-simDist.pm))
            localerrDG = sum(abs.(DGdist_t.pm-simDist.pm))
            for n in 1:NCells
                x = simDist.x[n]
                for i in 1:NPhases
                    if n==1
                        label1 = "fwdPH"
                        label2 = "DG"
                        label3 = "fwdME"
                    else
                        label1 = false
                        label2 = false
                        label3 = false
                    end
                    yvalsPH = [sum(dist_t[:,i,n])]
                    scatter!([x],yvalsPH,label=label1,subplot=i,color=:red,markershape=:x)
                    yvalsDG = [sum(DGdist_t.distribution[:,(1:NBases).+NBases*(n-1),i])]
                    scatter!([x],yvalsDG,label=label2,subplot=i,color=:blue,markershape=:rtriangle)
                    yvalsME = [sum(distME_t[:,i,n])]
                    scatter!([x],yvalsME,label=label3,subplot=i,color=:black,markershape=:ltriangle)
                    localerrfwdPH += abs(sum(yvalsPH-simDist.distribution[:,n,i]))
                    localerrDG += abs(sum(yvalsDG-simDist.distribution[:,n,i]))
                    localerrfwdME += abs(sum(yvalsME-simDist.distribution[:,n,i]))
                end
            end
            display(plot!())
            push!(globalerrfwdPH,localerrfwdPH)
            push!(globalerrDG,localerrDG)
            push!(globalerrfwdME,localerrfwdME)
            # plot!()
            # for i in 1:NPhases
            #     scatter!(simDist.x,simDist.distribution[:,:,i][:],subplot=i,label="sim")
            # end
            # display(plot!())

        end
        errfwdPH = [errfwdPH globalerrfwdPH]
        errDG = [errDG globalerrDG]
        errfwdME = [errfwdME globalerrfwdME]
        # plot!(vecNBases,log.(globalerrfwdPH),label=string(Δ),colour=c)
        # plot!(vecNBases,log.(globalerrfwdME),label=false,linestyle=:dot,colour=c,linewidth=4)
        # plot!(vecNBases,log.(globalerrDG),label=false,linestyle=:dash,color=c)
        # plot!(legend=:bottomleft,xlabel="Order",ylabel="log(error)",legendtitle="Δ")
        # display(plot!(title="-- PH,  solid DG"))
    end
    # plot(log.([5 2.5 1.25 1.25/2 1.25/4])[:], log.(errfwdPH[:,2:end]'),colour=[1 2 3 4 5],label=vecNBases')
    # plot!(log.([5 2.5 1.25 1.25/2 1.25/4])[:], log.(errfwdME[:,2:end]'),colour=[1 2 3 4 5],linestyle=:dot,label=false)
    # plot!(log.([5 2.5 1.25 1.25/2 1.25/4])[:], log.(errDG[:,2:end]'),linestyle=:dash,colour=[1 2 3 4 5],label=false)
    # plot!(legend=:bottomright,xlabel="log(Δ)",ylabel="log(error)",legendtitle="Order")
    # display(plot!(title="-- PH,  solid DG"))
end
