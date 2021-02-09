using LinearAlgebra, Plots, JSON, Jacobi
include("../src/SFFM.jl")
include("METools.jl")

include("stochasticApproximations.jl")
p = plot!()

T = [-2.0 2.0; 1.0 -1.0]#[-2.0 2.0 0; 0.5 -2.0 1.5; 0 2 -2]#
C = [1.0; -2.0]#; -1]
fn(x) = [ones(size(x)) ones(size(x))]# ones(size(x))]
Model = SFFM.MakeModel(;T=T,C=C,r=(r=fn,R=fn),Bounds=[0 10;-Inf Inf])
N₋ = sum(C.<=0)
N₊ = sum(C.>=0)
NPhases = length(C)

tim = 3
# τ = SFFM.FixedTime(T=tim)
# NSim = 500_000
# sims = SFFM.SimSFM(Model=Model,StoppingTime=τ,InitCondition=(φ=2*ones(Int,NSim),X=zeros(NSim)))

function e(i,order)
    e = zeros(order)
    e[i] = 1
    return e
end

let
    globalerrME = []
    globalerrDG = []
    orders = 1:2:23
    for order in orders
        μ = vecΔ
        ME = MakeME(CMEParams[order], mean = μ)#MakeErlang(order, mean = μ)#
        D = CMEParams[order]["D"]

        # if order==1
        #     tvec = [0]
        #     midpoints = 0
        # elseif order==2
        #     tvec=[0;1]
        # else
        #     h = (1)./(order-2)
        #     # tvec = [0;range(0+h/2,1-h/2,length=order-2);1]
        #     # tvec = range(0,2*(π./ME.Q[3,2]),length=order+1)[1:end-1]
        #     # tvec = ((-cos.(π.*range(0,1,length=order+1)).+1.0)./2*μ)[1:end-1]
        #     tvec = range(0,μ,length=order)[1:end]
        #     # tvec = ((Jacobi.zglj(order+1, 0, 0).+1.0)./2*μ)[1:end-1]#2*(π./ME.Q[3,2]))[1:end-1]
        #     # tvec4 = [0;range(0+h/2,1-h/2,length=order-1)]
        #     midpoints = [0;(tvec[1:end-1]+tvec[2:end])./2]
        #     # midpoints = range(0,μ,length=order)[1:end]
        # end
        # # tvec = range(0,1,length=order+1)
        # # tvec = tvec[1:end-1]
        # #[0;range(0.01,20,length=order-1)]
        # #[0;0.01;20]
        # #[0;range(0.1*μ,2*μ,length=order-1)]
        # #2*((exp(1).^(range(0,1,length=order))).-1)./(exp(1)-1)
        # H = zeros(order,order)
        # # H[1,:] = ME.α
        # for n in 1:order
        #     u = ME.α*exp(ME.Q*tvec[n])
        #     # u = ME.α*(ME.Q^-1)*(exp(ME.Q*tvec[n]*μ)-exp(ME.Q*tvec[n-1]*μ))
        #     u = u./sum(u)
        #     H[n,:] = u
        # end
        # # H[end,:] = ME.α*(ME.Q^-1)*[-exp(ME.Q*tvec[end]*μ)]
        #
        # MEf = (α = [1;zeros(order-1)]', Q = H*ME.Q*H^-1)
        # # MEf2 = (α = [0;1;zeros(order-2)]', Q = H*ME.Q*H^-1)
        #
        # # timesvector = 0:0.01:2*μ
        # # let
        # #     f = zeros(length(timesvector),order)
        # #     g = zeros(length(timesvector),order)
        # #     c = 0
        # #     for t in timesvector
        # #         c = c+1
        # #         g[c,:] = H*exp(ME.Q*t)*ME.q
        # #     end
        # #     plot(timesvector,g)
        # #     scatter!(tvec,zeros(size(tvec)))
        # # end
        #
        # # D = ME.q.*D
        # # D = I(order)[end:-1:1,:]
        # if order>1
        #     idx = sum(tvec.<=μ)+1
        # else
        #     idx = order
        # end
        # uvec = midpoints#tvec#[1:idx]
        # D = zeros(order,order)
        # for n in 2:length(uvec)
        #     dt = uvec[n]-uvec[n-1]
        #     u = (I-exp(ME.Q*dt))*exp(ME.Q*uvec[n-1])*ones(order)
        #
        #     # if n==2
        #     #     dt = (tvec[n]-tvec[n-1])/2
        #     #     u = (I-exp(MEf.Q*μ*dt))*ones(order)
        #     # else
        #     #     dt = tvec[n]-tvec[n-1]
        #     #     u = (I-exp(MEf.Q*μ*dt))*exp(MEf.Q*μ*midpoints[n-2])*ones(order)
        #     # end
        #     D[:, length(uvec)-n+2] = u
        # end
        # u = exp(ME.Q*uvec[end])*ones(order)
        # # u = exp(MEf.Q*μ*midpoints[end])*ones(order)
        # D[:,1] = u
        # D = D[:,end:-1:1]*H
        # D = integrateD(100000,CMEParams[string(order)])
        #
        # U = zeros(order,order)
        # for n in 2:length(uvec)
        #     dt = uvec[n]-uvec[n-1]
        #     u = e(n,order)'*((I-exp(ME.Q*dt))*exp(ME.Q*uvec[n-1]))^-1*ME.Q
        #     u = u./sum(u)
        #
        #     # if n==2
        #     #     dt = (tvec[n]-tvec[n-1])/2
        #     #     u = (I-exp(MEf.Q*μ*dt))*ones(order)
        #     # else
        #     #     dt = tvec[n]-tvec[n-1]
        #     #     u = (I-exp(MEf.Q*μ*dt))*exp(MEf.Q*μ*midpoints[n-2])*ones(order)
        #     # end
        #     U[:, n-1] = u
        # end
        # u = e(1,order)'*exp(ME.Q*uvec[end])^-1*ME.Q
        # u = u./sum(u)
        # # u = exp(MEf.Q*μ*midpoints[end])*ones(order)
        # U[:,end] = u
        # MEU = (α = ME.α*U^-1, Q = U*ME.Q*U^-1)
        # D = I(order)[:,end:-1:1]

        function MakeGlobalApprox(;NCells = 3,up, down,T,C,bkwd=false,jumpMatrixD=I)
            αup,Qup = up
            αdown,Qdown = down
            N₋ = sum(C.<=0)
            N₊ = sum(C.>=0)
            NPhases = length(C)
            order = length(αup)
            qup = -sum(Qup,dims=2)
            qdown = -sum(Qdown,dims=2)
            Q = zeros(NCells*order*NPhases,NCells*order*NPhases)
            for n in 1:NCells
                for i in 1:NPhases
                    idx = (1:order) .+ (n-1)*(order*NPhases) .+ (i-1)*order
                    if C[i]>0
                        Q[idx,idx] = Qup*abs(C[i])
                        if n<NCells
                            Q[idx,idx .+ order*NPhases] = qup*αup*abs(C[i])
                        end
                    elseif C[i]<0
                        Q[idx,idx] = Qdown*abs(C[i])
                        if n>1
                            Q[idx,idx .- order*NPhases] = qdown*αdown*abs(C[i])
                        end
                    end
                end
            end
            T₋₋ = T[C.<=0,C.<=0]
            T₊₋ = T[C.>=0,:].*((C.<0)')
            T₋₊ = T[C.<=0,:].*((C.>0)')
            T₊₊ = T[C.>=0,C.>=0]

            inLower = [kron(diagm(abs.(C).*(C.<0)),qdown)[:,C.<0]; zeros((NCells-1)*NPhases*order,N₋)]
            outLower = [kron(1,kron(T₋₊,αup)) zeros(N₋,N₊+(NCells-1)*NPhases*order)]
            inUpper = [zeros((NCells-1)*NPhases*order,N₊);kron(diagm(abs.(C).*(C.>0)),qup)[:,C.>0]]
            outUpper = [zeros(N₊,N₋+(NCells-1)*NPhases*order) kron(1,kron(T₊₋,αdown))]

            if bkwd
                Tdiag = diagm(diag(T))
                Toff = T-diagm(diag(T))
                T₊₋ = T.*((C.>0)*(C.<0)')
                T₋₊ = T.*((C.<0)*(C.>0)')
                Tchange = T₊₋ + T₋₊
                Tnochange = T-Tchange

                D = zeros(size(Q))
                for c in 1:NCells, i in 1:length(C), j in 1:length(C)
                    idxi = ((i-1)*order) .+ (1:order) .+ ((c-1)*order*NPhases)
                    idxj = ((j-1)*order) .+ (1:order) .+ ((c-1)*order*NPhases)
                    if i!=j
                        if C[i]>0 && C[j]<0
                            D[idxi,idxj] = T[i,j].*jumpMatrixD
                        elseif C[i]<0 && C[j]>0
                            D[idxi,idxj] = T[i,j].*jumpMatrixD
                        else
                            D[idxi,idxj] = T[i,j].*I(order)
                        end
                    else
                        D[idxi,idxj] = T[i,i].*I(order)
                    end
                end
                B = [
                    T₋₋ outLower;
                    inLower D+Q inUpper;
                    outUpper T₊₊;
                ]
            else
                B = [
                    T₋₋ outLower;
                    inLower kron(I(NCells),kron(T,I(order)))+Q inUpper;
                    outUpper T₊₊;
                ]
            end
            return Q, B
        end

        Δ = μ
        Nodes = collect(0:Δ:10)
        NCells = length(Nodes)-1
        Q, B = MakeGlobalApprox(
            NCells = NCells,
            up = ME,
            down = ME,
            T = T,
            C = C,
            bkwd = true,
            jumpMatrixD = D,
        )
        if order<6
            display(B)
        end

        # DGMesh = SFFM.MakeMesh(Model=Model,NBases=1,Nodes=collect(Nodes[1]:Δ/order:Nodes[end]),Basis="lagrange")
        DGMesh = SFFM.MakeMesh(Model=Model,NBases=order,Nodes=Nodes,Basis="lagrange")
        All = SFFM.MakeAll(Model=Model,Mesh=DGMesh)

        initDist = zeros(1,size(All.B.B,1))
        initDist[1] = 1

        temp = initDist*exp(Matrix(All.B.B)*tim)#SFFM.EulerDG(D=All.B.B,y=t,x0=initDist)#
        DGdist_t = SFFM.Coeffs2Dist(Model=Model,Mesh=DGMesh,Coeffs=temp,type="probability")

        initDist = zeros(1,size(B,1))
        initDist[1] = 1
        dist_t = initDist*exp(B*tim)#SFFM.EulerDG(D=B,y=t,x0=initDist)#
        display(dist_t)
        display(DGdist_t.distribution)
        pm_t = dist_t[[1:N₋;(end-N₊+1):end]]
        dist_t = reshape(dist_t[N₋+1:end-N₊],order,NPhases,NCells)

        Mesh = SFFM.MakeMesh(Model=Model,NBases=order,Nodes=Nodes,Basis="lagrange")
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
                yvalsME = [sum(dist_t[:,i,n])]
                # yvalsDG = [sum(DGdist_t.distribution[:,(1:order).+(n-1)*order,i])]
                yvalsDG = [sum(DGdist_t.distribution[:,n,i])]
                # scatter!([x],yvalsDG,label=label1,subplot=i,color=:blue,markershape=:rtriangle)
                # scatter!([x],yvalsME,label=label2,subplot=i,color=:black,markershape=:ltriangle)
                localerrME += abs(sum(yvalsME-simDist.distribution[:,n,i]))
                localerrDG += abs(sum(yvalsDG-simDist.distribution[:,n,i]))
            end
        end
        # display(q)
        # for i in 1:NPhases
        #     scatter!(q,simDist.x,simDist.distribution[:,:,i][:],subplot=i,label="sim")
        # end
        # display(q)
        push!(globalerrME,localerrME)
        push!(globalerrDG,localerrDG)
    end
    p = plot!(orders,log.(globalerrME),label="CME",linestyle=:dash,colour=2)
    plot!(p,orders,log.(globalerrDG),label="DG NBases",linestyle=:solid,colour=3)
    display(p)
end
# push!(globalerrME,localerrME)
# push!(globalerrDG,localerrDG)
# MẼ.q
# #
# timesvector = 0:0.01:2*π./ME.Q[3,2]
# let
#     f = zeros(length(timesvector),order)
#     g = zeros(length(timesvector),order)
#     c = 0
#     for t in timesvector
#         c = c+1
#         g[c,:] = H*exp(ME.Q*t)*ME.q
#     end
#     plot(timesvector,g)
# end
# #
# #
# NB = (α = [1;zeros(order-1)]', Q = MEf.Q)
# plot!(timesvector,density(timesvector,NB))
#
#
# plot(timesvector, orbit(timesvector,MEf)*D[:,end:-1:1], legend=:outertopright)
# plot!(timesvector, density(timesvector,MEf))
# scatter!(tvec, 0*tvec)
#
# MEtemp = (α = orbit(uvec[1]+uvec[2]/2,MEf)*D, Q = MEf.Q)
# plot(timesvector, density(timesvector,MEtemp), legend=:outertopright)
# for i in 2:length(uvec)
#     MEtemp = (α = orbit(uvec[i]+uvec[2]/2,MEf)*D, Q = MEf.Q)
#     plot!(timesvector, density(timesvector,MEtemp), legend=:outertopright)
# end
#
# plot!()
