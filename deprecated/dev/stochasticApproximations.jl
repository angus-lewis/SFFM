# using LinearAlgebra, Plots, JSON
include("src/SFFM.jl")
include("dev/METools.jl")

# define SFM
T =[-2.0 2.0 0; 1.0 -2.0 1; 4 3 -7]# [-2.0 2.0; 1.0 -1.0]#
C = [1.0; 2.0; 0]
fn(x) = [ones(size(x)) ones(size(x)) ones(size(x))]
model = SFFM.Model(;T=T,C=C,r=(r=fn,R=fn),Bounds=[0 10;-Inf Inf])
N₋ = sum(C.<=0)
N₊ = sum(C.>=0)
NPhases = length(C)

# construct global approximation

t = 3
τ = SFFM.FixedTime(T=t)
NSim = 400_000
sims = SFFM.SimSFM(model=model,StoppingTime=τ,InitCondition=(φ=2*ones(Int,NSim),X=zeros(NSim)))

p = plot()
vecΔ =  2#[5 2.5 1.25 1.25/2]
let
    vecNBases = [1,3,5,7,11,15,21]#,29]
    errPH = vecNBases
    errME = vecNBases
    errbkwdME = vecNBases
    errπME = vecNBases
    errDG = vecNBases
    c=0
    for Δ in vecΔ
        c+=1
        Nodes = collect(0:Δ:10)
        globalerrPH = []
        globalerrME = []
        globalerrbkwdME = []
        globalerrπME = []
        globalerrDG = []
        for NBases in vecNBases
            mesh = SFFM.MakeMesh(model=model,NBases=NBases,Nodes=Nodes,Basis="lagrange")
            simDist = SFFM.Sims2Dist(model=model,mesh=mesh,sims=sims,type="probability")

            # define generator for up approximation
            Erlang = MakeErlang(NBases, mean = Δ)

            NCells = length(Nodes)-1
            Q, B = MakeGlobalApprox(;
                NCells = NCells,
                up = Erlang,
                down = Erlang,
                T = T,
                C = C,
            )

            ME = MakeME(CMEParams[NBases],mean=Δ)

            QME, BME = MakeGlobalApprox(;
                NCells = NCells,
                up = ME,
                down = ME,
                T = T,
                C = C,
            )

            MEBkwd = reversal(ME)
            # αdownMEbkwd = αupME#[αupME[1]; [αupME[3:2:end] αupME[2:2:end]]'[:]]'
            # QdownMEbkwd = QupME#.*sum(-αdownMEbkwd*QupME'^-1)./Δ

            QMEbkwd, BMEbkwd = MakeGlobalApprox(;
                NCells = NCells,
                up = ME,
                down = ME,
                T = T,
                C = C,
                bkwd = false,
                D = [],
                plusI = false,
            )

            QMEπ, BMEπ = MakeGlobalApprox(;
                NCells = NCells,
                up = ME,
                down = ME,
                T = T,
                C = C,
                bkwd = true,
                D = [],
                plusI = false,
            )

            DGMesh = SFFM.MakeMesh(model=model,NBases=1,Nodes=collect(Nodes[1]:Δ/NBases:Nodes[end]),Basis="lagrange")
            All = SFFM.MakeAll(model=model,mesh=DGMesh)

            initDist = zeros(1,size(B,1))
            initDist[1] = 1

            temp = initDist*exp(Matrix(All.B.B)*t)#SFFM.EulerDG(D=All.B.B,y=t,x0=initDist)#
            DGdist_t = SFFM.Coeffs2Dist(model=model,mesh=DGMesh,Coeffs=temp,type="probability")

            dist_t = initDist*exp(B*t)#SFFM.EulerDG(D=B,y=t,x0=initDist)#
            pm_t = dist_t[[1:N₋;(end-N₊+1):end]]
            dist_t = reshape(dist_t[N₋+1:end-N₊],NBases,NPhases,NCells)

            distME_t = initDist*exp(BME*t)#SFFM.EulerDG(D=BME,y=t,x0=initDist)
            pmME_t = distME_t[[1:N₋;(end-N₊+1):end]]
            distME_t = reshape(distME_t[N₋+1:end-N₊],NBases,NPhases,NCells)

            distMEbkwd_t = initDist*exp(BMEbkwd*t)#SFFM.EulerDG(D=BME,y=t,x0=initDist)
            pmMEbkwd_t = distMEbkwd_t[[1:N₋;(end-N₊+1):end]]
            distMEbkwd_t = reshape(distMEbkwd_t[N₋+1:end-N₊],NBases,NPhases,NCells)

            distMEπ_t = initDist*exp(BMEπ*t)#SFFM.EulerDG(D=BME,y=t,x0=initDist)
            pmMEπ_t = distMEπ_t[[1:N₋;(end-N₊+1):end]]
            distMEπ_t = reshape(distMEπ_t[N₋+1:end-N₊],NBases,NPhases,NCells)

            # plot(layout = (NPhases,1))
            localerrPH = sum(abs.(pm_t-simDist.pm))
            localerrME = sum(abs.(pmME_t-simDist.pm))
            localerrbkwdME = sum(abs.(pmMEbkwd_t-simDist.pm))
            localerrπME = sum(abs.(pmMEπ_t-simDist.pm))
            localerrDG = sum(abs.(DGdist_t.pm-simDist.pm))
            for n in 1:NCells
                x = simDist.x[n]
                for i in 1:NPhases
                    if n==1
                        label1 = "PH"
                        label2 = "DG"
                        label3 = "ME"
                        label4 = "bkwdME"
                        label5 = "πME"
                    else
                        label1 = false
                        label2 = false
                        label3 = false
                        label4 = false
                        label5 = false
                    end
                    yvalsPH = [sum(dist_t[:,i,n])]
                    yvalsDG = [sum(DGdist_t.distribution[:,(1:NBases).+NBases*(n-1),i])]
                    yvalsME = [sum(distME_t[:,i,n])]
                    yvalsbkwdME = [sum(distMEbkwd_t[:,i,n])]
                    yvalsπME = [sum(distMEπ_t[:,i,n])]
                    # scatter!([x],yvalsPH,label=label1,subplot=i,color=:red,markershape=:x)
                    # scatter!([x],yvalsDG,label=label2,subplot=i,color=:blue,markershape=:rtriangle)
                    # scatter!([x],yvalsME,label=label3,subplot=i,color=:black,markershape=:ltriangle)
                    # scatter!([x],yvalsbkwdME,label=label4,subplot=i,color=:black,markershape=:utriangle)
                    localerrPH += abs(sum(yvalsPH-simDist.distribution[:,n,i]))
                    localerrDG += abs(sum(yvalsDG-simDist.distribution[:,n,i]))
                    localerrME += abs(sum(yvalsME-simDist.distribution[:,n,i]))
                    localerrbkwdME += abs(sum(yvalsbkwdME-simDist.distribution[:,n,i]))
                    localerrπME += abs(sum(yvalsπME-simDist.distribution[:,n,i]))
                end
            end
            push!(globalerrPH,localerrPH)
            push!(globalerrDG,localerrDG)
            push!(globalerrME,localerrME)
            push!(globalerrbkwdME,localerrbkwdME)
            push!(globalerrπME,localerrπME)
            # for i in 1:NPhases
            #     scatter!(simDist.x,simDist.distribution[:,:,i][:],subplot=i,label="sim")
            # end
            # display(plot!())
        end
        errPH = [errPH globalerrPH]
        errDG = [errDG globalerrDG]
        errME = [errME globalerrME]
        errbkwdME = [errbkwdME globalerrbkwdME]
        errπME = [errπME globalerrπME]
        # plot!(p,vecNBases,log.(globalerrPH),label=false,linestyle=:dot,colour=c,linewidth=2)
        # plot!(p,vecNBases,log.(globalerrME),label=false,linestyle=:dash,colour=c)
        # plot!(p,vecNBases,log.(globalerrbkwdME),label=false,linestyle=:dashdot,colour=c)
        # plot!(p,vecNBases,log.(globalerrπME),label=false,linestyle=:dashdotdot,colour=c)
        plot!(p,vecNBases,log.(globalerrDG),label="DG Δ",color=c)
        plot!(p,legend=:topright,xlabel="Order",ylabel="log(error)")
        display(plot!(p,title="DG"))
        display(globalerrbkwdME[end])
    end
    # plot(log.(vecΔ)[:], log.(errPH[:,2:end]'),colour=[1 2 3 4 5],linestyle=:dot,label=false,linewidth=2)
    # plot!(log.(vecΔ)[:], log.(errME[:,2:end]'),colour=[1 2 3 4 5],linestyle=:dash,label=false)
    # plot!(log.(vecΔ)[:], log.(errbkwdME[:,2:end]'),colour=[1 2 3 4 5],linestyle=:dashdot,label=false)
    # plot!(log.(vecΔ)[:], log.(errDG[:,2:end]'),colour=[1 2 3 4 5],label=vecNBases')
    # plot!(legend=:bottomright,xlabel="log(Δ)",ylabel="log(error)",legendtitle="Order")
    # plot!(title="... PH,  -- ME,  -. pois ME, Solid DG")
end
Δ = vecΔ

#
# let
#     vecNBases = [1,3,5,7,11,15,21]#,29]
#     vecΔ = [1.25/2]# 1.25/2]#[5 2.5 1.25 1.25/2 1.25/4]
#     errPH = vecNBases
#     errCoxian = vecNBases
#     errbkwdCoxian = vecNBases
#     errSomePH = vecNBases
#     errSomePHBkwd = vecNBases
#     errDG = vecNBases
#     plot()
#     c=0
#     for Δ in vecΔ
#         c+=1
#         Nodes = collect(0:Δ:10)
#         globalerrPH = []
#         globalerrCoxian = []
#         globalerrbkwdCoxian = []
#         globalerrSomePH = []
#         globalerrSomePHBkwd = []
#         globalerrDG = []
#         for NBases in vecNBases
#             mesh = SFFM.MakeMesh(model=model,NBases=NBases,Nodes=Nodes,Basis="lagrange")
#             simDist = SFFM.Sims2Dist(model=model,mesh=mesh,sims=sims,type="probability")
#
#             # define generator for up approximation
#             Erlang = MakeErlang(NBases, mean = Δ)
#
#             NCells = length(Nodes)-1
#             Q, B = MakeGlobalApprox(;
#                 NCells = NCells,
#                 up = Erlang,
#                 down = Erlang,
#                 T = T,
#                 C = C,
#             )
#
#             Coxian = MakeME(CMEParams[NBases], mean = Δ)#MakeSomeCoxian(NBases, mean = Δ)#
#
#             QCoxian, BCoxian = MakeGlobalApprox(;
#                 NCells = NCells,
#                 up = Coxian,
#                 down = Coxian,
#                 T = T,
#                 C = C,
#             )
#
#             CoxianBkwd = reversal(Coxian)
#
#             QCoxianbkwd, BCoxianbkwd = MakeGlobalApprox(;
#                 NCells = NCells,
#                 up = Coxian,
#                 down = CoxianBkwd,
#                 T = T,
#                 C = C,
#             )
#
#             SomePH = Coxian#MakeSomeCoxian(NBases, mean = Δ)#MakeSomePH(NBases, mean = Δ)
#
#             QSomePH, BSomePH = MakeGlobalApprox(;
#                 NCells = NCells,
#                 up = SomePH,
#                 down = SomePH,
#                 T = T,
#                 C = C,
#             )
#
#             # SomePHBkwd = reversal(SomePH)
#
#             QSomePHBkwd, BSomePHBkwd = MakeGlobalApprox(;
#                 NCells = NCells,
#                 up = SomePH,
#                 down = SomePH,
#                 T = T,
#                 C = C,
#                 bkwd = true,
#                 D = jumpMatrixD(SomePH)
#             )
#
#             DGMesh = SFFM.MakeMesh(model=model,NBases=1,Nodes=collect(Nodes[1]:Δ/NBases:Nodes[end]),Basis="lagrange")
#             All = SFFM.MakeAll(model=model,mesh=DGMesh)
#
#             initDist = zeros(1,size(B,1))
#             initDist[1] = 1
#
#             temp = initDist*exp(Matrix(All.B.B)*t)#SFFM.EulerDG(D=All.B.B,y=t,x0=initDist)#
#             DGdist_t = SFFM.Coeffs2Dist(model=model,mesh=DGMesh,Coeffs=temp,type="probability")
#
#             dist_t = initDist*exp(B*t)#SFFM.EulerDG(D=B,y=t,x0=initDist)#
#             pm_t = dist_t[[1:N₋;(end-N₊+1):end]]
#             dist_t = reshape(dist_t[N₋+1:end-N₊],NBases,NPhases,NCells)
#
#             distCoxian_t = initDist*exp(BCoxian*t)#SFFM.EulerDG(D=BCoxian,y=t,x0=initDist)
#             pmCoxian_t = distCoxian_t[[1:N₋;(end-N₊+1):end]]
#             distCoxian_t = reshape(distCoxian_t[N₋+1:end-N₊],NBases,NPhases,NCells)
#
#             distCoxianbkwd_t = initDist*exp(BCoxianbkwd*t)#SFFM.EulerDG(D=BCoxian,y=t,x0=initDist)
#             pmCoxianbkwd_t = distCoxianbkwd_t[[1:N₋;(end-N₊+1):end]]
#             distCoxianbkwd_t = reshape(distCoxianbkwd_t[N₋+1:end-N₊],NBases,NPhases,NCells)
#
#             distSomePH_t = initDist*exp(BSomePH*t)#SFFM.EulerDG(D=BCoxian,y=t,x0=initDist)
#             pmSomePH_t = distSomePH_t[[1:N₋;(end-N₊+1):end]]
#             distSomePH_t = reshape(distSomePH_t[N₋+1:end-N₊],NBases,NPhases,NCells)
#
#             distSomePHBkwd_t = initDist*exp(BSomePHBkwd*t)#SFFM.EulerDG(D=BCoxian,y=t,x0=initDist)
#             pmSomePHBkwd_t = distSomePHBkwd_t[[1:N₋;(end-N₊+1):end]]
#             distSomePHBkwd_t = reshape(distSomePHBkwd_t[N₋+1:end-N₊],NBases,NPhases,NCells)
#
#             # plot(layout = (NPhases,1))
#             localerrPH = sum(abs.(pm_t-simDist.pm))
#             localerrCoxian = sum(abs.(pmCoxian_t-simDist.pm))
#             localerrbkwdCoxian = sum(abs.(pmCoxianbkwd_t-simDist.pm))
#             localerrSomePH = sum(abs.(pmSomePH_t-simDist.pm))
#             localerrSomePHBkwd = sum(abs.(pmSomePHBkwd_t-simDist.pm))
#             localerrDG = sum(abs.(DGdist_t.pm-simDist.pm))
#             for n in 1:NCells
#                 x = simDist.x[n]
#                 for i in 1:NPhases
#                     if n==1
#                         label1 = "PH"
#                         label2 = "DG"
#                         label3 = "Coxian"
#                         label4 = "bkwdCoxian"
#                         label5 = "SomePH"
#                         label6 = "SomePHBkwd"
#                     else
#                         label1 = false
#                         label2 = false
#                         label3 = false
#                         label4 = false
#                         label5 = false
#                         label6 = false
#                     end
#                     yvalsPH = [sum(dist_t[:,i,n])]
#                     yvalsDG = [sum(DGdist_t.distribution[:,(1:NBases).+NBases*(n-1),i])]
#                     yvalsCoxian = [sum(distCoxian_t[:,i,n])]
#                     yvalsbkwdCoxian = [sum(distCoxianbkwd_t[:,i,n])]
#                     yvalsSomePH = [sum(distSomePH_t[:,i,n])]
#                     yvalsSomePHBkwd = [sum(distSomePHBkwd_t[:,i,n])]
#                     # scatter!([x],yvalsPH,label=label1,subplot=i,color=:red,markershape=:x)
#                     # scatter!([x],yvalsDG,label=label2,subplot=i,color=:blue,markershape=:rtriangle)
#                     # scatter!([x],yvalsCoxian,label=label3,subplot=i,color=:black,markershape=:ltriangle)
#                     # scatter!([x],yvalsbkwdCoxian,label=label4,subplot=i,color=:black,markershape=:utriangle)
#                     localerrPH += abs(sum(yvalsPH-simDist.distribution[:,n,i]))
#                     localerrDG += abs(sum(yvalsDG-simDist.distribution[:,n,i]))
#                     localerrCoxian += abs(sum(yvalsCoxian-simDist.distribution[:,n,i]))
#                     localerrbkwdCoxian += abs(sum(yvalsbkwdCoxian-simDist.distribution[:,n,i]))
#                     localerrSomePH += abs(sum(yvalsSomePH-simDist.distribution[:,n,i]))
#                     localerrSomePHBkwd += abs(sum(yvalsSomePHBkwd -simDist.distribution[:,n,i]))
#                 end
#             end
#             push!(globalerrPH,localerrPH)
#             push!(globalerrDG,localerrDG)
#             push!(globalerrCoxian,localerrCoxian)
#             push!(globalerrbkwdCoxian,localerrbkwdCoxian)
#             push!(globalerrSomePH,localerrSomePH)
#             push!(globalerrSomePHBkwd,localerrSomePHBkwd)
#             # for i in 1:NPhases
#             #     scatter!(simDist.x,simDist.distribution[:,:,i][:],subplot=i,label="sim")
#             # end
#             # display(plot!())
#         end
#         errPH = [errPH globalerrPH]
#         errDG = [errDG globalerrDG]
#         errCoxian = [errCoxian globalerrCoxian]
#         errbkwdCoxian = [errbkwdCoxian globalerrbkwdCoxian]
#         errSomePH = [errSomePH globalerrSomePH]
#         errSomePHBkwd = [errSomePHBkwd globalerrSomePHBkwd]
#         plot!(vecNBases,log.(globalerrPH),label="Erlang "*string(Δ),colour=c)
#         plot!(vecNBases,log.(globalerrDG),label="DG "*string(Δ),color=c,linestyle=:dash)
#         plot!(vecNBases,log.(globalerrCoxian),label="Coxian "*string(Δ),linestyle=:solid,colour=c+1,markershape=:+)
#         plot!(vecNBases,log.(globalerrbkwdCoxian),label="Cox. w rev. "*string(Δ),linestyle=:dash,colour=c+1,markershape=:+)
#         plot!(vecNBases,log.(globalerrSomePH),label="PH "*string(Δ),linestyle=:solid,colour=c+2)
#         plot!(vecNBases,log.(globalerrSomePHBkwd),label="PH. w rev. "*string(Δ),linestyle=:dash,colour=c+2)
#         plot!(legend=:outerright,xlabel="Order",ylabel="log(error)",legendtitle="Δ")
#         display(plot!(title="... PH,  -- Coxian,  -. bkwd Coxian,  Solid DG"))
#     end
#     # plot(log.(vecΔ)[:], log.(errPH[:,2:end]'),colour=[1 2 3 4 5],linestyle=:dot,label=false,linewidth=2)
#     # plot!(log.(vecΔ)[:], log.(errCoxian[:,2:end]'),colour=[1 2 3 4 5],linestyle=:dash,label=false)
#     # plot!(log.(vecΔ)[:], log.(errbkwdCoxian[:,2:end]'),colour=[1 2 3 4 5],linestyle=:dashdot,label=false)
#     # plot!(log.(vecΔ)[:], log.(errSomePH[:,2:end]'),colour=[1 2 3 4 5],linestyle=:dashdot,label=false)
#     # plot!(log.(vecΔ)[:], log.(errSomePHBkwd[:,2:end]'),colour=[1 2 3 4 5],linestyle=:dashdot,label=false)
#     # plot!(log.(vecΔ)[:], log.(errDG[:,2:end]'),colour=[1 2 3 4 5],label=vecNBases')
#     # plot!(legend=:bottomright,xlabel="log(Δ)",ylabel="log(error)",legendtitle="Order")
#     # plot!(title="... PH,  -- Coxian,  -. bkwd Coxian, Solid DG")
# end

#
# ME = MakeME(CMEParams[3],mean=2)#MakeSomeCoxian(21)#
# ~,~,πME = reversal(ME)
# γ = -1
# S = ME.Q-γ*I
# α = ME.α*(-S^-1)
# α = α./sum(α)
# MEpois = (α = α, Q = S)
# MEmix = (α = α, Q = ME.Q)
# αwhat = -ME.α*(ME.Q+γ*I)^-1
# αwhat = αwhat./sum(αwhat)
# MEwhat = (α = αwhat, Q = ME.Q)
# # A = sylvester(ME.Q,-ME.Q,γ*Matrix{Float64}(I(NBases)))
# t = 0:0.1:4
# densityME = density(t,ME)
# densityMEpois = density(t,MEpois)
# # densityMEmix = density(t,MEmix)
# densityMEwhat = density(t,MEwhat)
# plot(t,densityME)
# plot!(t,densityMEpois)
# # plot!(t,densityMEmix)
# plot!(t,densityMEwhat)
1
# a = CMEParams[3]["a"]
# b = CMEParams[3]["b"]
# c = CMEParams[3]["c"]
# omega = CMEParams[3]["omega"]
# kω = 1*omega
# Minv = [2*c*(1+kω^2) 0 0; 0 -b a; 0 a b]./(2*(1+kω^2))
# # Minv = [2*c*(1+kω^2) 0 0; 0 -a b; 0 -b -a]./(2*(1+kω^2))
# α, Q, q = MakeME(CMEParams[3], mean=CMEParams[3]["mu1"])
# println([q[1];q[3];q[2]]'*Minv)
# # println([q[1];-q[2];-q[3]]'*Minv)
# println(α)
#
# M = Minv^-1
#
# Fkkp1 = q*α*M
# display(Fkkp1)
# Fkkplus = -q*q'
# display(Fkkplus)
# display(-Q*M) # = (F+G)
# G = -Fkkplus-Q*M
# display(G) # G = -F + Q*M
# display(M)
#
# # check -(F+G)M⁻¹=Q
# display(Q)
# display(-(Fkkplus+G)*Minv)
#
# # phi at RHS
# display(q')
# # phi at LHS
# display(α*M)
# # inner products
# display(M)
#
# # for negative phases
# Fkkminus = (α*M)'*(α*M)
# display(Fkkminus)
# Q̃ = (G+Fkkminus)*Minv
# display(Q̃)
# display(eigen(Q̃).values)
#
#
# mesh = SFFM.MakeMesh(
#     model = model,
#     Nodes = collect(0:1:10),
#     NBases = 4,
#     Basis = "lagrange",
# )
#
# Mat = SFFM.MakeMatrices(mesh=mesh,model=model)
