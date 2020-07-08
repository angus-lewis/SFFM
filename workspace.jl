## Workspace
include("./SFFM.jl")
using LinearAlgebra

T = [-1.0 1.0; 1.0 -1.0]
C = [1.0;-2.0]

CM = diagm(0=>C)
A = CM*exp(CM^-1*T)*CM^-1
display(A)
x = -A[1,2]/A[2,2]
display([1 x]*CM*exp(CM^-1*T*1)*CM^-1*abs.(CM))

r = (r = function (x); [1.0.+0.01*x 1.0.+0.01*x].*ones(size(x)); end,
        R = function (x); [1*x.+0.01.*x.^2.0./2 1*x.+0.01.*x.^2.0./2]; end) # [1*(x.<=2.0).-2.0*(x.>1.0) -2.0*(x.<=2.0).+(x.>2.0)] # [1.0 -2.0].*ones(size(x))#

Model = SFFM.MakeModel(T=T,C=C,r=r,Bounds=[-1 1; -Inf Inf])

Nodes = collect(0.0:1:5.0)
MaxIters = 100
Fil = Dict{String,BitArray{1}}("1+" => Bool[1, 1, 0, 0, 0],
                                "10" => Bool[0, 0, 0, 0, 0],
                                "20" => Bool[0, 0, 0, 0, 0],
                                "2+" => Bool[1, 1, 1, 1, 1],
                                "2-" => Bool[0, 0, 0, 0, 0],
                                "1-" => Bool[0, 0, 1, 1, 1])
# Fil = Dict{String,BitArray{1}}("1+" => Bool[1, 1, 0, 0, 0],
#                                 "10" => Bool[0, 0, 0, 0, 0],
#                                 "20" => Bool[0, 0, 0, 0, 0],
#                                 "2+" => Bool[0, 0, 1, 1, 1],
#                                 "2-" => Bool[1, 1, 0, 0, 0],
#                                 "1-" => Bool[0, 0, 1, 1, 1])
#Dict{String,BitArray{1}}("1+" => r(Mesh.CellNodes[:])[2:NBases:end,1].>0, #trues(length(Nodes)-1),# 0],
#                                "10" => r(Mesh.CellNodes[:])[2:NBases:end,1].==0,
#                                "2+" => r(Mesh.CellNodes[:])[2:NBases:end,2].>0,
#                                "20" => r(Mesh.CellNodes[:])[2:NBases:end,2].==0, #falses(length(Nodes)-1),#, 1],
#                                "2-" => r(Mesh.CellNodes[:])[2:NBases:end,2].<0, #trues(length(Nodes)-1),#, 0],
#                                "1-" => r(Mesh.CellNodes[:])[2:NBases:end,1].<0) #falses(length(Nodes)-1) )#, 1])
NBases = 2

Mesh = SFFM.MakeMesh(Model=Model,Nodes=Nodes,NBases=NBases,Fil=Fil)
Matrices = SFFM.MakeMatrices(Model=Model,Mesh=Mesh,Basis="lagrange")
MatricesR = SFFM.MakeMatricesR(Model=Model,Mesh=Mesh)
B = SFFM.MakeB(Model=Model,Mesh=Mesh,Matrices=Matrices)
R = SFFM.MakeR(Model=Model,Mesh=Mesh)
D = SFFM.MakeD(Model=Model,Mesh=Mesh,R=R,B=B)
DR = SFFM.MakeDR(Matrices=Matrices,MatricesR=MatricesR,Model=Model,Mesh=Mesh,R=R,B=B)
# [D["++"]() D["+-"](); D["-+"]() D["--"]()]
display(D["++"](s=0))
display(DR.D(0))
display(sum(abs.(D["++"](s=0)-DR.D(0))))
display(D["++"](s=0)-DR.D(0))

Ψlegendre = SFFM.PsiFun(D=D,MaxIters=MaxIters,s=1)
display(Ψlegendre)
SFFM.MyPrint(Ψlegendre*repeat([1;zeros(NBases-1)],sum(Fil["-"])))

VinvtΨlegendreVinv = kron(I(sum(Fil["+"])),Matrices.Local.V.inv')*Ψlegendre*kron(I(sum(Fil["-"])),Matrices.Local.V.inv)
SFFM.MyPrint(sum(VinvtΨlegendreVinv,dims=2))

EvalD = Dict{String,Array{Float64}}("+-" => D["+-"](s=0))
Dimensions = size(EvalD["+-"])
for ℓ in ["++","--","-+"]
    EvalD[ℓ] = D[ℓ](s=0)
end
A = EvalD["++"]  #+ Ψ*EvalD["-+"]
B = EvalD["--"]*kron(I(sum(Fil["-"])),Matrices.Local.M)  #+ EvalD["-+"]*Ψ
C = EvalD["+-"]*kron(I(sum(Fil["-"])),Matrices.Local.M)  # + Ψlegendre*EvalD["-+"]*Ψlegendre
Psi = zeros(Float64,Dimensions)
OldPsi = Psi
for n = 1:MaxIters
    Psi = LinearAlgebra.sylvester(A,B,C)
    if maximum(abs.(OldPsi - Psi)) < 0
        flag = 0
        exitflag = string("Reached err tolerance in ",n,
            " iterations with error ",
            string(maximum(abs.(OldPsi - Psi))))
        break
    elseif any(isnan.(Psi))
        flag = 0
        exitflag = string("Produced NaNs at iteration ",n)
        break
    end
    OldPsi .= Psi
    A .= EvalD["++"] + Psi*EvalD["-+"]
    B .= (EvalD["--"] + EvalD["-+"]*Psi)*kron(I(sum(Fil["-"])),Matrices.Local.M)
    C .= (EvalD["+-"] - Psi*EvalD["-+"]*Psi)*kron(I(sum(Fil["-"])),Matrices.Local.M)
end
display(Psi) #

## lagrange
Matrices = SFFM.MakeMatrices(Model=Model,Mesh=Mesh,Basis="lagrange")
B = SFFM.MakeB(Model=Model,Mesh=Mesh,Matrices=Matrices)
R = SFFM.MakeR(Model=Model,Mesh=Mesh)
D = SFFM.MakeD(Model=Model,Mesh=Mesh,R=R,B=B)

Ψlagrange = SFFM.PsiFun(D=D,MaxIters=MaxIters,s=0)

## legendre
Ψt = kron(I(sum(Fil["+"])),
        Matrices.Local.V.inv')*Ψlegendre*kron(
                                    I(sum(Fil["-"])),Matrices.Local.V.V')
display(Ψt)
display(Ψlegendre)
display(Ψlagrange)
display(Ψt-Ψlagrange)

display(sum(Ψt,dims=2))
display(Ψlagrange*repeat(sum(Matrices.Local.M,dims=2),sum(Fil["-"])))
display(sum(VinvtΨlegendreVinv,dims=2))

##
for NBases in 1:4
    Mesh = SFFM.MakeMesh(Model=Model,Nodes=Nodes,NBases=NBases,Fil=Fil)
    Matrices = SFFM.MakeMatrices(Model=Model,Mesh=Mesh,Basis="legendre")
    B = SFFM.MakeB(Model=Model,Mesh=Mesh,Matrices=Matrices)
    R = SFFM.MakeR(Model=Model,Mesh=Mesh)
    D = SFFM.MakeD(Model=Model,Mesh=Mesh,R=R,B=B)
    #display(D["--"]()[(NBases+1):(2*NBases),(NBases+1):(2*NBases)])
    Ψ = SFFM.PsiFun(D=D,MaxIters=MaxIters)
    display(Ψ[1:NBases,1:NBases])
end

##
T = [-1 0 1 ;
     2 -2 0 ;
     1 1 -2 ];
C = diagm(0=>[3, -1, -2])
M = 1
A = C*exp(C^-1*T*M)*C^-1
a = [0]*C[diag(C).>0,diag(C).>0]^-1
b = [0 1]*abs.(C[diag(C).<0,diag(C).<0]^-1)
r =  ( b-a*A[diag(C).>0, diag(C).<0] )/A[diag(C).<0,diag(C).<0]
display([a r]*C*exp(C^-1*T*M)*C^-1*abs.(C))
display([a r]*C*exp(C^-1*T*0)*C^-1*abs.(C))
top = sum(([a r]*C*exp(C^-1*T*M)*C^-1*abs.(C))[diag(C).>0])
bottom = sum(([a r]*C*exp(C^-1*T*0)*C^-1*abs.(C))[diag(C).<0])
q = -sum(T,dims=2)

let s = 0
    for t in 0.0:0.0001:M
        temp = sum([a r]*C*exp(C^-1*T*t)*C^-1*q)
        s = s + 0.0001*temp
        if temp<-1e-8
            display(temp)
            display(t)
        end
    end
    display(top+bottom)
    display(s)
    display(top+bottom+s)
end


##
include("./SFFM.jl")
using Plots, LinearAlgebra, KernelDensity

T = [-2.0 1.0 1.0; 1.0 -2.0 1.0; 1.0 1.0 -2]
C = [1.0;-2.0;0]
r = (
    r =
        function (x)
            [1.0.+sin.(x) 2*(x.>0).*x.^2.0.+1 (x.>0).*x.+1]
        end, # r = function (x); [1.0.+0.01*x 1.0.+0.01*x 1*ones(size(x))]; end,
    R = function (x); [x.-cos.(x) 2*(x.>0).*x.^3/3.0.+1*x (x.>0).*x.^2/2.0.+1*x]; end # R = function (x); [1*x.+0.01.*x.^2.0./2 1*x.+0.01.*x.^2.0./2 1*x]; end
) # [1*(x.<=2.0).-2.0*(x.>1.0) -2.0*(x.<=2.0).+(x.>2.0)] # [1.0 -2.0].*ones(size(x))#
Bounds = [-Inf Inf;-Inf Inf]

Model = SFFM.MakeModel(T=T,C=C,r=r,Bounds=Bounds)

y = 2
NSim = 40000
IC = (φ=ones(Int,NSim), X=zeros(NSim), Y=zeros(NSim))
sims = SFFM.SimSFFM(
    Model=Model,
    StoppingTime=SFFM.InOutYLevel(y=y),
    InitCondition=IC
)
histogram(sims.X[sims.φ.==1])
Nodes = collect(-10.0:0.5:10.0)
Fil = Dict{String,BitArray{1}}("1+" => trues(length(Nodes)-1),
                                "2+" => trues(length(Nodes)-1),
                                "3+" => trues(length(Nodes)-1))
NBases = 2

Mesh = SFFM.MakeMesh(Model=Model,Nodes=Nodes,NBases=NBases,Fil=Fil)
Matrices = SFFM.MakeMatrices(Model=Model,Mesh=Mesh)
MatricesR = SFFM.MakeMatricesR(Model=Model,Mesh=Mesh)
B = SFFM.MakeB(Model=Model,Mesh=Mesh,Matrices=Matrices)
R = SFFM.MakeR(Model=Model,Mesh=Mesh)
D = SFFM.MakeD(Model=Model,Mesh=Mesh,R=R,B=B)
DR = SFFM.MakeDR(Matrices=Matrices,MatricesR=MatricesR,Model=Model,Mesh=Mesh,R=R,B=B)

function Integrater(; D, y, x0)
    h = 0.0001
    x = x0
    for t in h:h:y
        dx = h*x*D
        x = x+dx
    end
    return x
end
#x0 = ones(1,size(B.B,1))/size(B.B,1)
x0 = Matrix([
    zeros(Mesh.NBases*Mesh.NIntervals*2÷3);
    Mesh.Δ[1]; zeros(NBases-1); zeros(Mesh.NBases*Mesh.NIntervals*1÷3-NBases+1);
    zeros(Mesh.TotalNBases*2)
    ]')
yvalsR = Integrater(D=DR.DDict["++"](s=0), y = y, x0 = x0)[1:NBases:end]./Mesh.Δ[1]
yvals = Integrater(D=D["++"](s=0), y = y, x0 = x0)[1:NBases:end]./Mesh.Δ[1]
# Matrix([
#     zeros(Mesh.TotalNBases*2);
#     1;
#     zeros(Mesh.TotalNBases-1)
#     ]' * exp(D["++"](s=0)*y)
# )
p = plot(legend=:topleft)
Y = zeros(length(Nodes)-1,Model.NPhases)
YR = zeros(length(Nodes)-1,Model.NPhases)
let cum = 0
    for i in 1:Model.NPhases
        idx = findall(.!Fil[string(i)*"0"]) .- cum .+ (i - 1)*Mesh.NIntervals
        cum = cum + sum(Fil[string(i)*"0"])
        p = plot!(
            Mesh.CellNodes[1,.!Fil[string(i)*"0"]][:],
            yvals[idx],
            label="φ="*string(i)*" - D"
            )
        Y[:,i] = yvals[idx]
        p = plot!(
            Mesh.CellNodes[1,.!Fil[string(i)*"0"]][:],
            yvalsR[idx],
            label = "φ="*string(i)*" - DR"
            )
        YR[:,i] = yvalsR[idx]
    end
end
display(p)
H = zeros(length(Nodes)-1,Model.NPhases)
for whichφ in 1:Model.NPhases
    #pltvals = kde(sims.X[sims.φ.==whichφ])
    #p = histogram!(sims.X[sims.φ.==whichφ],bins=Nodes,normalize=:probability,alpha=0.2)
    # plot!(
    #     range(
    #         minimum(sims.X[sims.φ.==whichφ]),
    #         maximum(sims.X[sims.φ.==whichφ]),length=100
    #     ),
    #     z->pdf(pltvals,z)*sum(sims.φ.==whichφ)/length(sims.φ),
    #     label = "φ="*string(i)*" - sim"
    # )
    h = fit(Histogram,sims.X[sims.φ.==whichφ],Nodes)
    h = h.weights./sum(h.weights)*sum(sims.φ.==whichφ)/length(sims.φ)
    H[:,whichφ] = h
    p = plot!(Nodes[1:end-1]+diff(Nodes)/2,h,label="hist"*string(whichφ))
end

display(p)

err = H-Y
errR = H-YR
plot(Nodes[1:end-1]+diff(Nodes)/2,err,legend=:bottomleft)
plot!(Nodes[1:end-1]+diff(Nodes)/2,errR)
display(sum(abs.(err)*Mesh.Δ[1]))
display(sum(abs.(errR)*Mesh.Δ[1]))
display(abs.(err))
display(abs.(errR))
