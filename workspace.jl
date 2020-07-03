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

Model = SFFM.MakeModel(T=T,C=C,r=r,Signs=["+";"-";"0"],Bounds=[-1 1; -Inf Inf])

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
Matrices = SFFM.MakeMatrices(Model=Model,Mesh=Mesh)
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
1

function SimSFFM(;Model, StoppingTime, InitCondition)
    d = LinearAlgebra.diag(Model.T)
    P = (Model.T-LinearAlgebra.diagm(0=>d))./-d
    CumP = cumsum(P,dims=2)
    Λ = LinearAlgebra.diag(Model.T)

    M = size(InitCondition,1)
    tSims = Array{Float64,1}(undef,M)
    φSims = Array{Float64,1}(undef,M)
    XSims = Array{Float64,1}(undef,M)
    YSims = Array{Float64,1}(undef,M)
    nSims = Array{Float64,1}(undef,M)

    for m in 1:M
      t = 0.0
      φ = InitCondition[m,1]
      X = InitCondition[m,2]
      Y = InitCondition[m,3]
      n = 0
      while 1==1
        S = log(rand())/Λ[φ]
        t = t+S
        X = X+Model.C[φ]*S
        Y = Y+(Model.r.R(X)[φ]-Model.r.R(X-Model.C[φ]*S)[φ])/Model.C[φ]
        τ = StoppingTime(Model,t,φ,X,Y,n,S)
        if τ.Ind
          (tSims[m], φSims[m], XSims[m], YSims[m], nSims[m]) = τ.SFFM
          break
        end
        φ = findfirst(rand().<CumP[φ,:])
        n = n+1
      end
    end
    return (t=tSims,φ=φSims,X=XSims,Y=YSims,n=nSims)
end

function SimSFM(;Model, StoppingTime, InitCondition)
  d = LinearAlgebra.diag(Model.T)
  P = (Model.T-LinearAlgebra.diagm(0=>d))./-d
  CumP = cumsum(P,dims=2)
  Λ = LinearAlgebra.diag(Model.T)

  M = size(InitCondition,1)
  tSims = Array{Float64,1}(undef,M)
  φSims = Array{Float64,1}(undef,M)
  XSims = Array{Float64,1}(undef,M)
  nSims = Array{Float64,1}(undef,M)

  for m in 1:M
    t = 0.0
    φ = InitCondition[m,1]
    X = InitCondition[m,2]
    n = 0
    while 1==1
      S = log(rand())/Λ[φ]
      t = t+S
      X = X + Model.C[φ]*S
      τ = StoppingTime(Model=Model,t=t,φ=φ,X=X,n=n)
      if τ.Ind
        (tSims[m], φSims[m], XSims[m], nSims[m]) = τ.SFM
        break
      end
      φ = findfirst(rand().<CumP[φ,:])
      n = n+1
    end
  end
  return (t=tSims,φ=φSims,X=XSims,n=nSims)
end


function NJumpsFun(Model,t::Float64,φ,X,n::Int)
    Ind = n>=N
    SFM = (t,φ,X,n)
    return (Ind=Ind,SFM=SFM)
end

function NJumpsFun(Model,t::Float64,φ,X::Int,n::Int)
    Ind = n>=N
    SFM = (t,φ,X,n)
    return (Ind=Ind,SFM=SFM)
end





##
T = [-2.0 1.0 1.0; 1.0 -2.0 1.0; 1.0 1.0 -2]
C = [1.0;-2.0;0]
r = (
    r = function (x); [1.0.+0.01*x 1.0.+0.01*x 0*x]; end,
    R = function (x); [1*x.+0.01.*x.^2.0./2 1*x.+0.01.*x.^2.0./2 0*x]; end
) # [1*(x.<=2.0).-2.0*(x.>1.0) -2.0*(x.<=2.0).+(x.>2.0)] # [1.0 -2.0].*ones(size(x))#
Bounds = [-Inf 2;-Inf Inf]

Model = SFFM.MakeModel(T=T,C=C,r=r,Signs=["+";"-";"0"],Bounds=Bounds)
include("./SFFM.jl")
SFMsim = SFFM.SimSFM(
    Model=Model,
    StoppingTime=SFFM.NJumps(N=10),#SFFM.NJumps(N=10),#SFFM.FixedTime(T=10),
    InitCondition=repeat([1 0],1000,1)
)
histogram(SFMsim.X)
SFFMsim = SFFM.SimSFFM(
    Model=Model,
    StoppingTime=SFFM.NJumps(N=10),
    InitCondition=repeat([1 0 0],1000,1)
)
histogram(SFFMsim.X)
SFMsim = SFFM.SimSFM(
    Model=Model,
    StoppingTime=SFFM.FixedTime(T=10),
    InitCondition=repeat([1 0],1000,1)
)
histogram(SFMsim.X)
SFFMsim = SFFM.SimSFFM(
    Model=Model,
    StoppingTime=SFFM.FixedTime(T=10),
    InitCondition=repeat([1 0 0],1000,1)
)
histogram(SFFMsim.X)
SFMsim = SFFM.SimSFM(
    Model=Model,
    StoppingTime=SFFM.FirstExitX(u=-1,v=1),
    InitCondition=repeat([1 0],1000,1)
)
histogram(SFMsim.X)
SFFMsim = SFFM.SimSFFM(
    Model=Model,
    StoppingTime=SFFM.FirstExitX(u=-1,v=1),
    InitCondition=repeat([1 0 0],1000,1)
)
histogram(SFFMsim.X)
SFFMsim = SFFM.SimSFFM(
    Model=Model,
    StoppingTime=SFFM.InOutYLevel(y=4),
    InitCondition=repeat([1 0 0],1000,1)
)
histogram(SFFMsim.X)

function MakeModel(;
  T::Array{Float64},
  C::Array{Float64,1},
  r,
  Signs::Array{String,1} = ["+"; "-"; "0"],
  Bounds::Array{<:Number,2} = [-Inf Inf; -Inf Inf],
)
  # Make a 'Model' object which carries all the info we need to
  # know about the SFFM.
  # T - n×n Array{Float64}, a generator matrix of φ(t)
  # C - n×1 Array{Float64}, rates of the first fluid
  # Signs - n×1 Array{String}, the m∈{"+","-","0"} where Fᵢᵐ≂̸∅
  # IsBounded - Bool, whether the first fluid is bounded or not
  # r - array of rates for the second fluid,
  #     functions r(x) = [r₁(x) r₂(x) ... r_n(x)], where x is a column vector
  #
  # output is a NamedTuple with fields
  #                         .T, .C, .r, .Signs, .IsBounded, .NPhases, .NSigns

  NPhases = length(C)
  NSigns = length(Signs)
  println("Model.Field with Fields (.T, .C, .r, .Signs, .IsBounded, .NPhases,
            .NSigns)")
  IsBounded = true
  return (
    T = T,
    C = C,
    r = r,
    Signs = Signs,
    IsBounded = IsBounded,
    Bounds = Bounds,
    NPhases = NPhases,
    NSigns = NSigns,
  )
end

struct SFFModel
  T::Array{Float64}
  C::Array{Float64,1}
  r::NamedTuple{(:r, :R)}
  Signs::Array{String,1}
  Bounds::Array{<:Number,2}
  NPhases::Int
  NSigns::Int
  IsBounded::Bool

  function ModelGen(T,C,r)
      Signs::Array{String,1} = ["+"; "-"]
      Bounds::Array{<:Number,2} = [-Inf Inf; -Inf Inf]
      NPhases = length(C)
      NSigns = length(Signs)
      IsBounded = true
  end
end

struct mystruct
    a::Int
    b=sin(a)
end

function makeastruct(a)
    return (a=a, b=a+1)
end

mystruct
