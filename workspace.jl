## Workspace
# push!(LOAD_PATH,"/Users/a1627293/Documents/SFFM")
cd("/Users/angus2/Documents/SFFM")
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
Matrices = SFFM.MakeMatrices(Model=Model,Mesh=Mesh,Basis="legendre")
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
using Plots, LinearAlgebra, KernelDensity, StatsBase

## Define the model
T = [-2.0 1.0 1.0; 1.0 -2.0 1.0; 1.0 1.0 -2.0]
C = [1.0; -1.0; 0.0]
r = (
    r = function (x)
        [ones(size(x)) ones(size(x)) ones(size(x))]
    end, # r = function (x); [1.0.+0.01*x 1.0.+0.01*x 1*ones(size(x))]; end,
    R = function (x)
        [x x x]
    end, # R = function (x); [1*x.+0.01.*x.^2.0./2 1*x.+0.01.*x.^2.0./2 1*x]; end
) # [1*(x.<=2.0).-2.0*(x.>1.0) -2.0*(x.<=2.0).+(x.>2.0)] # [1.0 -2.0].*ones(size(x))#
Bounds = [-1 1; -Inf Inf]
Model = SFFM.MakeModel(T = T, C = C, r = r, Bounds = Bounds)

# in out Y-level
y = 1

## Simulate the model
NSim = 20000
# IC = (φ = ones(Int, NSim), X = zeros(NSim), Y = zeros(NSim))
# IC = (φ = 2 .*ones(Int, NSim), X = -10*ones(NSim), Y = zeros(NSim))
IC = (
    φ = sum(rand(NSim) .< [1 / 3 2 / 3 1], dims = 2),
    X = (Bounds[1,2]-Bounds[1,1]) .* rand(NSim) .- -Bounds[1,1],
    Y = zeros(NSim),
)
sims =
    SFFM.SimSFFM(Model = Model, StoppingTime = SFFM.InOutYLevel(y = y), InitCondition = IC)

## Define the mesh
Δ = 0.1
Nodes = collect(Bounds[1, 1]:Δ:Bounds[1, 2])
Fil = Dict{String,BitArray{1}}(
    "1+" => trues(length(Nodes) - 1),
    "2+" => trues(length(Nodes) - 1),
    "3+" => trues(length(Nodes) - 1),
    "p2+" => trues(1),
    "q1+" => trues(1),
    "p3+" => trues(1),
    "q3+" => trues(1),
)
NBases = 6
Mesh = SFFM.MakeMesh(Model = Model, Nodes = Nodes, NBases = NBases, Fil = Fil)

## Construct all DG operators
All = SFFM.MakeAll(Model = Model, Mesh = Mesh, Basis = "legendre")
Matrices = All.Matrices
MatricesR = All.MatricesR
B = All.B
R = All.R
D = All.D
DR = All.DR
MyD = SFFM.MakeMyD(Model = Model, Mesh = Mesh, MatricesR = MatricesR, B = B)

## initial condition
# x0 = Matrix(
#     [
#         zeros(sum(Model.C.<=0)) # LHS point mass
#         zeros(Mesh.NBases * Mesh.NIntervals * 1 ÷ 2) # phase 1
#         1 # phase 1
#         zeros(NBases - 1) # phase 1
#         zeros(Mesh.NBases * Mesh.NIntervals * 1 ÷ 2 - NBases) # phase 1
#         zeros(Mesh.TotalNBases * 2) # phases 2 and 3
#         zeros(sum(Model.C.>=0)) # RHS point mass
#     ]',
# )
x0 = Matrix(
    [
        zeros(sum(Model.C .<= 0)) # LHS point mass
        repeat([1.0; zeros(Float64, NBases-1)], Model.NPhases * Mesh.NIntervals, 1) ./
        (Model.NPhases * Mesh.NIntervals)
        zeros(sum(Model.C .>= 0)) # RHS point mass
    ]',
)

## DG approximations to exp(Dy)
yvalsR =
    SFFM.EulerDG(D = DR.DDict["++"](s = 0), y = y, x0 = x0, h = 0.0001)[[
        1:sum(Model.C .<= 0)
        sum(Model.C .<= 0).+1:1:end.-sum(Model.C .>= 0)
        end.-sum(Model.C .>= 0).+1:end
    ]]
yvalsR = [yvalsR[1:2]' sum(reshape(yvalsR[3:end-2],NBases,length(yvalsR[3:end-2])÷NBases),dims=1) yvalsR[end-1:end]']
yvals =
    SFFM.EulerDG(D = D["++"](s = 0), y = y, x0 = x0, h = 0.0001)[[
        1:sum(Model.C .<= 0)
        sum(Model.C .<= 0).+1:1:end.-sum(Model.C .>= 0)
        end.-sum(Model.C .>= 0).+1:end
    ]]
yvals = [yvals[1:2]' sum(reshape(yvals[3:end-2],NBases,length(yvals[3:end-2])÷NBases),dims=1) yvals[end-1:end]']
MyDyvals =
    SFFM.EulerDG(D = MyD.D(s = 0), y = y, x0 = x0, h = 0.0001)[[
        1:sum(Model.C .<= 0)
        sum(Model.C .<= 0).+1:1:end.-sum(Model.C .>= 0)
        end.-sum(Model.C .>= 0).+1:end
    ]]
MyDyvals = [MyDyvals[1:2]' sum(reshape(MyDyvals[3:end-2],NBases,length(MyDyvals[3:end-2])÷NBases),dims=1) MyDyvals[end-1:end]']
## analysis and plots

# plot solutions
p = plot(legend = false, layout = (3, 1))
Y = zeros((length(Nodes) - 1), Model.NPhases)
YR = zeros((length(Nodes) - 1), Model.NPhases)
MyY = zeros((length(Nodes) - 1), Model.NPhases)
let cum = 0
    for i = 1:Model.NPhases
        idx = findall(.!Fil[string(i)*"0"]) .- cum .+ (i - 1) * Mesh.NIntervals .+
            sum(Model.C .<= 0)
        cum = cum + sum(Fil[string(i)*"0"])*NBases
        p = plot!(
            (
                Mesh.CellNodes[1, .!Fil[string(i)*"0"]][:] +
                Mesh.CellNodes[end, .!Fil[string(i)*"0"]][:]
            ) / 2,
            yvals[idx],
            label = "φ=" * string(i) * " - D",
            subplot = i,
        )
        Y[:, i] = yvals[idx]
        # p = plot!(
        #     (
        #         Mesh.CellNodes[1, .!Fil[string(i)*"0"]][:] +
        #         Mesh.CellNodes[end, .!Fil[string(i)*"0"]][:]
        #     ) / 2,
        #     yvalsR[idx],
        #     label = "φ=" * string(i) * " - DR",
        #     subplot = i,
        # )
        # YR[:, i] = yvalsR[idx]
        p = plot!(
            (
                Mesh.CellNodes[1, .!Fil[string(i)*"0"]][:] +
                Mesh.CellNodes[end, .!Fil[string(i)*"0"]][:]
            ) / 2,
            MyDyvals[idx],
            label = "φ=" * string(i) * " - MyD",
            subplot = i,
        )
        MyY[:, i] = MyDyvals[idx]
    end
end
p = plot!(subplot = 1, legend = :topright)
pmdata = [
    [Nodes[1] * ones(sum(Model.C .<= 0)); Nodes[end] * ones(sum(Model.C .>= 0))]'
    yvals[[.!Mesh.Fil["p0"]; falses(Model.NPhases * Mesh.NIntervals); .!Mesh.Fil["q0"]]]'
    yvalsR[[.!Mesh.Fil["p0"]; falses(Model.NPhases * Mesh.NIntervals); .!Mesh.Fil["q0"]]]'
    MyDyvals[[.!Mesh.Fil["p0"]; falses(Model.NPhases * Mesh.NIntervals); .!Mesh.Fil["q0"]]]'
    [(sum(repeat(sims.X,1,Model.NPhases).*(sims.φ.==[1 2 3]).==Nodes[1],dims=1)./NSim)[Model.C.<=0];
    (sum(repeat(sims.X,1,Model.NPhases).*(sims.φ.==[1 2 3]).==Nodes[end],dims=1)./NSim)[Model.C.>=0]]'
]
SFFM.MyPrint([".";"pm";"pmR";"pmMyD";"sim"])
SFFM.MyPrint(pmdata)

display(p)

# plot sims
H = zeros(length(Nodes) - 1, Model.NPhases)
for whichφ = 1:Model.NPhases
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
    h = fit(
        Histogram,
        sims.X[(sims.φ.==whichφ) .& (sims.X.!=Nodes[1]) .& (sims.X.!=Nodes[end])],
        Nodes,
    )
    h = h.weights ./ NSim
    H[:, whichφ] = h
    #p = plot!(
    #    Nodes[1:end-1] + diff(Nodes) / 2,
    #    h,
    #    label = "hist" * string(whichφ),
    #)
    p = bar!(
        (Nodes[1:end-1] + Nodes[2:end]) / 2,
        h,
        alpha = 0.2,
        bar_width = Mesh.Δ,
        label = "sims",
        subplot = whichφ,
    )
end
display(p)

# display errors
err = H - Y
errR = H - YR
MyDerr = H - MyY
plot(
    Nodes[1:end-1] + diff(Nodes) / 2,
    err,
    label = "err",
    legend = :topleft,
    layout = (3, 1),
)
plot!(Nodes[1:end-1] + diff(Nodes) / 2, errR, label = "errR")
plot!(Nodes[1:end-1] + diff(Nodes) / 2, MyDerr, label = "MyDerr")
display(sum(abs.(err) ))
display(sum(abs.(errR)))
display(sum(abs.(MyDerr)))
display(abs.(err))
display(abs.(errR))
