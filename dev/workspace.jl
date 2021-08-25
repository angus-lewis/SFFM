using SFFM

## define a model
T = [0.0]
C = [1]

rfun(x) = x.*0
Rfun(x) = r(x)

r = (
    r = function (x)
        rfun(x)
    end,
    R = function (x)
        Rfun(x)
    end
)

bounds = [0 12; -Inf Inf]
model = SFFM.Model( T, C, r, Bounds = bounds)

orders = [1;3;5;7;11;13;15;21;25;27;33;;37;41]
errors_1 = []

Δtemp = 1/2 # the grid size; must have kΔ = 1 for some k due to discontinuity in r at 1
nodes = collect(0:Δtemp:bounds[1,2])

order = 3
println("order = "*string(order))
dgmesh = SFFM.DGMesh(
    model, 
    nodes, 
    order,
    Basis = "lagrange",
)
frapmesh = SFFM.FRAPMesh(
    model, 
    nodes, 
    order,
)
fvmesh = SFFM.FVMesh(
    model, 
    collect(0:Δtemp/order:bounds[1,2]), 
)
simmesh = SFFM.FVMesh(
    model, 
    nodes, 
)

trueprobs = zeros(Float64,1,SFFM.NIntervals(frapmesh),SFFM.NPhases(model))
truepos = convert(Int,ceil((1.2+eps())/Δtemp))
trueprobs[truepos] = 1
groundtruth = SFFM.SFFMProbability(
    [0],
    trueprobs,
    SFFM.CellNodes(frapmesh),
)

# DG
B_DG = SFFM.MakeB(model, dgmesh)
#ME
me = SFFM.MakeME(SFFM.CMEParams[order], mean = Δtemp)
B_ME = SFFM.MakeB(model, frapmesh, me)
# Erlang (this is the erlang which is equivalent to DG)
erlang = SFFM.MakeErlang(order, mean = Δtemp)
B_Erlang = SFFM.MakeB(model, frapmesh, erlang)
# meph (this is the erlang treated as an ME)
meph = SFFM.ME(erlang.a, erlang.S, erlang.s; D = SFFM.erlangDParams[string(order)])
B_MEPH = SFFM.MakeB(model, frapmesh, meph)
# FVM
B_FV = SFFM.MakeB(model, fvmesh, 3)

point = 0+eps()
pointIdx = convert(Int,ceil(point/Δtemp))
begin
    V = SFFM.vandermonde(order)
    theNodes = SFFM.CellNodes(dgmesh)[:,pointIdx]
    basisValues = zeros(length(theNodes))
    for n in 1:length(theNodes)
        basisValues[n] = prod(point.-theNodes[[1:n-1;n+1:end]])./prod(theNodes[n].-theNodes[[1:n-1;n+1:end]])
    end
    initpm = [
        zeros(sum(model.C.<=0)) # LHS point mass
        zeros(sum(model.C.>=0)) # RHS point mass
    ]
    initprobs = zeros(Float64,SFFM.NBases(dgmesh),SFFM.NIntervals(dgmesh),SFFM.NPhases(model))
    initprobs[:,pointIdx,1] = basisValues'*V.V*V.V'.*2/Δtemp
    initdist = SFFM.SFFMDensity(
        initpm,
        initprobs,
        SFFM.CellNodes(dgmesh),
    ) # convert to a distribution object so we can apply Dist2Coeffs
    # convert to Coeffs α in the DG context
    x0_DG = SFFM.Dist2Coeffs(model, dgmesh, initdist)
end
begin
    initpm = [
        zeros(sum(model.C.<=0)) # LHS point mass
        zeros(sum(model.C.>=0)) # RHS point mass
    ]
    initprobs = zeros(Float64,SFFM.NBases(frapmesh),SFFM.NIntervals(frapmesh),SFFM.NPhases(model))
    initprobs[:,pointIdx,1] = me.a
    initdist = SFFM.SFFMDensity(
        initpm,
        initprobs,
        SFFM.CellNodes(frapmesh),
    ) # convert to a distribution object so we can apply Dist2Coeffs
    # convert to Coeffs α in the DG context
    x0_ME = SFFM.Dist2Coeffs( model, frapmesh, initdist)
end
begin
    initpm = [
        zeros(sum(model.C.<=0)) # LHS point mass
        zeros(sum(model.C.>=0)) # RHS point mass
    ]
    initprobs = zeros(Float64,SFFM.NBases(frapmesh),SFFM.NIntervals(frapmesh),SFFM.NPhases(model))
    initprobs[1,pointIdx,1] = 1
    initdist = SFFM.SFFMDensity(
        initpm,
        initprobs,
        SFFM.CellNodes(frapmesh),
    ) # convert to a distribution object so we can apply Dist2Coeffs
    # convert to Coeffs α in the DG context
    x0_Erlang = SFFM.Dist2Coeffs( model, frapmesh, initdist)
end
begin
    initpm = [
        zeros(sum(model.C.<=0)) # LHS point mass
        zeros(sum(model.C.>=0)) # RHS point mass
    ]
    initprobs = zeros(Float64,SFFM.NBases(fvmesh),SFFM.NIntervals(fvmesh),SFFM.NPhases(model))
    initprobs[1,pointIdx,1] = 1
    initdist = SFFM.SFFMDensity(
        initpm,
        initprobs,
        SFFM.CellNodes(fvmesh),
    ) # convert to a distribution object so we can apply Dist2Coeffs
    # convert to Coeffs α in the DG context
    x0_FV = SFFM.Dist2Coeffs( model, fvmesh, initdist)
end

euler(B,x0) = SFFM.EulerDG( B, 1.2, x0, h = 0.0001) 
x1_DG = euler(B_DG.B, x0_DG)
x1_ME = euler(B_ME.B, x0_ME)
x1_Erlang = euler(B_Erlang.B, x0_Erlang)
x1_MEPH = euler(B_MEPH.B, x0_Erlang)
x1_FV = euler(B_FV.B, x0_FV)

x1_DG = SFFM.Coeffs2Dist(
    model,
    dgmesh,
    x1_DG,
    SFFM.SFFMProbability,
)
x1_ME = SFFM.Coeffs2Dist(
    model,
    frapmesh,
    x1_ME,
    SFFM.SFFMProbability,
)
x1_Erlang = SFFM.Coeffs2Dist(
    model,
    frapmesh,
    x1_Erlang,
    SFFM.SFFMProbability,
)
x1_MEPH = SFFM.Coeffs2Dist(
    model,
    frapmesh,
    x1_MEPH,
    SFFM.SFFMProbability,
)

x1_FV = SFFM.Coeffs2Dist(
    model,
    frapmesh,
    x1_FV,
    SFFM.SFFMProbability,
)

errVec_1 = (
    SFFM.starSeminorm(x1_DG, groundtruth),
    SFFM.starSeminorm(x1_ME, groundtruth),
    SFFM.starSeminorm(x1_Erlang, groundtruth),
    SFFM.starSeminorm(x1_MEPH, groundtruth),
    SFFM.starSeminorm(x1_FV, groundtruth),
)
# p = SFFM.plot(model, frapmesh, x1_ME,
#     color = 1, label = "DG")
# display(p)

# p = SFFM.plot(model, dgmesh, x1_DG,
#     color = 1, label = "DG", alpha = 0.25)
# p = SFFM.plot!(p, model, frapmesh, x1_ME, 
#     color = 2, label = "ME", alpha = 0.25)
# # SFFM.plot!(p, model, frapmesh, x1_Erlang, 
# #     color = 3, label = "Erlang", alpha = 0.25)
# # SFFM.plot!(p, model, frapmesh, x1_MEPH, 
# #     color = 5, label = "ME-PH", alpha = 0.25)
# SFFM.plot!(p, model, frapmesh, x1_FV, 
#     color = 7, label = "FV", alpha = 0.25)
# p = plot!(title = "approx dist at t=1.2; order = "*string(order), subplot = 1)
# display(p)

# push!(errors_1, errVec_1)

function vmult(u,v)
    s_1 = length(u)
    s_2 = length(v)
    if s_1 != s_2 
        throw(DomainError("Dimension mismatch"))
    end
    w = 0
    for i in 1:s_1
        w += u[i]*v[i]
    end
    return w
end

function mmult(A,B)
    s_1 = size(A,1)
    s_2 = size(A,2)
    s_3 = size(B,1)
    s_4 = size(B,2)
    if s_2 != s_3 
        throw(DomainError("Dimension mismatch"))
    end
    M = zeros(s_1,s_4)
    for i in 1:s_1
        for j in 1:s_4
            w = vmult(A[i,:],B[:,j])
            M[i,j] = w
        end
    end
    return M
end

function vmmult(u,A)
    s_1 = length(u)
    s_2 = size(A,1)
    s_3 = size(A,2)
    if s_1 != s_2 
        throw(DomainError("Dimension mismatch"))
    end
    v = zeros(1,s_3)
    for j in 1:s_3
        temp = 0 
        for i in 1:s_1
            temp += u[i]*A[i,j]
        end
        v[j] = temp
    end
    return v
end




# blocks = (left, middle, right)
# blocks = (down, stay, up)
s1 = rand(2:3)
s2 = rand(2:3)
s3 = rand(2:3)
blocks = (rand(s1,s1), rand(s1,s1), rand(s1,s1))
T = rand(s2,s2)
size_T = size(T,1)
C = rand(-2:2,s2)
signChangeIndex = zeros(Bool,size_T,size_T)
    for i in 1:size_T, j in 1:size_T
        if ((sign(C[i])!=0) && (sign(C[j])!=0))
            signChangeIndex[i,j] = (sign(C[i])!=sign(C[j]))
        elseif (sign(C[i])==0)
            signChangeIndex[i,j] = sign(C[j])>0
        elseif (sign(C[j])==0)
            signChangeIndex[i,j] = sign(C[i])>0            
        end
    end
size_blocks = size(blocks[1],1)
D = rand(s1,s1)
delta = cumsum(1:s3)
size_delta = length(delta)
u = rand(size_T*size_blocks*size_delta)#zeros(size_T*size_blocks*size_delta)# collect(1:(size_T*size_blocks*size_delta))
# u[1] = 1
size_u = length(u)
v = zeros(1,size_u)
for i in 1:size_T
    for j in 1:size_T
        if i == j
            for k in 1:size_delta
                k_idx = (i-1)*size_blocks*size_delta .+ (k-1)*size_blocks .+ (1:size_blocks)
                for ℓ in 1:size_delta
                    ℓ_idx = (i-1)*size_blocks*size_delta .+ (ℓ-1)*size_blocks .+ (1:size_blocks)
                    if k == ℓ+1
                        v[k_idx] += (u[ℓ_idx]'*blocks[3])'
                    elseif k == ℓ
                        v[k_idx] += (u[ℓ_idx]'*(blocks[2] + T[i,j]*I))'
                    elseif k == ℓ-1
                        v[k_idx] += (u[ℓ_idx]'*blocks[1])'
                    end
                end
            end
        elseif signChangeIndex[i,j]
            for k in 1:size_delta
                for ℓ in 1:size_delta
                    if k == ℓ
                        i_idx = (i-1)*size_blocks*size_delta .+ (k-1)*size_blocks .+ (1:size_blocks)
                        j_idx = (j-1)*size_blocks*size_delta .+ (k-1)*size_blocks .+ (1:size_blocks)
                        v[j_idx] += (u[i_idx]'*(T[i,j]*D))'
                    end
                end
            end
        else
            i_idx = (i-1)*size_blocks*size_delta .+ (1:size_blocks*size_delta)
            j_idx = (j-1)*size_blocks*size_delta .+ (1:size_blocks*size_delta)
            v[j_idx] += (u[i_idx]'*T[i,j])'
        end
    end
end    

full = zeros(size_u,size_u)
for i in 1:size_T
    for j in 1:size_T
        if i == j
            for k in 1:size_delta
                k_idx = (i-1)*size_blocks*size_delta .+ (k-1)*size_blocks .+ (1:size_blocks)
                for ℓ in 1:size_delta
                    ℓ_idx = (i-1)*size_blocks*size_delta .+ (ℓ-1)*size_blocks .+ (1:size_blocks)
                    if k == ℓ+1
                        full[ℓ_idx,k_idx] = (blocks[3])
                    elseif k == ℓ
                        full[ℓ_idx,k_idx] = (blocks[2])
                    elseif k == ℓ-1
                        full[ℓ_idx,k_idx]  = (blocks[1])
                    end
                end
            end
        end
    end
end  
Dtemp = kron(T.*signChangeIndex,kron(I(size_delta),D)) + kron(T.*.!signChangeIndex,kron(I(size_delta),I(size_blocks)))
full = Dtemp + full
sum(abs.(u'*full-v))








1