function MakeBFRAP(;model::SFFM.Model, mesh::SFFM.Mesh, me::SFFM.ME)
    N₊ = sum(model.C .>= 0)
    N₋ = sum(model.C .<= 0)

    F = Dict{String,SparseArrays.SparseMatrixCSC{Float64,Int64}}()
    UpDiagBlock = me.s*me.a
    LowDiagBlock = me.s*me.a
    for i = ["+","-"]
        F[i] = SparseArrays.spzeros(Float64, mesh.TotalNBases, mesh.TotalNBases)
        for k = 1:mesh.NIntervals
            idx = (1:mesh.NBases) .+ (k - 1) * mesh.NBases
            if k > 1
                idxup = (1:mesh.NBases) .+ (k - 2) * mesh.NBases
                if i=="+"
                    F[i][idxup, idx] = UpDiagBlock
                elseif i=="-"
                    F[i][idx, idxup] = LowDiagBlock
                end # end if C[i]
            end # end if k>1
        end # for k in ...
    end # for i in ...

    Q = Array{SparseArrays.SparseMatrixCSC{Float64,Int64},1}(undef,model.NPhases)
    for i = 1:model.NPhases
        Q[i] = abs(model.C[i]) * me.S
    end

    signChangeIndex = .!((sign.(model.C).<=0)*(sign.(model.C).<=0)')
    signChangeIndex = signChangeIndex - 
        LinearAlgebra.diagm(LinearAlgebra.diag(signChangeIndex))
    B = SparseArrays.spzeros(
        Float64,
        model.NPhases * mesh.TotalNBases + N₋ + N₊,
        model.NPhases * mesh.TotalNBases + N₋ + N₊,
    )
    B[N₋+1:end-N₊,N₋+1:end-N₊] = SparseArrays.kron(
            model.T.*signChangeIndex,
            SparseArrays.kron(SparseArrays.I(mesh.NIntervals),me.D)
        ) + SparseArrays.kron(
            model.T.*(1 .- signChangeIndex),
            SparseArrays.I(mesh.TotalNBases)
        )
    
    ## Make QBD index
    c = N₋
    QBDidx = zeros(Int, model.NPhases * mesh.TotalNBases + N₊ + N₋)
    for k = 1:mesh.NIntervals, i = 1:model.NPhases, n = 1:mesh.NBases
        c += 1
        QBDidx[c] = (i - 1) * mesh.TotalNBases + (k - 1) * mesh.NBases + n + N₋
    end
    QBDidx[1:N₋] = 1:N₋
    QBDidx[(end-N₊+1):end] = (model.NPhases * mesh.TotalNBases + N₋) .+ (1:N₊)
    
    # Boundary conditions
    T₋₋ = model.T[model.C.<=0,model.C.<=0]
    T₊₋ = model.T[model.C.>=0,:].*((model.C.<0)')
    T₋₊ = model.T[model.C.<=0,:].*((model.C.>0)')
    T₊₊ = model.T[model.C.>=0,model.C.>=0]
    # yuck
    inLower = [
        LinearAlgebra.kron(LinearAlgebra.diagm(abs.(model.C).*(model.C.<=0)),me.s)[:,model.C.<=0]; 
        LinearAlgebra.zeros((mesh.NIntervals-1)*model.NPhases*mesh.NBases,N₋)
    ]
    outLower = [
        LinearAlgebra.kron(1,kron(T₋₊,me.a)) LinearAlgebra.zeros(N₋,N₊+(mesh.NIntervals-1)*model.NPhases*mesh.NBases)
    ]
    inUpper = [
        LinearAlgebra.zeros((mesh.NIntervals-1)*model.NPhases*mesh.NBases,N₊);
        LinearAlgebra.kron(LinearAlgebra.diagm(abs.(model.C).*(model.C.>=0)),me.s)[:,model.C.>=0]
    ]
    outUpper = [
        LinearAlgebra.zeros(N₊,N₋+(mesh.NIntervals-1)*model.NPhases*mesh.NBases) LinearAlgebra.kron(1,kron(T₊₋,me.a))
    ]
    
    B[1:N₋,QBDidx] = [T₋₋ outLower]
    B[end-N₊+1:end,QBDidx] = [outUpper T₊₊]
    B[QBDidx[N₋+1:end-N₊],1:N₋] = inLower
    B[QBDidx[N₋+1:end-N₊],(end-N₊+1):end] = inUpper
    for i = 1:model.NPhases
        idx = ((i-1)*mesh.TotalNBases+1:i*mesh.TotalNBases) .+ N₋
        if model.C[i] > 0
            B[idx, idx] += model.C[i] * SparseArrays.kron(
                    SparseArrays.I(mesh.NIntervals),me.S
                    ) + model.C[i]*F["+"]
        elseif model.C[i] < 0
            B[idx, idx] += abs(model.C[i]) * SparseArrays.kron(
                SparseArrays.I(mesh.NIntervals),me.S
                ) + abs(model.C[i])*F["-"]
        end
    end

    BDict = MakeDict(B;model=model,mesh=mesh)

    return (BDict=BDict, B=B, QBDidx=QBDidx)
end

function MakeGlobalApprox(;NCells = 3,up, down,T,C,bkwd=false,D=[],plusI = false)
    αup,Qup = up
    αdown,Qdown = down
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
    T₊₋ = T[C.>=0,:].*((C.<0)')
    T₋₊ = T[C.<=0,:].*((C.>0)')
    T₊₊ = T[C.>=0,C.>=0]

    inLower = [kron(diagm(abs.(C).*(C.<0)),qdown)[:,C.<0]; zeros((NCells-1)*NPhases*NBases,N₋)]
    outLower = [kron(1,kron(T₋₊,αup)) zeros(N₋,N₊+(NCells-1)*NPhases*NBases)]
    inUpper = [zeros((NCells-1)*NPhases*NBases,N₊);kron(diagm(abs.(C).*(C.>0)),qup)[:,C.>0]]
    outUpper = [zeros(N₊,N₋+(NCells-1)*NPhases*NBases) kron(1,kron(T₊₋,αdown))]
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
    if bkwd
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
        for c in 1:NCells, i in 1:length(C), j in 1:length(C)
            idxi = ((i-1)*NBases) .+ (1:NBases) .+ ((c-1)*NBases*NPhases)
            idxj = ((j-1)*NBases) .+ (1:NBases) .+ ((c-1)*NBases*NPhases)
            if i!=j
                if C[i]>0 && C[j]<0
                    # display(Qup+T[i,i]*I(NBases)*plusI)
                    # display(Qup)
                    πtemp = -αup*(abs(C[i])*Qup-T[i,i]*I(NBases)*plusI)^-1
                    πtemp = πtemp./sum(πtemp)
                    D[idxi,idxj] = T[i,j].*repeat(πtemp,NBases,1)
                elseif C[i]<0 && C[j]>0
                    πtemp = -αdown*(abs(C[i])*Qdown-T[i,i]*I(NBases)*plusI)^-1
                    πtemp = πtemp./sum(πtemp)
                    D[idxi,idxj] = T[i,j].*repeat(πtemp,NBases,1)
                else
                    D[idxi,idxj] = T[i,j].*I(NBases)
                end
            else
                D[idxi,idxj] = T[i,i].*I(NBases)
            end
        end
        B = [
            T₋₋ outLower;
            inLower D+Q inUpper;
            outUpper T₊₊;
        ]
        # B = [
        #     T₋₋ outLower;
        #     inLower (#kron(I(NCells),kron(Tdiag,I(NBases)))
        #         +kron(I(NCells),kron(Tnochange,I2))
        #         +kron(I(NCells),kron(T₋₊,D^-1))
        #         +kron(I(NCells),kron(T₊₋,D))
        #         +Q) inUpper;
        #     outUpper T₊₊;
        # ]
    else
        B = [
            T₋₋ outLower;
            inLower kron(I(NCells),kron(T,I(NBases)))+Q inUpper;
            outUpper T₊₊;
        ]
    end
    # display(Q)
    return Q, B
end