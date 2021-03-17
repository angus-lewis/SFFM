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

    signChangeIndex = zeros(Bool,model.NPhases,model.NPhases)
    for i in 1:model.NPhases, j in 1:model.NPhases
        if ((sign(model.C[i])!=0) && (sign(model.C[j])!=0))
            signChangeIndex[i,j] = (sign(model.C[i])!=sign(model.C[j]))
        elseif (sign(model.C[i])==0)
            signChangeIndex[i,j] = sign(model.C[j])>0
        elseif (sign(model.C[j])==0)
            signChangeIndex[i,j] = sign(model.C[i])>0            
        end
    end
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
            B[idx, idx] += model.C[i] * (SparseArrays.kron(
                    SparseArrays.I(mesh.NIntervals), me.S
                    ) + F["+"])
        elseif model.C[i] < 0
            B[idx, idx] += abs(model.C[i]) * (SparseArrays.kron(
                SparseArrays.I(mesh.NIntervals), me.S
                ) + F["-"])
        end
    end

    BDict = MakeDict(B;model=model,mesh=mesh)

    return (BDict=BDict, B=B, QBDidx=QBDidx)
end
