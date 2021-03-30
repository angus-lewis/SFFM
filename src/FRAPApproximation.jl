struct FRAPMesh <: Mesh 
    NBases::Int
    CellNodes::Array{<:Real,2}
    Fil::Dict{String,BitArray{1}}
    Δ::Array{Float64,1}
    NIntervals::Int
    Nodes::Array{Float64,1}
    TotalNBases::Int
    Basis::String
    function FRAPMesh(
        model::SFFM.Model,
        Nodes::Array{<:Real,1},
        NBases::Int;
        Fil::Dict{String,BitArray{1}}=Dict{String,BitArray{1}}(),
        Basis::String = "lagrange",
    )
        ## Stencil specification
        NIntervals = length(Nodes) - 1 # the number of intervals
        Δ = (Nodes[2:end] - Nodes[1:end-1]) # interval width
        CellNodes = Array(((Nodes[1:end-1] + Nodes[2:end]) / 2 )')

        TotalNBases = NBases * NIntervals # the total number of bases in the stencil

        ## Construct the sets Fᵐ = ⋃ᵢ Fᵢᵐ, global index for sets of type m
        if isempty(Fil)
            idxPlus = model.r.r(Nodes[1:end-1].+Δ[:]/2).>0
            idxZero = model.r.r(Nodes[1:end-1].+Δ[:]/2).==0
            idxMinus = model.r.r(Nodes[1:end-1].+Δ[:]/2).<0
            for i in 1:model.NPhases
                Fil[string(i)*"+"] = idxPlus[:,i]
                Fil[string(i)*"0"] = idxZero[:,i]
                Fil[string(i)*"-"] = idxMinus[:,i]
                if model.C[i] .<= 0
                    Fil["p"*string(i)*"+"] = [model.r.r(model.Bounds[1,1])[i]].>0
                    Fil["p"*string(i)*"0"] = [model.r.r(model.Bounds[1,1])[i]].==0
                    Fil["p"*string(i)*"-"] = [model.r.r(model.Bounds[1,1])[i]].<0
                end
                if model.C[i] .>= 0
                    Fil["q"*string(i)*"+"] = [model.r.r(model.Bounds[1,end])[i]].>0
                    Fil["q"*string(i)*"0"] = [model.r.r(model.Bounds[1,end])[i]].==0
                    Fil["q"*string(i)*"-"] = [model.r.r(model.Bounds[1,end])[i]].<0
                end
            end
        end
        CurrKeys = keys(Fil)
        for ℓ in ["+", "-", "0"], i = 1:model.NPhases
            if !in(string(i) * ℓ, CurrKeys)
                Fil[string(i)*ℓ] = falses(NIntervals)
            end
            if !in("p" * string(i) * ℓ, CurrKeys) && model.C[i] <= 0
                Fil["p"*string(i)*ℓ] = falses(1)
            end
            if !in("p" * string(i) * ℓ, CurrKeys) && model.C[i] > 0
                Fil["p"*string(i)*ℓ] = falses(0)
            end
            if !in("q" * string(i) * ℓ, CurrKeys) && model.C[i] >= 0
                Fil["q"*string(i)*ℓ] = falses(1)
            end
            if !in("q" * string(i) * ℓ, CurrKeys) && model.C[i] < 0
                Fil["q"*string(i)*ℓ] = falses(0)
            end
        end
        for ℓ in ["+", "-", "0"]
            Fil[ℓ] = falses(NIntervals * model.NPhases)
            Fil["p"*ℓ] = trues(0)
            Fil["q"*ℓ] = trues(0)
            for i = 1:model.NPhases
                idx = findall(Fil[string(i)*ℓ]) .+ (i - 1) * NIntervals
                Fil[string(ℓ)][idx] .= true
                Fil["p"*ℓ] = [Fil["p"*ℓ]; Fil["p"*string(i)*ℓ]]
                Fil["q"*ℓ] = [Fil["q"*ℓ]; Fil["q"*string(i)*ℓ]]
            end
        end

        mesh = new(
            NBases,
            CellNodes,
            Fil,
            Δ,
            NIntervals,
            Nodes,
            TotalNBases,
            Basis,
        )
        println("UPDATE: DGMesh object created with fields ", fieldnames(SFFM.DGMesh))
        return mesh
    end
    function FRAPMesh()
        new(
            0,
            Array{Real,2}(undef,0,0),
            Dict{String,BitArray{1}}(),
            Array{Float64,1}(undef,0),
            0,
            Array{Float64,1}(undef,0),
            0,
            "",
        )
    end
end 

function MakeBFRAP(model::Model, mesh::FRAPMesh, me::ME)
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
        LinearAlgebra.kron(T₋₊,me.a) LinearAlgebra.zeros(N₋,N₊+(mesh.NIntervals-1)*model.NPhases*mesh.NBases)
    ]
    inUpper = [
        LinearAlgebra.zeros((mesh.NIntervals-1)*model.NPhases*mesh.NBases,N₊);
        LinearAlgebra.kron(LinearAlgebra.diagm(abs.(model.C).*(model.C.>=0)),me.s)[:,model.C.>=0]
    ]
    outUpper = [
        LinearAlgebra.zeros(N₊,N₋+(mesh.NIntervals-1)*model.NPhases*mesh.NBases) LinearAlgebra.kron(T₊₋,me.a)
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

    BDict = MakeDict(B, model, mesh)

    return (BDict=BDict, B=B, QBDidx=QBDidx)
end
