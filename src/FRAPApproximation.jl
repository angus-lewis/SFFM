struct FRAPMesh <: Mesh 
    Nodes::Array{Float64,1}
    NBases::Int
    Fil::Dict{String,BitArray{1}}
    function FRAPMesh(
        model::SFFM.Model,
        Nodes::Array{<:Real,1},
        NBases::Int;
        Fil::Dict{String,BitArray{1}}=Dict{String,BitArray{1}}(),
        v::Bool = false,
    )

        ## Construct the sets Fᵐ = ⋃ᵢ Fᵢᵐ, global index for sets of type m
        if isempty(Fil)
            Fil = MakeFil(model, Nodes)
        end

        mesh = new(
            Nodes,
            NBases,
            Fil,
        )
        v && println("UPDATE: DGMesh object created with fields ", fieldnames(SFFM.DGMesh))
        return mesh
    end
    function FRAPMesh()
        new(
            Array{Float64,1}(undef,0),
            0,
            Dict{String,BitArray{1}}(),
        )
    end
end 

"""

    NBases(mesh::FRAPMesh)

Number of bases in a cell
"""
NBases(mesh::FRAPMesh) = mesh.NBases


"""

    CellNodes(mesh::FRAPMesh)

The cell centre
"""
CellNodes(mesh::FRAPMesh) = Array(((mesh.Nodes[1:end-1] + mesh.Nodes[2:end]) / 2 )')

"""

    Basis(mesh::FRAPMesh)

Constant ""
"""
Basis(mesh::FRAPMesh) = ""

function MakeBFRAP(model::Model, mesh::FRAPMesh, me::ME)
    N₊ = sum(model.C .>= 0)
    N₋ = sum(model.C .<= 0)

    F = Dict{String,SparseArrays.SparseMatrixCSC{Float64,Int64}}()
    UpDiagBlock = me.s*me.a
    LowDiagBlock = me.s*me.a
    for i = ["+","-"]
        F[i] = SparseArrays.spzeros(Float64, TotalNBases(mesh), TotalNBases(mesh))
        for k = 1:NIntervals(mesh)
            idx = (1:NBases(mesh)) .+ (k - 1) * NBases(mesh)
            if k > 1
                idxup = (1:NBases(mesh)) .+ (k - 2) * NBases(mesh)
                if i=="+"
                    F[i][idxup, idx] = UpDiagBlock
                elseif i=="-"
                    F[i][idx, idxup] = LowDiagBlock
                end # end if C[i]
            end # end if k>1
        end # for k in ...
    end # for i in ...

    signChangeIndex = zeros(Bool,NPhases(model),NPhases(model))
    for i in 1:NPhases(model), j in 1:NPhases(model)
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
        NPhases(model) * TotalNBases(mesh) + N₋ + N₊,
        NPhases(model) * TotalNBases(mesh) + N₋ + N₊,
    )
    B[N₋+1:end-N₊,N₋+1:end-N₊] = SparseArrays.kron(
            model.T.*signChangeIndex,
            SparseArrays.kron(SparseArrays.I(NIntervals(mesh)),me.D)
        ) + SparseArrays.kron(
            model.T.*(1 .- signChangeIndex),
            SparseArrays.I(TotalNBases(mesh))
        )
    
    ## Make QBD index
    c = N₋
    QBDidx = zeros(Int, NPhases(model) * TotalNBases(mesh) + N₊ + N₋)
    for k = 1:NIntervals(mesh), i = 1:NPhases(model), n = 1:NBases(mesh)
        c += 1
        QBDidx[c] = (i - 1) * TotalNBases(mesh) + (k - 1) * NBases(mesh) + n + N₋
    end
    QBDidx[1:N₋] = 1:N₋
    QBDidx[(end-N₊+1):end] = (NPhases(model) * TotalNBases(mesh) + N₋) .+ (1:N₊)
    
    # Boundary conditions
    T₋₋ = model.T[model.C.<=0,model.C.<=0]
    T₊₋ = model.T[model.C.>=0,:].*((model.C.<0)')
    T₋₊ = model.T[model.C.<=0,:].*((model.C.>0)')
    T₊₊ = model.T[model.C.>=0,model.C.>=0]
    # yuck
    inLower = [
        LinearAlgebra.kron(LinearAlgebra.diagm(abs.(model.C).*(model.C.<=0)),me.s)[:,model.C.<=0]; 
        LinearAlgebra.zeros((NIntervals(mesh)-1)*NPhases(model)*NBases(mesh),N₋)
    ]
    outLower = [
        LinearAlgebra.kron(T₋₊,me.a) LinearAlgebra.zeros(N₋,N₊+(NIntervals(mesh)-1)*NPhases(model)*NBases(mesh))
    ]
    inUpper = [
        LinearAlgebra.zeros((NIntervals(mesh)-1)*NPhases(model)*NBases(mesh),N₊);
        LinearAlgebra.kron(LinearAlgebra.diagm(abs.(model.C).*(model.C.>=0)),me.s)[:,model.C.>=0]
    ]
    outUpper = [
        LinearAlgebra.zeros(N₊,N₋+(NIntervals(mesh)-1)*NPhases(model)*NBases(mesh)) LinearAlgebra.kron(T₊₋,me.a)
    ]
    
    B[1:N₋,QBDidx] = [T₋₋ outLower]
    B[end-N₊+1:end,QBDidx] = [outUpper T₊₊]
    B[QBDidx[N₋+1:end-N₊],1:N₋] = inLower
    B[QBDidx[N₋+1:end-N₊],(end-N₊+1):end] = inUpper
    for i = 1:NPhases(model)
        idx = ((i-1)*TotalNBases(mesh)+1:i*TotalNBases(mesh)) .+ N₋
        if model.C[i] > 0
            B[idx, idx] += model.C[i] * (SparseArrays.kron(
                    SparseArrays.I(NIntervals(mesh)), me.S
                    ) + F["+"])
        elseif model.C[i] < 0
            B[idx, idx] += abs(model.C[i]) * (SparseArrays.kron(
                SparseArrays.I(NIntervals(mesh)), me.S
                ) + F["-"])
        end
    end

    BDict = MakeDict(B, model, mesh)

    return (BDict=BDict, B=B, QBDidx=QBDidx)
end
