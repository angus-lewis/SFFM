"""

    FVMesh(
        model::SFFM.Model,
        Nodes::Array{Float64,1};
        Fil::Dict{String,BitArray{1}}=Dict{String,BitArray{1}}(),
    ) 

Constructor for a mesh for a finite volume scheme. 
    Inputs: 
     - `model::Model` a Model object
     - `Nodes::Array{Float64,1}` a vector specifying the cell edges
     - `Fil::Dict` an optional dictionary allocating the cells to the sets Fᵢᵐ
"""
struct FVMesh <: SFFM.Mesh 
    Nodes::Array{Float64,1}
    Fil::Dict{String,BitArray{1}}
    function FVMesh(
        model::SFFM.Model,
        Nodes::Array{Float64,1};
        Fil::Dict{String,BitArray{1}}=Dict{String,BitArray{1}}(),
    ) 
        ## Construct the sets Fᵐ = ⋃ᵢ Fᵢᵐ, global index for sets of type m
        if isempty(Fil)
            Fil = MakeFil(model, Nodes)
        end

        new(Nodes, Fil)
    end
end 


"""

    NBases(mesh::FVMesh)

Constant 1
"""
NBases(mesh::FVMesh) = 1


"""

    CellNodes(mesh::FVMesh)

The cell centres
"""
CellNodes(mesh::FVMesh) = Array(((mesh.Nodes[1:end-1] + mesh.Nodes[2:end]) / 2 )')

"""

    Basis(mesh::FVMesh)

Constant ""
"""
Basis(mesh::FVMesh) = ""

function interp(nodes, evalPt)
    order = length(nodes)
    polyCoefs = zeros(order)
    for n in 1:order
        notn = [1:n-1;n+1:order]
        polyCoefs[n] = prod(evalPt.-nodes[notn])./prod(nodes[n].-nodes[notn])
    end
    return polyCoefs
end


function MakeFVFlux(mesh::SFFM.Mesh, order::Int)
    nNodes = TotalNBases(mesh)
    F = zeros(Float64,nNodes,nNodes)
    ptsLHS = Int(ceil(order/2))
    interiorCoeffs = interp(CellNodes(mesh)[1:order],mesh.Nodes[ptsLHS+1])
    for n in 2:nNodes
        evalPt = mesh.Nodes[n]
        if n-ptsLHS-1 < 0
            nodesIdx = 1:order
            nodes = CellNodes(mesh)[nodesIdx]
            coeffs = interp(nodes,evalPt)
        elseif n-ptsLHS-1+order > nNodes
            nodesIdx = (nNodes-order+1):nNodes
            nodes = CellNodes(mesh)[nodesIdx]
            coeffs = interp(nodes,evalPt)
        else
            nodesIdx =  (n-ptsLHS-1) .+ (1:order)
            coeffs = interiorCoeffs
        end
        F[nodesIdx,n-1:n] += [-coeffs coeffs]./Δ(mesh)[1]
    end
    F[end-order+1:end,end] += -interp(CellNodes(mesh)[end-order+1:end],mesh.Nodes[end])./Δ(mesh)[1]
    return F
end

function MakeBFV(model::SFFM.Model, mesh::SFFM.Mesh, order::Int)
    N₊ = sum(model.C .>= 0)
    N₋ = sum(model.C .<= 0)

    F = SFFM.MakeFVFlux(mesh, order)

    B = SparseArrays.spzeros(
        Float64,
        NPhases(model) * NIntervals(mesh) + N₋ + N₊,
        NPhases(model) * NIntervals(mesh) + N₋ + N₊,
    )
    B[N₋+1:end-N₊,N₋+1:end-N₊] = SparseArrays.kron(
            model.T,
            SparseArrays.I(NIntervals(mesh))
        )

         ## Make QBD index
    c = N₋
    QBDidx = zeros(Int, NPhases(model) * NIntervals(mesh) + N₊ + N₋)
    for k = 1:NIntervals(mesh), i = 1:NPhases(model)
        c += 1
        QBDidx[c] = (i - 1) * NIntervals(mesh) + k + N₋
    end
    QBDidx[1:N₋] = 1:N₋
    QBDidx[(end-N₊+1):end] = (NPhases(model) * NIntervals(mesh) + N₋) .+ (1:N₊)
    
    # Boundary conditions
    T₋₋ = model.T[model.C.<=0,model.C.<=0]
    T₊₋ = model.T[model.C.>=0,:].*((model.C.<0)')
    T₋₊ = model.T[model.C.<=0,:].*((model.C.>0)')
    T₊₊ = model.T[model.C.>=0,model.C.>=0]
    # yuck
    begin 
        nodes = CellNodes(mesh)[1:order]
        coeffs = interp(nodes,mesh.Nodes[1])
        idxdown = ((1:order).+TotalNBases(mesh)*(findall(model.C .<= 0) .- 1)')[:] .+ N₋
        B[idxdown, 1:N₋] = LinearAlgebra.kron(
            LinearAlgebra.diagm(0 => model.C[model.C.<=0]),
            -coeffs./Δ(mesh)[1],
        )
    end
    # inLower = [
    #     SparseArrays.diagm(abs.(model.C).*(model.C.<=0))[:,model.C.<=0]; 
    #     SparseArrays.zeros((NIntervals(mesh)-1)*NPhases(model),N₋)
    # ]
    outLower = [
        T₋₊ SparseArrays.zeros(N₋,N₊+(NIntervals(mesh)-1)*NPhases(model))
    ]
    begin
        nodes = CellNodes(mesh)[end-order+1:end]
        coeffs = interp(nodes,mesh.Nodes[end])
        idxup =
            ((1:order).+TotalNBases(mesh)*(findall(model.C .>= 0) .- 1)')[:] .+
            (N₋ + TotalNBases(mesh) - order)
        B[idxup, (end-N₊+1):end] = LinearAlgebra.kron(
            LinearAlgebra.diagm(0 => model.C[model.C.>=0]),
            coeffs./Δ(mesh)[1],
        )
    end
    # inUpper = [
    #     SparseArrays.zeros((NIntervals(mesh)-1)*NPhases(model),N₊);
    #     (SparseArrays.diagm(abs.(model.C).*(model.C.>=0)))[:,model.C.>=0]
    # ]
    outUpper = [
        SparseArrays.zeros(N₊,N₋+(NIntervals(mesh)-1)*NPhases(model)) T₊₋
    ]
    
    B[1:N₋,QBDidx] = [T₋₋ outLower]
    B[end-N₊+1:end,QBDidx] = [outUpper T₊₊]
    # B[QBDidx[N₋+1:end-N₊],1:N₋] = inLower
    # B[QBDidx[N₋+1:end-N₊],(end-N₊+1):end] = inUpper
    for i = 1:NPhases(model)
        idx = ((i-1)*NIntervals(mesh)+1:i*NIntervals(mesh)) .+ N₋
        if model.C[i] > 0
            B[idx, idx] += model.C[i] * F
        elseif model.C[i] < 0
            B[idx, idx] += abs(model.C[i]) * F[end:-1:1,end:-1:1]
        end
    end

    BDict = SFFM.MakeDict(B,model,mesh)

    return (BDict = BDict, B = B, QBDidx = QBDidx)
end