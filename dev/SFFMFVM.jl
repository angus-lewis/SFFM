include(pwd()*"/src/SFFM.jl")
include(pwd()*"/examples/meNumerics/discontinuitiesModelDef.jl")

Δ = 1 
nodes = collect(0:Δ:bounds[1,2])
mesh = SFFM.MakeMesh(
    model = model, 
    Nodes = nodes, 
    NBases = 1,
    Basis = "lagrange",
)

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
    nNodes = length(mesh.CellNodes)
    F = zeros(Float64,nNodes,nNodes)
    ptsLHS = Int(ceil(order/2))
    interiorCoeffs = interp(mesh.CellNodes[1:order],mesh.Nodes[ptsLHS+1])
    for n in 2:nNodes
        evalPt = mesh.Nodes[n]
        if n-ptsLHS-1 < 0
            nodesIdx = 1:order
            nodes = mesh.CellNodes[nodesIdx]
            coeffs = interp(nodes,evalPt)
        elseif n-ptsLHS-1+order > nNodes
            nodesIdx = (nNodes-order+1):nNodes
            nodes = mesh.CellNodes[nodesIdx]
            coeffs = interp(nodes,evalPt)
        else
            nodesIdx =  (n-ptsLHS-1) .+ (1:order)
            coeffs = interiorCoeffs
        end
        F[nodesIdx,n-1:n] += [-coeffs coeffs]
    end
    F[end-order+1:end,end] += -interp(mesh.CellNodes[end-order+1:end],mesh.Nodes[end])
    return F
end

function MakeBFV(model::SFFM.Model, mesh::SFFM.Mesh, order::Int)
    N₊ = sum(model.C .>= 0)
    N₋ = sum(model.C .<= 0)

    F = MakeFVMFlux(mesh,order)

    B = SparseArrays.spzeros(
        Float64,
        model.NPhases * mesh.NIntervals + N₋ + N₊,
        model.NPhases * mesh.NIntervals + N₋ + N₊,
    )
    B[N₋+1:end-N₊,N₋+1:end-N₊] = SparseArrays.kron(
            model.T,
            SparseArrays.I(mesh.NIntervals)
        )

         ## Make QBD index
    c = N₋
    QBDidx = zeros(Int, model.NPhases * mesh.NIntervals + N₊ + N₋)
    for k = 1:mesh.NIntervals, i = 1:model.NPhases
        c += 1
        QBDidx[c] = (i - 1) * mesh.NIntervals + k + N₋
    end
    QBDidx[1:N₋] = 1:N₋
    QBDidx[(end-N₊+1):end] = (model.NPhases * mesh.NIntervals + N₋) .+ (1:N₊)
    
    # Boundary conditions
    T₋₋ = model.T[model.C.<=0,model.C.<=0]
    T₊₋ = model.T[model.C.>=0,:].*((model.C.<0)')
    T₋₊ = model.T[model.C.<=0,:].*((model.C.>0)')
    T₊₊ = model.T[model.C.>=0,model.C.>=0]
    # yuck
    inLower = [
        SparseArrays.diagm(abs.(model.C).*(model.C.<=0))[:,model.C.<=0]; 
        SparseArrays.zeros((mesh.NIntervals-1)*model.NPhases,N₋)
    ]
    outLower = [
        T₋₊ SparseArrays.zeros(N₋,N₊+(mesh.NIntervals-1)*model.NPhases)
    ]
    inUpper = [
        SparseArrays.zeros((mesh.NIntervals-1)*model.NPhases,N₊);
        (SparseArrays.diagm(abs.(model.C).*(model.C.>=0)))[:,model.C.>=0]
    ]
    outUpper = [
        SparseArrays.zeros(N₊,N₋+(mesh.NIntervals-1)*model.NPhases) T₊₋
    ]
    
    B[1:N₋,QBDidx] = [T₋₋ outLower]
    B[end-N₊+1:end,QBDidx] = [outUpper T₊₊]
    B[QBDidx[N₋+1:end-N₊],1:N₋] = inLower
    B[QBDidx[N₋+1:end-N₊],(end-N₊+1):end] = inUpper
    for i = 1:model.NPhases
        idx = ((i-1)*mesh.NIntervals+1:i*mesh.NIntervals) .+ N₋
        if model.C[i] > 0
            B[idx, idx] += model.C[i] * F
        elseif model.C[i] < 0
            B[idx, idx] += abs(model.C[i]) * F[end:-1:1,end:-1:1]
        end
    end

    BDict = SFFM.MakeDict(B;model=model,mesh=mesh)

    return (B = B, BDict = BDict, QBDidx = QBDidx)
end