# include(pwd()*"/src/SFFM.jl")
# include(pwd()*"/examples/meNumerics/discontinuitiesModelDef.jl")

struct FVMesh <: SFFM.Mesh 
    NBases::Int
    CellNodes::Array{<:Real,2}
    Fil::Dict{String,BitArray{1}}
    Δ::Array{Float64,1}
    NIntervals::Int
    Nodes::Array{Float64,1}
    TotalNBases::Int
    Basis::String
    function FVMesh(
        model::SFFM.Model,
        Nodes::Array{Float64,1};
        Fil::Dict{String,BitArray{1}}=Dict{String,BitArray{1}}(),
    )
        NBases = 1
        NIntervals = length(Nodes) - 1 # the number of intervals
        Δ = (Nodes[2:end] - Nodes[1:end-1]) # interval width
        CellNodes = Array(((Nodes[1:end-1] + Nodes[2:end]) / 2 )')

        TotalNBases = NBases * NIntervals
        
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
        new(NBases, CellNodes, Fil, Δ, NIntervals, Nodes, TotalNBases, "lagrange")
    end
end 

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
        F[nodesIdx,n-1:n] += [-coeffs coeffs]./mesh.Δ[1]
    end
    F[end-order+1:end,end] += -interp(mesh.CellNodes[end-order+1:end],mesh.Nodes[end])./mesh.Δ[1]
    return F
end

function MakeBFV(model::SFFM.Model, mesh::SFFM.Mesh, order::Int)
    N₊ = sum(model.C .>= 0)
    N₋ = sum(model.C .<= 0)

    F = SFFM.MakeFVFlux(mesh, order)

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
    begin 
        nodes = mesh.CellNodes[1:order]
        coeffs = interp(nodes,mesh.Nodes[1])
        idxdown = ((1:order).+mesh.TotalNBases*(findall(model.C .<= 0) .- 1)')[:] .+ N₋
        B[idxdown, 1:N₋] = LinearAlgebra.kron(
            LinearAlgebra.diagm(0 => model.C[model.C.<=0]),
            -coeffs./mesh.Δ[1],
        )
    end
    # inLower = [
    #     SparseArrays.diagm(abs.(model.C).*(model.C.<=0))[:,model.C.<=0]; 
    #     SparseArrays.zeros((mesh.NIntervals-1)*model.NPhases,N₋)
    # ]
    outLower = [
        T₋₊ SparseArrays.zeros(N₋,N₊+(mesh.NIntervals-1)*model.NPhases)
    ]
    begin
        nodes = mesh.CellNodes[end-order+1:end]
        coeffs = interp(nodes,mesh.Nodes[end])
        idxup =
            ((1:order).+mesh.TotalNBases*(findall(model.C .>= 0) .- 1)')[:] .+
            (N₋ + mesh.TotalNBases - order)
        B[idxup, (end-N₊+1):end] = LinearAlgebra.kron(
            LinearAlgebra.diagm(0 => model.C[model.C.>=0]),
            coeffs./mesh.Δ[1],
        )
    end
    # inUpper = [
    #     SparseArrays.zeros((mesh.NIntervals-1)*model.NPhases,N₊);
    #     (SparseArrays.diagm(abs.(model.C).*(model.C.>=0)))[:,model.C.>=0]
    # ]
    outUpper = [
        SparseArrays.zeros(N₊,N₋+(mesh.NIntervals-1)*model.NPhases) T₊₋
    ]
    
    B[1:N₋,QBDidx] = [T₋₋ outLower]
    B[end-N₊+1:end,QBDidx] = [outUpper T₊₊]
    # B[QBDidx[N₋+1:end-N₊],1:N₋] = inLower
    # B[QBDidx[N₋+1:end-N₊],(end-N₊+1):end] = inUpper
    for i = 1:model.NPhases
        idx = ((i-1)*mesh.NIntervals+1:i*mesh.NIntervals) .+ N₋
        if model.C[i] > 0
            B[idx, idx] += model.C[i] * F
        elseif model.C[i] < 0
            B[idx, idx] += abs(model.C[i]) * F[end:-1:1,end:-1:1]
        end
    end

    BDict = SFFM.MakeDict(B,model,mesh)

    return (BDict = BDict, B = B, QBDidx = QBDidx)
end