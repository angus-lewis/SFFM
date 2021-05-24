abstract type SFFMDistribution end

"""

    SFFMDistribution(
        pm::Array{<:Real},
        distribution::Array{<:Real,3},
        x::Array{<:Real},
        type::String,
    )

- `pm::Array{Float64}`: a vector containing the point masses, the first
    `sum(model.C.<=0)` entries are the left hand point masses and the last
    `sum(model.C.>=0)` are the right-hand point masses.
- `distribution::Array{Float64,3}`: "probability" or "density"` 
- `x::Array{Float64,2}`:
    - if `type="probability"` is a `1×NIntervals×NPhases` array
        containing the cell centers.
    - if `type="density"` is a `NBases×NIntervals×NPhases` array
        containing the cell nodes at which the denisty is evaluated.
- `type::String`: either `"probability"` or `"density"`. `"cumulative"` is
    not possible.
"""
struct SFFMDensity <: SFFMDistribution
    pm::Array{<:Real}
    distribution::Array{<:Real,3}
    x::Array{<:Real}
end
struct SFFMProbability <: SFFMDistribution
    pm::Array{<:Real}
    distribution::Array{<:Real,3}
    x::Array{<:Real}
end
struct SFFMCDF <: SFFMDistribution
    pm::Array{<:Real}
    distribution::Array{<:Real,3}
    x::Array{<:Real}
end

"""
Convert from a vector of coefficients for the DG system to a distribution.

    Coeffs2Dist(
        model::SFFM.Model,
        mesh::SFFM.Mesh,
        Coeffs;
        type::String = "probability",
    )

# Arguments
- `model`: a Model object
- `mesh`: a Mesh object as output from MakeMesh
- `Coeffs::Array`: a vector of coefficients from the DG method
- `type::String`: an (optional) declaration of what type of distribution you
    want to convert to. Options are `"probability"` to return the probabilities
    ``P(X(t)∈ D_k, φ(t) = i)`` where ``D_k``is the kth cell, `"cumulative"` to
    return the CDF evaluated at cell edges, or `"density"` to return an
    approximation to the density ar at the SFFM.CellNodes(mesh).

# Output
- a tuple with keys
(pm=pm, distribution=yvals, x=xvals, type=type)
    - `pm::Array{Float64}`: a vector containing the point masses, the first
        `sum(model.C.<=0)` entries are the left hand point masses and the last
        `sum(model.C.>=0)` are the right-hand point masses.
    - `distribution::Array{Float64,3}`:
        - if `type="cumulative"` returns a `2×NIntervals×NPhases` array
            containing the CDF evaluated at the cell edges as contained in
            `x` below. i.e. `distribution[1,:,i]` returns the cdf at the
            left-hand edges of the cells in phase `i` and `distribution[2,:,i]`
            at the right hand edges.
        - if `type="probability"` returns a `1×NIntervals×NPhases` array
            containing the probabilities ``P(X(t)∈ D_k, φ(t) = i)`` where ``D_k``
            is the kth cell.
        - if `type="density"` returns a `NBases×NIntervals×NPhases` array
            containing the density function evaluated at the cell nodes as
            contained in `x` below.
    - `x::Array{Float64,2}`:
        - if `type="cumulative"` returns a `2×NIntervals×NPhases` array
            containing the cell edges as contained. i.e. `x[1,:]`
            returns the left-hand edges of the cells and `x[2,:]` at the
            right-hand edges.
        - if `type="probability"` returns a `1×NIntervals×NPhases` array
            containing the cell centers.
        - if `type="density"` returns a `NBases×NIntervals×NPhases` array
            containing the cell nodes.
    - `type`: as input in arguments.
"""
function Coeffs2Dist(
    model::SFFM.Model,
    mesh::SFFM.DGMesh,
    Coeffs::AbstractArray,
    type::Type{T} = SFFMProbability,
    v::Bool = false,
) where {T<:SFFMDistribution} 

    V = SFFM.vandermonde(NBases(mesh))
    N₋ = sum(model.C .<= 0)
    N₊ = sum(model.C .>= 0)

    if type == SFFMDensity
        xvals = SFFM.CellNodes(mesh)
        if Basis(mesh) == "legendre"
            yvals = reshape(Coeffs[N₋+1:end-N₊], NBases(mesh), NIntervals(mesh), NPhases(model))
            for i in 1:NPhases(model)
                yvals[:,:,i] = V.V * yvals[:,:,i]
            end
            pm = [Coeffs[1:N₋]; Coeffs[end-N₊+1:end]]
        elseif Basis(mesh) == "lagrange"
            yvals =
                Coeffs[N₋+1:end-N₊] .* repeat(1.0 ./ V.w, NIntervals(mesh) * NPhases(model)) .*
                (repeat(2.0 ./ Δ(mesh), 1, NBases(mesh) * NPhases(model))'[:])
            yvals = reshape(yvals, NBases(mesh), NIntervals(mesh), NPhases(model))
            pm = [Coeffs[1:N₋]; Coeffs[end-N₊+1:end]]
        end
        if NBases(mesh) == 1
            yvals = [1;1].*yvals
            xvals = [SFFM.CellNodes(mesh)-Δ(mesh)'/2;SFFM.CellNodes(mesh)+Δ(mesh)'/2]
        end
    elseif type == SFFMProbability
        if NBases(mesh) > 1 
            xvals = SFFM.CellNodes(mesh)[1, :] + (Δ(mesh) ./ 2)
        else
            xvals = SFFM.CellNodes(mesh)
        end
        if Basis(mesh) == "legendre"
            yvals = (reshape(Coeffs[N₋+1:NBases(mesh):end-N₊], 1, NIntervals(mesh), NPhases(model)).*Δ(mesh)')./sqrt(2)
            pm = [Coeffs[1:N₋]; Coeffs[end-N₊+1:end]]
        elseif Basis(mesh) == "lagrange"
            yvals = sum(
                reshape(Coeffs[N₋+1:end-N₊], NBases(mesh), NIntervals(mesh), NPhases(model)),
                dims = 1,
            )
            pm = [Coeffs[1:N₋]; Coeffs[end-N₊+1:end]]
        end
    elseif type == SFFMCDF
        if NBases(mesh) > 1 
            xvals = SFFM.CellNodes(mesh)[[1;end], :]
        else
            xvals = [SFFM.CellNodes(mesh)-Δ(mesh)'/2;SFFM.CellNodes(mesh)+Δ(mesh)'/2]
        end
        if Basis(mesh) == "legendre"
            tempDist = (reshape(Coeffs[N₋+1:NBases(mesh):end-N₊], 1, NIntervals(mesh), NPhases(model)).*Δ(mesh)')./sqrt(2)
            pm = [Coeffs[1:N₋]; Coeffs[end-N₊+1:end]]
        elseif Basis(mesh) == "lagrange"
            tempDist = sum(
                reshape(Coeffs[N₋+1:end-N₊], NBases(mesh), NIntervals(mesh), NPhases(model)),
                dims = 1,
            )
            pm = [Coeffs[1:N₋]; Coeffs[end-N₊+1:end]]
        end
        tempDist = cumsum(tempDist,dims=2)
        temppm = zeros(Float64,1,2,NPhases(model))
        temppm[:,1,model.C.<=0] = pm[1:N₋]
        temppm[:,2,model.C.>=0] = pm[N₊+1:end]
        yvals = zeros(Float64,2,NIntervals(mesh),NPhases(model))
        yvals[1,2:end,:] = tempDist[1,1:end-1,:]
        yvals[2,:,:] = tempDist
        yvals = yvals .+ reshape(temppm[1,1,:],1,1,NPhases(model))
        pm[N₋+1:end] = pm[N₋+1:end] + yvals[end,end,model.C.>=0]
    end
    
    out = type(pm, yvals, xvals)
    v && println("UPDATE: distribution object created with keys ", fieldnames(type))
    return out
end
function Coeffs2Dist(
    model::SFFM.Model,
    mesh::Union{FRAPMesh, FVMesh},
    Coeffs::AbstractArray,
    type::Type{T} = SFFMProbability,
    v::Bool = false,
) where {T<:SFFMDistribution}

    if type != SFFMProbability
        args = [
            model;
            mesh;
            Coeffs;
            type;
        ]
        error("Input Error: no functionality other than 'probability' implemented, yet...")
    end
    
    N₋ = sum(model.C .<= 0)
    N₊ = sum(model.C .>= 0)
    
    xvals = SFFM.CellNodes(mesh)
    
    yvals = sum(
        reshape(Coeffs[N₋+1:end-N₊], NBases(mesh), NIntervals(mesh), NPhases(model)),
        dims = 1,
    )
    pm = [Coeffs[1:N₋]; Coeffs[end-N₊+1:end]]

    out = type(pm, yvals, xvals)
    v && println("UPDATE: distribution object created with keys ", fieldnames(type))
    return out
end

"""
Converts a distribution as output from `Coeffs2Dist()` to a vector of DG
coefficients.

    Dist2Coeffs(
        model::SFFM.Model,
        mesh::SFFM.Mesh,
        Distn::SFFMDistribution,
    )

# Arguments
- `model`: a Model object
- `mesh`: a Mesh object as output from MakeMesh
- `Distn::SFFMDistribution
    - if `type="probability"` is a `1×NIntervals×NPhases` array containing
        the probabilities ``P(X(t)∈ D_k, φ(t) = i)`` where ``D_k``
        is the kth cell.
    - if `type="density"` is a `NBases×NIntervals×NPhases` array containing
        either the density function evaluated at the cell nodes which are in
        `x` below, or, the inner product of the density function against the
        lagrange polynomials.

# Output
- `coeffs` a row vector of coefficient values of length
    `TotalNBases*NPhases + N₋ + N₊` ordered according to LH point masses, RH
    point masses, interior basis functions according to basis function, cell,
    phase. Used to premultiply operators such as B from `MakeB()`
"""
function Dist2Coeffs(
    model::SFFM.Model,
    mesh::SFFM.DGMesh,
    Distn::SFFMDensity,
)
    V = SFFM.vandermonde(NBases(mesh))
    theDistribution =
        zeros(Float64, NBases(mesh), NIntervals(mesh), NPhases(model))
    if Basis(mesh) == "legendre"
        theDistribution = Distn.distribution
        for i = 1:NPhases(model)
            theDistribution[:, :, i] = V.inv * theDistribution[:, :, i]
        end
    elseif Basis(mesh) == "lagrange"
        theDistribution .= Distn.distribution
        # convert to probability coefficients by multiplying by the
        # weights in V.w/2 and cell widths Δ
        theDistribution = ((V.w .* theDistribution).*(Δ(mesh) / 2)')[:]
    end
    # also put the point masses on the ends
    coeffs = [
        Distn.pm[1:sum(model.C .<= 0)]
        theDistribution[:]
        Distn.pm[sum(model.C .<= 0)+1:end]
    ]
    coeffs = Matrix(coeffs[:]')
    return coeffs
end

function Dist2Coeffs(
    model::SFFM.Model,
    mesh::SFFM.DGMesh,
    Distn::SFFMProbability,
)
    V = SFFM.vandermonde(NBases(mesh))
    theDistribution =
        zeros(Float64, NBases(mesh), NIntervals(mesh), NPhases(model))
    if Basis(mesh) == "legendre"
        # for the legendre basis the first basis function is ϕ(x)=Δ√2 and
        # all other basis functions are orthogonal to this. Hence, we map
        # the cell probabilities to the first basis function only.
        theDistribution[1, :, :] = Distn.distribution./Δ(mesh)'.*sqrt(2)
    elseif Basis(mesh) == "lagrange"
        theDistribution .= Distn.distribution
        # convert to probability coefficients by multiplying by the
        # weights in V.w/2
        theDistribution = (V.w .* theDistribution / 2)[:]
    end
    # also put the point masses on the ends
    coeffs = [
        Distn.pm[1:sum(model.C .<= 0)]
        theDistribution[:]
        Distn.pm[sum(model.C .<= 0)+1:end]
    ]
    coeffs = Matrix(coeffs[:]')
    return coeffs
end
function Dist2Coeffs(
    model::SFFM.Model,
    mesh::Union{SFFM.FRAPMesh,SFFM.FVMesh},
    Distn::SFFMDistribution
)
    
    # also put the point masses on the ends
    coeffs = [
        Distn.pm[1:sum(model.C .<= 0)]
        Distn.distribution[:]
        Distn.pm[sum(model.C .<= 0)+1:end]
    ]
    
    coeffs = Matrix(coeffs[:]')
    return coeffs
end

"""
Computes the error between distributions.

    starSeminorm(d1::SFFMProbability, d2::SFFMProbability)

# Arguments
- `d1`: a distribution object as output from `Coeffs2Dist` 
- `d2`: a distribution object as output from `Coeffs2Dist` 
"""
function starSeminorm(d1::SFFMProbability, d2::SFFMProbability)
    return sum(abs.(d1.pm-d2.pm)) + sum(abs.(d1.distribution-d2.distribution))
end
