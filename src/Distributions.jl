struct SFFMDistribution
    pm::Array{<:Real}
    distribution::Array{<:Real,3}
    x::Array{<:Real}
    type::String
end

"""
Convert from a vector of coefficients for the DG system to a distribution.

    Coeffs2Dist(
        model::SFFM.Model,
        mesh::SFFM.Mesh,
        Coeffs;
        type::String = "probability",
        probTransform::Bool = true,
    )

# Arguments
- `model`: a Model object
- `mesh`: a Mesh object as output from MakeMesh
- `Coeffs::Array`: a vector of coefficients from the DG method
- `type::String`: an (optional) declaration of what type of distribution you
    want to convert to. Options are `"probability"` to return the probabilities
    ``P(X(t)∈ D_k, φ(t) = i)`` where ``D_k``is the kth cell, `"cumulative"` to
    return the CDF evaluated at cell edges, or `"density"` to return an
    approximation to the density ar at the mesh.CellNodes.
- `probTransform::Bool` a boolean value specifying whether to transform to a
    probabilistic interpretation or not. Valid only for lagrange basis.

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
    mesh::SFFM.Mesh,
    Coeffs::AbstractArray;
    type::String = "probability",
    probTransform::Bool = true,
)
    V = SFFM.vandermonde(mesh.NBases)
    N₋ = sum(model.C .<= 0)
    N₊ = sum(model.C .>= 0)
    if !probTransform
        temp = reshape(Coeffs[N₋+1:end-N₊], mesh.NBases, mesh.NIntervals, model.NPhases)
        temp = V.w.*temp.*(mesh.Δ./2.0)'
        Coeffs = [Coeffs[1:N₋]; temp[:]; Coeffs[end-N₊+1:end]]
    end
    if type == "density"
        xvals = mesh.CellNodes
        if mesh.Basis == "legendre"
            yvals = reshape(Coeffs[N₋+1:end-N₊], mesh.NBases, mesh.NIntervals, model.NPhases)
            for i in 1:model.NPhases
                yvals[:,:,i] = V.V * yvals[:,:,i]
            end
            pm = [Coeffs[1:N₋]; Coeffs[end-N₊+1:end]]
        elseif mesh.Basis == "lagrange"
            yvals =
                Coeffs[N₋+1:end-N₊] .* repeat(1.0 ./ V.w, mesh.NIntervals * model.NPhases) .*
                (repeat(2.0 ./ mesh.Δ, 1, mesh.NBases * model.NPhases)'[:])
            yvals = reshape(yvals, mesh.NBases, mesh.NIntervals, model.NPhases)
            pm = [Coeffs[1:N₋]; Coeffs[end-N₊+1:end]]
        end
        if mesh.NBases == 1
            yvals = [1;1].*yvals
            xvals = [mesh.CellNodes-mesh.Δ'/2;mesh.CellNodes+mesh.Δ'/2]
        end
    elseif type == "probability"
        if mesh.NBases > 1 && typeof(mesh)==SFFM.DGMesh
            xvals = mesh.CellNodes[1, :] + (mesh.Δ ./ 2)
        else
            xvals = mesh.CellNodes
        end
        if mesh.Basis == "legendre"
            yvals = (reshape(Coeffs[N₋+1:mesh.NBases:end-N₊], 1, mesh.NIntervals, model.NPhases).*mesh.Δ')./sqrt(2)
            pm = [Coeffs[1:N₋]; Coeffs[end-N₊+1:end]]
        elseif mesh.Basis == "lagrange"
            yvals = sum(
                reshape(Coeffs[N₋+1:end-N₊], mesh.NBases, mesh.NIntervals, model.NPhases),
                dims = 1,
            )
            pm = [Coeffs[1:N₋]; Coeffs[end-N₊+1:end]]
        end
    elseif type == "cumulative"
        if mesh.NBases > 1 
            xvals = mesh.CellNodes[[1;end], :]
        else
            xvals = [mesh.CellNodes-mesh.Δ'/2;mesh.CellNodes+mesh.Δ'/2]
        end
        if mesh.Basis == "legendre"
            tempDist = (reshape(Coeffs[N₋+1:mesh.NBases:end-N₊], 1, mesh.NIntervals, model.NPhases).*mesh.Δ')./sqrt(2)
            pm = [Coeffs[1:N₋]; Coeffs[end-N₊+1:end]]
        elseif mesh.Basis == "lagrange"
            tempDist = sum(
                reshape(Coeffs[N₋+1:end-N₊], mesh.NBases, mesh.NIntervals, model.NPhases),
                dims = 1,
            )
            pm = [Coeffs[1:N₋]; Coeffs[end-N₊+1:end]]
        end
        tempDist = cumsum(tempDist,dims=2)
        temppm = zeros(Float64,1,2,model.NPhases)
        temppm[:,1,model.C.<=0] = pm[1:N₋]
        temppm[:,2,model.C.>=0] = pm[N₊+1:end]
        yvals = zeros(Float64,2,mesh.NIntervals,model.NPhases)
        yvals[1,2:end,:] = tempDist[1,1:end-1,:]
        yvals[2,:,:] = tempDist
        yvals = yvals .+ reshape(temppm[1,1,:],1,1,model.NPhases)
        pm[N₋+1:end] = pm[N₋+1:end] + yvals[end,end,model.C.>=0]
    end

    out = (pm=pm, distribution=yvals, x=xvals, type=type)
    println("UPDATE: distribution object created with keys ", keys(out))
    return out
end

"""
Converts a distribution as output from `Coeffs2Dist()` to a vector of DG
coefficients.

    Dist2Coeffs(
        model::SFFM.Model,
        mesh::SFFM.Mesh,
        Distn::NamedTuple{(:pm, :distribution, :x, :type)};
        probTransform::Bool = true,
    )

# Arguments
- `model`: a Model object
- `mesh`: a Mesh object as output from MakeMesh
- `Distn::NamedTuple{(:pm, :distribution, :x, :type)}`: a distribution object
    i.e. a `NamedTuple` with fields
    - `pm::Array{Float64}`: a vector containing the point masses, the first
        `sum(model.C.<=0)` entries are the left hand point masses and the last
        `sum(model.C.>=0)` are the right-hand point masses.
    - `distribution::Array{Float64,3}`:
        - if `type="probability"` is a `1×NIntervals×NPhases` array containing
            the probabilities ``P(X(t)∈ D_k, φ(t) = i)`` where ``D_k``
            is the kth cell.
        - if `type="density"` is a `NBases×NIntervals×NPhases` array containing
            either the density function evaluated at the cell nodes which are in
            `x` below, or, the inner product of the density function against the
            lagrange polynomials.
    - `x::Array{Float64,2}`:
        - if `type="probability"` is a `1×NIntervals×NPhases` array
            containing the cell centers.
        - if `type="density"` is a `NBases×NIntervals×NPhases` array
            containing the cell nodes at which the denisty is evaluated.
    - `type::String`: either `"probability"` or `"density"`. `:cumulative` is
        not possible.
- `probTransform::Bool` a boolean value specifying whether to transform to a
    probabilistic interpretation or not. Valid only for lagrange basis.

# Output
- `coeffs` a row vector of coefficient values of length
    `TotalNBases*NPhases + N₋ + N₊` ordered according to LH point masses, RH
    point masses, interior basis functions according to basis function, cell,
    phase. Used to premultiply operators such as B from `MakeB()`
"""
function Dist2Coeffs(
    model::SFFM.Model,
    mesh::SFFM.Mesh,
    Distn::NamedTuple{(:pm, :distribution, :x, :type)};
    probTransform::Bool = true,
)
    V = SFFM.vandermonde(mesh.NBases)
    theDistribution =
        zeros(Float64, mesh.NBases, mesh.NIntervals, model.NPhases)
    if mesh.Basis == "legendre"
        if Distn.type == "probability"
            # for the legendre basis the first basis function is ϕ(x)=Δ√2 and
            # all other basis functions are orthogonal to this. Hence, we map
            # the cell probabilities to the first basis function only.
            theDistribution[1, :, :] = Distn.distribution./mesh.Δ'.*sqrt(2)
        elseif Distn.type == "density"
            # if given density coefficients in lagrange form
            theDistribution = Distn.distribution
            for i = 1:model.NPhases
                theDistribution[:, :, i] = V.inv * theDistribution[:, :, i]
            end
        end
        # also put the point masses on the ends
        coeffs = [
            Distn.pm[1:sum(model.C .<= 0)]
            theDistribution[:]
            Distn.pm[sum(model.C .<= 0)+1:end]
        ]
    elseif mesh.Basis == "lagrange"
        theDistribution .= Distn.distribution
        if !probTransform
            theDistribution = (1.0./V.w) .* theDistribution .* (2.0./mesh.Δ')
        end
        if Distn.type == "probability"
            # convert to probability coefficients by multiplying by the
            # weights in V.w/2
            theDistribution = (V.w .* theDistribution / 2)[:]
        elseif Distn.type == "density"
            # convert to probability coefficients by multiplying by the
            # weights in V.w/2 and cell widths Δ
            theDistribution = ((V.w .* theDistribution).*(mesh.Δ / 2)')[:]
        end
        # also put the point masses on the ends
        coeffs = [
            Distn.pm[1:sum(model.C .<= 0)]
            theDistribution
            Distn.pm[sum(model.C .<= 0)+1:end]
        ]
    end
    coeffs = Matrix(coeffs[:]')
    return coeffs
end

"""
Computes the error between distributions.

    starSeminorm(
        d1::NamedTuple{(:pm, :distribution, :x, :type)},
        d2::NamedTuple{(:pm, :distribution, :x, :type)},
        )

# Arguments
- `d1`: a distribution object as output from `Coeffs2Dist` with
    `type="probability"``
- `d2`: a distribution object as output from `Coeffs2Dist` with
    `type="probability"``
"""
function starSeminorm(
    d1::NamedTuple{(:pm, :distribution, :x, :type)},
    d2::NamedTuple{(:pm, :distribution, :x, :type)},
    )
    if ((d1.type!="probability") || (d2.type!="probability"))
        throw(ArgumentError("distributions need to be of type probability"))
    end
    return sum(abs.(d1.pm-d2.pm)) + sum(abs.(d1.distribution-d2.distribution))
end