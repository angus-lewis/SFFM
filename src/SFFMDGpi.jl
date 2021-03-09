"""
Returns the DG approximation to the return probabilities ``ξ`` for the process
``Y(t)``.
NOTE: IMPLEMENTED FOR LAGRANGE BASIS ONLY

    MakeXi(;
        B::Dict{String,SparseArrays.SparseMatrixCSC{Float64,Int64}},
        Ψ::Array,
    )

# Arguments
- `B`: an object as returned by `MakeB`
- `Ψ::Array{Float64,2}`: an array as returned by `PsiFun`

# Output
- `ξ::Array{Float64,2}`: a row-vector of first return probabilities
"""
function MakeXi(;
    B::Dict{String,SparseArrays.SparseMatrixCSC{Float64,Int64}},
    Ψ::Array{Float64,2},
    probTransform::Bool = true,
    mesh=1,
    model=1,
)
    # BBullet = [B["--"] B["-0"]; B["0-"] B["00"]]
    # invB = inv(Matrix(Bbullet))

    # BBulletPlus = [B["-+"]; B["0+"]]

    # solve the linear system -[ξ 0] [Bmm Bm0; B₀₋ B00]^-1 [B₋₊; B₀₊]Ψ = ξ
    # writing out the system it turns out we only need the index -- and -0
    # blocks of the inverse. Wikipedia tells us that these are

    # the next two lines are a fancy way to do an inverse of a block matrix
    tempMat = inv(Matrix(B["00"]))
    invBmm = inv(B["--"] - B["-0"]*tempMat*B["0-"])
    invBm0 = -invBmm*B["-0"]*tempMat

    A = -(invBmm*B["-+"]*Ψ + invBm0*B["0+"]*Ψ + LinearAlgebra.I)
    b = zeros(1,size(B["--"],1))

    if probTransform
        A[:,1] .= 1.0 # normalisation conditions
        b[1] = 1.0 # normalisation conditions
    elseif !probTransform
        idx₋ = [mesh.Fil["p-"]; mesh.Fil["-"]; mesh.Fil["q-"]]
        if mesh.NBases>1
            w =
                2.0 ./ (
                    mesh.NBases *
                    (mesh.NBases - 1) *
                    Jacobi.legendre.(Jacobi.zglj(mesh.NBases, 0, 0), mesh.NBases - 1) .^ 2
                )
        else
            w = [2]
        end
        A[:,1] = [ones(sum(mesh.Fil["p-"]));
        repeat(w, sum(mesh.Fil["-"])).*repeat(repeat(mesh.Δ./2,model.NPhases,1)[mesh.Fil["-"]]', mesh.NBases, 1)[:];
        ones(sum(mesh.Fil["q-"]))]# normalisation conditions
        b[1] = 1.0 # normalisation conditions
    end

    ξ = b/A

    return ξ
end

"""
Returns the DG approximation to some quantities regarding the limiting
distribution of a SFFM. See Ouput below
NOTE: IMPLEMENTED FOR LAGRANGE BASIS ONLY

    MakeLimitDistMatrices(;
        B::Dict{String,SparseArrays.SparseMatrixCSC{Float64,Int64}},
        D::Dict{String,Any},
        R::Dict{String,SparseArrays.SparseMatrixCSC{Float64,Int64}},
        Ψ::Array{<:Real},
        ξ::Array{<:Real},
        mesh::Mesh,
    )

# Arguments
- `B`: an object as returned by `MakeB`
- `D`: an object as returned by `MakeD`
- `R`: an object as returned by `MakeR`
- `Ψ::Array{Float64,2}`: an array as returned by `PsiFun`
- `ξ::Array{Float64,2}`: an row-vector as returned by `XiFun`

# Output
marginalX, p, K
- `marginalX::Array{Float64,2}`: a row-vector of the marginal limiting
    distribution of the first buffer ``X(t)``.
- `p::Array{Float64,2}`: a row-vector of the distribution of ``X(t)`` at the
    times of first return of ``Y(t)`` to 0.
- `K::Array{Float64,2}`: the array in the operator exponential of the stationary
    distribution

"""
function MakeLimitDistMatrices(;
    B::Dict{String,SparseArrays.SparseMatrixCSC{Float64,Int64}},
    D::Dict{String,Any},
    R::Dict{String,SparseArrays.SparseMatrixCSC{Float64,Int64}},
    Ψ::Array{<:Real},
    ξ::Array{<:Real},
    mesh::Mesh,
    probTransform::Bool = true,
    model=1,
)
    B00inv = inv(Matrix(B["00"]))
    invBmm = inv(B["--"] - B["-0"]*B00inv*B["0-"])
    invBm0 = -invBmm*B["-0"]*B00inv

    αp = ξ * -[invBmm invBm0]

    K = D["++"]() + Ψ * D["-+"]()

    n₊, n₀ = size(B["+0"])
    n₋ = size(B["-+"],1)

    BBulletPlus = [B["-+"]; B["0+"]]

    αintegralPibullet = ((αp * BBulletPlus) / -K) * [R["+"] Ψ*R["-"]]
    αintegralPi0 = -αintegralPibullet * [B["+0"]; B["-0"]] * B00inv

    if probTransform
        α = sum(αintegralPibullet) + sum(αintegralPi0) + sum(αp)
    elseif !probTransform
        if mesh.NBases>1
            w = 2.0 ./ (
                    mesh.NBases *
                    (mesh.NBases - 1) *
                    Jacobi.legendre.(Jacobi.zglj(mesh.NBases, 0, 0), mesh.NBases - 1) .^ 2
                )
        else
            w = [2]
        end
        idx₊ = mesh.Fil["+"]
        idx₋ = mesh.Fil["-"]
        idx₀ = mesh.Fil["0"]
        a₋ = [ones(sum(mesh.Fil["p-"])); repeat(w, sum(mesh.Fil["-"])).*(repeat(repeat(mesh.Δ./2,model.NPhases,1)[idx₋]', mesh.NBases, 1)[:]); ones(sum(mesh.Fil["q-"]))]
        a₊ = [ones(sum(mesh.Fil["p+"])); repeat(w, sum(mesh.Fil["+"])).*(repeat(repeat(mesh.Δ./2,model.NPhases,1)[idx₊]', mesh.NBases, 1)[:]); ones(sum(mesh.Fil["q+"]))]
        a₀ = [ones(sum(mesh.Fil["p0"])); repeat(w, sum(mesh.Fil["0"])).*(repeat(repeat(mesh.Δ./2,model.NPhases,1)[idx₀]', mesh.NBases, 1)[:]); ones(sum(mesh.Fil["q0"]))]
        # display(a₋)
        # display(a₊)
        # display(a₀)
        # display(αintegralPibullet)
        # display(αintegralPi0)
        # display(αp)
        α = (αintegralPibullet*[a₊; a₋]) + (αintegralPi0*a₀) + (αp*[a₋;a₀])
    end

    p = αp ./ α
    integralPibullet = αintegralPibullet ./ α
    integralPi0 = αintegralPi0 ./ α

    marginalX = zeros(Float64, n₊ + n₋ + n₀)
    idx₊ = [mesh.Fil["p+"]; repeat(mesh.Fil["+"]', mesh.NBases, 1)[:]; mesh.Fil["q+"]]
    marginalX[idx₊] = integralPibullet[1:n₊]
    idx₋ = [mesh.Fil["p-"]; repeat(mesh.Fil["-"]', mesh.NBases, 1)[:]; mesh.Fil["q-"]]
    marginalX[idx₋] = integralPibullet[(n₊+1):end] + p[1:n₋]
    idx₀ = [mesh.Fil["p0"]; repeat(mesh.Fil["0"]', mesh.NBases, 1)[:]; mesh.Fil["q0"]]
    marginalX[idx₀] = integralPi0[:] + p[(n₋+1):end]

    return marginalX, p, K
end
