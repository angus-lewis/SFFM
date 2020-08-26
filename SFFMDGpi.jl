function MakeXi(;
    B::Dict{String,SparseArrays.SparseMatrixCSC{Float64,Int64}},
    Ψ::Array,
)
    # BBullet = [B["--"] B["-0"]; B["0-"] B["00"]]
    # invB = inv(Matrix(Bbullet))

    # BBulletPlus = [B["-+"]; B["0+"]]

    # solve the linear system -[ξ 0] [Bmm Bm0; B₀₋ B00]^-1 [B₋₊; B₀₊]Ψ = ξ
    # writing out the system it turns out we only need the index -- and -0
    # blocks of the inverse. Wikipedia tells us that these are
    tempMat = inv(Matrix(B["00"]))
    invBmm = inv(B["--"] - B["-0"]*tempMat*B["0-"])
    invBm0 = -invBmm*B["-0"]*tempMat

    A = -(invBmm*B["-+"]*Ψ + invBm0*B["0+"]*Ψ + LinearAlgebra.I)
    b = zeros(1,size(B["--"],1))

    A[:,1] .= 1.0 # normalisation conditions
    b[1] = 1.0 # normalisation conditions

    ξ = b/A

    return ξ
end


function MakeLimitDistMatrices(;
    B::Dict{String,SparseArrays.SparseMatrixCSC{Float64,Int64}},
    D::Dict{String,Any},
    R::Dict{String,SparseArrays.SparseMatrixCSC{Float64,Int64}},
    Ψ::Array{<:Real},
    ξ::Array{<:Real},
    Mesh,
)
    B00inv = inv(Matrix(B["00"]))
    invBmm = inv(B["--"] - B["-0"]*B00inv*B["0-"])
    invBm0 = -invBmm*B["-0"]*B00inv

    αp = ξ * -[invBmm invBm0]

    K = D["++"]() + Ψ * D["-+"]()

    n₊, n₀ = size(B["+0"])
    n₋ = size(B["-+"],1)

    BBulletPlus = [B["-+"]; B["0+"]]
    z₊₋ = SparseArrays.spzeros(Float64, n₊, n₋)
    RBullet = [R["+"] z₊₋; z₊₋' R["-"]]

    αintegralPibullet = ((αp * BBulletPlus) / -K) * [LinearAlgebra.I(n₊) Ψ] * RBullet
    αintegralPi0 = -αintegralPibullet * [B["+0"]; B["-0"]] * B00inv

    α = sum(αintegralPibullet) + sum(αintegralPi0) + sum(αp)

    p = αp ./ α
    integralPibullet = αintegralPibullet ./ α
    integralPi0 = αintegralPi0 ./ α

    marginalX = zeros(Float64, n₊ + n₋ + n₀)
    idx₊ = [Mesh.Fil["p+"]; repeat(Mesh.Fil["+"]', Mesh.NBases, 1)[:]; Mesh.Fil["q+"]]
    marginalX[idx₊] = integralPibullet[1:n₊]
    idx₋ = [Mesh.Fil["p-"]; repeat(Mesh.Fil["-"]', Mesh.NBases, 1)[:]; Mesh.Fil["q-"]]
    marginalX[idx₋] = integralPibullet[(n₊+1):end]
    idx₀ = [Mesh.Fil["p0"]; repeat(Mesh.Fil["0"]', Mesh.NBases, 1)[:]; Mesh.Fil["q0"]]
    marginalX[idx₀] = integralPi0

    return marginalX, p, integralPibullet, integralPi0, K
end
