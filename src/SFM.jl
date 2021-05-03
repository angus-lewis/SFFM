"""
Construct and evaluate ``Ψ(s)`` for a triditional SFM.

Uses newtons method to solve the Ricatti equation
``D⁺⁻(s) + Ψ(s)D⁻⁺(s)Ψ(s) + Ψ(s)D⁻⁻(s) + D⁺⁺(s)Ψ(s) = 0.``

    PsiFun( model::SFFM.Model; s = 0, MaxIters = 1000, err = 1e-8)

# Arguments
- `model`: a Model object
- `s::Real`: a value to evaluate the LST at
- `MaxIters::Int`: the maximum number of iterations of newtons method
- `err::Float64`: an error tolerance for terminating newtons method. Terminates
    when `max(Ψ_{n} - Ψ{n-1}) .< eps`.

# Output
- `Ψ(s)::Array{Float64,2}` the matrix ``Ψ``
"""
function PsiFunX( model::SFFM.Model; s = 0, MaxIters = 1000, err = 1e-8)
    SDict, TDict = modelDicts(model)

    T00inv = inv(TDict["00"] - s * LinearAlgebra.I)
    # construct the generator Q(s)
    Q =
        (1 ./ abs.(model.C[SDict["bullet"]])) .* (
            TDict["bulletbullet"] - s * LinearAlgebra.I -
            TDict["bullet0"] * T00inv * TDict["0bullet"]
        )

    QDict = Dict{String,Array}("Q" => Q)
    for ℓ in ["+" "-"], m in ["+" "-"]
        QDict[ℓ*m] = Q[SDict[ℓ], SDict[m]]
    end

    Ψ = zeros(Float64, length(SDict["+"]), length(SDict["-"]))
    A = QDict["++"]
    B = QDict["--"]
    D = QDict["+-"]
    # use netwons method to solve the Ricatti equation
    for n in 1:MaxIters
        Ψ = LinearAlgebra.sylvester(A,B,D)
        if maximum(abs.(sum(Ψ,dims=2).-1)) < err
            break
        end
        A = QDict["++"] + Ψ * QDict["-+"]
        B = QDict["--"] + QDict["-+"] * Ψ
        D = QDict["+-"] - Ψ * QDict["-+"] * Ψ
    end

    return Ψ
end

"""
Construct the vector ``ξ`` containing the distribution of the phase at the time
when ``X(t)`` first hits `0`.

    MakeXiX( model::SFFM.Model, Ψ::Array)

# Arguments
- `model`: a Model object
- `Ψ`: an array as output from `PsiFunX`

# Output
- the vector `ξ`
"""
function MakeXiX( model::SFFM.Model, Ψ::Array)
    # the system to solve is [ξ 0](-[B₋₋ B₋₀; B₀₋ B₀₀])⁻¹[B₋₊; B₀₊]Ψ = ξ
    # writing this out and using block inversion (as described on wikipedia)
    # we can solve this in the following way
    SDict, TDict = modelDicts(model)

    T00inv = inv(TDict["00"])
    invT₋₋ =
        inv(TDict["--"] - TDict["-0"] * T00inv * TDict["0-"])
    invT₋₀ = -invT₋₋ * TDict["-0"] * T00inv

    A =
        -(
            invT₋₋ * TDict["-+"] * Ψ + invT₋₀ * TDict["0+"] * Ψ +
            LinearAlgebra.I
        )
    b = zeros(1, size(TDict["--"], 1))
    A[:, 1] .= 1.0 # normalisation conditions
    b[1] = 1.0 # normalisation conditions

    ξ = b / A

    return ξ
end

"""
Construct the stationary distribution of the SFM

    StationaryDistributionX( model::SFFM.Model, Ψ::Array, ξ::Array)

# Arguments
- `model`: a Model object
- `Ψ`: an array as output from `PsiFunX`
- `ξ`: an array as returned from `MakeXiX`

# Output
- `pₓ::Array{Float64,2}`: the point masses of the SFM
- `πₓ(x)` a function with two methods
    - `πₓ(x::Real)`: for scalar inputs, returns the stationary density evaluated
        at `x` in all phases.
    - `πₓ(x::Array)`: for array inputs, returns an array with the same shape
        as is output by Coeff2Dist.
- `K::Array{Float64,2}`: the matrix in the exponential of the density.
"""
function StationaryDistributionX( model::SFFM.Model, Ψ::Array, ξ::Array)
    # using the same block inversion trick as in MakeXiX
    SDict, TDict = modelDicts(model)
    
    T00inv = inv(TDict["00"])
    invT₋₋ =
        inv(TDict["--"] - TDict["-0"] * T00inv * TDict["0-"])
    invT₋₀ = -invT₋₋ * TDict["-0"] * T00inv

    Q =
        (1 ./ abs.(model.C[SDict["bullet"]])) .* (
            TDict["bulletbullet"] -
            TDict["bullet0"] * T00inv * TDict["0bullet"]
        )

    QDict = Dict{String,Array}("Q" => Q)
    for ℓ in ["+" "-"], m in ["+" "-"]
        QDict[ℓ*m] = Q[SDict[ℓ], SDict[m]]
    end

    K = QDict["++"] + Ψ * QDict["-+"]

    A = -[invT₋₋ invT₋₀]

    # unnormalised values
    αpₓ = ξ * A

    απₓ = αpₓ *
        [TDict["-+"]; TDict["0+"]] *
        -inv(K) *
        [LinearAlgebra.I(length(SDict["+"])) Ψ] *
        LinearAlgebra.diagm(1 ./ abs.(model.C[SDict["bullet"]]))

    απₓ0 = -απₓ * [TDict["+0"];TDict["-0"]] * T00inv

    # normalising constant
    α = sum(αpₓ) + sum(απₓ) + sum(απₓ0)

    # normalised values
    # point masses
    pₓ = αpₓ/α
    # density method for scalar x-values
    function πₓ(x::Real)
        pₓ *
        [TDict["-+"]; TDict["0+"]] *
        exp(K*x) *
        [LinearAlgebra.I(length(SDict["+"])) Ψ] *
        LinearAlgebra.diagm(1 ./ abs.(model.C[SDict["bullet"]])) *
        [LinearAlgebra.I(sum(model.C .!= 0)) [TDict["+0"];TDict["-0"]] * T00inv]
    end
    # density method for arrays so that πₓ returns an array with the same shape
    # as is output by Coeff2Dist
    function πₓ(x::Array)
        temp = πₓ.(x)
        Evalπₓ = zeros(Float64, size(x,1), size(x,2), NPhases(model))
        for cell in 1:size(x,2)
            for basis in 1:size(x,1)
                Evalπₓ[basis,cell,:] = temp[basis,cell]
            end
        end
        return Evalπₓ
    end

    # CDF method for scalar x-values
    function Πₓ(x::Real)
        [pₓ zeros(1,sum(model.C.>0))] .+
        pₓ *
        [TDict["-+"]; TDict["0+"]] *
        (exp(K*x) - LinearAlgebra.I) / K *
        [LinearAlgebra.I(length(SDict["+"])) Ψ] *
        LinearAlgebra.diagm(1 ./ abs.(model.C[SDict["bullet"]])) *
        [LinearAlgebra.I(sum(model.C .!= 0)) [TDict["+0"];TDict["-0"]] * T00inv]
    end
    # CDF method for arrays so that Πₓ returns an array with the same shape
    # as is output by Coeff2Dist
    function Πₓ(x::Array)
        temp = Πₓ.(x)
        Evalπₓ = zeros(Float64, size(x,1), size(x,2), NPhases(model))
        for cell in 1:size(x,2)
            for basis in 1:size(x,1)
                Evalπₓ[basis,cell,:] = temp[basis,cell]
            end
        end
        return Evalπₓ
    end

    return pₓ, πₓ, Πₓ, K
end
