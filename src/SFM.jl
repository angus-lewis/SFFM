function PsiFunX(; Model::NamedTuple, s = 0, MaxIters = 1000, err = 1e-8)
    T00inv = inv(Model.TDict["00"] - s * LinearAlgebra.I)
    Q =
        (1 ./ abs.(Model.C[Model.SDict["bullet"]])) .* (
            Model.TDict["bulletbullet"] - s * LinearAlgebra.I -
            Model.TDict["bullet0"] * T00inv * Model.TDict["0bullet"]
        )

    QDict = Dict{String,Array}("Q" => Q)
    for ℓ in ["+" "-"], m in ["+" "-"]
        QDict[ℓ*m] = Q[Model.SDict[ℓ], Model.SDict[m]]
    end

    Ψ = zeros(Float64, length(Model.SDict["+"]), length(Model.SDict["-"]))
    A = QDict["++"]
    B = QDict["--"]
    D = QDict["+-"]

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

function MakeXiX(; Model::NamedTuple, Ψ::Array)
    T00inv = inv(Model.TDict["00"])
    invT₋₋ =
        inv(Model.TDict["--"] - Model.TDict["-0"] * T00inv * Model.TDict["0-"])
    invT₋₀ = -invT₋₋ * Model.TDict["-0"] * T00inv

    A =
        -(
            invT₋₋ * Model.TDict["-+"] * Ψ + invT₋₀ * Model.TDict["0+"] * Ψ +
            LinearAlgebra.I
        )
    b = zeros(1, size(Model.TDict["--"], 1))
    A[:, 1] .= 1.0 # normalisation conditions
    b[1] = 1.0 # normalisation conditions

    ξ = b / A

    return ξ
end

function StationaryDistributionX(; Model::NamedTuple, Ψ::Array, ξ::Array)
    T00inv = inv(Model.TDict["00"])
    invT₋₋ =
        inv(Model.TDict["--"] - Model.TDict["-0"] * T00inv * Model.TDict["0-"])
    invT₋₀ = -invT₋₋ * Model.TDict["-0"] * T00inv

    Q =
        (1 ./ abs.(Model.C[Model.SDict["bullet"]])) .* (
            Model.TDict["bulletbullet"] -
            Model.TDict["bullet0"] * T00inv * Model.TDict["0bullet"]
        )

    QDict = Dict{String,Array}("Q" => Q)
    for ℓ in ["+" "-"], m in ["+" "-"]
        QDict[ℓ*m] = Q[Model.SDict[ℓ], Model.SDict[m]]
    end

    K = QDict["++"] + Ψ * QDict["-+"]

    A = -[invT₋₋ invT₋₀]
    αpₓ = ξ * A

    απₓ = αpₓ *
        [Model.TDict["-+"]; Model.TDict["0+"]] *
        -inv(K) *
        [LinearAlgebra.I(length(Model.SDict["+"])) Ψ] *
        LinearAlgebra.diagm(1 ./ abs.(Model.C[Model.SDict["bullet"]]))

    απₓ0 = -απₓ * [Model.TDict["+0"];Model.TDict["-0"]] * T00inv

    α = sum(αpₓ) + sum(απₓ) + sum(απₓ0)

    pₓ = αpₓ/α
    function πₓ(x::Real)
        pₓ *
        [Model.TDict["-+"]; Model.TDict["0+"]] *
        exp(K*x) *
        [LinearAlgebra.I(length(Model.SDict["+"])) Ψ] *
        LinearAlgebra.diagm(1 ./ abs.(Model.C[Model.SDict["bullet"]])) *
        [LinearAlgebra.I(sum(Model.C .!= 0)) [Model.TDict["+0"];Model.TDict["-0"]] * T00inv]
    end

    function πₓ(x::Array)
        temp = πₓ.(x)
        Evalπₓ = zeros(Float64, size(x,1), size(x,2), Model.NPhases)
        for cell in 1:size(x,2)
            for basis in 1:size(x,1)
                Evalπₓ[basis,cell,:] = temp[basis,cell]
            end
        end
        return Evalπₓ
    end

    return pₓ, πₓ, K
end
