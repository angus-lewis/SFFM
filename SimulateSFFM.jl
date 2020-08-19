function SimSFM(;
    Model::NamedTuple{(:T, :C, :r, :Bounds, :NPhases)},
    StoppingTime::Function,
    InitCondition::NamedTuple{(:φ, :X)},
)
    # Simulates a SFM defined by Model until the StoppingTime has occured,
    # given the InitialConditions on (φ(0),X(0))
    # Model - A Model object as output from MakeModel
    # StoppingTime - A function which takes four arguments e.g.
    #   StoppingTime(;t,φ,X,n) where
    #   t is the current time, φ the current phase, X, the current level, n
    #   the number of transition to time t, and returns a tuple (Ind,SFM),
    #   where Ind is a boolen specifying whether the stopping time, τ, has
    #   occured or not, and SFM is a tuple (τ,φ(τ),X(τ),n).
    # InitCondition - M×2 Array with rows [φ(0) X(0)], M is the number of sims
    #   to be performed, e.g.~to simulate ten SFMs starting from
    #   [φ(0)=1 X(0)=0] then we set InitCondition = repeat([1 0], 10,1).

    # the transition matrix of the jump chain
    d = LinearAlgebra.diag(Model.T)
    P = (Model.T - LinearAlgebra.diagm(0 => d)) ./ -d
    CumP = cumsum(P, dims = 2)
    Λ = LinearAlgebra.diag(Model.T)

    M = length(InitCondition.φ)
    tSims = zeros(Float64, M)
    φSims = zeros(Float64, M)
    XSims = zeros(Float64, M)
    nSims = zeros(Float64, M)

    for m = 1:M
        SFM0 = (t = 0.0, φ = InitCondition.φ[m], X = InitCondition.X[m], n = 0)
        while 1 == 1
            S = log(rand()) / Λ[SFM0.φ] # generate exp random variable
            t = SFM0.t + S
            X = UpdateXt(Model = Model, SFM0 = SFM0, S = S)
            φ = findfirst(rand() .< CumP[SFM0.φ, :])
            n = SFM0.n + 1
            SFM = (t = t, φ = φ, X = X, n = n)
            τ = StoppingTime(Model, SFM, SFM0)
            if τ.Ind
                (tSims[m], φSims[m], XSims[m], nSims[m]) = τ.SFM
                break
            end
            SFM0 = SFM
        end
    end
    return (t = tSims, φ = φSims, X = XSims, n = nSims)
end

function SimSFFM(;
    Model::NamedTuple{(:T, :C, :r, :Bounds, :NPhases)},
    StoppingTime::Function,
    InitCondition::NamedTuple{(:φ, :X, :Y)},
)
    d = LinearAlgebra.diag(Model.T)
    P = (Model.T - LinearAlgebra.diagm(0 => d)) ./ -d
    CumP = cumsum(P, dims = 2)
    Λ = LinearAlgebra.diag(Model.T)

    M = length(InitCondition.φ)
    tSims = Array{Float64,1}(undef, M)
    φSims = Array{Float64,1}(undef, M)
    XSims = Array{Float64,1}(undef, M)
    YSims = Array{Float64,1}(undef, M)
    nSims = Array{Float64,1}(undef, M)

    for m = 1:M
        SFFM0 = (
            t = 0.0,
            φ = InitCondition.φ[m],
            X = InitCondition.X[m],
            Y = InitCondition.Y[m],
            n = 0.0,
        )
        if !(Model.Bounds[1, 1] <= SFFM0.X <= Model.Bounds[1, 2]) ||
           !(Model.Bounds[2, 1] <= SFFM0.Y <= Model.Bounds[2, 2]) ||
           !in(SFFM0.φ, 1:Model.NPhases)
            (tSims[m], φSims[m], XSims[m], YSims[m], nSims[m]) =
                (t = NaN, φ = NaN, X = NaN, Y = NaN, n = NaN)
        else
            while 1 == 1
                S = log(rand()) / Λ[SFFM0.φ]
                t = SFFM0.t + S
                X = UpdateXt(Model = Model, SFM0 = SFFM0, S = S)
                Y = UpdateYt(Model = Model, SFFM0 = SFFM0, S = S)
                φ = findfirst(rand() .< CumP[SFFM0.φ, :])
                n = SFFM0.n + 1.0
                SFFM = (t = t, φ = φ, X = X, Y = Y, n = n)
                τ = StoppingTime(Model, SFFM, SFFM0)
                if τ.Ind
                    (tSims[m], φSims[m], XSims[m], YSims[m], nSims[m]) = τ.SFFM
                    break
                end
                if t > 300.0
                    (tSims[m], φSims[m], XSims[m], YSims[m], nSims[m]) = SFFM
                    break
                end
                SFFM0 = SFFM
            end
        end
    end
    return (t = tSims, φ = φSims, X = XSims, Y = YSims, n = nSims)
end

function UpdateXt(;
    Model::NamedTuple{(:T, :C, :r, :Bounds, :NPhases)},
    SFM0::NamedTuple{(:t, :φ, :X, :Y, :n)},
    S::Real,
)
    # given the last position of a SFM, SFM0, a time step of size s, find the
    # position of X at time t
    X = min(
        max(SFM0.X + Model.C[SFM0.φ] * S, Model.Bounds[1, 1]),
        Model.Bounds[1, 2],
    )
    return X
end

function UpdateYt(;
    Model::NamedTuple{(:T, :C, :r, :Bounds, :NPhases)},
    SFFM0::NamedTuple{(:t, :φ, :X, :Y, :n)},
    S::Real,
)
    # given the last position of a SFFM, SFFM0, a time step of size s, find the
    # position of Y at time t
    if Model.C[SFFM0.φ] == 0
        Y = SFFM0.Y + S * Model.r.r(SFFM0.X)[SFFM0.φ]
    else
        X = UpdateXt(Model = Model, SFM0 = SFFM0, S = S)
        ind = (X.==Model.Bounds[1, :])[:]
        if any(ind)
            # time at which Xt hits u or v
            t0 = (Model.Bounds[1, ind][1] - SFFM0.X) / Model.C[SFFM0.φ]
            Y =
                SFFM0.Y +
                (Model.r.R(X)[SFFM0.φ] - Model.r.R(SFFM0.X)[SFFM0.φ]) /
                Model.C[SFFM0.φ] +
                Model.r.r(Model.Bounds[1, ind][1])[SFFM0.φ] * (S - t0)
        else
            Y =
                SFFM0.Y +
                (Model.r.R(X)[SFFM0.φ] - Model.r.R(SFFM0.X)[SFFM0.φ]) /
                Model.C[SFFM0.φ]
        end
    end
    return Y
end

function FixedTime(; T::Real)
    # Defines a simple stopping time, 1(t>T).
    # SFM method
    function FixedTimeFun(
        Model::NamedTuple{(:T, :C, :r, :Bounds, :NPhases)},
        SFM::NamedTuple{(:t, :φ, :X, :n)},
        SFM0::NamedTuple{(:t, :φ, :X, :n)},
    )
        Ind = SFM.t > T
        if Ind
            s = T - SFM0.t
            X = UpdateXt(Model = Model, SFM0 = SFM0, S = s)
            SFM = (t = T, φ = SFM0.φ, X = X, n = SFM0.n)
        end
        return (Ind = Ind, SFM = SFM)
    end
    # SFFM METHOD
    function FixedTimeFun(
        Model::NamedTuple{(:T, :C, :r, :Bounds, :NPhases)},
        SFFM::NamedTuple{(:t, :φ, :X, :Y, :n)},
        SFFM0::NamedTuple{(:t, :φ, :X, :Y, :n)},
    )
        Ind = SFFM.t > T
        if Ind
            s = T - SFFM0.t
            X = UpdateXt(Model = Model, SFM0 = SFFM0, S = s)
            Y = UpdateYt(Model = Model, SFFM0 = SFFM0, S = s)
            SFFM = (T, SFFM0.φ, X, Y, SFFM0.n)
        end
        return (Ind = Ind, SFFM = SFFM)
    end
    return FixedTimeFun
end

function NJumps(; N::Int)
    # Defines a simple stopping time, 1(n>N), where n is the number of jumps of φ.
    # SFM method
    function NJumpsFun(
        Model::NamedTuple{(:T, :C, :r, :Bounds, :NPhases)},
        SFM::NamedTuple{(:t, :φ, :X, :n)},
        SFM0::NamedTuple{(:t, :φ, :X, :n)},
    )
        Ind = SFM.n >= N
        return (Ind = Ind, SFM = SFM)
    end
    # SFFM method
    function NJumpsFun(
        Model::NamedTuple{(:T, :C, :r, :Bounds, :NPhases)},
        SFFM::NamedTuple{(:t, :φ, :X, :Y, :n)},
        SFFM0::NamedTuple{(:t, :φ, :X, :Y, :n)},
    )
        Ind = SFFM.n >= N
        return (Ind = Ind, SFFM = SFFM)
    end
    return NJumpsFun
end

function FirstExitX(; u::Real, v::Real)
    # Defines a first exit stopping time rule for the interval [u,v]
    # Inputs:
    #   u,v - scalars
    # Outputs:
    #   FirstExitFun(Model,t::Float64,φ,X,n::Int), a function with inputs;
    #   t is the current time, φ the current phase, X, the current level, n
    #   the number of transition to time t, and returns a tuple (Ind,SFM),
    #   where Ind is a boolen specifying whether the stopping time, τ, has
    #   occured or not, and SFM is a tuple (τ,φ(τ),X(τ),n).

    # SFM Method
    function FirstExitXFun(
        Model::NamedTuple{(:T, :C, :r, :Bounds, :NPhases)},
        SFM::NamedTuple{(:t, :φ, :X, :n)},
        SFM0::NamedTuple{(:t, :φ, :X, :n)},
    )
        Ind = SFM.X > v || SFM.X < u
        if Ind
            if SFM.X > v
                X = v
            else
                X = u
            end
            s = (X - SFM0.X) / Model.C[SFM0.φ] # can't exit with c = 0
            t = SFM0.t + s
            SFM = (t = t, φ = SFM0.φ, X = X, n = SFM0.n)
        end
        return (Ind = Ind, SFM = SFM)
    end
    # SFFM Method
    function FirstExitXFun(
        Model::NamedTuple{(:T, :C, :r, :Bounds, :NPhases)},
        SFFM::NamedTuple{(:t, :φ, :X, :Y, :n)},
        SFFM0::NamedTuple{(:t, :φ, :X, :Y, :n)},
    )
        Ind = SFFM.X > v || SFFM.X < u
        if Ind
            if SFFM.X > v
                X = v
            else
                X = u
            end
            s = (X - SFFM0.X) / Model.C[SFFM0.φ] # can't exit with c = 0.
            t = SFFM0.t + s
            Y = UpdateYt(Model = Model, SFFM0 = SFFM0, S = s)
            SFFM = (t, SFFM0.φ, X, Y, SFFM0.n)
        end
        return (Ind = Ind, SFFM = SFFM)
    end
    return FirstExitXFun
end

function FirstExitY(; u::Real, v::Real) #InOutYLevel(; y::Real)
    # Defines a first exit stopping time rule for the Y-in-out fluid hitting y
    # Inputs:
    #   y - scalar
    # Outputs:
    #   FirstExitFun(Model,t::Float64,φ,X,n::Int), a function with inputs;
    #   t is the current time, φ the current phase, X, the current level, n
    #   the number of transition to time t, and returns a tuple (Ind,SFM),
    #   where Ind is a boolen specifying whether the stopping time, τ, has
    #   occured or not, and SFM is a tuple (τ,φ(τ),X(τ),n).

    # SFFM Method
    function FirstExitYFun(
        Model::NamedTuple{(:T, :C, :r, :Bounds, :NPhases)},
        SFFM::NamedTuple{(:t, :φ, :X, :Y, :n)},
        SFFM0::NamedTuple{(:t, :φ, :X, :Y, :n)},
    )
        Ind = SFFM.Y < u || SFFM.Y > v
        if Ind
            idx = [SFFM.Y < u; SFFM.Y > v]
            boundaryHit = [u;v][idx][1]
            YFun(t) = UpdateYt(Model = Model, SFFM0 = SFFM0, S = t) - boundaryHit
            S = SFFM.t - SFFM0.t
            tstar = fzero(f = YFun, a = 0, b = S)
            X = UpdateXt(Model = Model, SFM0 = SFFM0, S = tstar)
            t = SFFM0.t + tstar
            Y = boundaryHit
            SFFM = (t, SFFM0.φ, X, Y, SFFM0.n)
        end
        return (Ind = Ind, SFFM = SFFM)
    end
    return FirstExitYFun
end

function fzero(; f::Function, a::Real, b::Real, err::Float64 = 1e-8)
    # finds zeros of f using the bisection method
    c = a + (b - a) / 2
    while a < c < b
        fc = f(c)
        if abs(fc) < err
            break
        end
        if f(a) * fc < 0
            a, b = a, c
        else
            a, b = c, b
        end
        c = a + (b - a) / 2
    end
    return c
end

function Sims2Dist(;
    Model::NamedTuple{(:T, :C, :r, :Bounds, :NPhases)},
    Mesh::NamedTuple{
        (
            :NBases,
            :CellNodes,
            :Fil,
            :Δ,
            :NIntervals,
            :Nodes,
            :TotalNBases,
            :Basis,
        ),
    },
    sims::NamedTuple{(:t, :φ, :X, :Y, :n)},
    type::String = "density",
)

    if type == "density"
        distribution = zeros(Float64, Mesh.NBases, Mesh.NIntervals, Model.NPhases)
    elseif type == "probability"
        distribution = zeros(Float64, 1, Mesh.NIntervals, Model.NPhases)
    end
    pm = zeros(Float64, sum(Model.C .<= 0) + sum(Model.C .>= 0))
    pc = 0
    qc = 0
    xvals = Mesh.CellNodes
    for i = 1:Model.NPhases
        whichsims =
            (sims.φ .== i) .&
            (sims.X .!= Model.Bounds[1, 1]) .&
            (sims.X .!= Model.Bounds[1, end])
        data = sims.X[whichsims]
        totalprob = sum(whichsims) / length(sims.φ)
        if type == "probability"
            h = StatsBase.fit(StatsBase.Histogram, data, Mesh.Nodes)
            h = h.weights ./ sum(h.weights) * totalprob
            distribution[:, :, i] = h
            xvals = Mesh.CellNodes[1, :] + Mesh.Δ / 2
        elseif type == "density"
            U = KernelDensity.kde(
                data,
                boundary = (Model.Bounds[1, 1], Model.Bounds[1, end]),
            )
            distribution[:, :, i] =
                reshape(
                    KernelDensity.pdf(U, Mesh.CellNodes[:])',
                    Mesh.NBases,
                    Mesh.NIntervals,
                ) * totalprob
        end

        if Model.C[i] <= 0
            pc = pc + 1
            whichsims = (sims.φ .== i) .& (sims.X .== Model.Bounds[1, 1])
            p = sum(whichsims) / length(sims.φ)
            pm[pc] = p
        end
        if Model.C[i] >= 0
            qc = qc + 1
            whichsims = (sims.φ .== i) .& (sims.X .== Model.Bounds[1, end])
            p = sum(whichsims) / length(sims.φ)
            pm[sum(Model.C .<= 0)+qc] = p
        end
    end
    if type == "density" && Mesh.NBases == 1
        distribution = [1; 1] .* distribution
        xvals = [Mesh.CellNodes-Mesh.Δ'/2;Mesh.CellNodes+Mesh.Δ'/2]
    end
    return (pm = pm, distribution = distribution, x = xvals, type = type)
end
