# include("../../src/SFFM.jl")
# using LinearAlgebra, Plots, JLD2, GLM
using GLM

# ## define the model(s)
# include("exampleModelDef.jl")

## load sims
@load pwd()*"/examples/paperNumerics/dump/sims.jld2" sims

## analytic X distribution for comparison
# construction
Ψₓ = SFFM.PsiFunX(Model = approxModel)
ξₓ = SFFM.MakeXiX(Model = approxModel, Ψ = Ψₓ)
pₓ, πₓ, Πₓ, Kₓ = SFFM.StationaryDistributionX(Model = approxModel, Ψ = Ψₓ, ξ = ξₓ)

## section 4.5: error for approximation of stationary distribution of X
Basis = "lagrange"
Δs = [1.6; 0.8; 0.4; 0.2; 0.1; 0.05]
NBasesRange = [1; 2; 3; 4]

πnorms = zeros(length(Δs), length(NBasesRange))
Ψnorms = zeros(length(Δs), length(NBasesRange))
times = Array{Any}(undef, length(Δs), length(NBasesRange))
gctimes = Array{Any}(undef, length(Δs), length(NBasesRange))
mems = Array{Any}(undef, length(Δs), length(NBasesRange))
alloc = Array{Any}(undef, length(Δs), length(NBasesRange))
approxSpec = Array{Any}(undef, length(Δs), length(NBasesRange))

for d = 1:length(Δs), n = 1:length(NBasesRange)
    if d + 2 * n <= 10
        # define the mesh for each iteration
        NBases = NBasesRange[n]
        Δ = Δs[d]
        println("Mesh details; Δ = ", Δ, " NBases = ", NBases)

        # collect time and memory stats
        ~, times[d, n], mems[d, n], gctimes[d, n], alloc[d, n] = @timed begin
            Nodes = collect(approxBounds[1, 1]:Δ:approxBounds[1, 2])
            Mesh = SFFM.MakeMesh(
                Model = approxModel,
                Nodes = Nodes,
                NBases = NBases,
                Basis = Basis,
            )
            approxSpec[d, n] = (Δ, NBases, Mesh.TotalNBases * approxModel.NPhases)

            # compute the marginal via DG
            All = SFFM.MakeAll(Model = approxModel, Mesh = Mesh, approxType = "projection")
            Ψ = SFFM.PsiFun(D = All.D)

            # the distribution of X when Y first returns to 0
            ξ = SFFM.MakeXi(B = All.B.BDict, Ψ = Ψ)

            marginalX, p, K = SFFM.MakeLimitDistMatrices(;
                B = All.B.BDict,
                D = All.D,
                R = All.R.RDict,
                Ψ = Ψ,
                ξ = ξ,
                Mesh = Mesh,
            )
        end
        # convert marginalX to a distribution for analysis
        DGStationaryDist = SFFM.Coeffs2Dist(
            Model = approxModel,
            Mesh = Mesh,
            Coeffs = marginalX,
            type = "probability",
        )

        ## stationary distirbution stuff
        # evaluate the analytic result given the mesh
        analyticX = (
            pm = [pₓ[:]; 0; 0],
            distribution =
                Πₓ(Matrix(Mesh.Nodes[2:end]')) - Πₓ(Matrix(Mesh.Nodes[1:end-1]')),
            x = Mesh.Nodes[1:end-1] + Mesh.Δ / 2,
            type = "probability",
        )

        # save them
        πnorms[d, n] = SFFM.starSeminorm(d1 = DGStationaryDist, d2 = analyticX)

##      ## now do Ψ
        ## turn sims into a cdf
        simprobs =
            SFFM.Sims2Dist(Model = simModel, Mesh = Mesh, sims = sims, type = "probability")

        # do DG for Ψ
        theNodes = Mesh.CellNodes[:, convert(Int, ceil(5 / Δ))]
        basisValues = zeros(length(theNodes))
        for n = 1:length(theNodes)
            basisValues[n] =
                prod(5.0 .- theNodes[[1:n-1; n+1:end]]) ./
                prod(theNodes[n] .- theNodes[[1:n-1; n+1:end]])
        end
        initpm = [
            zeros(sum(approxModel.C .<= 0)) # LHS point mass
            zeros(sum(approxModel.C .>= 0)) # RHS point mass
        ]
        initprobs = zeros(Float64, Mesh.NBases, Mesh.NIntervals, approxModel.NPhases)
        initprobs[:, convert(Int, ceil(5 / Δ)), 3] =
            basisValues' * All.Matrices.Local.V.V * All.Matrices.Local.V.V' .* 2 / Δ
        initdist =
            (pm = initpm, distribution = initprobs, x = Mesh.CellNodes, type = "density") # convert to a distribution object so we can apply Dist2Coeffs
        # convert to Coeffs α in the DG context
        x0 = SFFM.Dist2Coeffs(Model = approxModel, Mesh = Mesh, Distn = initdist)
        # the initial condition on Ψ is restricted to + states so find the + states
        plusIdx = [
            Mesh.Fil["p+"]
            repeat(Mesh.Fil["+"]', Mesh.NBases, 1)[:]
            Mesh.Fil["q+"]
        ]
        # get the elements of x0 in + states only
        x0 = x0[plusIdx]'

        # compute x-distribution at the time when Y returns to 0
        w = x0 * Ψ
        # this can occur in - states only, so find the - states
        minusIdx = [
            Mesh.Fil["p-"]
            repeat(Mesh.Fil["-"]', Mesh.NBases, 1)[:]
            Mesh.Fil["q-"]
        ]
        # then map to the whole state space for analysis
        z = zeros(
            Float64,
            Mesh.NBases * Mesh.NIntervals * approxModel.NPhases +
            sum(approxModel.C .<= 0) +
            sum(approxModel.C .>= 0),
        )
        z[minusIdx] = w

        # convert to a distribution object
        DGΨProbs = SFFM.Coeffs2Dist(
            Model = approxModel,
            Mesh = Mesh,
            Coeffs = z,
            type = "probability",
        )

        Ψnorms[d, n] = SFFM.starSeminorm(d1 = DGΨProbs, d2 = simprobs)
    else
        πnorms[d, n] = NaN
        Ψnorms[d, n] = NaN
        times[d, n], mems[d, n], gctimes[d, n], alloc[d, n] = NaN, NaN, NaN, NaN
    end
end

let p = plot()
    types = [:solid,:dash,:dashdot,:dashdotdot]
    for n = 1:length(NBasesRange)
        p = plot!(
            Δs,
            πnorms[:, n],
            xaxis = :log,
            yaxis = :log,
            xlabel = "log(Δ)",
            ylabel = "log error",
            label = "N_k = " * string(NBasesRange[n]),
            legend = :outertopright,
            linestyle = types[n],
            seriestype = :line,
            markershape = :auto,
            title = "Error",
            grid = false,
        )
    end
    display(p)
    # savefig(pwd()*"/examples/paperNumerics/dump/piErrorVsDelta.png")
end

let p = plot()
    types = [:solid,:dash,:dashdot,:dashdotdot,:solid,:dash]
    for d = 1:length(Δs)
        p = plot!(
            NBasesRange,
            πnorms[d, :],
            yaxis = :log,
            xlabel = "N_k",
            ylabel = "log error",
            label = "Δ = " * string(Δs[d]),
            legend = :outertopright,
            linestyle = types[d],
            seriestype = :line,
            markershape = :auto,
            title = "Error",
            grid = false,
        )
    end
    display(p)
    # savefig(pwd()*"/examples/paperNumerics/dump/piErrorVsNBases.png")
end

begin
    n = 1
    data = log.(πnorms[:,n])
    ind = .!isnan.(data)
    data = data[ind]
    println("rate of convergence for ", n, " basis functions is ")
    display(
        lm([ones(length(Δs[ind])) log.(Δs[ind])], data)
    )
    println("")

    n = 2
    data = log.(πnorms[:,n])
    ind = .!isnan.(data)
    data = data[ind]
    println("rate of convergence for ", n, " basis functions is ")
    display(
        lm([ones(length(Δs[ind])) log.(Δs[ind])], data)
    )
    println("")

    n = 3
    data = log.(πnorms[:,n])
    ind = .!isnan.(data)
    ind[4] = false
    data = data[ind]
    data = data
    println("rate of convergence for ", n, " basis functions is ")
    display(
        lm([ones(length(Δs[ind])) log.(Δs[ind])], data)
    )
    println("")
end

for d = 1:6
    data = log.(πnorms[d,:])
    ind = .!isnan.(data)
    data = data[ind]
    println("rate of convergence for Δ = ", Δs[d])
    display(
        lm([ones(length(NBasesRange[ind])) NBasesRange[ind]], data)
    )
    println("")
end

mp =
    (log.(πnorms[1:end-1, :]) .- log.(πnorms[2:end, :])) ./
    (log.(Δs[1:end-1]) - log.(Δs[2:end]))
np =
    (log.(πnorms[:, 1:end-1]) .- log.(πnorms[:, 2:end])) ./
    ((NBasesRange[1:end-1]) - (NBasesRange[2:end]))'

let q = plot()
    for n = 1:length(NBasesRange)
        q = plot!(
            Δs,
            Ψnorms[:, n],
            xaxis = :log,
            yaxis = :log,
            xlabel = "Δ",
            ylabel = "log error",
            label = "NBases = " * string(NBasesRange[n]),
            legend = :outertopright,
            seriestype = :line,
            markershape = :diamond,
            title = "Error for Ψ",
        )
    end
    display(q)
    # savefig(pwd()*"/examples/paperNumerics/dump/psiErrorVsDelta.png")
end

mq =
    (log.(Ψnorms[1:end-1, :]) .- log.(Ψnorms[2:end, :])) ./
    (log.(Δs[1:end-1]) - log.(Δs[2:end]))

stats = Array{Any}(undef,length(Δs)+1,length(NBasesRange)+1)
for n in 1:length(NBasesRange), d in 1:length(Δs)
    if d + 2 * n <= 10
        stats[d+1,n+1] = (πnorms[d,n], times[d,n], mems[d,n]/(1024*1024), approxSpec[d,n][3])
    else
        stats[d+1,n+1] = (0,0,0,0)
    end
end
stats[1,2:end] = NBasesRange
stats[2:end,1] = Δs
display(stats)
using NumericIO
innards = stats[2:end,2:end]
asciiexponentfmt = NumericIO.IOFormattingExpNum(
	"\\times10^{", false, '+', '-', NumericIO.ASCII_SUPERSCRIPT_NUMERALS
)
fmt = NumericIO.IOFormattingReal(asciiexponentfmt,
	ndigits=3, decpos=0, decfloating=true, eng=false, minus='-', inf="Inf"
)

begin
    println("\\begin{tabular}{ | c | l | l | l | l | l | }")
    println("\\hline")
    println("\\multirow{2}{*}{\\(\\Delta\\)} & \\multirow{2}{*}{}& \\multicolumn{4}{c|}{Number of basis functions, \\(N_k\\)} \\\\ \\cline{3-6}
          &      & 1            & 2           & 3           & 4           \\\\\\hline"
        )
    for d in 1:length(Δs)
        println(Δs[d], " &")
        println("
        \\(\\begin{array}{l}
            \\mbox{error}
            \\\\ \\mbox{time (sec)}
            \\\\ \\mbox{memory (MB)}
            \\\\ n_\\varphi
        \\end{array}\\)"
        )
        for n in 1:length(NBasesRange)
            println("&")
            if d + 2 * n <= 10
                println("
                \\(\\begin{array}{l}
                    ", formatted(innards[d,n][1],fmt),"}
                    \\\\",round(innards[d,n][2],sigdigits=3),"
                    \\\\",round(innards[d,n][3],sigdigits=3),"
                    \\\\",convert(Int,round(innards[d,n][4],digits=0)),"
                \\end{array}\\)"
                )
            else
                println(" - ")
            end
        end
        println("\\\\ \\hline")
    end
    println("\\end{tabular}")
end
