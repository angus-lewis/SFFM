include("../../src/SFFM.jl")
using LinearAlgebra, Plots, JLD2, GLM

## define the model(s)
include("exampleModelDef.jl")

## load sims
@load pwd()*"/examples/paperNumerics/dump/sims.jld2" sims

## analytic X distribution for comparison
# construction
Ψₓ = SFFM.PsiFunX( approxModel)
ξₓ = SFFM.MakeXiX( approxModel, Ψₓ)
pₓ, πₓ, Πₓ, Kₓ = SFFM.StationaryDistributionX( approxModel, Ψₓ, ξₓ)

## section 4.5: error for approximation of stationary distribution of X
Basis = "lagrange"
Δs = [1.6; 0.8; 0.4]
NBasesRange = [1; 3; 5; 7]

πnorms = zeros(length(Δs), length(NBasesRange))
Ψnorms = zeros(length(Δs), length(NBasesRange))
times = Array{Any}(undef, length(Δs), length(NBasesRange))
gctimes = Array{Any}(undef, length(Δs), length(NBasesRange))
mems = Array{Any}(undef, length(Δs), length(NBasesRange))
alloc = Array{Any}(undef, length(Δs), length(NBasesRange))
approxSpec = Array{Any}(undef, length(Δs), length(NBasesRange))

πnormsme = zeros(length(Δs), length(NBasesRange))
Ψnormsme = zeros(length(Δs), length(NBasesRange))
timesme = Array{Any}(undef, length(Δs), length(NBasesRange))
gctimesme = Array{Any}(undef, length(Δs), length(NBasesRange))
memsme = Array{Any}(undef, length(Δs), length(NBasesRange))
allocme = Array{Any}(undef, length(Δs), length(NBasesRange))
approxSpecme = Array{Any}(undef, length(Δs), length(NBasesRange))

πnormsfv = zeros(length(Δs), length(NBasesRange))
Ψnormsfv = zeros(length(Δs), length(NBasesRange))
timesfv = Array{Any}(undef, length(Δs), length(NBasesRange))
gctimesfv = Array{Any}(undef, length(Δs), length(NBasesRange))
memsfv = Array{Any}(undef, length(Δs), length(NBasesRange))
allocfv = Array{Any}(undef, length(Δs), length(NBasesRange))
approxSpecfv = Array{Any}(undef, length(Δs), length(NBasesRange))

for d = 1:length(Δs), n = 1:length(NBasesRange)
    # define the mesh for each iteration
    nBases = NBasesRange[n]
    Δtemp = Δs[d]
    println("mesh details; h = ", Δtemp, " NBases = ", nBases)

    # collect time and memory stats
    ~, times[d, n], mems[d, n], gctimes[d, n], alloc[d, n] = @timed begin
        Nodes = collect(approxBounds[1, 1]:Δtemp:approxBounds[1, 2])
        mesh = SFFM.DGMesh(
            approxModel,
            Nodes,
            nBases,
            Basis =  "lagrange",
        )
        approxSpec[d, n] = (Δtemp, nBases, SFFM.TotalNBases(mesh) * SFFM.NPhases(approxModel))

        # compute the marginal via DG
        All = SFFM.MakeAll( approxModel, mesh, approxType = "interpolation")
        Ψ = SFFM.PsiFun( All.D)

        # the distribution of X when Y first returns to 0
        ξ = SFFM.MakeXi( All.B.BDict, Ψ)

        marginalX, p, K = SFFM.MakeLimitDistMatrices(
            All.B.BDict,
            All.D,
            All.R.RDict,
            Ψ,
            ξ,
            mesh,
            approxModel,
        )
    end

    # collect time and memory stats
    ~, timesme[d, n], memsme[d, n], gctimesme[d, n], allocme[d, n] = @timed begin
        Nodes = collect(approxBounds[1, 1]:Δtemp:approxBounds[1, 2])
        mesh = SFFM.FRAPMesh(
            approxModel,
            Nodes,
            nBases,
        )
        dgmesh = SFFM.DGMesh(
            approxModel,
            Nodes,
            nBases,
            Basis =  "lagrange",
        )
        approxSpecme[d, n] = (Δtemp, nBases, SFFM.TotalNBases(mesh) * SFFM.NPhases(approxModel))

        # compute the marginal via FRAP approx
        me = SFFM.MakeME(SFFM.CMEParams[nBases], mean = SFFM.Δ(mesh)[1])
        B = SFFM.MakeBFRAP( approxModel, mesh, me)
        R = SFFM.MakeR( approxModel, dgmesh, approxType = "interpolation")
        D = SFFM.MakeD( mesh, B, R)
        Ψme = SFFM.PsiFun(D)

        # the distribution of X when Y first returns to 0
        ξme = SFFM.MakeXi( B.BDict, Ψme)

        marginalXme, pme, Kme = SFFM.MakeLimitDistMatrices(
            B.BDict,
            D,
            R.RDict,
            Ψme,
            ξme,
            mesh,
            approxModel,
        )
    end

    # collect time and memory stats
    ~, timesfv[d, n], memsfv[d, n], gctimesfv[d, n], allocfv[d, n] = @timed begin
        Nodes = collect(approxBounds[1, 1]:Δtemp:approxBounds[1, 2])
        mesh = SFFM.FVMesh(
            approxModel,
            Nodes,
        )
        approxSpecfv[d, n] = (Δtemp, nBases, SFFM.TotalNBases(mesh) * SFFM.NPhases(approxModel))

        # compute the marginal via FRAP approx
        B = SFFM.MakeBFV(approxModel, mesh, nBases)
        R = SFFM.MakeR( approxModel, mesh, approxType = "interpolation")
        D = SFFM.MakeD( mesh, B, R)
        Ψfv = SFFM.PsiFun(D)

        # the distribution of X when Y first returns to 0
        ξfv = SFFM.MakeXi( B.BDict, Ψfv)

        marginalXfv, pfv, Kfv = SFFM.MakeLimitDistMatrices(
            B.BDict,
            D,
            R.RDict,
            Ψfv,
            ξfv,
            mesh,
            approxModel,
        )
    end
    dgmesh = SFFM.DGMesh(
            approxModel,
            Nodes,
            nBases,
            Basis =  "lagrange",
        )
    frapmesh = SFFM.FRAPMesh(
            approxModel,
            Nodes,
            nBases,
        )
    fvmesh = SFFM.FVMesh(
        approxModel,
        Nodes,
    )
    # convert marginalX to a distribution for analysis
    DGStationaryDist = SFFM.Coeffs2Dist(
        approxModel,
        dgmesh,
        marginalX,
        type = "probability",
    )
    meStationaryDist = SFFM.Coeffs2Dist(
        approxModel,
        frapmesh,
        marginalXme,
        type = "probability",
    )
    fvStationaryDist = SFFM.Coeffs2Dist(
        approxModel,
        fvmesh,
        marginalXfv,
        type = "probability",
    )

    ## stationary distirbution stuff
    # evaluate the analytic result given the mesh
    analyticX = (
        pm = [pₓ[:]; 0; 0],
        distribution =
            Πₓ(Matrix(mesh.Nodes[2:end]')) - Πₓ(Matrix(mesh.Nodes[1:end-1]')),
        x = mesh.Nodes[1:end-1] + SFFM.Δ(mesh) / 2,
        type = "probability",
    )

    # save them
    πnorms[d, n] = SFFM.starSeminorm( DGStationaryDist, analyticX)
    πnormsme[d, n] = SFFM.starSeminorm( meStationaryDist, analyticX)
    πnormsfv[d, n] = SFFM.starSeminorm( fvStationaryDist, analyticX)
end

let p = plot()
    types = [:solid,:dash,:dashdot,:dashdotdot]
    for n = 1:length(NBasesRange)
        p = plot!(
            Δs,
            πnorms[:, n],
            xaxis = :log,
            yaxis = :log,
            xlabel = "h",
            ylabel = "error",
            label = "DG: N_k = " * string(NBasesRange[n]),
            legend = :outertopright,
            linestyle = types[n],
            seriestype = :line,
            # markershape = :auto,
            title = "Error",
            grid = false,
            color = :red,
        )
        p = plot!(
            Δs,
            πnormsme[:, n],
            label = "ME: N_k = " * string(NBasesRange[n]),
            linestyle = types[n],
            seriestype = :line,
            # markershape = :auto,
            color = :blue,
        )
        p = plot!(
            Δs,
            πnormsfv[:, n],
            label = "FV: N = " * string(NBasesRange[n]),
            linestyle = types[n],
            seriestype = :line,
            # markershape = :auto,
            color = :black,
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
            ylabel = "error",
            label = "DG: h = " * string(Δs[d]),
            legend = :outertopright,
            linestyle = types[d],
            seriestype = :line,
            # markershape = :auto,
            title = "Error",
            grid = false,
            color = :red,
        )
        p = plot!(
            NBasesRange,
            πnormsme[d, :],
            label = "ME: h = " * string(Δs[d]),
            linestyle = types[d],
            seriestype = :line,
            # markershape = :auto,
            colot = :blue,
        )
        p = plot!(
            NBasesRange,
            πnormsfv[d, :],
            label = "FV: h = " * string(Δs[d]),
            linestyle = types[d],
            seriestype = :line,
            # markershape = :auto,
            color = :black,
        )
    end
    display(p)
    # savefig(pwd()*"/examples/paperNumerics/dump/piErrorVsNBases.png")
end

begin
    n = 1
    data = log.(πnorms[:,n])
    println("DG rate of convergence for ", NBasesRange[n], " basis functions is ")
    display(
        lm([ones(length(Δs)) log.(Δs)], data)
    )
    println("")

    data = log.(πnormsme[:,n])
    ind = .!isnan.(data)
    data = data[ind]
    println("ME rate of convergence for ", NBasesRange[n], " basis functions is ")
    display(
        lm([ones(length(Δs[ind])) log.(Δs[ind])], data)
    )
    println("")

    data = log.(πnormsfv[:,n])
    ind = .!isnan.(data)
    data = data[ind]
    println("FV rate of convergence for ", NBasesRange[n], " basis functions is ")
    display(
        lm([ones(length(Δs[ind])) log.(Δs[ind])], data)
    )
    println("")

    n = 2
    data = log.(πnorms[:,n])
    println("DG rate of convergence for ", NBasesRange[n], " basis functions is ")
    display(
        lm([ones(length(Δs[ind])) log.(Δs[ind])], data)
    )
    println("")

    data = log.(πnormsme[:,n])
    println("ME rate of convergence for ", NBasesRange[n], " basis functions is ")
    display(
        lm([ones(length(Δs[ind])) log.(Δs[ind])], data)
    )
    println("")

    data = log.(πnormsfv[:,n])
    println("FV rate of convergence for ", NBasesRange[n], " basis functions is ")
    display(
        lm([ones(length(Δs[ind])) log.(Δs[ind])], data)
    )
    println("")

    n = 3
    data = log.(πnorms[:,n])
    println("DG rate of convergence for ", NBasesRange[n], " basis functions is ")
    display(
        lm([ones(length(Δs[ind])) log.(Δs[ind])], data)
    )
    println("")

    data = log.(πnormsme[:,n])
    println("ME rate of convergence for ", n, " basis functions is ")
    display(
        lm([ones(length(Δs[ind])) log.(Δs[ind])], data)
    )
    println("")

    data = log.(πnormsfv[:,n])
    println("FV rate of convergence for ", n, " basis functions is ")
    display(
        lm([ones(length(Δs[ind])) log.(Δs[ind])], data)
    )
    println("")
end

for d = 1:length(Δs)
    data = log.(πnorms[d,:])
    ind = .!isnan.(data)
    data = data[ind]
    println("DG rate of convergence for h = ", Δs[d])
    display(
        lm([ones(length(NBasesRange[ind])) NBasesRange[ind]], data)
    )
    println("")
end

for d = 1:length(Δs)
    data = log.(πnormsme[d,:])
    ind = .!isnan.(data)
    data = data[ind]
    println("ME rate of convergence for h = ", Δs[d])
    display(
        lm([ones(length(NBasesRange[ind])) NBasesRange[ind]], data)
    )
    println("")
end

for d = 1:length(Δs)
    data = log.(πnormsfv[d,:])
    ind = .!isnan.(data)
    data = data[ind]
    println("FV rate of convergence for h = ", Δs[d])
    display(
        lm([ones(length(NBasesRange[ind])) NBasesRange[ind]], data)
    )
    println("")
end

mp = (log.(πnorms[1:end-1, :]) .- log.(πnorms[2:end, :])) ./
    (log.(Δs[1:end-1]) - log.(Δs[2:end]))
mpme = (log.(πnormsme[1:end-1, :]) .- log.(πnormsme[2:end, :])) ./
    (log.(Δs[1:end-1]) - log.(Δs[2:end]))
mpfv = (log.(πnormsfv[1:end-1, :]) .- log.(πnormsfv[2:end, :])) ./
    (log.(Δs[1:end-1]) - log.(Δs[2:end]))    
np = (log.(πnorms[:, 1:end-1]) .- log.(πnorms[:, 2:end])) ./
    ((NBasesRange[1:end-1]) - (NBasesRange[2:end]))'
npme = (log.(πnormsme[:, 1:end-1]) .- log.(πnormsme[:, 2:end])) ./
    ((NBasesRange[1:end-1]) - (NBasesRange[2:end]))'
npme = (log.(πnormsfv[:, 1:end-1]) .- log.(πnormsfv[:, 2:end])) ./
    ((NBasesRange[1:end-1]) - (NBasesRange[2:end]))'

stats = Array{Any}(undef,length(Δs)+1,length(NBasesRange)+1)
statsme = Array{Any}(undef,length(Δs)+1,length(NBasesRange)+1)
statsfv = Array{Any}(undef,length(Δs)+1,length(NBasesRange)+1)
for n in 1:length(NBasesRange), d in 1:length(Δs)
        stats[d+1,n+1] = (πnorms[d,n], times[d,n], mems[d,n]/(1024*1024), approxSpec[d,n][3])
        statsme[d+1,n+1] = (πnormsme[d,n], timesme[d,n], memsme[d,n]/(1024*1024), approxSpecme[d,n][3])
        statsfv[d+1,n+1] = (πnormsfv[d,n], timesfv[d,n], memsfv[d,n]/(1024*1024), approxSpecfv[d,n][3])
end
stats[1,2:end] = NBasesRange
stats[2:end,1] = Δs
statsme[1,2:end] = NBasesRange
statsme[2:end,1] = Δs
statsfv[1,2:end] = NBasesRange
statsfv[2:end,1] = Δs

using NumericIO
innards = stats[2:end,2:end]
innardsme = statsme[2:end,2:end]
innardsfv = statsfv[2:end,2:end]
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
          &      & 1            & 3           & 5           & 7           \\\\\\hline"
        )
    for d in 1:length(Δs)
        println(Δs[d], " &")
        println("
        \\(\\begin{array}{l}
            \\mbox{ }\\\\
            \\mbox{error}
            \\\\ \\mbox{time (sec)}
            \\\\ \\mbox{memory (MB)}
            \\\\ n_\\varphi
        \\end{array}\\)"
        )
        for n in 1:length(NBasesRange)
            println("&")
                println("
                \\(\\begin{array}{l}
                    \\mbox{DG}\\\\
                    ", formatted(innards[d,n][1],fmt),"}
                    \\\\",round(innards[d,n][2],sigdigits=3),"
                    \\\\",round(innards[d,n][3],sigdigits=3),"
                    \\\\",convert(Int,round(innards[d,n][4],digits=0)),"
                \\end{array}\\)\\(\\begin{array}{l}
                \\mbox{ME}\\\\
                ", formatted(innardsme[d,n][1],fmt),"}
                \\\\",round(innardsme[d,n][2],sigdigits=3),"
                \\\\",round(innardsme[d,n][3],sigdigits=3),"
                \\\\",convert(Int,round(innardsme[d,n][4],digits=0)),"
            \\end{array}\\)\\(\\begin{array}{l}
            \\mbox{FV}\\\\
            ", formatted(innardsfv[d,n][1],fmt),"}
            \\\\",round(innardsfv[d,n][2],sigdigits=3),"
            \\\\",round(innardsfv[d,n][3],sigdigits=3),"
            \\\\",convert(Int,round(innardsfv[d,n][4],digits=0)),"
        \\end{array}\\)"
                )
        end
        println("\\\\ \\hline")
    end
    println("\\end{tabular}")
end
