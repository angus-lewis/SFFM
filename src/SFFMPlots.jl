"""
Add to a figure a plot of a SFM distribution.

    PlotSFM!(p;
        Model::NamedTuple{(:T, :C, :r, :Bounds, :NPhases, :SDict, :TDict)},
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
        Dist::NamedTuple{(:pm, :distribution, :x, :type)},
        color = 1,
    )

# Arguments
- `p::Plots.Plot{Plots.GRBackend}`: a plot object as initialised by `PlotSFM()`
- `Model`: A model tuple from `MakeModel`
- `Mesh`: A Mesh tuple from `MakeMesh`
- `Dist`: A distribution object as output from `Coeff2Dist` or `Sims2Dist`
- `color`: (optional) a colour specifier for the plot
"""
function PlotSFM!(p;
    Model::NamedTuple{(:T, :C, :r, :Bounds, :NPhases, :SDict, :TDict)},
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
    Dist::NamedTuple{(:pm, :distribution, :x, :type)},
    color = 1,
    label = false,
    marker = :none,
    seriestype = :line,
    markersize = 4,
    titles = false,
    jitter = 0,
)
    pc = 0
    qc = 0
    # yLimValues = (0.0, 0.0)
    for i = 1:Model.NPhases
        if Dist.type == "density"
            p = Plots.plot!(
                Dist.x,
                Dist.distribution[:, :, i],
                linecolor = color,
                subplot = i,
                title = "φ=" * string(i),
                ylabel = Dist.type,
                label = false,
                markershape = marker,
                markercolor = color,
                seriestype = seriestype,
                grid = false,
                markersize = markersize,
            )
        elseif Dist.type === "probability"
            p = Plots.bar!(
                Dist.x[:],
                Dist.distribution[:, :, i][:],
                alpha = 0.25,
                fillcolor = color,
                bar_width = Mesh.Δ,
                subplot = i,
                title = "φ=" * string(i),
                ylabel = Dist.type,
                label = false,
                grid = false,
            )
        end
        if Model.C[i] <= 0
            pc = pc + 1
            x = [Model.Bounds[1,1]]
            y = [Dist.pm[pc]]
            p = Plots.scatter!(
                x .- jitter/2 .+ jitter*rand(),
                y,
                markershape = :o,
                markercolor = color,
                markersize = 4,
                alpha = 0.3,
                subplot = i,
                label = false,
                grid = false,
            )
        end
        if Model.C[i] >= 0
            qc = qc + 1
            x = [Model.Bounds[1,end]]
            y = [Dist.pm[sum(Model.C .>= 0) + qc]]
            p = Plots.scatter!(
                x .- jitter/2 .+ jitter*rand(),
                y,
                markershape = :o,
                markercolor = color,
                markersize = 4,
                alpha = 0.3,
                subplot = i,
                label = false,
                grid = false,
            )
        end
        # yLimValues = (
        #     -0,
        #     max(maximum(Dist.distribution[:, :, i][:]), maximum(y))*1.025,
        # )
        p = Plots.plot!(
            # ylims = yLimValues,
            subplot = i,
            tickfontsize = 7,
            guidefontsize = 10,
            titlefontsize = 12,
        )
    end
    p = Plots.plot!(
        [],
        subplot = Model.NPhases+1,
        label = label,
        showaxis = false,
        grid = false,
        seriestype = seriestype,
        markershape = marker,
        markercolor = color,
        color = color,
        legendfontsize = 7,
    )

    return p
end # end PlotSFM!

"""
Initialise and plot a SFM distribution.

    PlotSFM(
        Model::NamedTuple{(:T, :C, :r, :Bounds, :NPhases, :SDict, :TDict)},
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
        Dist::NamedTuple{(:pm, :distribution, :x, :type)},
        color = 1,
    )

# Arguments
- `Model`: A model tuple from `MakeModel`
- `Mesh`: A Mesh tuple from `MakeMesh`
- `Dist`: A distribution object as output from `Coeff2Dist` or `Sims2Dist`
- `color`: (optional) a colour specifier for the plot

# Output
- a plot object of type `Plots.Plot{Plots.GRBackend}` with `NPhases` subplots
    containing a plot of the distribution for each phase.
"""
function PlotSFM(;
    Model::NamedTuple{(:T, :C, :r, :Bounds, :NPhases, :SDict, :TDict)},
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
    } = (
        NBases = 0,
        CellNodes = Float64[],
        Fil = Dict(),
        Δ = 0,
        NIntervals = 0,
        Nodes = 0,
        TotalNBases = 0,
        Basis = "",
    ),
    Dist::NamedTuple{(:pm, :distribution, :x, :type)} =
        (pm=Float64[],distribution=Float64[],x=Float64[],type=""),
    color = 1,
    label = false,
    marker = :none,
    seriestype = :line,
    markersize = 4,
    titles = false,
    jitter = 0,
)
    p = Plots.plot(layout = Plots.@layout([Plots.grid((Model.NPhases+1)÷2, 2) A{0.001w}]))
    if length(Dist.distribution) != 0
        p = SFFM.PlotSFM!(p;
            Model = Model,
            Mesh = Mesh,
            Dist = Dist,
            color = color,
            label = label,
            marker = marker,
            seriestype = seriestype,
            markersize = markersize,
            titles = titles,
            jitter = jitter,
        )
    end

    return p
end # end PlotSFM
