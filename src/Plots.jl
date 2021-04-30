"""
Add to a figure a plot of a SFM distribution.

    PlotSFM!(p,
        model::SFFM.Model,
        mesh::SFFM.Mesh,
        Dist::NamedTuple{(:pm, :distribution, :x, :type)};
        color = 1,
        label = false,
        marker = :none,
        seriestype = :line,
        markersize = 4,
        titles = false,
        jitter = 0,
    )

# Arguments
- `p::Plots.Plot{Plots.GRBackend}`: a plot object as initialised by `PlotSFM()`
- `Model`: A Model object
- `mesh`: A Mesh object from `MakeMesh`
- `Dist`: A distribution object as output from `Coeff2Dist` or `Sims2Dist`
- `color`: (optional) a colour specifier for the plot
"""
function PlotSFM!(
    p,
    model::Model,
    mesh::Mesh,
    Dist::NamedTuple{(:pm, :distribution, :x, :type)};
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
    for i = 1:NPhases(model)
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
                bar_width = Δ(mesh),
                subplot = i,
                title = "φ=" * string(i),
                ylabel = Dist.type,
                label = false,
                grid = false,
            )
        end
        if model.C[i] <= 0
            pc = pc + 1
            x = [model.Bounds[1,1]]
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
        if model.C[i] >= 0
            qc = qc + 1
            x = [model.Bounds[1,end]]
            y = [Dist.pm[sum(model.C .>= 0) + qc]]
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

        p = Plots.plot!(
            subplot = i,
            tickfontsize = 10,
            guidefontsize = 12,
            titlefontsize = 14,
        )
    end
    p = Plots.plot!(
        [],
        subplot = 2,
        label = label,
        seriestype = seriestype,
        markershape = marker,
        markercolor = color,
        color = color,
        legendfontsize = 10,
    )

    return p
end # end PlotSFM!

"""
Initialise and plot a SFM distribution.

    PlotSFM(
        model::SFFM.Model;
        mesh::SFFM.Mesh,
        Dist::NamedTuple{(:pm, :distribution, :x, :type)}
        color = 1,
        label = false,
        marker = :none,
        seriestype = :line,
        markersize = 4,
        titles = false,
        jitter = 0,
    )

# Arguments
- `Model`: A Model object
- `mesh`: A Mesh object from `MakeMesh`
- `Dist`: A distribution object as output from `Coeff2Dist` or `Sims2Dist`
- `color`: (optional) a colour specifier for the plot

# Output
- a plot object of type `Plots.Plot{Plots.GRBackend}` with `NPhases` subplots
    containing a plot of the distribution for each phase.
"""
function PlotSFM(
    model::SFFM.Model;
    mesh::SFFM.Mesh = SFFM.DGMesh(),
    dist::NamedTuple{(:pm, :distribution, :x, :type)} =
        (pm=Float64[],distribution=Float64[],x=Float64[],type=""),
    color = 1,
    label = false,
    marker = :none,
    seriestype = :line,
    markersize = 4,
    titles = false,
    jitter = 0,
)
    p = Plots.plot(layout = Plots.@layout(Plots.grid((NPhases(model)+1)÷2, 2)))
    if length(dist.distribution) != 0
        p = SFFM.PlotSFM!(
            p,
            model,
            mesh,
            dist,
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
