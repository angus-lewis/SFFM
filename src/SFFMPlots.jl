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
)
    pc = 0
    qc = 0
    yLimValues = (0.0, 0.0)
    for i = 1:Model.NPhases
        if Dist.type == "density"
            p = Plots.plot!(
                Dist.x,
                Dist.distribution[:, :, i],
                linecolor = color,
                subplot = i,
                title = "φ=" * string(i),
                ylabel = Dist.type,
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
            )
        end
        if Model.C[i] <= 0
            pc = pc + 1
            x = [Model.Bounds[1,1]]
            y = [Dist.pm[pc]]
            p = Plots.scatter!(
                x,
                y,
                markertype = :circle,
                markercolor = color,
                alpha = 0.25,
                subplot = i,
            )
        end
        if Model.C[i] >= 0
            qc = qc + 1
            x = [Model.Bounds[1,end]]
            y = [Dist.pm[sum(Model.C .>= 0) + qc]]
            p = Plots.scatter!(
                x,
                y,
                markertype = :circle,
                markercolor = color,
                alpha = 0.25,
                subplot = i,
            )
        end
        yLimValues = (
            -0.01,
            max(maximum(Dist.distribution[:, :, i][:]), maximum(y)) + 0.01,
        )
        p = Plots.plot!(ylims = yLimValues, subplot = i)
    end

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
    },
    Dist::NamedTuple{(:pm, :distribution, :x, :type)},
    color = 1,
)
    p = Plots.plot(legend = false, layout = ((Model.NPhases+1)÷2, 2))
    p = SFFM.PlotSFM!(p;
        Model = Model,
        Mesh = Mesh,
        Dist = Dist,
        color = color,
    )

    return p
end # end PlotSFM
