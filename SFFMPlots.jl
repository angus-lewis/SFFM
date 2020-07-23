function PlotSFM!(p;
    Model::NamedTuple{(:T, :C, :r, :IsBounded, :Bounds, :NPhases)},
    Mesh::NamedTuple{
        (
            :NBases,
            :CellNodes,
            :Fil,
            :Δ,
            :NIntervals,
            :MeshArray,
            :Nodes,
            :TotalNBases,
            :Basis,
        ),
    },
    Coeffs::Array{<:Real},
    type::String = "density",
    color = 1,
)
    (pm, yvals, xvals) =
        Coeffs2Dist(Model = Model, Mesh = Mesh, Coeffs = Coeffs, type = type)

    pc = 0
    qc = 0
    yLimValues = (0.0, 0.0)
    for i = 1:Model.NPhases
        if type == "density"
            p = Plots.plot!(
                xvals,
                yvals[:, :, i],
                linecolor = color,
                subplot = i,
                title = "φ=" * string(i),
                ylabel = type,
            )
        elseif type === "probability"
            p = Plots.bar!(
                xvals,
                yvals[:, :, i][:],
                alpha = 0.25,
                fillcolor = color,
                bar_width = Mesh.Δ,
                subplot = i,
                title = "φ=" * string(i),
                ylabel = type,
            )
        end
        if Model.C[i] <= 0
            pc = pc + 1
            x = [Mesh.CellNodes[1]]
            y = [pm[pc]]
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
            x = [Mesh.CellNodes[end]]
            y = [pm[sum(Model.C .>= 0) + qc]]
            p = Plots.scatter!(
                x,
                y,
                markertype = :circle,
                markercolor = color,
                alpha = 0.25,
                subplot = i,
            )
        end
        currYLims = Plots.ylims(p[i])
        yLimValues = (min(yLimValues[1], currYLims[1]), max(yLimValues[2], currYLims[2]))
        p = Plots.plot!(ylims = yLimValues, subplot = i)
    end

    return p
end # end PlotVt

function PlotSFM(;
    Model::NamedTuple{(:T, :C, :r, :IsBounded, :Bounds, :NPhases)},
    Mesh::NamedTuple{
        (
            :NBases,
            :CellNodes,
            :Fil,
            :Δ,
            :NIntervals,
            :MeshArray,
            :Nodes,
            :TotalNBases,
            :Basis,
        ),
    },
    Coeffs::Array{<:Real},
    type::String = "density",
    color = 1,
)
    p = Plots.plot(legend = false, layout = (Model.NPhases, 1))
    p = SFFM.PlotSFM!(p,
        Model = Model,
        Mesh = Mesh,
        Coeffs = Coeffs,
        type = type,
        color = color,
    )

    return p
end # end PlotVt

function PlotSFMSim!(p;
    Model::NamedTuple{(:T, :C, :r, :IsBounded, :Bounds, :NPhases)},
    Mesh::NamedTuple{
        (
            :NBases,
            :CellNodes,
            :Fil,
            :Δ,
            :NIntervals,
            :MeshArray,
            :Nodes,
            :TotalNBases,
            :Basis,
        ),
    },
    sims::NamedTuple{(:t, :φ, :X, :Y, :n)},
    type::String = "density",
    color = 1,
)

    yLimValues = (0.0, 0.0)
    vals = SFFM.Sims2Dist(Model = Model, Mesh = Mesh, sims = sims,type=type)
    pc = 0
    qc = 0
    for i = 1:Model.NPhases
        if type == "probability"
            h = vals.distribution[:, :, i][:]
            p = Plots.bar!(
                vals.x,
                h,
                alpha = 0.25,
                bar_width = Mesh.Δ,
                subplot = i,
                ylabel = type,
                color = color,
            )
        elseif type == "density"
            p = Plots.plot!(
                vals.x,
                vals.distribution[:,:,i],
                linecolor = color,
                subplot = i,
                title = "φ=" * string(i),
                ylabel = type,
            )
        end

        if Model.C[i] <= 0
            pc = pc + 1
            x = [Model.Bounds[1,1]]
            y = [vals.pm[pc]]
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
            y = [vals.pm[sum(Model.C.<=0)+qc]]
            p = Plots.scatter!(
                x,
                y,
                markertype = :circle,
                markercolor = color,
                alpha = 0.25,
                subplot = i,
            )
        end
        currYLims = Plots.ylims(p[i])
        yLimValues = (min(yLimValues[1], currYLims[1]), max(yLimValues[2], currYLims[2]))
        p = Plots.plot!(ylims = yLimValues, subplot = i)
    end

    return p
end

function PlotSFMSim(;
    Model::NamedTuple{(:T, :C, :r, :IsBounded, :Bounds, :NPhases)},
    Mesh::NamedTuple{
        (
            :NBases,
            :CellNodes,
            :Fil,
            :Δ,
            :NIntervals,
            :MeshArray,
            :Nodes,
            :TotalNBases,
            :Basis,
        ),
    },
    sims::NamedTuple{(:t, :φ, :X, :Y, :n)},
    type::String = "density",
    color = 1,
)
    p = Plots.plot(legend = false, layout = (Model.NPhases, 1))
    p = SFFM.PlotSFMSim!(p,
        Model = Model,
        Mesh = Mesh,
        sims = sims,
        type = type,
        color = color,
    )

    return p
end


function SFFMGIF(; a0, Nodes, B, Times, C, PointMass = true, YMAX = 1, labels = [])
    gifplt = Plots.@gif for n = 1:length(Times)
        Vt = a0 * exp(B * Times[n])
        Vt = reshape(Vt, (length(Nodes), length(C)))
        SFFM.PlotVt(
            Nodes = Nodes',
            Vt = Vt,
            C = C,
            PointMass = PointMass,
            YMAX = YMAX,
            labels = labels,
        )
    end
    return (gitplt = gifplt)
end
