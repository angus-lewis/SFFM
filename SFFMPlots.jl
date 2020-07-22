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
        Coeffs2Distn(Model = Model, Mesh = Mesh, Coeffs = Coeffs, type = type)

    cp = 0
    cq = 0
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
            cp = cp + 1
            x = [Mesh.CellNodes[1]]
            y = [pm[cp]]
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
            cq = cq + 1
            x = [Mesh.CellNodes[end]]
            y = [pm[cq]]
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
    H = zeros(Mesh.NIntervals, Model.NPhases)
    yLimValues = (0.0, 0.0)
    if type == "probability"
        (pm, H) = SFFM.Sims2Probs(Model = Model, Mesh = Mesh, sims = sims)
    elseif type == "density"
        (pm, U) = SFFM.Sims2PDF(Model = Model, Mesh = Mesh, sims = sims)
    end
    for i = 1:Model.NPhases
        whichsims =
            (sims.φ .== i) .&
            (sims.X .!= Mesh.CellNodes[1]) .&
            (sims.X .!= Mesh.CellNodes[end])
        data = sims.X[whichsims]
        if type == "probability"
            h = H[:, :, i][:]
            p = Plots.bar!(
                Mesh.Nodes,
                h,
                alpha = 0.25,
                bar_width = Mesh.Δ,
                subplot = i,
                ylabel = type,
            )
        elseif type == "density"
            p = Plots.plot!(
                Mesh.CellNodes,
                U[:, :, i],
                linecolor = color,
                subplot = i,
                title = "φ=" * string(i),
                ylabel = type,
            )
        end

        if Model.C[i] <= 0
            x = [Mesh.CellNodes[1]]
            whichsims = (sims.φ .== i) .& (sims.X .== Mesh.CellNodes[1])
            y = [sum(whichsims) / length(sims.φ)]
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
            x = [Mesh.CellNodes[end]]
            whichsims = (sims.φ .== i) .& (sims.X .== Mesh.CellNodes[end])
            y = [sum(whichsims) / length(sims.φ)]
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
    p = Plots.plot!(legend = false, layout = (Model.NPhases, 1))
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
