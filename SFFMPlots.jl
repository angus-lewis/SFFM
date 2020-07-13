import Plots
using Plots: Animation, frame, gif

function PlotVt(; Nodes, Vt, C, YMAX = 1, PointMass = true, labels = [])
    if isempty(labels)
        labels = string.(C)
    end
    Interval = [Nodes[1]; Nodes[end]]
    NNodes = length(Nodes)
    ΔEnds = [Nodes[2] - Nodes[1]; Nodes[end] - Nodes[end-1]]
    FivePercent = (Interval[2] - Interval[1]) * 0.025 .* [-1; 1]
    Plt = Plots.plot(xlims = Interval + FivePercent, ylims = (0, YMAX))
    for i = 1:length(C)
        if PointMass
            if C[i] > 0
                CtsIdx = [1; 1:NNodes-1]
                xIdx = [1; 2; 2:NNodes-1]
                PMIdx = NNodes
                ΔIdx = 2
            elseif C[i] < 0
                CtsIdx = [2:NNodes; NNodes]
                xIdx = [2:NNodes-1; NNodes - 1; NNodes]
                PMIdx = 1
                ΔIdx = 1
            else
                CtsIdx = 2:NNodes-1
                xIdx = CtsIdx
                PMIdx = [1; NNodes]
                ΔIdx = [1; 2]
            end # end if C[i]
            Plt = Plots.plot!(
                Nodes[xIdx],
                Vt[CtsIdx, i],
                label = string("Phase ", i, " rate: ", C[i], " and ", labels[i]),
                color = i,
            )
            Plt = Plots.scatter!(
                [Nodes[PMIdx]],
                [Vt[PMIdx, i] .* ΔEnds[ΔIdx]],
                marker = :hexagon,
                label = string("mass"),
                color = i,
            )
        else
            Plt = Plots.plot!(
                Nodes,
                Vt[:, i],
                label = string("Phase ", i, " rate: ", labels[i]),
                color = i,
            )
        end # end if PointMass
    end # end for i
    return (Plt = Plt)
end # end PlotVt

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
