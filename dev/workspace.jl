include("examples/meNumerics/discontinuities.jl")

d_fv = SFFM.SFMDistribution{SFFM.FVMesh}(x1_FV,model,fvmesh)
d_me = SFFM.SFMDistribution{SFFM.FRAPMesh}(x1_ME,model,frapmesh)
d_dg = SFFM.SFMDistribution{SFFM.DGMesh}(x1_DG,model,dgmesh)

plot(layout = SFFM.NPhases(model))
for i in SFFM.phases(model)
    f_fv(x) = SFFM.cdf(d_fv,model,x,i)
    f_me(x) = SFFM.cdf(d_me,model,x,i)
    f_dg(x) = SFFM.cdf(d_dg,model,x,i)
    plot!(f_fv,-1,13; subplot = i)
    plot!(f_me,-1,13; subplot = i)
    plot!(f_dg,-1,13; subplot = i)
end
# scatter!([f_fv(0)],[6],subplot=1)
# scatter!([f_me(0)],[6],subplot=1)
# scatter!([f_dg(0)],[6],subplot=1)
plot!()
