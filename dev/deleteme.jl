plot()
let
    λ = 1
    a₁ = 1
    a₂ = 0
    a₃ = 0
    na(t) = [0; a₁; a₁*λ*t+a₂; a₁*λ^2*t^2/2+a₂*λ*t+a₃]
    a(t) = na(t)./sum(na(t))
    nb(t) = [0; a₃*λ^2*t^2/2+a₂*λ*t+a₁; a₃*λ*t+a₂; a₃]
    b(t) = nb(t)./sum(nb(t))
    # a(t) = [t; a₁./sum([a₁ ;a₁*λ₁*t+a₂]) ; (a₁*λ₁*t+a₂)./sum([a₁ ;a₁*λ₁*t+a₂])]
    # b(t) = [t; -a(t)[2]; -a(t)[3]]
    # b(t) = [t; (a₁+a₂*λ₂*t)./sum([a₁+a₂*λ₂*t; a₂]); a₂./sum([a₁+a₂*λ₂*t; a₂])]
    # plot(xlims=(0,1),ylims=(0,1))#,layout=(2,1))
    plot(xlabel="a₁(t)",ylabel="a₂(t)",zlabel="a₃(t)") # xlabel="t",
    totaltime = 0
    for n in 1:8
        e₁ = -log(rand())/λ
        e₂ = -log(rand())/λ
        e₃ = -log(rand())/λ
        r = rand()
        erlangrnd = e₁*(r.<(a₁+a₂)) + e₂*(r.<a₁) + e₃
        h = erlangrnd/19
        for t in range(0,erlangrnd,length=20)[2:end]
            if n%2==1
                c = a(t)
                d = a(t-h)
                # display(plot!([c[2];d[2]],[c[3];d[3]],label=false,color=:blue,markershape=:x,seriestype=:line))
                display(plot3d!([d[2];c[2]],[d[3];c[3]],[d[4];c[4]],label=false,color=:blue,markershape=:rtriangle))
                # display(plot!(totaltime.+[c[1];d[1]],[c[2];d[2]],label=false,color=:blue,subplot=1))
                # display(plot!(totaltime.+[c[1];d[1]],[c[3];d[3]],label=false,color=:blue,subplot=2))
            else
                c = b(t)
                d = b(t-h)
                # display(plot!([c[2];d[2]],[c[3];d[3]],label=false,color=:red,markershape=:rtriangle,seriestype=:line))
                display(plot3d!([d[2];c[2]],[d[3];c[3]],[d[4];c[4]],label=false,color=:red,markershape=:ltriangle))
                # display(plot!(totaltime.+[c[1];d[1]],[c[2];d[2]],label=false,color=:red,subplot=1))
                # display(plot!(totaltime.+[c[1];d[1]],[c[3];d[3]],label=false,color=:red,subplot=2))
            end
            # display(c)
            dt = 0.01
            da = (a(t+dt)-a(t-dt))[2:end]./(2*dt)
            # da = da./sum(da)
            db = (b(t+dt)-b(t-dt))[2:end]./(2*dt)
            # db = db./sum(db)
            # display(da)
            # display(db)
            # display(da+db)
        end
        if n%2==1
            temp = a(erlangrnd)
            a₁ = temp[2]
            a₂ = temp[3]
            a₃ = temp[4]
        else
            temp = b(erlangrnd)
            a₁ = temp[2]
            a₂ = temp[3]
            a₃ = temp[4]
        end
        # s = a₁+a₂
        # a₁ = a₁/s
        # a₂ = a₂/s
        na(t) = [0; a₁; a₁*λ*t+a₂; a₁*λ^2*t^2/2+a₂*λ*t+a₃]
        a(t) = na(t)./sum(na(t))
        display(a(0))
        nb(t) = [0; a₃*λ^2*t^2/2+a₂*λ*t+a₁; a₃*λ*t+a₂; a₃]
        b(t) = nb(t)./sum(nb(t))
        display(b(0))
        # a(t) = [t; (a₁)./sum([a₁ ;a₁*λ₁*t+a₂]) ; (a₁*λ₁*t+a₂)./sum([a₁ ;a₁*λ₁*t+a₂])]
        # b(t) = [t; -a(t)[2]; -a(t)[3]]
        # b(t) = [t; (a₁+a₂*λ₂*t)./sum([a₁+a₂*λ₂*t; a₂]); a₂./sum([a₁+a₂*λ₂*t; a₂])]
    end
end


c = [0.5 0.5]*exp([-1 1; 0 -1])./sum([0.5 0.5]*exp([-1 1; 0 -1]))
dcdt = [0.5 0.5]*[-1 1; 0 -1] - [0.5 0.5]*[-1 1; 0 -1]*[1;1]*[0.5 0.5]
dcdt = -([0.5 0.5]*[-1 1; 0 -1] - [0.5 0.5]*[-1 1; 0 -1]*[1;1]*[0.5 0.5])
display(dcdt)


plot()
let
    λ = 1
    μ = 2
    γ = 3
    a₁ = 1
    a₂ = 0
    a₃ = 0
    # S = [-λ λ 0; 0 -μ μ; 0 0 -γ]
    S = [-2 1 1; 2 -5 1; 3 0 -4]
    t0 = 0
    avec = [a₁ a₂ a₃]*exp(S*t0)./sum([a₁ a₂ a₃]*exp(S*t0))
    d = -avec*S^-1
    Sr = (1.0./d)'.*S'.*d
    # Sr = [-λ 0 0; μ -μ 0; 0 γ -γ]
    # S = [-λ λ/2 λ/2; 0 -μ μ; γ 0 -γ]
    # Δ = diagm(eigen(S).vectors[:,end]./sum(eigen(S).vectors[:,end]))
    # Sr = Δ^-1*S'*Δ
    na(t) = [0 avec*exp(S*t)]
    a(t) = na(t)./sum(na(t))
    nb(t) = [0 avec*exp(Sr*t)]
    b(t) = nb(t)./sum(nb(t))
    # a(t) = [t; a₁./sum([a₁ ;a₁*λ₁*t+a₂]) ; (a₁*λ₁*t+a₂)./sum([a₁ ;a₁*λ₁*t+a₂])]
    # b(t) = [t; -a(t)[2]; -a(t)[3]]
    # b(t) = [t; (a₁+a₂*λ₂*t)./sum([a₁+a₂*λ₂*t; a₂]); a₂./sum([a₁+a₂*λ₂*t; a₂])]
    plot()#(xlims=(0,1),ylims=(0,1),zlims=(0,1))#,layout=(2,1))
    # plot(xlabel="a₁(t)",ylabel="a₂(t)") # xlabel="t",
    totaltime = 0
    for n in 1:6
        e₁ = -log(rand())/λ
        e₂ = -log(rand())/λ
        e₃ = -log(rand())/λ
        r = rand()
        erlangrnd = e₁*(r.<(a₁+a₂)) + e₂*(r.<a₁) + e₃
        h = erlangrnd/9
        for t in range(0,erlangrnd,length=10)[2:end]
            if n%2==1
                c = a(t)
                d = a(t-h)
                # display(plot!([c[2];d[2]],[c[3];d[3]],label=false,color=:blue,markershape=:x,seriestype=:line))
                display(plot3d!([d[2];c[2]],[d[3];c[3]],[d[4];c[4]],label=false,color=:blue,markershape=:rtriangle))
                # display(plot!(totaltime.+[c[1];d[1]],[c[2];d[2]],label=false,color=:blue,subplot=1))
                # display(plot!(totaltime.+[c[1];d[1]],[c[3];d[3]],label=false,color=:blue,subplot=2))
            else
                c = b(t)
                d = b(t-h)
                # display(plot!([c[2];d[2]],[c[3];d[3]],label=false,color=:red,markershape=:rtriangle,seriestype=:line))
                display(plot3d!([d[2];c[2]],[d[3];c[3]],[d[4];c[4]],label=false,color=:red,markershape=:ltriangle))
                # display(plot!(totaltime.+[c[1];d[1]],[c[2];d[2]],label=false,color=:red,subplot=1))
                # display(plot!(totaltime.+[c[1];d[1]],[c[3];d[3]],label=false,color=:red,subplot=2))
            end
            # display(c)
            dt = 0.01
            da = (a(t+dt)-a(t-dt))[2:end]./(2*dt)
            # da = da./sum(da)
            db = (b(t+dt)-b(t-dt))[2:end]./(2*dt)
            # db = db./sum(db)
            # display(da)
            # display(db)
            # display(da+db)
        end
        if n%2==1
            temp = a(erlangrnd)
            a₁ = temp[2]
            a₂ = temp[3]
            a₃ = temp[4]
        else
            temp = b(erlangrnd)
            a₁ = temp[2]
            a₂ = temp[3]
            a₃ = temp[4]
        end
        # s = a₁+a₂
        # a₁ = a₁/s
        # a₂ = a₂/s
        avec = [a₁ a₂ a₃]
        na(t) = [0 avec*exp(S*t)]
        a(t) = na(t)./sum(na(t))
        nb(t) = [0 avec*exp(Sr*t)]
        b(t) = nb(t)./sum(nb(t))
        # a(t) = [t; (a₁)./sum([a₁ ;a₁*λ₁*t+a₂]) ; (a₁*λ₁*t+a₂)./sum([a₁ ;a₁*λ₁*t+a₂])]
        # b(t) = [t; -a(t)[2]; -a(t)[3]]
        # b(t) = [t; (a₁+a₂*λ₂*t)./sum([a₁+a₂*λ₂*t; a₂]); a₂./sum([a₁+a₂*λ₂*t; a₂])]
    end
end

let
    α = [3 -1 -1]
    # C = [-1 1 0; 0 -2 2; 0 0 -3]
    # C = [-1 1 0; 0 -2 1; 1 1 -3]
    C = [-1 0 0; -2/3 -1 1; 2/3 -1 -1]
    c = -sum(C,dims=2)
    D = c*α
    Q = C + D
    E = eigen(Matrix(Q'))
    Eidx = abs.(E.values).<0.0001
    π = real.(E.vectors[:,Eidx])
    π = π./sum(π)
    Δ = diagm(0=>π[:])
    Δinv = diagm(0=>1 ./ π[:])

    Qr = Δinv*Q'*Δ
    Cr = Δinv*C'*Δ
    Dr = Δinv*D'*Δ
    cr = -sum(Cr,dims=2)

    idx = findfirst(abs.(cr).>sqrt(sqrt(eps())))[1]
    αr = Dr[idx,:]'./cr[idx]

    plot()

    # α = [1 0 0]
    na(t) = [0 α*exp(C*t)]
    a(t) = na(t)./sum(na(t))
    nb(t) = [0 α*exp(Cr*t)]
    b(t) = nb(t)./sum(nb(t))
    #plot(xlims=(-1.5,1.5),ylims=(-1.5,1.5))#,layout=(2,1))
    plot(xlabel="a₁(t)",ylabel="a₂(t)") # xlabel="t",
    for n in 1:14
        e₁ = log(rand())/C[1,1]
        e₂ = log(rand())/C[2,2]
        e₃ = log(rand())/C[3,3]
        r = rand()
        erlangrnd = e₁*(r.<(a(0)[1]+a(0)[2])) + e₂*(r.<a(0)[2]) + e₃
        h = erlangrnd/9
        for t in range(0,erlangrnd,length=10)[2:end]
            if true#n%2==1
                c = a(t)
                d = a(t-h)
                # display(plot!([c[2];d[2]],[c[3];d[3]],label=false,color=:blue,markershape=:x,seriestype=:line))
                display(plot!([d[3];c[3]],[d[4];c[4]],label=false,color=:blue,markershape=:rtriangle))
                # display(plot!(totaltime.+[c[1];d[1]],[c[2];d[2]],label=false,color=:blue,subplot=1))
                # display(plot!(totaltime.+[c[1];d[1]],[c[3];d[3]],label=false,color=:blue,subplot=2))
            else
                c = b(t)
                d = b(t-h)
                # display(plot!([c[2];d[2]],[c[3];d[3]],label=false,color=:red,markershape=:rtriangle,seriestype=:line))
                display(plot!([d[3];c[3]],[d[4];c[4]],label=false,color=:red,markershape=:ltriangle))
                # display(plot!(totaltime.+[c[1];d[1]],[c[2];d[2]],label=false,color=:red,subplot=1))
                # display(plot!(totaltime.+[c[1];d[1]],[c[3];d[3]],label=false,color=:red,subplot=2))
            end
        end
        if true#n%2==1
            temp = a(erlangrnd)
            a₁ = temp[2]
            a₂ = temp[3]
            a₃ = temp[4]
            α = [a₁ a₂ a₃]
        else
            temp = b(erlangrnd)
            a₁ = temp[2]
            a₂ = temp[3]
            a₃ = temp[4]
            α = [a₁ a₂ a₃]
        end
        na(t) = [0 α*exp(C*t)]
        a(t) = na(t)./sum(na(t))
        nb(t) = [0 α*exp(Cr*t)]
        b(t) = nb(t)./sum(nb(t))
    end
end
