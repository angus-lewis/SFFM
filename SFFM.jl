module SFFM
  import Plots, LinearAlgebra, SymPy
  using Plots: Animation, frame, gif

  function MakeModel(;T::Array,C::Array,r,Signs::Array{String}=["+","-","0"],IsBounded::Bool=true)
    NPhases = length(C)
    NSigns = length(Signs)
    println("Model.Field with Fields (.T, .C, .r, .Signs, .IsBounded, .NPhases, .NSigns)")
    return (T=T,C=C,r=r,Signs=Signs,IsBounded=IsBounded,NPhases=NPhases,NSigns=NSigns)
  end

  # function UniformMesh(;NNodes::Int=20, Interval::Array = [0 1])
  # #CreateUniformMesh is a function that makes a uniformly spaced mesh over the an Interval
  #   K = NNodes-1; # the number of intervals
  #   Δ = (Interval[2]-Interval[1])/K; # interval width, same for all
  #   Mesh = zeros(K,2);
  #   Nodes = Interval[1]:Δ:Interval[2];
  #   Mesh[:,1] = Nodes[1:end-1]; # Left-hand end points of each interval
  #   Mesh[:,2] = Nodes[2:end]; # Right-hand end points of each interval
  #   return (K=K, Δ=Δ, MeshArray=Mesh, Nodes=Nodes)
  # end

  # function NonUniformMesh(;Nodes)
  # #CreateUniformMesh is a function that makes a uniformly spaced mesh over the an Interval
  #   K = length(Nodes)-1; # the number of intervals
  #   Δ = (Nodes[2:end]-Nodes[1:end-1]); # interval width, same for all
  #   Mesh = zeros(K,2);
  #   Mesh[:,1] = Nodes[1:end-1]; # Left-hand end points of each interval
  #   Mesh[:,2] = Nodes[2:end]; # Right-hand end points of each interval
  #   return (NIntervals=K, Δ=Δ, MeshArray=Mesh, Nodes=Nodes)
  # end

  function MakeMesh(;Model,Nodes,NBases::Array{Int},Signs=["+","-","0"])
    NIntervals = length(Nodes)-1; # the number of intervals
    display(Nodes[1:end])
    Δ = (Nodes[2:end]-Nodes[1:end-1]); # interval width, same for all
    MeshArray = zeros(NIntervals,2);
    MeshArray[:,1] = Nodes[1:end-1]; # Left-hand end points of each interval
    MeshArray[:,2] = Nodes[2:end];
    TotalNBases = sum(NBases)
    Fil = Dict{String,Array{Bool}}()
    for i in 1:length(Model.C), ℓ in Model.Signs
      Fil[string(ℓ,i)] = falses(TotalNBases)
    end
    Bases = Array{Any}(undef,NIntervals)
    CumBases = [0;cumsum(NBases)]
    for row in 1:size(MeshArray)[1]
      if NBases[row] == 1
        Bases[row] = x -> [1]
      else
        Bases[row] = x -> [(x-MeshArray[row,1])/Δ[row] (MeshArray[row,2]-x)/Δ[row]]
      end
      NEvals = 10
      evalpoints = range(MeshArray[row,1]+1*Δ[row]/NEvals,MeshArray[row,2]-(NEvals-1)*Δ[row]/NEvals,length=NEvals)
      evals = Model.r.(evalpoints)
      idx = CumBases[row]+1:CumBases[row+1]
      for i in 1:Model.NPhases
        testvals = zeros(NEvals)
        for j in 1:NEvals
          testvals[j] = evals[j][i]
        end
        if all(testvals.>0)
          Fil[string("+",i)][idx] = trues(NBases[row])
        elseif all(testvals.<0)
          Fil[string("-",i)][idx] = trues(NBases[row])
        elseif all(testvals.==0)
          Fil[string("0",i)][idx] = trues(NBases[row])
        end
      end
    end
    for ℓ in Model.Signs
      Fil[string(ℓ)] = falses(TotalNBases*Model.NPhases)
      for i in 1:Model.NPhases
        i_idx = [falses((i-1)*TotalNBases);Fil[string(ℓ,i)];falses(Model.NPhases*TotalNBases-i*TotalNBases)] #
        Fil[string(ℓ)][i_idx] .= true
      end
    end
    println("Mesh.Field with Fields (.Bases, .NBases, .Fil, .Δ, .NIntervals, .MeshArray, .Nodes .TotalNBases)")
    return (Bases=Bases, NBases=NBases, Fil=Fil, Δ=Δ, NIntervals=NIntervals,
            MeshArray=MeshArray, Nodes=Nodes, TotalNBases=TotalNBases)
  end

  function CreateBlockDiagonalMatrix(;Mesh,Blocks::Dict,Factors)
  #CreateBlockDiagonalMatrix makes a matrix from diagonal block elements, given NBases
    BlockMatrix = zeros(Mesh.TotalNBases,Mesh.TotalNBases); # initialise an empty array to fill with blocks
    for i in 1:Mesh.NIntervals
      idx = sum(Mesh.NBases[1:i-1])+1:sum(Mesh.NBases[1:i]);
      BlockMatrix[idx,idx] = Blocks[string(Mesh.NBases[i])]*Factors[i];
    end
    return (BlockMatrix=BlockMatrix)
  end

  function PlotVt(; Nodes, Vt, C, YMAX=1, PointMass = true, labels=[])
    if isempty(labels)
      labels = string.(C)
    end
    Interval = [Nodes[1]; Nodes[end]]
    NNodes = length(Nodes)
    ΔEnds = [Nodes[2]-Nodes[1]; Nodes[end]-Nodes[end-1]]
    FivePercent = (Interval[2]-Interval[1])*0.025.*[-1;1]
    Plt=Plots.plot(xlims = Interval+FivePercent, ylims = (0,YMAX))
    for i in 1:length(C)
      if PointMass
        if C[i] > 0
          CtsIdx = [1;1:NNodes-1];
          xIdx = [1;2;2:NNodes-1];
          PMIdx = NNodes;
          ΔIdx = 2;
        elseif C[i] < 0
          CtsIdx = [2:NNodes;NNodes];
          xIdx = [2:NNodes-1;NNodes-1;NNodes];
          PMIdx = 1;
          ΔIdx = 1;
        else
          CtsIdx = 2:NNodes-1;
          xIdx = CtsIdx;
          PMIdx = [1;NNodes];
          ΔIdx = [1;2];
        end # end if C[i]
        Plt = Plots.plot!(Nodes[xIdx],Vt[CtsIdx,i],label=string("Phase ",i," rate: ",C[i]," and ", labels[i]),color=i);
        Plt = Plots.scatter!([Nodes[PMIdx]],[Vt[PMIdx,i].*ΔEnds[ΔIdx]],marker=:hexagon,label=string("mass"),color=i)
      else
        Plt = Plots.plot!(Nodes,Vt[:,i],label=string("Phase ",i," rate: ",labels[i]),color=i);
      end # end if PointMass
    end # end for i
    return (Plt=Plt)
  end # end PlotVt

  function CreateFluxMatrix(; Mesh, Model,
    Blocks::Dict=Dict{String,Array}("2PosDiagBlock" => [0 0;0 -1],
                      "2NegDiagBlock" => [1 0;0 0],
                      "22UpDiagBlock" => [0 0;1 0],
                      "22LowDiagBlock" => [0 -1;0 0],
                      "1PosDiagBlock" => [-1],
                      "1NegDiagBlock" => [1],
                      "11UpDiagBlock" => [1],
                      "11LowDiagBlock" => [-1],
                      "21UpDiagBlock" => [0; 1],
                      "12UpDiagBlock" => [1 0],
                      "21LowDiagBlock" => [0 -1],
                      "12LowDiagBlock" => [-1; 0]))
    F = zeros(Float64,Mesh.TotalNBases,Mesh.TotalNBases,Model.NPhases)
    CumBases = [0;cumsum(Mesh.NBases)];
    for i in 1:Model.NPhases
      for k in 1:length(Mesh.NBases)
        idx = CumBases[k]+1:CumBases[k+1]
        if Model.C[i] > 0
          F[idx,idx,i] = Blocks[string(Mesh.NBases[k],"PosDiagBlock")]
        elseif Model.C[i] < 0
          F[idx,idx,i] = Blocks[string(Mesh.NBases[k],"NegDiagBlock")]
        end # end if C[i]
        if k>1
          idxup = CumBases[k-1]+1:CumBases[k]
          if Model.C[i] > 0
              F[idxup,idx,i] = Blocks[string(Mesh.NBases[k-1],Mesh.NBases[k],"UpDiagBlock")]*(Mesh.Δ[k]/Mesh.NBases[k])/(Mesh.Δ[k-1]/Mesh.NBases[k-1])
          elseif Model.C[i] < 0
              F[idx,idxup,i] = Blocks[string(Mesh.NBases[k-1],Mesh.NBases[k],"LowDiagBlock")]*(Mesh.Δ[k-1]/Mesh.NBases[k-1])/(Mesh.Δ[k]/Mesh.NBases[k])
          end # end if C[i]
        end # end if k>2
      end # for k in ...
    end # end for i in NPhases
    if Model.IsBounded
      for i in 1:Model.NPhases
        if Model.C[i] < 0
          idx = 1:Mesh.NBases[1]
          F[idx,idx,i] .= 0
        elseif Model.C[i] > 0
          idx = CumBases[end-1]+1:CumBases[end]
          F[idx,idx,i] .= 0
        end # end if C[i]
      end # end for i ...
    end # end IsBounded
    return (F=F)
  end

  function MakeMatrices(;Model,Mesh)
    Gblock = Dict{String,Array}("1"=>[0], "2"=>[-1/2 1/2 ; -1/2 1/2]);
    Mblock = Dict{String,Array}("1"=>[1], "2"=>[1/3 1/6; 1/6 1/3]);
    MInvblock = Dict{String,Array}("1"=>[1], "2"=>[1/3 1/6; 1/6 1/3]\LinearAlgebra.I(2));
    G = SFFM.CreateBlockDiagonalMatrix(Mesh=Mesh,Blocks=Gblock,Factors=ones(Mesh.NIntervals))
    M = SFFM.CreateBlockDiagonalMatrix(Mesh=Mesh,Blocks=Mblock,Factors=Mesh.Δ)
    MInv = SFFM.CreateBlockDiagonalMatrix(Mesh=Mesh,Blocks=MInvblock,Factors=1 ./Mesh.Δ)
    F = SFFM.CreateFluxMatrix(Mesh=Mesh,Model=Model)
    Q = zeros(Float64,Mesh.TotalNBases,Mesh.TotalNBases,length(Model.C))
    for i in 1:Model.NPhases
      Q[:,:,i] = Model.C[i]*(G+F[:,:,i])*MInv;
    end
    println("Matrices.Fields with Fields (.G, .M, .MInv, F, .Q)")
    return (G=G,M=M,MInv=MInv,F=F,Q=Q)
  end

  function ApproximateB(;Model,Mesh,Matrices)
    # Make the approximation B
    B = zeros(Float64,Model.NPhases*Mesh.TotalNBases,Model.NPhases*Mesh.TotalNBases)
    Id = LinearAlgebra.I(Mesh.TotalNBases);
    for i in 1:Model.NPhases
      idx = (i-1)*Mesh.TotalNBases+1:i*Mesh.TotalNBases;
      B[idx,idx] = Matrices.Q[:,:,i];
    end
    B = B+LinearAlgebra.kron(Model.T,Id);
    # Make a Dictionary so that the blocks of B are easy to access
    BDict = Dict{String,Array}()
    for ℓ in Model.Signs, m in Model.Signs
      for i in 1:Model.NPhases, j in 1:Model.NPhases
        i_idx = [falses((i-1)*Mesh.TotalNBases);Mesh.Fil[string(ℓ,i)];falses(Model.NPhases*Mesh.TotalNBases-i*Mesh.TotalNBases)] #
        j_idx = [falses((j-1)*Mesh.TotalNBases);Mesh.Fil[string(m,j)];falses(Model.NPhases*Mesh.TotalNBases-j*Mesh.TotalNBases)] #
        BDict[string(ℓ,m,i,j)] = B[i_idx,j_idx]
      end
      BDict[string(ℓ,m)] = B[Mesh.Fil[ℓ],Mesh.Fil[m]]
    end
    CumBases = [0;cumsum(Mesh.NBases)]
    c = 0
    QBDidx = zeros(Int,Model.NPhases*Mesh.TotalNBases)
    for k in 1:Mesh.NIntervals, i in 1:Model.NPhases, n in 1:Mesh.NBases[k]
      c += 1
      QBDidx[c] = (i-1)*Mesh.TotalNBases+CumBases[k]+n
    end
    println("B.Fields with Fields (.BDict, .B, .QBDidx)")
    return (BDict=BDict, B=B, QBDidx=QBDidx)
  end

  function SFFMGIF(;a0,Nodes,B,Times,C,PointMass=true,YMAX=1, labels=[])
    gifplt = Plots.@gif for n in 1:length(Times)
      Vt = a0*exp(B*Times[n])
      Vt = reshape(Vt,(length(Nodes),length(C)))
      SFFM.PlotVt(Nodes = Nodes', Vt=Vt, C=C, PointMass = PointMass,YMAX=YMAX,labels=labels)
    end
    return (gitplt=gifplt)
  end

  # function MakeBDict(;Model,Mesh,B)
  #   BDict = Dict{String,Array}()
  #   for ℓ in Model.Signs, m in Model.Signs
  #     # Now make the big B operator
  #     Fl = falses(Mesh.TotalNBases*Model.NPhases)
  #     Fm = falses(Mesh.TotalNBases*Model.NPhases)
  #     for i in 1:Model.NPhases, j in 1:Model.NPhases
  #       i_idx = [falses((i-1)*Mesh.TotalNBases);Mesh.Fil[string(ℓ,i)];falses(Model.NPhases*Mesh.TotalNBases-i*Mesh.TotalNBases)] #
  #       j_idx = [falses((j-1)*Mesh.TotalNBases);Mesh.Fil[string(m,j)];falses(Model.NPhases*Mesh.TotalNBases-j*Mesh.TotalNBases)] #
  #       Fl[i_idx] .= true
  #       Fm[j_idx] .= true
  #       BDict[string(ℓ,m,i,j)] = B[i_idx,j_idx]
  #     end
  #     BDict[string(ℓ,m)] = B[Fl,Fm]
  #   end
  #   return (BDict=BDict)
  # end

  # function Assemble(;BDict,Model)
  #   BlockSize = size(BDict[string(Model.Signs[1],Model.Signs[1],1,1)])
  #   BSize = Model.NPhases*Model.NSigns*BlockSize[1]
  #   B = zeros(Float64,BSize,BSize)
  #   for ℓ in 1:Model.NSigns, i in 1:Model.NPhases, m in 1:Model.NSigns, j in 1:Model.NPhases
  #     RowIdx = (2*ℓ+i-3)*BlockSize[1] .+ (1:BlockSize[1])
  #     ColIdx = (2*m+j-3)*BlockSize[1] .+ (1:BlockSize[1])
  #     #B[RowIdx,ColIdx] = BDict[string(Signs[ℓ],Signs[m],i,j)]
  #   end
  #   return (B=B)
  # end

  # function MakeB(;Model,Mesh)
  #   Matrices = SFFM.MakeMatrices(Model=Model,Mesh=Mesh)
  #   Q = QApprox(Model=Model,Mesh=Mesh,Matrices=Matrices)
  #   B = BApprox(Model=Model,Mesh=Mesh,Q=Q)
  #   BDict = MakeBDict(Model=Model,Mesh=Mesh,B=B)
  #   return (B=B, BDict=BDict, Q=Q)
  # end

  function ApproximateR(;Model,Mesh)
    x = SymPy.Sym("x")
    CumBases = [0;cumsum(Mesh.NBases)]
    RDict = Dict{String,Array}()
    for i in 1:Model.NPhases
      RDict[string(i)] = zeros(Mesh.TotalNBases)
    end
    for i in 1:Model.NPhases, k in 1:Mesh.NIntervals, n in 1:Mesh.NBases[k]
      integrand(x) = Model.r(x)[i]*Mesh.Bases[k](x)[n]
      RDict[string(i)][CumBases[k]+n] = (1)./abs(SymPy.integrate(integrand(x),(x,Mesh.MeshArray[k,1],Mesh.MeshArray[k,2])))
    end
    R=zeros(Float64,Model.NPhases*Mesh.TotalNBases)
    for i in 1:Model.NPhases
      Ri = RDict[string(i)]
      R[(i-1)*Mesh.TotalNBases+1:i*Mesh.TotalNBases] = Ri
      for ℓ in Model.Signs
        RDict[string(ℓ,i)] = Ri[Mesh.Fil[string(ℓ,i)]]
      end
    end
    for ℓ in Model.Signs
      RDict[ℓ] = R[Mesh.Fil[ℓ]]
    end
    println("R.Fields with Fields (.RDict, .R)")
    return (RDict=RDict, R=R)
  end

  function ApproximateD(;RDict,BDict,Model,Mesh)
    DDict = Dict{String,Any}()
    for ℓ in Model.Signs, m in Model.Signs
      Id = LinearAlgebra.I(sum(Mesh.Fil[ℓ]))
      if in("0",Model.Signs)
        DDict[ℓ*m] = function(;s)
                        RDict[ℓ].*(BDict[ℓ*m]-s*Id +
                        BDict[string(ℓ,0)]*inv(s*Id-BDict["00"])*BDict[string(0,m)])
                      end
      else
        DDict[ℓ*m] = function(;s)
                        RDict[ℓ].*(BDict[ℓ*m]-s*Id)
                      end
      end
    end
    return (DDict=DDict)
  end

  function PsiFun(;s,DDict,MaxIters=100,err=1e-8)
    EvalD1 = DDict["+-"](s=s)
    Dimensions = size(EvalD1)
    EvalDDict = Dict{String,Array{Float64}}("+-" => EvalD1)
    for ℓ in ["++","--","-+"]
      EvalDDict[ℓ] = DDict[ℓ](s=s)
    end
    Psi = zeros(Float64,Dimensions)
    A = EvalDDict["++"]
    B = EvalDDict["--"]
    C = EvalDDict["-+"]
    OldPsi = Psi
    for n = 1:MaxIters
      display(n)
      display(LinearAlgebra.eigvals(A))
      display(LinearAlgebra.eigvals(B))
      display(LinearAlgebra.eigvals(C))
      Psi = LinearAlgebra.sylvester(A,B,C)
      if abs.(OldPsi - Psi) .< err
        break
      end
      A = EvalDDict["++"] + Psi*EvalDDict["-+"]
      B = EvalDDict["--"] + EvalDDict["-+"]*Psi
      C = EvalDDict["-+"] - Psi*EvalDDict["-+"]*Psi
    end
    return (Psi=Psi)
  end

end # end module
