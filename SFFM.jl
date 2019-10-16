module SFFM
  import Plots, LinearAlgebra, SymPy
  using Plots: Animation, frame, gif

  function MakeModel(;T::Array,C::Array,r,Signs::Array{String}=["+","-","0"],IsBounded::Bool=true)
    NPhases = length(C)
    NSigns = length(Signs)
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

  function NonUniformMesh(;Nodes)
  #CreateUniformMesh is a function that makes a uniformly spaced mesh over the an Interval
    K = length(Nodes)-1; # the number of intervals
    Δ = (Nodes[2:end]-Nodes[1:end-1]); # interval width, same for all
    Mesh = zeros(K,2);
    Mesh[:,1] = Nodes[1:end-1]; # Left-hand end points of each interval
    Mesh[:,2] = Nodes[2:end]; # Right-hand end points of each interval
    return (NIntervals=K, Δ=Δ, MeshArray=Mesh, Nodes=Nodes)
  end

  function MakeMesh(;Model,Nodes,NBases::Array{Int},Signs=["+","-","0"])
    NIntervals = length(Nodes)-1; # the number of intervals
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
        Bases[row] = x -> [(x-MeshArray[row,1])/MeshArray.Δ[row] (MeshArray[row,2]-x)/Δ[row]]
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
    Fl = Dict{String,Array}()
    for ℓ in Signs
      Fl[ℓ] = falses(TotalNBases*Model.NPhases)
      for i in 1:Model.NPhases
        Fl[ℓ] = falses(TotalNBases*Model.NPhases)
      end
    end
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
    return (G=G,M=M,MInv=MInv,F=F)
  end

  function QApprox(;Model,Mesh,Matrices)
    Q = zeros(Float64,Mesh.TotalNBases,Mesh.TotalNBases,length(Model.C))
    for i in 1:Model.NPhases
      Q[:,:,i] = Model.C[i]*(Matrices.G+Matrices.F[:,:,i])*Matrices.MInv;
    end
    return (Q=Q)
  end

  function BApprox(;Model,Mesh,Q)
    B = zeros(Float64,Model.NPhases*Mesh.TotalNBases,Model.NPhases*Mesh.TotalNBases)
    Id = Matrix{Float64}(LinearAlgebra.I,Mesh.TotalNBases,Mesh.TotalNBases);
    for i in 1:Model.NPhases
      idx = (i-1)*Mesh.TotalNBases+1:i*Mesh.TotalNBases;
      B[idx,idx] = Q[:,:,i];
    end
    B = B+kron(Model.T,Id);
    return (B=B)
  end

  function SFFMGIF(;a0,Nodes,B,Times,C,PointMass=true,YMAX=1, labels=[])
    gifplt = Plots.@gif for n in 1:length(Times)
      Vt = a0*exp(B*Times[n])
      Vt = reshape(Vt,(length(Nodes),length(C)))
      SFFM.PlotVt(Nodes = Nodes', Vt=Vt, C=C, PointMass = PointMass,YMAX=YMAX,labels=labels)
    end
    return (gitplt=gifplt)
  end

  function MakeBDict(;Model,Mesh,B)
    BDict = Dict{String,Array}()
    for ℓ in Model.Signs, m in Model.Signs
      # Now make the big B operator
      Fl = falses(Mesh.TotalNBases*Model.NPhases)
      Fm = falses(Mesh.TotalNBases*Model.NPhases)
      for i in 1:Model.NPhases, j in 1:Model.NPhases
        i_idx = [falses((i-1)*Mesh.TotalNBases);Mesh.Fil[string(ℓ,i)];falses(Model.NPhases*Mesh.TotalNBases-i*Mesh.TotalNBases)] #
        j_idx = [falses((j-1)*Mesh.TotalNBases);Mesh.Fil[string(m,j)];falses(Model.NPhases*Mesh.TotalNBases-j*Mesh.TotalNBases)] #
        Fl[i_idx] .= true
        Fm[j_idx] .= true
        BDict[string(ℓ,m,i,j)] = B[i_idx,j_idx]
      end
      BDict[string(ℓ,m)] = B[Fl,Fm]
    end
    return (BDict=BDict)
  end

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

  function MakeB(;Model,Mesh)
    Matrices = SFFM.MakeMatrices(Model=Model,Mesh=Mesh)
    Q = QApprox(Model=Model,Mesh=Mesh,Matrices=Matrices)
    B = BApprox(Model=Model,Mesh=Mesh,Q=Q)
    BDict = MakeBDict(Model=Model,Mesh=Mesh,B=B)
    return (B=B, BDict=BDict, Q=Q)
  end

  function Approximater(r,Bases,support)
    x = SymPy.Sym("x")
    RApprox = Array{Any}(undef,length(Bases))
    for i in 1:length(Bases)
      integrand(x) = r(x)*Bases[i](x)
      RApprox[i] = SymPy.integrate(integrand(x),(x,support[1],support[2]))
    end
    return (RApprox=RApprox)
  end

  function MakeR(;rArray,NBases,Nodes,Fil,Signs=["+","-","0"])
    K, Δ, Mesh, Node = CreateNonUniformMesh(Nodes=Nodes)
    TotalNBases = sum(NBases)
    RDict = Dict{String,Array{Float64,1}}()
    x = SymPy.Sym("x")
    for i in 1:Model.NPhases
      temp = Array{Float64,1}(undef,Mesh.TotalNBases);
      for k in 1:length(NBases)
        if NBases[k] == 1
          Bases = [x-x+1];
        elseif NBases[k] == 2
          Bases = [(x-Mesh[k,1])/(Mesh[k,2]-Mesh[k,1]) (Mesh[k,2]-x)/(Mesh[k,2]-Mesh[k,1])];
        end # end if NBases ...
        support = Mesh[k,:]
        temp[sum(NBases[1:k-1])+1:sum(NBases[1:k])] = abs.(Approximater(rArray[i],Bases,support));
      end # end for k in ...
      temp[temp.>0] .= 1.0 ./ temp[temp.>0]
      for m in Signs
        RDict[string(m,i)] = LinearAlgebra.diagm(Fil[string(m,i)])*temp
      end
    end# for i in ...
    TheDiagonal = zeros(Mesh.TotalNBases*length(Signs)*length(rArray))
    c = 0
    for m in Signs
      RDict[string(m)] = zeros(Float64,length(rArray)*Mesh.TotalNBases)
      for i in 1:length(rArray)
        c += 1
        TheDiagonal[(c-1)*Mesh.TotalNBases+1:c*Mesh.TotalNBases] = RDict[string(m,i)]
        RDict[string(m)][(i-1)*Mesh.TotalNBases+1:i*Mesh.TotalNBases] = RDict[string(m,i)]
      end
    end
    R = LinearAlgebra.diagm(TheDiagonal)
    return (R=R, RDict=RDict)
  end

  function MakeD(;RDict,BDict,Signs=["+","-"])
    DDict = Dict{String,Any}()
    for ℓ in Signs, m in Signs
      idx = string(ℓ,m)
      Id = LinearAlgebra.I(size(RDict[ℓ])[1])
      DDict[idx] = function(;s)
                    LinearAlgebra.diagm(RDict[ℓ])*(BDict[idx]-s*Id +
                    BDict[string(ℓ,0)]*inv(s*Id-BDict["00"])*BDict[string(0,m)])
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
