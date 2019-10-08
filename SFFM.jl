module SFFM
  import Plots, LinearAlgebra, SymPy
  using Plots: Animation, frame, gif

  function CreateUniformMesh(;NNodes::Int=20, Interval::Array = [0 1])
  #CreateUniformMesh is a function that makes a uniformly spaced mesh over the an Interval
    K = NNodes-1; # the number of intervals
    Δ = (Interval[2]-Interval[1])/K; # interval width, same for all
    Mesh = zeros(K,2);
    Nodes = Interval[1]:Δ:Interval[2];
    Mesh[:,1] = Nodes[1:end-1]; # Left-hand end points of each interval
    Mesh[:,2] = Nodes[2:end]; # Right-hand end points of each interval
    return K, Δ, Mesh, Nodes
  end

  function CreateNonUniformMesh(;Nodes)
  #CreateUniformMesh is a function that makes a uniformly spaced mesh over the an Interval
    K = length(Nodes)-1; # the number of intervals
    Δ = (Nodes[2:end]-Nodes[1:end-1]); # interval width, same for all
    Mesh = zeros(K,2);
    Mesh[:,1] = Nodes[1:end-1]; # Left-hand end points of each interval
    Mesh[:,2] = Nodes[2:end]; # Right-hand end points of each interval
    return K, Δ, Mesh, Nodes
  end

  function CreateBlockDiagonalMatrix(;NBases::Array{Int},MeshWidth::Array=[1],Blocks::Dict)
  #CreateBlockDiagonalMatrix makes a matrix from diagonal block elements, given NBases
    NMesh = length(NBases);
    TotalNBases = sum(NBases);
    if !any(size(MeshWidth).==NMesh)
      MeshWidth = ones(NMesh).*MeshWidth[1];
    end
    BlockMatrix = zeros(TotalNBases,TotalNBases); # initialise an empty array to fill with blocks
    for i in 1:NMesh
      idx = sum(NBases[1:i-1])+1:sum(NBases[1:i]);
      BlockMatrix[idx,idx] = Blocks[string(NBases[i])]*MeshWidth[i];
    end
    return BlockMatrix
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
    return Plt
  end # end PlotVt

  function CreateFluxMatrix(; NBases, C, Δ=[1], IsBounded = true,
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
    if length(Δ) .== 1
        Δ = ones(length(NBases))
    end
    TotalNBases = sum(NBases)
    NPhases = length(C)
    F = zeros(Float64,TotalNBases,TotalNBases,NPhases)
    for i in 1:NPhases
      for k in 1:length(NBases)
        idx = sum(NBases[1:k-1])+1:sum(NBases[1:k])
        if C[i] > 0
          F[idx,idx,i] = Blocks[string(NBases[k],"PosDiagBlock")]
        elseif C[i] < 0
          F[idx,idx,i] = Blocks[string(NBases[k],"NegDiagBlock")]
        end # end if C[i]
        if k>1
          idxup = sum(NBases[1:k-2])+1:sum(NBases[1:k-1])
          if C[i] > 0
              F[idxup,idx,i] = Blocks[string(NBases[k-1],NBases[k],"UpDiagBlock")]*(Δ[k]/NBases[k])/(Δ[k-1]/NBases[k-1])
          elseif C[i] < 0
              F[idx,idxup,i] = Blocks[string(NBases[k-1],NBases[k],"LowDiagBlock")]*(Δ[k-1]/NBases[k-1])/(Δ[k]/NBases[k])
          end # end if C[i]
        end # end if k>2
      end # for k in ...
    end # end for i in NPhases
    if IsBounded
      for i in 1:NPhases
        if C[i] < 0
          idx = 1:NBases[1]
          F[idx,idx,i] .= 0
        elseif C[i] > 0
          idx = sum(NBases[1:end-1])+1:sum(NBases[1:end])
          F[idx,idx,i] .= 0
        end # end if C[i]
      end # end for i ...
    end # end IsBounded
    return F
  end

  function QApprox(;C,F,G,M)
    TotalNBases = size(F)[1]
    NPhases = length(C)
    Q = zeros(Float64,TotalNBases,TotalNBases,length(C))
    for i in 1:NPhases
      Q[:,:,i] = C[i]*(G+F[:,:,i])*inv(M); #PreMat[:,:,i] = C[i]*(F[:,:,i]+G)*inv(M);
    end
    return Q
  end

  function SimpleBApprox(;T,Q)
    NPhases = size(T)[1]
    TotalNBases = size(Q)[1]
    B = zeros(Float64,NPhases*TotalNBases,NPhases*TotalNBases)
    Id = Matrix{Float64}(LinearAlgebra.I,TotalNBases,TotalNBases);
    for i in 1:NPhases
      idx = (i-1)*TotalNBases+1:i*TotalNBases;
      B[idx,idx] = Q[:,:,i];
    end
    B = B+kron(T,Id);
    return B
  end

  function SFFMGIF(;a0,Nodes,B,Times,C,PointMass=true,YMAX=1, labels=[])
    gifplt = Plots.@gif for n in 1:length(Times)
      Vt = a0*exp(B*Times[n])
      Vt = reshape(Vt,(length(Nodes),length(C)))
      SFFM.PlotVt(Nodes = Nodes', Vt=Vt, C=C, PointMass = PointMass,YMAX=YMAX,labels=labels)
    end
    return gifplt
  end

  function BApprox(;B,C,Fil::Dict,TotalNBases,YS=["+","-","0"])
    NPhases = length(C)
    NXPhases = length(YS)
    BFull = Dict{String,Array}() #zeros(Float64,NPhases*NXPhases*TotalNBases,NPhases*NXPhases*TotalNBases)
    for ℓ in YS, m in YS
      BFull[string(ℓ,m)] = zeros(NPhases*TotalNBases, NPhases*TotalNBases)
    end
    for i in 1:NPhases, j in 1:NPhases
      # Now make the big B operator
      for ℓ in YS, m in YS
        i_idx = [falses((i-1)*TotalNBases);Fil[string(ℓ,i)];falses(NPhases*TotalNBases-i*TotalNBases)]; # (i-1)*TotalNBases+1:i*TotalNBases;
        j_idx = [falses((j-1)*TotalNBases);Fil[string(m,j)];falses(NPhases*TotalNBases-j*TotalNBases)] # (j-1)*TotalNBases+1:j*TotalNBases;
        BFull[string(ℓ,m,i,j)] = B[i_idx,j_idx]
        #BFull[string(ℓ,m)][i_idx,j_idx] = BFull[string(ℓ,m,i,j)]
      end
      ##
    end
    return BFull
  end

  function Assemble(;BDict,C,YS=["+","-","0"])
    BlockSize = size(BDict[string(YS[1],YS[1],1,1)])
    NPhases = length(C)
    NYPhases = length(YS)
    BSize = NPhases*NYPhases*BlockSize[1]
    B = zeros(Float64,BSize,BSize)
    for ℓ in 1:NYPhases, i in 1:NPhases, m in 1:NYPhases, j in 1:NPhases
      RowIdx = (2*ℓ+i-3)*BlockSize[1] .+ (1:BlockSize[1])
      ColIdx = (2*m+j-3)*BlockSize[1] .+ (1:BlockSize[1])
      #B[RowIdx,ColIdx] = BDict[string(YS[ℓ],YS[m],i,j)]
    end
    return B
  end

  function MakeB(;T,C,Nodes,NBases,Fil,YS=["+","-","0"])
    K, Δ, Mesh, Nodes = SFFM.CreateNonUniformMesh(Nodes=Nodes);
    TotalNBases = sum(NBases)
    Gblock = Dict{String,Array}("1"=>[0], "2"=>[-1/2 1/2 ; -1/2 1/2]);
    Mblock = Dict{String,Array}("1"=>[1], "2"=>[1/3 1/6; 1/6 1/3]);
    G = CreateBlockDiagonalMatrix(NBases=NBases,Blocks=Gblock)
    M = CreateBlockDiagonalMatrix(NBases=NBases,MeshWidth=[Δ],Blocks=Mblock)
    F = CreateFluxMatrix(NBases = NBases, C=C, IsBounded=true)
    Q = QApprox(C=C,F=F,G=G,M=M)
    BSmall = SimpleBApprox(T=T,Q=Q)
    BDict = BApprox(B=BSmall;C=C,Fil=Fil,TotalNBases=TotalNBases,YS=YS)
    B = Assemble(BDict=BDict,C=C,YS=YS)
    return B, BDict, BSmall
  end

  function Approximater(r,Bases,support)
    x = SymPy.Sym("x")
    RApprox = Array{Any}(undef,length(Bases))
    for i in 1:length(Bases)
      integrand(x) = r(x)*Bases[i](x)
      RApprox[i] = SymPy.integrate(integrand(x),(x,support[1],support[2]))
    end
    return RApprox
  end

  function MakeR(;rArray,NBases,Nodes,Fil,YS=["+","-","0"])
    K, Δ, Mesh, Node = CreateNonUniformMesh(Nodes=Nodes)
    TotalNBases = sum(NBases)
    RDict = Dict{String,Array{Float64,1}}()#zeros(Float64,TotalNBases,TotalNBases,length(rArray))
    x = SymPy.Sym("x")
    for i in 1:length(rArray)
      temp = Array{Float64,1}(undef,TotalNBases);
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
      for m in YS
        RDict[string(m,i)] = LinearAlgebra.diagm(Fil[string(m,i)])*temp
      end
    end# for i in ...
    TheDiagonal = zeros(TotalNBases*length(YS)*length(rArray))
    c = 0
    for m in YS
      RDict[string(m)] = zeros(Float64,length(rArray)*TotalNBases)
      for i in 1:length(rArray)
        c += 1
        TheDiagonal[(c-1)*TotalNBases+1:c*TotalNBases] = RDict[string(m,i)]
        RDict[string(m)][(i-1)*TotalNBases+1:i*TotalNBases] = RDict[string(m,i)]
      end
    end
    R = LinearAlgebra.diagm(TheDiagonal)
    return R, RDict
  end

  function MakeD(;RDict,BDict,YS=["+","-"])
    DDict = Dict{String,Any}()
    for ℓ in YS, m in YS
      idx = string(ℓ,m)
      Id = LinearAlgebra.I(size(RDict[ℓ])[1])
      DDict[idx] = function(;s)
                    LinearAlgebra.diagm(RDict[ℓ])*(BDict[idx]-s*Id +
                    BDict[string(ℓ,0)]*inv(s*Id-BDict["00"])*BDict[string(0,m)])
                  end
    end
    return DDict
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
    return Psi
  end

end # end module
