### A Pluto.jl notebook ###
# v0.19.46

#> [frontmatter]
#> chapter = "1"
#> title = "Singular Value Decomposition"
#> tags = ["module1"]
#> layout = "layout.jlhtml"

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 74418960-a9e8-47ab-bf1b-31cdc2e0acb8
using PlutoPlotly, LinearAlgebra, PlutoUI, PlutoTeachingTools

# ╔═╡ 68546785-c79a-4e9b-a717-74120ead9b88
TableOfContents()

# ╔═╡ d2db3192-b5c9-11ee-0d3c-89a9f069d24c
md"""# Singular Value Decomposition (SVD)
```math
G = U\Sigma V^T
```
- SVD produces two sets of singular vectors (columns of matrices `U` and `V`). 
- For a matrix `G` of size $(m,n)$, these singular vectors form orthogonal axes in $\mathbf{R}^m$ and $\mathbf{R}^n$, respectively.
- In this demo, we consider a square matrix `G`: here, the matrix maps each of the columns in `V` to the corresponding columns in `U`.
- Notice that the orthogonal axes of `U` and `V` are similar (except for the sign) in the case when `G` is symmetric.
- If `G` is singular, we can rotate m to be perpendicular to the columns of G to see zero vector d; yes, the null space is orthogonal to the column space of a matrix.
"""

# ╔═╡ 5d11c5f1-3ecd-4264-902f-fa8fbb2d3c6b
ThreeColumn(md"""
$(@bind Atype MultiCheckBox(["Symmetric"=>"G is Symmetric", "Singular"=> "G is Singular"]))""",md"""$(@bind sampleG CounterButton("Resample G"))""", 
	md""" and  Rotate m$(@bind mphi Slider(range(0,2pi,length=1000)))""")

# ╔═╡ 63e61c32-c235-438b-978a-ae1a09fef886
md"Scale the columns of U and V with singular values? $(@bind scaleUV CheckBox())"

# ╔═╡ c712c1ee-ac74-4e3e-b9ed-f73ee505fdf6
md"## Orthogonal Matrices = Rotation in 2D = Invertible"

# ╔═╡ 7d6a1fad-6fb4-4b25-9c8a-9a9f581edfde
md"""
## Symmetric Matrices GᵀG and GGᵀ
`V` contains orthonormal eigenvectors of GᵀG, and `U` contains orthonormal eigenvectors of GGᵀ, where both of them share eigenvalues.
"""

# ╔═╡ 634364b6-05ca-4a54-a0cd-7d0581eaab55
begin
	sampleG; 
	G = randn(2,2)
	if("Symmetric" ∈ Atype)
		G=0.5*(G+G')
	end
	if("Singular" ∈ Atype)
		G[:,2] .= randn() .* G[:,1] 
	end
end

# ╔═╡ f98dab1c-16f1-418c-acda-3fe0b7554c9c
F=svd(G)

# ╔═╡ 6a2e234d-be6c-44a1-8fdd-9e5c6ad8ba01
begin
	m=[cos(mphi), sin(mphi)]
end

# ╔═╡ 75b51cd7-5d3d-4f0c-8750-d169edf0a3d1
n = F.Vt * m

# ╔═╡ cea5f070-70b4-4870-8a41-b0cb4a6ea26b
o = Diagonal(F.S) * n

# ╔═╡ 98338255-a250-4b22-8bc3-b3f11b22583b
m1=F.U*o

# ╔═╡ d94fc2d9-8881-4654-98d3-7fbb78bc1271
md"## Appendix"

# ╔═╡ 1fee9d48-46be-443a-9793-0b84e6de35a1
md"### Plots"

# ╔═╡ 4f9c48af-0714-40f8-a4e6-3228ba0b1393
function quiverplot(p1, p2=zeros(size(p1)); colors=fill("black", size(p1,2)), xylim=3.0, title="", names=string.(collect(1:size(p1,2))))
	vector_scale = 1 #scale factor in (0, 1] for the vector directions to avoid quiver overlapping
    arrow_scale = 0.25
    angle = π/9
    scaleratio = 1.0 #aspect ratio for the 2d plot
    d =1.0 #a scale factor in (0.9, 1]  for the already scaled direction; 
	p1=cat(p1,dims=2)
	p2=cat(p2,dims=2)
	@assert size(p1,2) == size(p2,2)
	plots = mapreduce(vcat, 1:size(p1,2)) do i
    	x = p2[1:1, i]
   	 	y = p2[2:2, i]
   	 	u = p1[1:1, i]
    	v = p1[2:2, i]
    (length(x) == length(y) == length(u) == length(v)) &&
                  vector_scale > 0 && arrow_scale > 0 ||
                  error("the vects x, y, u, v do not have the same length")
    u = vector_scale * scaleratio *u
    v = vector_scale * v
    end_x = x .+ u
    end_y = y .+ v
	function tuple_interleave(tu::Union{NTuple{3, Vector}, NTuple{4, Vector}}) 
   	 #auxilliary function to interleave elements of a NTuple of vectors, N=3 or 4
   	 zipped_data = collect(zip(tu...))
  	  vv_zdata = [collect(elem) for elem in zipped_data]
  	  return reduce(vcat, vv_zdata)
	end
	
    vect_nans = repeat([NaN],  length(x))
    barb_x = tuple_interleave((x, x .+ d*u, vect_nans))
    barb_y = tuple_interleave((y, y .+ d*v, vect_nans))

    barb_length = sqrt.((u/scaleratio) .^2 .+ v .^2)
    arrow_length = arrow_scale #* barb_length
    barb_angle = atan.(v, u/scaleratio)

    ang1 = barb_angle .+ angle
    ang2 = barb_angle .- angle

    seg1_x = arrow_length .* cos.(ang1)
    seg1_y = arrow_length .* sin.(ang1)

    seg2_x = arrow_length .* cos.(ang2)
    seg2_y = arrow_length .* sin.(ang2)

    arrowend1_x = end_x .- seg1_x *scaleratio
    arrowend1_y = end_y .- seg1_y
    arrowend2_x = end_x .- seg2_x *scaleratio
    arrowend2_y = end_y .- seg2_y
    arrow_x =  tuple_interleave((arrowend1_x, end_x, arrowend2_x, vect_nans))
    arrow_y =  tuple_interleave((arrowend1_y, end_y, arrowend2_y, vect_nans))

    barb = scatter(x=barb_x, y=barb_y, mode="lines", line_color=colors[i], name=names[i],showlegend=!(names[i]==""))
    arrow = scatter(x=arrow_x, y=arrow_y, mode="lines", line_color=colors[i],
                     fill="toself", fillcolor=colors[i], hoverinfo="skip", showlegend=false)
	return [barb, arrow]
	end
    layout = Layout(title=title,width=250, height=260, yaxis=attr(range=[-xylim, xylim]),xaxis=attr(range=[-xylim, xylim]), legend=attr(
        x=0.75,
        y=1,
        yanchor="bottom",
        xanchor="right",
        orientation="h"
    ),
                    showlegend=true)
    return plot(Plot(plots, layout, config=PlotConfig(staticPlot=true)))
end

# ╔═╡ 3539521e-d128-4c66-84ed-a85838ea1684
PlutoUI.ExperimentalLayout.hbox([quiverplot(G, title="Columns of G"), quiverplot(m, title="m", colors=["blue"], names=[""])])

# ╔═╡ 0f7e56b7-e71c-4e04-af8c-05dfb424860b
let
	I = scaleUV ? Diagonal(F.S) : Diagonal(ones(2))
	PlutoUI.ExperimentalLayout.hbox([quiverplot(F.U * I, title="Columns of U"), quiverplot(F.V * I, title="Columns of V")])
end

# ╔═╡ af442ee2-d0fc-4754-a1fb-917650237f4b
PlutoUI.ExperimentalLayout.hbox([quiverplot(hcat(m,o,m1), colors=["blue","magenta", "red"], names=["m","o","Gm"]), quiverplot(hcat(n,o), colors=["green", "magenta"], names=["n","o=Σn"]), quiverplot(hcat(m,n), colors=["blue", "green"], names=["m","n=Vᵀm"]), ])

# ╔═╡ 7a04a1a7-8a73-49a0-bb27-4731f8428ef1
quiverplot(G', colors=["blue", "green"], title="Row f G")

# ╔═╡ 82c2ce66-0d86-414d-868a-5d3aff5168dc
quiverplot(hcat(m,F.Vt*m), colors=["blue", "green"], names=["m","Vᵀm"])

# ╔═╡ f1311c00-8a40-4447-9416-8d0f1c512e33
quiverplot(hcat(F.Vt*m, F.Vt*F.V*m), colors=["blue", "green"], names=["m","VᵀVm"])

# ╔═╡ 126aaff1-428e-45e2-a353-f1afc437db7f
md"""### Resources
* [Wolfram Alpha Demo](https://demonstrations.wolfram.com/SingularValueDecomposition/)
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
PlutoPlotly = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
PlutoTeachingTools = "661c6b06-c737-4d37-b85c-46df65de6f69"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"

[compat]
PlutoPlotly = "~0.5.0"
PlutoTeachingTools = "~0.2.15"
PlutoUI = "~0.7.59"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.10.4"
manifest_format = "2.0"
project_hash = "81e0ff7e7ab3a7da04a4253674859ebf49f1a680"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BaseDirs]]
git-tree-sha1 = "cb25e4b105cc927052c2314f8291854ea59bf70a"
uuid = "18cc8868-cbac-4acf-b575-c8ff214dc66f"
version = "1.2.4"

[[deps.CodeTracking]]
deps = ["InteractiveUtils", "UUIDs"]
git-tree-sha1 = "7eee164f122511d3e4e1ebadb7956939ea7e1c77"
uuid = "da1fd8a2-8d9e-5ec2-8556-3022fb5608a2"
version = "1.3.6"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "b5278586822443594ff615963b0c09755771b3e0"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.26.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "b10d0b65641d57b8b4d5e234446582de5047050d"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.5"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "a1f44953f2382ebb937d60dafbe2deea4bd23249"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.10.0"

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

    [deps.ColorVectorSpace.weakdeps]
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "362a287c3aa50601b0bc359053d5c2468f0e7ce0"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.11"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "b6d6bfdd7ce25b0f9b2f6b3dd56b2673a66c8770"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.5"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JuliaInterpreter]]
deps = ["CodeTracking", "InteractiveUtils", "Random", "UUIDs"]
git-tree-sha1 = "7ae67d8567853d367e3463719356b8989e236069"
uuid = "aa1ae85d-cabe-5617-a682-6adf51b2e16a"
version = "0.9.34"

[[deps.LaTeXStrings]]
git-tree-sha1 = "50901ebc375ed41dbf8058da26f9de442febbbec"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.1"

[[deps.Latexify]]
deps = ["Format", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Requires"]
git-tree-sha1 = "ce5f5621cac23a86011836badfedf664a612cee4"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.5"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SparseArraysExt = "SparseArrays"
    SymEngineExt = "SymEngine"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.4.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.6.4+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LoweredCodeUtils]]
deps = ["JuliaInterpreter"]
git-tree-sha1 = "1ce1834f9644a8f7c011eb0592b7fd6c42c90653"
uuid = "6f1432cf-f94c-5a45-995e-cdbf5db27b0b"
version = "3.0.1"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "2fa9ee3e63fd3a4f7a9a4f4744a52f4856de82df"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.13"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+1"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.1.10"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.23+4"

[[deps.OrderedCollections]]
git-tree-sha1 = "dfdf5519f235516220579f949664f1bf44e741c5"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.3"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "8489905bcdbcfac64d1daa51ca07c0d8f0283821"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.1"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.10.0"

[[deps.PlotlyBase]]
deps = ["ColorSchemes", "Dates", "DelimitedFiles", "DocStringExtensions", "JSON", "LaTeXStrings", "Logging", "Parameters", "Pkg", "REPL", "Requires", "Statistics", "UUIDs"]
git-tree-sha1 = "56baf69781fc5e61607c3e46227ab17f7040ffa2"
uuid = "a03496cd-edff-5a9b-9e67-9cda94a718b5"
version = "0.8.19"

[[deps.PlutoHooks]]
deps = ["InteractiveUtils", "Markdown", "UUIDs"]
git-tree-sha1 = "072cdf20c9b0507fdd977d7d246d90030609674b"
uuid = "0ff47ea0-7a50-410d-8455-4348d5de0774"
version = "0.0.5"

[[deps.PlutoLinks]]
deps = ["FileWatching", "InteractiveUtils", "Markdown", "PlutoHooks", "Revise", "UUIDs"]
git-tree-sha1 = "8f5fa7056e6dcfb23ac5211de38e6c03f6367794"
uuid = "0ff47ea0-7a50-410d-8455-4348d5de0420"
version = "0.1.6"

[[deps.PlutoPlotly]]
deps = ["AbstractPlutoDingetjes", "Artifacts", "BaseDirs", "Colors", "Dates", "Downloads", "HypertextLiteral", "InteractiveUtils", "LaTeXStrings", "Markdown", "Pkg", "PlotlyBase", "Reexport", "TOML"]
git-tree-sha1 = "653b48f9c4170343c43c2ea0267e451b68d69051"
uuid = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
version = "0.5.0"

    [deps.PlutoPlotly.extensions]
    PlotlyKaleidoExt = "PlotlyKaleido"
    UnitfulExt = "Unitful"

    [deps.PlutoPlotly.weakdeps]
    PlotlyKaleido = "f2990250-8cf9-495f-b13a-cce12b45703c"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.PlutoTeachingTools]]
deps = ["Downloads", "HypertextLiteral", "LaTeXStrings", "Latexify", "Markdown", "PlutoLinks", "PlutoUI", "Random"]
git-tree-sha1 = "5d9ab1a4faf25a62bb9d07ef0003396ac258ef1c"
uuid = "661c6b06-c737-4d37-b85c-46df65de6f69"
version = "0.2.15"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "ab55ee1510ad2af0ff674dbcced5e94921f867a9"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.59"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "9306f6085165d270f7e3db02af26a400d580f5c6"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.3"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Revise]]
deps = ["CodeTracking", "Distributed", "FileWatching", "JuliaInterpreter", "LibGit2", "LoweredCodeUtils", "OrderedCollections", "REPL", "Requires", "UUIDs", "Unicode"]
git-tree-sha1 = "7b7850bb94f75762d567834d7e9802fc22d62f9c"
uuid = "295af30f-e4ad-537b-8983-00126c2a3abe"
version = "3.5.18"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.10.0"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.10.0"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.2.1+1"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.Tricks]]
git-tree-sha1 = "7822b97e99a1672bfb1b49b668a6d46d58d8cbcb"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.9"

[[deps.URIs]]
git-tree-sha1 = "67db6cc7b3821e19ebe75791a9dd19c9b1188f2b"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.8.0+1"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.52.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"
"""

# ╔═╡ Cell order:
# ╠═68546785-c79a-4e9b-a717-74120ead9b88
# ╟─d2db3192-b5c9-11ee-0d3c-89a9f069d24c
# ╟─5d11c5f1-3ecd-4264-902f-fa8fbb2d3c6b
# ╟─3539521e-d128-4c66-84ed-a85838ea1684
# ╟─63e61c32-c235-438b-978a-ae1a09fef886
# ╟─0f7e56b7-e71c-4e04-af8c-05dfb424860b
# ╟─af442ee2-d0fc-4754-a1fb-917650237f4b
# ╠═7a04a1a7-8a73-49a0-bb27-4731f8428ef1
# ╟─c712c1ee-ac74-4e3e-b9ed-f73ee505fdf6
# ╠═82c2ce66-0d86-414d-868a-5d3aff5168dc
# ╠═f1311c00-8a40-4447-9416-8d0f1c512e33
# ╟─7d6a1fad-6fb4-4b25-9c8a-9a9f581edfde
# ╠═f98dab1c-16f1-418c-acda-3fe0b7554c9c
# ╠═75b51cd7-5d3d-4f0c-8750-d169edf0a3d1
# ╠═cea5f070-70b4-4870-8a41-b0cb4a6ea26b
# ╠═98338255-a250-4b22-8bc3-b3f11b22583b
# ╠═634364b6-05ca-4a54-a0cd-7d0581eaab55
# ╠═6a2e234d-be6c-44a1-8fdd-9e5c6ad8ba01
# ╟─d94fc2d9-8881-4654-98d3-7fbb78bc1271
# ╠═74418960-a9e8-47ab-bf1b-31cdc2e0acb8
# ╟─1fee9d48-46be-443a-9793-0b84e6de35a1
# ╠═4f9c48af-0714-40f8-a4e6-3228ba0b1393
# ╟─126aaff1-428e-45e2-a353-f1afc437db7f
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
