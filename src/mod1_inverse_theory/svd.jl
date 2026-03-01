### A Pluto.jl notebook ###
# v0.20.21

#> [frontmatter]
#> chapter = "1"
#> title = "Singular Value Decomposition"
#> tags = ["module1"]
#> layout = "layout.jlhtml"

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
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
	md""" and  Rotate m$(rotm = @bind mphi Slider(range(0,2pi,length=1000)))""")

# ╔═╡ 63e61c32-c235-438b-978a-ae1a09fef886
md"Scale the columns of U and V with singular values? $(@bind scaleUV CheckBox())"

# ╔═╡ 40d42bb0-fdd4-434c-9653-ce23955dda77
md"Rotate m $(rotm)"

# ╔═╡ c712c1ee-ac74-4e3e-b9ed-f73ee505fdf6
md"## Orthogonal Matrices = Rotation in 2D = Invertible"

# ╔═╡ 019cb5dd-2d1a-4ca7-b8d6-dcb1682075dd
@bind θ Slider(-π/2:0.01:π/2, default=0.0, show_value=true)

# ╔═╡ 324ba76c-b5b1-4647-a0f3-b9ce6d148481
rot_ex = let
	n = 100
	
	# Anisotropic Gaussian cloud
	x = randn(n)
	y = 0.3 .* x .+ 0.4 .* randn(n)

	(; x, y, n)
end

# ╔═╡ 7d6a1fad-6fb4-4b25-9c8a-9a9f581edfde
md"""
## Symmetric Matrices GᵀG and GGᵀ
`V` contains orthonormal eigenvectors of GᵀG, and `U` contains orthonormal eigenvectors of GGᵀ, where both of them share eigenvalues.
"""

# ╔═╡ 634364b6-05ca-4a54-a0cd-7d0581eaab55
G = let
	sampleG; 
	G = randn(2,2)
	if("Symmetric" ∈ Atype)
		G=0.5*(G+G')
	end
	if("Singular" ∈ Atype)
		G[:,2] .= randn() .* G[:,1] 
	end
	G
	
	
end

# ╔═╡ 56fd07be-a44b-46b3-96fe-f2319a693845
eigen(G).vectors * eigen(G).vectors'

# ╔═╡ e35aeeee-a2f2-4367-bc93-6dbae02cd0d1
G

# ╔═╡ 69e856a0-24f9-4ca8-99d4-07638be81d31
G

# ╔═╡ 6a2e234d-be6c-44a1-8fdd-9e5c6ad8ba01
begin
	F=svd(G)
	m=[cos(mphi), sin(mphi)]
	n = F.Vt * m
	o = Diagonal(F.S) * n
	m1=F.U*o
end

# ╔═╡ a9895247-8038-4068-8d08-fc60c89d259b
F.U*o

# ╔═╡ 95c53541-0411-4035-8aa8-501a1f8366c5
F.S

# ╔═╡ d7e0ee1a-0658-467f-9d92-0a24548f4da5
F.U * F.V

# ╔═╡ fb440199-ff31-4423-84d0-8c840269f1bc
F.U

# ╔═╡ c23479dd-2eab-4f0c-9556-882f0e0bd940
m1

# ╔═╡ bec3fcb4-8e57-46cc-80d7-c03c113016d5
F.Vt

# ╔═╡ b8a6fbaa-a517-49fd-a4eb-0f6f5646f44e
F.U * Diagonal(F.S) * F.Vt

# ╔═╡ daf69c93-de5f-4afc-8ff8-1e2be73c13d2
let
	
	x, y, n = rot_ex
	
	X = hcat(x, y)
	Q = [cos(θ) -sin(θ);
     sin(θ)  cos(θ)]
	Q = F.Vt
	X_rot = X * Q'

	axis_len = 3.0


	plt = [
    scatter(
        x = X[:,1],
        y = X[:,2],
        mode = "markers",
        name = "Original",
        marker = attr(size=6)
    ),
    scatter(
        x = X_rot[:,1],
        y = X_rot[:,2],
        mode = "markers",
        name = "Rotated",
        marker = attr(size=6)
    )
]
	
layout = Layout(
    title = "Orthogonal Rotation: Same Cloud, New Viewpoint",
    xaxis = attr(scaleanchor="y"),
    yaxis = attr(scaleanchor="x"),
    legend = attr(x=0.02, y=0.98)
)


	
axes = [
    scatter(x=[0, axis_len], y=[0, 0], mode="lines", name="x-axis"),
    scatter(x=[0, 0], y=[0, axis_len], mode="lines", name="y-axis"),
    scatter(x=[0, axis_len*cos(θ)], y=[0, axis_len*sin(θ)],
            mode="lines", name="rotated x′"),
    scatter(x=[0, -axis_len*sin(θ)], y=[0, axis_len*cos(θ)],
            mode="lines", name="rotated y′")
]

plot(vcat(plt, axes),layout)
end

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
PlutoUI.ExperimentalLayout.hbox([quiverplot(hcat(m,n), title="Rotation", colors=["blue","magenta", "red"], names=["m","n=Vᵀm"]), quiverplot(hcat(n,o), colors=["magenta", "green"], title="Scaling", names=["n=Vᵀm","o=Σn"]), quiverplot(hcat(F.U*o,o), colors=["red", "green"], title="Rotation", names=["Uo","o"]), ])

# ╔═╡ 7a04a1a7-8a73-49a0-bb27-4731f8428ef1
quiverplot(G', colors=["blue", "green"], title="Rows of G")

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

julia_version = "1.12.4"
manifest_format = "2.0"
project_hash = "f5ab05980d345f5fc76996f3e0c3969bd9a49847"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

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
git-tree-sha1 = "26ec26c98ae1453c692efded2b17e15125a5bea1"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.28.0"

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
version = "1.3.0+1"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.7.0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

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
version = "1.11.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JuliaInterpreter]]
deps = ["CodeTracking", "InteractiveUtils", "Random", "UUIDs"]
git-tree-sha1 = "a729439c18f7112cbbd9fcdc1771ecc7f071df6a"
uuid = "aa1ae85d-cabe-5617-a682-6adf51b2e16a"
version = "0.9.39"

[[deps.JuliaSyntaxHighlighting]]
deps = ["StyledStrings"]
uuid = "ac6e5ff7-fb65-4e79-a425-ec3bc9c03011"
version = "1.12.0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

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
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.15.0+0"

[[deps.LibGit2]]
deps = ["LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.9.0+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "OpenSSL_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.3+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.12.0"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.LoweredCodeUtils]]
deps = ["JuliaInterpreter"]
git-tree-sha1 = "688d6d9e098109051ae33d126fcfc88c4ce4a021"
uuid = "6f1432cf-f94c-5a45-995e-cdbf5db27b0b"
version = "3.1.0"

[[deps.MIMEs]]
git-tree-sha1 = "1833212fd6f580c20d4291da9c1b4e8a655b128e"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.0.0"

[[deps.MacroTools]]
git-tree-sha1 = "72aebe0b5051e5143a079a4685a46da330a40472"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.15"

[[deps.Markdown]]
deps = ["Base64", "JuliaSyntaxHighlighting", "StyledStrings"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2025.11.4"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.3.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.29+0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.4+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "cc4054e898b852042d7b503313f7ad03de99c3dd"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.0"

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
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.12.1"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

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
git-tree-sha1 = "7e71a55b87222942f0f9337be62e26b1f103d3e4"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.61"

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
version = "1.11.0"

[[deps.REPL]]
deps = ["InteractiveUtils", "JuliaSyntaxHighlighting", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

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
deps = ["CodeTracking", "FileWatching", "JuliaInterpreter", "LibGit2", "LoweredCodeUtils", "OrderedCollections", "REPL", "Requires", "UUIDs", "Unicode"]
git-tree-sha1 = "9bb80533cb9769933954ea4ffbecb3025a783198"
uuid = "295af30f-e4ad-537b-8983-00126c2a3abe"
version = "3.7.2"

    [deps.Revise.extensions]
    DistributedExt = "Distributed"

    [deps.Revise.weakdeps]
    Distributed = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

    [deps.Statistics.weakdeps]
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

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
version = "1.11.0"

[[deps.Tricks]]
git-tree-sha1 = "6cae795a5a9313bbb4f60683f7263318fc7d1505"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.10"

[[deps.URIs]]
git-tree-sha1 = "67db6cc7b3821e19ebe75791a9dd19c9b1188f2b"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.3.1+2"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.15.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.64.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.7.0+0"
"""

# ╔═╡ Cell order:
# ╠═68546785-c79a-4e9b-a717-74120ead9b88
# ╟─d2db3192-b5c9-11ee-0d3c-89a9f069d24c
# ╟─5d11c5f1-3ecd-4264-902f-fa8fbb2d3c6b
# ╟─3539521e-d128-4c66-84ed-a85838ea1684
# ╟─63e61c32-c235-438b-978a-ae1a09fef886
# ╟─0f7e56b7-e71c-4e04-af8c-05dfb424860b
# ╟─40d42bb0-fdd4-434c-9653-ce23955dda77
# ╠═af442ee2-d0fc-4754-a1fb-917650237f4b
# ╠═a9895247-8038-4068-8d08-fc60c89d259b
# ╠═95c53541-0411-4035-8aa8-501a1f8366c5
# ╠═56fd07be-a44b-46b3-96fe-f2319a693845
# ╠═d7e0ee1a-0658-467f-9d92-0a24548f4da5
# ╠═fb440199-ff31-4423-84d0-8c840269f1bc
# ╠═e35aeeee-a2f2-4367-bc93-6dbae02cd0d1
# ╠═c23479dd-2eab-4f0c-9556-882f0e0bd940
# ╠═bec3fcb4-8e57-46cc-80d7-c03c113016d5
# ╠═b8a6fbaa-a517-49fd-a4eb-0f6f5646f44e
# ╠═69e856a0-24f9-4ca8-99d4-07638be81d31
# ╠═7a04a1a7-8a73-49a0-bb27-4731f8428ef1
# ╟─c712c1ee-ac74-4e3e-b9ed-f73ee505fdf6
# ╠═82c2ce66-0d86-414d-868a-5d3aff5168dc
# ╠═f1311c00-8a40-4447-9416-8d0f1c512e33
# ╟─019cb5dd-2d1a-4ca7-b8d6-dcb1682075dd
# ╠═daf69c93-de5f-4afc-8ff8-1e2be73c13d2
# ╠═324ba76c-b5b1-4647-a0f3-b9ce6d148481
# ╟─7d6a1fad-6fb4-4b25-9c8a-9a9f581edfde
# ╠═634364b6-05ca-4a54-a0cd-7d0581eaab55
# ╠═6a2e234d-be6c-44a1-8fdd-9e5c6ad8ba01
# ╟─d94fc2d9-8881-4654-98d3-7fbb78bc1271
# ╠═74418960-a9e8-47ab-bf1b-31cdc2e0acb8
# ╟─1fee9d48-46be-443a-9793-0b84e6de35a1
# ╠═4f9c48af-0714-40f8-a4e6-3228ba0b1393
# ╟─126aaff1-428e-45e2-a353-f1afc437db7f
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
