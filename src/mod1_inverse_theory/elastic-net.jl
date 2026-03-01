### A Pluto.jl notebook ###
# v0.20.21

#> [frontmatter]
#> chapter = "1"
#> layout = "layout.jlhtml"
#> tags = ["module1"]
#> title = "Lasso and Elastic Net"

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

# ╔═╡ d5f7245a-ee3e-4390-ba98-417cf941c92d
using HDF5

# ╔═╡ db63cba8-f6f7-49e5-852c-9de9ef5fc8c7
using PlutoPlotly, LinearAlgebra, PlutoUI, GLMNet, Statistics, StatsBase

# ╔═╡ c8057eec-5255-4aad-ab42-610488f9ea9e
TableOfContents()

# ╔═╡ c3661798-b931-11ee-21dc-7f6ee5fa1562
md"""
# Gravity Inversion (Lasso and Elastic Net)

In this exercise, we are interested in the numerical approximation of the gravity of an extended body. The gravitational potential at a point $P$ external to that body is given by Eq. (3.13),

```math
U_g = -G\, \int_x \int_y \int_z \frac{\rho(x,y,z)}{r(x,y,z)}\,dx\,dy\,dz\,,
```

where $G$ is the gravitational constant, $\rho(x,y,z)$ is the density at some location $(x,y,z)$, and $r(x,y,z)$ is the distance between $(x,y,z)$ and the observation point $P$.

To numerically solve the integral, we discretize the spatial domain into evenly-spaced grid points $(x_i, y_i, z_i)$, separated by some small distance $h$. We then obtain an approximation of $U_g$ by summing over all grid points,

```math
U_g \approx -G\,\sum_i \frac{\rho(x_i,y_i,z_i)}{r(x_i,y_i,z_i)}\,V\,,
```

where the small volume $V$ is given by $V=h^3\approx dx\,dy\,dz$. The product $\rho(x_i,y_i,z_i) V$ equals the mass $m_i$ contained in the small volume $V$. Renaming $r_i=r(x_i,y_i,z_i)$, we may rewrite the above equation as

```math
U_g \approx -G\,\sum_i \frac{m_i}{r_i}\,,
```

which is identical to equation (3.12) in the text.

An approximation that we will make throughout this exercise is that the Earth is contained in a small rectangular box. Though this is obviously not realistic, it greatly simplifies the calculations, while still illustrating the basic principles.
"""

# ╔═╡ 993c8d60-3146-438d-b2b6-c38195baf980
md"""
Elastic Net α $(@bind alphaselect Slider(range(0, 1, length=101), show_value=true, default=0.9))
"""

# ╔═╡ c2c1d192-8848-465e-a2d7-7260cb4e46fe
md"""
Constrain on density? 
$(@bind constraint CheckBox())
with maximum $(@bind maximum_density Slider(show_value=true, range(3000, 10000, length=10)))
"""

# ╔═╡ 4ce52e14-0a40-4cd9-82f1-f95ed2e4c938
@bind invG_type Select(["Pseudo Inverse", "Ridge Regression", "Lasso", "Elastic Net"])

# ╔═╡ a9a203e4-484a-47c8-beed-eeab49a72c51
# ╠═╡ disabled = true
#=╠═╡
h5open("gravity_data.h5", "w") do file
    write(file, "G", G) 
	 write(file, "d", data_observed)
	# alternatively, say "@write file A"
end
  ╠═╡ =#

# ╔═╡ 8caa697e-3c72-4d75-819f-a8907bef3fb7
md"## Error Bowls"

# ╔═╡ 74671a5e-4aec-4a51-a759-420d0b0cddf1
md"""

The objective function for Elastic Net proposed by Zou and Hastie (2005)
 is given by
```math
L(m, \lambda, \alpha) = \frac{1}{2}\|Gm-d\|_2^2 + P(m, \lambda, \alpha),
```where
```math
P(m, \lambda, \alpha) = \lambda\left((1-\alpha)\frac{1}{2}\|m\|^2_2 + \alpha \|m\|_1\right).
```
The hyperparameter to be tuned in the Elastic Net is the value for $\alpha$ where, $\alpha \in [0, 1]$
"""

# ╔═╡ 3c3751b1-3236-4ae2-ac4b-90e01076d920


# ╔═╡ 21ed6a59-7a4b-434f-9432-4047ca9520f7


# ╔═╡ e79fc893-233f-4e71-aac7-2c259d57cfcd
md"""
In addition, Zou and Hastie (2005) showed that L1–L2 norm regularization encourages a “grouping effect”, which means that the elements of the solution corresponding to the highly correlated columns of X have similar estimated values. By this enforcing of the grouping effect, the second drawback is mitigated. For the details of this point, please refer to section 2.3 of Zou and Hastie

As the result of L1–L2 norm regularization, the effect that the L1 norm regularized solution is overly concen- trated is resolved by introducing the L2 norm constraint, and at the same time, the sparse nature of the model is also provided by the L1 norm constraint
"""

# ╔═╡ 492e7b9a-8156-476f-aba8-a87e4eef9bea
md"## Physics"

# ╔═╡ 6d0655de-3a68-4990-bb3d-ffe2b2eba4fe
gravity_const=6.67508e-11

# ╔═╡ a30a6d2c-6057-4695-8b20-88a4163227d6
begin
	# Dimension of the computational domain [km].
	x_min=-50.0
	x_max=50.0
	z_min=0
	z_max=50
	
	nx=201
	nz=101
	
	# Coordinate axes.
	xgrid=range(x_min,stop=x_max,length=nx)
	dx=step(xgrid)
	zgrid=range(z_min,stop=z_max,length=nz)
	dz=step(zgrid)
	# Grid spacing [m] and cell volume
	A = step(xgrid)*step(zgrid)
	X = first.(Iterators.product(xgrid, zgrid))
	Z = last.(Iterators.product(xgrid, zgrid))
	# xv,zv=meshgrid(xgrid,zgrid)
	
	# # Define some density distribution.
	mtrue=zeros(nz, nx)
	mtrue[50:80,65:70].=10.0
	# mtrue[70:80,160:190].=5500.0
	mtrue[1:20,140:160].=10.0
	# mtrue[40:60,10:30].=5500.0
end;

# ╔═╡ f039c6e6-f5c4-42eb-95fc-586b711822f4
md"""
$(@bind x_bowl1 Slider(xgrid, show_value=true, default=mean(xgrid)))
$(@bind z_bowl1 Slider(zgrid, show_value=true, default=mean(zgrid)))
"""

# ╔═╡ e2da835d-fcdf-4a56-a632-362c4d62eecb
md"""
$(@bind x_bowl2 Slider(xgrid, show_value=true, default=mean(xgrid)))
$(@bind z_bowl2 Slider(zgrid, show_value=true, default=mean(zgrid)))
"""

# ╔═╡ 51f1666b-374b-43bb-9ac7-613837eddc77
mtrue

# ╔═╡ 2060cd03-e036-4acc-b54a-7fb56b068772
# Plot density distribution.


# ╔═╡ 4de006ac-c4a4-4b0a-b40a-ac9639af66cb
begin
	# Define observation points.
	xobs=xgrid
	zobs=-10.0.*ones(length(xobs))
	
	# Initialize gravitational potential.
	# U=zeros(nx)
	
	# # Loop over all observation points.
	# for k in range(length(x_obs))
	    
	#     r=np.sqrt((x_obs[k]-xv)^2 + (z_obs[k]-zv)^2)
	#     U[k]=-G*V*np.sum(rho/r)
	# end
end

# ╔═╡ cd60baa7-275f-46cc-95b9-9027aa10e828
begin
	G=mapreduce(hcat, xobs, zobs) do x, z
	r = @. sqrt(abs2(X-x) + abs2(Z-z)) * 1e3
	@. -gravity_const*A/r
	end
end

# ╔═╡ 5bb647b8-422b-4656-8df6-05bac55be032
function plot_error_bowl(
)

	ix1 = argmin(abs2.(xgrid .- x_bowl1))
	iz1 = argmin(abs2.(zgrid .- z_bowl1))
	ix2 = argmin(abs2.(xgrid .- x_bowl2))
	iz2 = argmin(abs2.(zgrid .- z_bowl2))

	m=zeros(nz, nx)
	m1=zeros(nz, nx)
	m[iz1, ix1] = 500.0
	m[iz2, ix2] = 500.0
	d=G*vec(m)
	rho_values=range(0, 1000, length=50)
	rho_iterator = Iterators.product(rho_values, rho_values)
	bowl=map(first.(rho_iterator), last.(rho_iterator)) do m11, m22
		fill!(m1, 0.0)
		m1[iz1, ix1] = m11
		m1[iz2, ix2] = m22
		abs2(norm(G*vec(m1)-d, 2))
	end
	@show size(bowl)
	
plot(
    surface(
        contours = attr(
            x=attr(show=true, start= 1.5, size=0.04, color="white"),
            x_end=2,
            z=attr(show=true, start= 0.5, size= 0.05),
            z_end=0.8
        ),
        x=rho_values, y=rho_values,
        z = bowl
    ),
    Layout(
        scene=attr(
            xaxis_nticks=20,
            zaxis_nticks=4,
            camera_eye=attr(x=0, y=-1, z=0.5),
            aspectratio=attr(x=1, y=1, z=0.2)
        )
    )
)
	# plot(heatmap(z=bowl))
end

# ╔═╡ 6ac71a81-5935-4e5a-a635-cf05ced40c6d
plot_error_bowl()

# ╔═╡ 16d73bbb-581a-4883-be01-4e0941f4264b
data_observed=G*vec(mtrue)

# ╔═╡ fb0433b4-83fe-4b2d-95f4-4ac812bf0768
plot(data_observed)

# ╔═╡ 4b227df5-711c-437b-88ed-ff02f060c2a2
Gpinv = pinv(G);

# ╔═╡ e195c809-d359-4a3d-ba0b-4926fbd841e0
@bind λ Slider(0.1:0.9)

# ╔═╡ 27e69011-3df9-4777-86fb-0412bdea76c2
qrG = qr(G)

# ╔═╡ 90cff54a-6967-4d81-a975-d400b3bb169f
G'*G

# ╔═╡ 6202aaa8-09ae-4202-8c4c-fb2841531eb4


# ╔═╡ 0f715e49-2311-438a-a5dd-1dff5310ad4a
F = svd(G)

# ╔═╡ 51193cf3-2464-48aa-b0a0-633dd05dbf72
plot(F.S)

# ╔═╡ f3cb239c-724b-4c4a-b735-fd5ec90cc614


# ╔═╡ 77ea2f3e-481a-4de4-aaf6-a5a8383bc721
md"## Inversion"

# ╔═╡ 4b9f074e-b263-46cb-8d4e-7fa5e7d5eda6
md"### Constraints"

# ╔═╡ 681f2718-c908-4e90-90ae-649568b8f73f
constraints = if(constraint)
	cat(zeros(nx*nz)', maximum_density*ones(nx*nz)', dims=1)
else
	cat(fill(-Inf, nx*nz)', fill(Inf, nx*nz)', dims=1)
end

# ╔═╡ 1e66f28f-5862-4532-a4a5-1abe1c39e44e
alphaused = if(invG_type == "Lasso")
	1.0
elseif(invG_type == "Ridge Regression")
	0.0
else
	alphaselect
end

# ╔═╡ 8d15596b-6beb-41cb-8c40-005d348ee9ca
path = glmnet(G, data_observed, alpha=alphaused, constraints=copy(constraints))

# ╔═╡ 92a0a1b3-31b1-4c1b-a005-44d288b0266a
md"""
Select Lasso Path λ $(@bind lambdaselect Slider(path.lambda, show_value=true, default=last(path.lambda)))
"""

# ╔═╡ f549df45-5ef8-4cd2-a5ea-8e730bcbfeb6
result = if(invG_type == "Pseudo Inverse")
	mhat = pinv(G) * data_observed
	data_predicted = G*mhat
	(; mhat, data_predicted, data_observed)
else
	mhat = path.betas[:,findfirst(x->x==lambdaselect, path.lambda)]
	data_predicted=GLMNet.predict(path, G)[:,findfirst(x->x==lambdaselect, path.lambda)]
	(; mhat, data_predicted, data_observed)
end

# ╔═╡ 1ee66663-8e88-431b-bbf5-32b17e4dac34
md"## Appendix"

# ╔═╡ 13ac643c-6e5c-4c5b-a7ea-e2748a5d953a
md"## Plots"

# ╔═╡ 475058ea-7e24-46a1-9b10-30cdb984b1ef
function plot_densitymap(rho, title="density distribution")
	R=reshape(rho, length(zgrid), length(xgrid))
	plot(heatmap(x=xgrid, y=zgrid, z=R), Layout(width=500, height=300, title=string(title, " [kg/m³]"), xaxis=attr(title="x [km]"), yaxis=attr(autorange="reversed",title="z [km]")))
end

# ╔═╡ 73f9f906-e9aa-4428-8568-44b5180a6e16
PlutoUI.ExperimentalLayout.vbox([plot_densitymap(mtrue, "true mass density"), plot_densitymap(result.mhat, "predicted mass density")])

# ╔═╡ 60928cf4-5eb5-4897-8cba-515114559e8c
function plot_data(data, title="data", names=fill("", size(data,2)))
	D=cat(data,dims=2)
	plot([scatter(x=xgrid, y=d, name=name) for (d, name) in zip(eachslice(D, dims=2), names)], Layout(width=500,height=300, title=title, xaxis=attr(title="x [km]"), yaxis=attr(title="gravitational potential")))
end

# ╔═╡ 746640db-eda8-4638-ad2b-c3b0d8f25f79
plot_data(hcat(result.data_predicted, data_observed), "observed vs predicted data", ["predicted", "observed"])

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
GLMNet = "8d5ece8b-de18-5317-b113-243142960cc6"
HDF5 = "f67ccb44-e63f-5c2f-98bd-6dc0ccc4ba2f"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
PlutoPlotly = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"

[compat]
GLMNet = "~0.7.4"
PlutoPlotly = "~0.4.6"
PlutoUI = "~0.7.60"
StatsBase = "~0.34.4"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.4"
manifest_format = "2.0"
project_hash = "2281cb20ff75e59e6db5adffffa25c5942990355"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

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
git-tree-sha1 = "bca794632b8a9bbe159d56bf9e31c422671b35e0"
uuid = "18cc8868-cbac-4acf-b575-c8ff214dc66f"
version = "1.3.2"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "b0fd3f56fa442f81e0a47815c92245acfaaa4e34"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.31.0"

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
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "362a287c3aa50601b0bc359053d5c2468f0e7ce0"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.11"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "9d8a54ce4b17aa5bdce0ea5c34bc5e7c340d16ad"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.18.1"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.3.0+1"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "DataStructures", "Future", "InlineStrings", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrecompileTools", "PrettyTables", "Printf", "Random", "Reexport", "SentinelArrays", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "d8928e9169ff76c6281f39a659f9bca3a573f24c"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.8.1"

[[deps.DataStructures]]
deps = ["OrderedCollections"]
git-tree-sha1 = "e357641bb3e0638d353c4b29ea0e40ea644066a6"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.19.3"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"
version = "1.11.0"

[[deps.Distributions]]
deps = ["AliasTables", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "fbcc7610f6d8348428f722ecbe0e6cfe22e672c6"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.123"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"
    DistributionsTestExt = "Test"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.DocStringExtensions]]
git-tree-sha1 = "7442a5dfe1ebb773c29cc2962a8980f47221d76c"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.5"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.7.0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "2f979084d1e13948a3352cf64a25df6bd3b4dca3"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.16.0"

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStaticArraysExt = "StaticArrays"
    FillArraysStatisticsExt = "Statistics"

    [deps.FillArrays.weakdeps]
    PDMats = "90014a1f-27ba-587c-ab20-58faa44d9150"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"
version = "1.11.0"

[[deps.GLMNet]]
deps = ["DataFrames", "Distributed", "Distributions", "Printf", "Random", "SparseArrays", "StatsBase", "glmnet_jll"]
git-tree-sha1 = "b873c384d3490304c18224b1d5554cdebaafb60b"
uuid = "8d5ece8b-de18-5317-b113-243142960cc6"
version = "0.7.4"

[[deps.HDF5]]
deps = ["Compat", "HDF5_jll", "Libdl", "MPIPreferences", "Mmap", "Preferences", "Printf", "Random", "Requires", "UUIDs"]
git-tree-sha1 = "e856eef26cf5bf2b0f95f8f4fc37553c72c8641c"
uuid = "f67ccb44-e63f-5c2f-98bd-6dc0ccc4ba2f"
version = "0.17.2"

    [deps.HDF5.extensions]
    MPIExt = "MPI"

    [deps.HDF5.weakdeps]
    MPI = "da04e1cc-30fd-572f-bb4f-1f8673147195"

[[deps.HDF5_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LazyArtifacts", "LibCURL_jll", "Libdl", "MPICH_jll", "MPIPreferences", "MPItrampoline_jll", "MicrosoftMPI_jll", "OpenMPI_jll", "OpenSSL_jll", "TOML", "Zlib_jll", "libaec_jll"]
git-tree-sha1 = "e94f84da9af7ce9c6be049e9067e511e17ff89ec"
uuid = "0234f1f7-429e-5d53-9886-15a909be8d59"
version = "1.14.6+0"

[[deps.Hwloc_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "XML2_jll", "Xorg_libpciaccess_jll"]
git-tree-sha1 = "3d468106a05408f9f7b6f161d9e7715159af247b"
uuid = "e33a78d0-f292-5ffc-b300-72abe9b543c8"
version = "2.12.2+0"

[[deps.HypergeometricFunctions]]
deps = ["LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "68c173f4f449de5b438ee67ed0c9c748dc31a2ec"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.28"

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
git-tree-sha1 = "0ee181ec08df7d7c911901ea38baf16f755114dc"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "1.0.0"

[[deps.InlineStrings]]
git-tree-sha1 = "8f3d257792a522b4601c24a577954b0a8cd7334d"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.4.5"

    [deps.InlineStrings.extensions]
    ArrowTypesExt = "ArrowTypes"
    ParsersExt = "Parsers"

    [deps.InlineStrings.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"
    Parsers = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.InvertedIndices]]
git-tree-sha1 = "6da3c4316095de0f5ee2ebd875df8721e7e0bdbe"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.3.1"

[[deps.IrrationalConstants]]
git-tree-sha1 = "b2d91fe939cae05960e760110b328288867b5758"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.6"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "0533e564aae234aff59ab625543145446d8b6ec2"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.7.1"

[[deps.JSON]]
deps = ["Dates", "Logging", "Parsers", "PrecompileTools", "StructUtils", "UUIDs", "Unicode"]
git-tree-sha1 = "b3ad4a0255688dcb895a52fafbaae3023b588a90"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "1.4.0"

    [deps.JSON.extensions]
    JSONArrowExt = ["ArrowTypes"]

    [deps.JSON.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"

[[deps.JuliaSyntaxHighlighting]]
deps = ["StyledStrings"]
uuid = "ac6e5ff7-fb65-4e79-a425-ec3bc9c03011"
version = "1.12.0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"
version = "1.11.0"

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

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "be484f5c92fad0bd8acfef35fe017900b0b73809"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.18.0+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.12.0"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "13ca9e2586b89836fd20cccf56e57e2b9ae7f38f"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.29"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.MIMEs]]
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

[[deps.MPICH_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Hwloc_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "MPIPreferences", "TOML"]
git-tree-sha1 = "9341048b9f723f2ae2a72a5269ac2f15f80534dc"
uuid = "7cb0a576-ebde-5e09-9194-50597f1243b4"
version = "4.3.2+0"

[[deps.MPIPreferences]]
deps = ["Libdl", "Preferences"]
git-tree-sha1 = "c105fe467859e7f6e9a852cb15cb4301126fac07"
uuid = "3da0fdf6-3ccc-4f1b-acd9-58baa6c99267"
version = "0.1.11"

[[deps.MPItrampoline_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "MPIPreferences", "TOML"]
git-tree-sha1 = "e214f2a20bdd64c04cd3e4ff62d3c9be7e969a59"
uuid = "f1f71cc9-e9ae-5b93-9b94-4fe0e1ad3748"
version = "5.5.4+0"

[[deps.Markdown]]
deps = ["Base64", "JuliaSyntaxHighlighting", "StyledStrings"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MicrosoftMPI_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bc95bf4149bf535c09602e3acdf950d9b4376227"
uuid = "9237b28f-5490-5468-be7b-bb81f5f5e6cf"
version = "10.1.4+3"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

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

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.7+0"

[[deps.OpenMPI_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Hwloc_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "MPIPreferences", "TOML", "Zlib_jll"]
git-tree-sha1 = "ab6596a9d8236041dcd59b5b69316f28a8753592"
uuid = "fe0851c0-eecd-5654-98d4-656369965a5c"
version = "5.0.9+0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.4+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1346c9208249809840c91b26703912dff463d335"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.6+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "05868e21324cede2207c6f0f466b4bfef6d5e7ee"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.1"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "e4cff168707d441cd6bf3ff7e4832bdf34278e4a"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.37"
weakdeps = ["StatsBase"]

    [deps.PDMats.extensions]
    StatsBaseExt = "StatsBase"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "7d2f8f21da5db6a806faf7b9b292296da42b2810"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.3"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.12.1"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PlotlyBase]]
deps = ["ColorSchemes", "Colors", "Dates", "DelimitedFiles", "DocStringExtensions", "JSON", "LaTeXStrings", "Logging", "Parameters", "Pkg", "REPL", "Requires", "Statistics", "UUIDs"]
git-tree-sha1 = "6256ab3ee24ef079b3afa310593817e069925eeb"
uuid = "a03496cd-edff-5a9b-9e67-9cda94a718b5"
version = "0.8.23"

    [deps.PlotlyBase.extensions]
    DataFramesExt = "DataFrames"
    DistributionsExt = "Distributions"
    IJuliaExt = "IJulia"
    JSON3Ext = "JSON3"

    [deps.PlotlyBase.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
    JSON3 = "0f8b85d8-7281-11e9-16c2-39a750bddbf1"

[[deps.PlutoPlotly]]
deps = ["AbstractPlutoDingetjes", "BaseDirs", "Colors", "Dates", "Downloads", "HypertextLiteral", "InteractiveUtils", "LaTeXStrings", "Markdown", "Pkg", "PlotlyBase", "Reexport", "TOML"]
git-tree-sha1 = "1ae939782a5ce9a004484eab5416411c7190d3ce"
uuid = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
version = "0.4.6"

    [deps.PlutoPlotly.extensions]
    PlotlyKaleidoExt = "PlotlyKaleido"
    UnitfulExt = "Unitful"

    [deps.PlutoPlotly.weakdeps]
    PlotlyKaleido = "f2990250-8cf9-495f-b13a-cce12b45703c"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Downloads", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "3ac7038a98ef6977d44adeadc73cc6f596c08109"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.79"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "36d8b4b899628fb92c2749eb488d884a926614d3"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.3"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "07a921781cab75691315adc645096ed5e370cb77"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.3.3"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "522f093a29b31a93e34eaea17ba055d850edea28"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.5.1"

[[deps.PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "PrecompileTools", "Printf", "REPL", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "c5a07210bd060d6a8491b0ccdee2fa0235fc00bf"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "3.1.2"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.PtrArrays]]
git-tree-sha1 = "1d36ef11a9aaf1e8b74dacc6a731dd1de8fd493d"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.3.0"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "9da16da70037ba9d701192e27befedefb91ec284"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.11.2"

    [deps.QuadGK.extensions]
    QuadGKEnzymeExt = "Enzyme"

    [deps.QuadGK.weakdeps]
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"

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
git-tree-sha1 = "62389eeff14780bfe55195b7204c0d8738436d64"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.1"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "5b3d50eb374cea306873b371d3f8d3915a018f0b"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.9.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "58cdd8fb2201a6267e1db87ff148dd6c1dbd8ad8"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.5.1+0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "ebe7e59b37c400f694f52b58c93d26201387da70"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.4.9"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "64d974c2e6fdf07f8155b5b2ca2ffa9069b608d9"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.2"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.12.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "f2685b435df2613e25fc10ad8c26dddb8640f547"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.6.1"

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

    [deps.SpecialFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "178ed29fd5b2a2cfc3bd31c13375ae925623ff36"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.8.0"

[[deps.StatsBase]]
deps = ["AliasTables", "DataAPI", "DataStructures", "IrrationalConstants", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "aceda6f4e598d331548e04cc6b2124a6148138e3"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.10"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "91f091a8716a6bb38417a6e6f274602a19aaa685"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.5.2"

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

    [deps.StatsFuns.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.StringManipulation]]
deps = ["PrecompileTools"]
git-tree-sha1 = "a3c1536470bf8c5e02096ad4853606d7c8f62721"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.4.2"

[[deps.StructUtils]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "9297459be9e338e546f5c4bedb59b3b5674da7f1"
uuid = "ec057cc2-7a8d-4b58-b3b3-92acb9f63b42"
version = "2.6.2"

    [deps.StructUtils.extensions]
    StructUtilsMeasurementsExt = ["Measurements"]
    StructUtilsTablesExt = ["Tables"]

    [deps.StructUtils.weakdeps]
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.8.3+2"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "f2c1efbc8f3a609aadf318094f8fc5204bdaf344"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.12.1"

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
git-tree-sha1 = "311349fd1c93a31f783f977a71e8b062a57d4101"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.13"

[[deps.URIs]]
git-tree-sha1 = "bef26fb046d031353ef97a82e3fdb6afe7f21b1a"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.6.1"

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

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "80d3930c6347cfce7ccf96bd3bafdf079d9c0390"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.13.9+0"

[[deps.Xorg_libpciaccess_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "4909eb8f1cbf6bd4b1c30dd18b2ead9019ef2fad"
uuid = "a65dc6b1-eb27-53a1-bb3e-dea574b5389e"
version = "0.18.1+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.3.1+2"

[[deps.glmnet_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "31adae3b983b579a1fbd7cfd43a4bc0d224c2f5a"
uuid = "78c6b45d-5eaf-5d68-bcfb-a5a2cb06c27f"
version = "2.0.13+0"

[[deps.libaec_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1aa23f01927b2dac46db77a56b31088feee0a491"
uuid = "477f73a3-ac25-53e9-8cc3-50b2fa2566f0"
version = "1.1.4+0"

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
# ╠═c8057eec-5255-4aad-ab42-610488f9ea9e
# ╟─c3661798-b931-11ee-21dc-7f6ee5fa1562
# ╟─92a0a1b3-31b1-4c1b-a005-44d288b0266a
# ╟─993c8d60-3146-438d-b2b6-c38195baf980
# ╟─c2c1d192-8848-465e-a2d7-7260cb4e46fe
# ╟─4ce52e14-0a40-4cd9-82f1-f95ed2e4c938
# ╟─746640db-eda8-4638-ad2b-c3b0d8f25f79
# ╠═73f9f906-e9aa-4428-8568-44b5180a6e16
# ╠═d5f7245a-ee3e-4390-ba98-417cf941c92d
# ╠═a9a203e4-484a-47c8-beed-eeab49a72c51
# ╟─8caa697e-3c72-4d75-819f-a8907bef3fb7
# ╟─74671a5e-4aec-4a51-a759-420d0b0cddf1
# ╠═6ac71a81-5935-4e5a-a635-cf05ced40c6d
# ╠═f039c6e6-f5c4-42eb-95fc-586b711822f4
# ╠═e2da835d-fcdf-4a56-a632-362c4d62eecb
# ╠═51f1666b-374b-43bb-9ac7-613837eddc77
# ╠═3c3751b1-3236-4ae2-ac4b-90e01076d920
# ╠═5bb647b8-422b-4656-8df6-05bac55be032
# ╠═21ed6a59-7a4b-434f-9432-4047ca9520f7
# ╠═e79fc893-233f-4e71-aac7-2c259d57cfcd
# ╟─492e7b9a-8156-476f-aba8-a87e4eef9bea
# ╠═6d0655de-3a68-4990-bb3d-ffe2b2eba4fe
# ╠═a30a6d2c-6057-4695-8b20-88a4163227d6
# ╠═2060cd03-e036-4acc-b54a-7fb56b068772
# ╠═4de006ac-c4a4-4b0a-b40a-ac9639af66cb
# ╠═cd60baa7-275f-46cc-95b9-9027aa10e828
# ╠═16d73bbb-581a-4883-be01-4e0941f4264b
# ╠═fb0433b4-83fe-4b2d-95f4-4ac812bf0768
# ╠═4b227df5-711c-437b-88ed-ff02f060c2a2
# ╠═e195c809-d359-4a3d-ba0b-4926fbd841e0
# ╠═27e69011-3df9-4777-86fb-0412bdea76c2
# ╠═90cff54a-6967-4d81-a975-d400b3bb169f
# ╠═6202aaa8-09ae-4202-8c4c-fb2841531eb4
# ╠═0f715e49-2311-438a-a5dd-1dff5310ad4a
# ╠═51193cf3-2464-48aa-b0a0-633dd05dbf72
# ╠═f3cb239c-724b-4c4a-b735-fd5ec90cc614
# ╟─77ea2f3e-481a-4de4-aaf6-a5a8383bc721
# ╟─4b9f074e-b263-46cb-8d4e-7fa5e7d5eda6
# ╠═681f2718-c908-4e90-90ae-649568b8f73f
# ╠═1e66f28f-5862-4532-a4a5-1abe1c39e44e
# ╠═8d15596b-6beb-41cb-8c40-005d348ee9ca
# ╠═f549df45-5ef8-4cd2-a5ea-8e730bcbfeb6
# ╟─1ee66663-8e88-431b-bbf5-32b17e4dac34
# ╠═db63cba8-f6f7-49e5-852c-9de9ef5fc8c7
# ╟─13ac643c-6e5c-4c5b-a7ea-e2748a5d953a
# ╠═475058ea-7e24-46a1-9b10-30cdb984b1ef
# ╠═60928cf4-5eb5-4897-8cba-515114559e8c
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
