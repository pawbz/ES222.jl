### A Pluto.jl notebook ###
# v0.20.21

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

# ╔═╡ 499b6d53-97a2-4ba8-a6d6-030ce1e6fa43
using PlutoPlotly, LinearAlgebra, Distributions, PlutoUI

# ╔═╡ c8945e8c-1425-4198-bf0f-f25bf48f79d4
md"# Discrete 1D Convolution: Linear Combination of Shifted Sources"

# ╔═╡ 2bd3beb4-fb9c-11f0-936e-8b0c12118131
md"""
## Key Insight
Convolution can be understood as a **linear combination of time-shifted copies** of the input signal.

Given:
- **Input signal**: `x[n]` (the source)
- **Filter/kernel**: `h[n]` (weights)

The output convolution `y[n]` is constructed by:
1. Creating shifted versions of `x[n]`: `x[n], x[n-1], x[n-2], ...`
2. Weighting each shift by the filter coefficients
3. Summing all weighted shifts together

This is exactly like the forward operator `G` in inverse theory — each column of `G` is a time-shifted version of the source!
"""

# ╔═╡ 2bd3c10a-fb9c-11f0-b2aa-a15534a70e30
md"## Interactive Controls"

# ╔═╡ 2bd3c15a-fb9c-11f0-942f-adea92d18d35
md"""
**Signal Parameters:**

Signal length: $(@bind signal_length Slider(5:15, show_value=true, default=10))

Signal type: $(@bind signal_type Select(["impulse", "step", "ramp", "gaussian", "sine"]))

---

**Filter Weights (4 coefficients):**

h[1]: $(@bind h1 Slider(-1.0:0.05:1.0, show_value=true, default=0.25))

h[2]: $(@bind h2 Slider(-1.0:0.05:1.0, show_value=true, default=0.5))

h[3]: $(@bind h3 Slider(-1.0:0.05:1.0, show_value=true, default=0.25))

h[4]: $(@bind h4 Slider(-1.0:0.05:1.0, show_value=true, default=0.0))

---

**Visualization:**

Show time step: $(@bind time_step Slider(1:20, show_value=true, default=5))

$(@bind regenerate Button("Regenerate"))
"""

# ╔═╡ 2bd3c2ba-fb9c-11f0-9ca3-25bd7caabd42
begin
	regenerate
	
	# Generate input signal based on type
	function generate_signal(type, len)
		if type == "impulse"
			x = zeros(len)
			x[1] = 1.0
		elseif type == "step"
			x = ones(len)
		elseif type == "ramp"
			x = 1.0:len
		elseif type == "gaussian"
			x = exp.(-((1:len .- len/2).^2) / (2 * (len/4)^2))
		elseif type == "sine"
			x = sin.(2π * (1:len) / len)
		end
		return x / maximum(abs.(x))  # Normalize
	end
	
	x = generate_signal(signal_type, signal_length)
end

# ╔═╡ 2bd3c43e-fb9c-11f0-a875-adc61e58a141
begin
	# Use filter weights directly from sliders
	h = [h1, h2, h3, h4]
	filter_length = 4
end

# ╔═╡ 2bd3c59c-fb9c-11f0-9225-c1d07827e730
begin
	# Compute full convolution (padding with zeros)
	out_len = signal_length + filter_length - 1
	y = zeros(out_len)
	
	for n in 1:out_len
		# At each output position, sum weighted shifted inputs
		for k in 1:filter_length
			if n - k + 1 >= 1 && n - k + 1 <= signal_length
				y[n] += h[k] * x[n - k + 1]
			end
		end
	end
end

# ╔═╡ 2bd3cdd0-fb9c-11f0-be36-a53a98f10843
let
	n_vis = min(time_step, out_len)
	
	# Subplot 1: Input Signal (stem plot)
	# Create stems as vertical lines
	stem_x_input = []
	stem_y_input = []
	for i in 1:signal_length
		append!(stem_x_input, [i, i, NaN])
		append!(stem_y_input, [0, x[i], NaN])
	end
	
	trace_input_stems = scatter(
		x=stem_x_input,
		y=stem_y_input,
		mode="lines",
		line=attr(color="steelblue", width=2),
		showlegend=false,
		xaxis="x1",
		yaxis="y1"
	)
	
	trace_input = scatter(
		x=1:signal_length,
		y=x,
		mode="markers",
		name="Input x[n]",
		marker=attr(color="steelblue", size=10, symbol="circle"),
		xaxis="x1",
		yaxis="y1"
	)
	
	# Subplot 2: Filter (impulse response) - stem plot
	# Create stems as vertical lines
	stem_x_filter = []
	stem_y_filter = []
	for i in 1:filter_length
		append!(stem_x_filter, [i, i, NaN])
		append!(stem_y_filter, [0, h[i], NaN])
	end
	
	trace_filter_stems = scatter(
		x=stem_x_filter,
		y=stem_y_filter,
		mode="lines",
		line=attr(color="coral", width=2),
		showlegend=false,
		xaxis="x2",
		yaxis="y2"
	)
	
	trace_filter = scatter(
		x=1:filter_length,
		y=h,
		mode="markers",
		name="Filter h[k]",
		marker=attr(color="coral", size=10, symbol="circle"),
		xaxis="x2",
		yaxis="y2"
	)
	
	# Subplot 3: Shifted and weighted signals (stacked area)
	traces_shifts = [scatter()]
	shift_colors = ["rgba(31, 119, 180, 0.7)", "rgba(255, 127, 14, 0.7)", "rgba(44, 160, 44, 0.7)", "rgba(214, 39, 40, 0.7)", "rgba(148, 103, 189, 0.7)"]
	
	for k in 1:filter_length
		weighted_signal = h[k] * x
		shift = k - 1
		x_shifted = vcat(zeros(shift), weighted_signal, zeros(out_len - shift - signal_length))
		
		push!(traces_shifts, scatter(
			x=1:out_len,
			y=x_shifted,
			mode="lines",
			name="h[$(k)]×x shifted by $(shift)",
			fill="tonexty",
			line=attr(color=shift_colors[min(k, length(shift_colors))], width=2),
			xaxis="x3",
			yaxis="y3"
		))
	end
	
	# Add a zero baseline for fill
	insert!(traces_shifts, 1, scatter(
		x=1:out_len,
		y=zeros(out_len),
		mode="lines",
		line=attr(color="rgba(0,0,0,0)"),
		showlegend=false,
		xaxis="x3",
		yaxis="y3"
	))
	
	# Subplot 4: Final output with current position highlighted
	trace_output = scatter(
		x=1:out_len,
		y=y,
		mode="lines+markers",
		name="Output y[n]",
		line=attr(color="crimson", width=3),
		marker=attr(size=6),
		xaxis="x4",
		yaxis="y4"
	)
	
	trace_current = scatter(
		x=[n_vis],
		y=[y[n_vis]],
		mode="markers",
		marker=attr(size=18, color="gold", symbol="star", line=attr(width=2, color="black")),
		name="Current (n=$(n_vis))",
		xaxis="x4",
		yaxis="y4"
	)
	
	# Combine all traces
	all_traces = vcat([trace_input_stems, trace_input, trace_filter_stems, trace_filter], traces_shifts, [trace_output, trace_current])
	
	layout = Layout(
		title="Convolution as Linear Combination (Time Step n=$(n_vis): y=$(round(y[n_vis], digits=3)))",
		xaxis1=attr(title="Index", domain=[0, 0.42], anchor="y1"),
		yaxis1=attr(title="Input Signal", domain=[0.58, 1], anchor="x1"),
		xaxis2=attr(title="Filter Index", domain=[0.58, 1], anchor="y2"),
		yaxis2=attr(title="Filter Weights", domain=[0.58, 1], anchor="x2"),
		xaxis3=attr(title="Time Index", domain=[0, 0.42], anchor="y3"),
		yaxis3=attr(title="Weighted & Shifted", domain=[0, 0.42], anchor="x3"),
		xaxis4=attr(title="Time Index", domain=[0.58, 1], anchor="y4"),
		yaxis4=attr(title="Output", domain=[0, 0.42], anchor="x4"),
		showlegend=true,
		legend=attr(x=1.02, y=0.5, xanchor="left"),
		width=1200,
		height=800,
		hovermode="closest",
		margin=attr(l=80, r=250, t=80, b=60)
	)
	
	WideCell(plot(all_traces, layout))
end

# ╔═╡ 2bd3d208-fb9c-11f0-a7cf-853bba122f48
let
	# Show the composition of y[n_vis] as a sum of weighted inputs
	n_vis = min(time_step, out_len)
	
	contributions = []
	labels = []
	filter_vals = []
	input_vals = []
	
	for k in 1:filter_length
		idx = n_vis - k + 1
		if idx >= 1 && idx <= signal_length
			contrib = h[k] * x[idx]
			push!(contributions, contrib)
			push!(labels, "h[$(k)]×x[$(idx)]")
			push!(filter_vals, h[k])
			push!(input_vals, x[idx])
		end
	end
	
	# Color stems by positive/negative contribution
	colors = [c >= 0 ? "seagreen" : "crimson" for c in contributions]
	
	# Create stems for contributions
	stem_traces = []
	for i in 1:length(contributions)
		stem_trace = scatter(
			x=[i, i],
			y=[0, contributions[i]],
			mode="lines",
			line=attr(color=colors[i], width=3),
			showlegend=false,
			hoverinfo="skip"
		)
		push!(stem_traces, stem_trace)
	end
	
	trace_contrib = scatter(
		x=1:length(contributions),
		y=contributions,
		mode="markers+text",
		name="Contributions",
		marker=attr(color=colors, size=12, symbol="circle", line=attr(width=2, color="black")),
		text=round.(contributions, digits=3),
		textposition="top center",
		hovertemplate="%{text}<br>Value: %{y:.4f}<extra></extra>",
		customdata=labels,
		hovertemplate="%{customdata}<br>Value: %{y:.4f}<extra></extra>"
	)
	
	# Add horizontal line at sum
	total = sum(contributions)
	trace_sum = scatter(
		x=[labels[1], labels[end]],
		y=[total, total],
		mode="lines",
		line=attr(color="black", width=3, dash="dash"),
		name="Sum = $(round(total, digits=3))",
		showlegend=true
	)
	
	layout_bar = Layout(
		title="Detailed Breakdown at Time n=$(n_vis)<br><b>y[$(n_vis)] = $(join(["$(round(filter_vals[i], digits=2))×$(round(input_vals[i], digits=2))" for i in 1:length(filter_vals)], " + ")) = $(round(total, digits=3))</b>",
		yaxis=attr(title="Contribution Value", zeroline=true, zerolinewidth=2),
		xaxis=attr(title="Term", tickmode="array", tickvals=1:length(labels), ticktext=labels),
		width=850,
		height=450,
		showlegend=true,
		plot_bgcolor="#f8f9fa",
		hovermode="closest"
	)
	
	plot(vcat(stem_traces, [trace_contrib, trace_sum]), layout_bar)
end

# ╔═╡ 2bd3c72c-fb9c-11f0-bd92-db091216a01e
md"""
**Input signal x[n]:** $(x)

**Filter h[n]:** $(round.(h, digits=3))

**Output length:** $out_len = $signal_length + $filter_length - 1
"""

# ╔═╡ 2bd3c84e-fb9c-11f0-912d-218a85d0136e
md"## Forward Operator (Convolution Matrix)"

# ╔═╡ 2bd3c8da-fb9c-11f0-b9a4-538d86c6fed2
begin
	# Build the convolution matrix (Toeplitz-like structure)
	# Each row is the filter applied at different time positions
	G = zeros(out_len, signal_length)
	
	for n in 1:out_len
		for k in 1:filter_length
			if n - k + 1 >= 1 && n - k + 1 <= signal_length
				G[n, n - k + 1] = h[k]
			end
		end
	end
end

# ╔═╡ 2bd3caa6-fb9c-11f0-98f9-8bd75ea80e30
begin
	md"""
	**Convolution Matrix G (forward operator):**
	
	Each row shows which input samples contribute to each output.
	Each column is a time-shifted version of the filter.
	
	```
	$(round.(G, digits=3))
	```
	"""
end

# ╔═╡ 2bd3cca4-fb9c-11f0-9c62-89dd4d7d934d
begin
	y_from_matrix = G * x
	error = norm(y - y_from_matrix)
	
	md"""
	**Verification:** 
	Convolution via matrix multiplication matches direct convolution (error: $(round(error, digits=10)))
	"""
end

# ╔═╡ 2bd3cd56-fb9c-11f0-8e99-d37711cdf01c
md"## Convolution Matrix Visualization"

# ╔═╡ 2bd3d3fa-fb9c-11f0-8eda-fd69cee7c812
md"## Interpretation & Key Concepts"

# ╔═╡ 2bd3d438-fb9c-11f0-b6f6-69644f2660cc
md"""
### Linear System Perspective
Convolution is a **linear operator** represented by matrix `G`:
- **Columns of G**: Each column is a time-shifted impulse response (time-shifted filter)
- **Matrix-vector product**: `y = Gx` computes the convolution
- **Each element `y[n]`**: A linear combination of input samples, weighted by filter coefficients

### Time-Domain Interpretation
At each output time `n`:
```
y[n] = h[1]×x[n] + h[2]×x[n-1] + h[3]×x[n-2] + ...
```

We're:
1. **Creating shifted versions** of the input (source shifted in time)
2. **Weighting each shift** by the filter coefficient
3. **Summing all weighted shifts** to get the output

### Physical Analogy
Think of `x[n]` as **point sources** activated at different times. 
The filter `h[k]` describes how each source **decays or spreads**.
The convolution output `y[n]` is the **superposition** of all delayed/attenuated sources.

### Connection to Inverse Theory
In inverse problems, we often have:
- `G`: Forward operator (convolution matrix)
- `x`: Unknown source (what we want to find)
- `y`: Observed data (what we measure)
- Goal: Solve `Gx = y` to recover the source

This notebook shows how `G` has a special **Toeplitz structure** arising from time-shifts!
"""

# ╔═╡ 2bd3cd57-fb9c-11f0-8e99-d37711cdf01d
let
	# Create heatmap of convolution matrix
	trace_heatmap = heatmap(
		z=G',
		x=1:out_len,
		y=1:signal_length,
		colorscale="RdBu",
		zmid=0,
		colorbar=attr(title="Weight"),
		text=round.(G', digits=2),
		texttemplate="%{text}",
		textfont=attr(size=10),
		hovertemplate="Output n=%{x}<br>Input m=%{y}<br>Weight=%{z:.3f}<extra></extra>"
	)
	
	layout_heatmap = Layout(
		title="Convolution Matrix G (Toeplitz Structure)<br>Each column = time-shifted filter",
		xaxis=attr(title="Output index n", side="bottom"),
		yaxis=attr(title="Input index m", autorange="reversed"),
		width=800,
		height=400,
		plot_bgcolor="white"
	)
	
	plot([trace_heatmap], layout_heatmap)
end

# ╔═╡ 2bd3cd58-fb9c-11f0-8e99-d37711cdf01e
md"## Multi-Panel Visualization: Linear Combination View"

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
PlutoPlotly = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"

[compat]
Distributions = "~0.25.123"
PlutoPlotly = "~0.6.5"
PlutoUI = "~0.7.79"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.4"
manifest_format = "2.0"
project_hash = "258aa15d3a4a15ad1e719821e21587a9626a38f0"

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

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "b0fd3f56fa442f81e0a47815c92245acfaaa4e34"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.31.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "67e11ee83a43eb71ddc950302c53bf33f0690dfe"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.1"
weakdeps = ["StyledStrings"]

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "8b3b6f87ce8f65a2b4f857528fd8d70086cd72b1"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.11.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "37ea44092930b1811e666c3bc38065d7d87fcc74"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.13.1"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.3.0+1"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataStructures]]
deps = ["OrderedCollections"]
git-tree-sha1 = "e357641bb3e0638d353c4b29ea0e40ea644066a6"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.19.3"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

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

[[deps.HashArrayMappedTries]]
git-tree-sha1 = "2eaa69a7cab70a52b9687c8bf950a5a93ec895ae"
uuid = "076d061b-32b6-4027-95e0-9a2c6f6d7e74"
version = "0.2.0"

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

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "b2d91fe939cae05960e760110b328288867b5758"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.6"

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

[[deps.Markdown]]
deps = ["Base64", "JuliaSyntaxHighlighting", "StyledStrings"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

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
deps = ["AbstractPlutoDingetjes", "Artifacts", "ColorSchemes", "Colors", "Dates", "Downloads", "HypertextLiteral", "InteractiveUtils", "LaTeXStrings", "Markdown", "Pkg", "PlotlyBase", "PrecompileTools", "Reexport", "ScopedValues", "Scratch", "TOML"]
git-tree-sha1 = "8acd04abc9a636ef57004f4c2e6f3f6ed4611099"
uuid = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
version = "0.6.5"

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

[[deps.ScopedValues]]
deps = ["HashArrayMappedTries", "Logging"]
git-tree-sha1 = "c3b2323466378a2ba15bea4b2f73b081e022f473"
uuid = "7e506255-f358-4e82-b7e4-beb19740aa63"
version = "1.5.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "9b81b8393e50b7d4e6d0a9f14e192294d3b7c109"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.3.0"

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
# ╟─c8945e8c-1425-4198-bf0f-f25bf48f79d4
# ╠═499b6d53-97a2-4ba8-a6d6-030ce1e6fa43
# ╟─2bd3beb4-fb9c-11f0-936e-8b0c12118131
# ╟─2bd3c10a-fb9c-11f0-b2aa-a15534a70e30
# ╟─2bd3c15a-fb9c-11f0-942f-adea92d18d35
# ╟─2bd3cdd0-fb9c-11f0-be36-a53a98f10843
# ╟─2bd3d208-fb9c-11f0-a7cf-853bba122f48
# ╠═2bd3c2ba-fb9c-11f0-9ca3-25bd7caabd42
# ╠═2bd3c43e-fb9c-11f0-a875-adc61e58a141
# ╠═2bd3c59c-fb9c-11f0-9225-c1d07827e730
# ╠═2bd3c72c-fb9c-11f0-bd92-db091216a01e
# ╠═2bd3c84e-fb9c-11f0-912d-218a85d0136e
# ╠═2bd3c8da-fb9c-11f0-b9a4-538d86c6fed2
# ╠═2bd3caa6-fb9c-11f0-98f9-8bd75ea80e30
# ╠═2bd3cca4-fb9c-11f0-9c62-89dd4d7d934d
# ╟─2bd3cd56-fb9c-11f0-8e99-d37711cdf01c
# ╟─2bd3d3fa-fb9c-11f0-8eda-fd69cee7c812
# ╟─2bd3d438-fb9c-11f0-b6f6-69644f2660cc
# ╠═2bd3cd57-fb9c-11f0-8e99-d37711cdf01d
# ╠═2bd3cd58-fb9c-11f0-8e99-d37711cdf01e
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
