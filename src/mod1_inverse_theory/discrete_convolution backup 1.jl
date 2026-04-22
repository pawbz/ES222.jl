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

# ╔═╡ intro_text
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

# ╔═╡ controls_section
md"## Interactive Controls"

# ╔═╡ controls_md
md"""
**Signal Parameters:**

Signal length: $(@bind signal_length Slider(5:15, show_value=true, default=10))

Signal type: $(@bind signal_type Select(["impulse", "step", "ramp", "gaussian", "sine"]))

---

**Filter Parameters:**

Filter length: $(@bind filter_length Slider(2:8, show_value=true, default=3))

Filter type: $(@bind filter_type Select(["box_average", "gaussian", "high_pass", "custom"]))

---

**Visualization:**

Show time step: $(@bind time_step Slider(1:20, show_value=true, default=5))

$(@bind regenerate Button("Regenerate"))
"""

# ╔═╡ generate_signal
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

# ╔═╡ generate_filter
begin
	# Generate filter based on type
	function generate_filter(type, len)
		if type == "box_average"
			h = ones(len) / len
		elseif type == "gaussian"
			c = len / 2
			h = exp.(-((1:len .- c).^2) / (2 * (len/4)^2))
			h = h / sum(h)
		elseif type == "high_pass"
			h = zeros(len)
			h[1] = 1.0
			h[div(len, 2)] = -0.5
			h = h / sum(abs.(h))
		elseif type == "custom"
			h = [0.25, 0.5, 0.25]
			if len != length(h)
				h = vcat(h, zeros(len - length(h)))
			end
		end
		return h
	end
	
	h = generate_filter(filter_type, filter_length)
end

# ╔═╡ compute_convolution
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

# ╔═╡ display_signal_info
md"""
**Input signal x[n]:** $(x)

**Filter h[n]:** $(round.(h, digits=3))

**Output length:** $out_len = $signal_length + $filter_length - 1
"""

# ╔═╡ matrix_view_section
md"## Forward Operator (Convolution Matrix)"

# ╔═╡ build_convolution_matrix
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

# ╔═╡ display_matrix
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

# ╔═╡ verify_convolution
begin
	y_from_matrix = G * x
	error = norm(y - y_from_matrix)
	
	md"""
	**Verification:** 
	Convolution via matrix multiplication matches direct convolution (error: $(round(error, digits=10)))
	"""
end

# ╔═╡ visualization_section
md"## Time-Domain Visualization"

# ╔═╡ plot_shifted_sources
begin
	t = 1:signal_length
	t_output = 1:out_len
	
	# Extract rows of G at selected time step
	n_vis = min(time_step, out_len)
	row = G[n_vis, :]
	
	# Create traces
	traces = []
	
	# Plot input signal
	push!(traces, scatter(
		x=t,
		y=x,
		mode="lines+markers",
		name="Input signal x[n]",
		line=attr(color="blue", width=2),
		marker=attr(size=6)
	))
	
	# Plot each weighted shifted version of input
	colors = ["lightblue", "lightgreen", "lightcoral", "lightyellow", "lightpink"]
	for k in 1:filter_length
		weighted_signal = h[k] * x
		shift = k - 1
		
		if shift < signal_length
			x_shifted = vcat(zeros(shift), weighted_signal)
			x_shifted = vcat(x_shifted, zeros(out_len - length(x_shifted)))
			
			push!(traces, scatter(
				x=1:out_len,
				y=x_shifted,
				mode="lines+markers",
				name="$(round(h[k], digits=2)) × x[n-$(shift)]",
				line=attr(color=colors[k], width=1, dash="dash"),
				marker=attr(size=4),
				opacity=0.6
			))
		end
	end
	
	# Plot output convolution
	push!(traces, scatter(
		x=t_output,
		y=y,
		mode="lines+markers",
		name="Output y[n] = Σ weighted shifts",
		line=attr(color="red", width=3),
		marker=attr(size=8)
	))
	
	# Highlight the current time step
	push!(traces, scatter(
		x=[n_vis],
		y=[y[n_vis]],
		mode="markers",
		marker=attr(size=15, color="red", symbol="star"),
		name="Current output",
		showlegend=false
	))
	
	layout = Layout(
		title="Time-Domain Convolution: y[n] = Σ h[k] × x[n-k+1]<br>Output at n=$(n_vis): y=$(round(y[n_vis], digits=3))",
		xaxis=attr(title="Time index n"),
		yaxis=attr(title="Amplitude"),
		showlegend=true,
		width=900,
		height=500,
		hovermode="x unified"
	)
	
	plot(traces, layout)
end

# ╔═╡ bar_chart_composition
begin
	# Show the composition of y[n_vis] as a sum of weighted inputs
	n_vis = min(time_step, out_len)
	
	contributions = []
	labels = []
	
	for k in 1:filter_length
		idx = n_vis - k + 1
		if idx >= 1 && idx <= signal_length
			contrib = h[k] * x[idx]
			push!(contributions, contrib)
			push!(labels, "h[$k]×x[$idx]")
		end
	end
	
	trace = bar(
		x=labels,
		y=contributions,
		name="Contributions",
		marker=attr(color="teal")
	)
	
	layout_bar = Layout(
		title="Composition of Output at n=$(n_vis)<br>y[$n_vis] = $(round(sum(contributions), digits=3))",
		yaxis=attr(title="Contribution value"),
		width=700,
		height=400,
		showlegend=false
	)
	
	plot([trace], layout_bar)
end

# ╔═╡ interpretation_section
md"## Interpretation & Key Concepts"

# ╔═╡ interpretation_text
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

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
PlutoPlotly = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"

[compat]
Distributions = "~0.25.115"
PlutoPlotly = "~0.4.6"
PlutoUI = "~0.7.60"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.1"
manifest_format = "2.0"
project_hash = "421ee1fff511dfef2c5c00bd6114a657b4771e81"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"
"""

# ╔═╡ Cell order:
# ╟─c8945e8c-1425-4198-bf0f-f25bf48f79d4
# ╟─intro_text
# ╟─controls_section
# ╟─controls_md
# ╠═generate_signal
# ╠═generate_filter
# ╠═compute_convolution
# ╟─display_signal_info
# ╟─matrix_view_section
# ╠═build_convolution_matrix
# ╟─display_matrix
# ╟─verify_convolution
# ╟─visualization_section
# ╟─plot_shifted_sources
# ╟─bar_chart_composition
# ╟─interpretation_section
# ╟─interpretation_text
# ╠═499b6d53-97a2-4ba8-a6d6-030ce1e6fa43
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
