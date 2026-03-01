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
md"# LASSO vs Ridge Regression: Seismic Source Deconvolution"

# ╔═╡ a33ef9ac-fb9d-11f0-aef3-2738233020c4
md"""
## The Deconvolution Problem

In seismic exploration, an **airgun source** fires and creates a wavelet that propagates through the earth. The recorded data is a **convolution** of:
- The airgun signature (known wavelet/filter)
- The earth's reflectivity (sparse signal we want to recover)

**Goal:** Given noisy recorded data and the airgun signature, recover the sparse reflectivity sequence.

## Regularization Approaches

### Ridge Regression (L2)
Penalizes large coefficients: `||Gx - d||² + λ₂||x||²`
- Produces **smooth** solutions
- All coefficients shrink proportionally
- Never exactly zero

### LASSO Regression (L1)
Penalizes absolute values: `||Gx - d||² + λ₁||x||₁`
- Produces **sparse** solutions
- Many coefficients become exactly zero
- Automatic feature selection

### Coordinate Descent
An iterative algorithm that optimizes one variable at a time while keeping others fixed.
"""

# ╔═╡ a33efdbc-fb9d-11f0-a81d-9ffc02850aa2
md"## Interactive Controls"

# ╔═╡ a33efe34-fb9d-11f0-b061-a349c8b76450
md"""
**Problem Setup:**

Reflectivity sparsity: $(@bind sparsity Slider(0.02:0.01:0.15, show_value=true, default=0.05))

Noise level (%): $(@bind noise_percent Slider(0:1:30, show_value=true, default=10))

---

**Ridge Regression (L2):**

λ₂ (Ridge): $(@bind lambda_ridge Slider(0:0.01:5.0, show_value=true, default=0.5))

---

**LASSO Regression (L1):**

λ₁ (LASSO): $(@bind lambda_lasso Slider(0:0.01:5.0, show_value=true, default=1.0))

---

Max iterations: $(@bind max_iter Slider(50:50:500, show_value=true, default=200))

$(@bind resample Button("Resample Data"))
"""

# ╔═╡ a33f010e-fb9d-11f0-bd8b-0b62c4571bc7
md"## Problem Setup: Forward Model"

# ╔═╡ a33f0184-fb9d-11f0-a575-71dd14204fdb
begin
	resample
	
	# Get airgun signature (from deconvolution.jl)
	function get_airgun()
		airgun_data = [0.0019455 0.31999; 0.0038911 0.66316; 0.0058366 0.86648; 
			0.0077821 0.96656; 0.0097276 1; 0.011673 0.93253; 0.013619 -0.34515; 
			0.015564 -0.44781; 0.01751 -0.48336; 0.019455 -0.5189; 0.021401 -0.55209; 
			0.023346 -0.59392; 0.025292 -0.68945; 0.027237 -0.81781; 0.029183 -0.91316; 
			0.031128 -0.78497; 0.033074 -0.38476; 0.035019 -0.076951; 0.036965 0.086752; 
			0.038911 0.21784; 0.040856 0.29985; 0.042802 0.32102; 0.044747 0.32677; 
			0.046693 0.33102; 0.048638 0.3338; 0.050584 0.33514; 0.052529 0.32802; 
			0.054475 0.29396; 0.05642 0.26859; 0.058366 0.25177; 0.060311 0.23896; 
			0.062257 0.23429; 0.064202 0.24808; 0.066148 0.27332; 0.068093 0.2945; 
			0.070039 0.31735; 0.071984 0.33727; 0.07393 0.34648; 0.075875 0.14177; 
			0.077821 -0.37562; 0.079767 -0.87791; 0.081712 -0.89059; 0.083658 -0.89326; 
			0.085603 -0.80043; 0.087549 -0.65179; 0.089494 -0.47611; 0.09144 -0.27827; 
			0.093385 -0.074001; 0.095331 0.10743; 0.097276 0.24002; 0.099222 0.29809]
		
		airgun_t = airgun_data[:, 1]
		airgun_sig = airgun_data[:, 2]
		return airgun_t, airgun_sig
	end
	
	airgun_t, airgun = get_airgun()
	n_wavelet = length(airgun)
	n_reflectivity = 200  # Reflectivity sequence length
	
	# Generate sparse reflectivity (few reflectors)
	x_true = zeros(n_reflectivity)
	n_spikes = max(2, round(Int, n_reflectivity * sparsity))
	spike_positions = rand(10:n_reflectivity-10, n_spikes)
	spike_amplitudes = randn(n_spikes) .* 0.8
	for (pos, amp) in zip(spike_positions, spike_amplitudes)
		x_true[pos] = amp
	end
	
	n_source = n_reflectivity  # For compatibility with rest of code
	n_data = n_reflectivity + n_wavelet - 1
end

# ╔═╡ 0aa0b3d5-e785-41b0-9193-fbe268c98c7e
plot(airgun_t, airgun, 
    Layout(title="Airgun Signature (Wavelet)", 
    xaxis_title="Time (s)", 
    yaxis_title="Amplitude", 
    width=700, height=400))

# ╔═╡ a33f03fc-fb9d-11f0-88e7-f57fc49ab40a
begin
	# Airgun is the wavelet (no need to generate Ricker)
	wavelet = airgun
end

# ╔═╡ a33f05a0-fb9d-11f0-91e8-5b072b9bd2ea
begin
	# Build convolution matrix (forward operator)
	function build_convolution_matrix(wavelet, n_source, n_data)
		n_wavelet = length(wavelet)
		G = zeros(n_data, n_source)
		
		for i in 1:n_source
			for j in 1:n_wavelet
				out_idx = i + j - 1
				if out_idx <= n_data
					G[out_idx, i] = wavelet[j]
				end
			end
		end
		return G
	end
	
	G = build_convolution_matrix(wavelet, n_source, n_data)
end

# ╔═╡ a33f0796-fb9d-11f0-a746-6b699aacd7b6
begin
	# Generate observed data with noise
	d_true = G * x_true
	noise_level = noise_percent / 100.0 * norm(d_true)
	noise = randn(n_data) * noise_level
	d_obs = d_true + noise
end

# ╔═╡ a33f08ac-fb9d-11f0-88ca-1f136753c450
md"""
**Problem dimensions:**
- Reflectivity length: $n_source samples
- Data length: $n_data samples
- Airgun wavelet length: $(length(wavelet)) samples
- True sparsity: $(sum(abs.(x_true) .> 0.01)) / $n_source reflectors
- SNR: $(round(20*log10(norm(d_true)/norm(noise)), digits=1)) dB
"""

# ╔═╡ a33f0a28-fb9d-11f0-a0e9-6581714db953
md"## Coordinate Descent Algorithms"

# ╔═╡ a33f0a98-fb9d-11f0-8a16-4b523bba358f
begin
	"""
	Ridge regression via coordinate descent
	Minimize: ||Gx - d||² + λ||x||²
	"""
	function ridge_coordinate_descent(G, d, λ; max_iter=200, tol=1e-6)
		n = size(G, 2)
		x = zeros(n)
		
		# Precompute for efficiency
		GtG = G' * G
		Gtd = G' * d
		
		history = [copy(x)]
		objectives = Float64[]
		
		for iter in 1:max_iter
			x_old = copy(x)
			
			# Update each coordinate
			for j in 1:n
				# Compute partial residual (excluding j)
				r_j = Gtd[j] - dot(GtG[j, :], x) + GtG[j, j] * x[j]
				
				# Update coordinate j
				x[j] = r_j / (GtG[j, j] + λ)
			end
			
			push!(history, copy(x))
			
			# Compute objective
			residual = G * x - d
			obj = dot(residual, residual) + λ * dot(x, x)
			push!(objectives, obj)
			
			# Check convergence
			if norm(x - x_old) < tol
				break
			end
		end
		
		return x, history, objectives
	end
end

# ╔═╡ a33f0e2e-fb9d-11f0-973b-995b38420393
begin
	"""
	LASSO regression via coordinate descent
	Minimize: ||Gx - d||² + λ||x||₁
	Uses soft thresholding
	"""
	function lasso_coordinate_descent(G, d, λ; max_iter=200, tol=1e-6)
		n = size(G, 2)
		x = zeros(n)
		
		# Precompute for efficiency
		GtG = G' * G
		Gtd = G' * d
		
		history = [copy(x)]
		objectives = Float64[]
		
		# Soft thresholding operator
		soft_threshold(z, γ) = sign(z) * max(abs(z) - γ, 0)
		
		for iter in 1:max_iter
			x_old = copy(x)
			
			# Update each coordinate
			for j in 1:n
				# Compute partial residual (excluding j)
				r_j = Gtd[j] - dot(GtG[j, :], x) + GtG[j, j] * x[j]
				
				# Soft thresholding update
				x[j] = soft_threshold(r_j, λ) / GtG[j, j]
			end
			
			push!(history, copy(x))
			
			# Compute objective
			residual = G * x - d
			obj = dot(residual, residual) + λ * sum(abs.(x))
			push!(objectives, obj)
			
			# Check convergence
			if norm(x - x_old) < tol
				break
			end
		end
		
		return x, history, objectives
	end
end

# ╔═╡ a33f123e-fb9d-11f0-92cc-7f321875af91
begin
	# Solve with Ridge
	x_ridge, hist_ridge, obj_ridge = ridge_coordinate_descent(
		G, d_obs, lambda_ridge, max_iter=max_iter
	)
	
	# Solve with LASSO
	x_lasso, hist_lasso, obj_lasso = lasso_coordinate_descent(
		G, d_obs, lambda_lasso, max_iter=max_iter
	)
	
	# Least squares (no regularization) for comparison
	x_ls = G \ d_obs
end

# ╔═╡ a33f1626-fb9d-11f0-b93e-9317b0073a06
begin
	t_source = 1:n_source
	
	trace_true = scatter(
		x=t_source,
		y=x_true,
		mode="markers",
		marker=attr(size=8, color="black", symbol="circle"),
		name="True Reflectivity",
		yaxis="y1"
	)
	
	trace_ls = scatter(
		x=t_source,
		y=x_ls,
		mode="lines",
		line=attr(color="gray", width=1, dash="dot"),
		name="Least Squares (no reg)",
		yaxis="y1"
	)
	
	trace_ridge = scatter(
		x=t_source,
		y=x_ridge,
		mode="lines+markers",
		line=attr(color="blue", width=2),
		marker=attr(size=4),
		name="Ridge (L2)",
		yaxis="y1"
	)
	
	trace_lasso = scatter(
		x=t_source,
		y=x_lasso,
		mode="lines+markers",
		line=attr(color="red", width=2),
		marker=attr(size=4),
		name="LASSO (L1)",
		yaxis="y1"
	)
	
	layout_sources = Layout(
		title="Recovery: LASSO vs Ridge",
		xaxis=attr(title="Time sample"),
		yaxis=attr(title="Amplitude", domain=[0, 1]),
		showlegend=true,
		width=900,
		height=500,
		hovermode="x unified"
	)
	
	WideCell(plot([trace_true, trace_ls, trace_ridge, trace_lasso], layout_sources))
end

# ╔═╡ a33f247c-fb9d-11f0-97b2-a9d8a9be0ff4
begin
	t_data = 1:n_data
	
	d_ridge = G * x_ridge
	d_lasso = G * x_lasso
	d_ls = G * x_ls
	
	trace_obs = scatter(
		x=t_data,
		y=d_obs,
		mode="lines",
		line=attr(color="black", width=1),
		name="Observed data (noisy)",
		opacity=0.5
	)
	
	trace_true_data = scatter(
		x=t_data,
		y=d_true,
		mode="lines",
		line=attr(color="green", width=2, dash="dash"),
		name="True data (no noise)"
	)
	
	trace_ridge_data = scatter(
		x=t_data,
		y=d_ridge,
		mode="lines",
		line=attr(color="blue", width=2),
		name="Ridge fit"
	)
	
	trace_lasso_data = scatter(
		x=t_data,
		y=d_lasso,
		mode="lines",
		line=attr(color="red", width=2),
		name="LASSO fit"
	)
	
	layout_data = Layout(
		title="Data Fit Comparison",
		xaxis=attr(title="Time sample"),
		yaxis=attr(title="Amplitude"),
		showlegend=true,
		width=900,
		height=400
	)
	
	WideCell(plot([trace_obs, trace_true_data, trace_ridge_data, trace_lasso_data], layout_data))
end

# ╔═╡ a33f271a-fb9d-11f0-8673-f1f9890402b0
begin
	trace_ridge_conv = scatter(
		x=1:length(obj_ridge),
		y=obj_ridge,
		mode="lines",
		line=attr(color="blue", width=2),
		name="Ridge objective"
	)
	
	trace_lasso_conv = scatter(
		x=1:length(obj_lasso),
		y=obj_lasso,
		mode="lines",
		line=attr(color="red", width=2),
		name="LASSO objective"
	)
	
	layout_conv = Layout(
		title="Objective Function Convergence",
		xaxis=attr(title="Iteration"),
		yaxis=attr(title="Objective value", type="log"),
		showlegend=true,
		width=800,
		height=400
	)
	
	plot([trace_ridge_conv, trace_lasso_conv], layout_conv)
end

# ╔═╡ a33f13ec-fb9d-11f0-8759-79b1e55198ad
md"""
**Results:**
- Ridge: $(sum(abs.(x_ridge) .> 0.01)) non-zero coefficients
- LASSO: $(sum(abs.(x_lasso) .> 0.01)) non-zero coefficients ($(round(100*sum(abs.(x_lasso) .> 0.01)/n_source, digits=1))% sparse)
- True: $(sum(abs.(x_true) .> 0.01)) non-zero coefficients
- Ridge converged in $(length(obj_ridge)) iterations
- LASSO converged in $(length(obj_lasso)) iterations
"""

# ╔═╡ a33f15b8-fb9d-11f0-957b-4ddb23071d7c
md"## Comparison: True Source vs Recovered Sources"

# ╔═╡ a33f26d4-fb9d-11f0-83aa-2b6f8b236fb9
md"## Convergence Analysis"

# ╔═╡ a33f2878-fb9d-11f0-998d-0d2568c583dc
md"## Error Metrics & Sparsity"

# ╔═╡ a33f28c8-fb9d-11f0-a364-777c9750987c
begin
	# Compute errors
	error_ridge = norm(x_ridge - x_true)
	error_lasso = norm(x_lasso - x_true)
	error_ls = norm(x_ls - x_true)
	
	# Data fit errors
	data_error_ridge = norm(G * x_ridge - d_obs)
	data_error_lasso = norm(G * x_lasso - d_obs)
	data_error_ls = norm(G * x_ls - d_obs)
	
	# Sparsity measures
	sparsity_ridge = sum(abs.(x_ridge) .> 0.01)
	sparsity_lasso = sum(abs.(x_lasso) .> 0.01)
	sparsity_true = sum(abs.(x_true) .> 0.01)
	
	md"""
	### Solution Quality
	
	| Method | Reflectivity Error | Data Misfit | Non-zeros | L1 Norm | L2 Norm |
	|--------|--------------|-------------|-----------|---------|---------|
	| **True** | 0.0 | $(round(norm(d_true - d_obs), digits=3)) | $sparsity_true | $(round(sum(abs.(x_true)), digits=2)) | $(round(norm(x_true), digits=2)) |
	| **Least Squares** | $(round(error_ls, digits=3)) | $(round(data_error_ls, digits=3)) | $(sum(abs.(x_ls) .> 0.01)) | $(round(sum(abs.(x_ls)), digits=2)) | $(round(norm(x_ls), digits=2)) |
	| **Ridge (L2)** | $(round(error_ridge, digits=3)) | $(round(data_error_ridge, digits=3)) | $sparsity_ridge | $(round(sum(abs.(x_ridge)), digits=2)) | $(round(norm(x_ridge), digits=2)) |
	| **LASSO (L1)** | $(round(error_lasso, digits=3)) | $(round(data_error_lasso, digits=3)) | $sparsity_lasso | $(round(sum(abs.(x_lasso)), digits=2)) | $(round(norm(x_lasso), digits=2)) |
	
	**Key Observations:**
	- LASSO achieves sparsity $(sparsity_lasso) vs true $(sparsity_true)
	- Ridge spreads energy across all coefficients
	- Both regularize better than unregularized least squares
	"""
end

# ╔═╡ a33f2c2e-fb9d-11f0-9b1c-29d3513c1f23
md"## Regularization Path"

# ╔═╡ a33f2c6a-fb9d-11f0-9f26-59f1211c7d20
begin
	# Compute solutions for different λ values
	λ_values = 10 .^ range(-4, -1, length=30)
	
	ridge_paths = []
	lasso_paths = []
	
	for λ in λ_values
		x_r, _, _ = ridge_coordinate_descent(G, d_obs, λ, max_iter=100)
		x_l, _, _ = lasso_coordinate_descent(G, d_obs, λ, max_iter=100)
		push!(ridge_paths, x_r)
		push!(lasso_paths, x_l)
	end
	
	# Plot paths for first few coefficients
	traces_ridge = []
	traces_lasso = []
	
	# Select a few representative coefficients to plot
	coef_indices = findall(x -> abs(x) > 0.1, x_true)
	if length(coef_indices) > 5
		coef_indices = coef_indices[1:5]
	end
	
	colors = ["blue", "red", "green", "purple", "orange"]
	
	for (i, idx) in enumerate(coef_indices)
		ridge_vals = [path[idx] for path in ridge_paths]
		lasso_vals = [path[idx] for path in lasso_paths]
		
		push!(traces_ridge, scatter(
			x=log10.(λ_values),
			y=ridge_vals,
			mode="lines",
			line=attr(color=colors[i], width=2),
			name="Coef $idx (Ridge)",
			showlegend=(i<=3)
		))
		
		push!(traces_lasso, scatter(
			x=log10.(λ_values),
			y=lasso_vals,
			mode="lines",
			line=attr(color=colors[i], width=2, dash="dash"),
			name="Coef $idx (LASSO)",
			showlegend=(i<=3)
		))
	end
	
	layout_path = Layout(
		title="Regularization Path: Coefficient Values vs λ",
		xaxis=attr(title="log₁₀(λ)"),
		yaxis=attr(title="Coefficient value"),
		showlegend=true,
		width=900,
		height=500
	)
	
	plot([traces_ridge..., traces_lasso...], layout_path)
end

# ╔═╡ a33f2fd0-fb9d-11f0-bc64-35d0eab46c26
md"## Key Takeaways"

# ╔═╡ a33f3016-fb9d-11f0-8357-5b221c052383
md"""
### Ridge Regression (L2 Regularization)
✓ **Smooth solutions**: All coefficients shrink proportionally  
✓ **Stable**: Less sensitive to noise  
✓ **Fast**: Simple closed-form or iterative solution  
✗ **Not sparse**: Cannot zero out irrelevant features  
✗ **Over-smooths**: Loses sharp features in sparse signals

**Best for:** Smooth signals, when all features matter

### LASSO Regression (L1 Regularization)
✓ **Sparse solutions**: Many coefficients become exactly zero  
✓ **Feature selection**: Automatically identifies important components  
✓ **Interpretable**: Few non-zero coefficients  
✗ **Sensitive to λ**: Requires careful tuning  
✗ **Biased**: Shrinks large coefficients too much

**Best for:** Sparse signals (like seismic sources), high-dimensional problems

### Coordinate Descent
✓ **Simple**: Update one variable at a time  
✓ **Efficient**: No matrix inversions needed  
✓ **Flexible**: Works for both L1 and L2  
✓ **Scalable**: Good for large problems  

**Algorithm:**
1. Initialize x = 0
2. For each coordinate j:
   - Fix all other coordinates
   - Update x[j] to minimize objective
3. Repeat until convergence

### Soft Thresholding (LASSO key)
```
soft_threshold(z, λ) = sign(z) × max(|z| - λ, 0)
```
This operator creates sparsity by:
- Setting small values to exactly zero
- Shrinking large values by λ

### Seismic Deconvolution Application
The earth's reflectivity is naturally **sparse** (few discrete reflectors). LASSO exploits this structure to recover sharp reflectivity spikes from seismic data convolved with the known airgun signature, while Ridge would over-smooth them.
"""

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
# ╟─a33ef9ac-fb9d-11f0-aef3-2738233020c4
# ╟─a33efdbc-fb9d-11f0-a81d-9ffc02850aa2
# ╟─a33efe34-fb9d-11f0-b061-a349c8b76450
# ╟─a33f1626-fb9d-11f0-b93e-9317b0073a06
# ╟─a33f247c-fb9d-11f0-97b2-a9d8a9be0ff4
# ╟─a33f271a-fb9d-11f0-8673-f1f9890402b0
# ╟─0aa0b3d5-e785-41b0-9193-fbe268c98c7e
# ╠═a33f010e-fb9d-11f0-bd8b-0b62c4571bc7
# ╠═a33f0184-fb9d-11f0-a575-71dd14204fdb
# ╠═a33f03fc-fb9d-11f0-88e7-f57fc49ab40a
# ╠═a33f05a0-fb9d-11f0-91e8-5b072b9bd2ea
# ╠═a33f0796-fb9d-11f0-a746-6b699aacd7b6
# ╠═a33f08ac-fb9d-11f0-88ca-1f136753c450
# ╠═a33f0a28-fb9d-11f0-a0e9-6581714db953
# ╠═a33f0a98-fb9d-11f0-8a16-4b523bba358f
# ╠═a33f0e2e-fb9d-11f0-973b-995b38420393
# ╠═a33f123e-fb9d-11f0-92cc-7f321875af91
# ╠═a33f13ec-fb9d-11f0-8759-79b1e55198ad
# ╟─a33f15b8-fb9d-11f0-957b-4ddb23071d7c
# ╠═a33f26d4-fb9d-11f0-83aa-2b6f8b236fb9
# ╟─a33f2878-fb9d-11f0-998d-0d2568c583dc
# ╠═a33f28c8-fb9d-11f0-a364-777c9750987c
# ╠═a33f2c2e-fb9d-11f0-9b1c-29d3513c1f23
# ╠═a33f2c6a-fb9d-11f0-9f26-59f1211c7d20
# ╠═a33f2fd0-fb9d-11f0-bc64-35d0eab46c26
# ╠═a33f3016-fb9d-11f0-8357-5b221c052383
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
