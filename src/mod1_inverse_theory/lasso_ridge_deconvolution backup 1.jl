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
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el)
        el
    end
    #! format: on
end

# ╔═╡ 499b6d53-97a2-4ba8-a6d6-030ce1e6fa43
using PlutoPlotly, LinearAlgebra, Distributions, PlutoUI

# ╔═╡ c8945e8c-1425-4198-bf0f-f25bf48f79d4
md"# LASSO vs Ridge Regression: Seismic Source Deconvolution"

# ╔═╡ intro_text
md"""
## The Deconvolution Problem

In seismic exploration, an **airgun source** fires and creates a wavelet that propagates through the earth. The recorded data is a **convolution** of:
- The source time function (what we want to recover)
- The earth's impulse response (reflectivity)

**Goal:** Given noisy recorded data, recover the sparse source signal.

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

# ╔═╡ controls_section
md"## Interactive Controls"

# ╔═╡ controls_md
md"""
**Problem Setup:**

Source sparsity: $(@bind sparsity Slider(0.1:0.05:0.5, show_value=true, default=0.2))

Wavelet width: $(@bind wavelet_width Slider(3:10, show_value=true, default=5))

Noise level (%): $(@bind noise_percent Slider(0:1:30, show_value=true, default=10))

---

**Ridge Regression (L2):**

λ₂ (Ridge): $(@bind lambda_ridge Slider(0:0.001:0.1, show_value=true, default=0.01))

---

**LASSO Regression (L1):**

λ₁ (LASSO): $(@bind lambda_lasso Slider(0:0.001:0.1, show_value=true, default=0.02))

---

Max iterations: $(@bind max_iter Slider(50:50:500, show_value=true, default=200))

$(@bind resample Button("Resample Data"))
"""

# ╔═╡ problem_setup
md"## Problem Setup: Forward Model"

# ╔═╡ generate_true_source
begin
	resample
	
	# Problem dimensions
	n_source = 100  # Source time samples
	n_data = 120    # Data samples (longer due to convolution)
	
	# Generate true sparse source (impulses)
	function generate_sparse_source(n, sparsity)
		x_true = zeros(n)
		n_spikes = max(1, round(Int, n * sparsity))
		spike_positions = rand(1:n, n_spikes)
		spike_amplitudes = randn(n_spikes) .* 2 .+ 1
		for (pos, amp) in zip(spike_positions, spike_amplitudes)
			x_true[pos] = amp
		end
		return x_true
	end
	
	x_true = generate_sparse_source(n_source, sparsity)
end

# ╔═╡ generate_wavelet
begin
	# Generate Ricker wavelet (Mexican hat)
	function ricker_wavelet(n, width)
		t = collect(1:n) .- (n/2)
		σ = width / 2.355  # Convert width to std dev
		w = (1 .- (t/σ).^2) .* exp.(-(t.^2)/(2*σ^2))
		return w / maximum(abs.(w))
	end
	
	wavelet = ricker_wavelet(2*wavelet_width+1, wavelet_width)
end

# ╔═╡ build_forward_operator
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

# ╔═╡ generate_data
begin
	# Generate observed data with noise
	d_true = G * x_true
	noise_level = noise_percent / 100.0 * norm(d_true)
	noise = randn(n_data) * noise_level
	d_obs = d_true + noise
end

# ╔═╡ display_problem
md"""
**Problem dimensions:**
- Source length: $n_source samples
- Data length: $n_data samples
- Wavelet length: $(length(wavelet)) samples
- True sparsity: $(sum(abs.(x_true) .> 0.01)) / $n_source spikes
- SNR: $(round(20*log10(norm(d_true)/norm(noise)), digits=1)) dB
"""

# ╔═╡ algorithms_section
md"## Coordinate Descent Algorithms"

# ╔═╡ ridge_coordinate_descent
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

# ╔═╡ lasso_coordinate_descent
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

# ╔═╡ solve_problems
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

# ╔═╡ display_results
md"""
**Results:**
- Ridge: $(sum(abs.(x_ridge) .> 0.01)) non-zero coefficients
- LASSO: $(sum(abs.(x_lasso) .> 0.01)) non-zero coefficients ($(round(100*sum(abs.(x_lasso) .> 0.01)/n_source, digits=1))% sparse)
- True: $(sum(abs.(x_true) .> 0.01)) non-zero coefficients
- Ridge converged in $(length(obj_ridge)) iterations
- LASSO converged in $(length(obj_lasso)) iterations
"""

# ╔═╡ visualization_section
md"## Comparison: True Source vs Recovered Sources"

# ╔═╡ plot_sources
begin
	t_source = 1:n_source
	
	trace_true = scatter(
		x=t_source,
		y=x_true,
		mode="markers",
		marker=attr(size=8, color="black", symbol="circle"),
		name="True Source",
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
		title="Source Recovery: LASSO vs Ridge",
		xaxis=attr(title="Time sample"),
		yaxis=attr(title="Amplitude", domain=[0, 1]),
		showlegend=true,
		width=900,
		height=500,
		hovermode="x unified"
	)
	
	plot([trace_true, trace_ls, trace_ridge, trace_lasso], layout_sources)
end

# ╔═╡ plot_data_fit
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
	
	plot([trace_obs, trace_true_data, trace_ridge_data, trace_lasso_data], layout_data)
end

# ╔═╡ convergence_section
md"## Convergence Analysis"

# ╔═╡ plot_convergence
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

# ╔═╡ error_metrics_section
md"## Error Metrics & Sparsity"

# ╔═╡ compute_metrics
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
	
	| Method | Source Error | Data Misfit | Non-zeros | L1 Norm | L2 Norm |
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

# ╔═╡ regularization_path_section
md"## Regularization Path"

# ╔═╡ plot_reg_path
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

# ╔═╡ interpretation_section
md"## Key Takeaways"

# ╔═╡ interpretation_text
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
The airgun source is naturally **sparse in time** (few impulses). LASSO exploits this structure to recover sharp source pulses, while Ridge would over-smooth them.
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
# ╟─problem_setup
# ╠═generate_true_source
# ╠═generate_wavelet
# ╠═build_forward_operator
# ╠═generate_data
# ╟─display_problem
# ╟─algorithms_section
# ╠═ridge_coordinate_descent
# ╠═lasso_coordinate_descent
# ╠═solve_problems
# ╟─display_results
# ╟─visualization_section
# ╟─plot_sources
# ╟─plot_data_fit
# ╟─convergence_section
# ╟─plot_convergence
# ╟─error_metrics_section
# ╟─compute_metrics
# ╟─regularization_path_section
# ╟─plot_reg_path
# ╟─interpretation_section
# ╟─interpretation_text
# ╠═499b6d53-97a2-4ba8-a6d6-030ce1e6fa43
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
