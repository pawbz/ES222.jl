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

# ╔═╡ f580a505-9db0-4652-9b8e-58a60d583219
TableOfContents()

# ╔═╡ c8945e8c-1425-4198-bf0f-f25bf48f79d4
md"# Gradient Vector and Hessian Matrix"

# ╔═╡ 02518918-f5e6-11f0-b392-0b817a3ef5eb
md"""
This notebook demonstrates the geometry of inverse problems, showing how the gradient vector and Hessian matrix relate to the cost function landscape. We'll visualize:

- **Gradient vectors**: Point in the direction of steepest ascent
- **Hessian matrix**: Describes the curvature of the cost function
- **Forward operator G**: Maps parameters to data
- **Tikhonov regularization**: Adds smoothness to solutions
"""

# ╔═╡ 02518ca6-f5e6-11f0-9304-2ba5c35b7cfd
md"## Interactive Controls"

# ╔═╡ f69ffb70-0136-4745-b0b9-233ae994fbb1
md"""
**Forward Operator G Parameters:**

Scale of column 1: $(@bind scale1 Slider(0.1:0.1:3.0, show_value=true, default=1.0))

Scale of column 2: $(@bind scale2 Slider(0.1:0.1:3.0, show_value=true, default=1.0))

Angle between columns (degrees): $(@bind angle_deg Slider(0:5:180, show_value=true, default=90))


Noise percentage: $(@bind noise_percent Slider(0:1:50, show_value=true, default=10))

---

**Regularization:**

Tikhonov parameter (λ): $(@bind lambda Slider(0:0.01:1.0, show_value=true, default=0.0))

---

$(@bind resample_data Button("Resample Data"))
"""

# ╔═╡ 02519016-f5e6-11f0-a96f-c96317f6e868
md"## True Model and Data Generation"

# ╔═╡ 02519084-f5e6-11f0-a437-15fd40aedcb3
m_true = [1.0, 1.0]  # True solution

# ╔═╡ 025190e8-f5e6-11f0-b104-11322479d181
begin
	# Convert angle to radians
	angle_rad = deg2rad(angle_deg)
	
	# Construct forward operator G (3x2 matrix)
	# Column 1: scaled vector in direction [1, 0, 0]
	col1 = scale1 * [1.0, 0.0001]
	
	# Column 2: scaled vector at specified angle from column 1
	# We rotate in the 2D subspace of the first two rows
	col2 = scale2 * [cos(angle_rad), sin(angle_rad)]
	
	G = hcat(col1, col2)
end

# ╔═╡ 0251925a-f5e6-11f0-8dfe-199e308ed3b8
begin
	resample_data  # Reactive to button
	
	# Generate synthetic data with noise
	d_true = G * m_true
	noise_level = noise_percent / 100.0 * norm(d_true)  # Noise as percentage of data norm
	noise = randn(size(d_true)...) * noise_level
	d_obs = d_true + noise
end

# ╔═╡ 025194a6-f5e6-11f0-ab61-b7b715aef91b
md"## Cost Function and Gradient"

# ╔═╡ 02519520-f5e6-11f0-ac4e-f52d91eca6b5
begin
	# Cost function: J(m) = ||Gm - d||^2 + λ||m||^2
	function cost_function(m, G, d, λ=0.0)
		residual = G * m - d
		data_misfit = dot(residual, residual)
		regularization = λ * dot(m, m)
		return data_misfit + regularization
	end
	
	# Gradient: ∇J(m) = 2G^T(Gm - d) + 2λm
	function gradient(m, G, d, λ=0.0)
		return 2 * G' * (G * m - d) + 2 * λ * m
	end
	
	# Hessian: H = 2G^TG + 2λI
	function hessian(G, λ=0.0)
		n = size(G, 2)
		return 2 * (G' * G + λ * I(n))
	end
end

# ╔═╡ 025196ce-f5e6-11f0-aa8a-7d2c3259b18e
H = hessian(G, lambda)

# ╔═╡ 02519732-f5e6-11f0-bf1e-b93070e09840
md"""
**Hessian Matrix H:**
```
$(round.(H, digits=3))
```

**Eigenvalues of H:** $(round.(eigvals(H), digits=3))

**Condition number:** $(round(cond(H), digits=2))
"""

# ╔═╡ 02519958-f5e6-11f0-a835-6b409af2f7ca
begin
	# Analytical solution: m_est = (G^TG + λI)^(-1)G^Td
	GTG = G' * G
	if lambda > 0
		m_est = inv(GTG + lambda * I(2)) * G' * d_obs
	else
		m_est = inv(GTG) * G' * d_obs
	end
end

# ╔═╡ 02519a7a-f5e6-11f0-8cfc-c7ec89e1eedd
md"""
**Estimated solution m_est:** $(round.(m_est, digits=3))

**True solution m_true:** $(round.(m_true, digits=3))

**Estimation error:** $(round(norm(m_est - m_true), digits=4))
"""

# ╔═╡ 02519bd8-f5e6-11f0-8fe4-55964d3b6646
md"## Cost Function Landscape with Gradients"

# ╔═╡ 02519c3c-f5e6-11f0-91fc-65f601a31b86
begin
	# Create a grid for contour plot
	m1_range = range(-0.5, 2.5, length=100)
	m2_range = range(-0.5, 2.5, length=100)
	
	# Compute cost function on grid
	J_grid = [cost_function([m1, m2], G, d_obs, lambda) 
	          for m2 in m2_range, m1 in m1_range]
end

# ╔═╡ 0251a696-f5e6-11f0-a010-975bd9b29476
begin
	# The Hessian describes the curvature around the minimum
	# Error ellipse is determined by eigenvalues/eigenvectors of H
	
	eig_result = eigen(H)
	eigenvalues = eig_result.values
	eigenvectors = eig_result.vectors
	
	# Generate ellipse points
	θ = range(0, 2π, length=100)
	
	# Ellipse in eigenspace (using inverse of sqrt of eigenvalues as radii)
	ellipse_eigen = hcat(
		cos.(θ) ./ sqrt(eigenvalues[1]),
		sin.(θ) ./ sqrt(eigenvalues[2])
	)
	
	# Rotate to original space
	ellipse_points = (eigenvectors * ellipse_eigen')'
	
	# Center at estimated solution and scale for visibility
	scale_factor = 0.5
	ellipse_x = m_est[1] .+ scale_factor * ellipse_points[:, 1]
	ellipse_y = m_est[2] .+ scale_factor * ellipse_points[:, 2]
	
	# Plot eigenvector directions (scaled by eigenvalues)
	# Normalize eigenvalues to reasonable plotting scale
	max_eigenvalue = maximum(eigenvalues)
	scale_by_eigs = 0.3 / sqrt(max_eigenvalue)
	
	eigen_trace1 = scatter(
		x=[m_est[1], m_est[1] + scale_by_eigs * sqrt(eigenvalues[1]) * eigenvectors[1, 1]],
		y=[m_est[2], m_est[2] + scale_by_eigs * sqrt(eigenvalues[1]) * eigenvectors[2, 1]],
		mode="lines+markers",
		line=attr(color="orange", width=3),
		marker=attr(size=8),
		name="Eigenvector 1 (λ=$(round(eigenvalues[1], digits=2)))"
	)
	
	eigen_trace2 = scatter(
		x=[m_est[1], m_est[1] + scale_by_eigs * sqrt(eigenvalues[2]) * eigenvectors[1, 2]],
		y=[m_est[2], m_est[2] + scale_by_eigs * sqrt(eigenvalues[2]) * eigenvectors[2, 2]],
		mode="lines+markers",
		line=attr(color="purple", width=3),
		marker=attr(size=8),
		name="Eigenvector 2 (λ=$(round(eigenvalues[2], digits=2)))"
	)
	
	# Contour plot (lighter)
	contour_trace2 = contour(
		x=collect(m1_range),
		y=collect(m2_range),
		z=J_grid,
		colorscale="Greys",
		showscale=false,
		contours=attr(
			showlabels=false
		),
		opacity=0.4,
		name="Cost Function",
		ncontours=15
	)
	
	# Error ellipse
	ellipse_trace = scatter(
		x=ellipse_x,
		y=ellipse_y,
		mode="lines",
		line=attr(color="red", width=3, dash="dash"),
		name="Error Ellipse"
	)
	
	# Solutions
	true_trace2 = scatter(
		x=[m_true[1]],
		y=[m_true[2]],
		mode="markers",
		marker=attr(size=15, color="white", symbol="star", line=attr(color="black", width=2)),
		name="True Solution"
	)
	
	est_trace2 = scatter(
		x=[m_est[1]],
		y=[m_est[2]],
		mode="markers",
		marker=attr(size=12, color="lime", symbol="x", line=attr(color="black", width=2)),
		name="Estimated Solution"
	)
	
	layout2 = Layout(
		title="Hessian Eigenvectors and Error Ellipse",
		xaxis=attr(title="Parameter m₁", range=(-0.5, 2.5), scaleanchor="y"),
		yaxis=attr(title="Parameter m₂", range=(-0.5, 2.5), scaleanchor="x"),
		width=700,
		height=700,
		showlegend=true
	)
	
	plot([contour_trace2, ellipse_trace, eigen_trace1, eigen_trace2, true_trace2, est_trace2], layout2)
end

# ╔═╡ 02519d7c-f5e6-11f0-8a36-773c8d5b8227
begin
	# Sample points for gradient vectors
	n_samples = 10
	m1_samples = range(0, 2, length=n_samples)
	m2_samples = range(0, 2, length=n_samples)
	
	grad_points = []
	grad_vectors = []
	
	for m1 in m1_samples
		for m2 in m2_samples
			m = [m1, m2]
			grad = gradient(m, G, d_obs, lambda)
			push!(grad_points, m)
			push!(grad_vectors, grad)
		end
	end
end

# ╔═╡ 02519eda-f5e6-11f0-b79d-01896d55555e
let
	# Create contour plot
	contour_trace = contour(
		x=collect(m1_range),
		y=collect(m2_range),
		z=J_grid,
		colorscale="Viridis",
		contours=attr(
			showlabels=true,
			labelfont=attr(size=10, color="white")
		),
		name="Objective Function",
		ncontours=20
	)
	
	# Plot gradient vectors (as arrows)
	arrow_x = Float64[]
	arrow_y = Float64[]
	
	# Find max gradient magnitude for scaling
	max_grad_magnitude = maximum([norm(g) for g in grad_vectors if norm(g) > 1e-6])
	
	for i in 1:length(grad_points)
		m = grad_points[i]
		grad = grad_vectors[i]
		
		# Get gradient magnitude
		grad_norm = norm(grad)
		if grad_norm > 1e-6
			# Scale arrow by gradient magnitude (negative for descent direction)
			# Normalize by max magnitude to keep reasonable scale
			grad_scaled = -0.25 * (grad_norm / max_grad_magnitude) * grad / grad_norm
			
			# Add line segment (with NaN to separate arrows)
			push!(arrow_x, m[1])
			push!(arrow_x, m[1] + grad_scaled[1])
			push!(arrow_x, NaN)
			
			push!(arrow_y, m[2])
			push!(arrow_y, m[2] + grad_scaled[2])
			push!(arrow_y, NaN)
		end
	end
	
	# Create single arrow trace with connected line segments
	arrow_trace = scatter(
		x=arrow_x,
		y=arrow_y,
		mode="lines+markers",
		line=attr(color="red", width=2),
		marker=attr(size=5, color="red"),
		showlegend=false,
		hoverinfo="skip"
	)
	
	# Plot true solution
	true_trace = scatter(
		x=[m_true[1]],
		y=[m_true[2]],
		mode="markers",
		marker=attr(size=15, color="white", symbol="star", line=attr(color="black", width=2)),
		name="True Solution"
	)
	
	# Plot estimated solution
	est_trace = scatter(
		x=[m_est[1]],
		y=[m_est[2]],
		mode="markers",
		marker=attr(size=12, color="lime", symbol="x", line=attr(color="black", width=2)),
		name="Estimated Solution"
	)
	
	# Combine all traces
	all_traces = [contour_trace, arrow_trace, true_trace, est_trace]
	
	layout = Layout(
		title="Objective Function Landscape with Gradient Vectors<br>(Red arrows point in steepest descent direction)",
		xaxis=attr(title="Parameter z₁", range=(-0.5, 2.5), scaleanchor="y"),
		yaxis=attr(title="Parameter z₂", range=(-0.5, 2.5), scaleanchor="x"),
		width=700,
		height=700,
		showlegend=true
	)
	
	plot(all_traces, layout)
end

# ╔═╡ 0251a62a-f5e6-11f0-8964-bbb596619344
md"## Error Ellipse (Hessian Geometry)"

# ╔═╡ 0251ae5c-f5e6-11f0-a6a1-a34afc3316dc
md"## Bias-Variance Trade-off"

# ╔═╡ 0251af06-f5e6-11f0-a0b0-536714bf0a31
begin
	# Compute bias and variance for different λ values
	lambda_values = 10 .^ range(-3, 1, length=50)
	
	n_trials = 100
	biases = Float64[]
	variances = Float64[]
	
	for λ_test in lambda_values
		estimates = []
		
		# Monte Carlo simulation
		for _ in 1:n_trials
			# Generate noisy data
			noise_trial = randn(2) * noise_level
			d_trial = d_true + noise_trial
			
			# Estimate solution
			GTG_trial = G' * G
			m_trial = inv(GTG_trial + λ_test * I(2)) * G' * d_trial
			push!(estimates, m_trial)
		end
		
		# Compute bias and variance
		mean_estimate = mean(estimates)
		bias = norm(mean_estimate - m_true)
		variance = mean([norm(est - mean_estimate)^2 for est in estimates])
		
		push!(biases, bias)
		push!(variances, variance)
	end
	
	mse = biases.^2 .+ variances
end

# ╔═╡ 0251b1c2-f5e6-11f0-a1fa-054cd04c5e2e
begin
	bias_trace = scatter(
		x=log10.(lambda_values),
		y=biases.^2,
		mode="lines",
		line=attr(color="blue", width=3),
		name="Bias²"
	)
	
	variance_trace = scatter(
		x=log10.(lambda_values),
		y=variances,
		mode="lines",
		line=attr(color="red", width=3),
		name="Variance"
	)
	
	mse_trace = scatter(
		x=log10.(lambda_values),
		y=mse,
		mode="lines",
		line=attr(color="green", width=3, dash="dash"),
		name="MSE (Bias² + Variance)"
	)
	
	# Current lambda
	current_lambda_trace = scatter(
		x=[log10(max(lambda, 1e-3))],
		y=[0],
		mode="markers",
		marker=attr(size=15, color="orange", symbol="diamond"),
		name="Current λ"
	)
	
	layout3 = Layout(
		title="Bias-Variance Trade-off",
		xaxis=attr(title="log₁₀(λ) - Regularization Parameter"),
		yaxis=attr(title="Error", type="log"),
		showlegend=true,
		width=700,
		height=500
	)
	
	plot([bias_trace, variance_trace, mse_trace, current_lambda_trace], layout3)
end

# ╔═╡ 0251b4e2-f5e6-11f0-b32a-55fb3c9da24d
md"""
## Interpretation

### Gradient Vectors
- Point in the direction of **steepest ascent** of the cost function
- The **negative gradient** points toward the minimum (steepest descent)
- Longer arrows indicate steeper slopes

### Hessian Matrix
- Describes the **curvature** of the cost function
- Eigenvalues indicate the **strength of curvature** in each direction
- Eigenvectors show the **principal directions** of curvature
- Large condition number → **ill-conditioned** problem (sensitive to noise)

### Error Ellipse
- Shows the **uncertainty** in parameter estimates
- Elongated ellipse → high correlation between parameters
- Circular ellipse → independent parameters

### Tikhonov Regularization (λ)
- **λ = 0**: No regularization, minimum bias but high variance
- **λ > 0**: Adds penalty on model norm, increases bias but reduces variance
- Optimal λ balances bias and variance to minimize total error

### Forward Operator Geometry
- **Angle ≈ 90°**: Columns nearly orthogonal → well-conditioned
- **Angle ≈ 0° or 180°**: Columns nearly parallel → ill-conditioned
- **Different scales**: Affects sensitivity to each parameter
"""

# ╔═╡ 8b9e36fa-f5ee-11f0-827c-53e1737fd864
md"## Residual Backpropagation (Data → Model)"

# ╔═╡ b521018d-2c7f-4404-8e0a-e22fe2f51b50
md"""
Residual angle w.r.t. G₁ (degrees): $(@bind residual_angle Slider(0:5:180, show_value=true, default=45))

Residual magnitude (data-space): $(@bind residual_mag Slider(0:0.05:2.0, show_value=true, default=0.5))
"""

# ╔═╡ 8b9e37b8-f5ee-11f0-a32c-63d48c6cc765
begin
	# Build a residual with user-chosen angle and magnitude relative to column 1 of G
	g1 = G[:, 1]
	g1_norm = norm(g1)
	g1_hat = g1_norm > 0 ? g1 / g1_norm : [1.0, 0.0]
	g1_perp = [-g1_hat[2], g1_hat[1]]
	
	θ_res = deg2rad(residual_angle)
	residual_demo = residual_mag * (cos(θ_res) * g1_hat + sin(θ_res) * g1_perp)
	
	# Backpropagate residual to model space (ignoring regularization term for clarity)
	backprop_grad = 2 .* (G' * residual_demo)
end

# ╔═╡ 8b9e3998-f5ee-11f0-b8cb-67dbbd566b57
begin
	# Helper to build arrow traces
	function arrow_trace(vec; label, color, xaxis="x", yaxis="y")
		scatter(
			x=[0, vec[1]],
			y=[0, vec[2]],
			mode="lines+markers",
			line=attr(color=color, width=3),
			marker=attr(size=8, color=color),
			name=label,
			xaxis=xaxis,
			yaxis=yaxis
		)
	end

	data_vectors = [g1_hat, residual_demo]
	data_span = maximum(abs, vcat([abs.(v) for v in data_vectors]...))
	data_span = max(data_span * 1.2, 1.0)

	model_vectors = [backprop_grad]
	model_span = maximum(abs, vcat([abs.(v) for v in model_vectors]...))
	model_span = max(model_span * 1.2, 1.0)

	traces = [
		arrow_trace(g1_hat; label="G₁ direction", color="gray"),
		arrow_trace(residual_demo; label="Residual r", color="red"),
		arrow_trace(backprop_grad; label="Backpropagated gradient 2Gᵀr", color="blue", xaxis="x2", yaxis="y2")
	]

	layout_bp = Layout(
		title="Residual Backpropagation: Data → Model Space",
		grid=attr(rows=1, columns=2, pattern="independent"),
		xaxis=attr(title="Data space d₁", range=(-data_span, data_span), scaleanchor="y"),
		yaxis=attr(title="d₂", range=(-data_span, data_span), scaleanchor="x"),
		xaxis2=attr(title="Model space m₁", range=(-model_span, model_span), scaleanchor="y2"),
		yaxis2=attr(title="m₂", range=(-model_span, model_span), scaleanchor="x2"),
		showlegend=true,
		width=900,
		height=450
	)

	plot(traces, layout_bp)
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
PlutoPlotly = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"

[compat]
Distributions = "~0.25.122"
PlutoPlotly = "~0.6.5"
PlutoUI = "~0.7.78"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.4"
manifest_format = "2.0"
project_hash = "a81188fdcd55a38d2e78cb2842d12c617d6bed01"

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
git-tree-sha1 = "3bc002af51045ca3b47d2e1787d6ce02e68b943a"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.122"

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
git-tree-sha1 = "5bfcd42851cf2f1b303f51525a54dc5e98d408a3"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.15.0"
weakdeps = ["PDMats", "SparseArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

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
git-tree-sha1 = "6122f9423393a2294e26a4efdf44960c5f8acb70"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.78"

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
# ╟─f580a505-9db0-4652-9b8e-58a60d583219
# ╟─c8945e8c-1425-4198-bf0f-f25bf48f79d4
# ╠═499b6d53-97a2-4ba8-a6d6-030ce1e6fa43
# ╟─02518918-f5e6-11f0-b392-0b817a3ef5eb
# ╟─02518ca6-f5e6-11f0-9304-2ba5c35b7cfd
# ╟─f69ffb70-0136-4745-b0b9-233ae994fbb1
# ╟─02519eda-f5e6-11f0-b79d-01896d55555e
# ╟─0251a696-f5e6-11f0-a010-975bd9b29476
# ╟─02519016-f5e6-11f0-a96f-c96317f6e868
# ╠═02519084-f5e6-11f0-a437-15fd40aedcb3
# ╠═025190e8-f5e6-11f0-b104-11322479d181
# ╠═0251925a-f5e6-11f0-8dfe-199e308ed3b8
# ╟─025194a6-f5e6-11f0-ab61-b7b715aef91b
# ╠═02519520-f5e6-11f0-ac4e-f52d91eca6b5
# ╠═025196ce-f5e6-11f0-aa8a-7d2c3259b18e
# ╠═02519732-f5e6-11f0-bf1e-b93070e09840
# ╠═02519958-f5e6-11f0-a835-6b409af2f7ca
# ╠═02519a7a-f5e6-11f0-8cfc-c7ec89e1eedd
# ╠═02519bd8-f5e6-11f0-8fe4-55964d3b6646
# ╠═02519c3c-f5e6-11f0-91fc-65f601a31b86
# ╠═02519d7c-f5e6-11f0-8a36-773c8d5b8227
# ╠═0251a62a-f5e6-11f0-8964-bbb596619344
# ╠═0251ae5c-f5e6-11f0-a6a1-a34afc3316dc
# ╠═0251af06-f5e6-11f0-a0b0-536714bf0a31
# ╠═0251b1c2-f5e6-11f0-a1fa-054cd04c5e2e
# ╠═0251b4e2-f5e6-11f0-b32a-55fb3c9da24d
# ╠═8b9e36fa-f5ee-11f0-827c-53e1737fd864
# ╠═8b9e37b8-f5ee-11f0-a32c-63d48c6cc765
# ╟─b521018d-2c7f-4404-8e0a-e22fe2f51b50
# ╠═8b9e3998-f5ee-11f0-b8cb-67dbbd566b57
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
