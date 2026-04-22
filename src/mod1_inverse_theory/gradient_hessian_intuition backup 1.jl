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
md"# Gradient Vector and Hessian Matrix Intuition"

# ╔═╡ intro_text
md"""
This notebook demonstrates the geometry of inverse problems, showing how the gradient vector and Hessian matrix relate to the cost function landscape. We'll visualize:

- **Gradient vectors**: Point in the direction of steepest ascent
- **Hessian matrix**: Describes the curvature of the cost function
- **Forward operator G**: Maps parameters to data
- **Tikhonov regularization**: Adds smoothness to solutions
"""

# ╔═╡ controls_section
md"## Interactive Controls"

# ╔═╡ f69ffb70-0136-4745-b0b9-233ae994fbb1
md"""
**Forward Operator G Parameters:**

Scale of column 1: $(@bind scale1 Slider(0.1:0.1:3.0, show_value=true, default=1.0))

Scale of column 2: $(@bind scale2 Slider(0.1:0.1:3.0, show_value=true, default=1.0))

Angle between columns (degrees): $(@bind angle_deg Slider(0:5:180, show_value=true, default=90))

---

**Regularization:**

Tikhonov parameter (λ): $(@bind lambda Slider(0:0.01:1.0, show_value=true, default=0.0))

---

$(@bind resample_data Button("Resample Data"))
"""

# ╔═╡ true_model
md"## True Model and Data Generation"

# ╔═╡ true_solution_cell
m_true = [1.0, 1.0]  # True solution

# ╔═╡ construct_G
begin
	# Convert angle to radians
	angle_rad = deg2rad(angle_deg)
	
	# Construct forward operator G (3x2 matrix)
	# Column 1: scaled vector in direction [1, 0, 0]
	col1 = scale1 * [1.0, 0.5, 0.3]
	
	# Column 2: scaled vector at specified angle from column 1
	# We rotate in the 2D subspace of the first two rows
	col2 = scale2 * [cos(angle_rad), sin(angle_rad), 0.4]
	
	G = hcat(col1, col2)
end

# ╔═╡ generate_data
begin
	resample_data  # Reactive to button
	
	# Generate synthetic data with noise
	d_true = G * m_true
	noise_level = 0.1
	noise = randn(3) * noise_level
	d_obs = d_true + noise
end

# ╔═╡ display_matrices
md"""
**Forward Operator G:**
```
$(round.(G, digits=3))
```

**True data d_true:** $(round.(d_true, digits=3))

**Observed data d_obs:** $(round.(d_obs, digits=3))
"""

# ╔═╡ cost_function_section
md"## Cost Function and Gradient"

# ╔═╡ cost_function_def
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

# ╔═╡ compute_hessian
H = hessian(G, lambda)

# ╔═╡ display_hessian
md"""
**Hessian Matrix H:**
```
$(round.(H, digits=3))
```

**Eigenvalues of H:** $(round.(eigvals(H), digits=3))

**Condition number:** $(round(cond(H), digits=2))
"""

# ╔═╡ analytical_solution
begin
	# Analytical solution: m_est = (G^TG + λI)^(-1)G^Td
	GTG = G' * G
	if lambda > 0
		m_est = inv(GTG + lambda * I(2)) * G' * d_obs
	else
		m_est = inv(GTG) * G' * d_obs
	end
end

# ╔═╡ display_solution
md"""
**Estimated solution m_est:** $(round.(m_est, digits=3))

**True solution m_true:** $(round.(m_true, digits=3))

**Estimation error:** $(round(norm(m_est - m_true), digits=4))
"""

# ╔═╡ contour_plot_section
md"## Cost Function Landscape with Gradients"

# ╔═╡ create_grid
begin
	# Create a grid for contour plot
	m1_range = range(-0.5, 2.5, length=100)
	m2_range = range(-0.5, 2.5, length=100)
	
	# Compute cost function on grid
	J_grid = [cost_function([m1, m2], G, d_obs, lambda) 
	          for m2 in m2_range, m1 in m1_range]
end

# ╔═╡ sample_gradients
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

# ╔═╡ plot_contours
begin
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
		name="Cost Function",
		ncontours=20
	)
	
	# Plot gradient vectors (as arrows)
	arrow_traces = []
	for i in 1:length(grad_points)
		m = grad_points[i]
		grad = grad_vectors[i]
		
		# Normalize gradient for visualization
		grad_norm = norm(grad)
		if grad_norm > 1e-6
			grad_scaled = -0.15 * grad / grad_norm  # Negative for descent direction
			
			push!(arrow_traces, scatter(
				x=[m[1], m[1] + grad_scaled[1]],
				y=[m[2], m[2] + grad_scaled[2]],
				mode="lines",
				line=attr(color="red", width=2),
				showlegend=false,
				hoverinfo="skip"
			))
			
			# Add arrowhead
			push!(arrow_traces, scatter(
				x=[m[1] + grad_scaled[1]],
				y=[m[2] + grad_scaled[2]],
				mode="markers",
				marker=attr(
					symbol="arrow",
					angle=atan(grad_scaled[2], grad_scaled[1]) * 180 / π - 90,
					size=10,
					color="red"
				),
				showlegend=false,
				hoverinfo="skip"
			))
		end
	end
	
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
	all_traces = [contour_trace, arrow_traces..., true_trace, est_trace]
	
	layout = Layout(
		title="Cost Function Landscape with Gradient Vectors<br>(Red arrows point in steepest descent direction)",
		xaxis=attr(title="Parameter m₁", range=(-0.5, 2.5)),
		yaxis=attr(title="Parameter m₂", range=(-0.5, 2.5)),
		width=700,
		height=700,
		showlegend=true
	)
	
	plot(all_traces, layout)
end

# ╔═╡ ellipse_section
md"## Error Ellipse (Hessian Geometry)"

# ╔═╡ plot_error_ellipse
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
	
	# Plot eigenvector directions
	eigen_trace1 = scatter(
		x=[m_est[1], m_est[1] + 0.3 * eigenvectors[1, 1]],
		y=[m_est[2], m_est[2] + 0.3 * eigenvectors[2, 1]],
		mode="lines+markers",
		line=attr(color="orange", width=3),
		marker=attr(size=8),
		name="Eigenvector 1 (λ=$(round(eigenvalues[1], digits=2)))"
	)
	
	eigen_trace2 = scatter(
		x=[m_est[1], m_est[1] + 0.3 * eigenvectors[1, 2]],
		y=[m_est[2], m_est[2] + 0.3 * eigenvectors[2, 2]],
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
		xaxis=attr(title="Parameter m₁", range=(-0.5, 2.5)),
		yaxis=attr(title="Parameter m₂", range=(-0.5, 2.5)),
		width=700,
		height=700,
		showlegend=true
	)
	
	plot([contour_trace2, ellipse_trace, eigen_trace1, eigen_trace2, true_trace2, est_trace2], layout2)
end

# ╔═╡ bias_variance_section
md"## Bias-Variance Trade-off"

# ╔═╡ compute_bias_variance
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
			noise_trial = randn(3) * noise_level
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

# ╔═╡ plot_bias_variance
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

# ╔═╡ interpretation_section
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
# ╟─f69ffb70-0136-4745-b0b9-233ae994fbb1
# ╟─true_model
# ╠═true_solution_cell
# ╠═construct_G
# ╠═generate_data
# ╟─display_matrices
# ╟─cost_function_section
# ╠═cost_function_def
# ╠═compute_hessian
# ╟─display_hessian
# ╠═analytical_solution
# ╟─display_solution
# ╟─contour_plot_section
# ╠═create_grid
# ╠═sample_gradients
# ╟─plot_contours
# ╟─ellipse_section
# ╟─plot_error_ellipse
# ╟─bias_variance_section
# ╠═compute_bias_variance
# ╟─plot_bias_variance
# ╟─interpretation_section
# ╠═499b6d53-97a2-4ba8-a6d6-030ce1e6fa43
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
