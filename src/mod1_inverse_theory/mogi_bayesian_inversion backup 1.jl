### A Pluto.jl notebook ###
# v0.20.21

#> [frontmatter]
#> layout = "layout.jlhtml"
#> title = "Bayesian Inversion: Mogi Model"
#> tags = ["module1"]
#> description = "Estimating volcano source parameters using Bayesian inference with the Mogi model."

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ 04d19c81-08f8-4f07-a0c7-f51c1b39d271
using PlutoUI, Plots, Distributions, Measures, PlutoTeachingTools, StatsPlots, LinearAlgebra

# ╔═╡ 02cd8bb4-d403-47f9-9de9-35d7b2b82bb8
ChooseDisplayMode()

# ╔═╡ 4c8ba5dc-a019-46ab-b7cc-b03cf4cee5f8
TableOfContents()

# ╔═╡ 91c82c81-8cfc-4b54-bed7-0592f5b39351
md"""
# Bayesian Inversion: Locating a Volcanic Pressure Source

In this notebook, we use **Bayesian inference** to estimate the location of a subsurface volcanic pressure source using the **Mogi model**. The Mogi model describes ground deformation caused by a point-like pressure source in an elastic half-space.

## The Problem

When a volcano inflates or deflates (e.g., due to magma movement), the ground surface deforms. We can measure:
- **Vertical displacement** (subsidence or uplift), denoted as $w$
- **Horizontal displacement** (radial motion), denoted as $u$

Our goal is to **invert** these observations to estimate:
- The **horizontal position** $(x_0)$ of the source
- The **depth** $(d)$ of the source
- The **pressure change** $(\Delta P)$ in the source

## Why Bayesian Inference?

Classical inversion gives us a "best fit" solution, but doesn't tell us:
- How **uncertain** our estimates are
- What **trade-offs** exist between parameters
- How **multiple measurements** affect our confidence

Bayesian inference addresses all of these by computing the **posterior probability distribution**:

```math
\sigma(m | d_{\text{obs}}) = \frac{\rho(d_{\text{obs}} | m) \cdot \rho(m)}{\int \rho(d_{\text{obs}} | m) \cdot \rho(m) \, dm}
```

where:
- $\rho(m)$ is the **prior**: what we know before seeing data
- $\rho(d_{\text{obs}} | m)$ is the **likelihood**: how well the model fits observations
- $\sigma(m | d_{\text{obs}})$ is the **posterior**: updated knowledge after seeing data
"""

# ╔═╡ d08d67f0-2468-485a-bf09-dd7a3d9f2e16
md"""
## The Mogi Model

The **Mogi model** (Mogi, 1958) describes surface displacement caused by a spherical pressure source buried in an elastic half-space.

### Model Geometry

- Source at depth $d$ (meters below surface)
- Source radius $a$ (meters), where $a \ll d$
- Horizontal position $x$ relative to source center
- Pressure change $\Delta P$ (MPa)

### Forward Model Equations

The **vertical displacement** $w$ (positive upward) and **horizontal displacement** $u$ (positive radially outward) at horizontal distance $x$ from the source are:

```math
w(x) = \frac{(1-\nu) a^3 d \Delta P}{G(x^2 + d^2)^{3/2}}
```

```math
u(x) = \frac{(1-\nu) a^3 x \Delta P}{G(x^2 + d^2)^{3/2}}
```

where:
- $G$ = shear modulus (typically 30,000 MPa)
- $\nu$ = Poisson's ratio (typically 0.25)
- $a$ = source radius (meters)
- $d$ = source depth (meters)
- $\Delta P$ = pressure change (MPa)

### Key Properties

1. **Radial symmetry**: Deformation is the same at all points equidistant from the source
2. **Sign of $u$**: Changes across the source (negative on one side, positive on the other)
3. **Sign of $w$**: 
   - Positive $\Delta P$ → uplift (inflation)
   - Negative $\Delta P$ → subsidence (deflation)
"""

# ╔═╡ 3b06a3a9-8a65-4fa9-a608-0e860670a3c4
# Mogi vertical displacement function
function mogi_vertical(x, a, d, G, ΔP, ν)
    """
    Calculate vertical surface displacement using Mogi model.
    
    Parameters:
    - x: horizontal distance from source (m)
    - a: source radius (m)
    - d: source depth (m)
    - G: shear modulus (MPa)
    - ΔP: pressure change (MPa)
    - ν: Poisson's ratio
    
    Returns:
    - w: vertical displacement (m)
    """
    w = (1 - ν) * a^3 * d * ΔP / (G * (x^2 + d^2)^(1.5))
    return w
end

# ╔═╡ 33f408d2-50ab-4cbe-a3c3-7d055b2bb5fd
# Mogi horizontal displacement function
function mogi_horizontal(x, a, d, G, ΔP, ν)
    """
    Calculate horizontal surface displacement using Mogi model.
    
    Parameters:
    - x: horizontal distance from source (m)
    - a: source radius (m)
    - d: source depth (m)
    - G: shear modulus (MPa)
    - ΔP: pressure change (MPa)
    - ν: Poisson's ratio
    
    Returns:
    - u: horizontal displacement (m)
    """
    u = (1 - ν) * a^3 * x * ΔP / (G * (x^2 + d^2)^(1.5))
    return u
end

# ╔═╡ a1b2c3d4-5e6f-7g8h-9i0j-1k2l3m4n5o6p
md"""
## Interactive Experiment Setup

Use the controls below to set up a synthetic experiment. In a real scenario, you would have actual GPS or InSAR measurements, but here we'll simulate observations to understand how Bayesian inference works.

### Experiment Parameters
"""

# ╔═╡ b1c2d3e4-6f7g-8h9i-0j1k-2l3m4n5o6p7q
md"""
**True Source Location (Unknown to "student"):**
- Horizontal position: $(@bind x_true Slider(-5000:100:5000, default=0, show_value=true)) meters
- Depth: $(@bind d_true Slider(2000:100:15000, default=8000, show_value=true)) meters

**True Pressure Change:**
- ΔP: $(@bind ΔP_true Slider(-500:10:500, default=-300, show_value=true)) MPa

**Observation Setup:**
- Measurement location: $(@bind x_obs Slider(-10000:500:10000, default=5000, show_value=true)) meters from origin
- Measurement uncertainty (σ): $(@bind σ_obs Slider(0.001:0.001:0.050, default=0.010, show_value=true)) meters

**Which data to use for inversion?**
- Use vertical displacement (w): $(@bind use_w CheckBox(default=true))
- Use horizontal displacement (u): $(@bind use_u CheckBox(default=true))
"""

# ╔═╡ c2d3e4f5-7g8h-9i0j-1k2l-3m4n5o6p7q8r
# Fixed model parameters
begin
	a = 500.0      # source radius (m) - kept small compared to depth
	G = 30000.0    # shear modulus (MPa)
	ν = 0.25       # Poisson's ratio
end

# ╔═╡ d3e4f5g6-8h9i-0j1k-2l3m-4n5o6p7q8r9s
md"""
## Generate Synthetic Observations

Based on your chosen "true" parameters, we generate synthetic observations with added noise to simulate real measurements.
"""

# ╔═╡ e4f5g6h7-9i0j-1k2l-3m4n-5o6p7q8r9s0t
# Generate "observed" data with noise
begin
	# Calculate true displacements at observation point
	# Note: x_obs is measured from origin, x_true is source location
	# So relative distance is (x_obs - x_true)
	x_rel = x_obs - x_true
	
	w_true = mogi_vertical(x_rel, a, d_true, G, ΔP_true, ν)
	u_true = mogi_horizontal(x_rel, a, d_true, G, ΔP_true, ν)
	
	# Add Gaussian noise
	w_obs = w_true + randn() * σ_obs
	u_obs = u_true + randn() * σ_obs
	
	# Store for display
	obs_summary = md"""
	**Synthetic Observations at x = $(x_obs) m:**
	- Vertical displacement: $(round(w_obs*1000, digits=2)) mm (true: $(round(w_true*1000, digits=2)) mm)
	- Horizontal displacement: $(round(u_obs*1000, digits=2)) mm (true: $(round(u_true*1000, digits=2)) mm)
	- Measurement uncertainty: ±$(round(σ_obs*1000, digits=2)) mm
	"""
end

# ╔═╡ f5g6h7i8-0j1k-2l3m-4n5o-6p7q8r9s0t1u
obs_summary

# ╔═╡ a2b3c4d5-6e7f-8g9h-0i1j-2k3l4m5n6o7p
md"""
## Prior Information

Before seeing any data, what do we know about the source location? This is encoded in our **prior distribution** $\rho(m)$.

For this problem, let's assume:
- The source is somewhere between 0-15 km depth (we know it's not at the surface or infinitely deep)
- The source could be anywhere horizontally within ±10 km
- We're less certain about exact depth than horizontal position

We'll use **Gaussian priors**:

```math
\rho(x_0) \sim \mathcal{N}(\mu_x, \sigma_x^2)
```
```math
\rho(d) \sim \mathcal{N}(\mu_d, \sigma_d^2)
```
"""

# ╔═╡ b3c4d5e6-7f8g-9h0i-1j2k-3l4m5n6o7p8q
md"""
**Prior Distribution Parameters:**

For horizontal position $x_0$:
- Mean: $(@bind μ_x Slider(-5000:500:5000, default=0, show_value=true)) meters
- Std dev: $(@bind σ_x Slider(1000:500:8000, default=5000, show_value=true)) meters

For depth $d$:
- Mean: $(@bind μ_d Slider(3000:500:12000, default=7000, show_value=true)) meters  
- Std dev: $(@bind σ_d Slider(500:500:5000, default=3000, show_value=true)) meters
"""

# ╔═╡ c4d5e6f7-8g9h-0i1j-2k3l-4m5n6o7p8q9r
# Define prior distributions
begin
	prior_x = Normal(μ_x, σ_x)
	prior_d = Normal(μ_d, σ_d)
	
	# Create 2D grid for visualization
	x_grid = range(-10000, 10000, length=100)
	d_grid = range(1000, 15000, length=100)
	
	# Evaluate 2D prior
	prior_2d = [pdf(prior_x, x) * pdf(prior_d, d) 
	            for d in d_grid, x in x_grid]
end;

# ╔═╡ d5e6f7g8-9h0i-1j2k-3l4m-5n6o7p8q9r0s
md"""
## Likelihood Function

The **likelihood** $\rho(d_{\text{obs}} | m)$ tells us: "If the source were at location $(x_0, d)$, how likely would we observe our measurements?"

We assume Gaussian measurement errors:

```math
\rho(w_{\text{obs}} | x_0, d) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(w_{\text{obs}} - w_{\text{pred}}(x_0, d))^2}{2\sigma^2}\right)
```

where $w_{\text{pred}}(x_0, d)$ is the displacement predicted by the Mogi model for a source at $(x_0, d)$.

### Why is Likelihood Important?

- Models that predict displacements **close** to observations get **high likelihood**
- Models that predict displacements **far** from observations get **low likelihood**
- The likelihood "shape" shows which models are compatible with data
"""

# ╔═╡ e6f7g8h9-0i1j-2k3l-4m5n-6o7p8q9r0s1t
# Likelihood function
function compute_likelihood_2d(x_grid, d_grid, x_obs, w_obs, u_obs, σ_obs, use_w, use_u)
    """
    Compute 2D likelihood over (x, d) parameter space.
    
    Note: We're assuming ΔP is known (fixed at true value) to reduce to 2D problem.
    In reality, you could extend this to 3D or higher dimensions.
    """
    likelihood = zeros(length(d_grid), length(x_grid))
    
    for (i, d) in enumerate(d_grid)
        for (j, x) in enumerate(x_grid)
            # Calculate relative distance from observation to source
            x_rel = x_obs - x
            
            # Predict displacements
            w_pred = mogi_vertical(x_rel, a, d, G, ΔP_true, ν)
            u_pred = mogi_horizontal(x_rel, a, d, G, ΔP_true, ν)
            
            # Compute log-likelihood (more numerically stable)
            log_lik = 0.0
            
            if use_w
                log_lik += logpdf(Normal(w_pred, σ_obs), w_obs)
            end
            
            if use_u
                log_lik += logpdf(Normal(u_pred, σ_obs), u_obs)
            end
            
            likelihood[i, j] = exp(log_lik)
        end
    end
    
    return likelihood
end

# ╔═╡ f7g8h9i0-1j2k-3l4m-5n6o-7p8q9r0s1t2u
# Compute likelihood
begin
	if use_w || use_u
		likelihood_2d = compute_likelihood_2d(
			x_grid, d_grid, x_obs, w_obs, u_obs, σ_obs, use_w, use_u
		)
	else
		likelihood_2d = ones(length(d_grid), length(x_grid))
	end
end;

# ╔═╡ g8h9i0j1-2k3l-4m5n-6o7p-8q9r0s1t2u3v
md"""
## Posterior Distribution

The **posterior** combines our prior knowledge with what we learned from observations:

```math
\sigma(x_0, d | d_{\text{obs}}) \propto \rho(d_{\text{obs}} | x_0, d) \cdot \rho(x_0, d)
```

This is **Bayes' theorem** in action! The posterior tells us:
- Which source locations are **most probable** given the data
- How **uncertain** we are (wide distribution = high uncertainty)
- **Trade-offs** between parameters (e.g., shallow + low ΔP might look like deep + high ΔP)

### Interpreting the Posterior

- **Narrow peak**: Data strongly constrain the source location
- **Broad distribution**: Multiple locations could explain the data
- **Multiple peaks**: Data are ambiguous (non-unique solution)
"""

# ╔═╡ h9i0j1k2-3l4m-5n6o-7p8q-9r0s1t2u3v4w
# Compute posterior (unnormalized)
posterior_2d = prior_2d .* likelihood_2d;

# ╔═╡ i0j1k2l3-4m5n-6o7p-8q9r-0s1t2u3v4w5x
# Normalize posterior
posterior_2d_norm = posterior_2d ./ sum(posterior_2d);

# ╔═╡ j1k2l3m4-5n6o-7p8q-9r0s-1t2u3v4w5x6y
md"""
## Visualization: Prior, Likelihood, Posterior

Below we show three heatmaps side-by-side:

1. **Prior** (left): What we knew before seeing data
2. **Likelihood** (middle): What the data tell us
3. **Posterior** (right): Combined knowledge = Prior × Likelihood

**Experiment:**
- Try using only vertical (w) or only horizontal (u) data - how does it change the posterior?
- Increase measurement uncertainty (σ) - what happens to the posterior width?
- Change the observation location - when does the problem become ambiguous?
"""

# ╔═╡ k2l3m4n5-6o7p-8q9r-0s1t-2u3v4w5x6y7z
# Plotting helper function
function plot_2d_distribution(data, x_grid, d_grid, title_str; show_true=false)
    plt = heatmap(
        x_grid ./ 1000,  # convert to km
        d_grid ./ 1000,  # convert to km
        data,
        xlabel="Horizontal Position (km)",
        ylabel="Depth (km)",
        title=title_str,
        color=:viridis,
        yflip=true,  # depth increases downward
        aspect_ratio=:equal,
        clims=(0, maximum(data)),
        colorbar_title="Probability Density"
    )
    
    if show_true
        scatter!(plt, [x_true/1000], [d_true/1000], 
                marker=:star5, markersize=10, color=:red, 
                label="True Location", legend=:topright)
    end
    
    return plt
end

# ╔═╡ l3m4n5o6-7p8q-9r0s-1t2u-3v4w5x6y7z8a
# Create comparison plot
begin
	p1 = plot_2d_distribution(prior_2d, x_grid, d_grid, "Prior ρ(m)")
	p2 = plot_2d_distribution(likelihood_2d, x_grid, d_grid, "Likelihood ρ(d|m)", show_true=true)
	p3 = plot_2d_distribution(posterior_2d_norm, x_grid, d_grid, "Posterior σ(m|d)", show_true=true)
	
	plot(p1, p2, p3, layout=(1, 3), size=(1400, 400), margin=5mm)
end

# ╔═╡ m4n5o6p7-8q9r-0s1t-2u3v-4w5x6y7z8a9b
md"""
## Marginal Distributions

Sometimes we're only interested in **one parameter** at a time. We can **marginalize** the posterior to get 1D distributions:

```math
\rho(x_0 | d_{\text{obs}}) = \int \sigma(x_0, d | d_{\text{obs}}) \, dd
```

This tells us about the horizontal position **regardless** of depth.
"""

# ╔═╡ n5o6p7q8-9r0s-1t2u-3v4w-5x6y7z8a9b0c
# Compute marginal distributions
begin
	# Marginal for x (integrate over d)
	marginal_x = sum(posterior_2d_norm, dims=1)[:]
	marginal_x ./= sum(marginal_x)  # normalize
	
	# Marginal for d (integrate over x)
	marginal_d = sum(posterior_2d_norm, dims=2)[:]
	marginal_d ./= sum(marginal_d)  # normalize
	
	# Plot marginals
	p_mx = plot(
		x_grid ./ 1000, marginal_x,
		xlabel="Horizontal Position (km)",
		ylabel="Probability Density",
		title="Marginal: ρ(x₀|d_obs)",
		label=nothing,
		linewidth=2,
		fill=(0, 0.3, :blue)
	)
	vline!(p_mx, [x_true/1000], label="True x₀", linewidth=2, color=:red, linestyle=:dash)
	
	p_md = plot(
		d_grid ./ 1000, marginal_d,
		xlabel="Depth (km)",
		ylabel="Probability Density",
		title="Marginal: ρ(d|d_obs)",
		label=nothing,
		linewidth=2,
		fill=(0, 0.3, :blue),
		yflip=false
	)
	vline!(p_md, [d_true/1000], label="True d", linewidth=2, color=:red, linestyle=:dash)
	
	plot(p_mx, p_md, layout=(1, 2), size=(900, 300), margin=5mm)
end

# ╔═╡ o6p7q8r9-0s1t-2u3v-4w5x-6y7z8a9b0c1d
md"""
## Quantifying Uncertainty

From the posterior, we can extract useful statistics:

**Maximum a posteriori (MAP) estimate:** The most probable location

**Credible intervals:** Ranges containing, say, 95% of the posterior probability

**Standard deviations:** How spread out is the posterior?
"""

# ╔═╡ p7q8r9s0-1t2u-3v4w-5x6y-7z8a9b0c1d2e
# Compute posterior statistics
begin
	# Find MAP estimate (maximum a posteriori)
	map_idx = argmax(posterior_2d_norm)
	map_d_idx, map_x_idx = Tuple(map_idx)
	map_x = x_grid[map_x_idx]
	map_d = d_grid[map_d_idx]
	
	# Compute posterior means (expected values)
	mean_x = sum(x_grid .* marginal_x)
	mean_d = sum(d_grid .* marginal_d)
	
	# Compute posterior standard deviations
	std_x = sqrt(sum((x_grid .- mean_x).^2 .* marginal_x))
	std_d = sqrt(sum((d_grid .- mean_d).^2 .* marginal_d))
	
	# Display results
	results_md = md"""
	### Inversion Results
	
	| Parameter | True Value | Prior Mean | Posterior Mean | Posterior Std | MAP Estimate |
	|-----------|------------|------------|----------------|---------------|--------------|
	| x₀ (km) | $(round(x_true/1000, digits=2)) | $(round(μ_x/1000, digits=2)) | $(round(mean_x/1000, digits=2)) | $(round(std_x/1000, digits=2)) | $(round(map_x/1000, digits=2)) |
	| d (km) | $(round(d_true/1000, digits=2)) | $(round(μ_d/1000, digits=2)) | $(round(mean_d/1000, digits=2)) | $(round(std_d/1000, digits=2)) | $(round(map_d/1000, digits=2)) |
	
	**Interpretation:**
	- If posterior std is **much smaller** than prior std → data were informative
	- If posterior mean is **close to true value** → inversion successful
	- Compare MAP vs. Mean: if different, posterior might be skewed or multi-modal
	"""
end

# ╔═╡ q8r9s0t1-2u3v-4w5x-6y7z-8a9b0c1d2e3f
results_md

# ╔═╡ r9s0t1u2-3v4w-5x6y-7z8a-9b0c1d2e3f4g
md"""
## Why Uncertainty Matters: A Demonstration

Let's see what happens when we have **very uncertain measurements** vs. **precise measurements**:

Try these experiments:
1. Set σ_obs = 0.001 m (1 mm) - very precise GPS
2. Set σ_obs = 0.050 m (50 mm) - less precise measurements

Notice how the likelihood and posterior change! With uncertain data:
- Likelihood becomes broader (many models fit the data)
- Posterior is less peaked (we remain uncertain about location)
- Prior has more influence on the result

This is why **reporting uncertainty is crucial** in geophysics:
- Helps decide if more measurements are needed
- Prevents overconfident interpretations
- Shows trade-offs between parameters
"""

# ╔═╡ s0t1u2v3-4w5x-6y7z-8a9b-0c1d2e3f4g5h
md"""
## What Happens with Multiple Observations?

Real surveys measure displacement at **many locations**. Each new observation provides additional constraints. Try this:

**Thought experiment** (you can extend the code):
- Observation at x = +5 km measures **uplift**
- Observation at x = -5 km also measures **uplift**
- Where must the source be? (Answer: somewhere near x = 0)

This demonstrates how **multiple data points** can:
- Break ambiguities (eliminate multi-modal posteriors)
- Dramatically reduce uncertainty
- Provide better depth resolution

In real inverse problems, we often have:
- 10-100+ GPS stations
- Satellite InSAR providing thousands of measurements
- Combined data types (GPS + InSAR + tilt + strain)
"""

# ╔═╡ t1u2v3w4-5x6y-7z8a-9b0c-1d2e3f4g5h6i
md"""
## Interactive Exploration: Key Insights

Use the sliders above to explore these key concepts:

### 1. Observation Location Matters
- Move x_obs close to x_true: easier to constrain depth
- Move x_obs far from source: mostly sensitive to shallow sources
- **Why?** Displacement amplitude decays with distance

### 2. Data Type Matters
- Use only w (vertical): may be ambiguous (shallow-strong vs deep-weak source)
- Use only u (horizontal): provides horizontal position
- Use both u and w: breaks ambiguities! 🎉

### 3. Prior Information Matters
- Tight prior (small σ_x, σ_d): posterior stays near prior even with data
- Wide prior (large σ_x, σ_d): data dominate the posterior
- **Trade-off:** Too tight prior can bias results; too wide gives less regularization

### 4. Measurement Uncertainty Matters
- Small σ_obs: sharp likelihood, data dominate
- Large σ_obs: broad likelihood, prior matters more
- **Reality check:** GPS typically 1-5 mm, InSAR 5-10 mm precision

"""

# ╔═╡ u2v3w4x5-6y7z-8a9b-0c1d-2e3f4g5h6i7j
md"""
## Historical Context: The 1914 Sakurajima Eruption

The Mogi model was developed to explain dramatic subsidence observed after the 1914 eruption of Sakurajima volcano in Japan. Precise leveling surveys showed:

- Up to **70+ cm subsidence** near the volcano
- **Radially symmetric pattern** extending 17+ km from the vent
- Pattern consistent with deflation of a subsurface magma chamber

Kiyoo Mogi's elegant solution (1958) remains widely used today for:
- Volcano monitoring and eruption forecasting
- Understanding magma chamber geometry
- Groundwater withdrawal studies
- CO₂ sequestration monitoring

**Key insight:** Even 70 years later, this simple model provides crucial insights into subsurface processes we cannot directly observe!
"""

# ╔═╡ v3w4x5y6-7z8a-9b0c-1d2e-3f4g5h6i7j8k
md"""
## Extensions and Advanced Topics

This notebook simplified several aspects. In real research:

### 1. More Parameters
- Invert for ΔP, x₀, d simultaneously (3D or higher)
- Include source radius a
- Account for topography
- Non-uniform elastic properties

### 2. More Sophisticated Methods
- **Markov Chain Monte Carlo (MCMC):** Sample the full posterior
- **Ensemble methods:** Multiple models with weights
- **Nonlinear optimization:** Gradient-based search

### 3. More Realistic Data
- Multiple observation points
- Time-series (track inflation/deflation over time)
- Combined InSAR + GPS + tilt + seismic
- Outlier detection and robust statistics

### 4. Model Limitations
- Assumes homogeneous half-space (real earth is layered)
- Ignores topography
- Point source (real chambers have finite extent)
- Elastic deformation (may have inelastic effects)

**Despite these simplifications, Mogi model remains incredibly useful for first-order understanding!**
"""

# ╔═╡ w4x5y6z7-8a9b-0c1d-2e3f-4g5h6i7j8k9l
md"""
## Summary: Key Takeaways

1. **Forward Problem:** Mogi model predicts surface deformation from subsurface pressure source

2. **Inverse Problem:** Use observations to infer source location and properties

3. **Bayesian Inference:** 
   - Prior: what we know before data
   - Likelihood: what data tell us
   - Posterior: combined knowledge (Prior × Likelihood)

4. **Uncertainty Quantification:**
   - Narrow posterior → well-constrained
   - Broad posterior → ambiguous/uncertain
   - Multiple data types → better constraints

5. **Why This Matters:**
   - Volcano monitoring and hazard assessment
   - Understanding magma movement
   - Quantifying confidence in interpretations
   - Guiding survey design (where to measure?)

**Most importantly:** Always report uncertainty! A point estimate without uncertainty is incomplete and potentially misleading.
"""

# ╔═╡ x5y6z7a8-9b0c-1d2e-3f4g-5h6i7j8k9l0m
md"""
## References

1. **Mogi, K. (1958).** Relations between the eruptions of various volcanoes and the deformations of the ground surfaces around them. *Bulletin of the Earthquake Research Institute*, 36, 99-134.

2. **Segall, P. (2010).** *Earthquake and Volcano Deformation.* Princeton University Press.

3. **Tarantola, A. (2005).** *Inverse Problem Theory and Methods for Model Parameter Estimation.* SIAM.

4. **Fichtner, A. (2021).** Lecture Notes on Inverse Theory. ETH Zurich.

5. **Mosegaard, K., & Tarantola, A. (2002).** Probabilistic approach to inverse problems. *International Geophysics Series*, 81(A), 237-268.

6. **Interactive Mogi Demo:** [GSC Community Codes](https://gscommunitycodes.usf.edu/geoscicommunitycodes/public/numeracy/numeracy_mogi/mogi.html)
"""

# ╔═╡ y6z7a8b9-0c1d-2e3f-4g5h-6i7j8k9l0m1n
md"""
## Appendix: Technical Notes

### Numerical Considerations

1. **Grid Resolution:** We use 100×100 grid points. Finer grids give smoother results but slower computation.

2. **Log-Likelihood:** We compute log-likelihood to avoid numerical underflow with very small probabilities.

3. **Normalization:** Posterior is normalized by dividing by the sum (discrete approximation to integral).

### Assumptions

1. **Known ΔP:** We fix ΔP at true value to reduce to 2D problem. Could extend to 3D.

2. **Single Observation:** Real surveys have 10-1000+ points.

3. **Gaussian Errors:** Assumes measurement errors are normally distributed (usually reasonable for GPS/InSAR).

4. **Independent Measurements:** If using both u and w, assumes errors are uncorrelated (may not be true with GPS).
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Measures = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoTeachingTools = "661c6b06-c737-4d37-b85c-46df65de6f69"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd"

[compat]
Distributions = "~0.25.123"
Measures = "~0.3.3"
Plots = "~1.41.1"
PlutoTeachingTools = "~0.4.6"
PlutoUI = "~0.7.79"
StatsPlots = "~0.15.8"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.4"
manifest_format = "2.0"
project_hash = "simplified_placeholder"

[[deps.Distributions]]
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.123"

[[deps.LinearAlgebra]]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.Measures]]
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.3"

[[deps.Plots]]
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.41.1"

[[deps.PlutoTeachingTools]]
uuid = "661c6b06-c737-4d37-b85c-46df65de6f69"
version = "0.4.6"

[[deps.PlutoUI]]
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.79"

[[deps.StatsPlots]]
uuid = "f3b207a7-027a-5e70-b257-86293d7955fd"
version = "0.15.8"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000003
PLUTO_NOTEBOOK_METADATA = """
# ╠═04d19c81-08f8-4f07-a0c7-f51c1b39d271
# ╠═02cd8bb4-d403-47f9-9de9-35d7b2b82bb8
# ╠═4c8ba5dc-a019-46ab-b7cc-b03cf4cee5f8
# ╟─91c82c81-8cfc-4b54-bed7-0592f5b39351
# ╟─d08d67f0-2468-485a-bf09-dd7a3d9f2e16
# ╠═3b06a3a9-8a65-4fa9-a608-0e860670a3c4
# ╠═33f408d2-50ab-4cbe-a3c3-7d055b2bb5fd
# ╟─a1b2c3d4-5e6f-7g8h-9i0j-1k2l3m4n5o6p
# ╟─b1c2d3e4-6f7g-8h9i-0j1k-2l3m4n5o6p7q
# ╠═c2d3e4f5-7g8h-9i0j-1k2l-3m4n5o6p7q8r
# ╟─d3e4f5g6-8h9i-0j1k-2l3m-4n5o6p7q8r9s
# ╠═e4f5g6h7-9i0j-1k2l-3m4n-5o6p7q8r9s0t
# ╟─f5g6h7i8-0j1k-2l3m-4n5o-6p7q8r9s0t1u
# ╟─a2b3c4d5-6e7f-8g9h-0i1j-2k3l4m5n6o7p
# ╟─b3c4d5e6-7f8g-9h0i-1j2k-3l4m5n6o7p8q
# ╠═c4d5e6f7-8g9h-0i1j-2k3l-4m5n6o7p8q9r
# ╟─d5e6f7g8-9h0i-1j2k-3l4m-5n6o7p8q9r0s
# ╠═e6f7g8h9-0i1j-2k3l-4m5n-6o7p8q9r0s1t
# ╠═f7g8h9i0-1j2k-3l4m-5n6o-7p8q9r0s1t2u
# ╟─g8h9i0j1-2k3l-4m5n-6o7p-8q9r0s1t2u3v
# ╠═h9i0j1k2-3l4m-5n6o-7p8q-9r0s1t2u3v4w
# ╠═i0j1k2l3-4m5n-6o7p-8q9r-0s1t2u3v4w5x
# ╟─j1k2l3m4-5n6o-7p8q-9r0s-1t2u3v4w5x6y
# ╠═k2l3m4n5-6o7p-8q9r-0s1t-2u3v4w5x6y7z
# ╟─l3m4n5o6-7p8q-9r0s-1t2u-3v4w5x6y7z8a
# ╟─m4n5o6p7-8q9r-0s1t-2u3v-4w5x6y7z8a9b
# ╟─n5o6p7q8-9r0s-1t2u-3v4w-5x6y7z8a9b0c
# ╟─o6p7q8r9-0s1t-2u3v-4w5x-6y7z8a9b0c1d
# ╠═p7q8r9s0-1t2u-3v4w-5x6y-7z8a9b0c1d2e
# ╟─q8r9s0t1-2u3v-4w5x-6y7z-8a9b0c1d2e3f
# ╟─r9s0t1u2-3v4w-5x6y-7z8a-9b0c1d2e3f4g
# ╟─s0t1u2v3-4w5x-6y7z-8a9b-0c1d2e3f4g5h
# ╟─t1u2v3w4-5x6y-7z8a-9b0c-1d2e3f4g5h6i
# ╟─u2v3w4x5-6y7z-8a9b-0c1d-2e3f4g5h6i7j
# ╟─v3w4x5y6-7z8a-9b0c-1d2e-3f4g5h6i7j8k
# ╟─w4x5y6z7-8a9b-0c1d-2e3f-4g5h6i7j8k9l
# ╟─x5y6z7a8-9b0c-1d2e-3f4g-5h6i7j8k9l0m
# ╟─y6z7a8b9-0c1d-2e3f-4g5h-6i7j8k9l0m1n
"""

# ╔═╡ Cell order:
# ╟─91c82c81-8cfc-4b54-bed7-0592f5b39351
# ╠═04d19c81-08f8-4f07-a0c7-f51c1b39d271
# ╠═02cd8bb4-d403-47f9-9de9-35d7b2b82bb8
# ╠═4c8ba5dc-a019-46ab-b7cc-b03cf4cee5f8
# ╟─d08d67f0-2468-485a-bf09-dd7a3d9f2e16
# ╠═3b06a3a9-8a65-4fa9-a608-0e860670a3c4
# ╠═33f408d2-50ab-4cbe-a3c3-7d055b2bb5fd
# ╟─a1b2c3d4-5e6f-7g8h-9i0j-1k2l3m4n5o6p
# ╟─b1c2d3e4-6f7g-8h9i-0j1k-2l3m4n5o6p7q
# ╠═c2d3e4f5-7g8h-9i0j-1k2l-3m4n5o6p7q8r
# ╟─d3e4f5g6-8h9i-0j1k-2l3m-4n5o6p7q8r9s
# ╠═e4f5g6h7-9i0j-1k2l-3m4n-5o6p7q8r9s0t
# ╟─f5g6h7i8-0j1k-2l3m-4n5o-6p7q8r9s0t1u
# ╟─a2b3c4d5-6e7f-8g9h-0i1j-2k3l4m5n6o7p
# ╟─b3c4d5e6-7f8g-9h0i-1j2k-3l4m5n6o7p8q
# ╠═c4d5e6f7-8g9h-0i1j-2k3l-4m5n6o7p8q9r
# ╟─d5e6f7g8-9h0i-1j2k-3l4m-5n6o7p8q9r0s
# ╠═e6f7g8h9-0i1j-2k3l-4m5n-6o7p8q9r0s1t
# ╠═f7g8h9i0-1j2k-3l4m-5n6o-7p8q9r0s1t2u
# ╟─g8h9i0j1-2k3l-4m5n-6o7p-8q9r0s1t2u3v
# ╠═h9i0j1k2-3l4m-5n6o-7p8q-9r0s1t2u3v4w
# ╠═i0j1k2l3-4m5n-6o7p-8q9r-0s1t2u3v4w5x
# ╟─j1k2l3m4-5n6o-7p8q-9r0s-1t2u3v4w5x6y
# ╠═k2l3m4n5-6o7p-8q9r-0s1t-2u3v4w5x6y7z
# ╟─l3m4n5o6-7p8q-9r0s-1t2u-3v4w5x6y7z8a
# ╟─m4n5o6p7-8q9r-0s1t-2u3v-4w5x6y7z8a9b
# ╟─n5o6p7q8-9r0s-1t2u-3v4w-5x6y7z8a9b0c
# ╟─o6p7q8r9-0s1t-2u3v-4w5x-6y7z8a9b0c1d
# ╠═p7q8r9s0-1t2u-3v4w-5x6y-7z8a9b0c1d2e
# ╟─q8r9s0t1-2u3v-4w5x-6y7z-8a9b0c1d2e3f
# ╟─r9s0t1u2-3v4w-5x6y-7z8a-9b0c1d2e3f4g
# ╟─s0t1u2v3-4w5x-6y7z-8a9b-0c1d2e3f4g5h
# ╟─t1u2v3w4-5x6y-7z8a-9b0c-1d2e3f4g5h6i
# ╟─u2v3w4x5-6y7z-8a9b-0c1d-2e3f4g5h6i7j
# ╟─v3w4x5y6-7z8a-9b0c-1d2e-3f4g5h6i7j8k
# ╟─w4x5y6z7-8a9b-0c1d-2e3f-4g5h6i7j8k9l
# ╟─x5y6z7a8-9b0c-1d2e-3f4g-5h6i7j8k9l0m
# ╟─y6z7a8b9-0c1d-2e3f-4g5h-6i7j8k9l0m1n
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
