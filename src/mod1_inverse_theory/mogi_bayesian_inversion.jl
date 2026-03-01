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

# ╔═╡ e5ee82b0-5742-4b34-9e38-f079d75a1ad5
using PlutoUI, Distributions, PlutoPlotly

# ╔═╡ 320c8808-098a-11f1-8de1-b1d853234290
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

# ╔═╡ 79eb11d5-d946-48db-9f10-ca6ffa3d7df5
md"""
## Interactive Experiment Setup

Use the controls below to set up a synthetic experiment. In a real scenario, you would have actual GPS or InSAR measurements, but here we'll simulate observations to understand how Bayesian inference works.

### Experiment Parameters
"""

# ╔═╡ a1ee4121-0b45-4d7c-9967-4b63b3747b1a
md"""
**True Source Location:**
- Horizontal position: $(@bind x_true Slider(-5000:100:5000, default=0, show_value=true)) meters
- Depth: $(@bind d_true Slider(2000:100:15000, default=8000, show_value=true)) meters

**True Pressure Change:**
- ΔP: $(@bind ΔP_true Slider(-500:10:500, default=-300, show_value=true)) MPa

**Observation Setup (10 stations):**
- Center location: $(@bind x_obs_center Slider(-10000:500:10000, default=0, show_value=true)) meters
- Array span: $(@bind x_obs_span Slider(1000:500:20000, default=10000, show_value=true)) meters
- Measurement uncertainty (σ): $(@bind σ_obs Slider(0.001:0.001:0.050, default=0.010, show_value=true)) meters

**Which data to use for inversion?**
- Use vertical displacement (w): $(@bind use_w CheckBox(default=true))
- Use horizontal displacement (u): $(@bind use_u CheckBox(default=true))
"""

# ╔═╡ 3fee472d-1599-4e3b-a22f-860b4a6850f3
md"""
**Prior Distribution Parameters:**

For horizontal position $x_0$:
- Mean: $(@bind μ_x Slider(-5000:500:5000, default=0, show_value=true)) meters
- Std dev: $(@bind σ_x Slider(1000:500:8000, default=5000, show_value=true)) meters

For depth $d$:
- Mean: $(@bind μ_d Slider(3000:500:12000, default=7000, show_value=true)) meters  
- Std dev: $(@bind σ_d Slider(500:500:5000, default=3000, show_value=true)) meters
"""

# ╔═╡ 39d40806-6bae-4044-a115-c9cbeba65625
md"""
## Marginal Distributions

Sometimes we're only interested in **one parameter** at a time. We can **marginalize** the posterior to get 1D distributions:

```math
\rho(x_0 | d_{\text{obs}}) = \int \sigma(x_0, d | d_{\text{obs}}) \, dd
```

This tells us about the horizontal position **regardless** of depth.
"""

# ╔═╡ 02f2687d-6cbb-47ba-a5ae-e978064ef535
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

# ╔═╡ e1c578d6-6d8a-448d-8ae5-1cb00ad14dbb
md"""
## Generate Synthetic Observations

Based on your chosen "true" parameters, we generate synthetic observations with added noise to simulate real measurements.
"""

# ╔═╡ d664b09d-4fd8-4e9c-8bf7-062a6b67bc0a
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

# ╔═╡ cd4fab19-6aff-4254-9824-d55e4a7c8692
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

# ╔═╡ f89bdd99-9418-4a44-8fd7-5ad087d760e2
# Fixed model parameters
begin
	a = 500.0      # source radius (m) - kept small compared to depth
	G = 30000.0    # shear modulus (MPa)
	ν = 0.25       # Poisson's ratio
end

# ╔═╡ d1dfb877-fa16-40a6-bc1d-180df15b4889
 # Station locations
x_obs_points = range(x_obs_center - x_obs_span / 2, x_obs_center + x_obs_span / 2, length=10)

# ╔═╡ c98b8476-877b-4b6f-b179-37387a98b12c
# Generate "observed" data with noise (10 stations)
begin
   
	
    # Relative distance from each station to source
    x_rel = x_obs_points .- x_true
	
    # True displacements
    w_true = mogi_vertical.(x_rel, a, d_true, G, ΔP_true, ν)
    u_true = mogi_horizontal.(x_rel, a, d_true, G, ΔP_true, ν)
	
    # Add Gaussian noise
    w_obs = w_true .+ randn(length(x_rel)) .* σ_obs
    u_obs = u_true .+ randn(length(x_rel)) .* σ_obs
	
    # Store for display
    obs_summary = md"""
    **Synthetic Observations (10 stations):**
    - Center: $(round(x_obs_center, digits=0)) m, span: $(round(x_obs_span, digits=0)) m
    - Mean vertical displacement: $(round(mean(w_obs)*1000, digits=2)) mm
    - Mean horizontal displacement: $(round(mean(u_obs)*1000, digits=2)) mm
    - Measurement uncertainty: ±$(round(σ_obs*1000, digits=2)) mm
    """
end

# ╔═╡ 7b2c65b3-ef5a-41b8-ae0a-7e5407f85a7b
let
    # Surface profile relative to the source (meters)
    x_profile = range(-12000, 12000, length=241)

    # Compute model predictions (meters)
    w_profile = [mogi_vertical(x, a, d_true, G, ΔP_true, ν) for x in x_profile]
    u_profile = [mogi_horizontal(x, a, d_true, G, ΔP_true, ν) for x in x_profile]

    # Convert to km and mm for plotting
    x_km = x_profile ./ 1000
    w_mm = w_profile .* 1000
    u_mm = u_profile .* 1000

    # Observation locations (10 stations) relative to source
    x_obs_points = range(x_obs_center - x_obs_span / 2, x_obs_center + x_obs_span / 2, length=10)
    x_obs_rel_km = (x_obs_points .- x_true) ./ 1000

    traces = [scatter()]
    if use_w
        push!(traces,
            scatter(
                x=x_km,
                y=w_mm,
                mode="lines",
                name="w (mm)",
                line=attr(color="white", width=3)
            )
        )
        push!(traces,
            scatter(
                x=x_obs_rel_km,
                y=(w_obs .* 1000),
                mode="markers",
                name="w obs",
                marker=attr(color="white", size=6, symbol="circle")
            )
        )
    end

    if use_u
        push!(traces,
            scatter(
                x=x_km,
                y=u_mm,
                mode="lines",
                name="u (mm)",
                line=attr(color="gold", width=3)
            )
        )
        push!(traces,
            scatter(
                x=x_obs_rel_km,
                y=(u_obs .* 1000),
                mode="markers",
                name="u obs",
                marker=attr(color="gold", size=6, symbol="circle")
            )
        )
    end

    # Mark observation locations along the surface
    push!(traces,
        scatter(
            x=x_obs_rel_km,
            y=zeros(length(x_obs_rel_km)),
            mode="markers",
            name="Observation x",
            marker=attr(color="red", size=7, symbol="circle")
        )
    )

    layout = Layout(
        title="Forward Model: Surface Displacement",
        xaxis=attr(title="x (km)", zeroline=true),
        yaxis=attr(title="Displacement (mm)", zeroline=true),
        margin=attr(l=60, r=20, t=50, b=50),
        showlegend=true,
        width=900,
        height=400,
        paper_bgcolor="rgb(6, 80, 64)",
        plot_bgcolor="rgb(6, 80, 64)",
        font=attr(color="white")
    )

    WideCell(plot(Plot(traces, layout)))
end

# ╔═╡ e8f39318-0d31-40b0-be32-723a19554080
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

# ╔═╡ 47f0ad9a-250f-4066-b2e6-f5c02f8a7068
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

# ╔═╡ 0a7e7ee0-3753-464c-b690-54a4b36d84eb
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

# ╔═╡ 125908ba-18cb-4214-8032-584cb5bd0ad0
# Likelihood function
function compute_likelihood_2d(x_grid, d_grid, x_obs_points, w_obs, u_obs, σ_obs, use_w, use_u)
    """
    Compute 2D likelihood over (x, d) parameter space.
    
    Note: We're assuming ΔP is known (fixed at true value) to reduce to 2D problem.
    In reality, you could extend this to 3D or higher dimensions.
    """
    likelihood = zeros(length(d_grid), length(x_grid))
    
    for (i, d) in enumerate(d_grid)
        for (j, x) in enumerate(x_grid)
            # Compute log-likelihood across all stations
            log_lik = 0.0

            for (x_obs, w_o, u_o) in zip(x_obs_points, w_obs, u_obs)
                x_rel = x_obs - x
                w_pred = mogi_vertical(x_rel, a, d, G, ΔP_true, ν)
                u_pred = mogi_horizontal(x_rel, a, d, G, ΔP_true, ν)

                if use_w
                    log_lik += logpdf(Normal(w_pred, σ_obs), w_o)
                end

                if use_u
                    log_lik += logpdf(Normal(u_pred, σ_obs), u_o)
                end
            end

            likelihood[i, j] = exp(log_lik)
        end
    end
    
    return likelihood
end

# ╔═╡ 7985480c-8d6a-4cc8-9b13-ba8ffd52a548
# Compute likelihood
begin
    if use_w || use_u
        likelihood_2d = compute_likelihood_2d(
            x_grid, d_grid, x_obs_points, w_obs, u_obs, σ_obs, use_w, use_u
        )
    else
        likelihood_2d = ones(length(d_grid), length(x_grid))
    end
end;

# ╔═╡ ee3052c9-db2c-4ad3-9aa1-4cc5fafb42e5
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

# ╔═╡ 02a1cb74-0930-4567-80a5-be0d9b1d750b
# Compute posterior (unnormalized)
posterior_2d = prior_2d .* likelihood_2d;

# ╔═╡ 99d4319f-7d97-4e16-84c8-0cc12bade43e
# Normalize posterior
posterior_2d_norm = posterior_2d ./ sum(posterior_2d);


# ╔═╡ 4cea48ad-3c98-4978-a1bd-a89579e4fa20
# Compute marginal distributions
begin
    # Marginal for x (integrate over d)
    marginal_x = sum(posterior_2d_norm, dims=1)[:]
    marginal_x ./= sum(marginal_x)  # normalize
	
    # Marginal for d (integrate over x)
    marginal_d = sum(posterior_2d_norm, dims=2)[:]
    marginal_d ./= sum(marginal_d)  # normalize

    # Plot marginal for x
    x_km = x_grid ./ 1000
    max_mx = maximum(marginal_x)
    traces_x = [
        scatter(
            x=x_km,
            y=marginal_x,
            mode="lines",
            line=attr(color="blue", width=2),
            fill="tozeroy",
            name="Posterior"
        ),
        scatter(
            x=[x_true / 1000, x_true / 1000],
            y=[0, max_mx],
            mode="lines",
            line=attr(color="red", width=2, dash="dash"),
            name="True x₀"
        )
    ]
    layout_x = Layout(
        title="Marginal: ρ(x₀|d_obs)",
        xaxis=attr(title="Horizontal Position (km)"),
        yaxis=attr(title="Probability Density"),
        margin=attr(l=60, r=20, t=50, b=50),
        showlegend=false,
        width=450,
        height=300
    )
    p_mx = plot(Plot(traces_x, layout_x))

    # Plot marginal for d
    d_km = d_grid ./ 1000
    max_md = maximum(marginal_d)
    traces_d = [
        scatter(
            x=d_km,
            y=marginal_d,
            mode="lines",
            line=attr(color="blue", width=2),
            fill="tozeroy",
            name="Posterior"
        ),
        scatter(
            x=[d_true / 1000, d_true / 1000],
            y=[0, max_md],
            mode="lines",
            line=attr(color="red", width=2, dash="dash"),
            name="True d"
        )
    ]
    layout_d = Layout(
        title="Marginal: ρ(d|d_obs)",
        xaxis=attr(title="Depth (km)"),
        yaxis=attr(title="Probability Density"),
        margin=attr(l=60, r=20, t=50, b=50),
        showlegend=false,
        width=450,
        height=300
    )
    p_md = plot(Plot(traces_d, layout_d))

    PlutoUI.ExperimentalLayout.hbox([p_mx, p_md])
end

# ╔═╡ d8cdb3bc-bc46-4913-b189-f3677e6ea2df
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

# ╔═╡ d4553119-153d-450c-b275-03a40822cdd1
# Plotting helper function
function plot_2d_distribution(data, x_grid, d_grid, title_str; show_true=false)
    x_km = x_grid ./ 1000
    d_km = d_grid ./ 1000

    traces = [
        contour(
            x=x_km,
            y=d_km,
            z=data,
            colorscale="Jet",
            showscale=false,
            contours=attr(
                coloring="heatmap",
                showlabels=false
            )
        )
    ]

    if show_true
        push!(traces,
            scatter(
                x=[x_true / 1000],
                y=[d_true / 1000],
                mode="markers",
                marker=attr(color="red", size=10, symbol="star"),
                name="True Location"
            )
        )
    end

    layout = Layout(
        title=title_str,
        xaxis=attr(title="Horizontal Position (km)"),
        yaxis=attr(title="Depth (km)", autorange="reversed", scaleanchor="x"),
        margin=attr(l=60, r=20, t=50, b=50),
        showlegend=false,
		showcolorbar=false,
        width=420,
        height=360
    )

    return plot(Plot(traces, layout))
end

# ╔═╡ b5f4a012-8667-4841-8cb8-be770b32f876
# Create comparison plot
begin
    p1 = plot_2d_distribution(prior_2d, x_grid, d_grid, "Prior ρ(m)")
    p2 = plot_2d_distribution(likelihood_2d, x_grid, d_grid, "Likelihood ρ(d|m)", show_true=true)
    p3 = plot_2d_distribution(posterior_2d_norm, x_grid, d_grid, "Posterior σ(m|d)", show_true=true)

    WideCell(PlutoUI.ExperimentalLayout.hbox([p1, p2, p3]))
end

# ╔═╡ a2c46aa4-cb2f-4bbc-942f-b53fa85d1905
md"""
## Quantifying Uncertainty

From the posterior, we can extract useful statistics:

**Maximum a posteriori (MAP) estimate:** The most probable location

**Credible intervals:** Ranges containing, say, 95% of the posterior probability

**Standard deviations:** How spread out is the posterior?
"""

# ╔═╡ 6683d74d-84bb-4b31-b5a7-b20e7853c896
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

# ╔═╡ 66bbb923-6104-4a0e-8926-e87f75215644
results_md

# ╔═╡ 7ea9c05d-32ff-4392-ba38-3477c421d61d
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

# ╔═╡ 8b64052e-a6e5-43ba-9401-1dc93a961866
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

# ╔═╡ 212fcb7f-f0e2-474f-b19a-977565576ed8
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

# ╔═╡ 38365ea5-fb78-4dde-b66d-0bdfcc0e45f3
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

# ╔═╡ 95310c9a-ab50-4a6f-8191-c8ff546d1e81
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

# ╔═╡ 4fa06655-9fac-4644-a60f-19d2ecd89b03
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

# ╔═╡ 464d55bf-43a5-4c60-9d29-741a98b19486
md"""
## References

1. **Mogi, K. (1958).** Relations between the eruptions of various volcanoes and the deformations of the ground surfaces around them. *Bulletin of the Earthquake Research Institute*, 36, 99-134.

2. **Segall, P. (2010).** *Earthquake and Volcano Deformation.* Princeton University Press.

3. **Tarantola, A. (2005).** *Inverse Problem Theory and Methods for Model Parameter Estimation.* SIAM.

4. **Fichtner, A. (2021).** Lecture Notes on Inverse Theory. ETH Zurich.

5. **Mosegaard, K., & Tarantola, A. (2002).** Probabilistic approach to inverse problems. *International Geophysics Series*, 81(A), 237-268.

6. **Interactive Mogi Demo:** [GSC Community Codes](https://gscommunitycodes.usf.edu/geoscicommunitycodes/public/numeracy/numeracy_mogi/mogi.html)
"""

# ╔═╡ df4a8488-eebf-493e-9f09-4a83e315cd88
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

# ╔═╡ 9c7b7b7a-9f2d-4f2a-8c4c-6a1c8d1b4a21
md"""
## Forward Model Demo (Surface Deformation)

This plot shows the **predicted surface deformation** as a function of horizontal distance for the current **true** source parameters.

- White curve: vertical displacement $w$ (uplift/subsidence)
- Yellow curve: horizontal displacement $u$

Use the **ΔP slider** to see how amplitude scales with pressure change, and toggle **u**/**w** to explore each component.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
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
project_hash = "362bce2240fbd1510b4f5e29c4ac97eb859eda78"

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
git-tree-sha1 = "28145feabf717c5d65c1d5e09747ee7b1ff3ed13"
uuid = "ec057cc2-7a8d-4b58-b3b3-92acb9f63b42"
version = "2.6.3"

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
# ╟─320c8808-098a-11f1-8de1-b1d853234290
# ╟─79eb11d5-d946-48db-9f10-ca6ffa3d7df5
# ╟─a1ee4121-0b45-4d7c-9967-4b63b3747b1a
# ╟─3fee472d-1599-4e3b-a22f-860b4a6850f3
# ╟─b5f4a012-8667-4841-8cb8-be770b32f876
# ╟─39d40806-6bae-4044-a115-c9cbeba65625
# ╟─4cea48ad-3c98-4978-a1bd-a89579e4fa20
# ╟─02f2687d-6cbb-47ba-a5ae-e978064ef535
# ╟─e1c578d6-6d8a-448d-8ae5-1cb00ad14dbb
# ╟─7b2c65b3-ef5a-41b8-ae0a-7e5407f85a7b
# ╠═d664b09d-4fd8-4e9c-8bf7-062a6b67bc0a
# ╠═cd4fab19-6aff-4254-9824-d55e4a7c8692
# ╠═f89bdd99-9418-4a44-8fd7-5ad087d760e2
# ╠═d1dfb877-fa16-40a6-bc1d-180df15b4889
# ╠═c98b8476-877b-4b6f-b179-37387a98b12c
# ╠═e8f39318-0d31-40b0-be32-723a19554080
# ╠═47f0ad9a-250f-4066-b2e6-f5c02f8a7068
# ╠═0a7e7ee0-3753-464c-b690-54a4b36d84eb
# ╟─125908ba-18cb-4214-8032-584cb5bd0ad0
# ╠═7985480c-8d6a-4cc8-9b13-ba8ffd52a548
# ╟─ee3052c9-db2c-4ad3-9aa1-4cc5fafb42e5
# ╠═02a1cb74-0930-4567-80a5-be0d9b1d750b
# ╠═99d4319f-7d97-4e16-84c8-0cc12bade43e
# ╟─d8cdb3bc-bc46-4913-b189-f3677e6ea2df
# ╠═d4553119-153d-450c-b275-03a40822cdd1
# ╟─a2c46aa4-cb2f-4bbc-942f-b53fa85d1905
# ╠═6683d74d-84bb-4b31-b5a7-b20e7853c896
# ╠═66bbb923-6104-4a0e-8926-e87f75215644
# ╠═7ea9c05d-32ff-4392-ba38-3477c421d61d
# ╠═8b64052e-a6e5-43ba-9401-1dc93a961866
# ╠═212fcb7f-f0e2-474f-b19a-977565576ed8
# ╠═38365ea5-fb78-4dde-b66d-0bdfcc0e45f3
# ╠═95310c9a-ab50-4a6f-8191-c8ff546d1e81
# ╠═4fa06655-9fac-4644-a60f-19d2ecd89b03
# ╟─464d55bf-43a5-4c60-9d29-741a98b19486
# ╠═df4a8488-eebf-493e-9f09-4a83e315cd88
# ╠═e5ee82b0-5742-4b34-9e38-f079d75a1ad5
# ╠═9c7b7b7a-9f2d-4f2a-8c4c-6a1c8d1b4a21
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
