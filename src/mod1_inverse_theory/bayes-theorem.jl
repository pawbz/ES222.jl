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

# ╔═╡ 2f95a58c-1f58-4c1d-9f51-c7c7c2b9b0ef
begin
	using PlutoUI
	using Random
	using Statistics
	using LaTeXStrings
	using TikzPictures
end

# ╔═╡ 5f1a2e3b-7c4d-5f2a-9d3e-1a2f4e5c6b7d
md"""
## Bayes' Theorem for Inverse Problems

### Goal
Infer hidden parameters $\mathbf{z}$ from observed data $\mathbf{x}_{\mathrm{obs}}$.

### Bayes' Theorem
$$p(\mathbf{z}\mid \mathbf{x}=\mathbf{x}_{\mathrm{obs}}) = \frac{p(\mathbf{x}=\mathbf{x}_{\mathrm{obs}}\mid \mathbf{z})\,p(\mathbf{z})}{p(\mathbf{x}=\mathbf{x}_{\mathrm{obs}})}$$

### Components

- **Prior:** $p(\mathbf{z})$ — encodes what we believe about $\mathbf{z}$ *before* observing data
- **Likelihood:** $p(\mathbf{x}=\mathbf{x}_{\mathrm{obs}}\mid \mathbf{z})$ — measures how well the data fit given model parameters  
- **Evidence:** $p(\mathbf{x}=\mathbf{x}_{\mathrm{obs}})$ — a normalizing constant (independent of $\mathbf{z}$)
- **Posterior:** $p(\mathbf{z}\mid \mathbf{x}=\mathbf{x}_{\mathrm{obs}})$ — the solution; our updated belief about $\mathbf{z}$

In the interactive visualization below, move the sliders and select different observation levels to see how the **posterior** (yellow bars) changes!
"""

# ╔═╡ 1b845f20-f192-4f06-8b4e-c4f7bdbb8b4f
md"""
---
**Forward model:** $x = \frac{2z}{v} + \text{noise}$  
We show how noise spreads the observations and changes the **posterior** over discrete thickness categories.

Move the sliders, then click **Resample** to redraw the circles.
"""

# ╔═╡ 4f74b95d-7b85-451b-8f38-31b4ffb4a3f0
md"""### Experiment controls"""

# ╔═╡ ccec4ec2-04cf-11f1-adee-dd8efaf97688
PlutoUI.ExperimentalLayout.hbox([
	md"**v (km/s):**",
	(@bind v Slider(4.5:0.1:9.5, default=7.0, show_value=true)),
	md"**Noise:**",
	(@bind noise_halfwidth Slider(0.0:0.01:3, default=0.5, show_value=true)),
	md"**Points:**",
	(@bind n_per_level Slider(25:5:100, default=25, show_value=true))
])

# ╔═╡ 9e90bff7-718e-4cad-a1b8-d146a9238921
md"""
### Observed Data"""

# ╔═╡ ccec51ba-04cf-11f1-94cc-cb76bc855fdd
@bind obs_level Radio(["very low", "low", "medium", "high", "very high"], default="medium")

# ╔═╡ d05c7b3a-04cb-11f1-94aa-d781b87d24c1
md"""### Prior probabilities"""

# ╔═╡ ccec5386-04cf-11f1-9e42-6bccc4fea46a
PlutoUI.ExperimentalLayout.hbox([
	md"**Thin:**",
	(@bind prior_thin Slider(0.01:0.01:2, default=1.0, show_value=true)),
	md"**Medium:**",
	(@bind prior_medium Slider(0.01:0.01:2, default=1.0, show_value=true)),
	md"**Thick:**",
	(@bind prior_thick Slider(0.01:0.01:2, default=1.0, show_value=true))
])

# ╔═╡ 2b8a8f41-1950-4d89-9f95-3b07f4a7b1ff
PlutoUI.ExperimentalLayout.hbox([
	md"**Random seed:**",
	(@bind base_seed Slider(1:1:999, default=42, show_value=true)),
	(@bind reseed CounterButton("Resample"))
])

# ╔═╡ 7b0d99e8-6f73-47f9-ae68-bd0b76734e58
md"""
**Current:**  
- $v$ = $(v) km/s  
- noise half-width = **$(noise_halfwidth)**  
- points per level = **$(n_per_level)**  
- observed level = **$(obs_level)**  
- seed = **$(base_seed)** (press *Resample* to redraw)

**Prior (unnormalized weights):**
- thin: **$(round(prior_thin, digits=2))**
- medium: **$(round(prior_medium, digits=2))**
- thick: **$(round(prior_thick, digits=2))**
"""

# ╔═╡ 5a8fe1b4-5ed2-4ff4-b8f2-1e8609c1c59a
begin
	# Model setup
	z_levels = [8.4, 6.3, 2.1]               # thin, medium, thick (arbitrary units)
	labels = ["thin", "medium", "thick"]
	y_levels = [3.0, 2.0, 1.0]               # visual rows

	# Observation level bins (5 discrete levels)
	obs_bins = Dict(
		"very low" => (0.0, 1.2),
		"low" => (1.2, 2.4),
		"medium" => (2.4, 3.6),
		"high" => (3.6, 4.8),
		"very high" => (4.8, 6.0)
	)
	x_low, x_high = obs_bins[obs_level]

	# Base travel-time per level
	x_base = 2 .* z_levels ./ v

	# Normalize prior weights
	prior_weights = [prior_thin, prior_medium, prior_thick]
	prior_normalized = prior_weights ./ sum(prior_weights)
	
	# Calculate number of samples for each level based on prior probabilities
	n_samples_per_level = round.(Int, n_per_level .* prior_normalized)
	# Ensure at least 1 sample per level if n_per_level > 0
	if n_per_level > 0
		n_samples_per_level = max.(n_samples_per_level, 1)
	end

	# Seed for reproducibility; reseed button increments a counter
	Random.seed!(base_seed + reseed)

	# Generate points (uniform noise) with prior-weighted samples
	points = map(1:3) do k
		n_samples = n_samples_per_level[k]
		x = x_base[k] .+ (2 .* rand(n_samples) .- 1) .* noise_halfwidth
		y = y_levels[k] .+ (rand(n_samples) .- 0.5) .* 0.25
		(; x, y)
	end

	# Count points in observed column
	counts = [sum((p.x .>= x_low) .& (p.x .<= x_high)) for p in points]
	posterior = counts ./ (sum(counts) == 0 ? 1 : sum(counts))

	(; z_levels, labels, y_levels, x_base, points, counts, posterior, x_low, x_high, prior_normalized)
end

# ╔═╡ 8b8f77f8-90a2-4c48-aea4-0d3d7393e2c2
begin
    # Build TikZ commands with interpolation
    point_cmds = String[]
    for (k, p) in enumerate(points)
        for j in eachindex(p.x)
            x = round(p.x[j], digits=2)
            y = round(p.y[j], digits=2)
            push!(point_cmds, "\\fill[cyan!80, opacity=0.7] ($(x), $(y)) circle (2.2pt);")
        end
    end

    posterior_cmds = String[]
    for (i, h) in enumerate(posterior)
        y0 = (3 - i) * 1.2
        y1 = y0 + 1.2
        hh = round(0.4 * h, digits=3)
        push!(posterior_cmds, "\\fill[yellow!70, opacity=0.7] (0,$(y0)) rectangle ($(hh),$(y1));")
    end

    prior_cmds = String[]
    for (i, h) in enumerate(prior_normalized)
        y0 = (3 - i) * 1.2
        y1 = y0 + 1.2
        hh = round(0.4 * h, digits=3)
        push!(prior_cmds, "\\fill[red!70, opacity=0.7] (0,$(y0)) rectangle ($(hh),$(y1));")
    end

    label_cmds = [
        "\\node[white, rotate=90, anchor=north east] at (0.3,-0.1) {very low};",
        "\\node[white, rotate=90, anchor=north east] at (1.5,-0.1) {low};",
        "\\node[white, rotate=90, anchor=north east] at (2.7,-0.1) {medium};",
        "\\node[white, rotate=90, anchor=north east] at (3.9,-0.1) {high};",
        "\\node[white, rotate=90, anchor=north east] at (5.1,-0.1) {very high};"
    ]

    preamble = """
  \\usepackage[T1]{fontenc}
  \\usepackage{lmodern}
  \\usepackage{tikz}
  \\usepackage{xcolor}
  \\usetikzlibrary{calc}
  """

        tp = TikzPicture("""
\\resizebox{6cm}{!}{%
\\begin{tikzpicture}[scale=1.0]
  \\def\\gridwidth{6}
  \\def\\gridheight{3.6}
  \\def\\margwidth{1.2}
    \\pgfmathsetmacro{\\xlow}{$(x_low)}
    \\pgfmathsetmacro{\\xhigh}{$(x_high)}

  % highlight observed column
    \\fill[yellow!20, opacity=0.3] (\\xlow,0) rectangle (\\xhigh,\\gridheight);
    \\pgfmathsetmacro{\\xcenter}{(\\xlow + \\xhigh) / 2}
    \\node[white] at (\\xcenter, \\gridheight+0.5) {observation};

  % main grid
  \\draw[thick, white] (0,0) rectangle (\\gridwidth,\\gridheight);
  \\foreach \\x in {1.2,2.4,3.6,4.8} {\\draw[thick, white] (\\x,0) -- (\\x,\\gridheight);}
  \\foreach \\y in {1.2,2.4} {\\draw[thick, white] (0,\\y) -- (\\gridwidth,\\y);}

  % axes labels
  $(join(label_cmds, "\n"))
  \\node[white, anchor=west] at (\\gridwidth+0.15,3.0) {thin};
  \\node[white, anchor=west] at (\\gridwidth+0.15,1.8) {medium};
  \\node[white, anchor=west] at (\\gridwidth+0.15,0.6) {thick};

  % points
  $(join(point_cmds, "\n"))

  % prior (left)
  \\begin{scope}[shift={(-\\margwidth,0)}]
    $(join(prior_cmds, "\n"))
    \\draw[white, thick] (0,0) -- (0,\\gridheight);
    \\node[white, rotate=90, anchor=south] at (0.0, -0.4) {prior \$p(z)\$};
  \\end{scope}

  % posterior (right)
  \\begin{scope}[shift={(8cm,0)}]
    $(join(posterior_cmds, "\n"))
    \\draw[white, thick] (0,0) -- (0,\\gridheight);
        \\node[white, rotate=90, anchor=south] at (0.0, -0.4) {posterior \$p(z|x_{\\mathrm{obs}})\$};
						      \\node[white, rotate=90, anchor=south] at (0.5, 0.0) {};
  \\end{scope}
\\end{tikzpicture}%
}
""", options="", preamble=preamble, width="30cm")

    WideCell(tp)
end

# ╔═╡ 4a69aa11-3e7f-457e-a5a6-b713d46ea2a2
md"""
## Count check (observed column)

- thin: **$(counts[1])**
- medium: **$(counts[2])**
- thick: **$(counts[3])**

Posterior $p(z\mid x_\mathrm{obs})$ (normalized counts):  
**$(round.(posterior, digits=2))**
"""

# ╔═╡ 1f53ccbe-54f3-45a8-88c0-4006d7d8aa6c
md"""
### Teaching notes
- The **highlighted column** is the observation bin.  
- Each redraw re-samples noise, so the **posterior moves** but remains consistent with counting.  
- Increase noise to see thicker tails and more mixing across bins.
"""

# ╔═╡ 3c8d2f1a-5e4c-4b3d-8a1c-9f8e7c6d5e4f
md"""
## Discrete Example: Crustal Thickness

### Hidden state ($z$): Crustal thickness category
- **Thin:** 20–30 km
- **Medium:** 30–45 km  
- **Thick:** 45–60 km

### Data ($x$): Discrete travel-time bins (seconds)
- **Very low:** 6–8 s
- **Low:** 8–11 s
- **Medium:** 11–14 s
- **High:** 14–17 s
- **Very high:** 17–20 s

### Simple forward model
$$t \approx \frac{2h}{v}, \quad v \approx 6~\text{km/s}$$

The visualization below shows how measurements (cyan circles) are distributed across different observation bins, and how the posterior distribution (yellow bars on right) is updated based on which bin you select.
"""

# ╔═╡ 734acb86-04cb-11f1-a988-2f4d1b061fa6
md"""
## What Does Solving an Inverse Problem Mean?

### Goal
Estimate the probability density of the hidden parameters $\mathbf{z}$.

### Key Points

1. **For a scalar case:** Learn $p(\mathbf{z}\mid \mathbf{x}=\mathbf{x}_{\mathrm{obs}})$
2. **Point estimate:** The mean of the posterior gives a single best estimate
3. **Uncertainty quantification:** The spread (variance/standard deviation) quantifies how much we don't know

### Interpretation

- The posterior distribution encodes all information about $\mathbf{z}$ given the observed data
- A narrow distribution = high confidence in the estimate
- A wide distribution = high uncertainty (many plausible values)
- The figure above shows a single-peaked (unimodal) posterior — but not all problems are this simple!
"""

# ╔═╡ 8f9a1b2c-3d4e-5f6a-7b8c-9d0e1f2a3b4c
md"""
## Sum Rule of Probability (Discrete Case)

### Statement
For a discrete random variable $z$ that can take values $z_1, z_2, \ldots, z_N$, the probabilities must sum to 1:

$$\sum_{i=1}^{N} p(z_i) = 1$$

This is a fundamental axiom of probability theory — the total probability across all possible outcomes must equal unity.

### Application to Inverse Problems

- **Prior distribution:** $\sum_i p(z_i) = 1$
- **Posterior distribution:** $\sum_i p(z_i \mid \mathbf{x}=\mathbf{x}_{\mathrm{obs}}) = 1$

The visualization below shows three discrete outcomes with probabilities that sum to exactly 1.0.
"""

# ╔═╡ 9e8b7c6d-4e5f-6a7b-8c9d-0e1f2a3b4c5d
begin
    # Example probabilities that sum to 1
    p1, p2, p3 = 0.25, 0.45, 0.30
    prob_labels = ["z₁", "z₂", "z₃"]
    prob_values = [p1, p2, p3]
    prob_sum = sum(prob_values)

    # Build TikZ bar chart
    bar_cmds = String[]
    for (i, (prob, label)) in enumerate(zip(prob_values, prob_labels))
        x_pos = (i - 1) * 1.5
        height = prob * 4  # Scale for visualization
        push!(bar_cmds, "\\fill[cyan!70] ($(x_pos),0) rectangle ($(x_pos+1.0),$(round(height, digits=3)));")
        push!(bar_cmds, "\\node[white, above] at ($(x_pos+0.5),$(round(height, digits=3))) {\$$(round(prob, digits=2))\$};")
        push!(bar_cmds, "\\node[white, below] at ($(x_pos+0.5),-0.3) {\$$(label)\$};")
    end

    sum_label = "\\node[white] at (2.25, -1.2) {Sum = \$$(round(prob_sum, digits=2))\$};"

    preamble_sum = """
  \\usepackage[T1]{fontenc}
  \\usepackage{lmodern}
  \\usepackage{tikz}
  \\usepackage{xcolor}
  """

    tp_sum = TikzPicture("""
\\begin{tikzpicture}[scale=1.2]
  % Title
  \\node[white] at (2.25, 2.5) {Sum Rule: \$\\sum_i p(z_i) = 1\$};
  
  % Draw bars
  $(join(bar_cmds, "\n  "))
  
  % Draw axes
  \\draw[white, thick, ->] (-0.5,0) -- (5,0) node[right] {\$z\$};
  \\draw[white, thick, ->] (-0.5,0) -- (-0.5,2) node[above] {\$p(z)\$};
  
  % Sum label
  $(sum_label)
\\end{tikzpicture}
""", options="", preamble=preamble_sum, width="20cm")

    WideCell(tp_sum)
end

# ╔═╡ 1a2b3c4d-5e6f-7a8b-9c0d-1e2f3a4b5c6d
md"""
## Product Rule of Probability (Discrete Case)

### Statement
The joint probability of two events can be expressed as:

$$p(z, x) = p(x \mid z) \, p(z) = p(z \mid x) \, p(x)$$

This is a fundamental relationship that connects:
- **Joint probability:** $p(z, x)$ — probability of both $z$ and $x$ occurring
- **Conditional probability:** $p(x \mid z)$ — probability of $x$ given $z$
- **Marginal probability:** $p(z)$ — probability of $z$ alone

### Application to Inverse Problems

The product rule is the foundation of Bayes' theorem. Rearranging:

$$p(z \mid x) = \frac{p(x \mid z) \, p(z)}{p(x)}$$

The visualization below demonstrates: $p(z, x) = p(x \mid z) \times p(z)$
"""

# ╔═╡ 2b3c4d5e-6f7a-8b9c-0d1e-2f3a4b5c6d7e
let
    # Example values for product rule
    p_z = 0.6          # P(z)
    p_x_given_z = 0.7  # P(x|z)
    p_z_and_x = p_z * p_x_given_z  # P(z,x) = P(x|z) * P(z)

    preamble_prod = """
  \\usepackage[T1]{fontenc}
  \\usepackage{lmodern}
  \\usepackage{tikz}
  \\usepackage{xcolor}
  """

    tp_prod = TikzPicture("""
\\begin{tikzpicture}[scale=1.2]
  % Title
  \\node[white] at (3.5, 4.0) {Product Rule: \$p(z,x) = p(x|z) \\cdot p(z)\$};
  
  % First box: P(z)
  \\fill[cyan!70] (0,0) rectangle (2,2.4);
  \\node[white] at (1,1.2) {\$p(z)\$};
  \\node[white] at (1,0.5) {\$$(round(p_z, digits=2))\$};
  
  % Multiplication symbol
  \\node[white] at (2.8,1.2) {\$\\times\$};
  
  % Second box: P(x|z)
  \\fill[yellow!70] (3.6,0) rectangle (5.6,2.8);
  \\node[white] at (4.6,1.7) {\$p(x|z)\$};
  \\node[white] at (4.6,0.7) {\$$(round(p_x_given_z, digits=2))\$};
  
  % Equals symbol
  \\node[white] at (6.4,1.2) {\$=\$};
  
  % Third box: P(z,x)
  \\fill[green!60] (7.2,0) rectangle (9.2,1.68);
  \\node[white] at (8.2,1.1) {\$p(z,x)\$};
  \\node[white] at (8.2,0.35) {\$$(round(p_z_and_x, digits=3))\$};
  
  % Labels below
  \\node[white] at (1,-0.5) {marginal};
  \\node[white] at (4.6,-0.5) {conditional};
  \\node[white] at (8.2,-0.5) {joint};
\\end{tikzpicture}
""", options="", preamble=preamble_prod)

    WideCell(tp_prod)
end

# ╔═╡ 3c4d5e6f-7a8b-9c0d-1e2f-3a4b5c6d7e8f
md"""
## Building Intuition: Counting Probabilities

### Interactive Grid Visualization

This interactive visualization helps build intuition for how probabilities are computed from data by **counting points**.

- **Grid:** Rows represent model parameters $z$ (thin, medium, thick), columns represent observations $x$ (very low through very high)
- **Points:** Each cyan circle is a data sample, randomly distributed according to the forward model
- **Select a cell:** Choose a row (parameter) and column (observation) to see all relevant probabilities

### Probabilities by Counting

- **Joint:** $p(z, x) = \\frac{\\text{# points in selected cell}}{\\text{# total points}}$
- **Marginal (z):** $p(z) = \\frac{\\text{# points in selected row}}{\\text{# total points}}$
- **Marginal (x):** $p(x) = \\frac{\\text{# points in selected column}}{\\text{# total points}}$
- **Conditional:** $p(x \\mid z) = \\frac{\\text{# points in selected cell}}{\\text{# points in selected row}}$
- **Posterior:** $p(z \\mid x) = \\frac{\\text{# points in selected cell}}{\\text{# points in selected column}}$

Move the sliders below to explore how these probabilities change!
"""

# ╔═╡ 4d5e6f7a-8b9c-0d1e-2f3a-4b5c6d7e8f9a
WideCell(PlutoUI.ExperimentalLayout.hbox([
	md"**Observation index (1=very low, 5=very high):**",
	(@bind obs_idx Slider(1:5, default=2, show_value=true)),
	md"**Parameter index (1=thin, 2=medium, 3=thick):**",
	(@bind param_idx Slider(1:3, default=1, show_value=true))
]))

# ╔═╡ 5e6f7a8b-9c0d-1e2f-3a4b-5c6d7e8f9a0b
begin
	# Generate random points for grid (use same structure as main visualization)
	Random.seed!(42)
	
	# Grid parameters
	z_names = ["thin", "medium", "thick"]
	x_names = ["very low", "low", "medium", "high", "very high"]
	
	# Generate points distributed across 3x5 grid
	n_points_per_cell = 8
	grid_points = []
	for z_idx in 1:3
		for x_idx in 1:5
			# Base position for cell
			x_base = (x_idx - 1) * 0.2 + 0.6
			y_base = (3 - z_idx) * 1.2 + 0.6
			
			# Add random points in this cell
			for _ in 1:n_points_per_cell
				x = x_base + (rand() - 0.5) * 0.9
				y = y_base + (rand() - 0.5) * 0.9
				push!(grid_points, (x=x, y=y, z_idx=z_idx, x_idx=x_idx))
			end
		end
	end
	
	# Count points in different regions
	n_total = length(grid_points)
	n_in_cell = count(p -> p.z_idx == param_idx && p.x_idx == obs_idx, grid_points)
	n_in_row = count(p -> p.z_idx == param_idx, grid_points)
	n_in_col = count(p -> p.x_idx == obs_idx, grid_points)
	
	# Compute probabilities
	p_joint = n_in_cell / n_total
	p_z = n_in_row / n_total
	p_x = n_in_col / n_total
	p_x_given_z = n_in_row > 0 ? n_in_cell / n_in_row : 0.0
	p_z_given_x = n_in_col > 0 ? n_in_cell / n_in_col : 0.0
	
	(; grid_points, n_total, n_in_cell, n_in_row, n_in_col, 
	   p_joint, p_z, p_x, p_x_given_z, p_z_given_x, z_names, x_names)
end

# ╔═╡ 6f7a8b9c-0d1e-2f3a-4b5c-6d7e8f9a0b1c
begin
	# Build TikZ visualization
	grid_pt_cmds = String[]
	for pt in grid_points
		# Highlight selected cell points in yellow, others in cyan
		if pt.z_idx == param_idx && pt.x_idx == obs_idx
			push!(grid_pt_cmds, L"""\fill[yellow, opacity=0.8] (%$(round(pt.x, digits=2)), %$(round(pt.y, digits=2))) circle (2pt);""")
		else
			push!(grid_pt_cmds, L"""\fill[cyan!60, opacity=0.6] (%$(round(pt.x, digits=2)), %$(round(pt.y, digits=2))) circle (1.5pt);""")
		end
	end
	
	# Selected cell coordinates
	sel_x_low = (obs_idx - 1) * 1.2
	sel_x_high = obs_idx * 1.2
	sel_y_low = (3 - param_idx) * 1.2
	sel_y_high = (4 - param_idx) * 1.2
	
	# Format probabilities for display
	z_label = z_names[param_idx]
	x_label = x_names[obs_idx]
	
	preamble_grid = """
  \\usepackage[T1]{fontenc}
  \\usepackage{lmodern}
  \\usepackage{tikz}
  \\usepackage{xcolor}
  \\usepackage{amsmath}
  """
	
	tp_grid = TikzPicture(L"""
						  \resizebox{10cm}{!}{%
	\begin{tikzpicture}[scale=1]
  % Highlight selected cell
  \fill[orange!30, opacity=0.4] (%$(sel_x_low),%$(sel_y_low)) rectangle (%$(sel_x_high),%$(sel_y_high));
  % Highlight selected row (z)
  \fill[red!20, opacity=0.2] (0,%$(sel_y_low)) rectangle (6.0,%$(sel_y_high));
  % Highlight selected column (x)
  \fill[blue!20, opacity=0.2] (%$(sel_x_low),0) rectangle (%$(sel_x_high),3.6);
  % Draw grid
  \draw[thick, white] (0,0) rectangle (6.0,3.6);
  \foreach \x in {1.2,2.4,3.6,4.8} {\draw[thick, white] (\x,0) -- (\x,3.6);}
  \foreach \y in {1.2,2.4} {\draw[thick, white] (0,\y) -- (6.0,\y);}
  % Row labels (z)
  \node[white, anchor=east] at (-0.1,3.0) {thin};
  \node[white, anchor=east] at (-0.1,1.8) {medium};
  \node[white, anchor=east] at (-0.1,0.6) {thick};
  % Column labels (x)
  \node[white, rotate=90, anchor=east] at (0.6,-0.15) {v.low};
  \node[white, rotate=90, anchor=east] at (1.8,-0.15) {low};
  \node[white, rotate=90, anchor=east] at (3.0,-0.15) {med};
  \node[white, rotate=90, anchor=east] at (4.2,-0.15) {high};
  \node[white, rotate=90, anchor=east] at (5.4,-0.15) {v.high};
						   \node[white, rotate=90, anchor=east] at (5.4,7) {};
						  % Plot points
  %$(join(grid_pt_cmds, L";"))	
						   % Probability formulas and values (right side)
  \node[white, anchor=west] at (6.5, 3.2) {
    Selected: \$z=%$(z_label)\$, \$x=%$(x_label)\$
  };
  \node[white, anchor=west] at (6.5, 2.6) {
    \$p(z,x) = %$(n_in_cell)/%$(n_total) = %$(round(p_joint, digits=3))\$
  };
  \node[white, anchor=west] at (6.5, 2.1) {
    \$p(z) = %$(n_in_row)/%$(n_total) = %$(round(p_z, digits=3))\$
  };
  \node[white, anchor=west] at (6.5, 1.6) {
    \$p(x) = %$(n_in_col)/%$(n_total) = %$(round(p_x, digits=3))\$
  };
  \node[white, anchor=west] at (6.5, 1.1) {
    \$p(x|z) = %$(n_in_cell)/%$(n_in_row) = %$(round(p_x_given_z, digits=3))\$
  }; 
  \node[white, anchor=west] at (6.5, 0.6) {
    \$p(z|x) = %$(n_in_cell)/%$(n_in_col) = %$(round(p_z_given_x, digits=3))\$
  };	
	\end{tikzpicture}
						  }
""", options=" ",width="30cm", preamble=preamble_grid)
	
	WideCell(tp_grid)
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
TikzPictures = "37f6aa50-8035-52d0-81c2-5a1d08754b2d"

[compat]
LaTeXStrings = "~1.4.0"
PlutoUI = "~0.7.79"
TikzPictures = "~3.5.1"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.4"
manifest_format = "2.0"
project_hash = "262d8c8d4cb91cdc83b03e1596e56e7f2743bb6c"

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

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1b96ea4a01afe0ea4090c5c8039690672dd13f2e"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.9+0"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "2ac646d71d0d24b44f3f8c84da8c9f4d70fb67df"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.4+0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "67e11ee83a43eb71ddc950302c53bf33f0690dfe"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.1"
weakdeps = ["StyledStrings"]

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.3.0+1"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.7.0"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "27af30de8b5445644e8ffe3bcb0d72049c089cf1"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.7.3+0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Zlib_jll"]
git-tree-sha1 = "f85dac9a96a01087df6e3a749840015a0ca3817d"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.17.1+0"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "2c5512e11c791d1baed2049c5652441b28fc6a31"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.4+0"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "b0036b392358c80d2d2124746c2bf3d48d457938"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.82.4+0"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a6dbda1fd736d60cc477d99f2e7a042acfa46e8"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.15+0"

[[deps.HarfBuzz_ICU_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "HarfBuzz_jll", "ICU_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "6ccbc4fdf65c8197738c2d68cc55b74b19c97ac2"
uuid = "655565e8-fb53-5cb3-b0cd-aec1ca0647ea"
version = "2.8.1+0"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

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

[[deps.ICU_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "20b6765a3016e1fca0c9c93c80d50061b94218b7"
uuid = "a51ab1cf-af8e-5615-a023-bc2c838bba6b"
version = "69.1.0+0"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "0ee181ec08df7d7c911901ea38baf16f755114dc"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "1.0.0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "0533e564aae234aff59ab625543145446d8b6ec2"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.7.1"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6893345fd6658c8e475d40155789f4860ac3b21"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.1.4+0"

[[deps.JuliaSyntaxHighlighting]]
deps = ["StyledStrings"]
uuid = "ac6e5ff7-fb65-4e79-a425-ec3bc9c03011"
version = "1.12.0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aaafe88dccbd957a8d82f7d05be9b69172e0cee3"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "4.0.1+0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "eb62a3deb62fc6d8822c0c4bef73e4412419c5d8"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "18.1.8+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1c602b1127f4751facb671441ca72715cc95938a"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.3+0"

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

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "27ecae93dd25ee0909666e6835051dd684cc035e"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+2"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "be484f5c92fad0bd8acfef35fe017900b0b73809"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.18.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "3acf07f130a76f87c041cfb2ff7d7284ca67b072"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.41.2+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "f04133fe05eff1667d2054c53d59f9122383fe05"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.7.2+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "2a7a12fc0a4e7fb773450d17975322aa77142106"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.41.2+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.12.0"

[[deps.LittleCMS_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll"]
git-tree-sha1 = "8e6a74641caf3b84800f2ccd55dc7ab83893c10b"
uuid = "d3a379c0-f9a3-5b72-a4c0-6bf4d2e8af0f"
version = "2.17.0+0"

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

[[deps.OpenJpeg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libtiff_jll", "LittleCMS_jll", "libpng_jll"]
git-tree-sha1 = "215a6666fee6d6b3a6e75f2cc22cb767e2dd393a"
uuid = "643b3616-a352-519d-856d-80112ee9badc"
version = "2.5.5+0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.4+0"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.44.0+1"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "db76b1ecd5e9715f3d043cec13b2ec93ce015d53"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.44.2+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.12.1"

    [deps.Pkg.extensions]
    REPLExt = "REPL"

    [deps.Pkg.weakdeps]
    REPL = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Downloads", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "3ac7038a98ef6977d44adeadc73cc6f596c08109"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.79"

[[deps.Poppler_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "JpegTurbo_jll", "LibCURL_jll", "Libdl", "Libtiff_jll", "OpenJpeg_jll", "libpng_jll"]
git-tree-sha1 = "7dbfb7f61c3aa5def7b7dad3fa344c1c2858a83b"
uuid = "9c32591e-4766-534b-9725-b71a8799265b"
version = "24.6.0+0"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "522f093a29b31a93e34eaea17ba055d850edea28"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.5.1"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
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

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.TikzPictures]]
deps = ["LaTeXStrings", "Poppler_jll", "tectonic_jll"]
git-tree-sha1 = "875854f63fbe215b554390efd249bfbef1418c31"
uuid = "37f6aa50-8035-52d0-81c2-5a1d08754b2d"
version = "3.5.1"

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

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "80d3930c6347cfce7ccf96bd3bafdf079d9c0390"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.13.9+0"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "9cce64c0fdd1960b597ba7ecda2950b5ed957438"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.8.2+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "b5899b25d17bf1889d25906fb9deed5da0c15b3b"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.12+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aa1261ebbac3ccc8d16558ae6799524c450ed16b"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.13+0"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "52858d64353db33a56e13c341d7bf44cd0d7b309"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.6+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "a4c0ee07ad36bf8bbce1c3bb52d21fb1e0b987fb"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.7+0"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "7ed9347888fac59a618302ee38216dd0379c480d"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.12+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXau_jll", "Xorg_libXdmcp_jll"]
git-tree-sha1 = "bfcaf7ec088eaba362093393fe11aa141fa15422"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.17.1+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a63799ff68005991f9d9491b6e95bd3478d783cb"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.6.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.3.1+2"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "446b23e73536f84e8037f5dce465e92275f6a308"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.7+1"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.15.0+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "6ab498eaf50e0495f89e7a5b582816e2efb95f64"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.54+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.64.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.7.0+0"

[[deps.tectonic_jll]]
deps = ["Artifacts", "Fontconfig_jll", "FreeType2_jll", "Graphite2_jll", "HarfBuzz_ICU_jll", "HarfBuzz_jll", "ICU_jll", "JLLWrappers", "Libdl", "OpenSSL_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "b62c5dcf5d80a82e40d58b908b8eca27a54f215b"
uuid = "d7dd28d6-a5e6-559c-9131-7eb760cdacc5"
version = "0.15.0+0"
"""

# ╔═╡ Cell order:
# ╠═2f95a58c-1f58-4c1d-9f51-c7c7c2b9b0ef
# ╟─5f1a2e3b-7c4d-5f2a-9d3e-1a2f4e5c6b7d
# ╟─1b845f20-f192-4f06-8b4e-c4f7bdbb8b4f
# ╟─4f74b95d-7b85-451b-8f38-31b4ffb4a3f0
# ╟─ccec4ec2-04cf-11f1-adee-dd8efaf97688
# ╟─9e90bff7-718e-4cad-a1b8-d146a9238921
# ╟─ccec51ba-04cf-11f1-94cc-cb76bc855fdd
# ╟─d05c7b3a-04cb-11f1-94aa-d781b87d24c1
# ╟─ccec5386-04cf-11f1-9e42-6bccc4fea46a
# ╟─2b8a8f41-1950-4d89-9f95-3b07f4a7b1ff
# ╟─8b8f77f8-90a2-4c48-aea4-0d3d7393e2c2
# ╟─7b0d99e8-6f73-47f9-ae68-bd0b76734e58
# ╠═5a8fe1b4-5ed2-4ff4-b8f2-1e8609c1c59a
# ╟─4a69aa11-3e7f-457e-a5a6-b713d46ea2a2
# ╟─1f53ccbe-54f3-45a8-88c0-4006d7d8aa6c
# ╟─3c8d2f1a-5e4c-4b3d-8a1c-9f8e7c6d5e4f
# ╟─734acb86-04cb-11f1-a988-2f4d1b061fa6
# ╟─8f9a1b2c-3d4e-5f6a-7b8c-9d0e1f2a3b4c
# ╟─9e8b7c6d-4e5f-6a7b-8c9d-0e1f2a3b4c5d
# ╟─1a2b3c4d-5e6f-7a8b-9c0d-1e2f3a4b5c6d
# ╠═2b3c4d5e-6f7a-8b9c-0d1e-2f3a4b5c6d7e
# ╠═3c4d5e6f-7a8b-9c0d-1e2f-3a4b5c6d7e8f
# ╠═5e6f7a8b-9c0d-1e2f-3a4b-5c6d7e8f9a0b
# ╟─4d5e6f7a-8b9c-0d1e-2f3a-4b5c6d7e8f9a
# ╠═6f7a8b9c-0d1e-2f3a-4b5c-6d7e8f9a0b1c
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
