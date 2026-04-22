### A Pluto.jl notebook ###
# v0.20.21

#> [frontmatter]
#> chapter = "1"
#> title = "Dimensionality Reduction & Random Projections"
#> layout = "layout.jlhtml"
#> description = "Understanding Johnson-Lindenstrauss theorem and Random Matrix Theory through interactive visualizations"

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

# ╔═╡ e796296e-9941-11ec-34fa-cbc6c2c3eaa0
begin
    using LinearAlgebra, Random, Plots, PlutoUI, PlutoTeachingTools
    using Statistics, Distributions, StatsPlots, Measures
end

# ╔═╡ 28b5ab19-2bce-4f37-985a-af550be109ef
TableOfContents()

# ╔═╡ 6bc469ef-3543-4fe7-8158-573981fe29f0
ChooseDisplayMode()

# ╔═╡ intro_section
md"""
# Dimensionality Reduction via Random Projections

## Why Care About Dimensionality Reduction?

In many inverse problems and data analysis tasks, we deal with **high-dimensional data**:
- Seismic waveforms: thousands of time samples
- Tomographic images: millions of pixels
- Climate models: thousands of spatial grid points

### The Curse of Dimensionality
As dimensions increase:
- 📈 **Computing distance/similarity becomes expensive** (O(d) operations per pair)
- 💾 **Storage grows linearly** with dimension
- 🐌 **Algorithms slow down dramatically** 
- 📊 **Visualization becomes impossible** (we can only see 2D/3D)

### The Solution: Random Projections
**Surprising fact**: We can project high-dimensional data to **much lower dimensions** while approximately preserving pairwise distances!

This notebook explores:
1. **Intuition**: 3D → 2D projections (visualizable)
2. **Johnson-Lindenstrauss Theorem**: 1000D → 20D (preserving distances!)
3. **Random Matrix Theory**: Why random projections work
4. **Applications**: Inverse problems and data compression
"""

# ╔═╡ motivation_examples
md"""
## Real-World Examples

**Example 1: Earthquake Catalog Analysis**
- Each earthquake: 100+ features (magnitude, depth, location, waveform characteristics)
- 10,000 earthquakes → 100D × 10,000 points
- Computing all pairwise similarities: $(10000 * 9999 / 2) = 49,995,000$ distance calculations
- **With 20D projection**: Same accuracy, 5× faster!

**Example 2: Climate Model Ensembles**
- Each model run: 1000s of spatial grid points
- Need to compare ensemble members
- Random projection enables fast similarity search

**Key Question**: How much can we compress without losing information?
"""

# ╔═╡ section_3d_to_2d
md"""
## Building Intuition: 3D → 2D Projections

Let's start with something we can visualize: projecting 3D points onto a 2D plane.

### The Setup
- Generate points in 3D space
- Project them onto a random 2D plane
- Compare distances before and after

**Key insight**: If we choose the projection carefully (or randomly!), distances are approximately preserved.
"""

# ╔═╡ demo_3d_controls
md"""
### Interactive 3D → 2D Demo

Number of points: $(@bind n_points_3d Slider(5:50, default=10, show_value=true))

Projection angle θ: $(@bind theta Slider(0:0.1:2π, default=π/4, show_value=true))

Random projection: $(@bind use_random_3d CheckBox(default=false))
"""

# ╔═╡ generate_3d_data
begin
	Random.seed!(42)
	
	# Generate 3D points on a sphere for better visualization
	n_pts_3d = n_points_3d
	X_3d = randn(3, n_pts_3d)
	X_3d = X_3d ./ sqrt.(sum(abs2, X_3d, dims=1)) .* (1 .+ 0.3*randn(1, n_pts_3d))
	
	# Compute original distances
	function pairwise_distances(X)
		n = size(X, 2)
		D = zeros(n, n)
		for i in 1:n
			for j in i+1:n
				D[i,j] = D[j,i] = norm(X[:,i] - X[:,j])
			end
		end
		return D
	end
	
	D_original_3d = pairwise_distances(X_3d)
	
	# Create projection matrix
	if use_random_3d
		# Random orthonormal 2D subspace
		P_3d = randn(2, 3)
		P_3d = qr(P_3d').Q[:, 1:2]'  # Orthonormalize
	else
		# Simple rotation in xy-plane
		P_3d = [cos(theta) sin(theta) 0;
		        -sin(theta) cos(theta) 0]
	end
	
	# Project to 2D
	X_2d = P_3d * X_3d
	D_projected_3d = pairwise_distances(X_2d)
	
	# Compute relative error
	relative_error_3d = norm(D_projected_3d - D_original_3d) / norm(D_original_3d)
	
	nothing
end

# ╔═╡ plot_3d_to_2d
begin
	# 3D scatter plot
	p1_3d = scatter(X_3d[1,:], X_3d[2,:], X_3d[3,:], 
		label="Original 3D points",
		xlabel="x", ylabel="y", zlabel="z",
		title="3D Data",
		markersize=6, alpha=0.7, color=:blue,
		camera=(30, 30))
	
	# Draw some connections to show structure
	for i in 1:min(5, n_pts_3d-1)
		plot!(p1_3d, [X_3d[1,i], X_3d[1,i+1]], 
			      [X_3d[2,i], X_3d[2,i+1]],
			      [X_3d[3,i], X_3d[3,i+1]], 
			      color=:gray, alpha=0.3, label="")
	end
	
	# 2D projected plot
	p2_3d = scatter(X_2d[1,:], X_2d[2,:],
		label="Projected 2D points",
		xlabel="u", ylabel="v",
		title="2D Projection (error: $(round(relative_error_3d*100, digits=1))%)",
		markersize=6, alpha=0.7, color=:red,
		aspect_ratio=:equal)
	
	# Draw same connections
	for i in 1:min(5, n_pts_3d-1)
		plot!(p2_3d, [X_2d[1,i], X_2d[1,i+1]], 
			      [X_2d[2,i], X_2d[2,i+1]], 
			      color=:gray, alpha=0.3, label="")
	end
	
	plot(p1_3d, p2_3d, layout=(1,2), size=(900, 400))
end

# ╔═╡ distance_comparison_3d
md"""
### Distance Preservation Analysis

The plot below compares original 3D distances vs. projected 2D distances. 
**Ideal case**: All points lie on the diagonal (perfect preservation).
"""

# ╔═╡ plot_distance_scatter_3d
begin
	# Extract upper triangle (unique pairs)
	idx_upper = findall(triu(ones(Bool, n_pts_3d, n_pts_3d), 1))
	d_orig_vec = [D_original_3d[i] for i in idx_upper]
	d_proj_vec = [D_projected_3d[i] for i in idx_upper]
	
	scatter(d_orig_vec, d_proj_vec,
		xlabel="Original 3D distance",
		ylabel="Projected 2D distance",
		title="Distance Preservation (3D → 2D)",
		label="Point pairs",
		alpha=0.6,
		markersize=4,
		legend=:bottomright)
	
	# Add y=x line (perfect preservation)
	max_d = maximum(d_orig_vec)
	plot!([0, max_d], [0, max_d], 
		line=:dash, color=:black, linewidth=2, 
		label="Perfect preservation")
end

# ╔═╡ observation_3d
md"""
### Observations

**Random vs. Structured Projections:**
- ✅ **Random projections** (checkbox on): Often preserve distances well
- ⚠️ **Structured projections** (checkbox off): Can compress or stretch depending on data alignment

**Key Insight**: Random projections are surprisingly good at preserving structure, even though they "throw away" information!

This hints at something deeper: **redundancy in high-dimensional data**.
"""

# ╔═╡ jl_theorem_intro
md"""
## The Johnson-Lindenstrauss Theorem

The 3D→2D demo hints at something remarkable. The **Johnson-Lindenstrauss (JL) theorem** formalizes this:

### Theorem Statement

For any set of **n** points in high-dimensional space, there exists a projection to **k** dimensions where:

$$k = O\left(\frac{\log n}{\epsilon^2}\right)$$

such that all pairwise distances are preserved within factor $(1±\epsilon)$:

$$(1-\epsilon) \|x_i - x_j\|^2 \leq \|f(x_i) - f(x_j)\|^2 \leq (1+\epsilon) \|x_i - x_j\|^2$$

### What This Means

- Original dimension **d** can be **anything** (even 1 million!)
- Target dimension **k** depends only on **n** (number of points) and **ε** (error tolerance)
- For n=1000 points with ε=0.1: **k ≈ 20 dimensions suffice!**

**Mind-blowing**: 1,000,000D → 20D while keeping distances accurate!
"""

# ╔═╡ jl_demo_params
md"""
## Johnson-Lindenstrauss Demo: 1000D → 20D

Let's verify this empirically with high-dimensional data.

### Parameters

Original dimension: $(@bind d_high Slider([100, 500, 1000, 2000], default=1000, show_value=true))

Number of points: $(@bind n_points_high Slider([50, 100, 200, 500, 1000], default=100, show_value=true))

Target dimension k: $(@bind k_low Slider(2:50, default=20, show_value=true))

Distortion tolerance ε: $(@bind epsilon Select([0.05 => "5%", 0.1 => "10%", 0.2 => "20%", 0.3 => "30%"]))
"""

# ╔═╡ jl_required_dimension
begin
	# Theoretical bound: k >= c * log(n) / ε²
	k_theoretical = ceil(Int, 8 * log(n_points_high) / epsilon^2)
	
	md"""
	**JL Theorem Prediction**: For $(n_points_high) points with $(epsilon*100)% error tolerance, need at least **$(k_theoretical) dimensions**.
	
	Current target: **$(k_low)** dimensions $(k_low >= k_theoretical ? "✅ (sufficient)" : "⚠️ (risky)")
	"""
end

# ╔═╡ generate_high_d_data
begin
	Random.seed!(123)
	
	# Generate high-dimensional data
	X_high = randn(d_high, n_points_high)
	
	# Normalize to unit sphere (helps with interpretation)
	for i in 1:n_points_high
		X_high[:,i] ./= norm(X_high[:,i])
	end
	
	# Add some structure (clusters)
	n_clusters = 3
	cluster_centers = randn(d_high, n_clusters)
	for i in 1:n_clusters
		cluster_centers[:,i] ./= norm(cluster_centers[:,i])
	end
	
	for i in 1:n_points_high
		cluster_id = mod1(i, n_clusters)
		X_high[:,i] = 0.7 * cluster_centers[:,cluster_id] + 0.3 * X_high[:,i]
		X_high[:,i] ./= norm(X_high[:,i])
	end
	
	# Original distances
	D_high_original = pairwise_distances(X_high)
	
	# Random projection matrix (Gaussian random)
	# Scale by 1/√k for distance preservation
	R_proj = randn(k_low, d_high) / sqrt(k_low)
	
	# Project
	X_low = R_proj * X_high
	D_high_projected = pairwise_distances(X_low)
	
	# Compute errors
	relative_errors_high = abs.(D_high_projected - D_high_original) ./ (D_high_original .+ 1e-10)
	max_relative_error = maximum(relative_errors_high)
	mean_relative_error = mean(relative_errors_high[D_high_original .> 0])
	
	# Count violations of ε-bound
	violations = count(relative_errors_high .> epsilon)
	total_pairs = n_points_high * (n_points_high - 1) / 2
	
	nothing
end

# ╔═╡ jl_results
begin
	md"""
	### Results: $(d_high)D → $(k_low)D Projection
	
	**Distance Preservation:**
	- Maximum relative error: **$(round(max_relative_error*100, digits=2))%**
	- Mean relative error: **$(round(mean_relative_error*100, digits=2))%**
	- JL bound violations: **$(violations) / $(Int(total_pairs))** pairs ($(round(violations/total_pairs*100, digits=2))%)
	
	**Interpretation:**
	$(if max_relative_error < epsilon
		"✅ **Success!** All distances within $(epsilon*100)% tolerance."
	elseif mean_relative_error < epsilon
		"⚠️ **Mostly works**: Mean error acceptable, but some outliers exceed tolerance."
	else
		"❌ **Need more dimensions**: Too much compression. Increase k or decrease n."
	end)
	
	**Compression ratio**: $(round(d_high/k_low, digits=1))× reduction in dimension!
	"""
end

# ╔═╡ plot_jl_distances
begin
	# Extract distances for plotting
	idx_upper_high = findall(triu(ones(Bool, n_points_high, n_points_high), 1))
	d_orig_high = [D_high_original[i] for i in idx_upper_high]
	d_proj_high = [D_high_projected[i] for i in idx_upper_high]
	
	p_scatter_high = scatter(d_orig_high, d_proj_high,
		xlabel="Original $(d_high)D distance",
		ylabel="Projected $(k_low)D distance",
		title="Distance Preservation $(d_high)D → $(k_low)D",
		label="Point pairs",
		alpha=0.3,
		markersize=3,
		legend=:bottomright)
	
	# Perfect preservation line
	max_d_high = maximum(d_orig_high)
	plot!(p_scatter_high, [0, max_d_high], [0, max_d_high], 
		line=:dash, color=:black, linewidth=2, label="Perfect")
	
	# Tolerance bounds
	plot!(p_scatter_high, [0, max_d_high], [0, max_d_high*(1-epsilon)], 
		line=:dash, color=:red, linewidth=1, label="$(epsilon*100)% bounds", alpha=0.5)
	plot!(p_scatter_high, [0, max_d_high], [0, max_d_high*(1+epsilon)], 
		line=:dash, color=:red, linewidth=1, label="", alpha=0.5)
	
	# Error distribution histogram
	p_hist = histogram(relative_errors_high[:],
		xlabel="Relative distance error",
		ylabel="Frequency",
		title="Error Distribution",
		bins=30,
		label="",
		color=:blue,
		alpha=0.6)
	
	# Mark epsilon threshold
	vline!(p_hist, [epsilon], line=:dash, color=:red, linewidth=2, label="ε threshold")
	
	plot(p_scatter_high, p_hist, layout=(1,2), size=(900, 400))
end

# ╔═╡ rmt_intro
md"""
## Random Matrix Theory: Why Does This Work?

The Johnson-Lindenstrauss theorem seems almost magical. **Random Matrix Theory** explains why.

### Key Concepts

**1. Concentration of Measure**
In high dimensions, random quantities concentrate around their mean:
- Distance between random points: ≈ √d
- Random dot products: ≈ 0
- Random projections preserve lengths on average

**2. Isotropic Random Projections**
A random matrix R with i.i.d. Gaussian entries N(0, 1/k) satisfies:
$$E[\|Rx\|^2] = \|x\|^2$$

The projection approximately preserves all directions **simultaneously** (with high probability).

**3. The Singular Value Distribution**
For large random matrices, singular values cluster predictably around 1.
"""

# ╔═╡ rmt_demo_setup
md"""
### RMT Demo: Singular Values of Random Projections

Dimension of projection: $(@bind k_rmt Slider(10:5:100, default=50, show_value=true))

Ambient dimension: $(@bind d_rmt Slider(100:100:1000, default=500, show_value=true))

Number of matrices: $(@bind n_matrices Slider(1:20, default=10, show_value=true))
"""

# ╔═╡ rmt_computation
begin
	Random.seed!(456)
	
	# Generate multiple random projection matrices
	singular_values_collection = []
	
	for _ in 1:n_matrices
		R_rmt = randn(k_rmt, d_rmt) / sqrt(k_rmt)
		svd_result = svd(R_rmt)
		push!(singular_values_collection, svd_result.S)
	end
	
	# Flatten all singular values
	all_svs = vcat(singular_values_collection...)
	
	nothing
end

# ╔═╡ plot_rmt_svd
begin
	# Histogram of singular values
	p_svd = histogram(all_svs,
		xlabel="Singular value",
		ylabel="Frequency",
		title="Singular Value Distribution ($(k_rmt)×$(d_rmt) random matrices)",
		bins=30,
		label="",
		color=:purple,
		alpha=0.6,
		normalize=:pdf)
	
	# Theoretical prediction: should cluster near 1
	vline!(p_svd, [1.0], line=:dash, color=:red, linewidth=3, 
		label="Expected value = 1")
	
	# Show spread
	mean_sv = mean(all_svs)
	std_sv = std(all_svs)
	
	annotate!(p_svd, 1.5, maximum(p_svd[1][2])*0.8, 
		text("μ = $(round(mean_sv, digits=3))\nσ = $(round(std_sv, digits=3))", 10, :left))
	
	p_svd
end

# ╔═╡ rmt_observation
md"""
### Observations

**Singular Value Concentration:**
- All singular values cluster tightly around **1.0**
- Standard deviation: $(round(std(all_svs), digits=3))
- This means the random projection is **nearly isometric** (distance-preserving)

**Why This Matters:**
If singular values ≈ 1, then for any vector x:
$$\|Rx\|^2 = \sum_i \sigma_i^2 \langle u_i, x \rangle^2 \approx \sum_i \langle u_i, x \rangle^2 = \|x\|^2$$

where σᵢ are singular values and uᵢ are left singular vectors.

**Physical Intuition**: Random projections act like "averaging" over many directions—they don't systematically amplify or shrink any particular direction.
"""

# ╔═╡ rmt_eigenvalue_demo
md"""
### Visualization: How Random Projections Transform Space

Let's visualize how a random projection transforms a unit sphere.
"""

# ╔═╡ sphere_transformation
begin
	Random.seed!(789)
	
	# Generate points on unit sphere in 3D
	n_sphere_pts = 200
	X_sphere = randn(3, n_sphere_pts)
	X_sphere = X_sphere ./ sqrt.(sum(abs2, X_sphere, dims=1))
	
	# Random projection 3D → 2D
	R_sphere = randn(2, 3) / sqrt(2)
	X_sphere_proj = R_sphere * X_sphere
	
	# Original sphere
	p_orig_sphere = scatter(X_sphere[1,:], X_sphere[2,:], X_sphere[3,:],
		title="Original: Unit Sphere in 3D",
		label="",
		markersize=2,
		alpha=0.5,
		camera=(30, 30),
		xlabel="x", ylabel="y", zlabel="z",
		aspect_ratio=:equal)
	
	# Projected ellipse
	p_proj_sphere = scatter(X_sphere_proj[1,:], X_sphere_proj[2,:],
		title="Projected: (Almost) Unit Circle in 2D",
		label="",
		markersize=2,
		alpha=0.5,
		xlabel="u", ylabel="v",
		aspect_ratio=:equal)
	
	# Add unit circle for reference
	theta_circle = range(0, 2π, length=100)
	plot!(p_proj_sphere, cos.(theta_circle), sin.(theta_circle),
		line=:dash, color=:red, linewidth=2, label="Unit circle")
	
	plot(p_orig_sphere, p_proj_sphere, layout=(1,2), size=(900, 400))
end

# ╔═╡ sphere_analysis
md"""
**Observation**: The projected points approximately fill a **unit circle** in 2D!

This demonstrates the isometric property:
- Random projection preserves the "size" of the original set
- Some distortion, but bounded by the concentration of measure phenomenon
"""

# ╔═╡ applications_section
md"""
## Applications to Inverse Problems

### 1. Fast Nearest Neighbor Search
**Problem**: Find similar earthquakes in a catalog (100D feature space, 10,000 events)

**Solution**: 
- Project to 20D using JL
- Build spatial index (k-d tree) in 20D
- Query time: $O(\log n)$ instead of $O(nd)$

### 2. Regularization via Random Projection
**Problem**: Ill-posed inverse problem G·m = d with huge parameter space

**Solution**:
- Parameterize: m = R^T·α where R is random k×d matrix with k ≪ d
- Solve compressed problem: (G·R^T)·α = d
- Effective regularization: restricts solution to random k-dimensional subspace

### 3. Sketching for Large-Scale Optimization
**Problem**: Least-squares with millions of parameters

**Solution**:
- Sketch equations: R·G·m ≈ R·d where R is random
- Solve smaller system (faster, less memory)
- Provably good approximation via JL lemma

"""

# ╔═╡ practical_considerations
md"""
## Practical Considerations

### Choosing Target Dimension k

From JL theorem: $k \geq \frac{8 \log n}{\epsilon^2}$

| Points (n) | ε=10% | ε=20% | ε=30% |
|-----------|-------|-------|-------|
| 100       | 37    | 12    | 6     |
| 1,000     | 55    | 17    | 8     |
| 10,000    | 74    | 23    | 11    |
| 100,000   | 92    | 29    | 14    |

**Rule of thumb**: k ≈ 10-50 dimensions handles thousands of points well.

### Random Projection Variants

1. **Gaussian**: $R_{ij} \sim N(0, 1/k)$ — theoretically optimal, but slow
2. **Rademacher**: $R_{ij} \in \{-1/\sqrt{k}, +1/\sqrt{k}\}$ — faster, almost as good
3. **Sparse**: $R_{ij} = 0$ with probability 2/3 — very fast, good for huge dimensions

### When JL Fails

- **Intrinsic low dimension**: If data already lies in k-dimensional subspace, can do better with PCA
- **Structure**: If distances have special patterns, structured methods (PCA, autoencoders) may preserve more
- **Very few points**: For n < 20, JL bound is pessimistic; other methods better
"""

# ╔═╡ interactive_classroom
md"""
## Classroom Exercise: Explore JL Parameters

Use the interactive demo above to investigate:

**Question 1**: Fix n=100 points, ε=10%. What's the minimum k that works?

**Question 2**: For k=20, what's the maximum n you can handle with ε=10%?

**Question 3**: Starting from d=1000, n=100, k=20, ε=10%:
- What happens if you increase d to 2000? Does it break?
- What happens if you increase n to 500? Does it break?

**Key Learning**: JL is **dimension-free**! The original dimension d doesn't matter (as long as d ≥ k).
"""

# ╔═╡ summary_section
md"""
## Summary

### Key Takeaways

1. **Johnson-Lindenstrauss Theorem** 🎯
   - Compress d dimensions → $O(\frac{\log n}{\epsilon^2})$ dimensions
   - Preserve all pairwise distances within (1±ε)
   - **Dimension-independent**: d can be anything!

2. **Random Matrix Theory** 🎲
   - Random projections are *nearly isometric*
   - Singular values concentrate around 1
   - Concentration of measure explains distance preservation

3. **Practical Impact** 🚀
   - **Speed**: 10-100× faster nearest neighbor search
   - **Memory**: 10-100× less storage
   - **Simplicity**: Just multiply by a random matrix!

### When to Use Random Projections

✅ **Good for:**
- Large-scale distance/similarity computations  
- Preprocessing for clustering/classification
- Sketching for optimization
- Data visualization (high-D → 2D/3D)

⚠️ **Consider alternatives:**
- Small datasets (n < 100): Use exact methods
- Known structure: PCA, sparse coding may be better
- Need interpretability: Random projections are "black box"

### Further Reading

- Original JL paper: Johnson & Lindenstrauss (1984)
- Modern analysis: Dasgupta & Gupta (2003)
- Applications: "Random Projections for Machine Learning" (Bingham & Mannila, 2001)
"""

# ╔═╡ appendix
md"""
## Appendix: Mathematical Details

### Proof Sketch of JL Theorem

**Key ingredient**: For any fixed unit vector v, the random variable $\|Rv\|^2$ concentrates around 1.

**Proof outline**:
1. Fix two points x, y. Let v = (x-y)/‖x-y‖
2. Show $P(|\|Rv\|^2 - 1| > \epsilon) < 2e^{-ck\epsilon^2}$ (Chernoff bound)
3. Union bound over all (n choose 2) pairs
4. Requirement: $n^2 e^{-ck\epsilon^2} < 1$ ⟹ $k > \frac{c \log n}{\epsilon^2}$

### Connection to Compressed Sensing

Random projections are related to **compressed sensing** (Candès, Tao, Donoho):
- Both use random linear measurements
- CS focuses on *reconstruction* (recover full signal from few measurements)
- JL focuses on *distances* (preserve geometry in low dimensions)

Common thread: **randomness reduces dimensionality** without losing essential information.
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
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd"

[compat]
Distributions = "~0.25.117"
Measures = "~0.3.2"
Plots = "~1.40.9"
PlutoTeachingTools = "~0.3.1"
PlutoUI = "~0.7.61"
StatsPlots = "~0.15.7"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.4"
manifest_format = "2.0"
project_hash = "abc123def456"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
"""

# ╔═╡ Cell order:
# ╠═28b5ab19-2bce-4f37-985a-af550be109ef
# ╠═6bc469ef-3543-4fe7-8158-573981fe29f0
# ╟─intro_section
# ╟─motivation_examples
# ╟─section_3d_to_2d
# ╟─demo_3d_controls
# ╟─generate_3d_data
# ╟─plot_3d_to_2d
# ╟─distance_comparison_3d
# ╟─plot_distance_scatter_3d
# ╟─observation_3d
# ╟─jl_theorem_intro
# ╟─jl_demo_params
# ╟─jl_required_dimension
# ╟─generate_high_d_data
# ╟─jl_results
# ╟─plot_jl_distances
# ╟─rmt_intro
# ╟─rmt_demo_setup
# ╟─rmt_computation
# ╟─plot_rmt_svd
# ╟─rmt_observation
# ╟─rmt_eigenvalue_demo
# ╟─sphere_transformation
# ╟─sphere_analysis
# ╟─applications_section
# ╟─practical_considerations
# ╟─interactive_classroom
# ╟─summary_section
# ╟─appendix
# ╠═e796296e-9941-11ec-34fa-cbc6c2c3eaa0
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
