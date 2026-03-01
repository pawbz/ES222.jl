### A Pluto.jl notebook ###
# v0.20.21

#> [frontmatter]
#> chapter = "1"
#> title = "Least Squares Line Fitting"
#> tags = ["module1"]
#> layout = "layout.jlhtml"

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

# ╔═╡ c9edde32-a12b-11ed-3dea-3be5d6546919
using LinearAlgebra, PlutoUI, PlutoTeachingTools, Symbolics, Distributions, PlutoPlotly, Printf, Random

# ╔═╡ 1768b806-0b79-416d-843e-7a2a64941706
TableOfContents()

# ╔═╡ 94bc7cd0-cfd1-4cc1-8930-f3fb37e0bcd4
md"# Fitting A Straight Line"

# ╔═╡ a055bbc2-cdc3-4d9b-93d0-d54accc5917a
md"We will consider a simple inverse problem here: Suppose that N temperature measurements Ti are made at times ti in the atmosphere. We assume that the temperature is a simple linear function of time. There are two model parameters in this problem, the slope of the straight line and its y-intercept. "

# ╔═╡ 4dbd9bb4-9354-47d5-ae0d-cc192c4ad28b
md"""
Configuration
$(@bind config MultiCheckBox(["Additive Gaussian Noise", "Outlier", "Constraint"], default=["Additive Gaussian", "Constraint"]))
"""

# ╔═╡ 3e9cfdd1-e2f2-4c75-be6a-b15540054871
md"""
Click to regenerate 1) $(@bind renoise Button("Noise")) 2) $(@bind reout Button("Outlier")) 3) $(@bind recon Button("Constraint"))
"""

# ╔═╡ 725659ff-77f1-46b5-9783-24100b3bc2c5
Markdown.MD(Markdown.Admonition("warning", "Intuition",
    [md"""
    Regenerate noise to notice that the variance along the y-intercept dimension is higher than the variance along the slope dimension. This means the estimate of the slope is less uncertain than the y-intercept. Notice the broad minimum along the y-intercept dimension compared to the sharp minimum along the slope dimension.
    """]
))

# ╔═╡ 23a93c6b-b53b-4433-8345-ab63e36e8a66
md"""
## Line Fitting
"""

# ╔═╡ 38106955-da38-4cbd-9d40-7e641acf9015
begin
    sloper = range(-2, stop=5, length=100) # range for slope
    yintr = range(-2, stop=5, length=100) # range for y intercept
end

# ╔═╡ 2d0a1baf-a070-4da6-8a86-e927dc0ff7f0
begin
    # choose the number of data points and control variable x
    N = 10 # change this to K
    x = range(1, stop=N)
end

# ╔═╡ 79c4f0a8-53b5-460a-a210-1a683709cbc6
# forward map
G = hcat(x, ones(N))

# ╔═╡ ad22aeb5-5366-434f-b735-98f8f7dd7101
G' * G

# ╔═╡ 33242a72-589f-47de-8127-b419957ca974
# choose true model parameters
mtrue = [3.0, 0.4]

# ╔═╡ 371db73a-9093-4fa1-8ebd-953ac4e5e090
md"""
Least-squares functional has the form
```math
J = (G\cdot m-d)^\top \cdot (G\cdot m-d)
```
which can be written as 
```math
J = m^\top G^\top G\,m\,+\,2\,m^\top\,G^\top\,d\,+ d^\top\,d
```
gradient w.r.t. $m$
```math
  \frac{\partial J}{\partial m} = 2\cdot G^\top \cdot (G\cdot m-d)
```
solution using the Moore-Penrose inverse
```math
(G^TG)^{-1}G^Td
```
"""

# ╔═╡ b5b75d3d-d67d-4e38-819d-8315bed6aef4
md"""
Notice that the columns of $G$ are linearly independent, therefore $G^\top G$ is invertible.
"""

# ╔═╡ 967f9eda-b39b-46b6-914b-c84d9acc056b
y_nonoise = G * mtrue

# ╔═╡ 419f6827-1dfd-40c9-bb73-53a9d5600edd
G⁻ᵍ = inv(transpose(G) * G) * transpose(G)

# ╔═╡ a9701326-dd26-473f-8534-d7400e4bc489
# m  = [slope, yintercept]

# ╔═╡ 131f48ac-45fe-455a-a8f0-e99e9ce05d7a
md"""
## Error Bowl
"""

# ╔═╡ 9df00f45-71be-41b5-9bbd-3ecba2493ddd
Markdown.MD(Markdown.Admonition("formula", "Gradient and Hessian of Linear and Quadratic Functions",
    [md"""
    A linear function of the form 
    ```math
    f(m) = a^\top\,m
    ```
    had gradient 
    $\nabla\,f = a$.
    The quadratic function of the form 
    ```math
    f(m) = m^\top\,A\,m
    ```
    has gradient $\nabla\,f=(A + A^\top)\,m$ and Hessian $\nabla^2\,f=A+A^\top$.
    These results can be derived by utilizing summation notation, taking partial derivatives, and subsequently recombining these partial derivatives into matrix form.
    """]
))

# ╔═╡ 3a22e16c-57b9-4b8f-b110-01edb1ed82b8
plot(heatmap(x=["1", "2"], y=["1", "2"], z=G' * G), Layout(title="Hessian Matrix", width=550))

# ╔═╡ d9acc5df-541b-448f-849f-228d6193b3dd
# generate observed data and add noise
begin
    renoise
    y_randnoise = copy(y_nonoise)
    # some random noise to the data
    if ("Additive Gaussian Noise" ∈ config)
        y_randnoise .+= randn(N) * 0.5
    end
end

# ╔═╡ 3db949ff-1b4f-484f-b5b8-2ab78b54fd51
begin
    reout
    y = copy(y_randnoise)
    # add an outlier
    if ("Outlier" ∈ config)
        y[rand(1:N)] *= 2.0
    end
end

# ╔═╡ ff5eddb8-b1f3-4d12-91bf-96438078f3cd
mest = G⁻ᵍ * y

# ╔═╡ b2766b9a-1287-42f1-bcc7-d0656a4a5f26
Jbowl = broadcast(Iterators.product(sloper, yintr)) do (m1, m2)
    sum(abs2.(G * [m1, m2] .- y))
end;

# ╔═╡ dceb94d9-4819-4d76-99d6-badbf8371469
md"""
## Constrained Problem
Number of constraints $P$
```math
Hm - h = 0
```
"""

# ╔═╡ e3205bc9-22d7-43bc-8068-4da0e6e71135
begin
    recon
    x1 = rand(Uniform(1, 4))
    y1 = rand(Uniform(extrema(y)...))
end

# ╔═╡ 27b428f0-015a-4eaa-9b16-75b69be4fe67
H = [x1, 1]'

# ╔═╡ ccb8342b-7d65-412c-b15a-2a28bc42f4da
h = [y1]

# ╔═╡ ff402573-bc3b-4b52-8549-2204d707aabe
md"""
```math
\begin{bmatrix}
G^TG &  H^T \\
H & 0 
\end{bmatrix} 
\begin{bmatrix}
m \\
\lambda 
\end{bmatrix}=
\begin{bmatrix}
G^Td  \\
h 
\end{bmatrix}
```
"""

# ╔═╡ f7e14126-37f2-4811-b1c6-b095f549d198
md"""### Lagrangian
The Lagrangian is a function of $m\in\mathbb{R}^N$ and $\lambda\in\mathbb{R}^P$, i.e., $N+P$ variables. We produce $N$ equations when differentiating w.r.t. each element of $m$ and $P$ equations when differentiating w.r.t. each element of $\lambda$.
"""

# ╔═╡ c5a8e002-35cd-4761-aa16-c2fdd1d7d270
md"""
Lagrangian:
```math
  \mathbb{L} = (G\cdot m-d)^\top \cdot (G\cdot m-d)+λ^\top \cdot (H\cdot m-h)
```

gradient w.r.t. $m$
```math
  \frac{\partial \mathbb{L}}{\partial m} = 2\cdot G^\top \cdot (G\cdot m-d)+H^\top \cdot \lambda
```

gradient w.r.t. $\lambda$
```math
  \frac{\partial \mathbb{L}}{\partial \lambda} = H\cdot m-h
```
"""

# ╔═╡ d520f33f-2347-4568-8b1e-7f0784ad515f
yc = [G' * y; y1]

# ╔═╡ ba7192d7-1973-4776-b44b-afd47727dbcd
Gc = [G'*G H'
    H [0]]

# ╔═╡ 61875817-4452-492d-b0f8-6d013ba414ef
mestc = inv(transpose(Gc) * Gc) * (transpose(Gc) * yc)

# ╔═╡ 6e6ac006-b078-4972-af27-ab24e3454e46
mestc

# ╔═╡ fa2989c0-b47b-4e3a-8d14-01d9433f2cd0
md"## Data Resolution Matrix"

# ╔═╡ 4a87186b-0ee3-417c-b61b-2facc44a2d94
Rd = G * G⁻ᵍ

# ╔═╡ 002dcf11-7259-45dd-af1f-568cc4880706
plot(heatmap(x=string.(1:size(Rd, 1)), y=string.(1:size(Rd, 1)), z=Rd), Layout(title="Data Resolution Matrix", width=550))

# ╔═╡ b3695eca-4652-4b3c-9d00-273fd6b5acf6
md"""
The diagonal elements of Data resolution matrix indicate how much weight a datum has in its own prediction. The diagonal elements are often singled out and called importance of the data. In this case, 
"""

# ╔═╡ f173dd19-506f-4b03-be65-dc3d0818f27c
plot(scatter(x=x, y=diag(Rd)), Layout(title="Data Importance", xlabel="x"))

# ╔═╡ e080a1f4-be4c-4512-bf16-ca4e32eb62b9
md"""
Curvature for the prediction error given by the second derivative 
"""

# ╔═╡ 63ebb87b-3fc2-4f83-ad82-b6c3aea17a4e
md"## Model Resolution Matrix"

# ╔═╡ 93779992-d2e2-452f-944e-009639f1d3db
Rm = G⁻ᵍ * G

# ╔═╡ 3a321ef1-6af2-431f-9e45-c8a512e92a9c
plot(heatmap(x=["1", "2"], y=["1", "2"], z=Rm), Layout(title="Model Resolution Matrix", width=550))

# ╔═╡ deca9062-a489-49b2-af63-ea9d2422efc2
md"""
# **Bayesian Line Fitting**
This notebook demonstrates **probabilistic line fitting** using **Bayesian inference** with a **closed-form solution**. 
 
In Bayesian linear regression, we assume a **linear model with Gaussian noise**:  

```math
y_i = m x_i + b + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2)
```

where:  
$y_i$ are the observed values.  
$x_i$ are the input data.  
$m$ (slope) and $b$ (intercept) are the unknown parameters.  
$\epsilon$ represents **Gaussian noise** with variance $\sigma^2$.  
    """

# ╔═╡ 3675562a-fed1-4d3f-be7a-29335204e56c
md"""Standard deviation of prior information $(@bind prior_std_input Slider(range(0.01, 1, length=100), default=0.1, show_value=true))"""

# ╔═╡ 849f32a0-8fda-49e7-b6e8-7968d71eb89a
md"""
Standard deviation of data noise 
$(@bind lh_std_input Slider(range(0.1, 10, length=100), default=0.1, show_value=true))
"""

# ╔═╡ b5195611-1309-42df-af5b-21eab516cb7d
md"""
    ## **Step 1: Define the Prior $P(m, b)$**  

    We assume a **Gaussian prior** over the parameters $m$ (slope) and $b$ (intercept):

    ```math
    P(m, b) = \mathcal{N} \left(
    \begin{bmatrix} m \\ b \end{bmatrix},
    \begin{bmatrix} \sigma_m^2 & 0 \\ 0 & \sigma_b^2 \end{bmatrix}
    \right)
    ```

    - The **prior mean vector** is:  
      ```math
      \mu_{\theta} =
      \begin{bmatrix} \mu_m \\ \mu_b \end{bmatrix}
      ```
    - The **prior covariance matrix** is:  
      ```math
      \Sigma_{\theta} =
      \begin{bmatrix} \sigma_m^2 & 0 \\ 0 & \sigma_b^2 \end{bmatrix}
      ```

    The **explicit probability density function** of the prior is:

    ```math
    P(\theta) = \frac{1}{\sqrt{(2\pi)^2 |\Sigma_{\theta}|}}
    \exp \left( -\frac{1}{2} (\theta - \mu_{\theta})^T \Sigma_{\theta}^{-1} (\theta - \mu_{\theta}) \right)
    ```

    where \( \theta = [m, b]^T \) is the parameter vector.
    """

# ╔═╡ 3060e567-b2a7-4ac7-b74b-dafbec4cb80b
### Step 2: Define the Likelihood ###
begin
    md"""
    ## **Step 2: Define the Likelihood \( P(Y | X, m, b) \)**  

    The likelihood function describes how the **observed data** is generated from  
    the true parameters. We assume **Gaussian noise**:

    ```math
    y_i = m x_i + b + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2)
    ```

    This means the **likelihood function** is:

    ```math
    P(Y | X, m, b) = \prod_{i=1}^{N} \mathcal{N}(y_i | m x_i + b, \sigma^2)
    ```

    Using matrix notation:
    - **Design matrix** \( X \):

      ```math
      X =
      \begin{bmatrix}
      x_1 & 1 \\
      x_2 & 1 \\
      \vdots & \vdots \\
      x_N & 1
      \end{bmatrix}
      ```

    - **Observation vector** \( Y \):

      ```math
      Y =
      \begin{bmatrix}
      y_1 \\ y_2 \\ \vdots \\ y_N
      \end{bmatrix}
      ```

    - **Parameter vector** \( \theta \):

      ```math
      \theta = \begin{bmatrix} m \\ b \end{bmatrix}
      ```

    The likelihood follows:

    ```math
    P(Y | X, \theta) =
    \frac{1}{\sqrt{(2\pi)^N |\sigma^2 I|}}
    \exp \left( -\frac{1}{2} (Y - X\theta)^T (\sigma^2 I)^{-1} (Y - X\theta) \right)
    ```
    """
end

# ╔═╡ 032d6aaf-1655-42fe-bc2c-0ea9b298cbf2

### Step 3: Compute the Posterior Using Bayes' Theorem ###
begin
    md"""
    ## **Step 3: Compute the Posterior \( P(\theta | X, Y) \) Using Bayes’ Theorem**  

    Using **Bayes' rule**:

    ```math
    P(\theta | X, Y) \propto P(Y | X, \theta) P(\theta)
    ```

    Substituting the **Gaussian prior** and **Gaussian likelihood**:

    ```math
    P(\theta | X, Y) \propto
    \exp \left( -\frac{1}{2} (Y - X\theta)^T (\sigma^2 I)^{-1} (Y - X\theta) \right)
    \exp \left( -\frac{1}{2} (\theta - \mu_{\theta})^T \Sigma_{\theta}^{-1} (\theta - \mu_{\theta}) \right)
    ```

    Expanding the exponents:

    ```math
    (Y - X\theta)^T (\sigma^2 I)^{-1} (Y - X\theta) = \frac{1}{\sigma^2} (Y - X\theta)^T (Y - X\theta)
    ```

    This leads to:

    ```math
    P(\theta | X, Y) \propto \exp \left( -\frac{1}{2} \left[ \theta^T A \theta - 2 \theta^T b + c \right] \right)
    ```

    where:

    - **Posterior precision matrix**:

      ```math
      A = X^T X / \sigma^2 + \Sigma_{\theta}^{-1}
      ```

    - **Posterior information vector**:

      ```math
      b = X^T Y / \sigma^2 + \Sigma_{\theta}^{-1} \mu_{\theta}
      ```

    Since this is the standard **Gaussian quadratic form**, we identify:

    - **Posterior covariance matrix**:

      ```math
      \Sigma_{\text{posterior}} = A^{-1} = \left( X^T X / \sigma^2 + \Sigma_{\theta}^{-1} \right)^{-1}
      ```

    - **Posterior mean**:

      ```math
      \mu_{\text{posterior}} = \Sigma_{\text{posterior}} \cdot b = \Sigma_{\text{posterior}} \left( X^T Y / \sigma^2 + \Sigma_{\theta}^{-1} \mu_{\theta} \right)
      ```
    """
end

# ╔═╡ fc6049b4-0312-4135-ad7b-26851708a156
md"## Appendix"

# ╔═╡ b2d3a472-ccda-4539-a8ce-6733012819eb
begin
	# Generate grid for prior 2D PDF
	grid_size = 256
	slope_range = range(-1, 1, length=grid_size)
	intercept_range = range(-1, 1, length=grid_size)
	grid = Iterators.product(slope_range, intercept_range)
	slope_grid = first.(grid)
	intercept_grid = last.(grid)
end;

# ╔═╡ be836b79-c3e9-4b02-9958-b4f2539b5f29
begin
    # Prior mean for slope and intercept
    mu_prior = [0.0, 0.0]

    # Prior standard deviations
    std_prior = [prior_std_input, prior_std_input]
    Sigma_prior = Diagonal(std_prior .^ 2)  # Prior covariance matrix

	# Compute prior PDF on the grid
    prior_dist = MvNormal(mu_prior, Sigma_prior)
    prior_pdf = reshape([pdf(prior_dist, [slope, intercept]) for (slope, intercept) in zip(slope_grid, intercept_grid)], grid_size, grid_size)

end;

# ╔═╡ 85152808-fa2c-4d71-bb96-85845367a12b
md"### Global Temperature Data

Hansen, J., R. Ruedy, Mki. Sato, and K. Lo, 2010: Global surface temperature change. Rev. Geophys., 48, RG4004, doi:10.1029/2010RG000345"

# ╔═╡ 1225f1fe-abf9-4a1d-816f-210e54c2ae3d
begin
    global_temp_data = [
        1965 -0.11; 1966 -0.03; 1967 -0.01; 1968 -0.04; 1969 0.08;
        1970 0.03; 1971 -0.10; 1972 0.00; 1973 0.14; 1974 -0.08;
        1975 -0.05; 1976 -0.16; 1977 0.12; 1978 0.01; 1979 0.08;
        1980 0.19; 1981 0.26; 1982 0.04; 1983 0.25; 1984 0.09;
        1985 0.04; 1986 0.12; 1987 0.27; 1988 0.31; 1989 0.19;
        1990 0.36; 1991 0.35; 1992 0.13; 1993 0.13; 1994 0.23;
        1995 0.37; 1996 0.29; 1997 0.39; 1998 0.56; 1999 0.32;
        2000 0.33; 2001 0.47; 2002 0.56; 2003 0.55; 2004 0.48;
        2005 0.62; 2006 0.55; 2007 0.58; 2008 0.44; 2009 0.58;
        2010 0.63
    ]

    # Extract years and temperature anomalies
    years = global_temp_data[:, 1]
    temp_anomalies = global_temp_data[:, 2]

    # Normalize years to avoid numerical issues (subtract mean)
    xtemp = years .- mean(years)
    ytemp = temp_anomalies

    md"""
    ## **Global Temperature Data**
    This dataset contains global temperature anomalies from **1965 to 2010**.
    """
end

# ╔═╡ 4b6aec66-93d7-4c20-83a4-87da5b481b9c
plot([
        scatter(x=global_temp_data[:, 1], y=global_temp_data[:, 2], name="a"),
    ], Layout(xaxis=attr(title="Year"), yaxis=attr(title="Temperature Anomaly (°C)")))

# ╔═╡ abbe0e38-1ab9-4695-a2cb-abdb3cd9e5d4
begin
	    Ntemp = length(xtemp)
	    # Construct design matrix X
	    Xtemp = hcat(xtemp, ones(Ntemp))  # [x 1] for intercept term
end

# ╔═╡ 8a9061d2-8900-4669-a0d9-f3691e71952c
begin
	# Noise variance (assumed known)
	
	std_lh = lh_std_input # Noise level
	Sigma_lh= Diagonal(fill(std_lh, length(ytemp)) .^ 2)
	
		# Compute prior PDF on the grid
		lh_dist = MvNormal(ytemp, Sigma_lh)
	    lh = reshape([pdf(lh_dist, Xtemp * [slope, intercept]) for (slope, intercept) in zip(slope_grid, intercept_grid)], grid_size, grid_size)
end

# ╔═╡ f09d0aeb-ba0e-40d2-9941-49e92b71c2a5
maximum(lh)

# ╔═╡ e8c9a972-7a84-4f45-937d-527c3d6d6f8e
heatmap(z=lh) |> plot

# ╔═╡ b5ac661a-94a3-424f-9628-9cccfcc0b8e7
begin


    # Compute posterior covariance
    Sigma_posterior = inv((Xtemp'Xtemp) / std_lh^2 + inv(Sigma_prior))

    # Compute posterior mean
    mu_posterior = Sigma_posterior * ((Xtemp' * ytemp) / std_lh^2 + inv(Sigma_prior) * mu_prior)

	# Compute posterior PDF on the grid
    posterior_dist = MvNormal(mu_posterior, Sigma_posterior)
    posterior_pdf = reshape([pdf(posterior_dist, [slope, intercept]) for (slope, intercept) in zip(slope_grid, intercept_grid)], grid_size, grid_size)

	
    # Generate posterior samples
    posterior_samples = rand(posterior_dist, 100)
    # Extract posterior slope and intercept samples
    slope_posterior_samples = posterior_samples[1, :]
    intercept_posterior_samples = posterior_samples[2, :]

    # # Sample from posterior
    # posterior_samples = rand(MvNormal(mu_posterior, Sigma_posterior), 10000)

    # # Extract posterior slope and intercept samples
    # slope_posterior_samples = posterior_samples[1, :]
    # intercept_posterior_samples = posterior_samples[2, :]

    md"""
    - Posterior mean for **slope**: $(@sprintf("%.3f", mu_posterior[1]))
    - Posterior mean for **intercept**: $(@sprintf("%.3f", mu_posterior[2]))
    """
end

# ╔═╡ 50924dce-1e92-4289-ae83-b3f9e540c7a8
begin
    # Compute posterior mean prediction
    y_mean = mu_posterior[1] .* xtemp .+ mu_posterior[2]

    # Compute prior mean regression line
    y_prior = mu_prior[1] .* xtemp .+ mu_prior[2]

    # Sampled posterior regression lines
    trace_samples = [
        scatter(x=xtemp, y=slope_posterior_samples[i] .* xtemp .+ intercept_posterior_samples[i],
            mode="lines", opacity=0.2, line=attr(color="gray"), showlegend=false)
        for i in 1:length(slope_posterior_samples)
    ]

    # Noisy data points
    trace_data = scatter(x=xtemp, y=ytemp, mode="markers",
        marker=attr(color="blue", size=6), name="Observed Data")

    # Posterior mean line
    trace_mean = scatter(x=xtemp, y=y_mean, mode="lines",
        line=attr(color="red", width=3), name="Posterior Mean Prediction")

    # Prior mean line
    trace_prior = scatter(x=xtemp, y=y_prior, mode="lines",
        line=attr(color="green", width=3, dash="dash"), name="Prior Mean Prediction")


    # Create figure
    layout = Layout(
        title="Bayesian Regression: Temperature Trend (1965-2010)",
        xaxis_title="Year Difference",
        yaxis_title="Temperature Anomaly (°C)",
        showlegend=true
    )

    md"""
    ## **Posterior Predictive Lines**
    - **Gray Lines:** Sampled regression lines from the posterior.
    - **Blue Points:** Observed temperature anomalies.
    - **Red Line:** Posterior mean trend.
    """

    plot([trace_data, trace_mean, trace_prior, trace_samples...], layout)
end

# ╔═╡ 7811d81f-0d19-4e7e-ae02-a3889ba178ef
begin
	prior_pdf_plot = plot([
	        surface(
	            x=slope_range, y=intercept_range, z=prior_pdf,
	            colorscale="Jet", opacity=0.9, showscale=false
	        )
	    ], Layout(
	        title="Prior",
	        scene=attr(
	            xaxis_title="Slope (m)",
	            yaxis_title="Intercept (b)",
	            zaxis_title="Probability Density",
	            zaxis=attr(range=[0, 1]),  # Fixed Z limits
	            aspectmode="cube"
	        )
	    ))
	
	    post_pdf_plot = plot([
	        surface(
	            x=slope_range, y=intercept_range, z=posterior_pdf,
	            colorscale="Jet", opacity=0.9, showscale=false
	        )
	    ], Layout(
	        title="Posterior",
	        scene=attr(
	            xaxis_title="Slope (m)",
	            yaxis_title="Intercept (b)",
	            zaxis_title="Probability Density",
	            zaxis=attr(range=[0, 1]),  # Fixed Z limits
	            aspectmode="cube"
	        )
	    ))


	 lh_plot = plot([
	        surface(
	            x=slope_range, y=intercept_range, z=lh,
	            colorscale="Jet", opacity=0.9, showscale=false
	        )
	    ], Layout(
	        title="Likelihood",
	        scene=attr(
	            xaxis_title="Slope (m)",
	            yaxis_title="Intercept (b)",
	            zaxis_title="Probability Density",
	            zaxis=attr(range=[0, 1]),  # Fixed Z limits
	            aspectmode="cube"
	        )
	    ))
end;

# ╔═╡ b3fdaab5-fa42-45ab-90ed-5eb4c7222169
begin
	p = [prior_pdf_plot lh_plot post_pdf_plot]  # Side-by-side arrangement
	    relayout!(p, height=400, width=700, title_text="")
	    p
end

# ╔═╡ 6ee10a18-d4bc-4c0e-9fcb-04b738bf7b1a
md"""
### References
- Useful resources for matrix calculus
  - [https://www.matrixcalculus.org/](https://www.matrixcalculus.org/)
  - [http://www.ee.ic.ac.uk/hp/staff/dmb/matrix/intro.html](http://www.ee.ic.ac.uk/hp/staff/dmb/matrix/intro.html)
"""

# ╔═╡ a62159c8-f4a3-46cb-82a2-fd92f8711382
md"### Plots"

# ╔═╡ c5e3add6-2446-4443-8f0d-708ceffd2033
pbowl = let
    s = [contour(x=yintr, y=sloper, z=Jbowl, colorscale="Hot", colorbar=attr(
            thickness=25,
            thicknessmode="pixels",
            len=0.3,
            lenmode="fraction",
            outlinewidth=0
        )),
        scatter(x=[mtrue[2]], y=[mtrue[1]], mode="markers", name="True Model", marker=attr(color=:green)),
        scatter(x=[mest[2]], y=[mest[1]], mode="markers", name="Estimated Model", marker=attr(color=:blue))
    ]
    if ("Constraint" ∈ config)
        push!(s, scatter(x=[mestc[2]], y=[mestc[1]], mode="markers", name="Constrained Model", marker=attr(color=:red)))
        push!(s, scatter(x=y1 .- sloper .* x1, y=yintr, mode="lines", name="Constraint", marker=attr(color=:red)))
    end
    plot(s, Layout(xaxis=attr(title="Intercept"), yaxis=attr(title="Slope")))
end;

# ╔═╡ 73007694-df0f-4fef-957a-b3c239c61bad
pline =
    let
        s = [scatter(x=x, y=y, mode="markers", name="Observations", marker=attr(color=:black)), scatter(x=x, y=G * mest, mode="lines", name="Estimated Model", line=attr(color=:blue)), scatter(x=x, y=G * mtrue, mode="lines", name="True Model", line=attr(color=:green))]
        if ("Constraint" ∈ config)
            push!(s, scatter(x=[x1], y=[y1], mode="markers", name="Constraint", marker=attr(color=:red)))
            push!(s, scatter(x=x, y=G * mestc[1:2], mode="lines", name="Constrained Model", line=attr(color=:red)))
        end
        plot(s, Layout(xaxis=attr(title="Year"), yaxis=attr(title="Temperature Anomaly (°C)")))
    end;

# ╔═╡ 6e8b13cc-f5f5-462f-94bd-06042cc9a7ad
PlutoUI.ExperimentalLayout.vbox([pline, pbowl])

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
PlutoPlotly = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
PlutoTeachingTools = "661c6b06-c737-4d37-b85c-46df65de6f69"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Symbolics = "0c5d862f-8b57-4792-8d23-62f2024744c7"

[compat]
Distributions = "~0.25.117"
PlutoPlotly = "~0.6.2"
PlutoTeachingTools = "~0.3.1"
PlutoUI = "~0.7.61"
Symbolics = "~6.29.0"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.4"
manifest_format = "2.0"
project_hash = "1ee6f5a0707f927150c3413b22b1c2071fe6aa59"

[[deps.ADTypes]]
git-tree-sha1 = "fb97701c117c8162e84dfcf80215caa904aef44f"
uuid = "47edcb42-4c32-4615-8424-f2b9edc5f35b"
version = "1.13.0"

    [deps.ADTypes.extensions]
    ADTypesChainRulesCoreExt = "ChainRulesCore"
    ADTypesConstructionBaseExt = "ConstructionBase"
    ADTypesEnzymeCoreExt = "EnzymeCore"

    [deps.ADTypes.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ConstructionBase = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.AbstractTrees]]
git-tree-sha1 = "2d9c9a55f9c93e8887ad391fbae72f8ef55e1177"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.4.5"

[[deps.Accessors]]
deps = ["CompositionsBase", "ConstructionBase", "Dates", "InverseFunctions", "MacroTools"]
git-tree-sha1 = "0ba8f4c1f06707985ffb4804fdad1bf97b233897"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.41"

    [deps.Accessors.extensions]
    AxisKeysExt = "AxisKeys"
    IntervalSetsExt = "IntervalSets"
    LinearAlgebraExt = "LinearAlgebra"
    StaticArraysExt = "StaticArrays"
    StructArraysExt = "StructArrays"
    TestExt = "Test"
    UnitfulExt = "Unitful"

    [deps.Accessors.weakdeps]
    AxisKeys = "94b1ba4f-4ee9-5380-92f1-94cde586c3c5"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    Requires = "ae029012-a4dd-5104-9daa-d747884805df"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "50c3c56a52972d78e8be9fd135bfb91c9574c140"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.1.1"
weakdeps = ["StaticArrays"]

    [deps.Adapt.extensions]
    AdaptStaticArraysExt = "StaticArrays"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.ArrayInterface]]
deps = ["Adapt", "LinearAlgebra"]
git-tree-sha1 = "017fcb757f8e921fb44ee063a7aafe5f89b86dd1"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.18.0"

    [deps.ArrayInterface.extensions]
    ArrayInterfaceBandedMatricesExt = "BandedMatrices"
    ArrayInterfaceBlockBandedMatricesExt = "BlockBandedMatrices"
    ArrayInterfaceCUDAExt = "CUDA"
    ArrayInterfaceCUDSSExt = "CUDSS"
    ArrayInterfaceChainRulesCoreExt = "ChainRulesCore"
    ArrayInterfaceChainRulesExt = "ChainRules"
    ArrayInterfaceGPUArraysCoreExt = "GPUArraysCore"
    ArrayInterfaceReverseDiffExt = "ReverseDiff"
    ArrayInterfaceSparseArraysExt = "SparseArrays"
    ArrayInterfaceStaticArraysCoreExt = "StaticArraysCore"
    ArrayInterfaceTrackerExt = "Tracker"

    [deps.ArrayInterface.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    CUDSS = "45b445bb-4962-46a0-9369-b4df9d0f772e"
    ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.Bijections]]
git-tree-sha1 = "d8b0439d2be438a5f2cd68ec158fe08a7b2595b7"
uuid = "e2ed5e7c-b2de-5872-ae92-c73ca462fb04"
version = "0.1.9"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "1713c74e00545bfe14605d2a2be1712de8fbcb58"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.25.1"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.CodeTracking]]
deps = ["InteractiveUtils", "UUIDs"]
git-tree-sha1 = "7eee164f122511d3e4e1ebadb7956939ea7e1c77"
uuid = "da1fd8a2-8d9e-5ec2-8556-3022fb5608a2"
version = "1.3.6"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "403f2d8e209681fcbd9468a8514efff3ea08452e"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.29.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "b10d0b65641d57b8b4d5e234446582de5047050d"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.5"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "a1f44953f2382ebb937d60dafbe2deea4bd23249"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.10.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "362a287c3aa50601b0bc359053d5c2468f0e7ce0"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.11"

[[deps.Combinatorics]]
git-tree-sha1 = "08c8b6831dc00bfea825826be0bc8336fc369860"
uuid = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
version = "1.0.2"

[[deps.CommonSolve]]
git-tree-sha1 = "0eee5eb66b1cf62cd6ad1b460238e60e4b09400c"
uuid = "38540f10-b2f7-11e9-35d8-d573e4eb0ff2"
version = "0.2.4"

[[deps.CommonWorldInvalidations]]
git-tree-sha1 = "ae52d1c52048455e85a387fbee9be553ec2b68d0"
uuid = "f70d9fcc-98c5-4d4a-abd7-e4cdeebd8ca8"
version = "1.0.0"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "8ae8d32e09f0dcf42a36b90d4e17f5dd2e4c4215"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.16.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.3.0+1"

[[deps.CompositeTypes]]
git-tree-sha1 = "bce26c3dab336582805503bed209faab1c279768"
uuid = "b152e2b5-7a66-4b01-a709-34e65c35f657"
version = "0.1.4"

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"
weakdeps = ["InverseFunctions"]

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

[[deps.ConstructionBase]]
git-tree-sha1 = "76219f1ed5771adbb096743bff43fb5fdd4c1157"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.8"
weakdeps = ["IntervalSets", "LinearAlgebra", "StaticArrays"]

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseLinearAlgebraExt = "LinearAlgebra"
    ConstructionBaseStaticArraysExt = "StaticArrays"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "1d0a14036acb104d9e89698bd408f63ab58cdc82"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.20"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "23163d55f885173722d1e4cf0f6110cdbaf7e272"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.15.1"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"
version = "1.11.0"

[[deps.Distributions]]
deps = ["AliasTables", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "03aa5d44647eaec98e1920635cdfed5d5560a8b9"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.117"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"
    DistributionsTestExt = "Test"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.DomainSets]]
deps = ["CompositeTypes", "IntervalSets", "LinearAlgebra", "Random", "StaticArrays"]
git-tree-sha1 = "a7e9f13f33652c533d49868a534bfb2050d1365f"
uuid = "5b8099bc-c8ec-5219-889f-1d9e522a28bf"
version = "0.7.15"

    [deps.DomainSets.extensions]
    DomainSetsMakieExt = "Makie"

    [deps.DomainSets.weakdeps]
    Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.7.0"

[[deps.DynamicPolynomials]]
deps = ["Future", "LinearAlgebra", "MultivariatePolynomials", "MutableArithmetics", "Reexport", "Test"]
git-tree-sha1 = "9a3ae38b460449cc9e7dd0cfb059c76028724627"
uuid = "7c1d4256-1411-5781-91ec-d7bc3513ac07"
version = "0.6.1"

[[deps.EnumX]]
git-tree-sha1 = "bdb1942cd4c45e3c678fd11569d5cccd80976237"
uuid = "4e289a0a-7415-4d19-859d-a7e5c4648b56"
version = "1.0.4"

[[deps.ExprTools]]
git-tree-sha1 = "27415f162e6028e81c72b82ef756bf321213b6ec"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.10"

[[deps.Expronicon]]
deps = ["MLStyle", "Pkg", "TOML"]
git-tree-sha1 = "fc3951d4d398b5515f91d7fe5d45fc31dccb3c9b"
uuid = "6b7a57c9-7cc1-4fdf-b7f5-e857abae3636"
version = "0.8.5"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "6a70198746448456524cb442b8af316927ff3e1a"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.13.0"
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

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.FunctionWrappers]]
git-tree-sha1 = "d62485945ce5ae9c0c48f124a84998d755bae00e"
uuid = "069b7b12-0de2-55c6-9aab-29f3d0a68a2e"
version = "1.1.3"

[[deps.FunctionWrappersWrappers]]
deps = ["FunctionWrappers"]
git-tree-sha1 = "b104d487b34566608f8b4e1c39fb0b10aa279ff8"
uuid = "77dc65aa-8811-40c2-897b-53d922fa7daf"
version = "0.1.3"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"
version = "1.11.0"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "83cf05ab16a73219e5f6bd1bdfa9848fa24ac627"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.2.0"

[[deps.HashArrayMappedTries]]
git-tree-sha1 = "2eaa69a7cab70a52b9687c8bf950a5a93ec895ae"
uuid = "076d061b-32b6-4027-95e0-9a2c6f6d7e74"
version = "0.2.0"

[[deps.HypergeometricFunctions]]
deps = ["LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "2bd56245074fab4015b9174f24ceba8293209053"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.27"

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
git-tree-sha1 = "b6d6bfdd7ce25b0f9b2f6b3dd56b2673a66c8770"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.5"

[[deps.IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

[[deps.IntegerMathUtils]]
git-tree-sha1 = "b8ffb903da9f7b8cf695a8bead8e01814aa24b30"
uuid = "18e54dd8-cb9d-406c-a71d-865a43cbb235"
version = "0.1.2"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.IntervalSets]]
git-tree-sha1 = "dba9ddf07f77f60450fe5d2e2beb9854d9a49bd0"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.10"
weakdeps = ["Random", "RecipesBase", "Statistics"]

    [deps.IntervalSets.extensions]
    IntervalSetsRandomExt = "Random"
    IntervalSetsRecipesBaseExt = "RecipesBase"
    IntervalSetsStatisticsExt = "Statistics"

[[deps.InverseFunctions]]
git-tree-sha1 = "a779299d77cd080bf77b97535acecd73e1c5e5cb"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.17"
weakdeps = ["Dates", "Test"]

    [deps.InverseFunctions.extensions]
    InverseFunctionsDatesExt = "Dates"
    InverseFunctionsTestExt = "Test"

[[deps.IrrationalConstants]]
git-tree-sha1 = "e2222959fbc6c19554dc15174c81bf7bf3aa691c"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.4"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "a007feb38b422fbdab534406aeca1b86823cb4d6"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.7.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JuliaInterpreter]]
deps = ["CodeTracking", "InteractiveUtils", "Random", "UUIDs"]
git-tree-sha1 = "4bf4b400a8234cff0f177da4a160a90296159ce9"
uuid = "aa1ae85d-cabe-5617-a682-6adf51b2e16a"
version = "0.9.41"

[[deps.JuliaSyntaxHighlighting]]
deps = ["StyledStrings"]
uuid = "ac6e5ff7-fb65-4e79-a425-ec3bc9c03011"
version = "1.12.0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

[[deps.Latexify]]
deps = ["Format", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Requires"]
git-tree-sha1 = "cd714447457c660382fe634710fb56eb255ee42e"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.6"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SparseArraysExt = "SparseArrays"
    SymEngineExt = "SymEngine"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"

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

[[deps.LoweredCodeUtils]]
deps = ["JuliaInterpreter"]
git-tree-sha1 = "688d6d9e098109051ae33d126fcfc88c4ce4a021"
uuid = "6f1432cf-f94c-5a45-995e-cdbf5db27b0b"
version = "3.1.0"

[[deps.MIMEs]]
git-tree-sha1 = "1833212fd6f580c20d4291da9c1b4e8a655b128e"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.0.0"

[[deps.MLStyle]]
git-tree-sha1 = "bc38dff0548128765760c79eb7388a4b37fae2c8"
uuid = "d8e11817-5142-5d16-987a-aa16d5891078"
version = "0.4.17"

[[deps.MacroTools]]
git-tree-sha1 = "72aebe0b5051e5143a079a4685a46da330a40472"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.15"

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

[[deps.MultivariatePolynomials]]
deps = ["ChainRulesCore", "DataStructures", "LinearAlgebra", "MutableArithmetics"]
git-tree-sha1 = "8d39779e29f80aa6c071e7ac17101c6e31f075d7"
uuid = "102ac46a-7ee4-5c85-9060-abc95bfdeaa3"
version = "0.5.7"

[[deps.MutableArithmetics]]
deps = ["LinearAlgebra", "SparseArrays", "Test"]
git-tree-sha1 = "9c0bc309df575c85422232eedfb74d5a9c155401"
uuid = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"
version = "1.6.3"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "cc0a5deefdb12ab3a096f00a6d42133af4560d71"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.2"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.3.0"

[[deps.OffsetArrays]]
git-tree-sha1 = "5e1897147d1ff8d98883cda2be2187dcf57d8f0c"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.15.0"
weakdeps = ["Adapt"]

    [deps.OffsetArrays.extensions]
    OffsetArraysAdaptExt = "Adapt"

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
git-tree-sha1 = "cc4054e898b852042d7b503313f7ad03de99c3dd"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.0"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "966b85253e959ea89c53a9abebbf2e964fbf593b"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.32"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "8489905bcdbcfac64d1daa51ca07c0d8f0283821"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.1"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.12.1"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PlotlyBase]]
deps = ["ColorSchemes", "Dates", "DelimitedFiles", "DocStringExtensions", "JSON", "LaTeXStrings", "Logging", "Parameters", "Pkg", "REPL", "Requires", "Statistics", "UUIDs"]
git-tree-sha1 = "56baf69781fc5e61607c3e46227ab17f7040ffa2"
uuid = "a03496cd-edff-5a9b-9e67-9cda94a718b5"
version = "0.8.19"

[[deps.PlutoHooks]]
deps = ["InteractiveUtils", "Markdown", "UUIDs"]
git-tree-sha1 = "072cdf20c9b0507fdd977d7d246d90030609674b"
uuid = "0ff47ea0-7a50-410d-8455-4348d5de0774"
version = "0.0.5"

[[deps.PlutoLinks]]
deps = ["FileWatching", "InteractiveUtils", "Markdown", "PlutoHooks", "Revise", "UUIDs"]
git-tree-sha1 = "8f5fa7056e6dcfb23ac5211de38e6c03f6367794"
uuid = "0ff47ea0-7a50-410d-8455-4348d5de0420"
version = "0.1.6"

[[deps.PlutoPlotly]]
deps = ["AbstractPlutoDingetjes", "Artifacts", "ColorSchemes", "Colors", "Dates", "Downloads", "HypertextLiteral", "InteractiveUtils", "LaTeXStrings", "Markdown", "Pkg", "PlotlyBase", "PrecompileTools", "Reexport", "ScopedValues", "Scratch", "TOML"]
git-tree-sha1 = "9ebe25fc4703d4112cc418834d5e4c9a4b29087d"
uuid = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
version = "0.6.2"

    [deps.PlutoPlotly.extensions]
    PlotlyKaleidoExt = "PlotlyKaleido"
    UnitfulExt = "Unitful"

    [deps.PlutoPlotly.weakdeps]
    PlotlyKaleido = "f2990250-8cf9-495f-b13a-cce12b45703c"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.PlutoTeachingTools]]
deps = ["Downloads", "HypertextLiteral", "Latexify", "Markdown", "PlutoLinks", "PlutoUI"]
git-tree-sha1 = "8252b5de1f81dc103eb0293523ddf917695adea1"
uuid = "661c6b06-c737-4d37-b85c-46df65de6f69"
version = "0.3.1"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "7e71a55b87222942f0f9337be62e26b1f103d3e4"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.61"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "9306f6085165d270f7e3db02af26a400d580f5c6"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.3"

[[deps.Primes]]
deps = ["IntegerMathUtils"]
git-tree-sha1 = "cb420f77dc474d23ee47ca8d14c90810cafe69e7"
uuid = "27ebfcd6-29c5-5fa9-bf4b-fb8fc14df3ae"
version = "0.5.6"

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

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.RecursiveArrayTools]]
deps = ["Adapt", "ArrayInterface", "DocStringExtensions", "GPUArraysCore", "IteratorInterfaceExtensions", "LinearAlgebra", "RecipesBase", "StaticArraysCore", "Statistics", "SymbolicIndexingInterface", "Tables"]
git-tree-sha1 = "fe9d37a17ab4d41a98951332ee8067f8dca8c4c2"
uuid = "731186ca-8d62-57ce-b412-fbd966d074cd"
version = "3.29.0"

    [deps.RecursiveArrayTools.extensions]
    RecursiveArrayToolsFastBroadcastExt = "FastBroadcast"
    RecursiveArrayToolsForwardDiffExt = "ForwardDiff"
    RecursiveArrayToolsMeasurementsExt = "Measurements"
    RecursiveArrayToolsMonteCarloMeasurementsExt = "MonteCarloMeasurements"
    RecursiveArrayToolsReverseDiffExt = ["ReverseDiff", "Zygote"]
    RecursiveArrayToolsSparseArraysExt = ["SparseArrays"]
    RecursiveArrayToolsStructArraysExt = "StructArrays"
    RecursiveArrayToolsTrackerExt = "Tracker"
    RecursiveArrayToolsZygoteExt = "Zygote"

    [deps.RecursiveArrayTools.weakdeps]
    FastBroadcast = "7034ab61-46d4-4ed7-9d0f-46aef9175898"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    MonteCarloMeasurements = "0987c9cc-fe09-11e8-30f0-b96dd679fdca"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Revise]]
deps = ["CodeTracking", "FileWatching", "JuliaInterpreter", "LibGit2", "LoweredCodeUtils", "OrderedCollections", "REPL", "Requires", "UUIDs", "Unicode"]
git-tree-sha1 = "9bb80533cb9769933954ea4ffbecb3025a783198"
uuid = "295af30f-e4ad-537b-8983-00126c2a3abe"
version = "3.7.2"
weakdeps = ["Distributed"]

    [deps.Revise.extensions]
    DistributedExt = "Distributed"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "852bd0f55565a9e973fcfee83a84413270224dc4"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.8.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "58cdd8fb2201a6267e1db87ff148dd6c1dbd8ad8"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.5.1+0"

[[deps.RuntimeGeneratedFunctions]]
deps = ["ExprTools", "SHA", "Serialization"]
git-tree-sha1 = "04c968137612c4a5629fa531334bb81ad5680f00"
uuid = "7e49a35a-f44a-4d26-94aa-eba1b4ca6b47"
version = "0.5.13"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SciMLBase]]
deps = ["ADTypes", "Accessors", "ArrayInterface", "CommonSolve", "ConstructionBase", "Distributed", "DocStringExtensions", "EnumX", "Expronicon", "FunctionWrappersWrappers", "IteratorInterfaceExtensions", "LinearAlgebra", "Logging", "Markdown", "PrecompileTools", "Preferences", "Printf", "RecipesBase", "RecursiveArrayTools", "Reexport", "RuntimeGeneratedFunctions", "SciMLOperators", "SciMLStructures", "StaticArraysCore", "Statistics", "SymbolicIndexingInterface"]
git-tree-sha1 = "ffed2507209da5b42c6881944ef41a340ab5449b"
uuid = "0bca4576-84f4-4d90-8ffe-ffa030f20462"
version = "2.74.1"

    [deps.SciMLBase.extensions]
    SciMLBaseChainRulesCoreExt = "ChainRulesCore"
    SciMLBaseMakieExt = "Makie"
    SciMLBasePartialFunctionsExt = "PartialFunctions"
    SciMLBasePyCallExt = "PyCall"
    SciMLBasePythonCallExt = "PythonCall"
    SciMLBaseRCallExt = "RCall"
    SciMLBaseZygoteExt = "Zygote"

    [deps.SciMLBase.weakdeps]
    ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
    PartialFunctions = "570af359-4316-4cb7-8c74-252c00c2016b"
    PyCall = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0"
    PythonCall = "6099a3de-0909-46bc-b1f4-468b9a2dfc0d"
    RCall = "6f49c342-dc21-5d91-9882-a32aef131414"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.SciMLOperators]]
deps = ["Accessors", "ArrayInterface", "DocStringExtensions", "LinearAlgebra", "MacroTools"]
git-tree-sha1 = "6149620767866d4b0f0f7028639b6e661b6a1e44"
uuid = "c0aeaf25-5076-4817-a8d5-81caf7dfa961"
version = "0.3.12"
weakdeps = ["SparseArrays", "StaticArraysCore"]

    [deps.SciMLOperators.extensions]
    SciMLOperatorsSparseArraysExt = "SparseArrays"
    SciMLOperatorsStaticArraysCoreExt = "StaticArraysCore"

[[deps.SciMLStructures]]
deps = ["ArrayInterface"]
git-tree-sha1 = "0444a37a25fab98adbd90baa806ee492a3af133a"
uuid = "53ae85a6-f571-4167-b2af-e1d143709226"
version = "1.6.1"

[[deps.ScopedValues]]
deps = ["HashArrayMappedTries", "Logging"]
git-tree-sha1 = "1147f140b4c8ddab224c94efa9569fc23d63ab44"
uuid = "7e506255-f358-4e82-b7e4-beb19740aa63"
version = "1.3.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "3bac05bc7e74a75fd9cba4295cde4045d9fe2386"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.1"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "e2cc6d8c88613c05e1defb55170bf5ff211fbeac"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.1"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "66e0a8e672a0bdfca2c3f5937efb8538b9ddc085"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.1"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.12.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "64cca0c26b4f31ba18f13f6c12af7c85f478cfde"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.5.0"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "e3be13f448a43610f978d29b7adf78c76022467a"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.12"
weakdeps = ["ChainRulesCore", "Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "192954ef1208c7019899fbf8049e717f92959682"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.3"

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
git-tree-sha1 = "1ff449ad350c9c4cbc756624d6f8a8c3ef56d3ed"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.0"

[[deps.StatsBase]]
deps = ["AliasTables", "DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "29321314c920c26684834965ec2ce0dacc9cf8e5"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.4"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "b423576adc27097764a90e163157bcfc9acf0f46"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.3.2"
weakdeps = ["ChainRulesCore", "InverseFunctions"]

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

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

[[deps.SymbolicIndexingInterface]]
deps = ["Accessors", "ArrayInterface", "RuntimeGeneratedFunctions", "StaticArraysCore"]
git-tree-sha1 = "fd2d4f0499f6bb4a0d9f5030f5c7d61eed385e03"
uuid = "2efcf032-c050-4f8e-a9bb-153293bab1f5"
version = "0.3.37"

[[deps.SymbolicLimits]]
deps = ["SymbolicUtils"]
git-tree-sha1 = "fabf4650afe966a2ba646cabd924c3fd43577fc3"
uuid = "19f23fe9-fdab-4a78-91af-e7b7767979c3"
version = "0.2.2"

[[deps.SymbolicUtils]]
deps = ["AbstractTrees", "ArrayInterface", "Bijections", "ChainRulesCore", "Combinatorics", "ConstructionBase", "DataStructures", "DocStringExtensions", "DynamicPolynomials", "IfElse", "LinearAlgebra", "MultivariatePolynomials", "NaNMath", "Setfield", "SparseArrays", "SpecialFunctions", "StaticArrays", "SymbolicIndexingInterface", "TermInterface", "TimerOutputs", "Unityper", "WeakValueDicts"]
git-tree-sha1 = "ae5e01353a02661d01514383b653d9de233274ea"
uuid = "d1185830-fcd6-423d-90d6-eec64667417b"
version = "3.15.0"

    [deps.SymbolicUtils.extensions]
    SymbolicUtilsLabelledArraysExt = "LabelledArrays"
    SymbolicUtilsReverseDiffExt = "ReverseDiff"

    [deps.SymbolicUtils.weakdeps]
    LabelledArrays = "2ee39098-c373-598a-b85f-a56591580800"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"

[[deps.Symbolics]]
deps = ["ADTypes", "ArrayInterface", "Bijections", "CommonWorldInvalidations", "ConstructionBase", "DataStructures", "DiffRules", "Distributions", "DocStringExtensions", "DomainSets", "DynamicPolynomials", "IfElse", "LaTeXStrings", "Latexify", "Libdl", "LinearAlgebra", "LogExpFunctions", "MacroTools", "Markdown", "NaNMath", "OffsetArrays", "PrecompileTools", "Primes", "RecipesBase", "Reexport", "RuntimeGeneratedFunctions", "SciMLBase", "Setfield", "SparseArrays", "SpecialFunctions", "StaticArraysCore", "SymbolicIndexingInterface", "SymbolicLimits", "SymbolicUtils", "TermInterface"]
git-tree-sha1 = "8bc0c65f76554ecff87a168893bd67dc5c55693f"
uuid = "0c5d862f-8b57-4792-8d23-62f2024744c7"
version = "6.29.0"

    [deps.Symbolics.extensions]
    SymbolicsForwardDiffExt = "ForwardDiff"
    SymbolicsGroebnerExt = "Groebner"
    SymbolicsLuxExt = "Lux"
    SymbolicsNemoExt = "Nemo"
    SymbolicsPreallocationToolsExt = ["PreallocationTools", "ForwardDiff"]
    SymbolicsSymPyExt = "SymPy"

    [deps.Symbolics.weakdeps]
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    Groebner = "0b43b601-686d-58a3-8a1c-6623616c7cd4"
    Lux = "b2108857-7c20-44ae-9111-449ecde12c47"
    Nemo = "2edaba10-b0f1-5616-af89-8c11ac63239a"
    PreallocationTools = "d236fae5-4411-538c-8e31-a6e3d9e00b46"
    SymPy = "24249f21-da20-56a4-8eb1-6a02cf4ae2e6"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "598cd7c1f68d1e205689b1c2fe65a9f85846f297"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.12.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.TermInterface]]
git-tree-sha1 = "d673e0aca9e46a2f63720201f55cc7b3e7169b16"
uuid = "8ea1fca8-c5ef-4a55-8b96-4e9afe9c9a3c"
version = "2.0.0"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.TimerOutputs]]
deps = ["ExprTools", "Printf"]
git-tree-sha1 = "d7298ebdfa1654583468a487e8e83fae9d72dac3"
uuid = "a759f4b9-e2f1-59dc-863e-4aeb61b1ea8f"
version = "0.5.26"

[[deps.Tricks]]
git-tree-sha1 = "6cae795a5a9313bbb4f60683f7263318fc7d1505"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.10"

[[deps.URIs]]
git-tree-sha1 = "67db6cc7b3821e19ebe75791a9dd19c9b1188f2b"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.1"

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

[[deps.Unityper]]
deps = ["ConstructionBase"]
git-tree-sha1 = "25008b734a03736c41e2a7dc314ecb95bd6bbdb0"
uuid = "a7c27f48-0311-42f6-a7f8-2c11e75eb415"
version = "0.1.6"

[[deps.WeakValueDicts]]
git-tree-sha1 = "98528c2610a5479f091d470967a25becfd83edd0"
uuid = "897b6980-f191-5a31-bcb0-bf3c4585e0c1"
version = "0.1.0"

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
# ╠═1768b806-0b79-416d-843e-7a2a64941706
# ╟─94bc7cd0-cfd1-4cc1-8930-f3fb37e0bcd4
# ╟─a055bbc2-cdc3-4d9b-93d0-d54accc5917a
# ╟─4dbd9bb4-9354-47d5-ae0d-cc192c4ad28b
# ╟─3e9cfdd1-e2f2-4c75-be6a-b15540054871
# ╟─6e8b13cc-f5f5-462f-94bd-06042cc9a7ad
# ╟─725659ff-77f1-46b5-9783-24100b3bc2c5
# ╠═23a93c6b-b53b-4433-8345-ab63e36e8a66
# ╠═ad22aeb5-5366-434f-b735-98f8f7dd7101
# ╠═38106955-da38-4cbd-9d40-7e641acf9015
# ╠═2d0a1baf-a070-4da6-8a86-e927dc0ff7f0
# ╠═79c4f0a8-53b5-460a-a210-1a683709cbc6
# ╠═33242a72-589f-47de-8127-b419957ca974
# ╠═3db949ff-1b4f-484f-b5b8-2ab78b54fd51
# ╠═371db73a-9093-4fa1-8ebd-953ac4e5e090
# ╠═b5b75d3d-d67d-4e38-819d-8315bed6aef4
# ╠═967f9eda-b39b-46b6-914b-c84d9acc056b
# ╠═419f6827-1dfd-40c9-bb73-53a9d5600edd
# ╠═ff5eddb8-b1f3-4d12-91bf-96438078f3cd
# ╠═a9701326-dd26-473f-8534-d7400e4bc489
# ╟─131f48ac-45fe-455a-a8f0-e99e9ce05d7a
# ╠═b2766b9a-1287-42f1-bcc7-d0656a4a5f26
# ╟─9df00f45-71be-41b5-9bbd-3ecba2493ddd
# ╠═3a22e16c-57b9-4b8f-b110-01edb1ed82b8
# ╠═d9acc5df-541b-448f-849f-228d6193b3dd
# ╠═dceb94d9-4819-4d76-99d6-badbf8371469
# ╠═e3205bc9-22d7-43bc-8068-4da0e6e71135
# ╠═27b428f0-015a-4eaa-9b16-75b69be4fe67
# ╠═ccb8342b-7d65-412c-b15a-2a28bc42f4da
# ╠═ff402573-bc3b-4b52-8549-2204d707aabe
# ╟─f7e14126-37f2-4811-b1c6-b095f549d198
# ╠═c5a8e002-35cd-4761-aa16-c2fdd1d7d270
# ╠═d520f33f-2347-4568-8b1e-7f0784ad515f
# ╠═61875817-4452-492d-b0f8-6d013ba414ef
# ╠═4b6aec66-93d7-4c20-83a4-87da5b481b9c
# ╠═6e6ac006-b078-4972-af27-ab24e3454e46
# ╠═ba7192d7-1973-4776-b44b-afd47727dbcd
# ╟─fa2989c0-b47b-4e3a-8d14-01d9433f2cd0
# ╠═4a87186b-0ee3-417c-b61b-2facc44a2d94
# ╠═002dcf11-7259-45dd-af1f-568cc4880706
# ╠═b3695eca-4652-4b3c-9d00-273fd6b5acf6
# ╠═f173dd19-506f-4b03-be65-dc3d0818f27c
# ╠═e080a1f4-be4c-4512-bf16-ca4e32eb62b9
# ╟─63ebb87b-3fc2-4f83-ad82-b6c3aea17a4e
# ╠═93779992-d2e2-452f-944e-009639f1d3db
# ╠═3a321ef1-6af2-431f-9e45-c8a512e92a9c
# ╟─deca9062-a489-49b2-af63-ea9d2422efc2
# ╟─3675562a-fed1-4d3f-be7a-29335204e56c
# ╟─849f32a0-8fda-49e7-b6e8-7968d71eb89a
# ╟─50924dce-1e92-4289-ae83-b3f9e540c7a8
# ╟─b3fdaab5-fa42-45ab-90ed-5eb4c7222169
# ╠═7811d81f-0d19-4e7e-ae02-a3889ba178ef
# ╟─b5195611-1309-42df-af5b-21eab516cb7d
# ╠═be836b79-c3e9-4b02-9958-b4f2539b5f29
# ╟─3060e567-b2a7-4ac7-b74b-dafbec4cb80b
# ╠═8a9061d2-8900-4669-a0d9-f3691e71952c
# ╠═f09d0aeb-ba0e-40d2-9941-49e92b71c2a5
# ╠═e8c9a972-7a84-4f45-937d-527c3d6d6f8e
# ╟─032d6aaf-1655-42fe-bc2c-0ea9b298cbf2
# ╠═abbe0e38-1ab9-4695-a2cb-abdb3cd9e5d4
# ╠═b5ac661a-94a3-424f-9628-9cccfcc0b8e7
# ╟─fc6049b4-0312-4135-ad7b-26851708a156
# ╠═c9edde32-a12b-11ed-3dea-3be5d6546919
# ╠═b2d3a472-ccda-4539-a8ce-6733012819eb
# ╟─85152808-fa2c-4d71-bb96-85845367a12b
# ╠═1225f1fe-abf9-4a1d-816f-210e54c2ae3d
# ╟─6ee10a18-d4bc-4c0e-9fcb-04b738bf7b1a
# ╟─a62159c8-f4a3-46cb-82a2-fd92f8711382
# ╠═c5e3add6-2446-4443-8f0d-708ceffd2033
# ╠═73007694-df0f-4fef-957a-b3c239c61bad
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
