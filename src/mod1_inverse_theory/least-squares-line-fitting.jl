### A Pluto.jl notebook ###
# v0.19.36

#> [frontmatter]
#> chapter = "1"
#> title = "Least Squares Line Fitting"
#> tags = ["module1"]
#> layout = "layout.jlhtml"

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ c9edde32-a12b-11ed-3dea-3be5d6546919
using LinearAlgebra, PlutoUI, PlutoTeachingTools, Symbolics, Distributions, PlutoPlotly

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

# ╔═╡ 419f6827-1dfd-40c9-bb73-53a9d5600edd
G⁻ᵍ = inv(transpose(G) * G) * transpose(G)

# ╔═╡ 967f9eda-b39b-46b6-914b-c84d9acc056b
y_nonoise = G * mtrue

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
plot(heatmap(x=["1", "2"],y=["1","2"], z=G' * G), Layout(title="Hessian Matrix", width=550))

# ╔═╡ f7e14126-37f2-4811-b1c6-b095f549d198
md"""### Lagrangian
The Lagrangian is a function of $m\in\mathbb{R}^N$ and $\lambda\in\mathbb{R}^P$, i.e., $N+P$ variables. We produce $N$ equations when differentiating w.r.t. each element of $m$ and $P$ equations when differentiating w.r.t. each element of $\lambda$.
"""

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

# ╔═╡ e3205bc9-22d7-43bc-8068-4da0e6e71135
begin
    recon
    x1 = rand(Uniform(1, 4))
    y1 = rand(Uniform(extrema(y)...))
end

# ╔═╡ 27b428f0-015a-4eaa-9b16-75b69be4fe67
H = [x1, 1]'

# ╔═╡ dceb94d9-4819-4d76-99d6-badbf8371469
md"""
## Constrained Problem
Number of constraints $P$
```math
Hm - h = 0
```
"""

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
plot(heatmap(x=string.(1:size(Rd,1)),y=string.(1:size(Rd,1)),z=Rd),Layout(title="Data Resolution Matrix", width=550))

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
plot(heatmap(x=["1","2"], y=["1","2"],z=Rm), Layout(title="Model Resolution Matrix", width=550))

# ╔═╡ fc6049b4-0312-4135-ad7b-26851708a156
md"## Appendix"

# ╔═╡ 85152808-fa2c-4d71-bb96-85845367a12b
md"### Global Temperature Data

Hansen, J., R. Ruedy, Mki. Sato, and K. Lo, 2010: Global surface temperature change. Rev. Geophys., 48, RG4004, doi:10.1029/2010RG000345"

# ╔═╡ 1225f1fe-abf9-4a1d-816f-210e54c2ae3d
 global_temp_data = [1965     -0.11
 1966     -0.03
 1967     -0.01
 1968     -0.04
 1969      0.08
 1970      0.03
 1971     -0.10
 1972      0.00
 1973      0.14
 1974     -0.08
 1975     -0.05
 1976     -0.16
 1977      0.12
 1978      0.01
 1979      0.08
 1980      0.19
 1981      0.26
 1982      0.04
 1983      0.25
 1984      0.09
 1985      0.04
 1986      0.12
 1987      0.27
 1988      0.31
 1989      0.19
 1990      0.36
 1991      0.35
 1992      0.13
 1993      0.13
 1994      0.23
 1995      0.37
 1996      0.29
 1997      0.39
 1998      0.56
 1999      0.32
 2000      0.33
 2001      0.47
 2002      0.56
 2003      0.55
 2004      0.48
 2005      0.62
 2006      0.55
 2007      0.58
 2008      0.44
 2009      0.58
 2010      0.63]

# ╔═╡ 4b6aec66-93d7-4c20-83a4-87da5b481b9c
plot([scatter(x=global_temp_data[:, 1], y=global_temp_data[:,2], name="a"),
	
], Layout(xaxis=attr(title="Year"), yaxis=attr(title="Temperature Anomaly (°C)")))

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
	s=[contour(x=yintr, y=sloper, z=Jbowl, colorscale="Hot", colorbar=attr(
	        thickness=25,
	        thicknessmode="pixels",
	        len=0.3,
	        lenmode="fraction",
	        outlinewidth=0
	    )), 
	scatter(x=[mtrue[2]], y=[mtrue[1]], mode="markers",name="True Model", marker=attr(color=:green)),
	scatter(x=[mest[2]], y=[mest[1]], mode="markers", name="Estimated Model", marker=attr(color=:blue))
	]
	if("Constraint" ∈ config)
		push!(s, scatter(x=[mestc[2]], y=[mestc[1]], mode="markers", name="Constrained Model", marker=attr(color=:red)) )
		push!(s, scatter(x=y1 .- sloper .* x1, y=yintr, mode="lines", name="Constraint", marker=attr(color=:red)) )
	end
	plot(s, Layout(xaxis=attr(title="Intercept"), yaxis=attr(title="Slope")))
end;

# ╔═╡ 73007694-df0f-4fef-957a-b3c239c61bad
pline = 
	let
s=[scatter(x=x, y=y, mode="markers", name="Observations", marker=attr(color=:black)), scatter(x=x, y=G*mest, mode="lines", name="Estimated Model", line=attr(color=:blue)), scatter(x=x, y=G*mtrue, mode="lines", name="True Model", line=attr(color=:green))]
		if("Constraint" ∈ config)
			push!(s, scatter(x=[x1], y=[y1], mode="markers", name="Constraint", marker=attr(color=:red)))
			push!(s, scatter(x=x, y=G* mestc[1:2], mode="lines", name="Constrained Model", line=attr(color=:red)))
		end
plot(s, Layout(xaxis=attr(title="Year"), yaxis=attr(title="Temperature Anomaly (°C)")))
	end;

# ╔═╡ 6e8b13cc-f5f5-462f-94bd-06042cc9a7ad
PlutoUI.ExperimentalLayout.vbox([pline, pbowl])

# ╔═╡ Cell order:
# ╠═1768b806-0b79-416d-843e-7a2a64941706
# ╟─94bc7cd0-cfd1-4cc1-8930-f3fb37e0bcd4
# ╠═a055bbc2-cdc3-4d9b-93d0-d54accc5917a
# ╠═4dbd9bb4-9354-47d5-ae0d-cc192c4ad28b
# ╟─3e9cfdd1-e2f2-4c75-be6a-b15540054871
# ╟─6e8b13cc-f5f5-462f-94bd-06042cc9a7ad
# ╠═725659ff-77f1-46b5-9783-24100b3bc2c5
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
# ╟─fc6049b4-0312-4135-ad7b-26851708a156
# ╠═c9edde32-a12b-11ed-3dea-3be5d6546919
# ╟─85152808-fa2c-4d71-bb96-85845367a12b
# ╟─1225f1fe-abf9-4a1d-816f-210e54c2ae3d
# ╟─6ee10a18-d4bc-4c0e-9fcb-04b738bf7b1a
# ╟─a62159c8-f4a3-46cb-82a2-fd92f8711382
# ╠═c5e3add6-2446-4443-8f0d-708ceffd2033
# ╠═73007694-df0f-4fef-957a-b3c239c61bad
