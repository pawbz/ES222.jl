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

# ╔═╡ 763a5056-c49e-11ee-0f48-874e88d8c2f4
using PlutoUI, LinearAlgebra, PlutoPlotly, DSP, ToeplitzMatrices, FFTW

# ╔═╡ 993a698e-63f7-4bf7-8403-c223658390a4
using HDF5

# ╔═╡ 2abf21a4-6204-486a-9325-bd07e3ad985f
TableOfContents()

# ╔═╡ c803646a-f37b-4316-898a-af80fd7b4d11
md"# Deconvolution"

# ╔═╡ d52bed16-b3cf-4d1c-8ade-4809f4da273c
md"""
## **Discrete Convolution**
    
Discrete convolution is an operation used in **signal processing, image processing, and deep learning**.
Given two discrete signals $f[n]$ and $g[n]$, their convolution is defined as:

```math
(f * g)[n] = \sum_{k=-\infty}^{\infty} f[k] g[n - k]
```

#### **How It Works**
1. **Flip** one of the signals (usually $g[n]$).
2. **Shift** it across $f[n]$.
3. **Multiply and Sum** at each step.

Below, we demonstrate discrete convolution.
"""

# ╔═╡ 8fbf3f1e-9b93-4771-8224-be833bc0ec22
@bind resetfg CounterButton("Reset")

# ╔═╡ 4143f162-fe52-4bc2-8083-fa91cfa114c9
md"""
|            | | | | |  |  |  |  | |
|:----------:|:----------:|:------------:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| f₁=$(@bind f1 Scrubbable(-1:0.1:1, default=0.9)) | f₂=$(@bind f2 Scrubbable(-1:0.1:1, default=0.0)) |    f₃=$(@bind f3 Scrubbable(-1:0.1:1, default=0.5))          | f₄=$(@bind f4 Scrubbable(-1:0.1:1, default=0.0)) | f₅=$(@bind f5 Scrubbable(-1:0.1:1, default=-0.9)) | f₆=$(@bind f6 Scrubbable(-1:0.1:1, default=0.0)) | f₇=$(@bind f7 Scrubbable(-1:0.1:1, default=0.0))| f₈=$(@bind f8 Scrubbable(-1:0.1:1, default=0.0)) | f₉=$(@bind f9 Scrubbable(-1:0.1:1, default=0.0)) | f₁₀=$(@bind f10 Scrubbable(-1:0.1:1, default=0.0)) |
| g₁=$(@bind g1 Scrubbable(-1:0.1:1, default=1.0))    | g₂=$(@bind g2 Scrubbable(-1:0.1:1, default=-1.0))  | g₃=$(@bind g3 Scrubbable(-1:0.1:1, default=0.0)) | g₄=$(@bind g4 Scrubbable(-1:0.1:1, default=0.0)) | g₅=$(@bind g5 Scrubbable(-1:0.1:1, default=0.0)) | g₆=$(@bind g6 Scrubbable(-1:0.1:1, default=0.0)) | g₇=$(@bind g7 Scrubbable(-1:0.1:1, default=0.0)) | g₈=$(@bind g8 Scrubbable(-1:0.1:1, default=0.0)) | g₉=$(@bind g9 Scrubbable(-1:0.1:1, default=0.0)) | g₁₀=$(@bind g10 Scrubbable(-1:0.1:1, default=0.0))  |

$(resetfg)
"""

# ╔═╡ d3c727a3-e7f4-45ce-84b1-8ecb2067a5a7
md"""
In the context of discrete convolution, the matrix `G = LowerTriangularToeplitz(g)` represents the Toeplitz matrix formulation of convolution, where g is the convolution kernel (or filter). This structure allows convolution to be expressed as a matrix-vector multiplication:

```math
(f \ast g) = G f,
```
where  $G$  is a lower triangular Toeplitz matrix, encoding shifted versions of $g$, and  $f$  is the input signal. This formulation is useful for understanding convolution as a linear transformation and efficiently computing convolutions using matrix operations.
"""

# ╔═╡ ddb89a12-4a7f-49a6-ae76-1264f3698958
f = [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10]

# ╔═╡ 9e0d2ba3-d700-4776-afcb-85cb21622b42
f

# ╔═╡ 2a544401-b722-45de-b202-84e74401171f
g = [g1, g2, g3, g4, g5, g6, g7, g8, g9, g10]

# ╔═╡ c73d160c-2094-4e7b-9fce-46680395c190
Gsimple = Array(LowerTriangularToeplitz(g))

# ╔═╡ 06393592-7a05-4ed4-9a3c-77219f2894fb
plot(heatmap(z=Gsimple))

# ╔═╡ bb1519d5-9495-4626-a28d-146bfa48e015
d_noise_less = Gsimple * f

# ╔═╡ 13d56b33-19d5-4d27-9356-587d8c947e4f
let
    p = make_subplots(rows=3, cols=1, shared_xaxes=true, vertical_spacing=0.02)
    add_trace!(p, scatter(x=1:length(d_noise_less), y=d_noise_less, name="conv(f,g)"), row=3, col=1)
    add_trace!(p, scatter(x=1:length(g), y=g, name="g"), row=2, col=1)
    add_trace!(p, scatter(x=1:length(f), y=f, name="f"), row=1, col=1)
    relayout!(p, title_text="Discrete Convolution", height=600, width=600)
    p
end

# ╔═╡ bd66f433-c586-4646-9814-8882d59a4db5
gsvd = svd(Gsimple);

# ╔═╡ ecbe9fda-a40a-4800-8c44-9ae67d88e3f4
G⁺ = gsvd.V * inv(diagm(gsvd.S)) * gsvd.U'

# ╔═╡ cb834d65-5476-4d38-982e-f811edef120c
msimple_est = G⁺ * d_noise_less

# ╔═╡ 471ba93f-fa75-487c-b5b6-485b2d317f66
let
    p = make_subplots(rows=2, cols=1, shared_xaxes=true, vertical_spacing=0.02)
    add_trace!(p, scatter(x=1:length(msimple_est), y=msimple_est, name="Estimated f "), row=2, col=1)
    add_trace!(p, scatter(x=1:length(f), y=f, name="True f"), row=1, col=1)
    relayout!(p, title_text="Deconvolution Without Noise", height=400, width=600)
    p
end

# ╔═╡ aa535f84-4338-46fe-8ef8-3d1d91e1fa22
plot(svd(G).S)

# ╔═╡ e6ef86b4-e67d-4553-9cc1-9c9a3060e186
rank(G)

# ╔═╡ b10e40aa-bd1e-41f1-a73c-950da681e584
G_pinv = pinv(G)

# ╔═╡ 1bc08a78-3f5d-43dc-bf5c-cbb33ad18cf3
plot(heatmap(z=G_pinv))

# ╔═╡ 2d17f016-2f66-4d1c-8cd7-4d8bb201e352
plot(G_pinv[:, 3])

# ╔═╡ 1cf485d5-1847-4ff7-b3ee-e52c4ef32133
plot(svd(G).U[:, end-1])

# ╔═╡ 39b70471-b546-420c-92b7-0a5159d4bc02
fhat = pinv(G) * d1

# ╔═╡ 5b098aa5-daa5-4817-a822-a954578b5fef
plot(fhat)

# ╔═╡ b29add71-bf30-4e12-b8e9-3babfd0614b3
plot(f)

# ╔═╡ 265d0778-8734-45ae-b37f-ad2a05b74563
md"## Deconvolution Using Airgun Signal"

# ╔═╡ ed8c354b-3566-45a3-8a71-3dc4ae6fad69
md"We aim to construct an inverse filter of the following
signal"

# ╔═╡ 01f102eb-a7cd-4cf8-a9fb-26a01a200fd5
md"Toeplitz matrix to perform convolution"

# ╔═╡ 81f5c864-8388-46eb-95e1-b0b295d6ebff
md"## Appendix"

# ╔═╡ b752fd54-2a58-4fb6-b4e1-e12a72bff7aa
"""
Computes the pseudo-inverse (or regular inverse if full rank) of a matrix A using SVD.

Parameters:
    - A: Input matrix (square or rectangular)
    - tol: Singular value threshold for filtering (default: 1e-6)

Returns:
    - A_inv: Inverse or pseudo-inverse of A
"""
function svd_inverse(A; tol=1e-6)

    # Perform Singular Value Decomposition (SVD)
    Asvd = svd(A, full=true)

    # Filter singular values based on tolerance
    S_inv = Diagonal([(s > tol * maximum(Asvd.S)) ? inv(s) : 0.0 for s in Asvd.S])

    # Compute pseudo-inverse using SVD components
    A_inv = Asvd.V * S_inv * Asvd.U'

    return A_inv
end

# ╔═╡ f567b59f-025e-4120-ab02-e0e021aa9c6c
# ╠═╡ disabled = true
#=╠═╡
Gairgun_pinv = pinv(Gairgun_noise)
  ╠═╡ =#

# ╔═╡ d3ceb1b0-c2de-44bd-8e56-4bf3d71354b6
svd_inverse_tol = @bind tol Slider(range(0, 0.5, length=1000), show_value=true)

# ╔═╡ 42176a5b-ec56-4c23-b8b5-41a0875e5a90
# m=zeros(size(airgun, 1)); m[im]=1.0; m[100]=-0.5; m[120]=0.8; d=Gairgun*m

# ╔═╡ 4daf46e8-ea43-4f2f-90a9-3453ffc55ee3
md"Noise $(noise_slider = @bind eps Slider(range(0.0, 10, length=1000), show_value=true))"

# ╔═╡ ac830045-7a9d-4f76-9da8-2685b6c62804
svd_inverse_tol

# ╔═╡ c28befbf-d91c-4e08-a743-d13c06b3aa26
svd_inverse_tol

# ╔═╡ 7652a4f4-2a9a-4ec9-88cc-b27a4a759b0d
noise_slider

# ╔═╡ 23b4f6ba-68f1-406b-a269-c09e4782995f
md"""
## **Coordinate Descent for Inverse Problems with $\ell_1$ or $\ell_2$ Regularization**  

Coordinate Descent is an iterative optimization algorithm that solves inverse problems of the form:  

```math
\min_x \| G x - b \|_2^2 + \lambda R(x)
```

where $G$ is the forward operator (matrix), $b$ is the observed data, and $R(x)$ is the chosen regularization term. The algorithm updates one coordinate $x_j$ at a time while keeping others fixed.  

**Key Steps of the Algorithm**  

1. **Initialization**  
   - Start with an initial guess $x = 0$.  
   - Compute preconditioned values $G^T G$ and $G^T b$ for efficiency.  

2. **Coordinate-wise Update for Each $j$**  
   - Compute the residual for coordinate $j$:  

     ```math
     r_j = (G^T b)_j - \sum_{k \neq j} (G^T G)_{jk} x_k
     ```

   - Update $x_j$ based on the chosen regularization:
     - **Lasso ($\ell_1$ Regularization)** (Soft-thresholding shrinkage):  

       ```math
       x_j =
       \begin{cases} 
       \frac{\operatorname{sign}(r_j) \cdot \max(|r_j| - \lambda, 0)}{(G^T G)_{jj}}, & \text{if } |r_j| > \lambda \\
       0, & \text{otherwise}
       \end{cases}
       ```

     - **Ridge ($\ell_2$ Regularization)** (Closed-form update):  

       ```math
       x_j = \frac{r_j}{(G^T G)_{jj} + \lambda}
       ```

3. **Convergence Check**  
   - Stop if the solution change is below a set tolerance:  

     ```math
     \| x_{\text{new}} - x_{\text{old}} \| < \epsilon
     ```

4. **Repeat Until Convergence**  
   - Iterate until the stopping criterion is met.

This method efficiently solves sparse regression problems (Lasso) and smooth solutions (Ridge) while iterating only over relevant coordinates.

"""

# ╔═╡ 70ca9052-27fa-477d-a3a5-55ec1ebd6116
    """
    Solves the inverse problem Ax = b using coordinate descent 
    with L1 (Lasso) or L2 (Ridge) regularization.

    Parameters:
    - A: Forward operator (matrix)
    - b: Observed data (vector)
    - λ: Regularization parameter
    - max_iter: Maximum number of iterations
    - tol: Convergence tolerance
    - norm_type: "L1" for Lasso, "L2" for Ridge

    Returns:
    - x: Solution vector
    """
function coordinate_descent(A, b; λ=0.1, max_iter=100, tol=1e-6, norm_type="L2")


    # Initialize solution
    m, n = size(A)
    x = zeros(n)  # Initial guess

    # Precompute necessary values
    AtA = A' * A
    Atb = A' * b
    diag_AtA = diag(AtA)  # Diagonal elements (used in coordinate updates)

    for iter in 1:max_iter
        x_old = copy(x)

        for j in 1:n
            # Compute residual excluding the j-th coordinate
            r_j = Atb[j] - sum(AtA[j, :] .* x) + AtA[j, j] * x[j]

            if norm_type == "L1"  # Lasso (L1)
                if abs(r_j) > λ
                    x[j] = sign(r_j) * max(abs(r_j) - λ, 0) / diag_AtA[j]
                else
                    x[j] = 0
                end
            elseif norm_type == "L2"  # Ridge (L2)
                x[j] = r_j / (diag_AtA[j] + λ)
            else
                error("norm_type must be either 'L1' or 'L2'")
            end
        end

        # Check for convergence
        if norm(x - x_old) < tol
            println("Converged in $iter iterations.")
            break
        end
    end

    return x
end

# ╔═╡ 32e85306-7fea-4b1b-8f3d-5c644020f0c0
noise_slider

# ╔═╡ 0a7354ec-56b2-4954-a924-56dd7e384de9
CD_lambda_slider = @bind CD_lambda Slider(range(0, 100,length=1000))

# ╔═╡ 065cef58-9fe9-4431-8d5d-2fe531278b43
noise_slider

# ╔═╡ dbfc6c64-0ef6-472c-b034-4237618317b6
CD_lambda_slider

# ╔═╡ fb696746-c51a-4331-9320-d5c3c9ccec6c
plot(m, Layout(title="True f(t)"))

# ╔═╡ 182fe741-7c23-4acc-98b2-b491483c508b


# ╔═╡ 29284a75-8dc4-4f16-8fbb-934c83e0a75b
plot(restored_d2)

# ╔═╡ 6536fc8d-2ef6-401f-8852-78db831d0bca
#=╠═╡
plot(Gairgun_pinv * d)
  ╠═╡ =#

# ╔═╡ d5d13ce3-e2e2-49d8-a033-fef5105e0cb1
#=╠═╡
plot((Gairgun_pinv*Gairgun)[:, 100])
  ╠═╡ =#

# ╔═╡ fd349977-6d95-44ca-90db-b4d092cb0efa
#=╠═╡
plot(heatmap(z=Gairgun_pinv * Gairgun))
  ╠═╡ =#

# ╔═╡ c6099f88-9061-437d-9e2d-cd38430ccf3d
#=╠═╡
plot(heatmap(z=Gairgun * Gairgun_pinv))
  ╠═╡ =#

# ╔═╡ 65a936ea-d82a-48ad-812d-bc3d81a90d72
plot(heatmap(z=G_pinv * G))

# ╔═╡ 28e8a487-fb76-49ad-a0d2-ba73d838bcfa
function get_airgun()
    airgun = [0.0019455 0.31999
        0.0038911 0.66316
        0.0058366 0.86648
        0.0077821 0.96656
        0.0097276 1
        0.011673 0.93253
        0.013619 -0.34515
        0.015564 -0.44781
        0.01751 -0.48336
        0.019455 -0.5189
        0.021401 -0.55209
        0.023346 -0.59392
        0.025292 -0.68945
        0.027237 -0.81781
        0.029183 -0.91316
        0.031128 -0.78497
        0.033074 -0.38476
        0.035019 -0.076951
        0.036965 0.086752
        0.038911 0.21784
        0.040856 0.29985
        0.042802 0.32102
        0.044747 0.32677
        0.046693 0.33102
        0.048638 0.3338
        0.050584 0.33514
        0.052529 0.32802
        0.054475 0.29396
        0.05642 0.26859
        0.058366 0.25177
        0.060311 0.23896
        0.062257 0.23429
        0.064202 0.24808
        0.066148 0.27332
        0.068093 0.2945
        0.070039 0.31735
        0.071984 0.33727
        0.07393 0.34648
        0.075875 0.14177
        0.077821 -0.37562
        0.079767 -0.87791
        0.081712 -0.89059
        0.083658 -0.89326
        0.085603 -0.80043
        0.087549 -0.65179
        0.089494 -0.47611
        0.09144 -0.27827
        0.093385 -0.074001
        0.095331 0.10743
        0.097276 0.24002
        0.099222 0.29809
        0.10117 0.31486
        0.10311 0.31903
        0.10506 0.32185
        0.107 0.32347
        0.10895 0.32406
        0.11089 0.31521
        0.11284 0.29185
        0.11479 0.26807
        0.11673 0.25313
        0.11868 0.23954
        0.12062 0.23031
        0.12257 0.22912
        0.12451 0.23333
        0.12646 0.23991
        0.1284 0.24646
        0.13035 0.25062
        0.1323 0.24505
        0.13424 0.20885
        0.13619 -0.015198
        0.13813 -0.2673
        0.14008 -0.34262
        0.14202 -0.42791
        0.14397 -0.42477
        0.14591 -0.41847
        0.14786 -0.40802
        0.14981 -0.36234
        0.15175 -0.2916
        0.1537 -0.19553
        0.15564 -0.093273
        0.15759 -0.0030751
        0.15953 0.089305
        0.16148 0.14768
        0.16342 0.18526
        0.16537 0.20179
        0.16732 0.20756
        0.16926 0.211
        0.17121 0.21146
        0.17315 0.20567
        0.1751 0.19503
        0.17704 0.18266
        0.17899 0.1717
        0.18093 0.16365
        0.18288 0.15677
        0.18482 0.14909
        0.18677 0.13862
        0.18872 0.12337
        0.19066 0.080138
        0.19261 0.0028842
        0.19455 -0.093647
        0.1965 -0.18345
        0.19844 -0.22087
        0.20039 -0.24756
        0.20233 -0.25953
        0.20428 -0.25981
        0.20623 -0.25433
        0.20817 -0.22079
        0.21012 -0.1688
        0.21206 -0.11249
        0.21401 -0.063461
        0.21595 -0.0069556
        0.2179 0.04728
        0.21984 0.080277
        0.22179 0.088373
        0.22374 0.094356
        0.22568 0.099036
        0.22763 0.1024
        0.22957 0.10443
        0.23152 0.10511
        0.23346 0.10345
        0.23541 0.098763
        0.23735 0.091573
        0.2393 0.082384
        0.24125 0.071706
        0.24319 0.060049
        0.24514 0.04736
        0.24708 0.030436
        0.24903 0.0072266
        0.25097 -0.024014
        0.25292 -0.090696
        0.25486 -0.14445
        0.25681 -0.16997
        0.25875 -0.16849
        0.2607 -0.16494
        0.26265 -0.15984
        0.26459 -0.15374
        0.26654 -0.14395
        0.26848 -0.12755
        0.27043 -0.10693
        0.27237 -0.084535
        0.27432 -0.062808
        0.27626 -0.042372
        0.27821 -0.020829
        0.28016 0.00026162
        0.2821 0.02051
        0.28405 0.043459
        0.28599 0.06294
        0.28794 0.07142
        0.28988 0.070491
        0.29183 0.068005
        0.29377 0.064614
        0.29572 0.060962
        0.29767 0.05753
        0.29961 0.054005
        0.30156 0.049899
        0.3035 0.044727
        0.30545 0.038004
        0.30739 0.027892
        0.30934 -0.0011971
        0.31128 -0.037038
        0.31323 -0.05843
        0.31518 -0.067327
        0.31712 -0.074797
        0.31907 -0.080552
        0.32101 -0.084303
        0.32296 -0.08576
        0.3249 -0.082045
        0.32685 -0.0714
        0.32879 -0.056901
        0.33074 -0.041651
        0.33268 -0.028728
        0.33463 -0.016548
        0.33658 -0.0030656
        0.33852 0.010982
        0.34047 0.024856
        0.34241 0.037819
        0.34436 0.049133
        0.3463 0.058061
        0.34825 0.063865
        0.35019 0.065807
        0.35214 0.063886
        0.35409 0.058991
        0.35603 0.051847
        0.35798 0.043181
        0.35992 0.033721
        0.36187 0.024001
        0.36381 0.010948
        0.36576 -0.0048216
        0.3677 -0.020755
        0.36965 -0.034907
        0.3716 -0.050599
        0.37354 -0.065825
        0.37549 -0.076856
        0.37743 -0.080056
        0.37938 -0.076418
        0.38132 -0.068503
        0.38327 -0.057628
        0.38521 -0.045108
        0.38716 -0.027993
        0.38911 -0.0018992
        0.39105 0.016215
        0.393 0.025808
        0.39494 0.033916
        0.39689 0.039877
        0.39883 0.043029
        0.40078 0.043154
        0.40272 0.041901
        0.40467 0.039605
        0.40661 0.036386
        0.40856 0.032366
        0.41051 0.027664
        0.41245 0.019005
        0.4144 0.0014679
        0.41634 -0.020129
        0.41829 -0.040631
        0.42023 -0.055326
        0.42218 -0.06826
        0.42412 -0.077979
        0.42607 -0.079818
        0.42802 -0.075647
        0.42996 -0.067681
        0.43191 -0.057143
        0.43385 -0.045255
        0.4358 -0.033241
        0.43774 -0.022322
        0.43969 -0.012652
        0.44163 -0.0013361
        0.44358 0.0096963
        0.44553 0.017938
        0.44747 0.020899
        0.44942 0.020899
        0.45136 0.020899
        0.45331 0.020899
        0.45525 0.020899
        0.4572 0.018299
        0.45914 0.010555
        0.46109 0.0002438
        0.46304 -0.010003
        0.46498 -0.018837
        0.46693 -0.028262
        0.46887 -0.037434
        0.47082 -0.045266
        0.47276 -0.051804
        0.47471 -0.058434
        0.47665 -0.064617
        0.4786 -0.069758
        0.48054 -0.073264
        0.48249 -0.074541
        0.48444 -0.074541
        0.48638 -0.074541
        0.48833 -0.074541
        0.49027 -0.074541
        0.49222 -0.074541
        0.49416 -0.071528
        0.49611 -0.061626
        0.49805 -0.047009
        0.5 -0.029947]

    airgun_t = airgun[:, 1]
    airgun = airgun[:, 2]
    # airgun .*= tukey(length(airgun), 0.5)
    # airgun = vcat(airgun, zeros(200))
    # nt=length(airgun)
    # responsetype = Bandpass(0.15, 0.35)
    # designmethod = Butterworth(2)
    # airgun = filtfilt(digitalfilter(responsetype, designmethod), airgun)


    # airgun .=  Toeplitz(vcat(ones(100), zeros(nt-100)),vcat([1],zeros(nt-1))) * airgun
    return airgun_t, airgun
end

# ╔═╡ 0d451ea3-e3d8-4f87-9f8e-9bcb39ccf1b6
# ╠═╡ show_logs = false
airgun_t, airgun = get_airgun()

# ╔═╡ 077b3192-cf54-4fe5-9643-ec681be0b13d
plot(airgun_t, airgun, Layout(width=500, height=250, xaxis=attr(title="time [s]"), yaxis=attr(title="amplitude"), title="Source Signal g(t)"))

# ╔═╡ 929ee3d8-044e-49b3-b4d5-22abf128d2e8
Gairgun = Array(LowerTriangularToeplitz(airgun));

# ╔═╡ 1ad3e30a-ffa0-46a6-a93f-e2bbc7136d2c
svd_Gairgun = svd(Gairgun)

# ╔═╡ 68686c13-af19-4cea-85ea-34a82a7d3570
plot(svd_Gairgun.S, Layout(title="Singular Values of G"))

# ╔═╡ b36c9a52-0e7b-4a34-a916-faead208c461
@bind isG Slider(1:length(svd_Gairgun.S), show_value=true)

# ╔═╡ 66eee112-45cc-44a0-b24b-ac7b672c187f
plot(heatmap(z=Gairgun), Layout(yaxis=attr(autorange="reversed"), title="G: Toeplitz Matrix To Perform Convolution With g(t) "))

# ╔═╡ 26c47017-5a7a-4ddf-9644-2e0246cc6f7b
plot(svd(Gairgun).U[:, isG], Layout(title="Singular Vector $(isG) of G"))

# ╔═╡ b8dd220a-faef-4091-ada1-44a9aa016fdd
Gairgun⁺ = svd_inverse(Gairgun, tol=tol)

# ╔═╡ 46b24cde-c224-4255-b49e-d76fef9952a2
d = Gairgun * m

# ╔═╡ 152c2c96-09e2-4679-95a6-bf062b4662b8
dnoise = eps * randn(length(d)) + d

# ╔═╡ bb96996a-a259-48b5-b657-a07cc9409cb5
plot(hcat(d, dnoise))

# ╔═╡ dd2a0b9e-096b-4c61-a50d-3ad99fc2bcfa
plot(randn(length(d)))

# ╔═╡ 5f184996-b3e8-4ef6-919a-12a3b45128b1
plot(pinv(Gairgun) * d, Layout(title="Estimated f(t)"))

# ╔═╡ 5ec8f71d-6f9c-4f19-9ed5-5bc2073d5fd6
plot(hcat(m, pinv(Gairgun, rtol=rtol_pinv) * dnoise))

# ╔═╡ 7447a2d2-e901-48f1-976d-48010c37a8fc
rank(Gairgun)

# ╔═╡ 6ff1d8f2-43c1-4232-860e-cdd957882938
begin
    m_pulses = exp.(-20000 * (airgun_t .- 0.15) .^ 2)
    m_pulses .+= -0.5 * exp.(-200000 * (airgun_t .- 0.2) .^ 2)
    m_pulses .+= 0.3 * exp.(-20000 * (airgun_t .- 0.25) .^ 2)
end

# ╔═╡ 61a30c42-a22b-439a-82bc-7346f4018ab7
d_pulses = Gairgun * m_pulses

# ╔═╡ cea0cdea-2dcf-4e5b-895a-5b26a6bfc6be
d_pulses_noise = d_pulses + eps*randn(length(d_pulses))

# ╔═╡ aadd65c5-361b-460e-8b46-ae27ac57d870
m_pulses_est = Gairgun⁺ * d_pulses_noise

# ╔═╡ eef0fa00-dbdc-4be7-8c65-d624612f5499
plot(d_pulses_noise)

# ╔═╡ bdba3e79-8784-4dff-925e-c3aeac83c74a
m_pulses_est_CD_1 = coordinate_descent(Gairgun, d_pulses_noise, λ=CD_lambda, norm_type="L1")

# ╔═╡ a945c1e8-851f-4ce3-b1ed-71fedae1c229
m_pulses_est_CD_2 = coordinate_descent(Gairgun, d_pulses_noise, λ=CD_lambda, norm_type="L2")

# ╔═╡ 6491c0d0-7055-4798-8a8e-6b4acb5d037d
plot(m_pulses)

# ╔═╡ f2aea8c1-d860-4a85-ae3b-fa43e8d598bf
let
    p = make_subplots(rows=2, cols=1, shared_xaxes=true, vertical_spacing=0.02)
    add_trace!(p, scatter(x=1:length(m_pulses_est), y=m_pulses_est, name="Estimated f "), row=2, col=1)
    add_trace!(p, scatter(x=1:length(m_pulses), y=m_pulses, name="True f"), row=1, col=1)
    relayout!(p, title_text="Deconvolution Without Noise", height=400, width=600)
    p
end

# ╔═╡ cc4a10c2-27c6-495b-9d08-0ab7bf597ae1
let
    p = make_subplots(rows=3, cols=1, shared_xaxes=true, vertical_spacing=0.02)
    add_trace!(p, scatter(x=1:length(m_pulses_est_CD_1), y=m_pulses_est_CD_1, name="Estimated m L1"), row=2, col=1)
	 add_trace!(p, scatter(x=1:length(m_pulses_est_CD_2), y=m_pulses_est_CD_2, name="Estimated m L2"), row=3, col=1)
    add_trace!(p, scatter(x=1:length(m_pulses), y=m_pulses, name="True m"), row=1, col=1)
    relayout!(p, title_text="Deconvolution With Noise", height=600, width=600)
    p
end

# ╔═╡ 224ab916-9f0a-4bb4-9a4d-408973056af4
begin
	m_sparse = zero(airgun_t)
	m_sparse[50] = 1.0
end

# ╔═╡ f6f8ee9c-2f1f-4e36-932c-a7d610e0cb91
d_sparse = Gairgun * m_sparse

# ╔═╡ 8e361e89-b787-4f5e-88c6-eb3f9431de93
d_sparse_noise = d_sparse + eps* randn(length(d_sparse))

# ╔═╡ 03d5e2e6-0ca5-4e45-a7f8-37acbdd5a5c8
m_sparse_est = Gairgun⁺ * d_sparse_noise

# ╔═╡ cc3dde37-fc3e-4e29-9c1a-8e4184038084
m_sparse_est_CD_1 = coordinate_descent(Gairgun, d_sparse_noise, λ=CD_lambda, norm_type="L1")

# ╔═╡ 25fba4c1-852a-4aa9-a631-1663fba7c437
m_sparse_est_CD_2 = coordinate_descent(Gairgun, d_sparse_noise, λ=CD_lambda, norm_type="L2")

# ╔═╡ e994ed04-1622-4966-adff-fb23cb037fc8
let
    p = make_subplots(rows=2, cols=1, shared_xaxes=true, vertical_spacing=0.02)
    add_trace!(p, scatter(x=1:length(m_sparse_est), y=m_sparse_est, name="Estimated f "), row=2, col=1)
    add_trace!(p, scatter(x=1:length(m_sparse), y=m_sparse, name="True f"), row=1, col=1)
    relayout!(p, title_text="Deconvolution With Noise", height=400, width=600)
    p
end

# ╔═╡ 8fdaa76a-11e7-4320-926f-3da264920167
let
    p = make_subplots(rows=3, cols=1, shared_xaxes=true, vertical_spacing=0.02)
    add_trace!(p, scatter(x=1:length(m_sparse_est_CD_1), y=m_sparse_est_CD_1, name="Estimated f "), row=2, col=1)
	 add_trace!(p, scatter(x=1:length(m_sparse_est_CD_2), y=m_sparse_est_CD_2, name="Estimated f "), row=3, col=1)
    add_trace!(p, scatter(x=1:length(m_sparse), y=m_sparse, name="True f"), row=1, col=1)
    relayout!(p, title_text="Deconvolution With Noise", height=600, width=600)
    p
end

# ╔═╡ f8c84f16-d1f7-479f-95f8-5808f254fba4
h5open("deconvolution.h5", "w") do file
    write(file, "data", dnoise)
    write(file, "impulse_response", airgun)
    write(file, "time", airgun_t)# alternatively, say "@write file A"
end

# ╔═╡ c9db7307-9ec8-421d-92d0-8326658e0c16
length(airgun)

# ╔═╡ f27ae334-0999-4d9a-9f65-81ffeeed98a3
length(airgun_t)

# ╔═╡ 9a9da957-900f-4ee4-9fc5-a8b519610552
@bind im Slider(1:length(airgun), show_value=true)

# ╔═╡ 64389d4b-8ec7-45d4-9534-8f32359fba1d
plot(abs.(rfft(airgun)))

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
DSP = "717857b8-e6f2-59f4-9121-6e50c889abd2"
FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
HDF5 = "f67ccb44-e63f-5c2f-98bd-6dc0ccc4ba2f"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
PlutoPlotly = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
ToeplitzMatrices = "c751599d-da0a-543b-9d20-d0a503d91d24"

[compat]
DSP = "~0.7.9"
FFTW = "~1.8.0"
HDF5 = "~0.17.1"
PlutoPlotly = "~0.4.4"
PlutoUI = "~0.7.55"
ToeplitzMatrices = "~0.8.3"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.4"
manifest_format = "2.0"
project_hash = "73205aa1faea6ed79f3e623f62d5199f90668ca8"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

    [deps.AbstractFFTs.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

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

[[deps.BaseDirs]]
git-tree-sha1 = "cb25e4b105cc927052c2314f8291854ea59bf70a"
uuid = "18cc8868-cbac-4acf-b575-c8ff214dc66f"
version = "1.2.4"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "26ec26c98ae1453c692efded2b17e15125a5bea1"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.28.0"

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

[[deps.ConstructionBase]]
git-tree-sha1 = "76219f1ed5771adbb096743bff43fb5fdd4c1157"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.8"

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseLinearAlgebraExt = "LinearAlgebra"
    ConstructionBaseStaticArraysExt = "StaticArrays"

    [deps.ConstructionBase.weakdeps]
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.DSP]]
deps = ["Compat", "FFTW", "IterTools", "LinearAlgebra", "Polynomials", "Random", "Reexport", "SpecialFunctions", "Statistics"]
git-tree-sha1 = "0df00546373af8eee1598fb4b2ba480b1ebe895c"
uuid = "717857b8-e6f2-59f4-9121-6e50c889abd2"
version = "0.7.10"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.7.0"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "7de7c78d681078f027389e067864a8d53bd7c3c9"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.8.1"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4d81ed14783ec49ce9f2e168208a12ce1815aa25"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+3"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "6a70198746448456524cb442b8af316927ff3e1a"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.13.0"

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

    [deps.FillArrays.weakdeps]
    PDMats = "90014a1f-27ba-587c-ab20-58faa44d9150"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"
version = "1.11.0"

[[deps.HDF5]]
deps = ["Compat", "HDF5_jll", "Libdl", "MPIPreferences", "Mmap", "Preferences", "Printf", "Random", "Requires", "UUIDs"]
git-tree-sha1 = "e856eef26cf5bf2b0f95f8f4fc37553c72c8641c"
uuid = "f67ccb44-e63f-5c2f-98bd-6dc0ccc4ba2f"
version = "0.17.2"

    [deps.HDF5.extensions]
    MPIExt = "MPI"

    [deps.HDF5.weakdeps]
    MPI = "da04e1cc-30fd-572f-bb4f-1f8673147195"

[[deps.HDF5_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LazyArtifacts", "LibCURL_jll", "Libdl", "MPICH_jll", "MPIPreferences", "MPItrampoline_jll", "MicrosoftMPI_jll", "OpenMPI_jll", "OpenSSL_jll", "TOML", "Zlib_jll", "libaec_jll"]
git-tree-sha1 = "82a471768b513dc39e471540fdadc84ff80ff997"
uuid = "0234f1f7-429e-5d53-9886-15a909be8d59"
version = "1.14.3+3"

[[deps.Hwloc_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "50aedf345a709ab75872f80a2779568dc0bb461b"
uuid = "e33a78d0-f292-5ffc-b300-72abe9b543c8"
version = "2.11.2+3"

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

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "0f14a5456bdc6b9731a5682f439a672750a09e48"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2025.0.4+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "e2222959fbc6c19554dc15174c81bf7bf3aa691c"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.4"

[[deps.IterTools]]
git-tree-sha1 = "42d5f897009e7ff2cf88db414a389e5ed1bdd023"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.10.0"

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

[[deps.JuliaSyntaxHighlighting]]
deps = ["StyledStrings"]
uuid = "ac6e5ff7-fb65-4e79-a425-ec3bc9c03011"
version = "1.12.0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"
version = "1.11.0"

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
git-tree-sha1 = "1833212fd6f580c20d4291da9c1b4e8a655b128e"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.0.0"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "oneTBB_jll"]
git-tree-sha1 = "5de60bc6cb3899cd318d80d627560fae2e2d99ae"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2025.0.1+1"

[[deps.MPICH_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Hwloc_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "MPIPreferences", "TOML"]
git-tree-sha1 = "7715e65c47ba3941c502bffb7f266a41a7f54423"
uuid = "7cb0a576-ebde-5e09-9194-50597f1243b4"
version = "4.2.3+0"

[[deps.MPIPreferences]]
deps = ["Libdl", "Preferences"]
git-tree-sha1 = "c105fe467859e7f6e9a852cb15cb4301126fac07"
uuid = "3da0fdf6-3ccc-4f1b-acd9-58baa6c99267"
version = "0.1.11"

[[deps.MPItrampoline_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "MPIPreferences", "TOML"]
git-tree-sha1 = "70e830dab5d0775183c99fc75e4c24c614ed7142"
uuid = "f1f71cc9-e9ae-5b93-9b94-4fe0e1ad3748"
version = "5.5.1+2"

[[deps.MacroTools]]
git-tree-sha1 = "72aebe0b5051e5143a079a4685a46da330a40472"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.15"

[[deps.Markdown]]
deps = ["Base64", "JuliaSyntaxHighlighting", "StyledStrings"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MicrosoftMPI_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bc95bf4149bf535c09602e3acdf950d9b4376227"
uuid = "9237b28f-5490-5468-be7b-bb81f5f5e6cf"
version = "10.1.4+3"

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

[[deps.OpenMPI_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "MPIPreferences", "TOML"]
git-tree-sha1 = "e25c1778a98e34219a00455d6e4384e017ea9762"
uuid = "fe0851c0-eecd-5654-98d4-656369965a5c"
version = "4.1.6+0"

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

[[deps.PlutoPlotly]]
deps = ["AbstractPlutoDingetjes", "BaseDirs", "Colors", "Dates", "Downloads", "HypertextLiteral", "InteractiveUtils", "LaTeXStrings", "Markdown", "Pkg", "PlotlyBase", "Reexport", "TOML"]
git-tree-sha1 = "1ae939782a5ce9a004484eab5416411c7190d3ce"
uuid = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
version = "0.4.6"

    [deps.PlutoPlotly.extensions]
    PlotlyKaleidoExt = "PlotlyKaleido"
    UnitfulExt = "Unitful"

    [deps.PlutoPlotly.weakdeps]
    PlotlyKaleido = "f2990250-8cf9-495f-b13a-cce12b45703c"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "7e71a55b87222942f0f9337be62e26b1f103d3e4"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.61"

[[deps.Polynomials]]
deps = ["LinearAlgebra", "OrderedCollections", "RecipesBase", "Requires", "Setfield", "SparseArrays"]
git-tree-sha1 = "56ff43adee1b1449fa151265b0088fe1c14dfd7d"
uuid = "f27b6e38-b328-58d1-80ce-0feddd5e7a45"
version = "4.0.14"

    [deps.Polynomials.extensions]
    PolynomialsChainRulesCoreExt = "ChainRulesCore"
    PolynomialsFFTWExt = "FFTW"
    PolynomialsMakieCoreExt = "MakieCore"
    PolynomialsMutableArithmeticsExt = "MutableArithmetics"

    [deps.Polynomials.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
    MakieCore = "20f20a25-4f0e-4fdf-b5d1-57303727442b"
    MutableArithmetics = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"

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

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

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

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

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

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.12.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "64cca0c26b4f31ba18f13f6c12af7c85f478cfde"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.5.0"

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

    [deps.SpecialFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"

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

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

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

[[deps.ToeplitzMatrices]]
deps = ["AbstractFFTs", "DSP", "FillArrays", "LinearAlgebra"]
git-tree-sha1 = "338d725bd62115be4ba7ffa891d85654e0bfb1a1"
uuid = "c751599d-da0a-543b-9d20-d0a503d91d24"
version = "0.8.5"

    [deps.ToeplitzMatrices.extensions]
    ToeplitzMatricesStatsBaseExt = "StatsBase"

    [deps.ToeplitzMatrices.weakdeps]
    StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"

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

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.3.1+2"

[[deps.libaec_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "f5733a5a9047722470b95a81e1b172383971105c"
uuid = "477f73a3-ac25-53e9-8cc3-50b2fa2566f0"
version = "1.1.3+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.15.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.64.0+1"

[[deps.oneTBB_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7d0ea0f4895ef2f5cb83645fa689e52cb55cf493"
uuid = "1317d2d5-d96f-522e-a858-c73665f53c3e"
version = "2021.12.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.7.0+0"
"""

# ╔═╡ Cell order:
# ╠═2abf21a4-6204-486a-9325-bd07e3ad985f
# ╟─c803646a-f37b-4316-898a-af80fd7b4d11
# ╟─d52bed16-b3cf-4d1c-8ade-4809f4da273c
# ╟─8fbf3f1e-9b93-4771-8224-be833bc0ec22
# ╟─4143f162-fe52-4bc2-8083-fa91cfa114c9
# ╟─13d56b33-19d5-4d27-9356-587d8c947e4f
# ╟─d3c727a3-e7f4-45ce-84b1-8ecb2067a5a7
# ╠═c73d160c-2094-4e7b-9fce-46680395c190
# ╠═06393592-7a05-4ed4-9a3c-77219f2894fb
# ╠═9e0d2ba3-d700-4776-afcb-85cb21622b42
# ╠═bb1519d5-9495-4626-a28d-146bfa48e015
# ╠═ddb89a12-4a7f-49a6-ae76-1264f3698958
# ╠═2a544401-b722-45de-b202-84e74401171f
# ╠═bd66f433-c586-4646-9814-8882d59a4db5
# ╠═ecbe9fda-a40a-4800-8c44-9ae67d88e3f4
# ╠═cb834d65-5476-4d38-982e-f811edef120c
# ╟─471ba93f-fa75-487c-b5b6-485b2d317f66
# ╠═aa535f84-4338-46fe-8ef8-3d1d91e1fa22
# ╠═e6ef86b4-e67d-4553-9cc1-9c9a3060e186
# ╠═b10e40aa-bd1e-41f1-a73c-950da681e584
# ╠═1bc08a78-3f5d-43dc-bf5c-cbb33ad18cf3
# ╠═2d17f016-2f66-4d1c-8cd7-4d8bb201e352
# ╠═1cf485d5-1847-4ff7-b3ee-e52c4ef32133
# ╠═39b70471-b546-420c-92b7-0a5159d4bc02
# ╠═5b098aa5-daa5-4817-a822-a954578b5fef
# ╠═b29add71-bf30-4e12-b8e9-3babfd0614b3
# ╟─265d0778-8734-45ae-b37f-ad2a05b74563
# ╟─ed8c354b-3566-45a3-8a71-3dc4ae6fad69
# ╟─077b3192-cf54-4fe5-9643-ec681be0b13d
# ╟─01f102eb-a7cd-4cf8-a9fb-26a01a200fd5
# ╠═929ee3d8-044e-49b3-b4d5-22abf128d2e8
# ╠═1ad3e30a-ffa0-46a6-a93f-e2bbc7136d2c
# ╟─66eee112-45cc-44a0-b24b-ac7b672c187f
# ╠═68686c13-af19-4cea-85ea-34a82a7d3570
# ╟─b36c9a52-0e7b-4a34-a916-faead208c461
# ╠═26c47017-5a7a-4ddf-9644-2e0246cc6f7b
# ╟─81f5c864-8388-46eb-95e1-b0b295d6ebff
# ╠═763a5056-c49e-11ee-0f48-874e88d8c2f4
# ╠═b752fd54-2a58-4fb6-b4e1-e12a72bff7aa
# ╠═f567b59f-025e-4120-ab02-e0e021aa9c6c
# ╟─d3ceb1b0-c2de-44bd-8e56-4bf3d71354b6
# ╠═b8dd220a-faef-4091-ada1-44a9aa016fdd
# ╠═61a30c42-a22b-439a-82bc-7346f4018ab7
# ╠═cea0cdea-2dcf-4e5b-895a-5b26a6bfc6be
# ╠═42176a5b-ec56-4c23-b8b5-41a0875e5a90
# ╠═6491c0d0-7055-4798-8a8e-6b4acb5d037d
# ╠═aadd65c5-361b-460e-8b46-ae27ac57d870
# ╠═eef0fa00-dbdc-4be7-8c65-d624612f5499
# ╟─4daf46e8-ea43-4f2f-90a9-3453ffc55ee3
# ╟─ac830045-7a9d-4f76-9da8-2685b6c62804
# ╠═f2aea8c1-d860-4a85-ae3b-fa43e8d598bf
# ╠═6ff1d8f2-43c1-4232-860e-cdd957882938
# ╠═224ab916-9f0a-4bb4-9a4d-408973056af4
# ╠═f6f8ee9c-2f1f-4e36-932c-a7d610e0cb91
# ╠═8e361e89-b787-4f5e-88c6-eb3f9431de93
# ╠═03d5e2e6-0ca5-4e45-a7f8-37acbdd5a5c8
# ╟─c28befbf-d91c-4e08-a743-d13c06b3aa26
# ╟─7652a4f4-2a9a-4ec9-88cc-b27a4a759b0d
# ╟─e994ed04-1622-4966-adff-fb23cb037fc8
# ╟─23b4f6ba-68f1-406b-a269-c09e4782995f
# ╠═70ca9052-27fa-477d-a3a5-55ec1ebd6116
# ╠═bdba3e79-8784-4dff-925e-c3aeac83c74a
# ╠═a945c1e8-851f-4ce3-b1ed-71fedae1c229
# ╟─32e85306-7fea-4b1b-8f3d-5c644020f0c0
# ╠═0a7354ec-56b2-4954-a924-56dd7e384de9
# ╠═cc4a10c2-27c6-495b-9d08-0ab7bf597ae1
# ╠═cc3dde37-fc3e-4e29-9c1a-8e4184038084
# ╠═25fba4c1-852a-4aa9-a631-1663fba7c437
# ╟─065cef58-9fe9-4431-8d5d-2fe531278b43
# ╠═dbfc6c64-0ef6-472c-b034-4237618317b6
# ╠═8fdaa76a-11e7-4320-926f-3da264920167
# ╠═993a698e-63f7-4bf7-8403-c223658390a4
# ╠═f8c84f16-d1f7-479f-95f8-5808f254fba4
# ╠═c9db7307-9ec8-421d-92d0-8326658e0c16
# ╠═f27ae334-0999-4d9a-9f65-81ffeeed98a3
# ╠═46b24cde-c224-4255-b49e-d76fef9952a2
# ╟─9a9da957-900f-4ee4-9fc5-a8b519610552
# ╠═fb696746-c51a-4331-9320-d5c3c9ccec6c
# ╠═5f184996-b3e8-4ef6-919a-12a3b45128b1
# ╠═bb96996a-a259-48b5-b657-a07cc9409cb5
# ╠═182fe741-7c23-4acc-98b2-b491483c508b
# ╠═5ec8f71d-6f9c-4f19-9ed5-5bc2073d5fd6
# ╠═152c2c96-09e2-4679-95a6-bf062b4662b8
# ╠═dd2a0b9e-096b-4c61-a50d-3ad99fc2bcfa
# ╠═29284a75-8dc4-4f16-8fbb-934c83e0a75b
# ╠═6536fc8d-2ef6-401f-8852-78db831d0bca
# ╠═7447a2d2-e901-48f1-976d-48010c37a8fc
# ╠═d5d13ce3-e2e2-49d8-a033-fef5105e0cb1
# ╠═fd349977-6d95-44ca-90db-b4d092cb0efa
# ╠═c6099f88-9061-437d-9e2d-cd38430ccf3d
# ╠═65a936ea-d82a-48ad-812d-bc3d81a90d72
# ╠═64389d4b-8ec7-45d4-9534-8f32359fba1d
# ╠═0d451ea3-e3d8-4f87-9f8e-9bcb39ccf1b6
# ╠═28e8a487-fb76-49ad-a0d2-ba73d838bcfa
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
