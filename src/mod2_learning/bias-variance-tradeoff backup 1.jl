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

# ╔═╡ 95b078b6-b86d-4f87-8c32-820e4d4e3b68
begin
    using Flux, PlutoUI, PlutoHooks
    using Statistics, Random, Distributions
    using Plots
end

# ╔═╡ 22bc9de8-2d0a-4d07-8f4b-2d8e4e958a1b
TableOfContents()

# ╔═╡ 8cd9d21b-4ad1-44f2-8887-f1076530af6f
md"""
# Bias–Variance Tradeoff with Neural Networks

This notebook is designed for teaching the **bias–variance tradeoff** interactively.

You can control:
- **Number of hidden units** (model complexity)
- **Measurement noise** (data uncertainty)
- **Resampling** of training data

We train an ensemble of neural networks and estimate, for each input $x$:
- **Bias**: $\mathbb{E}[\hat f(x)] - f(x)$
- **Variance**: $\mathbb{V}[\hat f(x)]$

Then we visualize:
- **Gray region** = variance around the ensemble mean
- **Red region** = bias gap between true function and ensemble mean
"""

# ╔═╡ 7b64a59f-5f78-43a7-bf7d-52a8d2dc4494
begin
    xmin = -3.0f0
    xmax = 3.0f0
    xgrid = collect(range(xmin, xmax, length=300))

    true_function(x) = 0.8f0 * sin(1.5f0 * x) + 0.35f0 * cos(2.8f0 * x)
end

# ╔═╡ 5e77d248-14d3-4053-9f33-d7af73f11026
md"""## Controls"""

# ╔═╡ f81a3af8-85b3-47b2-9f9d-7b3de2da1ded
function control_panel()
    PlutoUI.combine() do Child
        md"""
        Hidden units: $(Child("hidden_units", Slider(2:2:80, default=20, show_value=true)))

        Noise std: $(Child("noise_std", Slider(0.00:0.02:0.60, default=0.20, show_value=true)))

        $(Child("resample", CounterButton("Resample training data")))

        Seed: $(Child("seed", Slider(1:1:200, default=11, show_value=true)))
        """
    end
end

# ╔═╡ 8629a31c-8890-42d1-b780-52e7a2ef8de8
controls = @bind params control_panel()

# ╔═╡ 265b5cdf-808f-4c8f-995f-09d8e65295f1
md"""
### Experimental setup

- Training points per model: **50**
- Ensemble size: **30 models**
- Epochs per model: **250**
- Optimizer: **Adam**

Each ensemble member sees a different noisy sample and random initialization.
"""

# ╔═╡ 7567fce0-307b-4fc9-bfd0-d3f5739ea306
begin
    ntrain = 50
    nmodels = 30
    epochs = 250
    lr = 8f-3

    function make_model(hidden_units::Int)
        Flux.f32(Chain(
            Dense(1, hidden_units, tanh),
            Dense(hidden_units, 1)
        ))
    end

    function sample_dataset(rng::AbstractRNG, n::Int, noise_std::Float32)
        x = rand(rng, Uniform(xmin, xmax), n)
        y = true_function.(x) .+ noise_std .* randn(rng, Float32, n)
        return reshape(Float32.(x), 1, :), reshape(Float32.(y), 1, :)
    end

    mse(model, x, y) = mean((model(x) .- y) .^ 2)

    function train_one_model!(rng::AbstractRNG, hidden_units::Int, noise_std::Float32)
        xtrain, ytrain = sample_dataset(rng, ntrain, noise_std)
        model = make_model(hidden_units)
        opt = Flux.setup(Flux.Adam(lr), model)
        loader = Flux.DataLoader((xtrain, ytrain), batchsize=min(16, ntrain), shuffle=true, rng=rng)

        for _ in 1:epochs
            Flux.train!(model, loader, opt) do m, xb, yb
                mse(m, xb, yb)
            end
        end

        return model, xtrain, ytrain
    end
end

# ╔═╡ c3a74bc8-c5ef-4ef8-a4d1-b3c7ac49579d
# ╠═╡ show_logs = false
ensemble_result = @use_memo([params.hidden_units, params.noise_std, params.resample, params.seed]) do
    base_rng = MersenneTwister(params.seed + params.resample)

    xgrid_m = reshape(Float32.(xgrid), 1, :)
    preds = zeros(Float32, nmodels, length(xgrid))

    example_x = nothing
    example_y = nothing

    for i in 1:nmodels
        rng_i = MersenneTwister(rand(base_rng, 1:10^9))
        model_i, x_i, y_i = train_one_model!(rng_i, params.hidden_units, Float32(params.noise_std))
        preds[i, :] .= vec(model_i(xgrid_m))

        if i == 1
            example_x = x_i
            example_y = y_i
        end
    end

    pred_mean = vec(mean(preds, dims=1))
    pred_var = vec(var(preds, dims=1))
    pred_std = sqrt.(pred_var)

    ftrue = true_function.(Float32.(xgrid))
    bias = pred_mean .- ftrue

    # Scalar decomposition estimates on the evaluation grid
    bias2_scalar = mean(bias .^ 2)
    var_scalar = mean(pred_var)
    noise2_scalar = Float32(params.noise_std)^2
    expected_test_mse = bias2_scalar + var_scalar + noise2_scalar

    (; preds, pred_mean, pred_var, pred_std, ftrue, bias,
       bias2_scalar, var_scalar, noise2_scalar, expected_test_mse,
       example_x, example_y)
end

# ╔═╡ 97ecba36-6651-47cc-ad6d-f94e95f3d82e
md"""## Main visualization: bias and variance as shaded regions"""

# ╔═╡ 22c8d38b-f60b-4291-9030-001f6464d7e2
begin
    x = xgrid
    ftrue = ensemble_result.ftrue
    fmean = ensemble_result.pred_mean
    fstd = ensemble_result.pred_std

    # Gray variance band around the ensemble mean
    lower_var = fmean .- fstd
    upper_var = fmean .+ fstd

    # Red bias band between true function and ensemble mean
    lower_bias = min.(ftrue, fmean)
    upper_bias = max.(ftrue, fmean)

    p_main = plot(x, ftrue,
        label="True function f(x)",
        color=:black,
        lw=2,
        xlabel="x",
        ylabel="y",
        title="Bias–Variance Visualization")

    plot!(p_main, x, lower_var,
        fillrange=upper_var,
        fillalpha=0.25,
        color=:gray,
        label="Variance (±1σ around E[ŷ])")

    plot!(p_main, x, lower_bias,
        fillrange=upper_bias,
        fillalpha=0.25,
        color=:red,
        label="Bias gap |E[ŷ]-f|")

    plot!(p_main, x, fmean,
        color=:blue,
        lw=2,
        label="Ensemble mean E[ŷ(x)]")

    scatter!(p_main, vec(ensemble_result.example_x), vec(ensemble_result.example_y),
        color=:orange,
        alpha=0.45,
        ms=3,
        label="One sampled training set")

    p_main
end

# ╔═╡ 5720c37d-3ecd-4d40-bf70-cde9f4b6568d
md"""## Pointwise decomposition"""

# ╔═╡ 92c25b55-39ef-4b8a-8e60-a4ee5f7af808
begin
    p_terms = plot(xgrid, ensemble_result.bias .^ 2,
        label="Bias²(x)",
        color=:red,
        lw=2,
        xlabel="x",
        ylabel="Error contribution",
        title="Pointwise Bias² and Variance")

    plot!(p_terms, xgrid, ensemble_result.pred_var,
        label="Variance(x)",
        color=:gray,
        lw=2)

    hline!(p_terms, [ensemble_result.noise2_scalar],
        label="Noise² (irreducible)",
        color=:black,
        linestyle=:dash,
        lw=2)

    p_terms
end

# ╔═╡ 06f2638b-c71c-4c69-9bf7-1246116de4f9
md"""## Aggregate bias–variance decomposition"""

# ╔═╡ 239fb85b-c2c7-466e-92f5-082f73f2836e
begin
    labels = ["Bias²", "Variance", "Noise²", "Expected test MSE"]
    values = [
        ensemble_result.bias2_scalar,
        ensemble_result.var_scalar,
        ensemble_result.noise2_scalar,
        ensemble_result.expected_test_mse
    ]

    bar(labels, values,
        color=[:red, :gray, :black, :blue],
        alpha=0.75,
        legend=false,
        ylabel="Average contribution",
        title="E[(ŷ-f)²] decomposition")
end

# ╔═╡ b900de6f-e592-4042-a97a-c8c71f7ec646
md"""
## What to try in class

1. **Increase hidden units** with low noise:
   - Bias decreases (red area shrinks)
   - Variance often increases (gray area expands)

2. **Increase noise** while keeping hidden units fixed:
   - Variance grows
   - Irreducible noise floor increases

3. **Press resample repeatedly**:
   - Same settings, different datasets
   - Observe instability for high-complexity models

This is the practical essence of the bias–variance tradeoff.
"""

# ╔═╡ 9fe1f51e-f6ab-4f57-8028-5a43a6c8f16a
md"""
## Notes

- Hidden units control model complexity.
- The **gray band** shows model-to-model spread (variance).
- The **red band** shows systematic offset from the truth (bias).
- With noisy data, even perfect models cannot beat the noise floor.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoHooks = "0ff47ea0-7a50-410d-8455-4348d5de0420"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[compat]
Distributions = "~0.25.117"
Flux = "~0.16.5"
Plots = "~1.40.9"
PlutoHooks = "~0.0.5"
PlutoUI = "~0.7.61"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.4"
manifest_format = "2.0"
project_hash = "c86d3c7d3d6f2b119f8ec43c6ae5b2ab756ef273"
"""

# ╔═╡ Cell order:
# ╠═95b078b6-b86d-4f87-8c32-820e4d4e3b68
# ╠═22bc9de8-2d0a-4d07-8f4b-2d8e4e958a1b
# ╟─8cd9d21b-4ad1-44f2-8887-f1076530af6f
# ╠═7b64a59f-5f78-43a7-bf7d-52a8d2dc4494
# ╟─5e77d248-14d3-4053-9f33-d7af73f11026
# ╠═f81a3af8-85b3-47b2-9f9d-7b3de2da1ded
# ╠═8629a31c-8890-42d1-b780-52e7a2ef8de8
# ╟─265b5cdf-808f-4c8f-995f-09d8e65295f1
# ╠═7567fce0-307b-4fc9-bfd0-d3f5739ea306
# ╠═c3a74bc8-c5ef-4ef8-a4d1-b3c7ac49579d
# ╟─97ecba36-6651-47cc-ad6d-f94e95f3d82e
# ╠═22c8d38b-f60b-4291-9030-001f6464d7e2
# ╟─5720c37d-3ecd-4d40-bf70-cde9f4b6568d
# ╠═92c25b55-39ef-4b8a-8e60-a4ee5f7af808
# ╟─06f2638b-c71c-4c69-9bf7-1246116de4f9
# ╠═239fb85b-c2c7-466e-92f5-082f73f2836e
# ╟─b900de6f-e592-4042-a97a-c8c71f7ec646
# ╟─9fe1f51e-f6ab-4f57-8028-5a43a6c8f16a
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
