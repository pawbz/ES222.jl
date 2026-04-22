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

# ╔═╡ 6a6dcb6a-0da7-4a6c-8d60-5f8b7a8d9d43
using PlutoPlotly, Flux, PlutoUI, Statistics, Distributions, Random, PlutoHooks

# ╔═╡ 8f0a2c69-0d38-4d10-bf3f-4a4c70f0a86e
TableOfContents()

# ╔═╡ 7c8e375b-0aa2-4d35-8ecb-7ee7c915f870
md"""# Maximizing the Log-Likelihood
We fit a simple regression model and visualize how the conditional density $p(y\mid x)$ sharpens as training progresses. Use the slider to change the number of epochs, and switch between homoscedastic and heteroscedastic likelihoods.
"""

# ╔═╡ 8b580f69-0e10-4f3a-8a9a-52c63a2ed84d
begin
    xmin = -2.0
    xmax = 2.0
    xplot = collect(range(xmin, xmax, length=400))
end

# ╔═╡ c5f3b8c5-0b2b-4d67-9b21-f56c0d4f35f5
md"""## Controls"""

# ╔═╡ b22ee514-6c9f-45b5-9a4a-1f3a8a2852ad
function gui_controls()
    return PlutoUI.combine() do Child
        inputs = [
            md"""Model: $(Child("kind", Select(["Homoscedastic", "Heteroscedastic"])) )
            Epochs: $(Child("epochs", Slider(0:5:300, default=60, show_value=true)))
            Learning rate: $(Child("lr", Slider(1f-4:1f-4:5f-3, default=1f-3, show_value=true)))
            """,
            md"""Train samples: $(Child("ntrain", Slider(20:5:200, default=80, show_value=true)))
            Noise scale: $(Child("noise", Slider(0.05:0.05:0.6, default=0.2, show_value=true)))
            Add outliers: $(Child("outliers", CheckBox(true)))
            """,
            md"""$(Child("resample", CounterButton("Resample data")))
            $(Child("reinit", CounterButton("Reinitialize model")))
            Seed: $(Child("seed", Slider(1:1:100, default=11, show_value=true)))
            """
        ]
        md"""$(inputs)"""
    end
end

# ╔═╡ 62c239ce-7e90-4f4f-9851-08d5dd6bd2b0
controls = @bind params gui_controls()

# ╔═╡ 56b3b5d9-9c0c-4a7e-9dfe-37a89a2c9e5c
controls

# ╔═╡ dcc4dd8a-9b2f-4fc5-8f2f-f69a7fbf8b95
md"""## Data"""

# ╔═╡ 6ab0b2c4-1a49-4be7-bc7f-4dba6e4a36d9
# ╠═╡ show_logs = false
data = @use_memo([params.resample, params.ntrain, params.noise, params.outliers, params.seed]) do
    rng = MersenneTwister(params.seed)
    xtrain = rand(rng, Uniform(xmin, xmax), params.ntrain)

    # Heteroscedastic noise pattern: variance increases with x.
    base_sigma = 0.05
    slope_sigma = params.noise
    sigma_true = base_sigma .+ slope_sigma .* (xtrain .- xmin) ./ (xmax - xmin)

    ytrain = sin.(2 .* xtrain) .+ sigma_true .* randn(rng, params.ntrain)

    if params.outliers
        k = max(1, round(Int, 0.05 * params.ntrain))
        idx = rand(rng, 1:params.ntrain, k)
        ytrain[idx] .+= 4.0 .* randn(rng, k)
    end

    xtrain = reshape(Float32.(xtrain), 1, :)
    ytrain = reshape(Float32.(ytrain), 1, :)

    loader = Flux.DataLoader((xtrain, ytrain), batchsize=min(16, params.ntrain), shuffle=true, rng=rng)
    (; xtrain, ytrain, loader, sigma_true)
end

# ╔═╡ 0b505ef6-2f2e-4f7f-b27c-53252f6042a2
md"""## Models"""

# ╔═╡ 9f3d8e31-9dd5-4772-b6a1-0a93c9a6d30e
begin
    struct HomoModel
        w::Float32
        b::Float32
        log_sigma::Float32
    end

    Flux.@functor HomoModel

    (m::HomoModel)(x) = m.w .* x .+ m.b
    sigma(m::HomoModel) = Flux.softplus(m.log_sigma) + 1f-4

    struct HeteroModel
        w_mu::Float32
        b_mu::Float32
        w_logsigma::Float32
        b_logsigma::Float32
    end

    Flux.@functor HeteroModel

    function (m::HeteroModel)(x)
        mu = m.w_mu .* x .+ m.b_mu
        log_sigma = m.w_logsigma .* x .+ m.b_logsigma
        return mu, log_sigma
    end

    sigma_from_log(log_sigma) = Flux.softplus.(log_sigma) .+ 1f-4
end

# ╔═╡ b4e2f4e4-4f4c-4da9-a47d-2a1226f0d9f5
begin
    function init_model(kind::String, seed::Int)
        rng = MersenneTwister(seed)
        if kind == "Homoscedastic"
            return HomoModel(0.1f0 * randn(rng), 0.0f0, -1.0f0)
        else
            return HeteroModel(0.1f0 * randn(rng), 0.0f0, 0.0f0, -1.0f0)
        end
    end

    function nll(m::HomoModel, x, y)
        mu = m(x)
        s = sigma(m)
        return mean(@. 0.5f0 * log(2f0 * pi) + log(s) + 0.5f0 * ((y - mu) / s)^2)
    end

    function nll(m::HeteroModel, x, y)
        mu, log_sigma = m(x)
        s = sigma_from_log(log_sigma)
        return mean(@. 0.5f0 * log(2f0 * pi) + log(s) + 0.5f0 * ((y - mu) / s)^2)
    end

    function predict_mu_sigma(m::HomoModel, x)
        mu = m(x)
        s = fill(Float32(sigma(m)), size(mu))
        return mu, s
    end

    function predict_mu_sigma(m::HeteroModel, x)
        mu, log_sigma = m(x)
        s = sigma_from_log(log_sigma)
        return mu, s
    end
end

# ╔═╡ 0f3c23f2-1f4d-4c74-9db2-9f584f4e9fd6
base_model = @use_memo([params.kind, params.reinit, params.seed]) do
    init_model(params.kind, params.seed)
end

# ╔═╡ 9e2fba0f-cd2b-4e64-9d25-420925a6b8b3
# ╠═╡ show_logs = false
trained = @use_memo([params.epochs, params.lr, params.kind, params.reinit, params.resample, params.ntrain, params.noise, params.outliers, params.seed]) do
    model = deepcopy(base_model)
    opt = Flux.setup(Flux.Adam(params.lr), model)

    losses = Float32[]
    for epoch in 1:params.epochs
        Flux.train!(model, data.loader, opt) do m, x, y
            nll(m, x, y)
        end
        push!(losses, nll(model, data.xtrain, data.ytrain))
    end

    (; model, losses)
end

# ╔═╡ d8a2b6ce-0d3f-4c6a-bf2f-544c4b1a9f79
begin
    xplot_m = reshape(Float32.(xplot), 1, :)
    mu_pred, sigma_pred = predict_mu_sigma(trained.model, xplot_m)

    ytrue = sin.(2 .* xplot)
    mu_pred_v = vec(mu_pred)
    sigma_pred_v = vec(sigma_pred)
end

# ╔═╡ 20b0efdb-0879-4d88-b01b-9bbf30326b93
md"""## Fit and uncertainty"""

# ╔═╡ 0d0f7770-5bb1-4cc0-87a4-6b70baf02197
begin
    lower = mu_pred_v .- 2 .* sigma_pred_v
    upper = mu_pred_v .+ 2 .* sigma_pred_v

    traces = [
        scatter(x=vec(data.xtrain), y=vec(data.ytrain), mode="markers", name="data", marker=attr(color="rgba(185,113,80,0.45)", size=7)),
        scatter(x=xplot, y=ytrue, mode="lines", name="true mean", line=attr(color="black", width=1, dash="dot")),
        scatter(x=xplot, y=mu_pred_v, mode="lines", name="model mean", line=attr(color="rgba(36,122,115,1)", width=3)),
        scatter(x=xplot, y=lower, mode="lines", line=attr(color="rgba(36,122,115,0.2)"), showlegend=false),
        scatter(x=xplot, y=upper, mode="lines", fill="tonexty", name="+/- 2 sigma", fillcolor="rgba(36,122,115,0.2)", line=attr(color="rgba(36,122,115,0.2)"))
    ]

    plot(traces, Layout(width=650, height=400, title="Fit and predictive uncertainty", xaxis=attr(title="x"), yaxis=attr(title="y")))
end

# ╔═╡ 3c57a7a7-1d35-4b6b-8c96-8a9fcb7f8d6c
md"""## Conditional density slices"""

# ╔═╡ 5a4d9c5a-01c6-4d5b-90a0-77a28913fdb5
begin
    x_slices = [-1.2, 0.0, 1.2]
    density_traces = [
        scatter(x=vec(data.xtrain), y=vec(data.ytrain), mode="markers", name="data", marker=attr(color="rgba(185,113,80,0.25)", size=6))
    ]

    for x0 in x_slices
        x0m = reshape(Float32(x0), 1, 1)
        mu0, s0 = predict_mu_sigma(trained.model, x0m)
        mu0 = Float64(mu0[1])
        s0 = Float64(s0[1])
        ygrid = collect(range(mu0 - 3 * s0, mu0 + 3 * s0, length=200))
        pdf_vals = pdf.(Normal(mu0, s0), ygrid)
        scale = 0.35 / maximum(pdf_vals)
        push!(density_traces, scatter(x=x0 .+ scale .* pdf_vals, y=ygrid, mode="lines", name="p(y|x=$(round(x0, digits=1)))"))
        push!(density_traces, scatter(x=[x0], y=[mu0], mode="markers", marker=attr(color="rgba(36,122,115,1)", size=8), showlegend=false))
    end

    plot(density_traces, Layout(width=650, height=400, title="Slices of p(y|x)", xaxis=attr(title="x"), yaxis=attr(title="y")))
end

# ╔═╡ 0f6b0a4a-ff0c-4ee2-8f6f-8dfc6e61f729
md"""## Log-likelihood cost"""

# ╔═╡ 13cc01f3-4575-4b7d-b4b6-8e439a0c3fd1
begin
    last_loss = isempty(trained.losses) ? nll(trained.model, data.xtrain, data.ytrain) : trained.losses[end]
    md"""Current negative log-likelihood: **$(round(last_loss, digits=4))**"""
end

# ╔═╡ a2e3a6b8-16c6-449b-80bf-6cbfd3f6c931
begin
    loss_traces = [scatter(x=1:length(trained.losses), y=trained.losses, mode="lines+markers", name="NLL")]
    plot(loss_traces, Layout(width=650, height=320, title="NLL vs epochs", xaxis=attr(title="epoch"), yaxis=attr(title="NLL")))
end

# ╔═╡ c90bc7d8-c4b5-4b7f-8d40-07613aa8e1a6
md"""### Likelihood definition
For a Gaussian likelihood, the negative log-likelihood for one sample is

$$
\mathcal{L}(y, x) = \frac{1}{2}\log(2\pi) + \log\sigma(x) + \frac{1}{2}\Big(\frac{y - \mu(x)}{\sigma(x)}\Big)^2
$$

The homoscedastic model uses a constant $\sigma$, while the heteroscedastic model learns $\sigma(x)$.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c"
PlutoHooks = "0ff47ea0-7a50-410d-8455-4348d5de0774"
PlutoPlotly = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[compat]
Distributions = "~0.25.107"
Flux = "~0.14.13"
PlutoHooks = "~0.0.5"
PlutoPlotly = "~0.4.5"
PlutoUI = "~0.7.58"
"""

