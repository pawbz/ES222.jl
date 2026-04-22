using Flux
import Flux._show_children
import Flux._big_show
import Flux.DataLoader
using Printf
import Printf: @printf
using ProgressMeter
using Random
"""
    GaussianDiffusion(V::DataType, βs, data_shape, denoise_fn)

A Gaussian Diffusion Probalistic Model (DDPM) as introduced in "Denoising Diffusion Probabilistic Models" by Ho et. al (https://arxiv.org/abs/2006.11239).
"""
struct GaussianDiffusion{V<:AbstractVector}
    num_timesteps::Int
    data_shape::NTuple
    denoise_fn

    βs::V
    αs::V
    α_cumprods::V
    α_cumprod_prevs::V

    sqrt_α_cumprods::V
    sqrt_one_minus_α_cumprods::V
    sqrt_recip_α_cumprods::V
    sqrt_recip_α_cumprods_minus_one::V
    posterior_variance::V
    posterior_log_variance_clipped::V
    posterior_mean_coef1::V
    posterior_mean_coef2::V
end

Base.eltype(::Type{<:GaussianDiffusion{V}}) where {V} = V

Flux.@functor GaussianDiffusion
Flux.trainable(g::GaussianDiffusion) = (; g.denoise_fn)

function Base.show(io::IO, diffusion::GaussianDiffusion)
    V = typeof(diffusion).parameters[1]
    print(io, "GaussianDiffusion{$V}(")
    print(io, "num_timesteps=$(diffusion.num_timesteps)")
    print(io, ", data_shape=$(diffusion.data_shape)")
    print(io, ", denoise_fn=$(diffusion.denoise_fn)")
    num_buffers = 12
    buffers_size = Base.format_bytes(Base.summarysize(diffusion.βs) * num_buffers)
    print(io, ", buffers_size=$buffers_size")
    print(io, ")")
end

function GaussianDiffusion(V::DataType, βs::AbstractVector, data_shape::NTuple, denoise_fn)
    αs = 1 .- βs
    α_cumprods = cumprod(αs)
    α_cumprod_prevs = [1, (α_cumprods[1:end-1])...]

    sqrt_α_cumprods = sqrt.(α_cumprods)
    sqrt_one_minus_α_cumprods = sqrt.(1 .- α_cumprods)
    sqrt_recip_α_cumprods = 1 ./ sqrt.(α_cumprods)
    sqrt_recip_α_cumprods_minus_one = sqrt.(1 ./ α_cumprods .- 1)

    posterior_variance = βs .* (1 .- α_cumprod_prevs) ./ (1 .- α_cumprods)
    posterior_log_variance_clipped = log.(max.(posterior_variance, 1e-20))

    posterior_mean_coef1 = βs .* sqrt.(α_cumprod_prevs) ./ (1 .- α_cumprods)
    posterior_mean_coef2 = (1 .- α_cumprod_prevs) .* sqrt.(αs) ./ (1 .- α_cumprods)

    GaussianDiffusion{V}(
        length(βs),
        data_shape,
        denoise_fn,
        βs,
        αs,
        α_cumprods,
        α_cumprod_prevs,
        sqrt_α_cumprods,
        sqrt_one_minus_α_cumprods,
        sqrt_recip_α_cumprods,
        sqrt_recip_α_cumprods_minus_one,
        posterior_variance,
        posterior_log_variance_clipped,
        posterior_mean_coef1,
        posterior_mean_coef2
    )
end

"""
    linear_beta_schedule(num_timesteps, β_start=0.0001f0, β_end=0.02f0)
"""
function linear_beta_schedule(num_timesteps::Int, β_start=0.0001f0, β_end=0.02f0)
    scale = convert(typeof(β_start), 1000 / num_timesteps)
    β_start *= scale
    β_end *= scale
    range(β_start, β_end; length=num_timesteps)
end



"""
    cosine_beta_schedule(num_timesteps, s=0.008)

Cosine schedule as proposed in "Improved Denoising Diffusion Probabilistic Models" by Nichol, Dhariwal (https://arxiv.org/abs/2102.09672)
"""
function cosine_beta_schedule(num_timesteps::Int, s=0.008)
    t = range(0, num_timesteps; length=num_timesteps + 1)
    α_cumprods = (cos.((t / num_timesteps .+ s) / (1 + s) * π / 2)) .^ 2
    α_cumprods = α_cumprods / α_cumprods[1]
    βs = 1 .- α_cumprods[2:end] ./ α_cumprods[1:(end-1)]
    clamp!(βs, 0, 0.999)
    βs
end

## extract input[idxs] and reshape for broadcasting across a batch.
function _extract(input, idxs::AbstractVector{Int}, shape::NTuple)
    reshape(input[idxs], (repeat([1], length(shape) - 1)..., :))
end

"""
    q_sample(diffusion, x_start, timesteps, noise)
    q_sample(diffusion, x_start, timesteps; to_device=cpu)

The forward process ``q(x_t | x_0)``. Diffuse the data for a given number of diffusion steps.
"""
function q_sample(
    diffusion::GaussianDiffusion,
    x_start::AbstractArray,
    timesteps::AbstractVector{Int},
    noise::AbstractArray
)
    coeff1 = _extract(diffusion.sqrt_α_cumprods, timesteps, size(x_start))
    coeff2 = _extract(diffusion.sqrt_one_minus_α_cumprods, timesteps, size(x_start))
    coeff1 .* x_start + coeff2 .* noise
end

function q_sample(
    diffusion::GaussianDiffusion,
    x_start::AbstractArray,
    timesteps::AbstractVector{Int}
    ; to_device=cpu
)
    T = eltype(eltype(diffusion))
    noise = randn(T, size(x_start)) |> to_device
    timesteps = timesteps |> to_device
    q_sample(diffusion, x_start, timesteps, noise)
end

function q_sample(
    diffusion::GaussianDiffusion,
    x_start::AbstractArray{T,N},
    timestep::Int; to_device=cpu
) where {T,N}
    timesteps = fill(timestep, size(x_start, N)) |> to_device
    q_sample(diffusion, x_start, timesteps; to_device=to_device)
end

"""
    q_posterior_mean_variance(diffusion, x_start, x_t, timesteps)

Compute the mean and variance for the ``q_{posterior}(x_{t-1} | x_t, x_0) = q(x_t | x_{t-1}, x_0) q(x_{t-1} | x_0) / q(x_t | x_0)``
where `x_0 = x_start`. 
The ``q_{posterior}`` is a Bayesian estimate of the reverse process ``p(x_{t-1} | x_{t})`` where ``x_0`` is known.
"""
function q_posterior_mean_variance(diffusion::GaussianDiffusion, x_start::AbstractArray, x_t::AbstractArray, timesteps::AbstractVector{Int})
    coeff1 = _extract(diffusion.posterior_mean_coef1, timesteps, size(x_t))
    coeff2 = _extract(diffusion.posterior_mean_coef2, timesteps, size(x_t))
    posterior_mean = coeff1 .* x_start + coeff2 .* x_t
    posterior_variance = _extract(diffusion.posterior_variance, timesteps, size(x_t))
    posterior_mean, posterior_variance
end

"""
    predict_start_from_noise(diffusion, x_t, timesteps, noise)

Predict an estimate for the ``x_0`` based on the forward process ``q(x_t | x_0)``.
"""
function predict_start_from_noise(diffusion::GaussianDiffusion, x_t::AbstractArray, timesteps::AbstractVector{Int}, noise::AbstractArray)
    coeff1 = _extract(diffusion.sqrt_recip_α_cumprods, timesteps, size(x_t))
    coeff2 = _extract(diffusion.sqrt_recip_α_cumprods_minus_one, timesteps, size(x_t))
    coeff1 .* x_t - coeff2 .* noise
end

function denoise(diffusion::GaussianDiffusion, x::AbstractArray, timesteps::AbstractVector{Int})
    noise = diffusion.denoise_fn(x, timesteps)
    x_start = predict_start_from_noise(diffusion, x, timesteps, noise)
    x_start, noise
end

"""
    p_sample(diffusion, x, timesteps, noise; 
        clip_denoised=true, add_noise=true)

The reverse process ``p(x_{t-1} | x_t, t)``. Denoise the data by one timestep.
"""
function p_sample(
    diffusion::GaussianDiffusion, x::AbstractArray, timesteps::AbstractVector{Int}, noise::AbstractArray;
    clip_denoised::Bool=true, add_noise::Bool=true
)
    x_start, pred_noise = denoise(diffusion, x, timesteps)
    if clip_denoised
        clamp!(x_start, -1, 1)
    end
    posterior_mean, posterior_variance = q_posterior_mean_variance(diffusion, x_start, x, timesteps)
    x_prev = posterior_mean
    if add_noise
        x_prev += sqrt.(posterior_variance) .* noise
    end
    x_prev, x_start
end

"""
    p_sample_loop(diffusion, shape; clip_denoised=true, to_device=cpu)
    p_sample_loop(diffusion, batch_size; options...)

Generate new samples and denoise it to the first time step.
See `p_sample_loop_all` for a version which returns values for all timesteps.
"""
function p_sample_loop(diffusion::GaussianDiffusion, shape::NTuple; clip_denoised::Bool=true, to_device=cpu)
    T = eltype(eltype(diffusion))
    x = randn(T, shape) |> to_device
    @showprogress "Sampling..." for t in diffusion.num_timesteps:-1:1
        timesteps = fill(t, shape[end]) |> to_device
        noise = randn(T, size(x)) |> to_device
        x, x_start = p_sample(diffusion, x, timesteps, noise; clip_denoised=clip_denoised, add_noise=(t != 1))
    end
    x
end

function p_sample_loop(diffusion::GaussianDiffusion, batch_size::Int; options...)
    p_sample_loop(diffusion, (diffusion.data_shape..., batch_size); options...)
end

"""
    p_sample_loop_all(diffusion, shape; clip_denoised=true, to_device=cpu)
    p_sample_loop_all(diffusion, batch_size; options...)

Generate new samples and denoise them to the first time step. Return all samples where the last dimension is time.
See `p_sample_loop` for a version which returns only the final sample.
"""
function p_sample_loop_all(diffusion::GaussianDiffusion, shape::NTuple; clip_denoised::Bool=true, to_device=cpu)
    T = eltype(eltype(diffusion))
    x = randn(T, shape) |> to_device
    x_all = Array{T}(undef, size(x)..., 0) |> to_device
    x_start_all = Array{T}(undef, size(x)..., 0) |> to_device
    tdim = ndims(x_all)
    @showprogress "Sampling..." for t in diffusion.num_timesteps:-1:1
        timesteps = fill(t, shape[end]) |> to_device
        noise = randn(T, size(x)) |> to_device
        x, x_start = p_sample(diffusion, x, timesteps, noise; clip_denoised=clip_denoised, add_noise=(t != 1))
        x_all = cat(x_all, x, dims=tdim)
        x_start_all = cat(x_start_all, x_start, dims=tdim)
    end
    x_all, x_start_all
end
"""
x is xstart
"""
function p_sample_loop_all_xstart(diffusion::GaussianDiffusion, xstart; clip_denoised::Bool=true, to_device=cpu)
    T = eltype(eltype(diffusion))
    shape = size(xstart)
    x = T.(xstart) |> to_device
    x_all = Array{T}(undef, size(x)..., 0) |> to_device
    x_start_all = Array{T}(undef, size(x)..., 0) |> to_device
    tdim = ndims(x_all)
    @showprogress "Sampling..." for t in diffusion.num_timesteps:-1:1
        timesteps = fill(t, shape[end]) |> to_device
        noise = randn(T, size(x)) |> to_device
        x, x_start = p_sample(diffusion, x, timesteps, noise; clip_denoised=clip_denoised, add_noise=(t != 1))
        x_all = cat(x_all, x, dims=tdim)
        x_start_all = cat(x_start_all, x_start, dims=tdim)
    end
    x_all, x_start_all
end


function p_sample_loop_all(diffusion::GaussianDiffusion, batch_size::Int=16; options...)
    p_sample_loop_all(diffusion, (diffusion.data_shape..., batch_size); options...)
end

"""
    p_losses(diffusion, loss, x_start, timesteps, noise)
    p_losses(diffusion, loss, x_start; to_device=cpu)

Sample from ``q(x_t | x_0)`` and return the loss for the predicted noise.
"""
function p_losses(diffusion::GaussianDiffusion, loss, x_start::AbstractArray, timesteps::AbstractVector{Int}, noise::AbstractArray)
    x = q_sample(diffusion, x_start, timesteps, noise)
    model_out = diffusion.denoise_fn(x, timesteps)
    loss(model_out, noise)
end

function p_losses(diffusion::GaussianDiffusion, loss, x_start::AbstractArray{T,N}; to_device=cpu) where {T,N}
    timesteps = rand(1:diffusion.num_timesteps, size(x_start, N)) |> to_device
    noise = randn(eltype(eltype(diffusion)), size(x_start)) |> to_device
    p_losses(diffusion, loss, x_start, timesteps, noise)
end


abstract type AbstractParallel end

_maybe_forward(layer::AbstractParallel, x::AbstractArray, ys::AbstractArray...) = layer(x, ys...)
_maybe_forward(layer::Parallel, x::AbstractArray, ys::AbstractArray...) = layer(x, ys...)
_maybe_forward(layer, x::AbstractArray, ys::AbstractArray...) = layer(x)

"""
    ConditionalChain(layers...)

Based off `Flux.Chain` except takes in multiple inputs. 
If a layer is of type `AbstractParallel` it uses all inputs else it uses only the first one.
The first input can therefore be conditioned on the other inputs.
"""
struct ConditionalChain{T<:Union{Tuple,NamedTuple}} <: AbstractParallel
    layers::T
end
Flux.@functor ConditionalChain

ConditionalChain(xs...) = ConditionalChain(xs)
function ConditionalChain(; kw...)
    :layers in keys(kw) && throw(ArgumentError("a Chain cannot have a named layer called `layers`"))
    isempty(kw) && return ConditionalChain(())
    ConditionalChain(values(kw))
end

Flux.@forward ConditionalChain.layers Base.getindex, Base.length, Base.first, Base.last,
Base.iterate, Base.lastindex, Base.keys, Base.firstindex

Base.getindex(c::ConditionalChain, i::AbstractArray) = ConditionalChain(c.layers[i]...)

function (c::ConditionalChain)(x, ys...)
    for layer in c.layers
        x = _maybe_forward(layer, x, ys...)
    end
    x
end

function Base.show(io::IO, c::ConditionalChain)
    print(io, "ConditionalChain(")
    Flux._show_layers(io, c.layers)
    print(io, ")")
end

function _big_show(io::IO, m::ConditionalChain{T}, indent::Int=0, name=nothing) where {T<:NamedTuple}
    println(io, " "^indent, isnothing(name) ? "" : "$name = ", "ConditionalChain(")
    for k in Base.keys(m.layers)
        _big_show(io, m.layers[k], indent + 2, k)
    end
    if indent == 0
        print(io, ") ")
        _big_finale(io, m)
    else
        println(io, " "^indent, ")", ",")
    end
end

"""
    ConditionalSkipConnection(layers, connection)

The output is equivalent to `connection(layers(x, ys...), x)`.
Based off Flux.SkipConnection except it passes multiple arguments to layers.
"""
struct ConditionalSkipConnection{T,F} <: AbstractParallel
    layers::T
    connection::F
end

Flux.@functor ConditionalSkipConnection

function (skip::ConditionalSkipConnection)(x, ys...)
    skip.connection(skip.layers(x, ys...), x)
end

function Base.show(io::IO, b::ConditionalSkipConnection)
    print(io, "ConditionalSkipConnection(", b.layers, ", ", b.connection, ")")
end

### Show. Copied from Flux.jl/src/layers/show.jl

for T in [
    :ConditionalChain, ConditionalSkipConnection
]
    @eval function Base.show(io::IO, m::MIME"text/plain", x::$T)
        if get(io, :typeinfo, nothing) === nothing  # e.g. top level in REPL
            Flux._big_show(io, x)
        elseif !get(io, :compact, false)  # e.g. printed inside a Vector, but not a Matrix
            Flux._layer_show(io, x)
        else
            show(io, x)
        end
    end
end

_show_children(c::ConditionalChain) = c.layers


# SinusoidalPositionEmbedding

"""
    SinusoidalPositionEmbedding(dim_embedding::Int, max_length::Int=1000)

A position encoding layer for a matrix of size `dim_embedding`. `max_len` is the maximum acceptable length of input.

For each a pair of rows `(2i, 2i+1)` and a position `k`, the encoding is calculated as:

    W[2i, k] = sin(pos/(1e4^(2i/dim_embedding)))
    W[2i + 1, k] = cos(pos/(1e4^(2i/dim_embedding)))

"""
struct SinusoidalPositionEmbedding{W<:AbstractArray}
    weight::W
end

Flux.@functor SinusoidalPositionEmbedding
Flux.trainable(emb::SinusoidalPositionEmbedding) = (;) # not trainable

function SinusoidalPositionEmbedding(in::Int, out::Int)
    W = make_positional_embedding(out, in)
    SinusoidalPositionEmbedding(W)
end

function make_positional_embedding(dim_embedding::Int, seq_length::Int=1000; n::Int=10000)
    embedding = Matrix{Float32}(undef, dim_embedding, seq_length)
    for pos in 1:seq_length
        for row in 0:2:(dim_embedding-1)
            denom = 1.0 / (n^(row / (dim_embedding - 2)))
            embedding[row+1, pos] = sin(pos * denom)
            embedding[row+2, pos] = cos(pos * denom)
        end
    end
    embedding
end

(m::SinusoidalPositionEmbedding)(x::Integer) = m.weight[:, x]
(m::SinusoidalPositionEmbedding)(x::AbstractVector) = NNlib.gather(m.weight, x)
(m::SinusoidalPositionEmbedding)(x::AbstractArray) = reshape(m(vec(x)), :, size(x)...)

function Base.show(io::IO, m::SinusoidalPositionEmbedding)
    print(io, "SinusoidalPositionEmbedding(", size(m.weight, 2), " => ", size(m.weight, 1), ")")
end



### normalize

function normalize_zero_to_one(x)
    x_min, x_max = extrema(x)
    x_norm = (x .- x_min) ./ (x_max - x_min)
    x_norm
end

function normalize_neg_one_to_one(x)
    2 * normalize_zero_to_one(x) .- 1
end





### Training
function train!(loss, model, data::Flux.DataLoader, opt_state, val_data;
    num_epochs::Int=10,
    save_after_epoch::Bool=false,
    save_dir::String="",
    prob_uncond::Float64=0.0,
)
    history = Dict(
        "epoch_size" => length(data),
        "mean_batch_loss" => Float64[],
        "val_loss" => Float64[],
        "batch_size" => data.batchsize,
    )
    for epoch = 1:num_epochs
        print(stderr, "") # clear stderr for Progress
        progress = Progress(length(data); desc="epoch $epoch/$num_epochs")
        total_loss = 0.0
        for (idx, x) in enumerate(data)
            if (x isa Tuple)
                y = prob_uncond == 0.0 ?
                    x[2] :
                    randomly_set_unconditioned(x[2]; prob_uncond=prob_uncond)
                x_splat = (x[1], y)
            else
                x_splat = (x,)
            end
            batch_loss, grads = Flux.withgradient(model) do m
                loss(m, x_splat...)
            end
            total_loss += batch_loss
            Flux.update!(opt_state, model, grads[1])
            ProgressMeter.next!(
                progress; showvalues=[("batch loss", @sprintf("%.5f", batch_loss))]
            )
        end
        if save_after_epoch
            path = joinpath(save_dir, "model_epoch=$(epoch).bson")
            let model = cpu(model) # keep main model on device
                BSON.bson(path, Dict(:model => model))
            end
        end
        push!(history["mean_batch_loss"], total_loss / length(data))
        @sprintf("mean batch loss: %.5f ; ", history["mean_batch_loss"][end])
        update_history!(model, history, loss, val_data; prob_uncond=prob_uncond)
    end
    history
end

function randomly_set_unconditioned(
    labels::AbstractVector{Int}; prob_uncond::Float64=0.20
)
    # with probability prob_uncond we train without class conditioning
    labels = copy(labels)
    batch_size = length(labels)
    is_not_class_cond = rand(batch_size) .<= prob_uncond
    labels[is_not_class_cond] .= 1
    labels
end

function update_history!(model, history, loss, val_data; prob_uncond::Float64=0.0)
    val_loss = batched_loss(loss, model, val_data; prob_uncond=prob_uncond)
    push!(history["val_loss"], val_loss)
    @printf("val loss: %.5f", history["val_loss"][end])
    println("")
end

function batched_loss(loss, model, data::DataLoader; prob_uncond::Float64=0.0)
    total_loss = 0.0
    for x in data
        if (x isa Tuple)
            y = prob_uncond == 0.0 ?
                x[2] :
                randomly_set_unconditioned(x[2]; prob_uncond=prob_uncond)
            x_splat = (x[1], y)
        else
            x_splat = (x,)
        end
        total_loss += loss(model, x_splat...)
    end
    total_loss /= length(data)
end

"""
    split_validation(rng, data[, labels]; frac=0.1)

Splits `data` and `labels` into two datasets of size `1-frac` and `frac` respectively.

Warning: this function duplicates `data`.
"""
function split_validation(rng::Random.AbstractRNG, data::AbstractArray; frac=0.1)
    nsamples = size(data)[end]
    idxs = randperm(rng, nsamples)
    ntrain = nsamples - floor(Int, frac * nsamples)
    inds_start = ntuple(Returns(:), ndims(data) - 1)
    train_data = data[inds_start..., idxs[1:ntrain]]
    val_data = data[inds_start..., idxs[(ntrain+1):end]]
    train_data, val_data
end

function split_validation(rng::Random.AbstractRNG, data::AbstractArray, labels::AbstractVecOrMat; frac=0.1)
    nsamples = size(data)[end]
    idxs = randperm(rng, nsamples)
    ntrain = nsamples - floor(Int, frac * nsamples)
    inds_start = ntuple(Returns(:), ndims(data) - 1)
    ## train data
    idxs_train = idxs[1:ntrain]
    train_data = data[inds_start..., idxs_train]
    train_labels = ndims(labels) == 2 ? labels[:, idxs_train] : labels[idxs_train]
    ## validation data
    idxs_val = idxs[(ntrain+1):end]
    val_data = data[inds_start..., idxs_val]
    val_labels = ndims(labels) == 2 ? labels[:, idxs_val] : labels[idxs_val]
    (train_data, train_labels), (val_data, val_labels)
end

"""
    batched_metric(g, f, data::DataLoader, g=identity)

Caculates `f(g(x), y)` for each `(x, y)` in data and returns a weighted sum by batch size.
If `f` takes the mean this will recover the full sample mean.
Reduces memory load for `f` and `g`. 
"""
function batched_metric(g, f, data::DataLoader)
    result = 0.0
    num_observations = 0
    for (x, y) in data
        metric = f(g(x), y)
        batch_size = count_observations(x)
        result += metric * batch_size
        num_observations += batch_size
    end
    result / num_observations
end

count_observations(data::D) where {D<:DataLoader} = count_observations(data.data)
count_observations(data::Tuple) = count_observations(data[1])
count_observations(data::AbstractArray{<:Any,N}) where {N} = size(data, N)
count_observations(data) = length(data)

