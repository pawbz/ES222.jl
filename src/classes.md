---
title: "Lecture Schedule"
tags: ["welcome"]
order: 5
layout: "md.jlmd"
---

<style>
main a img {
    width: 5rem;
    margin: 1rem;
}
</style>

0. This course explores the unifying theme of inverse, learning, and design problems within the context of Earth System Science. These problems are fundamental across various domains, such as geophysics, climate modeling, geology, and environmental science. This course is built on several case studies and encourages a “learn by doing” approach, allowing students to actively engage with real-world problems.

# Discrete Linear Inverse Problems

1. Formulation of discrete linear inverse problem `Gm=d`; row and column interpretations of matrix-vector `Gm` and matrix-matrix `GH` products.
2. Overdetermined and underdetermined systems; linear (in)dependence, basis vectors; independence-dimension inequality.
3. Left (right) inverses of a matrix with independent column (row) vectors; Gram matrix; pseudo inverses; solving linear equations with the inverses.
4. Solving inverse problems via matrix factorization; `CR` column times row and `QR` factorizations.
5. Least-squares; coordinate descent algorithm, lasso, ridge regression, elastic-net.
6. Pseudoinverse via SVD; importance of setting tolerance with noisy data.
7. Gradient, Hessian and Jacobian matrices and their interpretations.
8. Data and model resolution matrices.

## Examples

1. Fitting a polynomial using temperature anomaly data.
2. Chemical analysis of ocean sediments.
3. Analysis of spatial patterns in atmospheric pressure.
4. Formulation of gravity inversion.
5. Demonstration of elastic-net using the gravity problem - assignment.
6. Demonstration of SVD using deconvolution problem.

# Probabilistic Inverse Problems

1. Bayesian inference for inverse problems; mathematical formulation: posterior = likelihood × prior / evidence.
2. Uncertainty quantification.
3. Monte Carlo methods.

# Learning Problems

1. Learning problems; shallow neural networks.
2. Why deep learning; loss functions; maximum likelihood approach for regression and classification problems.
3. Bias-variance tradeoff.
4. K-means clustering; Gaussian mixture models.
5. Application of unsupervised learning and generative AI in Earth sciences: variational autoencoders & diffusion models.

## Examples

1. Seismic event detection and magnitude prediction.
2. Clustering of geological features from remote sensing data.
3. Generating synthetic models of geological structures, based on training data derived from geological surveys.

# Design Problems

1. Design optimization and importance of experiment design in data collection.
2. Spatial and temporal sampling strategies.
3. Bayesian experimental design.

## Examples

1. Design a seismometer network for optimal seismic hazard analysis.
2. Optimizing sampling and proxy selection for reconstructing Holocene climate changes.
