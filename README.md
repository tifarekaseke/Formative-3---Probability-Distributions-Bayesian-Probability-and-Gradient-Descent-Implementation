# Formative 3 - Probability Distributions, Bayesian Probability, and Gradient Descent

This repository contains the implementation for the Formative 3 assignment.

## Datasets Used

1. Galton Parents and Children Heights Dataset  
   Link: https://www.kaggle.com/datasets/jacopoferretti/parents-heights-vs-children-heights-galton-data

2. IMDb Dataset of 50K Movie Reviews  
   Link: http://kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

## Files

- `formative3_notebook.ipynb` - Full notebook containing Part 1, Part 2, and Part 4 code.
- `manual_gradient_descent.md` - Typed manual calculations for Part 3.
- `manual_gradient_descent.pdf` - Printable version of the manual calculations.
- `data/GaltonFamilies.csv` - Height dataset.
- `data/IMDB Dataset.csv` - Movie reviews dataset.

## Part 1: EM Algorithm

We use the Galton height dataset and compare children heights with father heights. The labels are removed during modeling, so the data is treated as a mixture of two Gaussian distributions. The EM algorithm is implemented from scratch using NumPy.

The notebook prints:

- Initial state and iterations
- Means, variances, mixing coefficients, and log-likelihood
- Posterior probability for a random test height

## Part 2: Bayesian Probability

We calculate `P(Positive | keyword)` for selected sentiment keywords using Bayes' Theorem.

Selected keywords:

- Positive indicators: `great`, `excellent`, `amazing`, `love`
- Negative indicators: `bad`, `boring`, `terrible`, `waste`

The code calculates:

- Prior: P(Positive)
- Likelihood: P(keyword | Positive)
- Marginal: P(keyword)
- Posterior: P(Positive | keyword)

## Part 3 and 4: Gradient Descent

The manual section shows matrix multiplication, prediction, MSE, chain rule gradients, and parameter updates.

The code section uses SciPy for numerical derivative demonstration and implements gradient descent step-by-step. It also creates two Matplotlib plots:

1. How `m` and `b` change over iterations
2. How the error changes over iterations

## Presentation Note

Each student should explain one part of the code and one part of the mathematics. During the presentation, run the notebook live and enter a random test height when asked by the coach.
