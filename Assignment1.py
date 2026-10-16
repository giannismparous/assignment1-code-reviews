"""CSE 242 Assignment 1 — submit as Assignment1.py.

Fill # YOUR ANSWER HERE blocks with commented reasoning; fill function bodies
at # YOUR CODE HERE. Keep signatures. See README.md and INTERFACE.md.
Functions must not mutate inputs, prompt for input, or perform network I/O.
Only code inside the main guard should save/show figures.
"""
import numpy as np


# PART 1 (20 points), Q1: Show computations for the given A and B.
# YOUR ANSWER HERE
# A = [[3, 4], [2, 1]], B = [[1, 2], [0, -1]]
#
# AB = [[3*1 + 4*0, 3*2 + 4*(-1)],
#       [2*1 + 1*0, 2*2 + 1*(-1)]]
#    = [[3, 2], [2, 3]]
#
# BA = [[1*3 + 2*2,    1*4 + 2*1],
#       [0*3 + (-1)*2, 0*4 + (-1)*1]]
#    = [[7, 6], [-2, -1]]
# AB != BA, so matrix multiplication is not commutative.
#
# A + B = [[3+1, 4+2], [2+0, 1+(-1)]] = [[4, 6], [2, 0]]
# (A + B)^T = [[4, 2], [6, 0]]
#
# B^T = [[1, 0], [2, -1]]
# A B^T = [[3*1 + 4*2, 3*0 + 4*(-1)],
#          [2*1 + 1*2, 2*0 + 1*(-1)]]
#       = [[11, -4], [4, -1]]
#
# trace(A) = 3 + 1 = 4
# trace(AB) = 3 + 3 = 6
# Check: trace(BA) = 7 + (-1) = 6 = trace(AB), as Q3 predicts.
def matrix_operations(A=None, B=None):
    """Return AB, BA, (A+B).T, A@B.T, trace(A), trace(AB).

    Defaults: A=[[3,4],[2,1]], B=[[1,2],[0,-1]]. Other inputs are square
    real matrices of matching size. Return a tuple in the specified order.
    """
    # YOUR CODE HERE
    if A is None:
        A = [[3, 4], [2, 1]]
    if B is None:
        B = [[1, 2], [0, -1]]
    # np.array copies, so the caller's inputs are never modified.
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)
    AB = A @ B
    BA = B @ A
    sum_transpose = (A + B).T
    A_Bt = A @ B.T
    return AB, BA, sum_transpose, A_Bt, float(np.trace(A)), float(np.trace(AB))


# P1 Q2: Derive the given M eigenpairs and verify Mv=lambda*v.
# YOUR ANSWER HERE
# M = [[2, -1], [-1, 2]]
#
# Characteristic polynomial:
# det(M - lambda I) = (2 - lambda)^2 - (-1)(-1)
#                   = lambda^2 - 4 lambda + 4 - 1
#                   = lambda^2 - 4 lambda + 3
#                   = (lambda - 1)(lambda - 3)
# Eigenvalues: lambda1 = 1, lambda2 = 3.
#
# lambda1 = 1: M - I = [[1, -1], [-1, 1]]
#   Row 1 gives v1 - v2 = 0, so v1 = v2. Eigenvector v = [1, 1].
# lambda2 = 3: M - 3I = [[-1, -1], [-1, -1]]
#   Row 1 gives -v1 - v2 = 0, so v2 = -v1. Eigenvector v = [1, -1].
#
# Verify Mv = lambda v:
# M [1, 1]  = [2*1 + (-1)*1, (-1)*1 + 2*1]     = [1, 1]  = 1 * [1, 1]
# M [1, -1] = [2*1 + (-1)(-1), (-1)*1 + 2(-1)] = [3, -3] = 3 * [1, -1]
#
# The eigenvectors are orthogonal ([1,1].[1,-1] = 0), as expected for a
# symmetric matrix. The code uses np.linalg.eigh, which is built for
# symmetric matrices. It returns real eigenvalues in ascending order and an
# orthonormal set of eigenvector columns, so a repeated eigenvalue still gets
# a full basis of its eigenspace. eigh returns unit vectors, so it reports
# [1, 1]/sqrt(2) and [1, -1]/sqrt(2) up to sign.
def eigensystem(M=None):
    """Return (values, vectors) for real symmetric M (default [[2,-1],[-1,2]]).

    Vectors are columns paired with values; order, sign, and nonzero scale
    are free. Vectors must span each eigenspace, including repeated roots.
    """
    # YOUR CODE HERE
    if M is None:
        M = [[2, -1], [-1, 2]]
    M = np.array(M, dtype=float)
    values, vectors = np.linalg.eigh(M)
    return values, vectors


# P1 Q3: Prove trace(AB)=trace(BA) for compatible matrices.
# YOUR ANSWER HERE
# The result holds for any A of size m x n and B of size n x m. The matrices
# need not be square. AB is m x m and BA is n x n, so both traces are defined
# even when m != n.
#
# By the definition of the matrix product, (AB)_ii = sum_{k=1..n} A_ik B_ki.
# The trace sums the diagonal entries, so
#   trace(AB) = sum_{i=1..m} sum_{k=1..n} A_ik B_ki
#
# The sums are finite, so their order can be swapped. Scalar multiplication
# commutes, so A_ik B_ki = B_ki A_ik. This gives
#   trace(AB) = sum_{k=1..n} sum_{i=1..m} B_ki A_ik
#
# The inner sum is (BA)_kk = sum_{i=1..m} B_ki A_ik by the same definition.
# Therefore
#   trace(AB) = sum_{k=1..n} (BA)_kk = trace(BA)
#
# Q1 confirms this: trace(AB) = 6 = trace(BA).

# P1 Q4: Show the given C is invertible, and compute its inverse step by step.
# YOUR ANSWER HERE
# C = [[4, 7], [2, 6]]
#
# Invertibility: det(C) = 4*6 - 7*2 = 24 - 14 = 10 != 0. A square matrix is
# invertible if and only if its determinant is nonzero, so C is invertible.
#
# Inverse: for a 2x2 matrix [[a, b], [c, d]] with nonzero determinant, the
# adjugate formula gives
#   [[a, b], [c, d]]^-1 = (1 / (ad - bc)) [[d, -b], [-c, a]]
# Here a = 4, b = 7, c = 2, d = 6, so
#   C^-1 = (1/10) [[6, -7], [-2, 4]] = [[0.6, -0.7], [-0.2, 0.4]]
#
# The code uses np.linalg.inv instead of this formula because the adjugate
# formula above only covers 2x2 matrices and the function must handle n x n.
#
# Verify C C^-1 = I:
#   [[4*0.6 + 7*(-0.2), 4*(-0.7) + 7*0.4],
#    [2*0.6 + 6*(-0.2), 2*(-0.7) + 6*0.4]]
# = [[2.4 - 1.4, -2.8 + 2.8], [1.2 - 1.2, -1.4 + 2.4]] = [[1, 0], [0, 1]]
def matrix_inverse(C=None):
    """Return inverse of nonsingular square C; default C=[[4,7],[2,6]]."""
    # YOUR CODE HERE
    if C is None:
        C = [[4, 7], [2, 6]]
    # Float copy, so the caller's input is never modified.
    C = np.array(C, dtype=float)
    return np.linalg.inv(C)


# PART 2 (50 points), Q1: Normalize c*3**k/k!, derive expectation and variance.
# YOUR ANSWER HERE
# P(X = k) = c 3^k / k! for k = 0, 1, 2, ...
# The derivation below uses a general rate r > 0 and then sets r = 3.
#
# Normalization: the probabilities must sum to 1. The Taylor series of the
# exponential is e^r = sum_{k=0..inf} r^k / k!, so
#   1 = sum_{k=0..inf} c r^k / k! = c e^r
# Therefore c = e^-r. For r = 3, c = e^-3 (about 0.0498).
#
# Expectation: the k = 0 term is zero, and k / k! = 1 / (k-1)! for k >= 1.
#   E[X] = sum_{k=0..inf} k e^-r r^k / k!
#        = e^-r sum_{k=1..inf} r^k / (k-1)!
# Shift the index with j = k - 1:
#        = e^-r r sum_{j=0..inf} r^j / j!
#        = e^-r r e^r = r
# For r = 3, E[X] = 3.
#
# Variance: first compute E[X(X-1)]. The k = 0 and k = 1 terms are zero, and
# k(k-1) / k! = 1 / (k-2)! for k >= 2.
#   E[X(X-1)] = sum_{k=2..inf} e^-r r^k / (k-2)!
# Shift the index with j = k - 2:
#             = e^-r r^2 sum_{j=0..inf} r^j / j!
#             = e^-r r^2 e^r = r^2
# Then E[X^2] = E[X(X-1)] + E[X] = r^2 + r, and by Q6
#   Var(X) = E[X^2] - E[X]^2 = r^2 + r - r^2 = r
# For r = 3, Var(X) = 3.
#
# This is the Poisson distribution with rate 3, and the results agree with its
# known mean and variance.
def distribution_moments(rate=3.):
    """Return (c, mean, variance) for P(X=k)=c*rate**k/k!, k>=0, rate>0."""
    # YOUR CODE HERE
    rate = float(rate)
    c = float(np.exp(-rate))
    mean = rate
    variance = rate
    return c, mean, variance


# P2 Q2: Show reasoning for intersection, conditional, and union probabilities.
# YOUR ANSWER HERE
# Given P(A) = 0.5, P(B) = 0.3, P(A|B) = 0.6.
#
# Intersection: by the definition of conditional probability,
# P(A|B) = P(A n B) / P(B), so
#   P(A n B) = P(A|B) P(B) = 0.6 * 0.3 = 0.18
#
# Conditional: by Bayes' rule,
#   P(B|A) = P(A|B) P(B) / P(A) = 0.18 / 0.5 = 0.36
#
# Union: by inclusion-exclusion, the intersection is counted twice in
# P(A) + P(B), so it is subtracted once.
#   P(A u B) = P(A) + P(B) - P(A n B) = 0.5 + 0.3 - 0.18 = 0.62
def event_probabilities(p_a=.5, p_b=.3, p_a_given_b=.6):
    """Return (P(A intersect B), P(B|A), P(A union B)).

    Inputs describe a valid distribution with p_a,p_b>0.
    """
    # YOUR CODE HERE
    p_a_and_b = p_a_given_b * p_b
    p_b_given_a = p_a_and_b / p_a
    p_a_or_b = p_a + p_b - p_a_and_b
    return float(p_a_and_b), float(p_b_given_a), float(p_a_or_b)


# P2 Q3: Derive MLE/MAP for H,H,T,H,T, including the prior and maximization.
# YOUR ANSWER HERE
# Let theta = P(heads), h = number of heads, n = number of tosses. The data
# H,H,T,H,T gives h = 3 and n = 5. The tosses are independent, so the
# likelihood of the observed sequence is
#   L(theta) = theta^h (1 - theta)^(n-h),   theta in [0, 1]
#
# MLE, interior case 0 < h < n: take the log.
#   log L(theta) = h log(theta) + (n - h) log(1 - theta)
#   d/dtheta log L = h / theta - (n - h) / (1 - theta)
# Setting the derivative to zero gives h (1 - theta) = (n - h) theta, so
#   theta_MLE = h / n
# The second derivative -h / theta^2 - (n - h) / (1 - theta)^2 is negative,
# so log L is concave and this point is the maximum. L is zero at both
# endpoints, so no boundary point beats it.
# For the data, theta_MLE = 3/5 = 0.6.
#
# MLE, boundary cases: the derivative argument does not apply, because one of
# the log terms is absent and the maximum sits on the boundary.
#   h = 0: L(theta) = (1 - theta)^n is strictly decreasing on [0, 1], so the
#          maximum is at theta = 0.
#   h = n: L(theta) = theta^n is strictly increasing on [0, 1], so the
#          maximum is at theta = 1.
# Both equal h / n, so the formula holds for every 0 <= h <= n. n >= 1, so
# there is no division by zero.
#
# MAP: the prior p(theta) = 2 theta is a valid density on [0, 1] because
# the integral of 2 theta from 0 to 1 is 1. By Bayes' rule the posterior is
# proportional to likelihood times prior:
#   p(theta | data) proportional to theta^h (1 - theta)^(n-h) * 2 theta
#                   proportional to theta^(h+1) (1 - theta)^(n-h)
# The prior adds one to the exponent of theta, as if one extra head had been
# observed.
#
# MAP, interior case h < n: take the log.
#   d/dtheta log p = (h + 1) / theta - (n - h) / (1 - theta)
# Setting the derivative to zero gives (h + 1)(1 - theta) = (n - h) theta, so
#   theta_MAP = (h + 1) / (n + 1)
# The log posterior is concave by the same argument as for the MLE, and the
# posterior is zero at both endpoints, so this point is the maximum. This
# includes h = 0, where the posterior theta (1 - theta)^n has its mode at
# 1 / (n + 1) > 0. The prior pulls the estimate off the boundary at 0.
# For the data, theta_MAP = 4/6 = 2/3.
#
# MAP, boundary case h = n: the posterior is theta^(n+1), strictly increasing
# on [0, 1], so the maximum is at theta = 1. The formula gives
# (n + 1) / (n + 1) = 1, so it also holds here. n + 1 >= 2, so there is no
# division by zero.
def coin_estimates(heads=3, tosses=5):
    """Return (MLE, MAP) using prior density 2*theta on [0,1].

    0<=heads<=tosses, positive integer tosses. Include boundary maxima.
    """
    # YOUR CODE HERE
    h = int(heads)
    n = int(tosses)
    # MLE: boundary maxima at h = 0 and h = n, interior critical point otherwise.
    if h == 0:
        mle = 0.0
    elif h == n:
        mle = 1.0
    else:
        mle = h / n
    # MAP: posterior proportional to theta^(h+1) (1 - theta)^(n-h).
    # h = n makes the posterior increasing, so the maximum is at 1.
    if h == n:
        map_estimate = 1.0
    else:
        map_estimate = (h + 1) / (n + 1)
    return float(mle), float(map_estimate)


# P2 Q4: Derive both Gaussian MLEs for [1,3,5,7] step by step.
# YOUR ANSWER HERE
# Samples x_1, ..., x_n are i.i.d. N(mu, sigma^2). The log-likelihood is the
# sum of the log densities:
#   l(mu, sigma^2) = -(n/2) log(2 pi) - (n/2) log(sigma^2)
#                    - (1 / (2 sigma^2)) sum_i (x_i - mu)^2
#
# Partial derivative with respect to mu:
#   dl/dmu = (1 / sigma^2) sum_i (x_i - mu)
# Setting it to zero gives sum_i x_i - n mu = 0, so
#   mu_MLE = (1/n) sum_i x_i
#
# Partial derivative with respect to sigma^2:
#   dl/d(sigma^2) = -n / (2 sigma^2) + (1 / (2 sigma^4)) sum_i (x_i - mu)^2
# Setting it to zero and multiplying by 2 sigma^4 gives
# -n sigma^2 + sum_i (x_i - mu)^2 = 0. Substituting mu = mu_MLE,
#   sigma^2_MLE = (1/n) sum_i (x_i - mu_MLE)^2
# The divisor is n, not n - 1. The MLE is the biased estimator.
#
# For the data [1, 3, 5, 7], n = 4:
#   mu_MLE = (1 + 3 + 5 + 7) / 4 = 16 / 4 = 4
#   deviations: 1-4 = -3, 3-4 = -1, 5-4 = 1, 7-4 = 3
#   squared deviations: 9, 1, 1, 9, with sum 20
#   sigma^2_MLE = 20 / 4 = 5
# The unbiased n - 1 version would give 20 / 3, which is not the MLE.
def gaussian_mle(data=None):
    """Return (mean MLE, variance MLE); default data=[1,3,5,7].

    Other inputs are finite 1D samples of length>=2 with nonzero variance.
    """
    # YOUR CODE HERE
    if data is None:
        data = [1, 3, 5, 7]
    # Float copy, so the caller's input is never modified.
    x = np.array(data, dtype=float)
    mean = np.mean(x)
    # ddof=0 gives the 1/n divisor of the MLE.
    variance = np.var(x, ddof=0)
    return float(mean), float(variance)


# P2 Q5: Derive the Gaussian posterior mode for the given observation/prior.
# YOUR ANSWER HERE
# One observation x ~ N(mu, sigma^2) with known variance sigma^2, and prior
# mu ~ N(mu0, tau^2). Both sigma^2 and tau^2 are variances.
# Given values: x = 5, sigma^2 = 4, mu0 = 0, tau^2 = 1.
#
# By Bayes' rule, p(mu | x) is proportional to p(x | mu) p(mu). Taking logs
# and dropping terms that do not depend on mu,
#   log p(mu | x) = -(x - mu)^2 / (2 sigma^2) - (mu - mu0)^2 / (2 tau^2) + const
# This is a quadratic in mu with negative leading coefficient, so the
# posterior is Gaussian and its mode is the unique critical point.
#
# Derivative with respect to mu:
#   d/dmu log p(mu | x) = (x - mu) / sigma^2 - (mu - mu0) / tau^2
# Setting it to zero gives
#   x / sigma^2 + mu0 / tau^2 = mu (1 / sigma^2 + 1 / tau^2)
#   mu_MAP = (x / sigma^2 + mu0 / tau^2) / (1 / sigma^2 + 1 / tau^2)
# This is the average of x and mu0 weighted by their precisions 1 / sigma^2
# and 1 / tau^2. For a Gaussian the mode equals the mean, so this is also the
# posterior mean.
#
# For the given values:
#   mu_MAP = (5/4 + 0/1) / (1/4 + 1/1) = 1.25 / 1.25 = 1
# The prior has precision 1 and the observation has precision 1/4, so the
# estimate is pulled strongly from x = 5 toward the prior mean 0.
def gaussian_map(x=5., observation_variance=4., prior_mean=0., prior_variance=1.):
    """Return posterior mode for one observation and a Gaussian mean prior.

    Both variances are positive; inputs are variances, not standard deviations.
    """
    # YOUR CODE HERE
    observation_precision = 1.0 / observation_variance
    prior_precision = 1.0 / prior_variance
    numerator = x * observation_precision + prior_mean * prior_precision
    return float(numerator / (observation_precision + prior_precision))


# P2 Q6: Prove Var(X)=E[X**2]-E[X]**2.
# YOUR ANSWER HERE
# Let mu = E[X], a constant. By definition Var(X) = E[(X - mu)^2].
# Expand the square:
#   (X - mu)^2 = X^2 - 2 mu X + mu^2
# By linearity of expectation, and because mu is a constant,
#   E[(X - mu)^2] = E[X^2] - 2 mu E[X] + mu^2
#                 = E[X^2] - 2 mu^2 + mu^2
#                 = E[X^2] - mu^2
# Therefore Var(X) = E[X^2] - E[X]^2.

# PART 3 (30 points), Q1: Define covariance and derive Var(u.T@X)=u.T@Sigma@u.
# YOUR ANSWER HERE
# Let X be a random vector in R^d with mean mu = E[X]. The covariance matrix
# is defined as
#   Sigma = E[(X - mu)(X - mu)^T]
# It is d x d, and its (i, j) entry is Cov(X_i, X_j) = E[(X_i - mu_i)(X_j - mu_j)].
# Sigma is symmetric because (X - mu)(X - mu)^T is symmetric.
#
# Let u be a fixed vector in R^d. The projection u^T X is a scalar random
# variable.
#
# Step 1, its mean: by linearity of expectation, E[u^T X] = u^T E[X] = u^T mu.
#
# Step 2, definition of variance:
#   Var(u^T X) = E[(u^T X - u^T mu)^2] = E[(u^T (X - mu))^2]
#
# Step 3, rewrite the square: u^T (X - mu) is a scalar, so it equals its own
# transpose (X - mu)^T u. Hence
#   (u^T (X - mu))^2 = u^T (X - mu) (X - mu)^T u
#
# Step 4, pull out the constants: u is not random, so by linearity of
# expectation
#   E[u^T (X - mu)(X - mu)^T u] = u^T E[(X - mu)(X - mu)^T] u = u^T Sigma u
#
# Therefore Var(u^T X) = u^T Sigma u.
# A variance is never negative, so u^T Sigma u >= 0 for every u. This means
# Sigma is positive semidefinite, and all its eigenvalues are >= 0.

# P3 Q2: Describe the Gaussian generator and observed correlation.
# YOUR ANSWER HERE
# generate_data draws n_samples i.i.d. points from the bivariate Gaussian
# N(mean, covariance) using np.random.default_rng(seed).multivariate_normal.
# A fixed seed gives the same samples on every call, so results are
# reproducible. The defaults are mean = [2, -1] and
# covariance = [[3, 1.2], [1.2, 1]].
#
# The true correlation between the two coordinates is
#   rho = Cov(X_1, X_2) / sqrt(Var(X_1) Var(X_2)) = 1.2 / sqrt(3 * 1) = 1.2 / sqrt(3)
#       = about 0.69
# This is a clear positive correlation. When X_1 is above its mean, X_2 tends
# to be above its mean too, so the scatter plot forms an ellipse tilted
# upward. X_1 has variance 3 and X_2 has variance 1, so the cloud is also
# wider along the horizontal axis than the vertical one.
#
# With the defaults (500 samples, seed 0), the sample mean is about
# [2.15, -0.93] and the sample correlation is about 0.70. Both are close to
# the true values, as expected for 500 samples.
def generate_data(n_samples=500, seed=0, mean=None, covariance=None):
    """Return reproducible Gaussian samples (n_samples,2) using NumPy and seed.

    Defaults: mean=[2,-1], covariance=[[3,1.2],[1.2,1]]. Custom mean has shape
    (2,), covariance (2,2) is positive definite. Do not mutate input arrays.
    """
    # YOUR CODE HERE
    if mean is None:
        mean = [2., -1.]
    if covariance is None:
        covariance = [[3., 1.2], [1.2, 1.]]
    # Float copies, so the caller's inputs are never modified.
    mean = np.array(mean, dtype=float)
    covariance = np.array(covariance, dtype=float)
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(mean, covariance, size=int(n_samples))


# P3 Q3: Link centering/covariance/eigendecomposition to the derivation and explain the plot.
# YOUR ANSWER HERE
# PCA looks for the unit direction u along which the projected data u^T X has
# the largest variance. By Q1, that variance is u^T Sigma u, so the first
# principal component is
#   u_1 = argmax_u u^T Sigma u   subject to   ||u|| = 1
# Without the constraint the objective could be made arbitrarily large by
# scaling u, so the constraint ||u||^2 = u^T u = 1 is needed.
#
# Lagrangian: L(u, lambda) = u^T Sigma u - lambda (u^T u - 1).
# Sigma is symmetric, so the gradient with respect to u is 2 Sigma u - 2 lambda u.
# Setting it to zero gives
#   Sigma u = lambda u
# So every critical point is an eigenvector of Sigma. At such a point the
# objective equals u^T Sigma u = lambda u^T u = lambda. The maximum is
# therefore the largest eigenvalue, reached at its eigenvector. The second
# component maximizes the same objective among unit vectors orthogonal to
# u_1, which gives the eigenvector of the second largest eigenvalue. Each
# eigenvalue is the variance of the data along its component.
#
# How each step in pca() maps to this derivation:
#   (i) Centering: subtracting the sample mean replaces X with X - mu, the
#       quantity inside Sigma = E[(X - mu)(X - mu)^T].
#   (ii) Covariance: Sigma is estimated by centered^T centered / (n - 1). This
#        is the sample version of the expectation, with the unbiased n - 1
#        divisor.
#   (iii) Eigendecomposition: np.linalg.eigh solves Sigma u = lambda u, the
#         Lagrangian condition above. Sigma is symmetric, so eigh returns real
#         eigenvalues and orthonormal eigenvectors. This holds even when two
#         eigenvalues are tied (any orthonormal basis of the shared eigenspace
#         is returned) or when Sigma is rank-deficient (an eigenvalue is 0).
#   (iv) Ordering: eigh returns eigenvalues in ascending order. They are
#        reversed so that the first component has the largest variance.
#   (v) Clipping: by Q1 the eigenvalues are >= 0. Floating-point error can
#       produce tiny negative values in the rank-deficient case, so these are
#       clipped to 0.
#   (vi) Scores: centered @ components gives u_j^T (x_i - mu) for each point i
#        and component j, the coordinates of the data in the new basis. The
#        sample variance of column j of the scores equals eigenvalue j.
#
# For the default data, the eigenvalues are about 3.46 and 0.41. The first
# component points at about 25 degrees above the horizontal axis (up to
# sign), along the long axis of the tilted ellipse. The true covariance gives
# eigenvalues 3.56 and 0.44 and the same angle, so the sample estimate is
# close. In the scatter plot, the first component arrow runs along the
# direction of greatest spread, and the second is perpendicular to it along
# the short axis of the ellipse. Arrow lengths are scaled by the standard
# deviation sqrt(eigenvalue) along each component.
def pca(X):
    """Fit NumPy PCA without mutating finite X of shape (n,2), n>=2.

    Return dict: mean (2,), centered (n,2), covariance (2,2), eigenvalues (2,),
    components (2,2), scores (n,2). Use covariance divisor n-1, descending
    eigenvalues, orthonormal eigenvectors as columns, scores=centered@components.
    Signs/tied-eigenspace bases are free. Include rank-deficient and tied cases.
    NumPy eigh/SVD are allowed; fitted library PCA is not.
    """
    # YOUR CODE HERE
    # Float copy, so the caller's input is never modified.
    X = np.array(X, dtype=float)
    n = X.shape[0]
    # (i) Centering.
    mean = X.mean(axis=0)
    centered = X - mean
    # (ii) Sample covariance with divisor n - 1.
    covariance = centered.T @ centered / (n - 1)
    # (iii) Eigendecomposition of the symmetric covariance.
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    # (iv) eigh is ascending, so reverse to descending.
    eigenvalues = eigenvalues[::-1]
    components = eigenvectors[:, ::-1]
    # (v) Remove tiny negative values caused by floating-point error.
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    # (vi) Project the centered data onto the components.
    scores = centered @ components
    return {
        'mean': mean,
        'centered': centered,
        'covariance': covariance,
        'eigenvalues': eigenvalues,
        'components': components,
        'scores': scores,
    }


# P3 Q4: Interpret the variance fractions and dimensionality reduction tradeoff.
# YOUR ANSWER HERE
# The fraction of variance explained by component j is
#   lambda_j / (lambda_1 + ... + lambda_d)
# Each eigenvalue is the variance along its component (Q3), and the
# eigenvalues sum to trace(Sigma), the total variance of the data. So the
# fractions are nonnegative and sum to 1.
#
# For the true covariance [[3, 1.2], [1.2, 1]], the eigenvalues are
# 2 +/- sqrt(2.44), about 3.56 and 0.44, so PC1 explains about 89.1% of the
# variance and PC2 about 10.9%. For the default sample (500 points, seed 0),
# the fractions are about 89.4% and 10.6%.
#
# Projecting each point onto PC1 alone reduces the data from 2D to 1D and
# keeps about 89% of the variance. The tradeoff:
#   (i) Benefits: the data is stored with half as many numbers (compression),
#       and later computations run on a smaller representation. If the small
#       variance along PC2 is mostly noise, discarding it also denoises the
#       data.
#   (ii) Cost: the roughly 11% of variance along PC2 is lost. Points that
#        differ only along PC2 become identical after projection, and the
#        reconstruction error is exactly the discarded variance. If that
#        direction carries real signal, for example the feature that separates
#        two classes, the 1D projection removes it.
# PCA keeps the directions of largest variance, which is not always the same
# as the directions most useful for a given task.
def explained_variance(eigenvalues):
    """Return same-shape fractions for nonnegative eigenvalues with positive sum."""
    # YOUR CODE HERE
    # Float copy, so the caller's input is never modified.
    eigenvalues = np.array(eigenvalues, dtype=float)
    return eigenvalues / eigenvalues.sum()


# P3 Q2/Q3/Q4: Provide labeled figures and discuss their meaning.
# YOUR ANSWER HERE
# make_plots returns one figure with two panels.
#
# Left panel (Q2 and Q3): a scatter plot of the samples with the two
# principal directions drawn as arrows from the sample mean. Each arrow has
# length sqrt(eigenvalue), the standard deviation of the data along that
# component. The cloud is an ellipse tilted upward, which shows the positive
# correlation from Q2. The PC1 arrow runs along the long axis of the ellipse,
# about 25 degrees above horizontal, and is about 1.86 long. The PC2 arrow is
# perpendicular to it along the short axis and is about 0.64 long. The axes
# use equal scaling, so the arrows appear perpendicular as they are.
#
# Right panel (Q4): a bar chart of the explained variance fraction of each
# component, with the value printed on each bar. PC1 accounts for about 89%
# and PC2 for about 11%, which shows that a 1D projection onto PC1 keeps most
# of the variance.
def make_plots(X, result):
    """Return a Matplotlib Figure (or sequence of Figures), without show/save.

    Plot samples and principal directions through their mean, and explained
    variance fractions. Label axes. Plot quality is manually reviewed.
    """
    # YOUR CODE HERE
    import matplotlib.pyplot as plt

    X = np.array(X, dtype=float)
    mean = np.array(result['mean'], dtype=float)
    eigenvalues = np.array(result['eigenvalues'], dtype=float)
    components = np.array(result['components'], dtype=float)
    fractions = explained_variance(eigenvalues)

    fig, (ax_scatter, ax_bar) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: samples and principal directions from the mean.
    ax_scatter.scatter(X[:, 0], X[:, 1], s=10, alpha=0.4, color='tab:gray',
                       label='Samples')
    ax_scatter.scatter(mean[0], mean[1], s=60, color='black', marker='x',
                       zorder=3, label='Sample mean')
    colors = ['tab:red', 'tab:blue']
    for j in range(components.shape[1]):
        # Eigenvector signs are arbitrary. For readability, flip each one so
        # its largest-magnitude entry is positive. This changes only the
        # drawing, not the result.
        u = components[:, j]
        if u[np.argmax(np.abs(u))] < 0:
            u = -u
        direction = u * np.sqrt(eigenvalues[j])
        ax_scatter.quiver(mean[0], mean[1], direction[0], direction[1],
                          angles='xy', scale_units='xy', scale=1,
                          color=colors[j % len(colors)], width=0.008, zorder=4,
                          label=f'PC{j + 1} (length sqrt(eigenvalue) = '
                                f'{np.sqrt(eigenvalues[j]):.2f})')
    ax_scatter.set_aspect('equal', adjustable='datalim')
    ax_scatter.set_xlabel('$x_1$')
    ax_scatter.set_ylabel('$x_2$')
    ax_scatter.set_title('Samples and principal directions')
    ax_scatter.legend(loc='lower right', fontsize=9)

    # Right: explained variance fractions.
    labels = [f'PC{j + 1}' for j in range(len(fractions))]
    bars = ax_bar.bar(labels, fractions, color=colors[:len(fractions)],
                      label='Fraction of total variance')
    for bar, fraction in zip(bars, fractions):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, fraction + 0.01,
                    f'{fraction:.1%}', ha='center', va='bottom')
    ax_bar.set_ylim(0, 1.05)
    ax_bar.set_xlabel('Principal component')
    ax_bar.set_ylabel('Explained variance fraction')
    ax_bar.set_title('Explained variance by component')
    ax_bar.legend(loc='upper right')

    fig.tight_layout()
    return fig


# REFLECTION: contributions, tasks completed, and external resources used.
# YOUR ANSWER HERE
# Name: Ioannis Barous
#
# Tasks completed: all parts. Part 1 (matrix operations, eigensystem, trace
# proof, matrix inverse), Part 2 (normalizing the distribution and its
# moments, event probabilities, coin MLE and MAP, Gaussian MLE and MAP,
# variance identity) and Part 3 (covariance projection derivation, data
# generation, PCA, explained variance, plots).
#
# Contributions: I worked on this assignment individually.
#
# External resources: I used Claude (Anthropic's AI assistant) to help draft
# the written derivations and the code. I specified the approach for each
# question, checked the derivations, and tested every function on the
# default inputs and on other inputs, including boundary, tied and
# rank-deficient cases. I also used the NumPy and Matplotlib documentation.

if __name__ == '__main__':
    # YOUR CODE HERE: call your functions and save/display the requested figures.
    # This block does not run when the grader imports your functions.
    np.set_printoptions(precision=4, suppress=True)

    print('Part 1')
    AB, BA, sum_transpose, A_Bt, trace_A, trace_AB = matrix_operations()
    print('AB =\n', AB)
    print('BA =\n', BA)
    print('(A + B)^T =\n', sum_transpose)
    print('A B^T =\n', A_Bt)
    print('trace(A) =', trace_A, ' trace(AB) =', trace_AB)
    values, vectors = eigensystem()
    print('eigenvalues of M =', values)
    print('eigenvectors of M (columns) =\n', vectors)
    print('C^-1 =\n', matrix_inverse())

    print('\nPart 2')
    c, mean, variance = distribution_moments()
    print(f'c = {c:.6f}, E[X] = {mean}, Var(X) = {variance}')
    p_and, p_b_given_a, p_or = event_probabilities()
    print(f'P(A and B) = {p_and:.4f}, P(B|A) = {p_b_given_a:.4f}, '
          f'P(A or B) = {p_or:.4f}')
    mle, map_estimate = coin_estimates()
    print(f'coin MLE = {mle:.4f}, coin MAP = {map_estimate:.4f}')
    mu_mle, var_mle = gaussian_mle()
    print(f'Gaussian MLE mean = {mu_mle}, variance = {var_mle}')
    print(f'Gaussian MAP = {gaussian_map()}')

    print('\nPart 3')
    X = generate_data()
    result = pca(X)
    print('sample mean =', result['mean'])
    print('sample covariance =\n', result['covariance'])
    print('sample correlation =', np.corrcoef(X, rowvar=False)[0, 1])
    print('eigenvalues =', result['eigenvalues'])
    print('components (columns) =\n', result['components'])
    print('explained variance fractions =', explained_variance(result['eigenvalues']))

    fig = make_plots(X, result)
    fig.savefig('pca.png', dpi=150)
    print('\nSaved figure to pca.png')
