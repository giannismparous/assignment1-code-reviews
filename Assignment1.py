"""CSE 242 Assignment 1 — submit as Assignment1.py.

Fill # YOUR ANSWER HERE blocks with commented reasoning; fill function bodies
at # YOUR CODE HERE. Keep signatures. See README.md and INTERFACE.md.
Functions must not mutate inputs, prompt for input, or perform network I/O.
Only code inside the main guard should save/show figures.
"""
import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D


# PART 1 (20 points), Q1: Show computations for the given A and B.
# Given A = [[3, 4],      B = [[1,  2],
#            [2, 1]]           [0, -1]]
#
# Rule: (XY)_ij = sum_k X_ik * Y_kj  (row i of X dotted with column j of Y).
#
# AB:
#   (AB)_11 = 3*1 + 4*0    = 3 + 0  = 3
#   (AB)_12 = 3*2 + 4*(-1) = 6 - 4  = 2
#   (AB)_21 = 2*1 + 1*0    = 2 + 0  = 2
#   (AB)_22 = 2*2 + 1*(-1) = 4 - 1  = 3
#   => AB = [[3, 2],
#            [2, 3]]
#
# BA:
#   (BA)_11 = 1*3 + 2*2    = 3 + 4  = 7
#   (BA)_12 = 1*4 + 2*1    = 4 + 2  = 6
#   (BA)_21 = 0*3 + (-1)*2 = 0 - 2  = -2
#   (BA)_22 = 0*4 + (-1)*1 = 0 - 1  = -1
#   => BA = [[ 7,  6],
#            [-2, -1]]
#   Note AB != BA: matrix multiplication is not commutative.
#
# (A+B)^T:
#   A + B = [[3+1, 4+2], [2+0, 1+(-1)]] = [[4, 6], [2, 0]]
#   Transpose swaps rows and columns, (X^T)_ij = X_ji:
#   => (A+B)^T = [[4, 2],
#                 [6, 0]]
#
# AB^T:
#   B^T = [[1,  0],
#          [2, -1]]
#   (AB^T)_11 = 3*1 + 4*2    = 3 + 8  = 11
#   (AB^T)_12 = 3*0 + 4*(-1) = 0 - 4  = -4
#   (AB^T)_21 = 2*1 + 1*2    = 2 + 2  = 4
#   (AB^T)_22 = 2*0 + 1*(-1) = 0 - 1  = -1
#   => AB^T = [[11, -4],
#              [ 4, -1]]
#
# Traces (sum of diagonal entries):
#   trace(A)  = 3 + 1 = 4
#   trace(AB) = 3 + 3 = 6
#   Check: trace(BA) = 7 + (-1) = 6 = trace(AB), as Q3 proves in general.
def matrix_operations(A=None, B=None):
    """Return AB, BA, (A+B).T, A@B.T, trace(A), trace(AB).

    Defaults: A=[[3,4],[2,1]], B=[[1,2],[0,-1]]. Other inputs are square
    real matrices of matching size. Return a tuple in the specified order.
    """
    # Step 1: fill in the defaults from the problem statement.
    if A is None:
        A = [[3, 4], [2, 1]]
    if B is None:
        B = [[1, 2], [0, -1]]

    # Step 2: copy into new float arrays so the caller's inputs are never mutated.
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)

    # Step 3: check that both inputs are square and of the same size,
    # because AB, BA and A+B are all needed.
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square 2D matrix")
    if B.shape != A.shape:
        raise ValueError("B must be square with the same shape as A")

    # Step 4: compute each requested quantity (same order as the derivation above).
    AB = A @ B
    BA = B @ A
    sum_transpose = (A + B).T
    AB_transpose = A @ B.T
    trace_A = float(np.trace(A))
    trace_AB = float(np.trace(AB))

    return AB, BA, sum_transpose, AB_transpose, trace_A, trace_AB


# P1 Q2: Derive the given M eigenpairs and verify Mv=lambda*v.
# Given M = [[ 2, -1],
#            [-1,  2]]
#
# Step 1: characteristic polynomial, det(M - lambda*I) = 0.
#   M - lambda*I = [[2-lambda,   -1    ],
#                   [  -1,     2-lambda]]
#   det = (2-lambda)(2-lambda) - (-1)(-1)
#       = (2-lambda)^2 - 1
#       = lambda^2 - 4*lambda + 4 - 1
#       = lambda^2 - 4*lambda + 3
#
# Step 2: solve lambda^2 - 4*lambda + 3 = 0.
#   (lambda - 1)(lambda - 3) = 0  =>  lambda_1 = 1, lambda_2 = 3.
#   Check: sum = 1 + 3 = 4 = trace(M); product = 1*3 = 3 = det(M) = 4 - 1.
#
# Step 3: eigenvectors, solve (M - lambda*I) v = 0.
#   lambda_1 = 1:  M - I = [[ 1, -1],
#                           [-1,  1]]
#     Row 1: v1 - v2 = 0  =>  v1 = v2  =>  v = [1, 1]  (unit: [1, 1]/sqrt(2)).
#   lambda_2 = 3:  M - 3I = [[-1, -1],
#                            [-1, -1]]
#     Row 1: -v1 - v2 = 0  =>  v2 = -v1  =>  v = [1, -1]  (unit: [1, -1]/sqrt(2)).
#   The two eigenvectors are orthogonal ([1,1].[1,-1] = 1 - 1 = 0), which is
#   expected because M is symmetric.
#
# Step 4: verify M v = lambda v.
#   M [1, 1]  = [2*1 + (-1)*1, (-1)*1 + 2*1]    = [1, 1]   = 1 * [1, 1]   OK
#   M [1, -1] = [2*1 + (-1)(-1), (-1)*1 + 2(-1)] = [3, -3] = 3 * [1, -1]  OK
#   With unit vectors: M [0.7071, 0.7071] = [0.7071, 0.7071] = 1 * v1, and
#   M [0.7071, -0.7071] = [2.1213, -2.1213] = 3 * v2.
def eigensystem(M=None):
    """Return (values, vectors) for real symmetric M (default [[2,-1],[-1,2]]).

    Vectors are columns paired with values; order, sign, and nonzero scale
    are free. Vectors must span each eigenspace, including repeated roots.
    """
    # Step 1: fill in the default matrix from the problem statement.
    if M is None:
        M = [[2, -1], [-1, 2]]

    # Step 2: copy into a new float array so the caller's input is never mutated.
    M = np.array(M, dtype=float)

    # Step 3: check the input. np.linalg.eigh reads only one triangle of M, so
    # a non-symmetric input would silently give wrong answers; reject it instead.
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError("M must be a square 2D matrix")
    if not np.allclose(M, M.T):
        raise ValueError("M must be symmetric")

    # Step 4: eigh solves det(M - lambda*I) = 0 for a symmetric matrix. It
    # returns real eigenvalues in ascending order and orthonormal eigenvectors
    # as the columns of `vectors` (column i pairs with values[i]). Because the
    # columns are orthonormal, a repeated eigenvalue still gets as many
    # independent vectors as its multiplicity, so each eigenspace is spanned
    # (e.g. M = I gives the full basis [[1, 0], [0, 1]]).
    values, vectors = np.linalg.eigh(M)

    return values, vectors


# P1 Q3: Prove trace(AB)=trace(BA) for compatible matrices.
# Let A be m x n and B be n x m, so AB is m x m and BA is n x n (both square,
# so both traces are defined).
#
# Step 1: write the diagonal entries using the product rule.
#   (AB)_ii = sum_{k=1..n} A_ik * B_ki          for i = 1..m
#   (BA)_kk = sum_{i=1..m} B_ki * A_ik          for k = 1..n
#
# Step 2: the trace is the sum of the diagonal entries.
#   trace(AB) = sum_{i=1..m} (AB)_ii = sum_{i=1..m} sum_{k=1..n} A_ik * B_ki
#
# Step 3: the double sum is finite, so the order of summation can be swapped,
# and scalar multiplication commutes (A_ik * B_ki = B_ki * A_ik):
#   trace(AB) = sum_{k=1..n} sum_{i=1..m} B_ki * A_ik
#
# Step 4: the inner sum is exactly (BA)_kk from Step 1, so
#   trace(AB) = sum_{k=1..n} (BA)_kk = trace(BA).   QED
#
# Example from Q1: trace(AB) = 3 + 3 = 6 and trace(BA) = 7 + (-1) = 6.
# Note: AB and BA can even have different sizes (m x m vs n x n), but their
# traces are always equal.

# P1 Q4: Show the given C is invertible, and compute its inverse step by step.
# Given C = [[4, 7],
#            [2, 6]]
#
# Step 1: a square matrix is invertible if and only if det(C) != 0.
#   det(C) = 4*6 - 7*2 = 24 - 14 = 10  != 0  =>  C is invertible.
#
# Step 2: for a 2x2 matrix [[a, b], [c, d]] with ad - bc != 0,
#   inverse = (1 / (ad - bc)) * adj(C),  where adj(C) = [[d, -b], [-c, a]]
#   (swap the diagonal entries, negate the off-diagonal entries).
#   Here a=4, b=7, c=2, d=6:
#   adj(C) = [[ 6, -7],
#             [-2,  4]]
#   C^-1 = (1/10) * [[ 6, -7],  = [[ 0.6, -0.7],
#                    [-2,  4]]     [-0.2,  0.4]]
#
# Step 3: verify C * C^-1 = I.
#   (1,1): 4*0.6 + 7*(-0.2)  = 2.4 - 1.4 = 1
#   (1,2): 4*(-0.7) + 7*0.4  = -2.8 + 2.8 = 0
#   (2,1): 2*0.6 + 6*(-0.2)  = 1.2 - 1.2 = 0
#   (2,2): 2*(-0.7) + 6*0.4  = -1.4 + 2.4 = 1
#   => C * C^-1 = [[1, 0], [0, 1]] = I.  OK
def matrix_inverse(C=None):
    """Return inverse of nonsingular square C; default C=[[4,7],[2,6]]."""
    # Step 1: fill in the default matrix from the problem statement.
    if C is None:
        C = [[4, 7], [2, 6]]

    # Step 2: copy into a new float array so the caller's input is never mutated.
    C = np.array(C, dtype=float)

    # Step 3: the inverse is only defined for square matrices.
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a square 2D matrix")

    # Step 4: compute the inverse. The 2x2 adjugate formula above does not
    # generalize to any size, so use np.linalg.inv, which works for any n x n
    # matrix and raises np.linalg.LinAlgError if C is singular (det(C) = 0).
    return np.linalg.inv(C)


# PART 2 (50 points), Q1: Normalize c*3**k/k!, derive expectation and variance.
# P(X=k) = c * r^k / k!,  k = 0, 1, 2, ...   (here r = rate = 3).
#
# (a) Normalization constant c.
#   Probabilities must sum to 1:  sum_{k>=0} c * r^k / k! = 1.
#   Exponential series: sum_{k>=0} r^k / k! = e^r.
#   => c * e^r = 1  =>  c = e^(-r).
#   For r = 3:  c = e^(-3) ~= 0.049787.
#   So P(X=k) = e^(-r) r^k / k!, which is the Poisson(r) distribution.
#
# (b) Expectation.
#   E[X] = sum_{k>=0} k * e^(-r) r^k / k!
#        The k=0 term is 0, and k/k! = 1/(k-1)! for k >= 1:
#        = e^(-r) * sum_{k>=1} r^k / (k-1)!
#        Substitute j = k-1 and pull out one factor of r:
#        = e^(-r) * r * sum_{j>=0} r^j / j!
#        = e^(-r) * r * e^r = r.
#   For r = 3:  E[X] = 3.
#
# (b) Variance, using the E[X(X-1)] trick.
#   E[X(X-1)] = sum_{k>=0} k(k-1) * e^(-r) r^k / k!
#        The k=0 and k=1 terms are 0, and k(k-1)/k! = 1/(k-2)! for k >= 2:
#        = e^(-r) * sum_{k>=2} r^k / (k-2)!
#        Substitute j = k-2 and pull out r^2:
#        = e^(-r) * r^2 * sum_{j>=0} r^j / j! = e^(-r) * r^2 * e^r = r^2.
#   E[X^2] = E[X(X-1)] + E[X] = r^2 + r.
#   Var(X) = E[X^2] - (E[X])^2 = r^2 + r - r^2 = r      (identity proved in Q6).
#   For r = 3:  Var(X) = 3.
def distribution_moments(rate=3.):
    """Return (c, mean, variance) for P(X=k)=c*rate**k/k!, k>=0, rate>0."""
    # Step 1: convert to float and check the domain (the series needs rate > 0).
    rate = float(rate)
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError("rate must be a finite positive number")

    # Step 2: c = e^(-rate) from the exponential series sum rate^k/k! = e^rate.
    c = float(np.exp(-rate))

    # Step 3: Poisson(rate) has mean = rate and variance = rate (derived above).
    mean = rate
    variance = rate

    return c, mean, variance


# P2 Q2: Show reasoning for intersection, conditional, and union probabilities.
# Given P(A) = 0.5, P(B) = 0.3, P(A|B) = 0.6.
#
# (a) Product rule: P(A|B) = P(A ∩ B) / P(B)  =>  P(A ∩ B) = P(A|B) * P(B).
#   P(A ∩ B) = 0.6 * 0.3 = 0.18.
#
# (b) Bayes' rule: P(B|A) = P(A ∩ B) / P(A) = P(A|B) P(B) / P(A).
#   P(B|A) = 0.18 / 0.5 = 0.36.
#
# (c) Inclusion-exclusion: P(A ∪ B) = P(A) + P(B) - P(A ∩ B)
#   (A ∩ B is counted in both P(A) and P(B), so subtract it once).
#   P(A ∪ B) = 0.5 + 0.3 - 0.18 = 0.62.
#
# Sanity checks: 0.18 <= min(0.5, 0.3), and 0.5 <= 0.62 <= 1.
def event_probabilities(p_a=.5, p_b=.3, p_a_given_b=.6):
    """Return (P(A intersect B), P(B|A), P(A union B)).

    Inputs describe a valid distribution with p_a,p_b>0.
    """
    # Step 1: convert to floats and check each input is a valid probability.
    p_a, p_b, p_a_given_b = float(p_a), float(p_b), float(p_a_given_b)
    if not (0 < p_a <= 1 and 0 < p_b <= 1):
        raise ValueError("p_a and p_b must be in (0, 1]")
    if not 0 <= p_a_given_b <= 1:
        raise ValueError("p_a_given_b must be in [0, 1]")

    # Step 2: product rule, P(A ∩ B) = P(A|B) * P(B).
    p_intersection = p_a_given_b * p_b

    # Step 3: check the inputs are consistent with one distribution:
    # P(A ∩ B) cannot exceed P(A), and P(A ∪ B) cannot exceed 1.
    tol = 1e-12
    if p_intersection > p_a + tol or p_a + p_b - p_intersection > 1 + tol:
        raise ValueError("inputs do not describe a valid probability distribution")

    # Step 4: Bayes' rule, P(B|A) = P(A ∩ B) / P(A)  (p_a > 0, so this is defined).
    p_b_given_a = p_intersection / p_a

    # Step 5: inclusion-exclusion, P(A ∪ B) = P(A) + P(B) - P(A ∩ B).
    p_union = p_a + p_b - p_intersection

    return p_intersection, p_b_given_a, p_union


# P2 Q3: Derive MLE/MAP for H,H,T,H,T, including the prior and maximization.
# Data: H, H, T, H, T  =>  n = 5 tosses, h = 3 heads, n - h = 2 tails.
# Tosses are independent Bernoulli(theta), theta = P(H).
#
# (a) MLE.
#   Likelihood: L(theta) = prod_i P(x_i | theta) = theta^h * (1-theta)^(n-h)
#                        = theta^3 * (1-theta)^2.
#   Log-likelihood (same maximizer, since log is increasing):
#     l(theta) = h*log(theta) + (n-h)*log(1-theta).
#   Derivative, set to 0 (for 0 < theta < 1):
#     dl/dtheta = h/theta - (n-h)/(1-theta) = 0
#     => h(1-theta) = (n-h)theta  =>  h = n*theta  =>  theta_MLE = h/n.
#   Second derivative: -h/theta^2 - (n-h)/(1-theta)^2 < 0, so it is a maximum.
#   theta_MLE = 3/5 = 0.6.
#
#   Boundary cases (the log is undefined at 0 or 1, so check L directly):
#     h = 0: L(theta) = (1-theta)^n is decreasing on [0,1] => max at theta = 0.
#     h = n: L(theta) = theta^n is increasing on [0,1]     => max at theta = 1.
#   Both agree with h/n, so theta_MLE = h/n for every 0 <= h <= n.
#
# (b) MAP with prior P(theta) = 2*theta on [0, 1].
#   Check the prior is a valid density: integral_0^1 2*theta d(theta) = 1.
#   Posterior is proportional to likelihood * prior:
#     p(theta | data) ∝ theta^h (1-theta)^(n-h) * 2*theta ∝ theta^(h+1) (1-theta)^(n-h)
#   (a Beta(h+2, n-h+1) density). The constant 2 does not change the argmax.
#   Log-posterior: (h+1)*log(theta) + (n-h)*log(1-theta) + const.
#   Derivative, set to 0:
#     (h+1)/theta - (n-h)/(1-theta) = 0
#     => (h+1)(1-theta) = (n-h)theta  =>  h+1 = (n+1)theta
#     => theta_MAP = (h+1)/(n+1).
#   Second derivative: -(h+1)/theta^2 - (n-h)/(1-theta)^2 < 0, so it is a maximum.
#   theta_MAP = (3+1)/(5+1) = 4/6 = 2/3 ~= 0.6667.
#   The prior favors larger theta, so the MAP is pulled above the MLE of 0.6.
#
#   Boundary cases:
#     h = n: posterior ∝ theta^(n+1) is increasing on [0,1] => max at theta = 1,
#            which equals (n+1)/(n+1) = 1.
#     h = 0: posterior ∝ theta (1-theta)^n is 0 at both ends and positive inside,
#            so the max is the interior point from the derivative, 1/(n+1).
#            (It is NOT at theta = 0, because the prior 2*theta is 0 there.)
#   So theta_MAP = (h+1)/(n+1) for every 0 <= h <= n.
def coin_estimates(heads=3, tosses=5):
    """Return (MLE, MAP) using prior density 2*theta on [0,1].

    0<=heads<=tosses, positive integer tosses. Include boundary maxima.
    """
    # Step 1: check the inputs are integer counts with 0 <= heads <= tosses
    # and tosses >= 1 (bool is rejected because it is not a count).
    for name, value in (("heads", heads), ("tosses", tosses)):
        if isinstance(value, bool) or not float(value).is_integer():
            raise ValueError(f"{name} must be an integer")
    heads, tosses = int(heads), int(tosses)
    if tosses < 1 or not 0 <= heads <= tosses:
        raise ValueError("need tosses >= 1 and 0 <= heads <= tosses")

    # Step 2: MLE. The interior solution is h/n; the boundary cases h = 0 and
    # h = n (likelihood monotone on [0,1]) give 0 and 1, which h/n also gives.
    if heads == 0:
        mle = 0.0
    elif heads == tosses:
        mle = 1.0
    else:
        mle = heads / tosses

    # Step 3: MAP. Posterior ∝ theta^(h+1) (1-theta)^(n-h). For h = n it is
    # increasing, so the max is at the boundary theta = 1; otherwise the
    # derivative gives the interior maximum (h+1)/(n+1) (this includes h = 0,
    # where the prior makes the posterior 0 at theta = 0).
    if heads == tosses:
        map_estimate = 1.0
    else:
        map_estimate = (heads + 1) / (tosses + 1)

    return mle, map_estimate


# P2 Q4: Derive both Gaussian MLEs for [1,3,5,7] step by step.
# Model: x_i ~ N(mu, sigma^2) independently, i = 1..n.
#
# Step 1: likelihood and log-likelihood.
#   L(mu, sigma^2) = prod_i (2*pi*sigma^2)^(-1/2) * exp(-(x_i - mu)^2 / (2 sigma^2))
#   l(mu, sigma^2) = -(n/2) log(2*pi) - (n/2) log(sigma^2)
#                    - (1/(2 sigma^2)) * sum_i (x_i - mu)^2
#
# Step 2: derivative with respect to mu, set to 0.
#   dl/dmu = (1/sigma^2) * sum_i (x_i - mu) = 0
#   => sum_i x_i - n*mu = 0  =>  mu_MLE = (1/n) sum_i x_i  (the sample mean).
#
# Step 3: derivative with respect to sigma^2 (treat s = sigma^2 as the variable).
#   dl/ds = -n/(2s) + (1/(2 s^2)) * sum_i (x_i - mu)^2 = 0
#   Multiply by 2 s^2:  -n*s + sum_i (x_i - mu)^2 = 0
#   => sigma^2_MLE = (1/n) sum_i (x_i - mu_MLE)^2   (divisor n, not n-1).
#   (The n-1 version is the unbiased estimator; the MLE is slightly biased low.)
#
# Step 4: plug in X = {1, 3, 5, 7}, n = 4.
#   mu_MLE = (1 + 3 + 5 + 7) / 4 = 16 / 4 = 4.
#   Deviations: 1-4 = -3, 3-4 = -1, 5-4 = 1, 7-4 = 3.
#   Squares: 9, 1, 1, 9  => sum = 20.
#   sigma^2_MLE = 20 / 4 = 5.
def gaussian_mle(data=None):
    """Return (mean MLE, variance MLE); default data=[1,3,5,7].

    Other inputs are finite 1D samples of length>=2 with nonzero variance.
    """
    # Step 1: fill in the default dataset from the problem statement.
    if data is None:
        data = [1, 3, 5, 7]

    # Step 2: copy into a new float array so the caller's input is never mutated,
    # and check it is a finite 1D sample with at least 2 points.
    x = np.array(data, dtype=float)
    if x.ndim != 1 or x.size < 2:
        raise ValueError("data must be a 1D sample with at least 2 values")
    if not np.all(np.isfinite(x)):
        raise ValueError("data must be finite")

    # Step 3: mu_MLE = sample mean (from dl/dmu = 0).
    mean = float(np.mean(x))

    # Step 4: sigma^2_MLE = average squared deviation, divisor n (from dl/ds = 0).
    variance = float(np.mean((x - mean) ** 2))

    return mean, variance


# P2 Q5: Derive the Gaussian posterior mode for the given observation/prior.
# Model: x | mu ~ N(mu, s2) with known s2; prior mu ~ N(m0, t2).
# Given x = 5, s2 = 4, m0 = 0, t2 = 1.
#
# Step 1: posterior ∝ likelihood * prior.
#   p(mu | x) ∝ exp(-(x - mu)^2 / (2 s2)) * exp(-(mu - m0)^2 / (2 t2))
#
# Step 2: log-posterior (dropping constants that do not depend on mu).
#   log p(mu | x) = -(x - mu)^2 / (2 s2) - (mu - m0)^2 / (2 t2) + const
#
# Step 3: derivative with respect to mu, set to 0.
#   d/dmu = (x - mu)/s2 - (mu - m0)/t2 = 0
#   => x/s2 + m0/t2 = mu * (1/s2 + 1/t2)
#   => mu_MAP = (x/s2 + m0/t2) / (1/s2 + 1/t2).
#   Second derivative: -(1/s2 + 1/t2) < 0, so this is a maximum.
#   Equivalently, completing the square shows the posterior is Gaussian with
#   precision 1/s2 + 1/t2, and its mode equals its mean, the formula above.
#   Interpretation: a precision-weighted average of the observation x
#   (weight 1/s2) and the prior mean m0 (weight 1/t2).
#
# Step 4: plug in x = 5, s2 = 4, m0 = 0, t2 = 1.
#   mu_MAP = (5/4 + 0/1) / (1/4 + 1/1) = 1.25 / 1.25 = 1.
#   The prior (precision 1) is 4x more confident than the observation
#   (precision 1/4), so the estimate is pulled strongly from 5 toward 0.
def gaussian_map(x=5., observation_variance=4., prior_mean=0., prior_variance=1.):
    """Return posterior mode for one observation and a Gaussian mean prior.

    Both variances are positive; inputs are variances, not standard deviations.
    """
    # Step 1: convert to floats and check both variances are positive.
    x, prior_mean = float(x), float(prior_mean)
    observation_variance = float(observation_variance)
    prior_variance = float(prior_variance)
    if observation_variance <= 0 or prior_variance <= 0:
        raise ValueError("variances must be positive")

    # Step 2: precisions (inverse variances) are the weights in the formula.
    obs_precision = 1.0 / observation_variance
    prior_precision = 1.0 / prior_variance

    # Step 3: precision-weighted average from setting d/dmu log-posterior = 0.
    return (obs_precision * x + prior_precision * prior_mean) / (obs_precision + prior_precision)


# P2 Q6: Prove Var(X)=E[X**2]-E[X]**2.
# Let mu = E[X] (a constant). By definition Var(X) = E[(X - mu)^2].
#
# Step 1: expand the square.
#   (X - mu)^2 = X^2 - 2*mu*X + mu^2
#
# Step 2: apply linearity of expectation, E[aU + bV + c] = aE[U] + bE[V] + c.
#   E[(X - mu)^2] = E[X^2] - 2*mu*E[X] + mu^2
#
# Step 3: substitute E[X] = mu.
#   = E[X^2] - 2*mu^2 + mu^2
#   = E[X^2] - mu^2
#
# Therefore Var(X) = E[X^2] - (E[X])^2.   QED
# (Used in Q1: Var(X) = (r^2 + r) - r^2 = r.)

# PART 3 (30 points), Q1: Define covariance and derive Var(u.T@X)=u.T@Sigma@u.
# Let X be a random vector in R^d and u a fixed (non-random) unit vector, ||u|| = 1.
#
# Definitions:
#   Mean:        mu    = E[X]                      (a d-vector)
#   Covariance:  Sigma = E[(X - mu)(X - mu)^T]     (a d x d matrix)
#     Entry-wise Sigma_ij = Cov(X_i, X_j); Sigma is symmetric (Sigma_ij = Sigma_ji)
#     and positive semi-definite (shown at the end of Step 5).
#
# Step 1: the projection u^T X is a scalar random variable, and its mean is
#   E[u^T X] = u^T E[X] = u^T mu        (linearity of expectation; u is constant).
#
# Step 2: definition of variance for a scalar, Var(Y) = E[(Y - E[Y])^2]:
#   Var(u^T X) = E[(u^T X - u^T mu)^2] = E[(u^T (X - mu))^2]
#   (factor out u^T, since matrix multiplication distributes over subtraction).
#
# Step 3: u^T (X - mu) is a scalar, and a scalar equals its own transpose, so
#   u^T (X - mu) = (X - mu)^T u. Write the square as the product of the two forms:
#   (u^T (X - mu))^2 = u^T (X - mu) (X - mu)^T u.
#   => Var(u^T X) = E[u^T (X - mu)(X - mu)^T u]
#
# Step 4: u is constant, so linearity lets it move outside the expectation
#   (E[A Z B] = A E[Z] B for constant matrices A, B):
#   Var(u^T X) = u^T E[(X - mu)(X - mu)^T] u
#
# Step 5: the middle term is exactly the definition of Sigma:
#   Var(u^T X) = u^T Sigma u.   QED
#   Since a variance is >= 0, u^T Sigma u >= 0 for every u, so Sigma is PSD
#   and all of its eigenvalues are >= 0.
#
# Why the top eigenvector maximizes the projected variance (Lagrange multipliers):
#   Maximize u^T Sigma u subject to u^T u = 1.
#   Lagrangian: J(u, lambda) = u^T Sigma u - lambda (u^T u - 1).
#   Gradient: dJ/du = 2 Sigma u - 2 lambda u = 0  =>  Sigma u = lambda u.
#   So every stationary point is an eigenvector of Sigma, and at such a point
#   u^T Sigma u = u^T (lambda u) = lambda (u^T u) = lambda.
#   The projected variance therefore equals the eigenvalue, and it is largest for
#   the eigenvector with the largest eigenvalue: this is the first principal
#   component. Repeating with the extra constraint u perpendicular to the
#   earlier components gives the next eigenvectors in order of decreasing
#   eigenvalue.

# P3 Q2: Describe the Gaussian generator and observed correlation.
# Generator: X ~ N(mean, covariance) in 2D, with default
#   mean = [2, -1],  covariance = [[3, 1.2], [1.2, 1]].
#   The covariance is positive definite: det = 3*1 - 1.2^2 = 1.56 > 0 and 3 > 0.
#   np.random.default_rng(seed).multivariate_normal factorizes the covariance
#   (Sigma = L L^T) and returns mean + L z with z ~ N(0, I). This gives
#   samples whose covariance is Sigma. The same seed always gives the same samples.
#
# Correlation structure:
#   Theoretical correlation = Cov(X1, X2) / sqrt(Var(X1) Var(X2))
#                           = 1.2 / sqrt(3 * 1) ~= 0.693 (strong positive).
#   With n_samples = 500, seed = 0, the sample statistics are:
#     sample mean ~= [2.152, -0.935]          (true [2, -1])
#     sample covariance ~= [[2.916, 1.166],   (true [[3, 1.2],
#                           [1.166, 0.953]]          [1.2, 1]])
#     sample correlation ~= 0.699             (true 0.693)
#   The scatter plot is a tilted ellipse: when feature 1 is above its mean,
#   feature 2 tends to be above its mean too. The ellipse is wider along
#   feature 1 (variance 3) than feature 2 (variance 1), so its long axis is
#   tilted only about 25 degrees above the feature-1 axis, not 45 degrees.
#   The small differences from the true values are sampling noise, which
#   shrinks like 1/sqrt(n).
def generate_data(n_samples=500, seed=0, mean=None, covariance=None):
    """Return reproducible Gaussian samples (n_samples,2) using NumPy and seed.

    Defaults: mean=[2,-1], covariance=[[3,1.2],[1.2,1]]. Custom mean has shape
    (2,), covariance (2,2) is positive definite. Do not mutate input arrays.
    """
    # Step 1: fill in the defaults from the assignment.
    if mean is None:
        mean = [2., -1.]
    if covariance is None:
        covariance = [[3., 1.2], [1.2, 1.]]

    # Step 2: copy into new float arrays so the caller's inputs are never mutated.
    mean = np.array(mean, dtype=float)
    covariance = np.array(covariance, dtype=float)

    # Step 3: check the inputs: integer sample count >= 1, mean shape (2,),
    # covariance shape (2, 2), symmetric and positive definite (Cholesky
    # succeeds only for positive definite matrices).
    if isinstance(n_samples, bool) or not float(n_samples).is_integer() or n_samples < 1:
        raise ValueError("n_samples must be a positive integer")
    if mean.shape != (2,):
        raise ValueError("mean must have shape (2,)")
    if covariance.shape != (2, 2) or not np.allclose(covariance, covariance.T):
        raise ValueError("covariance must be a symmetric (2, 2) matrix")
    try:
        np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError:
        raise ValueError("covariance must be positive definite") from None

    # Step 4: draw reproducible samples from a generator seeded with `seed`.
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(mean, covariance, size=int(n_samples))


# P3 Q3: Link centering/covariance/eigendecomposition to the derivation and explain the plot.
# Each step of pca() below follows the Q1 derivation:
#   1. Centering: subtract the sample mean, X_c = X - mean. This estimates the
#      (X - mu) term in Sigma = E[(X - mu)(X - mu)^T].
#   2. Covariance: Sigma_hat = X_c^T X_c / (n - 1). This is the sample version of
#      E[(X - mu)(X - mu)^T]. The n - 1 divisor makes it unbiased, because one degree
#      of freedom is spent estimating the mean.
#   3. Eigendecomposition: Q1 showed the directions maximizing u^T Sigma u with
#      ||u|| = 1 are the solutions of Sigma u = lambda u. Sigma_hat is symmetric,
#      so np.linalg.eigh gives real eigenvalues and orthonormal eigenvectors.
#   4. Selection: the eigenvalue equals the variance of the data projected on
#      that eigenvector (u^T Sigma u = lambda), so sorting eigenvalues in
#      descending order ranks the components from most to least variance.
#   5. Scores: scores = X_c @ components gives each sample's coordinates u_k^T x_c
#      along each component. The sample variance of score column k equals
#      lambda_k, and the score columns are uncorrelated.
#
# Results for the default data (n = 500, seed = 0):
#   eigenvalues ~= [3.458, 0.411]   (true [3.562, 0.438])
#   PC1 ~= +/-[0.907, 0.422] (about 25 degrees above the feature-1 axis),
#   PC2 ~= +/-[-0.422, 0.907] (perpendicular to PC1).
# On the plot, both directions are drawn from the sample mean. PC1 follows the
# long axis of the elliptical cloud, and PC2 follows its short axis.
#
# Special cases handled:
#   - Rank-deficient data (e.g. all points on a line) gives a zero eigenvalue.
#     Round-off can make it slightly negative, so it is clipped to 0 because a
#     covariance matrix is PSD (Q1).
#   - Tied eigenvalues (e.g. an isotropic cloud) have no unique direction. Any
#     orthonormal basis of the tied eigenspace is valid, and eigh returns one.
def pca(X):
    """Fit NumPy PCA without mutating finite X of shape (n,2), n>=2.

    Return dict: mean (2,), centered (n,2), covariance (2,2), eigenvalues (2,),
    components (2,2), scores (n,2). Use covariance divisor n-1, descending
    eigenvalues, orthonormal eigenvectors as columns, scores=centered@components.
    Signs/tied-eigenspace bases are free. Include rank-deficient and tied cases.
    NumPy eigh/SVD are allowed; fitted library PCA is not.
    """
    # Step 0: copy into a new float array (so X is never mutated) and check the
    # input is a finite 2D array with at least 2 samples (n - 1 > 0).
    X = np.array(X, dtype=float)
    if X.ndim != 2 or X.shape[0] < 2:
        raise ValueError("X must be a 2D array with at least 2 rows")
    if not np.all(np.isfinite(X)):
        raise ValueError("X must be finite")
    n = X.shape[0]

    # Step 1 (Q1: X - mu): estimate mu with the sample mean and center the data.
    mean = X.mean(axis=0)
    centered = X - mean

    # Step 2 (Q1: Sigma = E[(X - mu)(X - mu)^T]): sample covariance, divisor n - 1.
    covariance = centered.T @ centered / (n - 1)
    # Enforce exact symmetry (round-off can break it slightly) before eigh.
    covariance = (covariance + covariance.T) / 2

    # Step 3 (Q1: Lagrange condition Sigma u = lambda u): eigendecomposition of the
    # symmetric covariance. eigh returns ascending eigenvalues and orthonormal
    # eigenvector columns, including an orthonormal basis for tied eigenvalues.
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    # Step 4 (Q1: projected variance = lambda): sort descending so component 1
    # carries the most variance, and reorder the columns to keep the pairs.
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    components = eigenvectors[:, order]

    # Covariance is PSD, so any negative eigenvalue is round-off; clip it to 0.
    eigenvalues = np.clip(eigenvalues, 0.0, None)

    # Optional sign convention (signs are free): make the largest-magnitude
    # entry of each component positive, so results do not depend on eigh's
    # arbitrary sign choice. Flipping a column keeps it a unit eigenvector.
    largest = np.argmax(np.abs(components), axis=0)
    signs = np.sign(components[largest, np.arange(components.shape[1])])
    signs[signs == 0] = 1.0
    components = components * signs

    # Step 5: project the centered data onto the components, score_k = u_k^T x_c.
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
# Total variance = trace(Sigma) = sum of eigenvalues (the trace is invariant
# under the orthogonal change of basis to the eigenvectors). So component k
# explains the fraction  lambda_k / sum_j lambda_j  of the total variance.
#
# Default data (n = 500, seed = 0):
#   fractions ~= [0.894, 0.106]   (theoretical [0.891, 0.109], from
#   eigenvalues 2 +/- sqrt(2.44) ~= 3.562, 0.438 of the true covariance).
#   PC1 alone keeps about 89% of the variance, and PC2 holds only about 11%.
#
# Dimensionality reduction:
#   Keeping only PC1 (projecting 2D -> 1D) keeps ~89% of the variance and
#   stores 1 number per sample instead of 2. The discarded variance
#   lambda_2 ~= 0.411 (~11% of the total) equals the squared reconstruction
#   error summed over samples divided by n - 1, i.e. what is lost along PC2.
#   PCA's choice is optimal: no other 1D linear projection keeps more
#   variance (Q1), so none has a smaller reconstruction error.
#
# Tradeoff:
#   Fewer components mean more compression, less noise and faster downstream
#   models, but more information lost. More components mean higher fidelity
#   but less reduction. A common rule is to keep the smallest k whose
#   cumulative explained variance reaches a threshold (e.g. 90-95%). Here 1
#   component gives ~89%, just under 90%, and 2 give 100% but no reduction.
#   The gain is small in 2D but grows in high dimensions, where a few
#   components often explain most of the variance. The limitation is that PCA
#   only captures linear structure and depends on feature scale (standardize
#   features first when their units differ).
def explained_variance(eigenvalues):
    """Return same-shape fractions for nonnegative eigenvalues with positive sum."""
    # Step 1: copy into a new float array so the caller's input is never mutated.
    eigenvalues = np.array(eigenvalues, dtype=float)

    # Step 2: check the docstring's preconditions: finite, nonnegative, sum > 0.
    if not np.all(np.isfinite(eigenvalues)) or np.any(eigenvalues < 0):
        raise ValueError("eigenvalues must be finite and nonnegative")
    total = eigenvalues.sum()
    if total <= 0:
        raise ValueError("eigenvalues must have a positive sum")

    # Step 3: fraction of total variance, lambda_k / sum_j lambda_j (same shape).
    return eigenvalues / total

# P3 Q2/Q3/Q4: Provide labeled figures and discuss their meaning.
# Figures saved by the main block (default data, n = 500, seed = 0):
#
# data_scatter.png (Q2): a plain scatter of the samples. The cloud is an
#   ellipse tilted upward to the right: x1 and x2 rise together, with sample
#   correlation r = 0.699 (theory 1.2/sqrt(3) ~= 0.693). The cloud is wider
#   along x1 (variance ~2.92) than along x2 (variance ~0.95).
#
# pca_plots.png, left panel (Q3): the same samples, their mean (black X at
#   about (2.15, -0.93)), and both principal directions drawn from the mean
#   with length 2*sqrt(lambda_k), i.e. two standard deviations along that
#   component. PC1 (orange, length 3.72) points along the long axis of the
#   ellipse, about 25 degrees above the x1 axis. PC2 (aqua, length 1.28) is
#   perpendicular to it along the short axis. The axes use an equal aspect
#   ratio, so the right angle between the components is shown faithfully.
#   This matches Q1: the direction u maximizing u^T Sigma u is the top
#   eigenvector, and the spread of the data along each arrow is
#   sqrt(u^T Sigma u) = sqrt(lambda).
#
# pca_plots.png, right panel (Q4): the explained variance fractions,
#   PC1 = 89.4% and PC2 = 10.6%. Almost all of the variation lies along one
#   direction, so projecting the 2D data onto PC1 alone keeps most of the
#   information (see the Q4 discussion above for the tradeoff).
# Colors: data points use one hue (blue); PC1 (orange) and PC2 (aqua) keep the
# same color in both panels, so each component reads as one entity. Every arrow
# and bar also has a text label, so identity never depends on color alone.
_DATA_COLOR = '#2a78d6'
_PC_COLORS = ('#eb6834', '#1baf7a', '#6250d6', '#e87ba4')
_INK = '#0b0b0b'
_INK_MUTED = '#52514e'
_GRID = '#e4e3df'


def _style_axes(ax):
    """Recessive grid and spines so the data marks stand out (helper)."""
    ax.grid(True, color=_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(_INK_MUTED)
    ax.tick_params(colors=_INK_MUTED)


def make_scatter(X):
    """Return a Figure with a plain labeled scatter of 2D data X (for P3 Q2)."""
    # Step 1: copy X so the caller's array is never mutated.
    X = np.array(X, dtype=float)

    # Step 2: build the figure without pyplot, so importing this module or
    # calling this function never needs a display (saving is done in main).
    fig = Figure(figsize=(6, 5), layout='constrained')
    ax = fig.add_subplot()
    _style_axes(ax)
    ax.scatter(X[:, 0], X[:, 1], s=14, alpha=0.5, color=_DATA_COLOR,
               edgecolors='none', label=f'samples (n = {len(X)})')
    r = np.corrcoef(X, rowvar=False)[0, 1]
    ax.set_title(f'Synthetic correlated Gaussian data (sample r = {r:.3f})', color=_INK)
    ax.set_xlabel('Feature 1 ($x_1$)', color=_INK)
    ax.set_ylabel('Feature 2 ($x_2$)', color=_INK)
    ax.set_aspect('equal', adjustable='datalim')
    ax.legend(loc='upper left', frameon=False)
    return fig


def make_plots(X, result):
    """Return a Matplotlib Figure (or sequence of Figures), without show/save.

    Plot samples and principal directions through their mean, and explained
    variance fractions. Label axes. Plot quality is manually reviewed.
    """
    # Step 1: copy the inputs so neither X nor the pca() result is mutated.
    X = np.array(X, dtype=float)
    mean = np.array(result['mean'], dtype=float)
    eigenvalues = np.array(result['eigenvalues'], dtype=float)
    components = np.array(result['components'], dtype=float)
    fractions = explained_variance(eigenvalues)
    names = [f'PC{k + 1}' for k in range(len(eigenvalues))]
    colors = [_PC_COLORS[k % len(_PC_COLORS)] for k in range(len(eigenvalues))]

    # Step 2: create a two-panel figure without pyplot (no show/save here).
    fig = Figure(figsize=(12, 5), layout='constrained')
    ax_data, ax_var = fig.subplots(1, 2, width_ratios=[1.3, 1])

    # Step 3 (left panel): samples, their mean, and each principal direction
    # drawn from the mean with length 2*sqrt(lambda_k), i.e. two standard
    # deviations of the data projected on that component (Q1: Var = lambda).
    _style_axes(ax_data)
    ax_data.scatter(X[:, 0], X[:, 1], s=14, alpha=0.35, color=_DATA_COLOR,
                    edgecolors='none', label=f'samples (n = {len(X)})')
    handles = [ax_data.collections[0]]
    tips = []
    for k in range(len(eigenvalues)):
        direction = components[:, k] * 2 * np.sqrt(eigenvalues[k])
        tip = mean + direction
        tips.append(tip)
        ax_data.annotate('', xy=tip, xytext=mean,
                         arrowprops=dict(arrowstyle='-|>', color=colors[k],
                                         lw=2.5, mutation_scale=18,
                                         shrinkA=0, shrinkB=0))
        # Label just beyond the arrow tip, pushed outward along the direction.
        norm = np.linalg.norm(direction)
        offset = direction / norm * 0.35 if norm > 0 else np.zeros(2)
        ax_data.text(*(tip + offset), names[k], color=_INK, fontsize=11,
                     fontweight='bold', ha='center', va='center')
        handles.append(Line2D([], [], color=colors[k], lw=2.5,
                              label=f'{names[k]}: 2$\\sqrt{{\\lambda_{k + 1}}}$ = '
                                    f'{2 * np.sqrt(eigenvalues[k]):.2f}'))
    handles.append(ax_data.scatter(*mean, s=120, marker='X', color=_INK,
                                   edgecolors='white', linewidths=1.5, zorder=3,
                                   label=f'mean ({mean[0]:.2f}, {mean[1]:.2f})'))
    # annotate() does not autoscale, so include the arrow tips in the limits.
    ax_data.update_datalim(np.vstack(tips + [mean]))
    ax_data.autoscale_view()
    ax_data.set_aspect('equal', adjustable='datalim')
    ax_data.set_title('Data with principal directions from the mean', color=_INK)
    ax_data.set_xlabel('Feature 1 ($x_1$)', color=_INK)
    ax_data.set_ylabel('Feature 2 ($x_2$)', color=_INK)
    ax_data.legend(handles=handles, loc='upper left', frameon=True,
                   framealpha=0.9, edgecolor=_GRID)

    # Step 4 (right panel): explained variance fraction of each component,
    # with the percentage written above each bar.
    _style_axes(ax_var)
    ax_var.grid(False, axis='x')
    positions = np.arange(len(fractions))
    bars = ax_var.bar(positions, fractions, width=0.6, color=colors,
                      edgecolor='white', linewidth=2)
    for bar, fraction in zip(bars, fractions):
        ax_var.text(bar.get_x() + bar.get_width() / 2, fraction + 0.02,
                    f'{100 * fraction:.1f}%', ha='center', va='bottom',
                    color=_INK, fontsize=12, fontweight='bold')
    ax_var.set_xticks(positions, names)
    ax_var.set_ylim(0, 1.1)
    ax_var.set_title('Explained variance by principal component', color=_INK)
    ax_var.set_xlabel('Principal component', color=_INK)
    ax_var.set_ylabel('Fraction of total variance', color=_INK)

    return fig


# REFLECTION: contributions, tasks completed, and external resources used.
# - Contributions: I worked on this assignment individually.
# - Tasks completed: all parts - Part 1 (linear algebra), Part 2 (probability,
#   MLE, MAP) and Part 3 (PCA derivation, NumPy implementation and plots).
# - Resources: I used an AI assistant to help draft the derivations, code and
#   tests; the full conversation log is submitted to Canvas per the course AI
#   policy. I also used the NumPy and Matplotlib documentation.
# - Verification: I checked the results with a test script
#   (test_assignment1.py) and against the hand-computed values in the
#   derivations.

if __name__ == '__main__':
    # This block does not run when the grader imports your functions.
    # Step 1: generate the default dataset (500 samples, seed 0).
    X = generate_data()

    # Step 2: fit PCA and compute the explained variance fractions.
    result = pca(X)
    fractions = explained_variance(result['eigenvalues'])

    # Step 3: print the results (printing is allowed only here).
    np.set_printoptions(precision=4, suppress=True)
    print('Part 1')
    for name, value in zip(['AB', 'BA', '(A+B)^T', 'AB^T', 'trace(A)', 'trace(AB)'],
                           matrix_operations()):
        print(f'  {name} =\n{value}' if np.ndim(value) else f'  {name} = {value}')
    values, vectors = eigensystem()
    print(f'  eigenvalues of M = {values}\n  eigenvectors (columns) =\n{vectors}')
    print(f'  C^-1 =\n{matrix_inverse()}')
    print('Part 2')
    print('  (c, E[X], Var(X)) =', distribution_moments())
    print('  (P(A and B), P(B|A), P(A or B)) =', event_probabilities())
    print('  (coin MLE, coin MAP) =', coin_estimates())
    print('  (Gaussian mean MLE, variance MLE) =', gaussian_mle())
    print('  Gaussian MAP of mu =', gaussian_map())
    print('Part 3')
    print(f'  sample correlation = {np.corrcoef(X, rowvar=False)[0, 1]:.4f}'
          f' (theory 1.2/sqrt(3) = {1.2 / np.sqrt(3):.4f})')
    print(f'  mean = {result["mean"]}')
    print(f'  covariance =\n{result["covariance"]}')
    print(f'  eigenvalues = {result["eigenvalues"]}')
    print(f'  components (columns) =\n{result["components"]}')
    print(f'  explained variance fractions = {fractions}')

    # Step 4: save the figures (saving is allowed only here).
    make_scatter(X).savefig('data_scatter.png', dpi=150)
    make_plots(X, result).savefig('pca_plots.png', dpi=150)
    print('Saved data_scatter.png and pca_plots.png')
