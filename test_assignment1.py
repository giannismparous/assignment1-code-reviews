"""Plain-assert tests for Assignment1.py. Run with: python test_assignment1.py"""
import copy
import math

import numpy as np

from Assignment1 import (coin_estimates, distribution_moments, eigensystem,
                         event_probabilities, explained_variance, gaussian_map,
                         gaussian_mle, generate_data, make_plots, matrix_inverse,
                         matrix_operations, pca)


def close(a, b, tol=1e-9):
    return np.allclose(a, b, atol=tol, rtol=0)


def test_matrix_operations():
    AB, BA, sum_t, AB_t, tr_A, tr_AB = matrix_operations()
    assert close(AB, [[3, 2], [2, 3]])
    assert close(BA, [[7, 6], [-2, -1]])
    assert close(sum_t, [[4, 2], [6, 0]])
    assert close(AB_t, [[11, -4], [4, -1]])
    assert close(tr_A, 4) and close(tr_AB, 6)
    # General square input of another size.
    rng = np.random.default_rng(0)
    A, B = rng.normal(size=(3, 3)), rng.normal(size=(3, 3))
    out = matrix_operations(A, B)
    assert close(out[0], A @ B) and close(out[1], B @ A)
    assert close(out[2], (A + B).T) and close(out[3], A @ B.T)
    assert close(out[4], np.trace(A)) and close(out[5], np.trace(A @ B))


def test_eigensystem():
    M = np.array([[2., -1.], [-1., 2.]])
    values, vectors = eigensystem()
    assert close(sorted(values), [1, 3])
    expected = {1: np.array([1., 1.]), 3: np.array([1., -1.])}
    for k, lam in enumerate(values):
        v = vectors[:, k]
        assert close(M @ v, lam * v)                      # M v = lambda v
        u = expected[int(round(lam))]
        cosine = abs(v @ u) / (np.linalg.norm(v) * np.linalg.norm(u))
        assert close(cosine, 1)                           # parallel (sign/scale free)
    # Repeated eigenvalue: the vectors must still span the whole eigenspace.
    values, vectors = eigensystem(np.eye(3))
    assert close(values, [1, 1, 1]) and np.linalg.matrix_rank(vectors) == 3


def test_matrix_inverse():
    C_inv = matrix_inverse()
    assert close(C_inv, [[0.6, -0.7], [-0.2, 0.4]])
    assert close(np.array([[4, 7], [2, 6]]) @ C_inv, np.eye(2))


def test_distribution_moments():
    c, mean, var = distribution_moments()
    assert close(c, math.exp(-3)) and close(mean, 3) and close(var, 3)
    c, mean, var = distribution_moments(1.5)
    assert close(c, math.exp(-1.5)) and close(mean, 1.5) and close(var, 1.5)


def test_event_probabilities():
    p_and, p_b_given_a, p_or = event_probabilities()
    assert close(p_and, 0.18) and close(p_b_given_a, 0.36) and close(p_or, 0.62)


def test_coin_estimates():
    mle, map_ = coin_estimates()
    assert close(mle, 0.6) and close(map_, 2 / 3)
    # Edge case heads = 0: MLE at the boundary 0; MAP interior at 1/(n+1).
    mle, map_ = coin_estimates(0, 5)
    assert close(mle, 0) and close(map_, 1 / 6)
    # Edge case heads = tosses: both maxima at the boundary theta = 1.
    mle, map_ = coin_estimates(5, 5)
    assert close(mle, 1) and close(map_, 1)


def test_gaussian():
    mu, var = gaussian_mle()
    assert close(mu, 4) and close(var, 5)
    assert close(gaussian_map(), 1.0)


def check_pca(X):
    r = pca(X)
    n = len(X)
    V, w = r['components'], r['eigenvalues']
    assert r['mean'].shape == (2,) and r['centered'].shape == (n, 2)
    assert r['covariance'].shape == (2, 2) and w.shape == (2,)
    assert V.shape == (2, 2) and r['scores'].shape == (n, 2)
    assert close(r['covariance'], np.cov(X, rowvar=False))       # matches np.cov
    assert close(V.T @ V, np.eye(2))                              # orthonormal
    assert w[0] >= w[1] >= 0                                      # descending, >= 0
    assert close(r['covariance'] @ V, V * w)                      # eigenpairs
    assert close(r['scores'], r['centered'] @ V)
    return r


def test_pca():
    check_pca(generate_data())
    # Rank-deficient: all points on the line y = 2x + 1.
    t = np.linspace(-1, 1, 50)
    r = check_pca(np.column_stack([t, 2 * t + 1]))
    assert close(r['eigenvalues'][1], 0)
    # Tied: 4 symmetric points scaled so the covariance is exactly the identity.
    X_tied = np.sqrt(1.5) * np.array([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]])
    r = check_pca(X_tied)
    assert close(r['covariance'], np.eye(2)) and close(r['eigenvalues'], [1, 1])


def test_explained_variance():
    fractions = explained_variance(pca(generate_data())['eigenvalues'])
    assert fractions.shape == (2,) and close(fractions.sum(), 1)
    assert close(explained_variance([3., 1.]), [0.75, 0.25])


def test_inputs_not_mutated():
    A, B = np.array([[3., 4.], [2., 1.]]), np.array([[1., 2.], [0., -1.]])
    A0, B0 = A.copy(), B.copy()
    matrix_operations(A, B)
    assert np.array_equal(A, A0) and np.array_equal(B, B0)
    M = np.array([[2., -1.], [-1., 2.]]); M0 = M.copy(); eigensystem(M)
    assert np.array_equal(M, M0)
    C = np.array([[4., 7.], [2., 6.]]); C0 = C.copy(); matrix_inverse(C)
    assert np.array_equal(C, C0)
    data = np.array([1., 3., 5., 7.]); data0 = data.copy(); gaussian_mle(data)
    assert np.array_equal(data, data0)
    mean, cov = np.array([2., -1.]), np.array([[3., 1.2], [1.2, 1.]])
    mean0, cov0 = mean.copy(), cov.copy(); generate_data(10, 0, mean, cov)
    assert np.array_equal(mean, mean0) and np.array_equal(cov, cov0)
    X = generate_data(); X0 = X.copy(); r = pca(X)
    assert np.array_equal(X, X0)
    ev = r['eigenvalues']; ev0 = ev.copy(); explained_variance(ev)
    assert np.array_equal(ev, ev0)
    r0 = copy.deepcopy(r); make_plots(X, r)
    assert np.array_equal(X, X0) and all(np.array_equal(r[k], r0[k]) for k in r)


if __name__ == '__main__':
    tests = [f for name, f in sorted(globals().items())
             if name.startswith('test_') and callable(f)]
    for test in tests:
        test()
        print(f'PASS {test.__name__}')
    print(f'All {len(tests)} tests passed.')
