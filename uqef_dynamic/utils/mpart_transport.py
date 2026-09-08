"""MParT-based triangular transport maps for the particle-filtering pipeline.

Purpose
-------
The particle filter yields, at every timestep, an ensemble of posterior parameter
samples theta_i with a non-standard, correlated, bounded distribution. Polynomial
chaos expansions need inputs in a *standard* space (independent standard normals
for Hermite bases). A triangular transport map S provides that:

    z = S(theta)        forward:  posterior  ->  standard Gaussian
    theta = S^{-1}(z)   inverse:  standard Gaussian -> posterior

Once z_i = S(theta_i) is available alongside the model outputs Q_i, a PCE can be
fitted per timestep by plain regression in z-space:

    Q(z) ~= sum_alpha c_alpha * He_alpha(z)

ORIENTATION (the easy thing to get wrong)
-----------------------------------------
MParT works column-major: it expects arrays shaped (n_dim, n_samples).
The pipeline stores samples row-major: (n_samples, n_dim).
Every public function here takes and returns the PIPELINE orientation
(n_samples, n_dim) and transposes internally, so callers never deal with it.

Install
-------
MParT is an optional dependency:

    pip install mpart          # or: conda install -c conda-forge mpart

All entry points degrade gracefully when it is absent; call is_available()
to check before use.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import multivariate_normal

try:  # optional dependency
    import mpart as mt
    _MPART_IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - depends on environment
    mt = None
    _MPART_IMPORT_ERROR = exc


__all__ = [
    "is_available",
    "TransportMapResult",
    "fit_transport_map",
    "forward",
    "inverse",
]


def is_available():
    """True when the mpart package can be imported."""
    return mt is not None


def _require_mpart():
    if mt is None:
        raise ImportError(
            "MParT is required for this operation but could not be imported "
            f"({_MPART_IMPORT_ERROR}). Install it with 'pip install mpart' or "
            "'conda install -c conda-forge mpart'."
        ) from _MPART_IMPORT_ERROR


def _as_mpart(X):
    """(n_samples, n_dim) pipeline layout -> (n_dim, n_samples) MParT layout."""
    X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    if X.ndim != 2:
        raise ValueError(f"Expected a 2-D array (n_samples, n_dim), got shape {X.shape}.")
    return np.asfortranarray(X.T)


def _as_pipeline(Z):
    """(n_dim, n_samples) MParT layout -> (n_samples, n_dim) pipeline layout."""
    return np.ascontiguousarray(np.asarray(Z, dtype=np.float64).T)


class TransportMapResult:
    """A fitted triangular map plus the metadata needed to reuse it.

    Attributes:
        tri_map:      the fitted MParT triangular map object.
        n_dim:        dimension of the parameter space.
        max_order:    total polynomial order used.
        coeffs:       optimized coefficient vector (copy).
        param_names:  optional list of parameter names, for bookkeeping.
        optimizer:    the scipy OptimizeResult from fitting.
    """

    def __init__(self, tri_map, n_dim, max_order, coeffs,
                 param_names=None, optimizer=None):
        self.tri_map = tri_map
        self.n_dim = n_dim
        self.max_order = max_order
        self.coeffs = np.asarray(coeffs, dtype=np.float64).copy()
        self.param_names = list(param_names) if param_names is not None else None
        self.optimizer = optimizer

    def forward(self, X):
        """theta -> z. X is (n_samples, n_dim); returns (n_samples, n_dim)."""
        return forward(self, X)

    def inverse(self, Z):
        """z -> theta. Z is (n_samples, n_dim); returns (n_samples, n_dim)."""
        return inverse(self, Z)

    def diagnostics(self, X):
        """Quality of the Gaussianization, computed on samples X.

        For a well-fitted map the pushforward z should have mean ~0 and
        covariance ~I. Returns a dict with those, plus the maximum absolute
        deviation of the covariance from the identity.
        """
        Z = self.forward(X)
        mean = Z.mean(axis=0)
        cov = np.cov(Z, rowvar=False)
        cov = np.atleast_2d(cov)
        return {
            "mean": mean,
            "cov": cov,
            "max_abs_mean": float(np.max(np.abs(mean))),
            "max_abs_cov_minus_I": float(np.max(np.abs(cov - np.eye(self.n_dim)))),
        }


def fit_transport_map(parameter_samples, max_order=2, param_names=None,
                      map_options=None, gtol=1e-3, maxiter=500, verbose=False):
    """Fit a triangular transport map pushing samples toward a standard Gaussian.

    Maximises the map-induced log-likelihood

        sum_i [ log rho(S(x_i)) + log|det grad S(x_i)| ]

    over the map coefficients, with rho the standard normal density. This is the
    standard MParT density-estimation objective.

    Args:
        parameter_samples: (n_samples, n_dim) posterior parameter ensemble.
        max_order:         total polynomial order of the map. 1 is affine
                           (fast, captures only mean/covariance); 2-3 captures
                           skew and mild nonlinearity. Cost grows quickly with
                           dimension, so prefer 2 for 7-D problems.
        param_names:       optional names, carried on the result for bookkeeping.
        map_options:       optional mpart.MapOptions; a default is used if None.
        gtol, maxiter:     BFGS stopping controls.
        verbose:           print optimizer progress and fit diagnostics.

    Returns:
        TransportMapResult

    Raises:
        ImportError: if mpart is not installed.
        ValueError:  if the sample array is malformed or too small.
    """
    _require_mpart()

    X = np.asarray(parameter_samples, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"Expected (n_samples, n_dim), got shape {X.shape}.")
    n_samples, n_dim = X.shape
    if n_samples <= n_dim:
        raise ValueError(
            f"Need more samples than dimensions to fit a map; got "
            f"n_samples={n_samples}, n_dim={n_dim}.")

    # Degenerate directions (a parameter collapsed to a single value) make the
    # log-determinant term singular. Surface that clearly rather than letting
    # BFGS wander.
    stds = X.std(axis=0)
    degenerate = np.where(stds < 1e-12)[0]
    if degenerate.size:
        names = ([param_names[i] for i in degenerate] if param_names is not None
                 else degenerate.tolist())
        raise ValueError(
            f"Parameter(s) {names} have (near-)zero variance across the ensemble; "
            "a transport map cannot be fitted. This usually means particle "
            "degeneracy collapsed the posterior.")

    X_mp = _as_mpart(X)
    rho = multivariate_normal(np.zeros(n_dim), np.eye(n_dim))

    if map_options is None:
        map_options = mt.MapOptions()
    tri_map = mt.CreateTriangular(n_dim, n_dim, max_order, map_options)

    def objective(coeffs):
        tri_map.SetCoeffs(coeffs)
        z = tri_map.Evaluate(X_mp)                 # (n_dim, n_samples)
        log_det = tri_map.LogDeterminant(X_mp)     # (n_samples,)
        return -np.sum(rho.logpdf(z.T) + log_det) / n_samples

    def gradient(coeffs):
        tri_map.SetCoeffs(coeffs)
        z = tri_map.Evaluate(X_mp)
        # d/dcoeffs of log rho(S(x)) is -S(x) contracted with the coeff jacobian
        grad_rho = -tri_map.CoeffGrad(X_mp, z)
        grad_log_det = tri_map.LogDeterminantCoeffGrad(X_mp)
        return -np.sum(grad_rho + grad_log_det, axis=1) / n_samples

    coeffs_init = tri_map.CoeffMap()
    res = minimize(objective, coeffs_init, jac=gradient, method="BFGS",
                   options={"gtol": gtol, "maxiter": maxiter, "disp": verbose})
    tri_map.SetCoeffs(res.x)

    result = TransportMapResult(tri_map, n_dim, max_order, res.x,
                                param_names=param_names, optimizer=res)
    if verbose:
        d = result.diagnostics(X)
        print(f"[mpart_transport] converged={res.success} "
              f"nit={getattr(res, 'nit', '?')} obj={res.fun:.6f}")
        print(f"[mpart_transport] pushforward |mean|max={d['max_abs_mean']:.4f} "
              f"|cov-I|max={d['max_abs_cov_minus_I']:.4f}")
    return result


def forward(fitted, parameter_samples):
    """Push samples from the target (posterior) to the reference (standard Gaussian).

    Args:
        fitted:            TransportMapResult from fit_transport_map.
        parameter_samples: (n_samples, n_dim) in the target space.

    Returns:
        (n_samples, n_dim) array in the standard-Gaussian reference space,
        suitable as PCE inputs with a probabilist-Hermite basis.
    """
    _require_mpart()
    X_mp = _as_mpart(parameter_samples)
    if X_mp.shape[0] != fitted.n_dim:
        raise ValueError(
            f"Sample dimension {X_mp.shape[0]} does not match map dimension {fitted.n_dim}.")
    return _as_pipeline(fitted.tri_map.Evaluate(X_mp))


def inverse(fitted, reference_samples):
    """Pull samples from the reference (standard Gaussian) back to the target.

    Needed when a PCE built in z-space must be interpreted, or evaluated, at
    parameter values in the original physical space.

    Args:
        fitted:            TransportMapResult from fit_transport_map.
        reference_samples: (n_samples, n_dim) standard-Gaussian points.

    Returns:
        (n_samples, n_dim) array in the target (parameter) space.
    """
    _require_mpart()
    Z_mp = _as_mpart(reference_samples)
    if Z_mp.shape[0] != fitted.n_dim:
        raise ValueError(
            f"Sample dimension {Z_mp.shape[0]} does not match map dimension {fitted.n_dim}.")
    # MParT's triangular Inverse takes (prefix, rhs); for a square map the
    # prefix is the point itself and is ignored beyond providing the shape.
    return _as_pipeline(fitted.tri_map.Inverse(Z_mp, Z_mp))
