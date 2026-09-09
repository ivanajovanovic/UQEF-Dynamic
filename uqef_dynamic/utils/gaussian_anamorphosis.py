"""Gaussian anamorphosis: rank-based marginal transform to a standard Gaussian.

    z = G^{-1}(F(y))

with F the empirical CDF of the samples and G the standard normal CDF
(Fan et al. 2016, Appendix A1; Simon & Bertino 2009). Used here for the same
purpose as the MParT transport map — turning particle-filter posterior parameter
samples into standard-Gaussian inputs for a Hermite PCE — and exposing the same
forward/inverse interface, so either backend can drive transport_timeseries.

How it compares to a triangular transport map:

  + Rank-based, so it is indifferent to marginal shape. It handles the spiky,
    high-kurtosis, heavily clustered marginals a particle filter produces, which
    a low-order polynomial map cannot.
  + ~500x faster (a sort, no optimisation) and it cannot fail to converge.
  + No dependency on mpart.
  - It is a MARGINAL transform: each marginal becomes standard normal by
    construction, but cross-correlations are left untouched. A joint transport
    map whitens them; this does not. If the PCE basis relies on the inputs being
    jointly independent, that difference matters.
  - The inverse is only defined over the sampled range; z beyond the training
    support clamps to the extreme sample values rather than extrapolating.

Because the marginals are standard normal BY CONSTRUCTION, a marginal normality
test on the output of this transform is uninformative. Judge it on the joint
diagnostics (off-diagonal correlation) instead.
"""

import numpy as np
from scipy import stats


__all__ = ["AnamorphosisResult", "fit_anamorphosis", "forward", "inverse"]


class AnamorphosisResult:
    """A fitted anamorphosis. Mirrors mpart_transport.TransportMapResult."""

    def __init__(self, sorted_samples, plotting_positions, param_names=None,
                 tie_handling="average"):
        self.sorted_samples = sorted_samples          # (n_samples, n_dim), sorted per column
        self.plotting_positions = plotting_positions  # (n_samples,)
        self.n_dim = sorted_samples.shape[1]
        self.param_names = list(param_names) if param_names is not None else None
        self.tie_handling = tie_handling

    def forward(self, X):
        return forward(self, X)

    def inverse(self, Z):
        return inverse(self, Z)

    def diagnostics(self, X):
        """Joint diagnostics. Marginal normality is automatic here, so the
        informative numbers are the covariance and off-diagonal correlation."""
        Z = self.forward(X)
        mean = Z.mean(axis=0)
        cov = np.atleast_2d(np.cov(Z, rowvar=False))
        corr = np.atleast_2d(np.corrcoef(Z, rowvar=False))
        off = np.abs(corr - np.eye(self.n_dim))
        return {
            "mean": mean,
            "cov": cov,
            "max_abs_mean": float(np.max(np.abs(mean))),
            "max_abs_cov_minus_I": float(np.max(np.abs(cov - np.eye(self.n_dim)))),
            "max_abs_offdiag_corr": float(np.max(off)) if self.n_dim > 1 else 0.0,
        }


def fit_anamorphosis(samples, param_names=None, tie_handling="average"):
    """Fit a per-parameter empirical-CDF transform.

    Args:
        samples:       (n_samples, n_dim) posterior parameter ensemble.
        param_names:   optional names, carried on the result.
        tie_handling:  "average" gives tied samples the same rank, hence the same
                       z (standard, matches Fan et al.). "ordinal" breaks ties by
                       position, giving distinct z values — useful when a
                       downstream step cannot cope with duplicates, at the cost of
                       an arbitrary ordering within a tie.

    Returns:
        AnamorphosisResult
    """
    X = np.asarray(samples, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"Expected (n_samples, n_dim), got shape {X.shape}.")
    n = X.shape[0]
    if n < 2:
        raise ValueError(f"Need at least 2 samples, got {n}.")
    if tie_handling not in ("average", "ordinal"):
        raise ValueError("tie_handling must be 'average' or 'ordinal'.")
    return AnamorphosisResult(np.sort(X, axis=0), (np.arange(n) + 0.5) / n,
                              param_names=param_names, tie_handling=tie_handling)


def forward(fitted, samples):
    """theta -> z. Ranks the samples and maps through the normal quantile function."""
    X = np.asarray(samples, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != fitted.n_dim:
        raise ValueError(f"Expected (n_samples, {fitted.n_dim}), got {X.shape}.")
    r = stats.rankdata(X, method=fitted.tie_handling, axis=0)
    return stats.norm.ppf((r - 0.5) / X.shape[0])


def inverse(fitted, reference_samples):
    """z -> theta, by interpolating the stored empirical quantile function.

    Values of z outside the training support clamp to the smallest/largest
    training sample; the empirical CDF carries no information beyond its range.
    """
    Z = np.asarray(reference_samples, dtype=np.float64)
    if Z.ndim != 2 or Z.shape[1] != fitted.n_dim:
        raise ValueError(f"Expected (n_samples, {fitted.n_dim}), got {Z.shape}.")
    p = stats.norm.cdf(Z)
    out = np.empty_like(p)
    for j in range(fitted.n_dim):
        out[:, j] = np.interp(p[:, j], fitted.plotting_positions,
                              fitted.sorted_samples[:, j])
    return out
