import numpy as np
from scipy.optimize import OptimizeResult
from scipy.spatial.distance import cdist
from itertools import combinations
from functools import partial

def nchoosek(nn, kk):
    try:  # SciPy >= 0.19
        from scipy.special import comb
    except ImportError:
        from scipy.misc import comb
    result = np.asarray(np.round(comb(nn, kk)), dtype=int)
    if result.ndim == 0:
        result = result.item()
        # result = np.asscalar(result)
    return result

# ==========================================================

def get_morris_trajectory(nvars, nlevels, eps=0):
    r"""
    Compute a morris trajectory used to compute elementary effects

    Parameters
    ----------
    nvars : integer
        The number of variables

    nlevels : integer
        The number of levels used for to define the morris grid.

    eps : float
        Set grid used defining the morris trajectory to [eps,1-eps].
        This is needed when mapping the morris trajectories using inverse
        CDFs of unbounded variables

    Returns
    -------
    trajectory : np.ndarray (nvars,nvars+1)
        The Morris trajectory which consists of nvars+1 samples
    """
    assert nlevels % 2 == 0
    delta = nlevels/((nlevels-1)*2)
    samples_1d = np.linspace(eps, 1-eps, nlevels)

    initial_point = np.random.choice(samples_1d, nvars)
    shifts = np.diag(np.random.choice([-delta, delta], nvars))
    trajectory = np.empty((nvars, nvars+1))
    trajectory[:, 0] = initial_point
    for ii in range(nvars):
        trajectory[:, ii+1] = trajectory[:, ii].copy()
        if (trajectory[ii, ii]-delta) >= 0 and (trajectory[ii, ii]+delta) <= 1:
            trajectory[ii, ii+1] += shifts[ii]
        elif (trajectory[ii, ii]-delta) >= 0:
            trajectory[ii, ii+1] -= delta
        elif (trajectory[ii, ii]+delta) <= 1:
            trajectory[ii, ii+1] += delta
        else:
            raise Exception('This should not happen')
    return trajectory


def get_morris_samples(nvars, nlevels, ntrajectories, eps=0, icdfs=None):
    r"""
    Compute a set of Morris trajectories used to compute elementary effects

    Notes
    -----
    The choice of nlevels must be linked to the choice of ntrajectories.
    For example, if a large number of possible levels is used ntrajectories
    must also be high, otherwise if ntrajectories is small effort will be
    wasted because many levels will not be explored. nlevels=4 and
    ntrajectories=10 is often considered reasonable.

    Parameters
    ----------
    nvars : integer
        The number of variables

    nlevels : integer
        The number of levels used for to define the morris grid.

    ntrajectories : integer
        The number of Morris trajectories requested

    eps : float
        Set grid used defining the Morris trajectory to [eps,1-eps].
        This is needed when mapping the morris trajectories using inverse
        CDFs of unbounded variables

    icdfs : list (nvars)
        List of inverse CDFs functions for each variable

    Returns
    -------
    trajectories : np.ndarray (nvars,ntrajectories*(nvars+1))
        The Morris trajectories
    """
    if icdfs is None:
        icdfs = [lambda x: x]*nvars
    assert len(icdfs) == nvars

    trajectories = np.hstack([get_morris_trajectory(nvars, nlevels, eps)
                              for n in range(ntrajectories)])
    for ii in range(nvars):
        trajectories[ii, :] = icdfs[ii](trajectories[ii, :])
    return trajectories


def get_morris_elementary_effects(samples, values):
    r"""
    Get the Morris elementary effects from a set of trajectories.

    Parameters
    ----------
    samples : np.ndarray (nvars,ntrajectories*(nvars+1))
        The morris trajectories

    values : np.ndarray (ntrajectories*(nvars+1),nqoi)
        The values of the vecto-valued target function with nqoi quantities
        of interest (QoI)

    Returns
    -------
    elem_effects : np.ndarray(nvars,ntrajectories,nqoi)
        The elementary effects of each variable for each trajectory and QoI
    """
    nvars = samples.shape[0]
    nqoi = values.shape[1]
    assert samples.shape[1] % (nvars+1) == 0
    assert samples.shape[1] == values.shape[0]
    ntrajectories = samples.shape[1]//(nvars+1)
    elem_effects = np.empty((nvars, ntrajectories, nqoi))
    ix1 = 0
    for ii in range(ntrajectories):
        ix2 = ix1+nvars
        delta = np.diff(samples[:, ix1+1:ix2+1]-samples[:, ix1:ix2]).max()
        assert delta > 0
        elem_effects[:, ii] = (values[ix1+1:ix2+1]-values[ix1:ix2])/delta
        ix1 = ix2+1
    return elem_effects


def get_morris_sensitivity_indices(elem_effects):
    r"""
    Compute the Morris sensitivity indices mu and sigma from the elementary
    effects computed for a set of trajectories.

    Mu is the mu^\star from Campolongo et al.

    Parameters
    ----------
    elem_effects : np.ndarray(nvars,ntrajectories,nqoi)
        The elementary effects of each variable for each trajectory and
        quantity of interest (QoI)

    Returns
    -------
    mu : np.ndarray (nvars, nqoi)
        The sensitivity of each output to each input. Larger mu corresponds to
        higher sensitivity

    sigma: np.ndarray(nvars, nqoi)
        A measure of the non-linearity and/or interaction effects of each input
        for each output. Low values suggest a linear realationship between
        the input and output. Larger values suggest a that the output is
        nonlinearly dependent on the input and/or the input interacts with
        other inputs
    """
    mu = np.absolute(elem_effects).mean(axis=1)
    assert mu.shape == (elem_effects.shape[0], elem_effects.shape[2])
    sigma = np.std(elem_effects, axis=1)
    return mu, sigma


def print_morris_sensitivity_indices(mu, sigma, qoi=0):
    str_format = "{:<3} {:>10} {:>10}"
    print(str_format.format(" ", "mu*", "sigma"))
    str_format = "{:<3} {:10.5f} {:10.5f}"
    for ii in range(mu.shape[0]):
        print(str_format.format(f'Z_{ii+1}', mu[ii, qoi], sigma[ii, qoi]))


def downselect_morris_trajectories(samples, ntrajectories):
    nvars = samples.shape[0]
    assert samples.shape[1] % (nvars+1) == 0
    ncandidate_trajectories = samples.shape[1]//(nvars+1)
    # assert 10*ntrajectories<=ncandidate_trajectories

    trajectories = np.reshape(
        samples, (nvars, nvars+1, ncandidate_trajectories), order='F')

    distances = np.zeros((ncandidate_trajectories, ncandidate_trajectories))
    for ii in range(ncandidate_trajectories):
        for jj in range(ii+1):
            distances[ii, jj] = cdist(
                trajectories[:, :, ii].T, trajectories[:, :, jj].T).sum()
            distances[jj, ii] = distances[ii, jj]

    get_combinations = combinations(
        np.arange(ncandidate_trajectories), ntrajectories)
    ncombinations = nchoosek(ncandidate_trajectories, ntrajectories)
    print('ncombinations', ncombinations)
    # values = np.empty(ncombinations)
    best_index = None
    best_value = -np.inf
    for ii, index in enumerate(get_combinations):
        value = np.sqrt(np.sum(
            [distances[ix[0], ix[1]]**2 for ix in combinations(index, 2)]))
        if value > best_value:
            best_value = value
            best_index = index

    samples = trajectories[:, :, best_index].reshape(
        nvars, ntrajectories*(nvars+1), order='F')
    return samples


class SensitivityResult(OptimizeResult):
    pass


def morris_sensitivities(fun, variable, ntrajectories,
                         nlevels=4):
    r"""
    Compute sensitivity indices by constructing an adaptive polynomial chaos
    expansion.

    Parameters
    ----------
    fun : callable
        The function being analyzed

        ``fun(z) -> np.ndarray``

        where ``z`` is a 2D np.ndarray with shape (nvars,nsamples) and the
        output is a 2D np.ndarray with shape (nsamples,nqoi)

    variable : :py:class:`pyapprox.variables.IndependentMarginalsVariable`
         Object containing information of the joint density of the inputs z
         which is the tensor product of independent and identically distributed
         uniform variables.

    ntrajectories : integer
        The number of Morris trajectories requested


    nlevels : integer
        The number of levels used for to define the morris grid.

    Returns
    -------
    result : :class:`pyapprox.analysis.sensitivity_analysis.SensitivityResult`
         Result object with the following attributes

    mu : np.ndarray (nvars,nqoi)
        The sensitivity of each output to each input. Larger mu corresponds to
        higher sensitivity

    sigma: np.ndarray (nvars,nqoi)
        A measure of the non-linearity and/or interaction effects of each input
        for each output. Low values suggest a linear realationship between
        the input and output. Larger values suggest a that the output is
        nonlinearly dependent on the input and/or the input interacts with
        other inputs

    samples : np.ndarray(nvars,ntrajectories*(nvars+1))
        The coordinates of each morris trajectory

    values : np.ndarray(nvars,nqoi)
        The values of ``fun`` at each sample in ``samples``
    """

    nvars = variable.num_vars()
    icdfs = [v.ppf for v in variable.marginals()]
    samples = get_morris_samples(nvars, nlevels, ntrajectories, icdfs=icdfs)
    values = fun(samples)
    elem_effects = get_morris_elementary_effects(samples, values)
    mu, sigma = get_morris_sensitivity_indices(elem_effects)

    return SensitivityResult(
        {'morris_mu': mu,
         'morris_sigma': sigma,
         'samples': samples, 'values': values})