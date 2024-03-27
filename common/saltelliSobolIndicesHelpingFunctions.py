import numpy as np
np.random.seed(10)

# Functions needed for calculating Sobol indices using MC samples and Saltelli method


def rank_stats(f, N, dim):
    """
    This function only estimates global Sobol indices
    Example how Ravi calls the function - mean, var, sobol = rank_stats(ishigami.hf, N, dim)
    :param f: model to be evaluated
    :param N: number of samples
    :param dim: stochastic dimensionality
    :return:
    """
    samples = np.random.uniform(0, 1, (N, dim))
    Y = f(samples)
    mean = np.mean(Y)
    var = np.var(Y, ddof=1)
    sobol = np.zeros(dim)
    # N_j = np.zeros((N, dim), dtype=np.int32)
    mean_of_square = np.mean(Y**2)
    px = samples.argsort(axis=0)
    pi_j = px.argsort(axis=0) + 1
    argpiinv = (pi_j % N) + 1
    # print(pi_j, argpiinv)
    for i in range(dim):
        N_j = px[argpiinv[:, i] - 1, i]
        # print(i, N_j)
        YN_j = Y[N_j]
        sobol[i] = (np.mean(Y*YN_j) - mean**2)/var
    return mean, var, sobol


def compute_total_sobol_indices_with_n_samples(samples, Y, D, N):
    """
    : param samples: UQEF.Samples.parameters; should be of the size NxD
    :param Y:  function evaluations dim(Y) = (N x t)
    :param D: Stochastic dimension
    :param N: Number of samples
    :return:
    """
    mean = np.mean(Y, axis=0)
    variance = np.var(Y, axis=0, ddof=1)
    denominator = np.where(variance, variance, 1)

    px = samples.argsort(axis=0)  # samples are NxD; this will sort each column and return indices
    pi_j = px.argsort(axis=0) + 1
    argpiinv = (pi_j % N) + 1

    # s_t = np.zeros(D)
    s_t = []
    for j in range(D):
        # numerator = np.mean((fA - fAB[j]) ** 2, axis=0) / 2  # np.mean((fA-fAB[j].T)**2, -1)
        N_j = px[argpiinv[:, j] - 1, j]
        YN_j = Y[N_j, :]
        # print(f"DEBUGGING - shape of YN_j - {YN_j.shape}")
        numerator = (np.mean(Y * YN_j, axis=0) - mean**2)
        s_t_j = numerator[0] / denominator[0]
        s_t.append(s_t_j)
    return np.asfarray(s_t)


def _power(my_list):
    return [ x**2 for x in my_list ]


def _separate_output_values(Y, D, N):
    """
    Input:
    Y - function evaluations in A, B and fAB points
    dim(Y) = (N*(2+D) x t)
    D - Stochastic dimension
    N - Number of samples
    Return:
    fA - function evaluations based on m1 Nxt
    fB - function evaluations based on m2 Nxt
    fAB - array of function evaluations based on m1 with some rows from m2,
    len(fAB) = D;
    """
    fA = Y[0:N, :]
    fB = Y[N:2*N, :]

    fAB = []
    for j in range(D):
        start = (j + 2)*N
        end = start + N
        temp = np.array(Y[start:end, :])
        fAB.append(temp)

    return fA, fB, fAB


def _separate_output_values_j(Y, D, N, j):
    """
    Input:
        Y - function evaluations in fA, fB and fAB points
    dim(Y) = (N*(2+D) x t)
    D - Stochastic dimension
    N - Number of samples
    j - in range(0,D)
    Return:
    fA - function evaluations based on m1 Nxt
    fB - function evaluations based on m2 Nxt
    fAB - function evaluations based on m1 with jth row from m2
    """

    fA = Y[0:N, :]
    fB = Y[N:2*N, :]
    start = (j + 2)*N
    end = start + N
    fAB = Y[start:end, :]

    return fA, fB, fAB

# ====================================================
# Functions for computing first and total order sensitivity indices based on MC samples -
# required number of total model runs is N*(2+dim)
# ====================================================


def _Sens_m_sample(Y, D, N, code=4, do_printing=False):
    """
    First order sensitivity indices
    code ==
    1 - Chaospy
    2 - Homma(1996) & Sobolo (2007)
    3 - Saltelli 2010
    4 - Jensen.
    """
    fA, fB, fAB = _separate_output_values(Y, D, N)

    if do_printing:
        print(f"fA shape: {fA.shape}")
        print(f"fB shape: {fB.shape}")
        print(f"fAB len: {len(fAB)}")
        print(f"fAB[0].shape: {fAB[0].shape}")

    # variance = np.var(fA, -1)
    variance = np.var(fA, axis=0, ddof=1)
    denominator = np.where(variance, variance, 1)

    if do_printing:
        print(f"variance.shape: {variance.shape}")
        print(f"variance: {variance}")
        print(f"denominator.shape: {denominator.shape}")
        print(f"denominator: {denominator}")

    if code == 1 or code == 2:
        # mean = .5*(np.mean(fA) + np.mean(fB))
        mean = .5 * (np.mean(fA, axis=0) + np.mean(fB, axis=0))

    if code == 1:
        fA -= mean
        fB -= mean

    # 1: np.mean(fB*((fAB[j].T-mean)-fA), -1)/denominator or np.mean(fB*((fAB[j]-mean)-fA),axis=0)/denominator
    # 2: (np.mean(fB*fAB[j].T, -1) - mean**2)/denominator or (np.mean(fB*fAB[j], axis=0) - mean**2)/denominator
    # 3: np.mean(fB*(fAB[j].T-fA), -1)/denominator or np.mean(fB*(fAB[j]-fA), axis=0)/denominator
    # 4: s_i_j = 1 - np.mean((fAB[j]-fB)**2, axis=0)/(2*denominator) or 1 - np.mean((fAB[j].T-fB)**2, -1)/(2*denominator)
    s_i = []
    for j in range(D):
        if code == 1:
            numerator = np.mean(fB * ((fAB[j] - mean) - fA), axis=0)
        elif code == 2:
            numerator = np.mean(fB * fAB[j], axis=0) - mean ** 2
        elif code == 3:
            # np.dot(fB, (fAB[j]-fA))
            numerator = np.mean(fB * (fAB[j] - fA), axis=0)
        elif code == 4:
            temp = np.mean((fAB[j] - fB) ** 2, axis=0) / 2
            numerator = denominator - temp
        else:
            raise
        s_i_j = numerator[0] / denominator[0]
        s_i.append(s_i_j)

    return np.asfarray(s_i)


def _Sens_t_sample(Y, D, N, code=4, do_printing=False):
    """
    First order sensitivity indices
    code ==
    1 - Chaospy
    2 - Homma 1996
    3 - Sobol 2007
    4 - Saltelli 2010 & Jensen
    """
    fA, fB, fAB = _separate_output_values(Y, D, N)

    if do_printing:
        print(f"fA shape: {fA.shape}")
        print(f"fB shape: {fB.shape}")
        print(f"fAB len: {len(fAB)}")
        print(f"fAB[0].shape: {fAB[0].shape}")

    # variance = np.var(fA, -1)
    variance = np.var(fA, axis=0, ddof=1)
    denominator = np.where(variance, variance, 1)

    if do_printing:
        print(f"variance.shape: {variance.shape}")
        print(f"variance: {variance}")
        print(f"denominator.shape: {denominator.shape}")
        print(f"denominator: {denominator}")

    if code == 2:
        # mean = .5*(np.mean(fA) + np.mean(fB))
        mean = .5 * (np.mean(fA, axis=0) + np.mean(fB, axis=0))

    s_t = []
    for j in range(D):
        if code == 1:
            # TODO Note - this version sometimes gives negative values
            numerator = denominator - np.mean((fAB[j]-B)**2, axis=0)/2  # denominator - np.mean((fAB[j].T-fB)**2, -1)/2
        elif code == 2:
            numerator = denominator - (np.mean(fA*fAB[j], axis=0) - mean**2)  # (np.mean(fA*fAB[j].T, -1) - mean**2)
        elif code == 3:
            numerator = np.mean(fA*(A-fAB[j]),  axis=0)  # np.mean(fA*(A-fAB[j].T), -1)
        elif code == 4:
            numerator = np.mean((fA-fAB[j])**2, axis=0)/2  # np.mean((fA-fAB[j].T)**2, -1)
        else:
            raise
        s_t_j = numerator[0] / denominator[0]
        s_t.append(s_t_j)

    return np.asfarray(s_t)