import numpy as np

#helper function
def _power(my_list):
    return [ x**2 for x in my_list ]

# Functions needed for calculating Sobol indices using MC samples and Saltelli method


def _separate_output_values(Y, D, N):
    """
    Input:
    dim(Y) = (N*(2+D) x t)
    D - Stochastic dimension
    N - Numer of samples
    Return:
    A - function evaluations based on m1 Nxt
    B - function evaluations based on m2 Nxt
    A_B - array of function evaluations based on m1 with some rows from m2,
    len(A_B) = D;
    """
    A = Y[0:N,:]
    B = Y[N:2*N,:]

    A_B = []
    for j in range(D):
        start = (j + 2)*N
        end = start + N
        temp = np.array(Y[start:end, :])
        A_B.append(temp)

    #return A.T, B.T, A_B
    return A, B, A_B


def _separate_output_values_j(Y, D, N, j):
    """
    Input:
    dim(Y) = (N*(2+D) x t)
    D - Stochastic dimension
    N - Numer of samples
    j - in range(0,D)
    Return:
    A - function evaluations based on m1 Nxt
    B - function evaluations based on m2 Nxt
    A_B - function evaluations based on m1 with jth row from m2
    """

    A = Y[0:N, :]
    B = Y[N:2*N, :]
    start = (j + 2)*N
    end = start + N
    A_B = Y[start:end, :]

    return A, B, A_B

#####################################


def _Sens_m_sample(Y, D, N, code=4, do_printing=False):
    """
    First order sensitivity indices
    code ==
    1 - Chaospy
    2 - Homma(1996) & Sobolo (2007)
    3 - Saltelli 2010
    4 - Jensen.
    """
    A, B, A_B = _separate_output_values(Y, D, N)

    if do_printing:
        print(f"A shape: {A.shape}")
        print(f"b shape: {B.shape}")
        print(f"A_B len: {len(A_B)}")
        print(f"A_B[0].shape: {A_B[0].shape}")

    # variance = np.var(A, -1)
    variance = np.var(A, axis=0, ddof=1)
    denumerator = np.where(variance, variance, 1)

    if do_printing:
        print(f"variance.shape: {variance.shape}")
        print(f"variance: {variance}")
        print(f"denumerator.shape: {denumerator.shape}")
        print(f"denumerator: {denumerator}")

    if code == 1 or code == 2:
        # mean = .5*(np.mean(A) + np.mean(B))
        mean = .5 * (np.mean(A, axis=0) + np.mean(B, axis=0))

    if code == 1:
        A -= mean
        B -= mean

    # 1: np.mean(B*((A_B[j].T-mean)-A), -1)/denumerator or np.mean(B*((A_B[j]-mean)-A),axis=0)/denumerator
    # 2: (np.mean(B*A_B[j].T, -1) - mean**2)/denumerator or (np.mean(B*A_B[j], axis=0) - mean**2)/denumerator
    # 3: np.mean(B*(A_B[j].T-A), -1)/denumerator or np.mean(B*(A_B[j]-A), axis=0)/denumerator
    # 4: s_i_j = 1 - np.mean((A_B[j]-B)**2, axis=0)/(2*denumerator) or 1 - np.mean((A_B[j].T-B)**2, -1)/(2*denumerator)
    s_i = []
    for j in range(D):
        if code == 1:
            numerator = np.mean(B * ((A_B[j] - mean) - A), axis=0)
        elif code == 2:
            numerator = np.mean(B * A_B[j], axis=0) - mean ** 2
        elif code == 3:
            # np.dot(B, (A_B[j]-A))
            numerator = np.mean(B * (A_B[j] - A), axis=0)
        elif code == 4:
            temp = np.mean((A_B[j] - B) ** 2, axis=0) / 2
            numerator = denumerator - temp
        else:
            raise
        s_i_j = numerator[0] / denumerator[0]
        s_i.append(s_i_j)

    return np.array(s_i)


def _Sens_t_sample(Y, D, N, code=4, do_printing=False):
    """
    First order sensitivity indices
    code ==
    1 - Chaospy
    2 - Homma 1996
    3 - Sobol 2007
    4 - Saltelli 2010 & Jensen
    """
    A, B, A_B = _separate_output_values(Y, D, N)

    if do_printing:
        print(f"A shape: {A.shape}")
        print(f"b shape: {B.shape}")
        print(f"A_B len: {len(A_B)}")
        print(f"A_B[0].shape: {A_B[0].shape}")

    # variance = np.var(A, -1)
    variance = np.var(A, axis=0, ddof=1)
    denumerator = np.where(variance, variance, 1)

    if do_printing:
        print(f"variance.shape: {variance.shape}")
        print(f"variance: {variance}")
        print(f"denumerator.shape: {denumerator.shape}")
        print(f"denumerator: {denumerator}")

    if code == 2:
        # mean = .5*(np.mean(A) + np.mean(B))
        mean = .5 * (np.mean(A, axis=0) + np.mean(B, axis=0))

    s_t = []
    for j in range(D):
        if code == 1:
            # TODO this gives negative sometimes
            numerator = denumerator - np.mean((A_B[j]-B)**2, axis=0)/2  # denumerator - np.mean((A_B[j].T-B)**2, -1)/2
        elif code == 2:
            numerator = denumerator - (np.mean(A*A_B[j], axis=0) - mean**2)  # (np.mean(A*A_B[j].T, -1) - mean**2)
        elif code == 3:
            numerator = np.mean(A*(A-A_B[j]),  axis=0)  # np.mean(A*(A-A_B[j].T), -1)
        elif code == 4:
            numerator = np.mean((A-A_B[j])**2, axis=0)/2  # np.mean((A-A_B[j].T)**2, -1)
        else:
            raise
        s_t_j = numerator[0] / denumerator[0]
        s_t.append(s_t_j)

    return np.array(s_t)

#####################################


def _Sens_m_sample_1(Y, D, N):
    """
    First order sensitivity indices - Chaospy
    """
    A, B, A_B = _separate_output_values(Y, D, N)

    #variance = np.var(A, -1)
    variance = np.var(A, axis=0)

    mean = .5*(np.mean(A) + np.mean(B))
    #mean = .5*(np.mean(A, axis=0) + np.mean(B, axis=0))
    A -= mean
    B -= mean

    out = [
        #np.mean(B*((A_B[j].T-mean)-A), -1) /
        np.mean(B*((A_B[j]-mean)-A),axis=0) /
        np.where(variance, variance, 1)
        for j in range(D)
        ]

    return np.array(out)


def _Sens_m_sample_2(Y, D, N):
    """
    First order sensitivity indices - Homma(1996) & Sobolo (2007)
    """
    A, B, A_B = _separate_output_values(Y, D, N)

    #variance = np.var(A, -1)
    variance = np.var(A, axis=0)

    #mean = .5*(np.mean(A) + np.mean(B))
    mean = .5*(np.mean(A, axis=0) + np.mean(B, axis=0))

    out = [
        #(np.mean(B*A_B[j].T, -1) - mean**2) /
        (np.mean(B*A_B[j], axis=0) - mean**2) /
        np.where(variance, variance, 1)
        for j in range(D)
        ]

    return np.array(out)


def _Sens_m_sample_3(Y, D, N):
    """
    First order sensitivity indices - Saltelli 2010.
    """
    A, B, A_B = _separate_output_values(Y, D, N)

    #variance = np.var(A, -1)
    variance = np.var(A, axis=0, ddof=1)
    denumerator = np.where(variance, variance, 1)

    #out = [
    #    #np.mean(B*(A_B[j].T-A), -1) /
    #    np.mean(B*(A_B[j]-A), axis=0) /
    #    np.where(variance, variance, 1)
    #    for j in range(D)
    #    ]
    s_i = []
    for j in range(D):
        #np.dot(B, (A_B[j]-A))
        numerator = np.mean(B*(A_B[j]-A), axis=0)
        s_i_j = numerator[0] / denumerator[0]
        s_i.append(s_i_j)

    return np.array(s_i)


def _Sens_m_sample_4(Y, D, N):
    """
    First order sensitivity indices - Jensen.
    """
    A, B, A_B = _separate_output_values(Y, D, N)

    #variance = np.var(A, -1)
    variance = np.var(A, axis=0, ddof=1)
    denumerator = np.where(variance, variance, 1)

    s_i = []
    print(f"D={D}")
    for j in range(D):
        temp = np.mean((A_B[j]-B)**2, axis=0)/2
        s_i_j = (denumerator[0] - temp[0]) / denumerator[0]
        s_i.append(s_i_j)

    return np.array(s_i)


#####################################
#####################################

def _Sens_t_sample_1(Y, D, N):
    """
    Total order sensitivity indices - Chaospy
    """
    A, B, A_B = _separate_output_values(Y, D, N)

    #variance = np.var(A, -1)
    variance = np.var(A, axis=0)

    out = [
        #1-np.mean((A_B[j].T-B)**2, -1) /
        1-np.mean((A_B[j]-B)**2, axis=0) /
        (2*np.where(variance, variance, 1))
        for j in range(D)
        ]

    return np.array(out)


def _Sens_t_sample_2(Y, D, N):
    """
    Total order sensitivity indices - Homma 96
    """
    A, B, A_B = _separate_output_values(Y, D, N)

    #mean = .5*(np.mean(A) + np.mean(B))
    mean = .5*(np.mean(A, axis=0) + np.mean(B, axis=0))

    #variance = np.var(A, -1)
    variance = np.var(A, axis=0)

    out = [
        #1-(np.mean(A*A_B[j].T, -1) - mean**2)/
        1-(np.mean(A*A_B[j], axis=0) - mean**2)/
        np.where(variance, variance, 1)
        for j in range(D)
    ]

    return np.array(out)


def _Sens_t_sample_3(Y, D, N):
    """
    Total order sensitivity indices - Sobel 2007
    """
    A, B, A_B = _separate_output_values(Y, D, N)

    #variance = np.var(A, -1)
    variance = np.var(A, axis=0, ddof=1)
    print(f"variance.shape: {variance.shape}")
    print(f"variance: {variance}")

    denumerator = np.where(variance, variance, 1)
    print(f"denumerator.shape: {denumerator.shape}")
    print(f"denumerator: {denumerator}")

    out = [
        #np.mean(A*(A-A_B[j].T), -1) /
        np.mean(A*(A-A_B[j]),  axis=0) /
        np.where(variance, variance, 1)
        for j in range(D)
        ]

    return np.array(out)


def _Sens_t_sample_4(Y, D, N):
    """
    Total order sensitivity indices - Saltelli 2010 & Jensen
    """
    A, B, A_B = _separate_output_values(Y, D, N)

    #variance = np.var(A, -1)
    #variance = np.var(A, -1)
    variance = np.var(A, axis=0, ddof=1)
    print(f"variance.shape: {variance.shape}")
    print(f"variance: {variance}")

    denumerator = np.where(variance, variance, 1)
    print(f"denumerator.shape: {denumerator.shape}")
    print(f"denumerator: {denumerator}")

    s_t = []
    for j in range(D):
        temp = np.mean((A-A_B[j])**2, axis=0)  # np.mean((A-A_B[j].T)**2, -1)
        s_t_j = temp[0] / (2*denumerator[0])
        s_t.append(s_t_j)

    return np.array(s_t)

#####################################
# Old code
#####################################


def _separate_output_values_old(Y, D, N):
    """
    Return:
    A - function evaluations based on m1 Nxt
    B - function evaluations based on m2 Nxt
    A_B - function evaluations based on m1 with some rows from m2 Nxt
    """

    A_B = np.zeros((N, D))

    A = np.tile(Y[0:N], (D,1))
    A = A.T

    B = np.tile(Y[N:2*N], (D,1))
    B = B.T

    for j in range(D):
        start = (j + 2)*N
        end = start + N
        A_B[:, j] = (Y[start:end]).T

    return A, B, A_B

def _first_order(A, AB, B):
    # First order estimator following Saltelli et al. 2010 CPC, normalized by
    # sample variance
    #return np.mean(B * (AB - A), axis=0) / np.var(np.r_[A, B], axis=0)
    #return np.mean(B * (AB - A), axis=0, dtype=np.float64) / np.var(np.concatenate((A, B), axis=0), axis=0, ddof=1, dtype=np.float64)
    return np.mean(B * (AB - A), axis=0, dtype=np.float64) / np.var(np.r_[A, B], axis=0, ddof=1, dtype=np.float64)

def _total_order(A, AB, B):
    # Total order estimator following Saltelli et al. 2010 CPC, normalized by
    # sample variance
    #return 0.5 * np.mean((A - AB) ** 2, axis=0) / np.var(np.r_[A, B], axis=0)
    #return np.mean(B * (AB - A), axis=0, dtype=np.float64) / np.var(np.concatenate((A, B), axis=0), axis=0, ddof=1, dtype=np.float64)
    return np.mean(B * (AB - A), axis=0, dtype=np.float64) / np.var(np.r_[A, B], axis=0, ddof=1, dtype=np.float64)
