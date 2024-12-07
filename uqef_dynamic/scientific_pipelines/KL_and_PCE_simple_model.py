from collections import defaultdict
from scipy.linalg import sqrtm
import scipy.special
import numpy as np
from numpy.linalg import eig
import pandas as pd
from typing import List, Optional, Dict, Any, Union, Tuple

import chaospy as cp

# =========================================================
# Utility Functions
# =========================================================

time_column_name = TIME_COLUMN_NAME = 'TimeStamp'
index_column_name = INDEX_COLUMN_NAME = "Index_run"
single_qoi_column = "model"
single_qoi_column_centered = single_qoi_column + "_centered"

def transformation_of_parameters(samples, distribution_r, distribution_q):
    """
    :param samples: array of samples from distribution_r
    :param distribution_r: 'standard' distribution
    :param distribution_q: 'user-defined' distribution
    :return: array of samples from distribution_q
    """
    return distribution_q.inv(distribution_r.fwd(samples))

def model(t, alpha, beta, l):
    return l*np.exp(-alpha*t)*(np.cos(beta*t)+alpha/beta*np.sin(beta*t))

def running_the_model_and_generating_df(
    model, 
    t: Union[np.array, np.ndarray, List[Union[int, float]]], 
    parameters: np.ndarray,
    time_column_name: str=TIME_COLUMN_NAME,
    index_column_name: str= INDEX_COLUMN_NAME,
    **kwargs
    ) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Runs the model over time and parameters, and generates a DataFrame with the simulation results.

    Args:
        model: The model to be run.
        t: The time array or list.
        parameters: The array of parameters used to stimulate the model. Expected to be of shape dim x number_of_nodes
        time_column_name: The name of the column to store time values in the DataFrame. Defaults to TIME_COLUMN_NAME.
        index_column_name: The name of the column to store index values in the DataFrame. Defaults to INDEX_COLUMN_NAME.

    Returns:
        A tuple containing the model runs as a numpy array and the simulation results as a pandas DataFrame.
    """
    model_runs = np.empty((parameters.shape[1], len(t)))
    for idx, single_node in enumerate(parameters.T):
        model_runs[idx] = model(t, *single_node, **kwargs)

    list_of_single_df = []
    for idx, single_node in enumerate(parameters.T):
        df_temp = pd.DataFrame(model_runs[idx], columns=[single_qoi_column])
        df_temp[time_column_name] = t
        df_temp[index_column_name] = idx
        tuple_column = [tuple(single_node)] * len(df_temp)
        df_temp['Parameters'] = tuple_column
        list_of_single_df.append(df_temp)

    df_simulation_result = pd.concat(
        list_of_single_df, ignore_index=True, sort=False, axis=0)
    return model_runs, df_simulation_result

# =========================================================
# Definition of the different parameters and variables
# =========================================================

# Definition of uncertain parameters
dim = 3
alpha_dist = cp.Uniform(3/8, 5/8)
beta_dist = cp.Uniform(10/4, 15/4)
l_dist = cp.Uniform(-5/4, -3/4)
param_names = ['apha', 'beta', 'l']
joint_dist = cp.J(alpha_dist, beta_dist, l_dist)

alpha_dist_standard = cp.Uniform(-1, 1)
beta_dist_standard = cp.Uniform(-1, 1)
l_dist_standard = cp.Uniform(-1, 1)
joint_dist_standard = cp.J(alpha_dist_standard, beta_dist_standard, l_dist_standard)

numSamples = numEvaluations = N = 10**3 #150
N_quad = 20
time_quadrature = t = np.linspace(0, 10, N_quad)
t_final = 10
t_starting = 0
numCollocationPointsPerDim = q = 8
order = 3 # 3
total_number_of_nodes = np.power(numCollocationPointsPerDim+1,dim)
c_number = scipy.special.binom(dim+order, dim)
print(f"Max order of polynomial: {order}")
print(f"Number of quadrature points in 1D: {numCollocationPointsPerDim+1}")
print(f"Total number of points in {dim}D space: {total_number_of_nodes}")
print(f"Total number of expansion coefficients in {dim}D space: {int(c_number)}")
print(f"Total number of time-stamps: {len(t)}")

rule = 'g'
growth = False
sparse = False
nodes_quad, weights_quad = cp.generate_quadrature(
    numCollocationPointsPerDim, joint_dist_standard, rule=rule, growth=growth, sparse=sparse)
parameters_quad = transformation_of_parameters(nodes_quad, joint_dist_standard, joint_dist)

poly_rule = 'three_terms_recurrence'  # 'three_terms_recurrence', 'gram_schmidt', 'cholesky'
poly_normed = True
polynomial_expansion, norms = cp.generate_expansion(
    order, joint_dist_standard, rule=poly_rule, normed=poly_normed, retall=True)

# =========================================================
# Running the model and generating the DataFrame
# Computing the centered outputs
# =========================================================

model_runs_quad, df_simulation_result_quad = running_the_model_and_generating_df(model, t, parameters_quad)
    
# adding a column with centered results to the df_simulation_result_quad
grouped = df_simulation_result_quad.groupby([time_column_name,])
groups = grouped.groups
for key, val_indices in groups.items():
        qoi_values = df_simulation_result_quad.loc[val_indices.values][single_qoi_column].values
        # compute mean
        mean = np.dot(qoi_values, weights_quad)
        df_simulation_result_quad.loc[val_indices, single_qoi_column_centered] = \
        df_simulation_result_quad.loc[val_indices, single_qoi_column] - mean

# =========================================================
# Computing the Covariance Matrix
# =========================================================

N = total_number_of_nodes
centered_outputs_quadrature = np.empty((N, N_quad))

grouped = df_simulation_result_quad.groupby([index_column_name,])
groups = grouped.groups
for key, val_indices in groups.items():
    centered_outputs_quadrature[int(key), :] = df_simulation_result_quad.loc[val_indices, single_qoi_column_centered].values
    
covariance_matrix = np.empty((N_quad, N_quad))
for c in range(N_quad):
    for s in range(N_quad):
        covariance_matrix[s, c] = np.dot(weights_quad, centered_outputs_quadrature[:, c]*centered_outputs_quadrature[:,s])

# plt.imshow(covariance_matrix, cmap='hot', interpolation='nearest')
# plt.colorbar()
# plt.title('Covariance Matrix')

# =========================================================
# Solving Discrte Eigenvalue Problem
# =========================================================

assert len(t)==N_quad
h = (t_final - t_starting)/(N_quad-1)
# Note: weights_quad are in stohastic space, and weights are in time-domian
weights = [h for i in range(N_quad)]
assert len(t)==len(weights)
weights[0] /= 2
weights[-1] /= 2

K = covariance_matrix
# Check if the approximation of the covarriance matrix is symmetric
# temp = np.array_equal(covariance_matrix, covariance_matrix.T)
W = np.diag(weights)
sqrt_W = sqrtm(W)
LHS = sqrt_W@K@sqrt_W

B = np.identity(LHS.shape[0])
# Solve the generalized eigenvalue problem
eigenvalues, eigenvectors = eig(LHS)
idx = eigenvalues.argsort()[::-1]   # Sort by descending real part of eigenvalues
sorted_eigenvalues = eigenvalues[idx]
sorted_eigenvectors = eigenvectors[:, idx]
eigenvalues = sorted_eigenvalues
eigenvectors = sorted_eigenvectors

# plt.yscale("log")
# plt.plot(eigenvalues, 'x')
# eigenvalues_real = np.asfarray([element.real for element in eigenvalues])
# eigenvalues_real_scaled = eigenvalues_real/eigenvalues_real[0]
# plt.yscale("log")
# plt.plot(eigenvalues_real_scaled, 'x')

# Inversion of the Matrix
final_eigenvectors = np.linalg.inv(sqrt_W)@eigenvectors

# =========================================================
# Kl Expansion
# =========================================================

Var_kl_approx = np.sum(eigenvalues)
print(f"Var_kl_approx - {Var_kl_approx}")

N_kl =  8 # [2, 4, 6, 8, 10]
weights = np.asfarray(weights)

f_kl_eval_at_params = np.empty((N_kl, N))
# weights @ centered_outputs_quadrature[k,:] @ final_eigenvectors[:,i]
for i in range(N_kl):
    for k in range(N):
        f_kl_eval_at_params[i, k] = 0
        for m in range(N_quad):
            f_kl_eval_at_params[i, k] += weights[m]*centered_outputs_quadrature.T[m,k]*final_eigenvectors[m,i]

f_kl_surrogate_dict = {}
f_kl_surrogate_coefficients = []
for i in range(N_kl):
    f_kl_surrogate_dict[i] = {}
    f_kl_gPCE, f_kl_coeff = cp.fit_quadrature(polynomial_expansion, nodes_quad, weights_quad, f_kl_eval_at_params[i,:], retall=True)
    f_kl_surrogate_dict[i]["gPCE"] = f_kl_gPCE
    f_kl_surrogate_dict[i]["coeff"] = f_kl_coeff
    f_kl_surrogate_coefficients.append(np.asfarray(f_kl_coeff))

f_kl_surrogate_coefficients = np.asfarray(f_kl_surrogate_coefficients)
print(f"DEBUGGINH f_kl_surrogate_coefficients.shape - {f_kl_surrogate_coefficients.shape}")
# =========================================================
# Generalized Sobol Indices
# =========================================================

dic = polynomial_expansion.todict()
alphas = []
for idx in range(len(polynomial_expansion)):
    expons = np.array([key for key, value in dic.items() if value[idx]])
    alphas.append(tuple(expons[np.argmax(expons.sum(1))]))

index = np.array([any(alpha) for alpha in alphas])

dict_of_num = defaultdict(list)
for idx in range(len(alphas[0])):
    dict_of_num[idx] = []

for i in range(f_kl_surrogate_coefficients.shape[0]):
    coefficients = np.asfarray(f_kl_surrogate_coefficients[i,:])
    variance = np.sum(coefficients[index] ** 2, axis=0)
    for idx in range(len(alphas[0])):
        index_local = np.array([alpha[idx] > 0 for alpha in alphas])  # Compute the total Sobol indices;
        dict_of_num[idx].append(np.sum(coefficients[index_local] ** 2, axis=0))

denum = Var_kl_approx
    
for idx in range(len(alphas[0])):
    param_name = param_names[idx]
    num = np.sum(np.asfarray(dict_of_num[idx]), axis=0)
    s_tot_generalized = num/denum
    print(f"Generalized Total Sobol Index computed based on the PCE of KL expansion for {param_name} is {s_tot_generalized}")
    # with open(fileName, 'a') as file:
    #     # Write each variable to the file followed by a newline character
    #     file.write(f'{param_name}: {s_tot_generalized}\n')

# =========================================================
# Re-evaluating the surrogate
# =========================================================

# numSamples = 10**3
# rule = 'random'
# nodes_to_eval_kl_surrogate = joint_dist_standard.sample(size=numSamples, rule=rule).round(4)
# surrogate_eval = np.zeros((len(t),nodes_to_eval_kl_surrogate.shape[1]))
# mean_vector = df_stat_gpce['E'].to_numpy()  # one needs access to df_stat_gpce - or compute a mean in a different way...
# for m in range(len(t)):
#     surrogate_eval[m,:] = mean_vector[m]
#     for i in range(N_kl):
#         surrogate_eval[m,:] += f_kl_surrogate_dict[i]["gPCE"](*nodes_to_eval_kl_surrogate)*final_eigenvectors[m,i]
