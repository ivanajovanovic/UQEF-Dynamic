# Transport Map Part in Pipeline to Gaussianize the output of PF

import mpart as mt
import numpy as np
from scipy.optimize import minimize
from scipy.stats import multivariate_normal
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler


def transform_parameters_with_transport_map(parameter_samples_matrix: np.ndarray):    
    """Transforms the parameters from exponential distribution to standard Gauss."""
    """Returns: transformed samples, transport_map object (need for inverse), scaler object (need for inverse)"""
    # shape[0] should be dimension, shape[1] should be num_points
    # MParT expects the input sample matrix of shape (dim, num_samples)
    if parameter_samples_matrix.shape[0] > parameter_samples_matrix.shape[1]:
        print("Transposing the parameter samples matrix to ensure shape is (dim, num_samples)")
        parameter_samples_matrix = parameter_samples_matrix.T
        
    print("dim: ", parameter_samples_matrix.shape[0])
    print("num_samples: ", parameter_samples_matrix.shape[1])
    print("Input sample matrix shape:", parameter_samples_matrix.shape)
    
    
    dim = parameter_samples_matrix.shape[0]
    rho1 = multivariate_normal(np.zeros(dim), np.eye(dim))
    
        
    ### LOGARITHM

    for i in range(parameter_samples_matrix.shape[0]):
        if np.all(parameter_samples_matrix[i, :] <= 0): 
        ### NEED TO DECIDE BASED ON X0: data strictly positive (log) - data strictly negative (-log)
            parameter_samples_matrix[i, :] = -np.log(-parameter_samples_matrix[i, :])
        else:
            parameter_samples_matrix[i, :] = np.log(parameter_samples_matrix[i, :])
        #X[i, :] = np.log1p(X[i, :])


    ### SCALING
    parameter_scaler = StandardScaler()
    parameter_samples_matrix = parameter_scaler.fit_transform(parameter_samples_matrix.T).T
    

    def obj(coeffs, tri_map, x):
        """ Evaluates the log-likelihood of the samples using the map-induced density. """
        num_points = x.shape[1]
        tri_map.SetCoeffs(coeffs)

        # Compute the map-induced density at each point
        map_of_x = tri_map.Evaluate(x)
        rho_of_map_of_x = rho1.logpdf(map_of_x.T)
        log_det = tri_map.LogDeterminant(x)

        # Return the negative log-likelihood of the entire dataset
        return -np.sum(rho_of_map_of_x + log_det) / num_points

    def grad_obj(coeffs, tri_map, x):
        """ Returns the gradient of the log-likelihood objective wrt the map parameters. """
        num_points = x.shape[1]
        tri_map.SetCoeffs(coeffs)

        # Evaluate the map
        map_of_x = tri_map.Evaluate(x)

        # Now compute the inner product of the map jacobian (\nabla_w S) and the gradient (which is just -S(x) here)
        grad_rho_of_map_of_x = -tri_map.CoeffGrad(x, map_of_x)

        # Get the gradient of the log determinant with respect to the map coefficients
        grad_log_det = tri_map.LogDeterminantCoeffGrad(x)

        return -np.sum(grad_rho_of_map_of_x + grad_log_det, 1) / num_points

   
    ## COEFF PARAMETRIZTATION

    # Options for the transport map
    map_options = mt.MapOptions()
    map_options.basisType = mt.BasisTypes.ProbabilistHermite
    max_order = 4

    tri_map = mt.CreateTriangular(
        dim, dim, max_order, map_options
    )

    
    ## OPTIMIZATION

    coeffs_init = tri_map.CoeffMap()

    res = minimize(
        obj,
        coeffs_init,
        args=(tri_map, parameter_samples_matrix),
        jac=grad_obj,
        method='L-BFGS-B',
        options={'gtol': 1e-2, 'disp': True}
    )
    


    if not res.success:
        raise RuntimeError("Optimization failed: " + res.message)


    tri_map.SetCoeffs(res.x)  # Update map with optimized coefficients

    mapped_samples = tri_map.Evaluate(parameter_samples_matrix)
    
      
    ## ACCURACY CHECKS

    print('Mean of mapped samples:', np.mean(mapped_samples, axis=1))
    print('Covariance of mapped samples:', np.cov(mapped_samples))        
    

    # Tranform back to original shape
    if parameter_samples_matrix.shape[1] > parameter_samples_matrix.shape[0]:
        print("Transposing the mapped samples back to original shape")
        mapped_samples = mapped_samples.T
    
    
    return mapped_samples, tri_map, parameter_scaler









def transform_states_with_transport_map(states_matrix: np.ndarray):    
    """Transforms the parameters from exponential distribution to standard Gauss."""
    """Returns: transformed samples, transport_map object (need for inverse), scaler object (need for inverse)"""
    # shape[0] should be dimension, shape[1] should be num_points
    # MParT expects the input sample matrix of shape (dim, num_samples)
    
    # delete index and watershed_size columns
    states_matrix = np.delete(states_matrix, [4, 5], axis=0)

    
    if states_matrix.shape[0] > states_matrix.shape[1]:
        print("Transposing the states matrix to ensure shape is (dim, num_samples)")
        states_matrix = states_matrix.T
        
    print("dim: ", states_matrix.shape[0])
    print("num_samples: ", states_matrix.shape[1])
    print("Input sample matrix shape:", states_matrix.shape)
    
    
    dim = states_matrix.shape[0]
    rho1 = multivariate_normal(np.zeros(dim), np.eye(dim))
    
        
    ### LOGARITHM

    for i in range(3):
        min_val = np.min(states_matrix[i, :])
        print(min_val)
        if min_val <= 0:
            shift = abs(min_val) + 1  # tiny epsilon to avoid log(0)
            # min_positive = np.min(X[i, states_matrix[i, :] > 0])
            print(f"SHIFT param {i} by {shift}")
            states_matrix[i, :] = states_matrix[i, :] + shift
            states_matrix[i, :] = np.log1p(states_matrix[i, :])
        else:
            states_matrix[i, :] = np.log(states_matrix[i, :])


    ### SCALING
    states_scaler = StandardScaler()
    states_matrix = states_scaler.fit_transform(states_matrix.T).T
    

    def obj(coeffs, tri_map, x):
        """ Evaluates the log-likelihood of the samples using the map-induced density. """
        num_points = x.shape[1]
        tri_map.SetCoeffs(coeffs)

        # Compute the map-induced density at each point
        map_of_x = tri_map.Evaluate(x)
        rho_of_map_of_x = rho1.logpdf(map_of_x.T)
        log_det = tri_map.LogDeterminant(x)

        # Return the negative log-likelihood of the entire dataset
        return -np.sum(rho_of_map_of_x + log_det) / num_points

    def grad_obj(coeffs, tri_map, x):
        """ Returns the gradient of the log-likelihood objective wrt the map parameters. """
        num_points = x.shape[1]
        tri_map.SetCoeffs(coeffs)

        # Evaluate the map
        map_of_x = tri_map.Evaluate(x)

        # Now compute the inner product of the map jacobian (\nabla_w S) and the gradient (which is just -S(x) here)
        grad_rho_of_map_of_x = -tri_map.CoeffGrad(x, map_of_x)

        # Get the gradient of the log determinant with respect to the map coefficients
        grad_log_det = tri_map.LogDeterminantCoeffGrad(x)

        return -np.sum(grad_rho_of_map_of_x + grad_log_det, 1) / num_points

   
    ## COEFF PARAMETRIZTATION

    # Options for the transport map
    map_options = mt.MapOptions()
    map_options.basisType = mt.BasisTypes.ProbabilistHermite
    map_options.nugget = 1e-4  # stability
    max_order = 4

    tri_map = mt.CreateTriangular(
        dim, dim, max_order, map_options
    )

    
    ## OPTIMIZATION

    coeffs_init = tri_map.CoeffMap()

    res = minimize(
        obj,
        coeffs_init,
        args=(tri_map, states_matrix),
        jac=grad_obj,
        method='L-BFGS-B',
        options={'gtol': 1e-2, 'disp': True}
    )
    


    if not res.success:
        raise RuntimeError("Optimization failed: " + res.message)


    tri_map.SetCoeffs(res.x)  # Update map with optimized coefficients

    mapped_states = tri_map.Evaluate(states_matrix)
    
      
    ## ACCURACY CHECKS

    print('Mean of mapped samples:', np.mean(mapped_states, axis=1))
    print('Covariance of mapped samples:', np.cov(mapped_states))        
    

    # Tranform back to original shape
    if states_matrix.shape[1] > states_matrix.shape[0]:
        print("Transposing the mapped samples back to original shape")
        mapped_states = mapped_states.T
    
    
    return mapped_states, tri_map, states_scaler