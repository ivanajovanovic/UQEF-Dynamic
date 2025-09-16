# Transport Map Part in Pipeline to Gaussianize the output of PF

import mpart as mt
import numpy as np
from scipy.optimize import minimize
from scipy.stats import multivariate_normal
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler


def transform_samples_with_transport_map(parameter_samples_matrix: np.ndarray):    
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
    scaler = StandardScaler()
    parameter_samples_matrix = scaler.fit_transform(parameter_samples_matrix.T).T
    

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
    # if parameter_samples_matrix.shape[1] > parameter_samples_matrix.shape[0]:
    #     print("Transposing the mapped samples back to original shape")
    #     mapped_samples = mapped_samples.T
    
    
    # MAPPED SAMPLES IS NOW [dim, num_samples]!!!!!!!
    return mapped_samples, tri_map, scaler
