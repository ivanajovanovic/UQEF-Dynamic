# Transport Map Part in Pipeline to Gaussianize the output of PF

import mpart as mt
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.optimize import minimize



def transform_samples_with_transport_map(parameter_samples_matrix: np.ndarray):
    print("test")
    print("dim: ", parameter_samples_matrix.shape[1])
    
    dim = parameter_samples_matrix.shape[1]
    
    
    rho1 = multivariate_normal(np.zeros(dim),np.eye(dim))
    
    
    
    def obj(coeffs, tri_map,x):
        """ Evaluates the log-likelihood of the samples using the map-induced density. """
        num_points = x.shape[1]
        tri_map.SetCoeffs(coeffs)

        # Compute the map-induced density at each point 
        map_of_x = tri_map.Evaluate(x)
        rho_of_map_of_x = rho1.logpdf(map_of_x.T)
        log_det = tri_map.LogDeterminant(x)

        # Return the negative log-likelihood of the entire dataset
        return -np.sum(rho_of_map_of_x + log_det)/num_points


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
        
        return -np.sum(grad_rho_of_map_of_x + grad_log_det, 1)/num_points
    
    
    
    ## COEFF PARAMETRIZTATION
    
    # Options for the transport map
    map_options = mt.MapOptions()
    # map_options.basisType = mt.BasisTypes.ProbabilistHermite
    max_order = 3

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
        method='BFGS',
        options={'gtol': 1e-2, 'disp': True}
    )

    tri_map.SetCoeffs(res.x)  # Update map with optimized coefficients

    mapped_samples = tri_map.Evaluate(parameter_samples_matrix)
    print('Mean of mapped samples:', np.mean(mapped_samples, axis=1))
    print('Covariance of mapped samples:', np.cov(mapped_samples))


    return mapped_samples
