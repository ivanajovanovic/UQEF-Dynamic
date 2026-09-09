"""Legacy triangular transport map backend (bundled toolbox).

Wraps uqef_dynamic/utils/transport_map.py — the Ramgraber-style triangular map
toolbox vendored in this repository. Kept as the "legacy" backend of
transport_timeseries.gaussianize_parameter_samples so existing runs reproduce;
new work should prefer the mpart or anamorphosis backends.

Moved here verbatim from particle_filtering_pipeline.py to keep that script to
the filter itself.
"""

import numpy as np

from uqef_dynamic.utils import transport_map


__all__ = ["define_the_transport_map_parameterization",
           "transform_samples_with_transport_map"]


def define_the_transport_map_parameterization(D, maxorder=5):
    # =============================================================================
    # Define the transport map parameterization
    # =============================================================================

    # Next, we define the map component functions used in the triangular transport 
    # map. The map definition requires two lists of lists: one for the monotone 
    # part (basis functions which do depend on the last argument) and one for the
    # nonmonotone part (basis functions which do not depend on the last argument).
    # Each entry in those lists is another list that defines the basis functions.
    # Polynomial basis functions are lists of integers, with a potential keyword
    # such as 'HF' appended to mark it as a Hermite function. RBFs or related basis
    # functions are defined as strings such as 'RBF 0' or 'iRBF 7'. 
    # 
    # Example: --------------------------------------------------------------------
    #
    # monotone = [
    #   [ [0] ],
    #   [ [1], [0,0,1,'HF'] ] ]
    # nonmonotone = [
    #   [ [] ],
    #   [ [], [0], [0,0], [0,0,'HF], 'RBF 0'] ]
    #
    # Explanation: ----------------------------------------------------------------
    #
    # Monotone [list]
    #   |
    #   └―― Map component 1 [list] (last argument: entry x_{0})
    #   |       |
    #   |       └―― [0] Basis function 1 (linear term for entry 0)
    #   |
    #   └―― Map component 2 [list] (last argument: entry x_{1})
    #           |
    #           └―― [1] Basis function 1 (linear term for entry 1)
    #           |
    #           └―― [0,0,1,'HF'] Basis function 2 (cross-term: quadratic Hermite function for entry 0, linear Hermite function for entry 1)
    #
    # Nonmonotone [list]
    #   |
    #   └―― Map component 1 [list] (valid arguments: constant)
    #   |       |
    #   |       └―― [] Basis function 1 (constant term)
    #   |
    #   └―― Map component 2 [list] (Valid arguments: consant, x_{0})
    #           |
    #           └―― [] Basis function 1 (constant term)
    #           |
    #           └―― [0] Basis function 2 (linear term for entry x_{0})
    #           |
    #           └―― [0,0] Basis function 3 (quadratic term for entry x_{0})
    #           |
    #           └―― [0,0,'HF'] Basis function 4 (quadratic Hermite function for entry x_{0})
    #           |
    #           └―― 'RBF 0' Basis function 5 (radial basis function for entry x_{0})

    # Create empty lists for the map component specifications
    monotone    = []
    nonmonotone = []

    # Here, we try  different form of map parameterization. Let's try using maps
    # with separable monotonicity. These are often much more efficient, but do not
    # allow for cross-terms or nonmonotone basis functions in the 'monotone' list.
    for k in range(D):
        
        # Level 1: Add an empty list entry for each map component function
        monotone.append([])
        nonmonotone.append([]) # An empty list "[]" denotes a constant
        
        # Level 2: We initiate the nonmonotone terms with a constant
        nonmonotone[-1].append([])

        # Nonmonotone part --------------------------------------------------------

        # Go through every polynomial order
        for order in range(maxorder):
            
            # We only have non-constant nonmonotone terms past the first map 
            # component, and we already added the constant term earlier, so only do
            # this for the second map component function (k > 0).
            if k > 0: 
                
                # The nonmonotone basis functions can be as nonmonotone as we want.
                # Hermite functions are generally a good choice.
                nonmonotone[-1].append([k-1]*(order+1)+['HF'])
                
        # Monotone part -----------------------------------------------------------
        
        # Let's get more fancy with the monotone part this time. If the order  we 
        # specified is one, then use a linear term. Otherwise, use a few monotone 
        # special functions: Left edge terms, integrated radial basis functions, 
        # and right edge terms
        
        # The specified order is one
        if maxorder == 1:
            
            # Then just add a linear term
            monotone[-1].append([k])
            
        # Otherweise, the order is greater than one. Let's use special terms.
        else:
            
            # Add a left edge term. The order matters for these special terms. 
            # While they are placed according to marginal quantiles, they are 
            # placed from left to right. We want the left edge term to be left.
            monotone[-1].append('LET '+str(k))
                    
            # Lets only add maxorder-1 iRBFs
            for order in range(maxorder-1):
                
                # Add an integrated radial basis function
                monotone[-1].append('iRBF '+str(k))
        
            # Then add a right edge term 
            monotone[-1].append('RET '+str(k))
    return  monotone, nonmonotone


def transform_samples_with_transport_map(parameter_samples_matrix):
    # =============================================================================
    # Use transport map to transform current parameter samples to standard Gaussian
    # =============================================================================
    # Define the transport map parameterization
    # Create empty lists for the map component specifications
    monotone    = []
    nonmonotone = []
    # require polynomial basis terms up to order 5
    maxorder    = 1
    monotone, nonmonotone = define_the_transport_map_parameterization(D=parameter_samples_matrix.shape[1], maxorder=maxorder)
    # =============================================================================
    # Create the transport map object
    # =============================================================================
    # With the map parameterization (nonmonotone, monotone) defined and the target
    # samples (X) obtained, we can start creating the transport map object.
    # To begin, delete any map object which might already exist.
    if "tm" in globals():
        del tm

    # Create the transport map object tm
    tm     = transport_map.transport_map(
        monotone                = monotone,                 # Specify the monotone parts of the map component function
        nonmonotone             = nonmonotone,              # Specify the nonmonotone parts of the map component function
        X                       = parameter_samples_matrix, # = np.random.uniform(size=(N,D)), # Dummy input A N-by-D matrix of training samples (N = ensemble size, D = variable space dimension)
        polynomial_type         = "hermite function",       # What types of polynomials did we specify? The option 'Hermite functions' here are re-scaled probabilist's Hermites, to avoid numerical overflow for higher-order terms
        monotonicity            = "separable monotonicity",   # Are we ensuring monotonicity through 'integrated rectifier' or 'separable monotonicity'?
        standardize_samples     = True,                     # Standardize the training ensemble X? Should always be True
        workers                 = 1,                        # Number of workers for the parallel optimization.
        # quadrature_input        = {                         # Keywords for the Gaussian quadrature used for integration
        #     'order'         : 25,
        #     'adaptive'      : False,
        #     'threshold'     : 1E-9,
        #     'verbose'       : False,
        #     'increment'     : 6}
        # regularization          = "l2",
        # regularization_lambda   = lmbda,
        verbose                 = False
        )

    # Optimize the transport maps. This takes a while, it's an extremeley complicated map.
    tm.optimize()

    # Store the coefficients in a dictionary
    dict_coeffs = {
        'coeffs_mon'    : tm.coeffs_mon,
        'coeffs_nonmon' : tm.coeffs_nonmon}
    
    # Save the dictionary
    # pickle.dump(dict_coeffs,open('dict_coeffs_order='+str(maxorder)+'.p','wb'))
    # =============================================================================
    # Apply the map
    # =============================================================================  
    # -----------------------------------------------------------------------------
    # forward map from the target to the reference
    # -----------------------------------------------------------------------------
    # we apply the map forward. This transforms samples from
    # the target into samples from the reference (a standard Gaussian)

    # We can evaluate the forward map with the following command:
    Z_gen   = tm.map(parameter_samples_matrix)
    # =============================================================================
    return Z_gen
