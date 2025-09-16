# chaospy surrogate model construction

import numpy as np
import chaospy 
import transport_pce_pipeline
import pathlib

from uqef_dynamic.utils import utility
from uqef_dynamic.utils import transport_map
from uqef_dynamic.models.hbv_sask import hbvsask_utility as hbv
from uqef_dynamic.models.hbv_sask import HBVSASKModel as hbvmodel





def setup_HBV():
    working_dir_name=f"trial_single_run_hbvsaskmodel_7d_filtering_III"
    hbv_model_data_path = pathlib.Path("/home/christoph/projects/thesis_code/HBV-SASK-data")
    # change 6D to 10D
    configuration_file = pathlib.Path('/home/christoph/projects/thesis_code/HBV-SASK-py-tool/configurations/configuration_hbv_6D.json')
    # configuration_file = pathlib.Path('/home/christoph/projects/thesis_code/UQEF-Dynamic/data/configurations/configuration_hbv_10D.json')
    inputModelDir = hbv_model_data_path
    basin = "Oldman_Basin"  # 'Banff_Basin'
    workingDir = hbv_model_data_path / basin / "model_runs" / working_dir_name
    directory_for_saving_plots = workingDir
    if not str(directory_for_saving_plots).endswith("/"):
        directory_for_saving_plots = str(directory_for_saving_plots) + "/"

    # Creating Model Object
    writing_results_to_a_file = False
    plotting = False
    createNewFolder = False # create a separate folder to save results for each model run
    hbvsaskModelObject = hbvmodel.HBVSASKModel(
        configurationObject=configuration_file,
        inputModelDir=inputModelDir,
        workingDir=workingDir,
        basin=basin,
        writing_results_to_a_file=writing_results_to_a_file,
        plotting=plotting
    )
    
    return hbvsaskModelObject




def construct_polynomial_chaos_expansion(mean_state_values_dict, transport_map, scaler, pce_samples):
    
    # mean_state_values_dict: dictionary with keys {SWE, SMS, S1, S2}
    # transport_map
    # scaler
    # pce_samples: (num_samples, dim) --> (particle_count, 7)

    hbvsaskModelObject = setup_HBV()
    num_samples = pce_samples


    # !! standard_parameters (dim, num_samples) / (dim, 1) [chaospy format, transportmap format, standard_params_matrix format]
    def inverse(standard_parameters):
        """Inverse parameter distribution from reference (SNV) back to target (original / exponential)"""
        
        # print()
        # print("== SAMPLED PARAMETERS ==")
        # print(standard_parameters)
        # print()
        
        # Inverse transport map approximation
        X_reconstruct = transport_map.Inverse(np.empty((0, standard_parameters.shape[1])), standard_parameters)
        # print("After transport_map.Inverse: any nan?", np.any(np.isnan(X_reconstruct)), "any inf?", np.any(np.isinf(X_reconstruct)))
        # print("After inverse map:", X_reconstruct)
    
        # Inverse scaling
        X_reconstruct = scaler.inverse_transform(X_reconstruct.T).T
        # print("After scaler.inverse_transform: any nan?", np.any(np.isnan(X_reconstruct)), "any inf?", np.any(np.isinf(X_reconstruct)))
        # print("After inverse scaling:", X_reconstruct)
    
        # Inverse logarithm
        for i in range(X_reconstruct.shape[0]):
            X_reconstruct[i, :] = np.exp(X_reconstruct[i, :])
        
        # print("After exponentiation:", X_reconstruct)

        # print("After np.exp: any nan?", np.any(np.isnan(X_reconstruct)), "any inf?", np.any(np.isinf(X_reconstruct)))

        
        return X_reconstruct


    # inverted_parameters: (dim, 1)
    def evaluate(inverted_parameters):
        """Evaluates the hydrological model for given parameters in original (target / exponential) form"""
        
        
        # print()
        # print("== INVERTED PARAMETERS ==")
        # print(inverted_parameters)
        # print()
        
        param_names = ['TT', 'C0', 'beta', 'ETF', 'FC', 'FRAC', 'K2']
        param_dict = {name: float(val) for name, val in zip(param_names, inverted_parameters)}
        # print("=== PARAM DICT ===")
        # print(param_dict)
        
        # TODO: check argument values
        # unique_index_model_run, y_t_model, y_t_observed, x_t_plus_1, parameter_value_dict
        _, y_t_model, _, _, _ = transport_pce_pipeline.run_model_single_time_stamp_single_particle(
            hbvsaskModelObject=hbvsaskModelObject,
            date_of_interest=hbvsaskModelObject.end_date,
            parameter_value_dict=param_dict, # ! dictionary
            state_values_dict=mean_state_values_dict
            )
        # print("original_parameters shape:", np.shape(original_parameters))
        # print("original_parameters:", original_parameters)
    
        # result = hbvsaskModelObject.run_model_single_time_stamp(hbvsaskModelObject.end_date, parameters=original_parameters, raise_exception_on_model_break=True)
        
        return y_t_model
        


    


    distribution_r = chaospy.J(
        chaospy.Normal(0, 1), 
        chaospy.Normal(0, 1), 
        chaospy.Normal(0, 1),
        chaospy.Normal(0, 1), 
        chaospy.Normal(0, 1), 
        chaospy.Normal(0, 1),
        chaospy.Normal(0, 1))
    
    
    # i.i.d. sample in standard Gauss
    # !! samples_r (dim, num_samples) [chaospy format]
    samples_r = distribution_r.sample(num_samples, rule="latin_hypercube") # sobol quasi random --> try pure random (Latin Hypercube)
    print(f"num_samples: ", samples_r.shape)
    
    # DEFAULT_PAR_VALUES_DICT = {'TT': 0.0, 'C0': 0.5, 'ETF': 0.2, 'FC': 250,
    #                        'beta': 2.0, 'FRAC': 0.3, 'K2': 0.05,  'LP': 0.5,
    #                        'K1': 0.5, 'alpha': 2.0, 
    #                        'UBAS': 1, 'PM': 1, "M": 1.0, "VAR_M": 1e-4}
    
    # TT, C0, beta, ETF, FC, FRAC, K2
    default_sample = [[0.0], [0.5], [2.0], [0.2], [250], [0.3], [0.05]]
    
    
    # DEFAULT_PAR_INFO_DICT = {
    #     'TT': {"lower": -4.0, "upper": 4.0, "default": 0.0},
    #     'C0': {"lower": 0.0, "upper": 5.0, "default": 0.5},
    #     'ETF': {"lower": 0.0, "upper": 1.0, "default": 0.2},
    #     'LP': {"lower": 0.0, "upper": 1.0, "default": 0.5},
    #     'FC': {"lower": 50.0, "upper": 1000.0, "default": 250.0},
    #     'beta': {"lower": 1.0, "upper": 3.0, "default": 2.0},
    #     'FRAC': {"lower": 0.1, "upper": 0.9, "default": 0.3},
    #     'K1': {"lower": 0.05, "upper": 1.0, "default": 0.5},
    #     'alpha': {"lower": 1.0, "upper": 3.0, "default": 2.0},
    #     'K2': {"lower": 0.0, "upper": 0.1, "default": 0.05},
    #     'UBAS': {"lower": 1.0, "upper": 3.0, "default": 1.0},
    #     'PM':{"lower": 0.5, "upper": 2.0, "default": 1.0},
    #     'M':{"lower": 0.9, "upper": 1.0, "default": 1.0},
    #     'VAR_M':{"lower": 1e-5, "upper": 1e-3, "default": 1e-4}
    # }
    
    # TT, C0, beta, ETF, FC, FRAC, K2   
    # PARAMETER BOUNDS
    # inverted_min_max = np.array([[-4.0, 4.0], [0.0, 5.0], [1.0, 3.0], [0.0, 1.0], [50.0, 500.0], [0.1, 0.9], [0.0, 0.1]])
    inverted_min_max = np.array([[-13.0, 13.0], [0.0, 5.0], [0.0, 5.0], [0.0, 2.0], [0.0, 700.0], [0.0, 0.3], [0.0, 0.1]])
    lower_bounds = inverted_min_max[:, 0]
    upper_bounds = inverted_min_max[:, 1]   
    
    print("TODO: inverse! need to adjust logarithm for different distributions!")
    samples_q_list = []
    samples_r_list = []
    countBounds = 0
    countNaN = 0
    
    # !! samples_r (dim, num_samples) [chaospy format]
    for i in range(samples_r.shape[1]):
        try:
            sample = samples_r[:, i:i+1]  # shape (n_params, 1)
            # print("Original sample: ", sample)
            sample_q = inverse(sample)
            
            if np.any(np.isnan(sample_q)):
                # samples_q_list.append(default_sample)
                countNaN += 1
                continue
            
            # check if inverted sample within physical bounds
            
            checksample = np.asarray(sample_q).flatten()
            within_bounds = (checksample >= lower_bounds) & (checksample <= upper_bounds)
            all_within_bounds = np.all(within_bounds)
            all_within_bounds = True
            
            if not all_within_bounds:
                # print("not in bound: ", sample_q)
                # print(sample)
                # print(f"NaN detected in inverted sample {i}:")
                # print("Input sample_r:", sample.flatten())
                # print("Output sample_q:", sample_q.flatten())
                # samples_q_list.append(default_sample)
                # samples_q_list.append(default_sample)
                countBounds += 1
            else:   
                samples_q_list.append(sample_q)
                # filter samples_r for those samples that are invalid
                samples_r_list.append(sample)
        except Exception as e:
            print(f"Exception for sample {i}: {e}")
            print("Input sample_r:", sample.flatten())
            
    # samples_q: (dim, num_samples)
    # samples_q = np.hstack(samples_q_list)
    
    print("finished inversion")
    print(f"{num_samples - (countNaN + countBounds)}/{num_samples} samples were correctly inversed / could be used for chaospy")
    print(f"{countNaN} times there was an issue with NaN")
    print(f"{countBounds} times the values were out of bounds")


    # TODO: check expansion order
    expansion = chaospy.generate_expansion(3, distribution_r)
    print("chaos: generated expansion")
    
    
    evaluations = []
    valid_samples_q = []
    valid_samples_r = []
    
    countSkip = 0
    
    for sample_q, sample_r in zip(samples_q_list, samples_r_list):
        y = evaluate(sample_q)
        # Refuse abnormal outputs (e.g., outside [0, 100])
        if np.isnan(y) or np.isinf(y) or y < 0 or y > 600: 
            print(f"Abnormal evaluation: {y}, skipping sample.")
            countSkip += 1
            continue
        evaluations.append(y)
        valid_samples_q.append(sample_q.reshape(-1, 1))
        valid_samples_r.append(sample_r)
    
    print("Done evaluations in chaos")
    print(f"Skipped {countSkip}/{num_samples - (countNaN + countBounds)} samples")
    print(f"Remaining samples: {num_samples - (countNaN + countBounds + countSkip)}")
    
    evaluations = np.array(evaluations)
    valid_samples_r = np.hstack(valid_samples_r)   
    
    print("Approximating model...")
    # samples_r_list = np.hstack(samples_r_list)
    # model_approx = chaospy.fit_regression(expansion, samples_r_list, evaluations)
    model_approx = chaospy.fit_regression(expansion, valid_samples_r, evaluations)
    
    mean = chaospy.E(model_approx, distribution_r)
    std = chaospy.Std(model_approx, distribution_r)

    # print(model_approx)
    print(f"chaospy approximated mean: {mean}")
    print(f"chaospy approximated std: {std}")
    
    
    return model_approx