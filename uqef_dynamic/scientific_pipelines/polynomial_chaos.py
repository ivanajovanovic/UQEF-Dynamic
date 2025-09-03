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




def construct_polynomial_chaos_expansion(standard_parameter_matrix: np.ndarray, mean_state_values_dict,transport_map, scaler):
    

    hbvsaskModelObject = setup_HBV()

    def inverse(standard_parameters):
        """Inverse parameter distribution from reference (SNV) back to target (original / exponential)"""
        
        print()
        print("== SAMPLED PARAMETERS ==")
        print(standard_parameters)
        print()
        
        # Inverse transport map approximation
        X_reconstruct = transport_map.Inverse(np.empty((0, standard_parameters.shape[1])), standard_parameters)
        print("After transport_map.Inverse: any nan?", np.any(np.isnan(X_reconstruct)), "any inf?", np.any(np.isinf(X_reconstruct)))

        # Inverse scaling
        X_reconstruct = scaler.inverse_transform(X_reconstruct.T).T
        print("After scaler.inverse_transform: any nan?", np.any(np.isnan(X_reconstruct)), "any inf?", np.any(np.isinf(X_reconstruct)))

        # Inverse logarithm
        print("TODO: need to adjust logarithm for different distributions!")
        for i in range(X_reconstruct.shape[0]):
            X_reconstruct[i, :] = np.exp(X_reconstruct[i, :])
        print("After np.exp: any nan?", np.any(np.isnan(X_reconstruct)), "any inf?", np.any(np.isinf(X_reconstruct)))

        
        return X_reconstruct


    
    def evaluate(inverted_parameters):
        """Evaluates the hydrological model for given parameters in original (target / exponential) form"""
        
        
        print()
        print("== INVERTED PARAMETERS ==")
        print(inverted_parameters)
        print()
        
        # TODO: check argument values
        # unique_index_model_run, y_t_model, y_t_observed, x_t_plus_1, parameter_value_dict
        _, y_t_model, _, _, _ = transport_pce_pipeline.run_model_single_time_stamp_single_particle(
            hbvsaskModelObject=hbvsaskModelObject,
            date_of_interest=hbvsaskModelObject.end_date,
            parameter_value_dict=inverted_parameters, # ! dictionary
            state_values_dict=mean_state_values_dict,
            print_debug=True
            )
        # print("original_parameters shape:", np.shape(original_parameters))
        # print("original_parameters:", original_parameters)
    
        # result = hbvsaskModelObject.run_model_single_time_stamp(hbvsaskModelObject.end_date, parameters=original_parameters, raise_exception_on_model_break=True)
        
        return y_t_model
        


    mus = standard_parameter_matrix.mean(axis=0)
    sigmas = standard_parameter_matrix.std(axis=0)

    print("== Standard parameter samples matrix metric ==")
    print(mus)
    print(sigmas)


    distribution_r = chaospy.J(
        chaospy.Normal(0, 1), 
        chaospy.Normal(0, 1), 
        chaospy.Normal(0, 1),
        chaospy.Normal(0, 1), 
        chaospy.Normal(0, 1), 
        chaospy.Normal(0, 1),
        chaospy.Normal(0, 1))
    
    
    # i.i.d. sample in standard Gauss
    samples_r = distribution_r.sample(1000, rule="sobol") # sobol quasi random --> try pure random
    print(f"num_samples: ", samples_r.shape)
    
    # DEFAULT_PAR_VALUES_DICT = {'TT': 0.0, 'C0': 0.5, 'ETF': 0.2, 'FC': 250,
    #                        'beta': 2.0, 'FRAC': 0.3, 'K2': 0.05,  'LP': 0.5,
    #                        'K1': 0.5, 'alpha': 2.0, 
    #                        'UBAS': 1, 'PM': 1, "M": 1.0, "VAR_M": 1e-4}
    
    default_sample = [[0.0], [0.5], [2.0], [0.2], [250], [0.3], [0.05]]
    
    samples_q_list = []
    for i in range(samples_r.shape[1]):
        try:
            sample = samples_r[:, i:i+1]  # shape (n_params, 1)
            sample_q = inverse(sample)
            if np.any(np.isnan(sample_q)):
                print(f"NaN detected in inverted sample {i}:")
                print("Input sample_r:", sample.flatten())
                print("Output sample_q:", sample_q.flatten())
                samples_q_list.append(default_sample)
            else:   
                samples_q_list.append(sample_q)
        except Exception as e:
            print(f"Exception for sample {i}: {e}")
            print("Input sample_r:", sample.flatten())
    samples_q = np.hstack(samples_q_list)
    
    print("finished inversion")

    expansion = chaospy.generate_expansion(3, distribution_r)
    print("chaos: generated expansion")
    evaluations = np.array([evaluate(sample) for sample in samples_q.T])
    print("Done evaluations in chaos")