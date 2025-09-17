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
    
    hbvsaskModelObject = hbvmodel.HBVSASKModel(
        configurationObject=configuration_file,
        inputModelDir=inputModelDir,
        workingDir=workingDir,
        basin=basin,
        writing_results_to_a_file=writing_results_to_a_file,
        plotting=plotting
    )
    
    return hbvsaskModelObject




def construct_polynomial_chaos_expansion(mean_state_values_dict, transport_map, scaler, pce_samples, eval_cap, expansion_order):
    
    # mean_state_values_dict: dictionary with keys {SWE, SMS, S1, S2}
    # transport_map
    # scaler
    # pce_samples: (num_samples, dim) --> (particle_count, 7)

    hbvsaskModelObject = setup_HBV()
    num_samples = pce_samples


    # !! standard_parameters (dim, num_samples) / (dim, 1) [chaospy format, transportmap format, standard_params_matrix format]
    def inverse(standard_parameters):
        """Inverse parameter distribution from reference (SNV) back to target (original / exponential)"""
        
        
        # Inverse transport map approximation
        X_reconstruct = transport_map.Inverse(np.empty((0, standard_parameters.shape[1])), standard_parameters)
    
        # Inverse scaling
        X_reconstruct = scaler.inverse_transform(X_reconstruct.T).T

    
        # Inverse logarithm
        for i in range(X_reconstruct.shape[0]):
            X_reconstruct[i, :] = np.exp(X_reconstruct[i, :])
        
        return X_reconstruct


    # inverted_parameters: (dim, 1)
    def evaluate(inverted_parameters):
        """Evaluates the hydrological model for given parameters in original (target / exponential) form"""
        
        
        
        param_names = ['TT', 'C0', 'beta', 'ETF', 'FC', 'FRAC', 'K2']
        param_dict = {name: float(val) for name, val in zip(param_names, inverted_parameters)}
  
        
        # unique_index_model_run, y_t_model, y_t_observed, x_t_plus_1, parameter_value_dict
        _, y_t_model, _, _, _ = transport_pce_pipeline.run_model_single_time_stamp_single_particle(
            hbvsaskModelObject=hbvsaskModelObject,
            date_of_interest=hbvsaskModelObject.end_date,
            parameter_value_dict=param_dict, # ! dictionary
            state_values_dict=mean_state_values_dict
            )
        
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
    samples_r = distribution_r.sample(num_samples, rule="latin_hypercube")
    print(f"num_samples: ", samples_r.shape)
        
    
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
            sample = samples_r[:, i:i+1]  # shape: (n_params, 1)
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
                countBounds += 1
            else:   
                samples_q_list.append(sample_q)
                # filter samples_r for those samples that are invalid
                samples_r_list.append(sample)
        except Exception as e:
            print(f"Exception for sample {i}: {e}")
            print("Input sample_r:", sample.flatten())
    
    
    print("chaospy - finished inversion")
    print(f"{num_samples - (countNaN + countBounds)}/{num_samples} samples were correctly inversed / could be used for chaospy")
    print(f"{countNaN} times there was an issue with NaN")
    print(f"{countBounds} times the inverted parameter values were out of bounds")


    # TODO: check expansion order
    expansion = chaospy.generate_expansion(expansion_order, distribution_r)
    print(f"chaos: generated expansion of order {expansion_order}.")
    
    
    evaluations = []
    valid_samples_q = []
    valid_samples_r = []
    
    countSkip = 0
    
    for sample_q, sample_r in zip(samples_q_list, samples_r_list):
        y = evaluate(sample_q)
        # Refuse abnormal outputs
        if np.isnan(y) or np.isinf(y) or y < 0 or y > eval_cap: 
            print(f"Abnormal evaluation: {y}, skipping sample.")
            countSkip += 1
            continue
        evaluations.append(y)
        valid_samples_q.append(sample_q.reshape(-1, 1))
        valid_samples_r.append(sample_r)
    
    print(f"Done evaluations in chaos. Limit of evaluations was {eval_cap}.")
    print(f"Skipped {countSkip}/{num_samples - (countNaN + countBounds)} samples")
    print(f"Remaining samples: {num_samples - (countNaN + countBounds + countSkip)}")
    
    evaluations = np.array(evaluations)
    valid_samples_r = np.hstack(valid_samples_r)   
    
    print("Approximating model...")
    model_approx = chaospy.fit_regression(expansion, valid_samples_r, evaluations)
    
    mean = chaospy.E(model_approx, distribution_r)
    std = chaospy.Std(model_approx, distribution_r)

    print(f"chaospy approximated mean: {mean}")
    print(f"chaospy approximated std: {std}")
    
    
    return model_approx, mean, std