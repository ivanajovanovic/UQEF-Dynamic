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
        # Inverse transport map approximation
        X_reconstruct = transport_map.Inverse(np.empty((0, standard_parameters.shape[1])), standard_parameters)
        # Inverse scaling
        X_reconstruct = scaler.inverse_transform(X_reconstruct.T).T
        # Inverse logarithm
        print("TODO: need to adjust logarithm for different distributions!")
        for i in range(X_reconstruct.shape[0]):
            X_reconstruct[i, :] = np.exp(X_reconstruct[i, :])
        
        
        return X_reconstruct


    
    def evaluate(original_parameters):
        """Evaluates the hydrological model for given parameters in original (target / exponential) form"""
        
        # TODO: check argument values
        result = transport_pce_pipeline.run_model_single_time_stamp_single_particle(
            hbvsaskModelObject=hbvsaskModelObject,
            date_of_interest=hbvsaskModelObject.start_date_predictions,
            parameter_value_dict=original_parameters,
            state_values_dict=mean_state_values_dict
            )
        
        return result
        


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
    samples_r = distribution_r.sample(1000, rule="sobol")
    print(f"num_samples: ", samples_r.shape)
    samples_q = inverse(samples_r)
    print("finished inversion")

    expansion = chaospy.generate_expansion(3, distribution_r)
    print("chaos: generated expansion")
    evaluations = np.array([evaluate(sample) for sample in samples_q.T])
    print("Done evaluations in chaos")