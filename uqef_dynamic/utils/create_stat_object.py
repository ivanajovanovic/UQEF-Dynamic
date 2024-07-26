from uqef_dynamic.models.larsim import LarsimStatistics
from uqef_dynamic.models.linearDampedOscillator import LinearDampedOscillatorStatistics
from uqef_dynamic.models.ishigami import IshigamiStatistics
from uqef_dynamic.models.productFunction import ProductFunctionStatistics
from uqef_dynamic.models.hbv_sask import HBVSASKStatistics
from uqef_dynamic.models.pybamm import pybammStatistics
# from uqef_dynamic.models.time_dependent_baseclass import time_dependent_statistics


def create_statistics_object(configuration_object, uqsim_args_dict, workingDir, model="hbvsask"):
    """
    Note: hardcoded for a couple of currently supported models
    :param configuration_object:
    :param uqsim_args_dict:
    :param workingDir:
    :param model: "larsim" | "hbvsask"
    :return:
    """
    # TODO make this function more general or move it somewhere else
    if model == "larsim":
        statisticsObject = LarsimStatistics.LarsimStatistics(configuration_object, workingDir=workingDir,
                                                                   parallel_statistics=uqsim_args_dict[
                                                                       "parallel_statistics"],
                                                                   mpi_chunksize=uqsim_args_dict["mpi_chunksize"],
                                                                   unordered=False,
                                                                   uq_method=uqsim_args_dict["uq_method"],
                                                                   compute_Sobol_t=uqsim_args_dict["compute_Sobol_t"],
                                                                   compute_Sobol_m=uqsim_args_dict["compute_Sobol_m"])
    elif model == "hbvsask":
        statisticsObject = HBVSASKStatistics.HBVSASKStatistics(
            configurationObject=configuration_object,
            workingDir=workingDir,
            inputModelDir=uqsim_args_dict["inputModelDir"],
            sampleFromStandardDist=uqsim_args_dict["sampleFromStandardDist"],
            parallel_statistics=uqsim_args_dict["parallel_statistics"],
            mpi_chunksize=uqsim_args_dict["mpi_chunksize"],
            uq_method=uqsim_args_dict["uq_method"],
            compute_Sobol_t=uqsim_args_dict["compute_Sobol_t"],
            compute_Sobol_m=uqsim_args_dict["compute_Sobol_m"],
            compute_Sobol_m2=uqsim_args_dict["compute_Sobol_m2"]
        )
    elif model == "battery":
        statisticsObject = pybammStatistics.pybammStatistics(
            configurationObject=configuration_object,
            workingDir=workingDir,
            inputModelDir=uqsim_args_dict["inputModelDir"],
            sampleFromStandardDist=uqsim_args_dict["sampleFromStandardDist"],
            parallel_statistics=uqsim_args_dict["parallel_statistics"],
            mpi_chunksize=uqsim_args_dict["mpi_chunksize"],
            unordered=False,
            uq_method=uqsim_args_dict["uq_method"],
            compute_Sobol_t=uqsim_args_dict["compute_Sobol_t"],
            compute_Sobol_m=uqsim_args_dict["compute_Sobol_m"],
            compute_Sobol_m2=uqsim_args_dict["compute_Sobol_m2"],
            save_all_simulations=uqsim_args_dict["save_all_simulations"],
            collect_and_save_state_data=uqsim_args_dict["collect_and_save_state_data"],
            store_qoi_data_in_stat_dict=uqsim_args_dict["store_qoi_data_in_stat_dict"],
            store_gpce_surrogate_in_stat_dict=uqsim_args_dict["store_gpce_surrogate_in_stat_dict"],
            instantly_save_results_for_each_time_step=uqsim_args_dict["instantly_save_results_for_each_time_step"]
        )
    else:
        raise ValueError("Model not supported")
        # statisticsObject = time_dependent_statistics.TimeDependentStatistics(
        #     configurationObject=configuration_object,
        #     workingDir=workingDir,
        #     inputModelDir=uqsim_args_dict["inputModelDir"],
        #     sampleFromStandardDist=uqsim_args_dict["sampleFromStandardDist"],
        #     parallel_statistics=uqsim_args_dict["parallel_statistics"],
        #     mpi_chunksize=uqsim_args_dict["mpi_chunksize"],
        #     uq_method=uqsim_args_dict["uq_method"],
        #     compute_Sobol_t=uqsim_args_dict["compute_Sobol_t"],
        #     compute_Sobol_m=uqsim_args_dict["compute_Sobol_m"],
        #     compute_Sobol_m2=uqsim_args_dict["compute_Sobol_m2"]
        # )

    return statisticsObject