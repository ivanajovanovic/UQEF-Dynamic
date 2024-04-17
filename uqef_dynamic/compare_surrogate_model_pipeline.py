# read the assumed prior distribution over parameters
import chaospy as cp
from collections import defaultdict
import inspect
import matplotlib.pyplot as plt
import pandas as pd
import time

from common import utility


def fetch_and_evaluate_gpce_model_produced_by_uqef_statistics_class(
        statisticsObject, configurationObject, num_evaluations=100, sampling_rule="halton"):
    list_of_single_distr = []
    for param in configurationObject["parameters"]:
        # for now this is hard-coded
        if param["distribution"]=="Uniform":
            list_of_single_distr.append(cp.Uniform(param["lower"], param["upper"]))
    joint = cp.J(*list_of_single_distr)
    joint_standard = cp.J(cp.Uniform(), cp.Uniform(), cp.Uniform(), cp.Uniform(), cp.Uniform())
    joint_standard_min_1_1 = cp.J(
        cp.Uniform(lower=-1, upper=1),cp.Uniform(lower=-1, upper=1), cp.Uniform(lower=-1, upper=1),
        cp.Uniform(lower=-1, upper=1), cp.Uniform(lower=-1, upper=1)
    )

    samples_to_evaluate_gPCE = joint.sample(num_evaluations, rule=sampling_rule) # 'sobol' 'random'
    samples_to_evaluate_gPCE_transformed = utility.transformation_of_parameters_var1(
        samples_to_evaluate_gPCE, joint, joint_standard_min_1_1)

    print(samples_to_evaluate_gPCE_transformed.shape)

    gPCE_model = defaultdict()
    for single_date in statisticsObject.pdTimesteps:
        gPCE_model[single_date] = statisticsObject.result_dict["Q_cms"][single_date]['gPCE']

    print(type(gPCE_model[list(gPCE_model.keys())[0]]))
    print(gPCE_model[list(gPCE_model.keys())[0]].values)

    start = time.time()

    gPCE_model_evaluated = defaultdict()
    for single_date in statisticsObject.pdTimesteps:
        gPCE_model = statisticsObject.result_dict["Q_cms"][single_date]['gPCE']
        gPCE_model_evaluated[single_date] = gPCE_model(samples_to_evaluate_gPCE_transformed.T)

    end = time.time()
    runtime = end - start
    print(f"Time needed for evaluating {samples_to_evaluate_gPCE_transformed.shape[1]} \
    gPCE model for {len(statisticsObject.pdTimesteps)} time steps is: {runtime}")

    return gPCE_model_evaluated


def creted_df_from_gpce_model_evaluations(gPCE_model_evaluated):
    gpc_eval_df = pd.DataFrame.from_dict(gPCE_model_evaluated, orient="index", columns=range(1000))
    # Re-compute mean...
    gpc_eval_df['new_E'] = gpc_eval_df.mean(numeric_only=True, axis=1)
    gpc_eval_df = gpc_eval_df.loc[:, ['new_E', ]]
    return gpc_eval_df


if __name__ == '__main__':
    # Trying out some simlpe thinkgs
    # plt.hist(samples_1d, bins=50, density=True, alpha=0.5)
    # t = np.linspace(-3, 3, 400)
    # distribution = cp.GaussianKDE(samples_1d, h_mat=0.05**2)
    # plt.plot(t, distribution.pdf(t), label="0.05")
    # plt.legend()
    # plt.show()
    statisticsObject = None
    configurationObject = None

    gPCE_model_evaluated = fetch_and_evaluate_gpce_model_produced_by_uqef_statistics_class(
        statisticsObject, configurationObject, num_evaluations=100, sampling_rule="halton")

    # Looking closer for a particular time-step
    # learn the distribution of gPCE evaluations for a particular time step
    date_QoI = pd.Timestamp('2006-05-27 00:00:00')
    samples_of_gPCE_evals = gPCE_model_evaluated[date_QoI]
    distribution = cp.GaussianKDE(samples_of_gPCE_evals, h_mat=0.05)

    gpc_eval_df = creted_df_from_gpce_model_evaluations(gPCE_model_evaluated)
    print(gpc_eval_df)

