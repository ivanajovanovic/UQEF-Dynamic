"""
This module contains a framework to summarize the distance between the model simulations and corresponding observations
by calculating different likelihood values.

@author: Ivana Jovanovic
"""
import pandas as pd
import os
import numpy as np
import sys


def _prepare_data_to_calculate_likelihood(DF, column_name='Value') -> np.ndarray:
    #TODO-Ivana check when DF only contains TimeStampas as index column and Value column
    if isinstance(DF, pd.core.frame.DataFrame):
        return DF[column_name].values
    elif isinstance(DF, pd.core.series.Series):
        return DF.values
    elif isinstance(DF, np.ndarray):
        return DF
    else:
        raise Exception('You want to calculate likelihood using data from the following type {}, which is not accepted! \
        Accepted are: pandas.DataFrame, pandas.Series and numpy.ndarray()'.format(type(DF)))


def calculateMAE(measuredDF, simulatedDF, measuredDF_column_name='Value', simulatedDF_column_name='Value', **kwargs):
    measuredDF = _prepare_data_to_calculate_likelihood(DF=measuredDF, column_name=measuredDF_column_name)
    simulatedDF = _prepare_data_to_calculate_likelihood(DF=simulatedDF, column_name=simulatedDF_column_name)

    if measuredDF.size == simulatedDF.size:
        return np.mean(np.abs(measuredDF - simulatedDF))
    else:
        return np.nan


def calculateMSE(measuredDF, simulatedDF, measuredDF_column_name='Value', simulatedDF_column_name='Value', **kwargs):
    measuredDF = _prepare_data_to_calculate_likelihood(DF=measuredDF, column_name=measuredDF_column_name)
    simulatedDF = _prepare_data_to_calculate_likelihood(DF=simulatedDF, column_name=simulatedDF_column_name)

    if measuredDF.size == simulatedDF.size:
        #squared_error = np.square(np.subtract(measuredDF, simulatedDF))
        squared_error = (measuredDF - simulatedDF)**2
        return np.nanmean(squared_error)
    else:
        return np.nan


def calculateRMSE(measuredDF, simulatedDF, measuredDF_column_name='Value', simulatedDF_column_name='Value', **kwargs):
    measuredDF = _prepare_data_to_calculate_likelihood(DF=measuredDF, column_name=measuredDF_column_name)
    simulatedDF = _prepare_data_to_calculate_likelihood(DF=simulatedDF, column_name=simulatedDF_column_name)

    if measuredDF.size == simulatedDF.size:
        squared_error = np.square(np.subtract(measuredDF, simulatedDF))
        return np.sqrt(np.nanmean(squared_error))
    else:
        return np.nan


def calculateNRMSE(measuredDF, simulatedDF, measuredDF_column_name='Value', simulatedDF_column_name='Value', **kwargs):
    measuredDF = _prepare_data_to_calculate_likelihood(DF=measuredDF, column_name=measuredDF_column_name)
    simulatedDF = _prepare_data_to_calculate_likelihood(DF=simulatedDF, column_name=simulatedDF_column_name)

    if measuredDF.size == simulatedDF.size:
        try:
            return calculateRMSE(measuredDF, simulatedDF, measuredDF_column_name, simulatedDF_column_name) \
                   / (np.max(measuredDF) - np.min(measuredDF))
        except ZeroDivisionError:
            return np.nan
    else:
        return np.nan


def calculateRSR(measuredDF, simulatedDF, measuredDF_column_name='Value', simulatedDF_column_name='Value', **kwargs):
    """
    RMSE-observations standard deviation ratio
    Corresponding paper:
    Moriasi, Arnold, Van Liew, Bingner, Harmel, Veith, 2007, Model Evaluation Guidelines for Systematic Quantification of Accuracy in Watershed Simulations
    output:
        rsr: RMSE-observations standard deviation ratio
    """
    measuredDF = _prepare_data_to_calculate_likelihood(DF=measuredDF, column_name=measuredDF_column_name)
    simulatedDF = _prepare_data_to_calculate_likelihood(DF=simulatedDF, column_name=simulatedDF_column_name)

    if measuredDF.size == simulatedDF.size:
        try:
            return calculateRMSE(measuredDF, simulatedDF, measuredDF_column_name, simulatedDF_column_name) / np.std(
                measuredDF)
        except ZeroDivisionError:
            return np.nan
    else:
        return np.nan


def calculateBIAS(measuredDF, simulatedDF, measuredDF_column_name='Value', simulatedDF_column_name='Value', **kwargs):
    """
    Bias
        .. math::
         Bias=\\frac{1}{N}\\sum_{i=1}^{N}(e_{i}-s_{i})
    :measuredDF: Observed data to compared with simulation data.
    :type: pandas.DataFrame or pandas.Series or np.array
    :simulatedDF: simulation data to compared with evaluation data
    :type: pandas.DataFrame or pandas.Series or np.array
    :return: Bias
    :rtype: float
    """
    measuredDF = _prepare_data_to_calculate_likelihood(DF=measuredDF, column_name=measuredDF_column_name)
    simulatedDF = _prepare_data_to_calculate_likelihood(DF=simulatedDF, column_name=simulatedDF_column_name)

    if measuredDF.size == simulatedDF.size:
        residual = np.subtract(measuredDF, simulatedDF)
        # return float(np.nansum(residual) / len(measuredDF))
        return float(np.abs(np.nanmean(residual)))
    else:
        return np.nan


def calculatePBIAS(measuredDF, simulatedDF, measuredDF_column_name='Value', simulatedDF_column_name='Value', **kwargs):
    """
    Bias
        .. math::
         Bias=sum_{i=1}^{N}(e_{i}-s_{i})/sum_{i=1}^{N}()*100
    """
    measuredDF = _prepare_data_to_calculate_likelihood(DF=measuredDF, column_name=measuredDF_column_name)
    simulatedDF = _prepare_data_to_calculate_likelihood(DF=simulatedDF, column_name=simulatedDF_column_name)

    if measuredDF.size == simulatedDF.size:
        residual = np.subtract(measuredDF, simulatedDF)
        try:
            return float(np.nansum(residual) / np.nansum(measuredDF)) * 100
        except ZeroDivisionError:
            return np.nan
    else:
        return np.nan


def calculateROCE(measuredDF, simulatedDF, measuredDF_column_name='Value', simulatedDF_column_name='Value', **kwargs):
    measuredDF = _prepare_data_to_calculate_likelihood(DF=measuredDF, column_name=measuredDF_column_name)
    simulatedDF = _prepare_data_to_calculate_likelihood(DF=simulatedDF, column_name=simulatedDF_column_name)
    if measuredDF.size == simulatedDF.size:
        try:
            return np.abs(np.nansum(simulatedDF) - np.nansum(measuredDF)) / np.nansum(measuredDF)
        except ZeroDivisionError:
            return np.nan
    else:
        return np.nan


def calculateNSE(measuredDF, simulatedDF, measuredDF_column_name='Value', simulatedDF_column_name='Value', **kwargs):
    """
    Nash-Sutcliffe model efficinecy
        .. math::
         NSE = 1-\\frac{\\sum_{i=1}^{N}(e_{i}-s_{i})^2}{\\sum_{i=1}^{N}(e_{i}-\\bar{e})^2}
    :measuredDF: Observed data to compared with simulation data.
    :type: pandas.DataFrame or pandas.Series or np.array
    :simulatedDF: simulation data to compared with evaluation data
    :type: pandas.DataFrame or pandas.Series or np.array
    :return: Nash-Sutcliff model efficiency
    :rtype: float
    """
    measuredDF = _prepare_data_to_calculate_likelihood(DF=measuredDF, column_name=measuredDF_column_name)
    simulatedDF = _prepare_data_to_calculate_likelihood(DF=simulatedDF, column_name=simulatedDF_column_name)

    if measuredDF.size == simulatedDF.size:
        #numerator = np.square(np.subtract(measuredDF, simulatedDF))
        #denominator = np.square(np.subtract(measuredDF, np.mean(measuredDF)))
        numerator = (measuredDF - simulatedDF)**2
        denominator = (measuredDF - np.nanmean(measuredDF))**2
        try:
            return float(1 - (np.nansum(numerator) / np.nansum(denominator)))
        except ZeroDivisionError:
            return np.nan
    else:
        return np.nan


def calculateLogNSE(measuredDF, simulatedDF, measuredDF_column_name='Value', simulatedDF_column_name='Value', **kwargs):
    """
    log Nash-Sutcliffe model efficiency
        .. math::
         NSE = 1-\\frac{\\sum_{i=1}^{N}(log(e_{i})-log(s_{i}))^2}{\\sum_{i=1}^{N}(log(e_{i})-log(\\bar{e})^2}-1)*-1
    :measuredDF: Observed data to compared with simulation data.
    :type: pandas.DataFrame or pandas.Series or np.array
    :simulatedDF: simulation data to compared with evaluation data
    :type: pandas.DataFrame or pandas.Series or np.array
    :epsilon: Value which is added to simulation and evaluation data to errors when simulation or evaluation data has zero values
    :type: float or list

    :return: log Nash-Sutcliffe model efficiency
    :rtype: float
    """
    measuredDF = _prepare_data_to_calculate_likelihood(DF=measuredDF, column_name=measuredDF_column_name)
    simulatedDF = _prepare_data_to_calculate_likelihood(DF=simulatedDF, column_name=simulatedDF_column_name)

    epsilon = kwargs.get('epsilon', None)
    if not epsilon or epsilon is None or epsilon == 0:
        epsilon = sys.float_info.epsilon

    if measuredDF.size == simulatedDF.size:
        measuredDF = measuredDF + epsilon
        simulatedDF = simulatedDF + epsilon
        #numerator = np.square(np.subtract(np.log(measuredDF), np.log(simulatedDF)))
        #denominator = np.square(np.subtract(np.log(measuredDF), np.mean(np.log(measuredDF))))
        numerator = (np.log(measuredDF) - np.log(simulatedDF))**2
        denominator = (np.log(measuredDF) - np.mean(np.log(measuredDF)))**2
        try:
            return float(1 - np.nansum(numerator) / np.nansum(denominator))
        except ZeroDivisionError:
            return np.nan
    else:
        return np.nan


def calculateLogGaussian(measuredDF, simulatedDF, measuredDF_column_name='Value', simulatedDF_column_name='Value',
                         **kwargs):
    """
    Logarithmic Gaussian probability distribution of the error/residual signal
    :measuredDF: Observed data to compared with simulation data.
    :type: pandas.DataFrame or pandas.Series or np.array
    :simulatedDF: simulation data to compared with evaluation data
    :type: pandas.DataFrame or pandas.Series or np.array
    :return: Logarithmic Gaussian (Normals) probability distribution
    :rtype: float
    """
    measuredDF = _prepare_data_to_calculate_likelihood(DF=measuredDF, column_name=measuredDF_column_name)
    simulatedDF = _prepare_data_to_calculate_likelihood(DF=simulatedDF, column_name=simulatedDF_column_name)

    scale = np.mean(measuredDF) / 10
    if scale < .01:
        scale = .01
    if measuredDF.size == simulatedDF.size:
        y = (np.array(measuredDF) - np.array(simulatedDF)) / scale
        normpdf = -y**2 / 2 - np.log(np.sqrt(2 * np.pi))
        return np.mean(normpdf)
    else:
        return np.nan


def calculateCorrelationCoefficient(measuredDF, simulatedDF, measuredDF_column_name='Value',
                                    simulatedDF_column_name='Value', **kwargs):
    """
    Correlation Coefficient
        .. math::
         r = \\frac{\\sum ^n _{i=1}(e_i - \\bar{e})(s_i - \\bar{s})}{\\sqrt{\\sum ^n _{i=1}(e_i - \\bar{e})^2} \\sqrt{\\sum ^n _{i=1}(s_i - \\bar{s})^2}}
    :measuredDF: Observed data to compared with simulation data.
    :type: pandas.DataFrame or pandas.Series or np.array
    :simulatedDF: simulation data to compared with evaluation data
    :type: pandas.DataFrame or pandas.Series or np.array
    :return: Corelation Coefficient
    :rtype: float
    """
    measuredDF = _prepare_data_to_calculate_likelihood(DF=measuredDF, column_name=measuredDF_column_name)
    simulatedDF = _prepare_data_to_calculate_likelihood(DF=simulatedDF, column_name=simulatedDF_column_name)

    if measuredDF.size == simulatedDF.size:
        correlation_coefficient = np.corrcoef(measuredDF, simulatedDF)[0, 1]
        return correlation_coefficient
    else:
        return np.nan


def calculateKGE(measuredDF, simulatedDF, measuredDF_column_name='Value', simulatedDF_column_name='Value', **kwargs):
    """
    code from - https://github.com/thouska/spotpy/tree/master/spotpy
    Kling-Gupta Efficiency
    Corresponding paper:
    Gupta, Kling, Yilmaz, Martinez, 2009, Decomposition of the mean squared error and NSE performance criteria: Implications for improving hydrological modelling
    output:
        kge: Kling-Gupta Efficiency
    optional_output:
        cc: correlation
        alpha: ratio of the standard deviation
        beta: ratio of the mean
    """
    measuredDF = _prepare_data_to_calculate_likelihood(DF=measuredDF, column_name=measuredDF_column_name)
    simulatedDF = _prepare_data_to_calculate_likelihood(DF=simulatedDF, column_name=simulatedDF_column_name)

    return_all = kwargs.get('return_all', False)

    if measuredDF.size == simulatedDF.size:
        cc = np.corrcoef(measuredDF, simulatedDF)[0, 1]
        alpha = np.std(simulatedDF) / np.std(measuredDF)
        beta = np.sum(simulatedDF) / np.sum(measuredDF)
        kge = 1 - np.sqrt((cc - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
        if return_all:
            return kge, cc, alpha, beta
        else:
            return kge
    else:
        return np.nan


#####################################################


def calculateKGE_non_parametric(measuredDF, simulatedDF, measuredDF_column_name='Value',
                                simulatedDF_column_name='Value', **kwargs):
    """
    code from - https://github.com/thouska/spotpy/tree/master/spotpy
    Non parametric Kling-Gupta Efficiency
    Corresponding paper:
    Pool, Vis, and Seibert, 2018 Evaluating model performance: towards a non-parametric variant of the Kling-Gupta efficiency, Hydrological Sciences Journal.
    output:
        kge: Kling-Gupta Efficiency

    author: Nadine Maier and Tobias Houska
    optional_output:
        cc: correlation
        alpha: ratio of the standard deviation
        beta: ratio of the mean
    """
    measuredDF = _prepare_data_to_calculate_likelihood(DF=measuredDF, column_name=measuredDF_column_name)
    simulatedDF = _prepare_data_to_calculate_likelihood(DF=simulatedDF, column_name=simulatedDF_column_name)

    return_all = kwargs.get('return_all', False)

    if measuredDF.size == simulatedDF.size:
        ### pandas version of Separmann correlation coefficient
        a = pd.DataFrame({'eva': measuredDF, 'sim': simulatedDF})
        cc = a.ix[:,1].corr(a.ix[:,0], method='spearman')
        fdc_sim = np.sort(simulatedDF / (np.nanmean(simulatedDF) * len(simulatedDF)))
        fdc_obs = np.sort(measuredDF / (np.nanmean(measuredDF) * len(measuredDF)))
        alpha = 1 - 0.5 * np.nanmean(np.abs(fdc_sim - fdc_obs))
        beta = np.mean(simulatedDF) / np.mean(measuredDF)
        kge = 1 - np.sqrt((cc - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
        if return_all:
            return kge, cc, alpha, beta
        else:
            return kge
    else:
        return np.nan


def calculateCovariance(measuredDF, simulatedDF, measuredDF_column_name='Value', simulatedDF_column_name='Value',
                        **kwargs):
    """
    Covariance
        .. math::
         Covariance = \\frac{1}{N} \\sum_{i=1}^{N}((e_{i} - \\bar{e}) * (s_{i} - \\bar{s}))
    :measuredDF: Observed data to compared with simulation data.
    :type: pandas.DataFrame or pandas.Series or np.array
    :simulatedDF: simulation data to compared with evaluation data
    :type: pandas.DataFrame or pandas.Series or np.array
    :return: Covariance
    :rtype: float
    """
    measuredDF = _prepare_data_to_calculate_likelihood(DF=measuredDF, column_name=measuredDF_column_name)
    simulatedDF = _prepare_data_to_calculate_likelihood(DF=simulatedDF, column_name=simulatedDF_column_name)

    if measuredDF.size == simulatedDF.size:
        covariance = np.mean((measuredDF - np.mean(measuredDF))*(simulatedDF - np.mean(simulatedDF)))
        return covariance
    else:
        return np.nan


def calculateBraviasPearson(measuredDF, simulatedDF, measuredDF_column_name='Value', simulatedDF_column_name='Value',
                            **kwargs):
    measuredDF = _prepare_data_to_calculate_likelihood(DF=measuredDF, column_name=measuredDF_column_name)
    simulatedDF = _prepare_data_to_calculate_likelihood(DF=simulatedDF, column_name=simulatedDF_column_name)

    mean_measured = np.mean(measuredDF)
    mean_simulated = np.mean(simulatedDF)
    term1 = np.subtract(measuredDF, mean_measured)
    term2 = np.subtract(simulatedDF, mean_measured)
    term3 = np.subtract(simulatedDF, mean_simulated)
    term4 = np.prod(term1, term2)
    term5 = np.square(term1)
    term6 = np.square(term3)
    r_2 = np.square(np.sum(term4))/(np.sum(term5)*np.sum(term6))
    return r_2

#####################################################


_all_functions = [calculateMAE, calculateMSE,
                  calculateRMSE, calculateNRMSE, calculateRSR,
                  calculateBIAS, calculatePBIAS, calculateROCE,
                  calculateNSE, calculateLogNSE,
                  calculateLogGaussian, calculateCorrelationCoefficient,
                  calculateKGE]


def calculate_all_functions(measuredDF, simulatedDF, measuredDF_column_name='Value', simulatedDF_column_name='Value',
                            **kwargs):
    """
    Calculates all objective functions from
    and returns the results as a list of name/value pairs
    :param measuredDF: a sequence of evaluation data
    :type pandas.DataFrame or pandas.Series or np.array
    :param simulatedDF: a sequence of simulation data
    :type pandas.DataFrame or pandas.Series or np.array
    :return: A list of (name, value) tuples or dictionary
    """

    result = {}
    for f in _all_functions:
        # Check if the name is not private and attr is a function but not this
        try:
            #result.append((f.__name__, f(measuredDF, simulatedDF)))
            result[f.__name__] = f(measuredDF, simulatedDF, measuredDF_column_name, simulatedDF_column_name, **kwargs)
        except:
            #result.append((f.__name__, np.nan))
            result[f.__name__] = np.nan

    return result
