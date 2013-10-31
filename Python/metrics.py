"""
Created: 2013-10-31 14:56:21

Author: Spencer Lyon

Various tools useful for doing standard econometric analysis.

"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as sm


__all__ = ['iv2sls', 'bootstrap', 'wald']


def iv2sls(form, data, instruments=None):
    """
    Perform instrumental variables two-state least squares regression.

    As an example in this docstring imagine we wish to estimate the
    model given by

    .. math::

        y = b_1 + b_2 a + b_3 b + b_4 c

    where c is an endogenous regressor to be instrumented using
    variables d and e.

    Parameters
    ==========
    form : str
        A statsmodels formula for the model you would like to estimate.
        This should contain all regressors that are included in the
        theoretical model, including the endogenous regressors. For the
        example above, this would be `'y ~ a + b + c'`

    data : pandas.DataFrame
        The pandas DataFrame object containing the data for the
        regression.

    instruments : tuple
        A 2 element tuple where each element is a list. The first list
        contains strings giving the column name of all variables from
        `form` that are endogenous and should be instrumented out. The
        second list is also a list of strings, but they are the
        instruments to be used. In the example from above this would be
        `(['c'], ['d', 'e'])`

    Returns
    =======
    fit : statsmodels.regression.linear_model.RegressionResultsWrapper
        The statsmodels fit object corresponding to the instrumental
        variables two-stage least squares estimator for the model.

    TODO: Add correction for standard errors and t-stat/p-value

    """
    end_reg, inst = instruments
    exog, endog = form.split("~")
    s2_form = form

    for i in end_reg:
        s1_form = i + ' ~ ' + endog.replace(i, " + ".join(inst))
        s1_fit = sm.ols(s1_form, data=data).fit()
        pred_name = 'inst_%s' % (i)
        data[pred_name] = s1_fit.predict()
        s2_form = s2_form.replace(i, pred_name)

    fit = sm.ols(s2_form, data=data).fit()

    return fit


def bootstrap(fit, reps=100):
    """
    Simple bootstrap function to estimate asymptotic variance of OLS
    parameter estimate.

    Parameters
    ==========
    fit : statsmodels.regression.linear_model.RegressionResultsWrapper
        This is the fit obtained through sm.ols(args).fit(). It contains
        all information regarding the model, as well as provides access
        to the model's data.

    reps : int, optional(default=100)
        The number of repetitions to

    Returns
    =======
    mean : np.array
        The estimated mean from the bootstrap

    var : np.array
        The estimated asymptotic covariance matrix from the bootstrap

    """
    b = fit.params.values
    X = fit.model.data.exog
    y = fit.model.data.endog
    n = int(fit.nobs)  # Number of observations

    b_R = np.empty((reps, b.size))
    for i in range(reps):
        # Sample with replacement
        to_choose = list(np.random.choice(n, n, replace=True))
        y_i = y[to_choose]
        x_i = X[to_choose]

        # Estimate b(r)
        b_R[i, :] = np.linalg.lstsq(x_i, y_i)[0]

    est_var = np.cov(b_R, rowvar=0, ddof=1)

    return b_R.mean(0), est_var


def wald(s, res):
    """
    Perform wald test of single constraint.

    Parameters
    ==========
    s : string
        A statsmodels string representing the constraint.

    fit : statsmodels.regression.linear_model.RegressionResultsWrapper
        The fit from the regression that should be used to perform the
        test. This is typically the result of calling sm.ols(args).fit()

    Returns
    =======
    t_value : float
        The wald statistic (t-value).

    p_value : float
        The p-value associated with the test statistic.

    """
    msg = 'Wald test %s\n\tt_stat: %.4e\n\tp-value:%.4e\n'
    t = res.t_test(s)
    print(msg % (s, t.tvalue, t.pvalue))

    return t.tvalue, t.pvalue
