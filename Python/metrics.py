"""
Created: 2013-10-31 14:56:21

Author: Spencer Lyon

Various tools useful for doing standard econometric analysis.

"""
import numpy as np
import numpy.linalg as la
import pandas as pd
import statsmodels.formula.api as sm
from scipy import stats


__all__ = ['iv2sls', 'bootstrap', 'wald', "wu_test", 'cluster_se']


def iv2sls(form, data, instruments=None):
    """
    Perform instrumental variables two-stage least squares regression.

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

    correct_se : pd.DataFrame
        A pandas Series with the correct standard errors, t-statistics,
        and p-values

    Notes
    =====
    Note that if you ask for the printout of fit, the standard errors,
    t statistics, and p-vales will all be incorrect. They should instead
    be based on the

    """
    end_reg, inst = instruments
    endog, exog = form.split("~")
    s2_form = form
    z_form = endog + ' ~ ' + " + ".join(inst)
    for i in map(str.strip, exog.split("+")):
        if i not in end_reg:
            z_form = z_form + " + " + i

    for i in end_reg:
        s1_form = i + ' ~ ' + exog.replace(i, " + ".join(inst))
        s1_fit = sm.ols(s1_form, data=data).fit()
        pred_name = 'inst_%s' % (i)
        data[pred_name] = s1_fit.predict()
        s2_form = s2_form.replace(i, pred_name)

    fit = sm.ols(s2_form, data=data).fit()

    X = sm.ols(form, data=data).data.exog
    Z = sm.ols(z_form, data=data).data.exog

    good_var = la.inv((X.T.dot(Z).dot(la.inv(np.dot(Z.T, Z)).dot(Z.T).dot(X))))
    good_se = np.diag(good_var)
    t_values = fit.params / good_se
    p_values = stats.t.sf(np.abs(t_values), fit.df_resid) * 2

    good_table = pd.DataFrame({'Standard Errors': good_se,
                               't': t_values,
                               'P>|t|': p_values},
                              index=fit.params.index)

    return fit, good_table


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


def wu_test(form, data, variable):
    """
    Perform the Wu endogeneity test. This test is carried out in 3
    steps:

    1. Regress the variable in question on all other exogenous variables
    2. Add the residuals from the aforementioned regression to the main
       model
    3. Examine the p-value associated with the residual term from the
       updated model from part 2. A statistically significant coeff
       indicates that the tested variable is indeed endogenous.

    Parameters
    ==========
    form : str
        The statsmodels (patsy) formula for the model

    data : pandas.DataFrame
        The pandas DataFrame holding the data for the regression

    variable : str
        The string naming the variable (column) for which to perform
        the test

    Returns
    =======
    fit : statsmodels.regression.linear_model.RegressionResultsWrapper
        The statsmodels fit object associated with the Wu test.
    """
    endog, exog = form.split("~")
    s2_form = form

    o_exog = map(str.strip, exog.split('+'))
    o_exog.remove(variable)

    s1_form = variable + ' ~ ' + " + ".join(o_exog)
    s1_fit = sm.ols(s1_form, data=data).fit()
    res_name = 'resid_%s' % (variable)
    data[res_name] = s1_fit.resid
    s2_form += " + %s" % (res_name)

    fit = sm.ols(s2_form, data=data).fit()

    p_val = fit.pvalues['resid_EXP']
    endog_bool = 'not' if p_val >= 0.05 else 'is'
    msg = "WU TEST: The p_value of the added residual is %.4e"
    msg += "\n\t This %s significant at the alpha=0.05 level\n\n"
    print(msg % (p_val, endog_bool))

    return fit


def cluster_se(fit, gp_name):
    """
    Compute robust "clustered" standard errors.

    Parameters
    ==========
    fit : statsmodels.regression.linear_model.RegressionResultsWrapper
        The statsmodels fit object obtained from the original regression

    gp_name : str
        The name of the group on which the clustering should happen.
        This needs to be the name of a column in the original DataFrame
        used to create and fit the model.

    Returns
    =======
    ser : pd.Series
        A pandas Series with the variable names and robust standard
        errors.

    """
    from statsmodels.stats.sandwich_covariance import cov_cluster
    grp = fit.model.data.frame[gp_name]
    se = np.diag(cov_cluster(fit, grp)) ** (1/2.)
    return pd.Series(se, index=fit.params.index)
