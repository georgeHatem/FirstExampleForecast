import pytest

import forecasthelper as fh

import pandas as pd
import random as rand
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess
from sklearn.multioutput import RegressorChain
from datetime import datetime
from sys import maxsize

def test__fixdate_basecase():
    inputParam = pd.DataFrame({'Other': [1, 2]}, index=pd.Series([datetime(year=1970, month=1, day=1), datetime(year=1970, month=1, day=2)], name='Date'))
    desired = 'datetime64'
    assert fh._fixDate(inputParam).index.inferred_type == desired

def test__fixdate_incolumns():
    inputParam = pd.DataFrame({'Date': pd.Series([datetime(year=1970, month=1, day=1), datetime(year=1970, month=1, day=2)]),
                                 'Other': [1, 2]})
    desired = 'datetime64'
    assert fh._fixDate(inputParam).index.inferred_type == desired

def test__fixdate_integer_neat():
    inputParam = pd.DataFrame({'Date':  pd.Series([datetime(year=1970, month=1, day=1), datetime(year=1970, month=1, day=2)]).apply(lambda x: x.value),
                                 'Other': [1, 2]})
    desired = 'datetime64'
    assert fh._fixDate(inputParam).index.inferred_type == desired

def test__fixdate_string_neat():
    inputParam = pd.DataFrame({'Date':  pd.Series([datetime(year=1970, month=1, day=1), datetime(year=1970, month=1, day=2)]).apply(lambda x: x.isoformat()),
                                 'Other': [1, 2]})
    desired = 'datetime64'
    assert fh._fixDate(inputParam).index.inferred_type == desired

def test__fixdate_integer_random():
    inputParam = pd.DataFrame({'Date':  [rand.randrange(maxsize), rand.randrange(maxsize)],
                                 'Other': [1, 2]})
    desired = 'datetime64'
    assert fh._fixDate(inputParam).index.inferred_type == desired

def test__fixdate_string():
    inputParam = pd.DataFrame({'Date':  ["1970-01-01", "1/2/1970"],
                                 'Other': [1, 2]})
    desired = 'datetime64'
    assert fh._fixDate(inputParam).index.inferred_type == desired

def test__fixdate_emptyerror():
    inputParam = pd.DataFrame()
    with pytest.raises(ValueError):
        fh._fixDate(inputParam)

def test__fixdate_failerror():
    inputParam = pd.DataFrame({'Date': ["hi", "bye"], 'Other': [1, 2]})
    with pytest.raises(TypeError):
        fh._fixDate(inputParam)

def test_detrend_target():
    inputParam = [pd.DataFrame({'Other': [1, 2]}, index=pd.Series([datetime(year=1970, month=1, day=1), datetime(year=1970, month=1, day=2)], name='Date')), "Other"]
    dptest = DeterministicProcess(index=inputParam[0].index, constant=True, order=1, drop=True)
    lrtest = LinearRegression(fit_intercept=False).fit(dptest.in_sample(), inputParam[0]['Other'])
    
    desired = [pd.DataFrame(inputParam[0]['Other'] - lrtest.predict(dptest.in_sample())), lrtest.predict(dptest.in_sample()), dptest.in_sample()]

    output = fh.detrend(*inputParam)
    assert output[0].equals(desired[0])
    assert (output[1].predict(dptest.in_sample()) == desired[1]).all()
    assert output[2].in_sample().equals(desired[2])

def test_detrend_notarget():
    inputParam = [pd.DataFrame({'Other': [1, 2]}, index=pd.Series([datetime(year=1970, month=1, day=1), datetime(year=1970, month=1, day=2)], name='Date')), None]
    dptest = DeterministicProcess(index=inputParam[0].index, constant=True, order=1, drop=True)
    lrtest = LinearRegression(fit_intercept=False).fit(dptest.in_sample(), inputParam[0]['Other'])
    
    desired = [pd.DataFrame(inputParam[0]['Other'] - lrtest.predict(dptest.in_sample())), lrtest.predict(dptest.in_sample()), dptest.in_sample()]

    output = fh.detrend(*inputParam)
    assert output[0].equals(desired[0])
    assert (output[1].predict(dptest.in_sample()) == desired[1]).all()
    assert output[2].in_sample().equals(desired[2])

def test_deseason_target():
    inputParam = [pd.DataFrame({'Other': [1, 2, 3]}, index=pd.Series([datetime(year=1970, month=1, day=1), datetime(year=1970, month=1, day=2), datetime(year=1970, month=1, day=3)], name='Date')), ('A', 2), "Other"]
    fouriertest = CalendarFourier(freq='A', order=2)
    dptest = DeterministicProcess(index=inputParam[0].index, constant=True, order=1, seasonal=True, additional_terms=[fouriertest], drop=True)
    lrtest = LinearRegression(fit_intercept=False).fit(dptest.in_sample(), inputParam[0]['Other'])
    
    desired = [pd.DataFrame(inputParam[0]['Other'] - lrtest.predict(dptest.in_sample())), lrtest.predict(dptest.in_sample()), dptest.in_sample()]

    output = fh.deseason(*inputParam)
    assert output[0].equals(desired[0])
    assert (output[1].predict(dptest.in_sample()) == desired[1]).all()
    assert output[2].in_sample().equals(desired[2])

def test_deseason_notarget():
    inputParam = [pd.DataFrame({'Other': [1, 2, 3]}, index=pd.Series([datetime(year=1970, month=1, day=1), datetime(year=1970, month=1, day=2), datetime(year=1970, month=1, day=3)], name='Date')), ('A', 2), None]
    fouriertest = CalendarFourier(freq='A', order=2)
    dptest = DeterministicProcess(index=inputParam[0].index, constant=True, order=1, seasonal=True, additional_terms=[fouriertest], drop=True)
    lrtest = LinearRegression(fit_intercept=False).fit(dptest.in_sample(), inputParam[0]['Other'])
    
    desired = [pd.DataFrame(inputParam[0]['Other'] - lrtest.predict(dptest.in_sample())), lrtest.predict(dptest.in_sample()), dptest.in_sample()]

    output = fh.deseason(*inputParam)
    assert output[0].equals(desired[0])
    assert (output[1].predict(dptest.in_sample()) == desired[1]).all()
    assert output[2].in_sample().equals(desired[2])

def test_multi_shift_target_lagwsteps():
    inputParam = [pd.DataFrame({'Other': [1, 2, 3]}, index=pd.Series([datetime(year=1970, month=1, day=1), datetime(year=1970, month=1, day=2), datetime(year=1970, month=1, day=3)], name='Date')), 2, [1, 2], True, "Other"]
    desired = pd.DataFrame(index=inputParam[0].index)
    for i in [1, 2]:
        desired["Other_lag_" + str(i)] = inputParam[0].shift(i)
    
    assert fh.multi_shift(*inputParam).equals(desired)

def test_multi_shift_target_lagwosteps():
    inputParam = [pd.DataFrame({'Other': [1, 2, 3]}, index=pd.Series([datetime(year=1970, month=1, day=1), datetime(year=1970, month=1, day=2), datetime(year=1970, month=1, day=3)], name='Date')), 2, [], True, "Other"]
    desired = pd.DataFrame(index=inputParam[0].index)
    for i in [1, 2]:
        desired["Other_lag_" + str(i)] = inputParam[0].shift(i)
    
    assert fh.multi_shift(*inputParam).equals(desired)

def test_multi_shift_target_lead():
    inputParam = [pd.DataFrame({'Other': [1, 2, 3]}, index=pd.Series([datetime(year=1970, month=1, day=1), datetime(year=1970, month=1, day=2), datetime(year=1970, month=1, day=3)], name='Date')), 2, [], False, "Other"]
    desired = pd.DataFrame(index=inputParam[0].index)
    for i in range(1, 3):
        desired["Other_lead_" + str(i)] = inputParam[0].shift(-i)
    
    assert fh.multi_shift(*inputParam).equals(desired)

def test_forecast():
    inputParam = [pd.DataFrame({'Other': [1, 2, 3, 4, 5, 6]}, index=pd.Series([datetime(year=1970, month=1, day=1), datetime(year=1970, month=1, day=2), datetime(year=1970, month=1, day=3),
                                                                                datetime(year=1970, month=1, day=4), datetime(year=1970, month=1, day=5), datetime(year=1970, month=1, day=6)], name='Date')),
                    2,
                    DecisionTreeRegressor(),
                    {'param_grid': {'max_leaf_nodes': [2, 3]}, 'cv': 2, 'scoring': 'neg_mean_absolute_error'},
                    'Other',
                    pd.DataFrame(),
                    pd.DataFrame({'Other_lead_1': [2, 3, 4, 5, 6, 4], 'Other_lead_2': [3, 4, 5, 6, 7, 5]}, index=pd.Series([datetime(year=1970, month=1, day=1), datetime(year=1970, month=1, day=2), datetime(year=1970, month=1, day=3),
                                                                                datetime(year=1970, month=1, day=4), datetime(year=1970, month=1, day=5), datetime(year=1970, month=1, day=6)], name='Date'))]
    
    desired = [pd.DataFrame({'Other_0': [1, 2],'Other_1': [2, 3], 'Other_2': [3, 4]}, index=pd.Series([datetime(year=1970, month=1, day=7), datetime(year=1970, month=1, day=8)], name='Date')), RegressorChain(DecisionTreeRegressor())]

    output = fh.forecast(*inputParam)
    assert output[0].shape == desired[0].shape
    assert (output[0].index == desired[0].index).all()
    assert type(output[1])== type(desired[1])
    assert output[1]._estimator_type == desired[1]._estimator_type

def test_forecast_nocv():
    inputParam = [pd.DataFrame({'Other': [1, 2, 3, 4, 5, 6]}, index=pd.Series([datetime(year=1970, month=1, day=1), datetime(year=1970, month=1, day=2), datetime(year=1970, month=1, day=3),
                                                                                datetime(year=1970, month=1, day=4), datetime(year=1970, month=1, day=5), datetime(year=1970, month=1, day=6)], name='Date')),
                    2,
                    DecisionTreeRegressor(),
                    None,
                    'Other',
                    pd.DataFrame(),
                    pd.DataFrame({'Other_lead_1': [2, 3, 4, 5, 6, 4], 'Other_lead_2': [3, 4, 5, 6, 7, 5]}, index=pd.Series([datetime(year=1970, month=1, day=1), datetime(year=1970, month=1, day=2), datetime(year=1970, month=1, day=3),
                                                                                datetime(year=1970, month=1, day=4), datetime(year=1970, month=1, day=5), datetime(year=1970, month=1, day=6)], name='Date'))]
    
    desired = [pd.DataFrame({'Other_0': [1, 2],'Other_1': [2, 3], 'Other_2': [3, 4]}, index=pd.Series([datetime(year=1970, month=1, day=7), datetime(year=1970, month=1, day=8)], name='Date')), RegressorChain(DecisionTreeRegressor())]

    output = fh.forecast(*inputParam)
    assert output[0].shape == desired[0].shape
    assert (output[0].index == desired[0].index).all()
    assert type(output[1])== type(desired[1])
    assert output[1]._estimator_type == desired[1]._estimator_type