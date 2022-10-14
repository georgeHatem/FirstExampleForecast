#
# Set of helper functions for producing forecasts
#
#
# Author: George Hatem      Last Updated: 2022-10-13
#

import re
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.multioutput import RegressorChain
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess

def _fixDate(din: pd.DataFrame) -> pd.DataFrame:
    """Private method used to check the input for a datetime feature in index and add it if possible."""
    if din.empty:
        raise ValueError("Input pandas.DataFrame is empty.")

    d = din.copy() #does not modify input dataframe

    if d.index.inferred_type != "datetime64":
        dateLabel = list(filter(re.compile(".*date.*|.*time.*", re.IGNORECASE).match, d.columns.tolist()))
        if len(dateLabel) > 0:
            d.set_index(dateLabel[0], inplace=True)

    if d.index.inferred_type in ["integer", "string"]:
        try:
            d.set_index(d.index.to_series().apply(lambda x: pd.to_datetime(x)), inplace=True)
        except:
            raise TypeError("Index is not DatetimeIndex and could not be parsed into DatetimeIndex.")

    return d
    


def detrend(din: pd.DataFrame, target: str = None) -> [pd.DataFrame, LinearRegression, DeterministicProcess]:
    """
    detrend: Returns a list of a pandas.DataFrame of the detrended input 
    feature with the same index and the model used to make this detrended output
    
    Input: 
        - din: A pandas.DataFrame of the series to detrend. Must be a numerical value. 
            A single feature should be present unless the target is given.
        - target: (Optional) Name of the feature to detrend
    Output:
        - (pd.DataFrame, LinearRegression): Single Feature pandas.DataFrame of detrended input, 
            model describing trend without intercept, statsmodels DeterministicProcess of target feature
    """
    d = _fixDate(din)

    if d.columns.size != 1 and target == None:
        raise ValueError("Could not determine target feature due to having too many options and no target specified.")

    dp = DeterministicProcess(
            index=d.index,
            constant=True,
            order=1,
            drop=True
            )

    if target == None:
        trendReg = LinearRegression(fit_intercept=False).fit(dp.in_sample(), d[d.columns[0]])
        dret = d[d.columns[0]] - trendReg.predict(dp.in_sample())
    else:
        trendReg = LinearRegression(fit_intercept=False).fit(dp.in_sample(), d[target])
        dret = d[target] - trendReg.predict(dp.in_sample())

    return [pd.DataFrame(dret), trendReg, dp]

def deseason(din: pd.DataFrame, season: (str, int), target: str = None) -> [pd.DataFrame, LinearRegression, DeterministicProcess]:
    """
    deseason: Returns a list of a pandas.DataFrame of the deseasoned input 
    feature with the same index and the model used to make this deseasoned output
    as well as a DeterministicProcess of the input

    Input: 
        - din: A pandas.DataFrame of the series to deseason. Must be a numerical value. 
            A single feature should be present unless the target is given.
        - season: Tuple of frequency timedelta values and how many of it past which variance drops
        - target: (Optional) Name of the feature to deseason
    Output:
        - (DataFrame, LinearRegression, DeterministicProcess): Single feature pandas.DataFrame of detrended input, 
            model describing trend, and statsmodels DeterministicProcess of input
    """
    d = _fixDate(din)

    if d.columns.size != 1 and target == None:
        raise ValueError("Could not determine target feature due to having too many options and no target specified.")

    fourier = CalendarFourier(freq=season[0], order=season[1])

    dp = DeterministicProcess(
            index=d.index,
            constant=True,
            order=1,
            seasonal=True,
            additional_terms=[fourier],
            drop=True
            )

    if target == None:
        seasonReg = LinearRegression().fit(dp.in_sample(), d[d.columns[0]])
        dret = d[0] - seasonReg.predict(dp.in_sample())
    else:
        seasonReg = LinearRegression().fit(dp.in_sample(), d[target])
        dret = d[target] - seasonReg.predict(dp.in_sample())

    return [pd.DataFrame(dret), seasonReg, dp]

def multi_shift(din: pd.DataFrame, steps: int, lag_steps: list = [], sign: bool = True, target: str = None) -> pd.DataFrame:
    """
    lag: Returns lags or leads to the input target feature up to input 'steps' times.

    Input:
        - din: Pandas.DataFrame with feature to lag. 
            Must have a single feature unless the target is given.
        - steps: Number of lags to produce
        - sign: determines whether to output is a lag or a leads. True shifts forward, False shifts back.
        - target: (Optional) Name of feature to lag or lead
    """
    d = _fixDate(din)

    if d.columns.size != 1 and target == None:
        raise ValueError("Could not determine target feature due to having too many options and no target specified.")

    dret = pd.DataFrame(index=d.index)
    if target == None:
        feature_name = d.columns[0]
    else:
        feature_name = target

    for i in range(1, steps + 1):
        if sign:
            if i in lag_steps or lag_steps == []:
                dret[str(feature_name) + "_lag_" + str(i)] = d[feature_name].shift(i)
        else:
            dret[str(feature_name) + "_lead_" + str(i)] = d[feature_name].shift(-i)

    return dret

def forecast(din: pd.DataFrame, length: int, model: object, cvparam: list = None, target: str = None, lags: pd.DataFrame = pd.DataFrame(), leads: pd.DataFrame = pd.DataFrame()) -> [pd.DataFrame, object]:
    """
    forcast: Returns a pandas.DataFrame of the forecast over an input time series for the given target feature.
        Optionally: input lags and leads pandas.DataFrame
    
    Input:
        - din: pandas.DataFrame including the target feature. 
            Must have a single feature unless the target is given.
        - length: number of time periods to forecast forward
        - model: Instance of modeling algorithm must be base model will be fed to sklearn.multioutput.RegressorChain()
        - cvparam: (Optional) cross-validation prameters besides the model for sklearn.model_selection.GridSearchCV.
            eg: "GridSearchCV(model, *cvparam)" where cvparam = [param_grid, cv, scoring].
        - target: (Optional) Name of feature to forcast on
        - lags: (Optional) Minimum 1 lag will be used. This method imputes mean value in NA
        - lag_steps: (Optional) list of lag shifts provided eg: lag_1, lag_2, lag_5 corresponds to [1, 2, 5]
        - leads: (Optional) Without current value eg: shift(0). This method imputes mean value in NA
    Output:
        - DataFrame: forecast predictions
        - object: best performing model
    """
    d = _fixDate(din)

    if d.columns.size != 1 and target == None:
        raise ValueError("Could not determine target feature due to having too many options and no target specified.")

    if target == None:
        feature_name = d.columns[0]
    else:
        feature_name = target

    if lags.empty:
        lags = pd.DataFrame(index=d.index)
        lags[feature_name + "_lag_1"] = d[feature_name].shift(1)

    lag_length = lags.shape[1]

    if leads.empty:
        leads = pd.DataFrame(index=d.index)
        lead_length = 10
        for i in range(1, lead_length + 1):
            leads[feature_name + "_lead_" + str(i)] = d[feature_name].shift(-i)
    else:
        lead_length = leads.shape[1]

    imputer = SimpleImputer(strategy="median", copy=False)
    imputer.fit_transform(lags)
    imputer.fit_transform(leads)

    future = d[[feature_name]].join(leads)

    if cvparam != None:
        model = GridSearchCV(model, *cvparam).fit(past, leads).best_estimator_

    date = d.index.to_series().apply(lambda x: x.toordinal())
    foreDate = (np.arange(length, dtype=int) + 1) * int(date.diff().mean()) + date.iloc[-1]
    foreData = pd.DataFrame((datetime.fromordinal(i) for i in foreDate), columns=["Date"])

    try:
        model = RegressorChain(model)
        model.fit(lags, future)
    except:
        raise TypeError("Could not fit given model through an sklearn.multioutput.RegressorChain()")

    #Reserve Memory
    foreRes = pd.DataFrame(np.zeros((length, lead_length + 1)), columns=list(feature_name + "_" + str(i) for i in range(0, lead_length + 1, 1)), index=range(length))
    foreRes.iloc[0] = model.predict( lags.iloc[[-1]] )
    for i in range(1, length):
        foreRes.iloc[i] = model.predict( lags.iloc[:, :lag_length].iloc[[i - 1]] )

    return [foreRes.set_index(foreData["Date"]), model]
