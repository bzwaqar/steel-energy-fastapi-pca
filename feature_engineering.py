"""
Shared feature-engineering function used inside the saved model.joblib pipeline.

CRITICAL: This exact file (feature_engineering.py) must also be copied into the FastAPI
app folder, next to main.py, before loading model.joblib there. joblib/pickle stores a
*reference* to this function (module name + function name), not its code -- if the
module isn't importable at load time with the same name, unpickling fails with:
    AttributeError: Can't get attribute 'engineer_features' on <module '__main__' ...>
That's why this is NOT defined inline in the notebook (which would bind it to
'__main__', a module main.py doesn't share) -- it's written to its own .py file instead.
"""

import numpy as np
import pandas as pd


def engineer_features(df):
    """Raw dataframe (with 'date' + raw columns) -> engineered feature dataframe.
    Deterministic, no fitting required -- safe to apply identically to train/test/raw
    single-row inference input.
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])

    hour = df['date'].dt.hour
    month = df['date'].dt.month
    weekday = df['date'].dt.dayofweek

    df['day'] = df['date'].dt.day
    df['is_weekend'] = (weekday >= 5).astype(int)
    df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    df['month_sin'] = np.sin(2 * np.pi * month / 12)
    df['month_cos'] = np.cos(2 * np.pi * month / 12)

    df = df.drop(columns=['date'])
    return df
