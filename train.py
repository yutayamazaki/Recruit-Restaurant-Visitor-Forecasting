from datetime import datetime
import logging
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

sns.set_style('darkgrid')


def get_logger(filename: str) -> logging.Logger:
    format_ = '%(asctime)s:%(levelname)s:%(message)s'
    logging.basicConfig(filename=filename, level=logging.INFO,
                        format=format_)
    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler())
    return logger


def is_holiday(x: str) -> int:
    if x in ['Sunday', 'Saturday']:
        return 1
    else:
        return 0


def date_encode_cos(ser: pd.Series) -> np.ndarray:
    return np.cos(2 * np.pi * ser / ser.max())


def date_encode_sin(ser: pd.Series) -> np.ndarray:
    return np.sin(2 * np.pi * ser / ser.max())


n_splits = 5
LOG_DIR = './logs'

logger_path = pathlib.PurePath(LOG_DIR, f'{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
logger = get_logger(logger_path)

if __name__ == '__main__':
    # Data Preprocessing
    air_res = pd.read_csv('input/air_reserve.csv')
    df_id = pd.read_csv('input/store_id_relation.csv')
    air_store = pd.read_csv('input/air_store_info.csv')
    hpg_store = pd.read_csv('input/hpg_store_info.csv')
    air_visit = pd.read_csv('./input/air_visit_data.csv')

    df_air = pd.merge(air_visit, air_store, on='air_store_id')
    assert df_air.shape == (air_visit.shape[0], 7), 'Fuckin shape'

    # Feature Engineering
    df_air['air_prefecture'] = \
        df_air['air_area_name'].apply(lambda x: x.split(' ')[0])

    df_air['year'] = df_air['visit_date'].apply(lambda x: x.split('-')[0])
    df_air['month'] = df_air['visit_date'].apply(lambda x: x.split('-')[1])
    df_air['day'] = df_air['visit_date'].apply(lambda x: x.split('-')[2])
    for c in ['year', 'month', 'day']:
        df_air[c] = df_air[c].astype(int)
        df_air[f'{c}_sin'] = date_encode_sin(df_air[c])
        df_air[f'{c}_cos'] = date_encode_cos(df_air[c])
        df_air.drop([c], axis=1, inplace=True)

    df_day = pd.read_csv('./input/date_info.csv')
    prev_len = len(df_air)
    df_air = pd.merge(df_air, df_day, left_on='visit_date', right_on='calendar_date')
    assert len(df_air) == prev_len, 'Fuckin shape'

    saturday_or_sunday = df_air['day_of_week'].apply(is_holiday)
    df_air['holiday_flg'] = saturday_or_sunday | df_air['holiday_flg']

    categoricals = ['air_store_id', 'air_genre_name', 'air_area_name',
                    'air_prefecture', 'day_of_week']
    for c in categoricals:
        encoder = df_air[c].value_counts()
        df_air[f'{c}_cnt_enc'] = df_air[c].map(encoder)

        le = LabelEncoder().fit(df_air[c].values)
        df_air[f'{c}'] = le.transform(df_air[c].values)

    # Training
    X = df_air.drop(['visit_date', 'visitors', 'calendar_date'], axis=1)
    y = np.log1p(df_air['visitors'].values)

    cv = KFold(n_splits=n_splits)
    cv_scores = []
    feat_imp = np.zeros_like(X.columns)
    feat_imp = pd.DataFrame(np.zeros_like(X.columns), columns=['importance'], index=X.columns)
    for cv_idx, (train_idx, valid_idx) in enumerate(cv.split(X, y)):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]
        
        model = GradientBoostingRegressor()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_valid)

        rmse = np.sqrt(metrics.mean_squared_error(y_valid, y_pred))
        cv_scores.append(rmse)
        feat_imp['importance'] += model.feature_importances_ / n_splits

        logger.info(f'[CV: {cv_idx+1}/{n_splits}]  [RMSE: {rmse:.5f}]')

    logger.info(f'Mean RMSE: {np.mean(cv_scores)}')
    feat_imp = feat_imp['importance'].sort_values(ascending=False)
    logger.info(f'Feature Importance')
    logger.info(f'{feat_imp}')
    plt.barh(feat_imp.index, feat_imp)
    plt.show()