from datetime import datetime
import logging
import pathlib

from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import KFold, GroupKFold
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
    df_id = pd.read_csv('input/store_id_relation.csv')
    hpg_store = pd.read_csv('input/hpg_store_info.csv')

    air_visit = pd.read_csv('input/air_visit_data.csv')
    air_store = pd.read_csv('input/air_store_info.csv')
    hpg_store = pd.merge(hpg_store, df_id, on='hpg_store_id', how='left').dropna(axis=0, how='any')
    air_store = pd.merge(air_store, hpg_store[['hpg_genre_name', 'air_store_id']], on='air_store_id', how='left')
    air_store['hpg_genre_name'].fillna('unk', inplace=True)
    df_air = pd.merge(air_visit, air_store, how='left', on='air_store_id')
    assert df_air.shape[0] == air_visit.shape[0], 'Fuckin shape'

    data = {
        'ar': pd.read_csv('input/air_reserve.csv'),
        'hr': pd.read_csv('input/hpg_reserve.csv'),
    }
    for key in ['ar', 'hr']:
        if key == 'hr':
            data[key] = pd.merge(
                data[key], df_id,
                how='left',
                on=['hpg_store_id']).dropna(axis=0, how='any')

        data[key]['visit_date'] = \
            data[key]['visit_datetime'].apply(lambda x: x.split(' ')[0])
        data[key]['diff_res_and_vis'] = (
            pd.to_datetime(data[key]['visit_datetime']) - \
                pd.to_datetime(data[key]['reserve_datetime'])).dt.days

        groupby = data[key].groupby(['visit_date', 'air_store_id'],
                                    as_index=False)
        reserve_mean = groupby.mean()
        reserve_sum = groupby.sum()

        for method in ['mean', 'sum']:
            columns = {
                'visit_date': 'visit_date',
                'air_store_id': 'air_store_id',
                'reserve_visitors': f'reserve_visitors_{method}_{key}',
                'diff_res_and_vis': f'diff_res_and_vis_{method}_{key}'
            }
            if method == 'mean':
                reserve_mean.rename(columns=columns, inplace=True)
            elif method == 'sum':
                reserve_sum.rename(columns=columns, inplace=True)

        reserve_feats = pd.merge(reserve_mean, reserve_sum, how='inner',
                                 on=['air_store_id', 'visit_date'])

        prev_shape = df_air.shape[0]
        df_air = pd.merge(df_air, reserve_feats,
                          how='left', on=['visit_date', 'air_store_id'])
        assert df_air.shape[0] == prev_shape, 'Fuckin shape'
        new_cols = [
            f'reserve_visitors_mean_{key}', f'reserve_visitors_sum_{key}',
            f'diff_res_and_vis_mean_{key}', f'diff_res_and_vis_sum_{key}'
        ]
        for c in new_cols:
            fill_val = df_air[c].mean()
            df_air[c].fillna(fill_val, inplace=True)

    # Feature Engineering
    reserve_cols = [
        'reserve_visitors_mean', 'reserve_visitors_sum',
        'diff_res_and_vis_mean', 'diff_res_and_vis_sum'
    ]
    for c in reserve_cols:
        df_air[c] = df_air[f'{c}_hr'] + df_air[f'{c}_ar']

    df_air['air_prefecture'] = \
        df_air['air_area_name'].apply(lambda x: x.split(' ')[0])

    df_air['latitude_minus_mean'] = \
        df_air['latitude'] - df_air['latitude'].mean()
    df_air['longitude_minus_mean'] = \
        df_air['longitude'] - df_air['longitude'].mean()
    df_air['longitude_plus_latitude'] = \
        df_air['longitude'] + df_air['latitude']

    df_air['year'] = df_air['visit_date'].apply(lambda x: x.split('-')[0])
    df_air['month'] = df_air['visit_date'].apply(lambda x: x.split('-')[1])
    df_air['day'] = df_air['visit_date'].apply(lambda x: x.split('-')[2])
    df_air['visit_date'] = pd.to_datetime(df_air['visit_date'])
    df_air['week'] = df_air['visit_date'].dt.week
    df_air['visit_date'] = df_air['visit_date'].astype(str)
    for c in ['year', 'month', 'week', 'day']:
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
    df_air['prev_day_is_holiday'] = df_air['holiday_flg'].shift().fillna(0)
    df_air['next_day_is_holiday'] = df_air['holiday_flg'].shift(-1).fillna(0)

    df_air['genre'] = df_air['air_genre_name'] + df_air['hpg_genre_name']
    df_air['genre_area'] = df_air['genre'] + df_air['air_prefecture']

    categoricals = [
        'air_store_id', 'air_genre_name', 'air_area_name', 'air_prefecture',
        'day_of_week', 'hpg_genre_name', 'genre', 'genre_area'
    ]
    for c in categoricals:
        encoder = df_air[c].value_counts()
        df_air[f'{c}_cnt_enc'] = df_air[c].map(encoder)

        le = LabelEncoder().fit(df_air[c].values)
        df_air[f'{c}'] = le.transform(df_air[c].values)

    mean_vistors = df_air['visitors'].mean()
    std_vistors = df_air['visitors'].std()
    for span in [7, 30]:
        by_id = df_air.groupby('air_store_id')['visitors']
        rolling = by_id.rolling(span, min_periods=1)

        rolling_mean = rolling.mean().shift(1).fillna(mean_vistors)
        df_air[f'ma{span}'] = rolling_mean.reset_index(0, drop=True)
        rolling_max = rolling.max().shift(1).fillna(mean_vistors)
        df_air[f'max_in_{span}'] = rolling_max.reset_index(0, drop=True)
        rolling_min = rolling.min().shift(1).fillna(mean_vistors)
        df_air[f'min_in_{span}'] = rolling_min.reset_index(0, drop=True)
        rolling_std = rolling.std().shift(1).fillna(std_vistors)
        df_air[f'std_in_{span}'] = rolling_std.reset_index(0, drop=True)

    # Training
    X = df_air.drop(['visit_date', 'visitors', 'calendar_date'], axis=1)
    y = np.log1p(df_air['visitors'].values)

    # cv = KFold(n_splits=n_splits)
    cv = GroupKFold(n_splits=n_splits)
    cv_scores = []
    feat_imp = pd.DataFrame(np.zeros_like(X.columns), columns=['importance'], index=X.columns)
    for cv_idx, (train_idx, valid_idx) in enumerate(cv.split(X, y, df_air['air_store_id'])):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]

        model = CatBoostRegressor(
            loss_function='RMSE',
            learning_rate=0.1,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_valid)

        rmse = np.sqrt(metrics.mean_squared_error(y_valid, y_pred))
        cv_scores.append(rmse)
        feat_imp['importance'] += model.feature_importances_ / n_splits

    for idx in range(n_splits):
        logger.info(f'[CV: {idx+1}/{n_splits}]  [RMSE: {cv_scores[idx]:.5f}]')

    logger.info(f'RMSLE: {np.mean(cv_scores)}±{np.std(cv_scores)}')
    feat_imp = feat_imp['importance'].sort_values(ascending=False)
    logger.info(f'Feature Importance')
    logger.info(f'{feat_imp}')
    plt.barh(feat_imp.index, feat_imp)
    plt.show()