import lore
import lore.io
from datetime import datetime, timedelta
import logging
import os, math
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
import pickle
import pytz
import sqlalchemy as scm
from sqlalchemy.dialects import postgresql

logger = logging.getLogger(__name__)

HOURS = ['hour_6', 'hour_7', 'hour_8', 'hour_9', 'hour_10',
          'hour_11', 'hour_12', 'hour_13', 'hour_14', 'hour_15',
          'hour_16', 'hour_17', 'hour_18', 'hour_19', 'hour_20',
          'hour_21', 'hour_22', 'hour_23']
WEEKDAYS = ['weekday_0', 'weekday_1', 'weekday_2', 'weekday_3',
            'weekday_4', 'weekday_5', 'weekday_6']

class RealTimeGapForecast(object):

    def __init__(self,
                 zone_id,
                 mode='scoring',
                 running_time=datetime.now(),
                 days=56):

        # Initialize parameters
        self._zone_id = zone_id
        self._mode = mode
        self._days = days
        self._end_dt = datetime(running_time.year, running_time.month, running_time.day)
        self._end_dt_str = str(self._end_dt.date())

        # Read in DataFrames
        self._time_zone = self._get_time_zone()
        self._orders = self._get_orders()
        self._orders_created = pd.DataFrame()
        self._orders_ends = pd.DataFrame()
        self._orders_delivered = pd.DataFrame()
        self._capacity = self._get_capacity()
        self._shoppers = self._get_shoppers()
        self._lost_demand = self._get_lost_demand()

        self._data = None
        self._features = None
        self._linear_model = None
        self._xgb = None
        self._filename_lr = './medley/realtime_gap_forecast/production/v1/model/v1_linear_model_' \
                            + str(self._zone_id) + '.pkl'
        self._filename_xgb = './medley/realtime_gap_forecast/production/v1/model/v1_xgb_model_' \
                             + str(self._zone_id) + '.pkl'

        logger.info('Real Time Gap Forecast V1 for Zone %d initialized.' % (self._zone_id))

    def _get_time_zone(self):
        time_zone = lore.io.logistics.dataframe(
            """select time_zone_name from logistics_zones where zone_id = '{}'""".format(self._zone_id))

        return time_zone['time_zone_name'][0]

    def _get_orders(self):

        if self._mode == 'training':
            df = lore.io.snowflake.dataframe(filename='realtime_gap_forecast/v1/rtgf_orders_training',
                                             zone_id=self._zone_id, date=self._end_dt_str, days=self._days)
        else:
            df = lore.io.orders.dataframe(filename='realtime_gap_forecast/v1/rtgf_orders_scoring',
                                          zone_id=self._zone_id, utc_now=datetime.utcnow())

        return df

    def _get_capacity(self):

        if self._mode == 'training':
            df = lore.io.snowflake.dataframe(filename='realtime_gap_forecast/v1/rtgf_capacity_training',
                                             zone_id=self._zone_id, date=self._end_dt_str, days=self._days)
        else:
            dates = [self._end_dt.date(), (self._end_dt + timedelta(days=1)).date()]
            df = lore.io.logistics.dataframe(filename='realtime_gap_forecast/v1/rtgf_capacity_scoring',
                                          zone_id=self._zone_id, dates=tuple(dates), utc_now=datetime.utcnow())

        return df

    def _get_shoppers(self):

        if self._mode == 'training':
            df = lore.io.snowflake.dataframe(filename='realtime_gap_forecast/v1/rtgf_shoppers_training',
                                             zone_id=self._zone_id, date=self._end_dt_str, days=self._days)
        else:
            df = lore.io.logistics.dataframe(filename='realtime_gap_forecast/v1/rtgf_shoppers_scoring',
                                          zone_id=self._zone_id, utc_now=datetime.utcnow())

        return df

    def _get_lost_demand(self):

        if self._mode == 'training':
            df = lore.io.snowflake.dataframe(filename='realtime_gap_forecast/v1/rtgf_lost_demand_training',
                                             zone_id=self._zone_id, date=self._end_dt_str, days=self._days)
        else:
            df = pd.DataFrame({'zone_id': pd.Series([], dtype='int'),
                               'date': pd.Series([], dtype='datetime64[ns]'),
                               'hour': pd.Series([], dtype='int'),
                               'created_at': pd.Series([], dtype='datetime64[ns]'),
                               'daily_ld': pd.Series([], dtype='int'),
                               'deliveries': pd.Series([], dtype='int'),
                               'demand': pd.Series([], dtype='float'),
                               'hourly_ld': pd.Series([], dtype='float')})

        df['ld_percent'] = df['hourly_ld'] / df['demand']

        return df

    def _feature_engineering(self):

        # Convert to string then back to datetime to avoid wired timezone bug.
        # Convert to string
        self._shoppers['created_at'] = self._shoppers['created_at'].map(lambda x: str(x)[:19])
        self._capacity['created_at'] = self._capacity['created_at'].map(lambda x: str(x)[:19])
        self._orders['time_at'] = self._orders['time_at'].map(lambda x: str(x)[:19])
        self._lost_demand['created_at'] = self._lost_demand['created_at'].map(lambda x: str(x)[:19])

        # Convert to datetime
        self._capacity['created_at'] = pd.to_datetime(self._capacity['created_at'], infer_datetime_format=True)
        self._shoppers['created_at'] = pd.to_datetime(self._shoppers['created_at'], infer_datetime_format=True)
        self._lost_demand['created_at'] = pd.to_datetime(self._lost_demand['created_at'], infer_datetime_format=True)
        self._shoppers.dropna(subset=['eligible_shoppers'], inplace=True)
        self._shoppers.fillna(0, inplace=True)
        self._orders['time_at'] = pd.to_datetime(self._orders['time_at'], infer_datetime_format=True)
        self._orders_created = self._orders[self._orders['label'] == 'created'].copy()
        self._orders_ends = self._orders[self._orders['label'] == 'window_ends'].copy()
        self._orders_delivered = self._orders[self._orders['label'] == 'delivered'].copy()


        # Create created hour
        self._capacity['created_hour'] = self._capacity['created_at'].map(lambda x:
                                                                    datetime(x.year, x.month, x.day, int(x.hour)))
        self._shoppers['created_hour'] = self._shoppers['created_at'].map(lambda x:
                                                                    datetime(x.year, x.month, x.day, int(x.hour)))
        self._lost_demand['created_hour'] = self._lost_demand['created_at'].map(lambda x:
                                                                    datetime(x.year, x.month, x.day, int(x.hour)))

        self._orders_created['created_hour'] = self._orders_created['time_at'].map(lambda x:
                                                                    datetime(x.year, x.month, x.day, int(x.hour)))
        self._orders_delivered['delivered_hour'] = self._orders_delivered['time_at'].map(lambda x:
                                                                    datetime(x.year, x.month, x.day, int(x.hour)))
        # If window_ends_at is 8:00pm, then it should be delivered at 7:00pm - 8:00pm.
        self._orders_ends['window_ends_hour'] = self._orders_ends['time_at'].map(lambda x:
                        datetime(x.year, x.month, x.day, int(x.hour) - 1) if x.minute == 0 and x.second == 0
                        else datetime(x.year, x.month, x.day, int(x.hour)))

        return

    def _get_train_score_data(self):

        self._feature_engineering()

        # Group into hours
        df_cpcty_created = self._capacity[['created_hour', 'capacity_on']].groupby('created_hour') \
            .mean().reset_index().rename(columns={'capacity_on': 'availability'})
        df_shppr_created = self._shoppers[['created_hour', 'eligible_shoppers',
            'working_shoppers', 'shoppers_having_batches']] \
            .groupby('created_hour').mean().reset_index()
        df_created = self._orders_created[['created_hour', 'order_id']].groupby('created_hour') \
            .count().reset_index().rename(columns={'order_id': 'created_cnt'})
        df_window_ends = self._orders_ends[['window_ends_hour', 'order_id']].groupby('window_ends_hour') \
            .count().reset_index().rename(columns={'order_id': 'window_ends_cnt'})
        df_delivered = self._orders_delivered[['delivered_hour', 'order_id']].groupby('delivered_hour') \
            .count().reset_index().rename(columns={'order_id': 'delivered_cnt'})

        # Scheduled orders as a feature
        df_scheduled = self._orders_ends[self._orders_ends['delivery_type'].isin(['limited_availability', 'scheduled'])] \
            [['window_ends_hour', 'order_id']].groupby('window_ends_hour') \
            .count().reset_index().rename(columns={'order_id': 'scheduled_cnt'})

        # Working and having batch percent
        df_shppr_created['working_pct'] = (df_shppr_created['working_shoppers']
                                           / df_shppr_created['eligible_shoppers'])
        df_shppr_created['coverage_pct'] = (df_shppr_created['shoppers_having_batches']
                                            / df_shppr_created['eligible_shoppers'])

        # Merge
        data = df_window_ends.merge(df_scheduled,
                                    left_on=['window_ends_hour'],
                                    right_on=['window_ends_hour'],
                                    how='left')

        data = data.merge(df_created,
                          left_on=['window_ends_hour'],
                          right_on=['created_hour'],
                          how='left')

        data = data.merge(df_delivered,
                          left_on=['window_ends_hour'],
                          right_on=['delivered_hour'],
                          how='left')

        data = data.merge(df_cpcty_created,
                          left_on=['window_ends_hour'],
                          right_on=['created_hour'],
                          how='left')

        data = data.merge(df_shppr_created[['created_hour', 'eligible_shoppers', 'working_pct', 'coverage_pct']],
                          left_on=['window_ends_hour'],
                          right_on=['created_hour'],
                          how='left')

        data = data.merge(self._lost_demand[['created_hour', 'ld_percent']],
                          left_on=['window_ends_hour'],
                          right_on=['created_hour'],
                          how='left')

        data = data[['window_ends_hour', 'window_ends_cnt', 'created_cnt', 'delivered_cnt', 'scheduled_cnt',
                     'availability', 'ld_percent', 'eligible_shoppers', 'working_pct', 'coverage_pct']].copy()
        data.rename(columns={'window_ends_hour': 'time'}, inplace=True)
        data.fillna(value={'created_cnt': 0, 'delivered_cnt': 0, 'scheduled_cnt': 0,
                           'availability': 0,
                           'eligible_shoppers': 0, 'working_pct': 0, 'coverage_pct': 0},
                    inplace=True)

        return data

    def _prepare_train_score_data(self):

        data = self._get_train_score_data()

        if self._mode == 'training':
            data['gap'] = data['window_ends_cnt'] * (1 + data['ld_percent']) - data['delivered_cnt']

        data['weekday'] = data['time'].map(lambda x: x.weekday())
        data['hour'] = data['time'].map(lambda x: x.hour)

        # Prepare lag data
        # Run at 10:30AM(10AM), Lag 1hr is 9AM(9-10AM), Lag 2 hr is 8AM(8-9AM).
        data['time_add1hr'] = data['time'].map(lambda x: x + timedelta(hours=1))
        data['time_add2hr'] = data['time'].map(lambda x: x + timedelta(hours=2))

        # Adding lag feature and create label
        # Lag 1 hour
        df_lag1 = data[['time_add1hr', 'window_ends_cnt', 'created_cnt', 'delivered_cnt',
                        'availability',
                        'eligible_shoppers', 'working_pct', 'coverage_pct']].copy()
        col_rename_t1 = {_s: _s + '_lag1' for _s in ['window_ends_cnt', 'created_cnt', 'delivered_cnt',
                                                     'availability',
                                                     'eligible_shoppers', 'working_pct', 'coverage_pct']}
        df_lag1.rename(columns=col_rename_t1, inplace=True)

        # Lag 2 hour
        df_lag2 = data[['time_add2hr', 'window_ends_cnt', 'created_cnt', 'delivered_cnt',
                        'availability',
                        'eligible_shoppers', 'working_pct', 'coverage_pct']].copy()
        col_rename_t2 = {_s: _s + '_lag2' for _s in ['window_ends_cnt', 'created_cnt', 'delivered_cnt',
                                                     'availability',
                                                     'eligible_shoppers', 'working_pct', 'coverage_pct']}
        df_lag2.rename(columns=col_rename_t2, inplace=True)

        # Training Label and Naive Forecast
        # Run at 10:30AM(10AM), predict 2hr leading time is 1PM.
        # Naive forecast is yesterday's 1-2PM.
        data['time_sub3hr'] = data['time'].map(lambda x: x - timedelta(hours=3))
        data['time_add1dy'] = data['time'].map(lambda x: x + timedelta(hours=21))

        # Lag 1 Day
        if self._mode == 'training':
            df_lag3 = data[['time_add1dy', 'gap']].copy()
            col_rename_t3 = {'gap': 'gap_forecast_naive'}
            df_lag3.rename(columns=col_rename_t3, inplace=True)

            # Training Label
            # Scheduled feature is the number of scheduled orders for the prediction hour
            df_lead3 = data[['time_sub3hr', 'gap', 'scheduled_cnt']].copy()
            df_lead3.rename(columns={'gap': 'gap_forecast_label',
                                     'scheduled_cnt': 'scheduled_feature'}, inplace=True)
        else:
            # Scheduled feature is the number of scheduled orders for the prediction hour
            df_lead3 = data[['time_sub3hr', 'scheduled_cnt']].copy()
            df_lead3.rename(columns={'scheduled_cnt': 'scheduled_feature'}, inplace=True)

        # Merge DataFrame
        if self._mode == 'training':
            data = data[['time', 'hour', 'weekday',
                     'window_ends_cnt', 'created_cnt', 'delivered_cnt', 'scheduled_cnt',
                     'availability',
                     'eligible_shoppers', 'working_pct', 'coverage_pct',
                     'gap']].merge(df_lag1,
                                   left_on='time',
                                   right_on='time_add1hr',
                                   how='left')
        else:
            data = data[['time', 'hour', 'weekday',
                         'window_ends_cnt', 'created_cnt', 'delivered_cnt', 'scheduled_cnt',
                         'availability',
                         'eligible_shoppers', 'working_pct', 'coverage_pct']].merge(df_lag1,
                                       left_on='time',
                                       right_on='time_add1hr',
                                       how='left')

        data = data.merge(df_lag2,
                          left_on='time',
                          right_on='time_add2hr',
                          how='left')

        data = data.merge(df_lead3,
                          left_on='time',
                          right_on='time_sub3hr',
                          how='left')

        if self._mode == 'training':
            data = data.merge(df_lag3,
                              left_on='time',
                              right_on='time_add1dy',
                              how='left')

        # Dummy variables for hour and weekday
        data = data.join(pd.get_dummies(data['hour'], prefix='hour'))
        data = data.join(pd.get_dummies(data['weekday'], prefix='weekday'))
        for _col in [_ for _ in HOURS + WEEKDAYS if _ not in data.columns.tolist()]:
            data[_col] = 0

        return data

    def _training(self, data):

        # Hypoparameter training
        # Reset index, otherwise there is problem with my own CViterator
        # Get 7 weeks of data
        data = data[data['time'] >= self._end_dt - timedelta(days=self._days - 7)]
        data.reset_index(drop=True, inplace=True)
        data.fillna(0, inplace=True)

        myCViterator = []
        for time_filter in [self._end_dt - timedelta(days=21), self._end_dt - timedelta(days=14)]:
            trainIndices = data[data.time < time_filter].index.values.astype(int)
            testIndices = data[(data.time >= time_filter) &
                               (data.time < time_filter + timedelta(days=7))].index.values.astype(int)
            myCViterator.append((trainIndices, testIndices))

        # Model Training
        from sklearn.linear_model import RidgeCV

        train_data = data[data['time'] < self._end_dt - timedelta(days=7)][self._features]
        train_label = data[data['time'] < self._end_dt - timedelta(days=7)]['gap_forecast_label']

        # Define Model Parameters
        grid = RidgeCV(alphas=[0.001, 0.01, 0.1, 1, 10, 100], cv=myCViterator)

        grid.fit(train_data, train_label)

        # Save to local
        if os.environ.get("LORE_ENV") == 'development':
            pickle.dump(grid, open(self._filename_lr, 'wb'))

        # Save to S3
        # Save to S3
        filename = 'v1_linear_model_' + str(self._zone_id) + '.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(grid, f)
        pathname = os.path.join('realtime_gap_forecast', filename)
        lore.io.upload(filename, pathname)
        os.remove(filename)

        self._linear_model = grid

        logger.info({_k: round(_v, 4) for _k, _v in zip(self._features, grid.coef_)})
        logger.info(grid.alpha_)

        # XGBoost Regression
        xgb = XGBRegressor(n_jobs=-1, silent=1, subsample=0.9, eval_metric='rmse')

        params_random = {
            "max_depth": [2, 3, 4, 5, 6, 7, 8, 9],
            "n_estimators": [20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140],
            "min_child_weight": [1, 2, 3, 4, 5, 6],
            "gamma": [i / 10.0 for i in range(3, 6)]
        }

        model = RandomizedSearchCV(estimator=xgb, param_distributions=params_random, cv=myCViterator)
        model.fit(train_data, train_label)
        logger.info(model.best_params_)

        # Save to local
        if os.environ.get("LORE_ENV") == 'development':
            pickle.dump(model, open(self._filename_xgb, 'wb'))

        # Save to S3
        filename = 'v1_xgb_model_' + str(self._zone_id) + '.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
        pathname = os.path.join('realtime_gap_forecast', filename)
        lore.io.upload(filename, pathname)
        os.remove(filename)

        self._xgb = model

        return

    def _scoring(self, data):

        # Load linear model from s3
        filename = 'realtime_gap_forecast/v1_linear_model_' + str(self._zone_id) + '.pkl'
        file_path = lore.io.download(filename)
        linear_model = pickle.load(open(file_path, 'rb'))
        os.remove(file_path)

        # Load XGB model from s3
        filename = 'realtime_gap_forecast/v1_xgb_model_' + str(self._zone_id) + '.pkl'
        file_path = lore.io.download(filename)
        xgb_model = pickle.load(open(file_path, 'rb'))
        os.remove(file_path)

        df_forecast = data.copy()
        data = data.fillna(0)
        df_forecast['forecast_linear'] = linear_model.predict(data[self._features])
        df_forecast['forecast_xgb'] = xgb_model.best_estimator_.predict(data[self._features])

        df_forecast['forecast_linear'] = df_forecast['forecast_linear'].map(lambda x: round(x, 1))
        df_forecast['forecast_xgb'] = df_forecast['forecast_xgb'].map(lambda x: round(x, 1))

        # Real Time Check
        df_forecast['gap'] = df_forecast['window_ends_cnt'] - df_forecast['delivered_cnt']
        df_forecast['time_sub3hr'] = df_forecast['time'].map(lambda x: x - timedelta(hours=3))
        df_lead3 = df_forecast[['time_sub3hr', 'gap']].copy()
        df_lead3.rename(columns={'gap': 'forecast_label'}, inplace=True)

        df_forecast = df_forecast.merge(df_lead3,
                                        left_on='time',
                                        right_on='time_sub3hr',
                                        how='left')

        running_local_hour = datetime.now(pytz.timezone(self._time_zone)).hour
        self._write_forecast(df_forecast[df_forecast['time'].dt.hour == running_local_hour] \
            [['time', 'scheduled_feature', 'forecast_linear', 'forecast_xgb']])

        return

    def train_score(self):

        data = self._prepare_train_score_data()

        cols = ['time', 'window_ends_cnt', 'created_cnt', 'delivered_cnt', 'scheduled_cnt',
                'availability',
                'eligible_shoppers', 'working_pct', 'coverage_pct']
        self._features = HOURS + ['weekday_0', 'weekday_1', 'weekday_2', 'weekday_3',
                                'weekday_4', 'weekday_5', 'weekday_6',
                                'scheduled_feature',
                                'window_ends_cnt_lag1', 'created_cnt_lag1', 'delivered_cnt_lag1',
                                'availability_lag1',
                                'eligible_shoppers_lag1', 'working_pct_lag1', 'coverage_pct_lag1',
                                'window_ends_cnt_lag2', 'created_cnt_lag2', 'delivered_cnt_lag2',
                                'availability_lag2',
                                'eligible_shoppers_lag2', 'working_pct_lag2', 'coverage_pct_lag2']
        label = ['gap', 'gap_forecast_label', 'gap_forecast_naive']

        if self._mode == 'training':
            data = data[cols + self._features + label]
            self._training(data)
        elif len(data) > 0:
            data = data[cols + self._features]
            self._scoring(data)

        self._data = data

        return

    def _write_forecast(self, df):

        cols_to_write = ['zone_id', 'date', 'hour', 'forecast_type', 'forecast_name',
                         'data', 'created_at']

        if len(df) > 0:
            df['zone_id'] = self._zone_id
            df['time_forecasted'] = df['time'].map(lambda x: x + timedelta(hours=3))
            df['date'] = df['time_forecasted'].dt.date
            df['hour'] = df['time_forecasted'].dt.hour
            df['forecast_type'] = 'gap'
            df['forecast_name'] = 'v1_realtime_gap_forecast'
            df['gap'] = df['forecast_xgb']
            cols_to_data = ['gap', 'forecast_xgb', 'forecast_linear', 'scheduled_feature']
            df['data'] = df[cols_to_data].to_dict(orient='records')
            df['created_at'] = datetime.utcnow()

            self._write_to_realtime_gap_forecasts(df[cols_to_write].to_dict(orient='records'))
            logger.info(
                'Real Time Gap Forecast V1 for Zone %d uploaded successful.' % (self._zone_id))

        return

    def check_model_performance(self):

        if self._mode == 'training':

            data = self._data
            data.fillna(0, inplace=True)

            # Check Model Performance
            df_forecast = data[data['time'] >= self._end_dt - timedelta(days=7)][
                ['time', 'gap_forecast_label', 'gap_forecast_naive']]
            df_forecast['gap_forecast_linear'] = self._linear_model.predict(data[data['time'] >= self._end_dt - timedelta(days=7)][self._features])
            df_forecast['gap_forecast_xgb'] = self._xgb.best_estimator_.predict(data[data['time'] >= self._end_dt - timedelta(days=7)][self._features])
            df_forecast['naive_residual'] = df_forecast['gap_forecast_naive'] - df_forecast['gap_forecast_label']
            df_forecast['linear_residual'] = df_forecast['gap_forecast_linear'] - df_forecast['gap_forecast_label']
            df_forecast['xgb_residual'] = df_forecast['gap_forecast_xgb'] - df_forecast['gap_forecast_label']

            # Correlation Score
            print('Correlation Naive:')
            print(df_forecast[['gap_forecast_label', 'gap_forecast_naive']].corr().iloc[0]['gap_forecast_naive'])

            print('\nCorrelation linear:')
            print(
                df_forecast[['gap_forecast_label', 'gap_forecast_linear']].corr().iloc[0]['gap_forecast_linear'])

            print('\nCorrelation xgb:')
            print(df_forecast[['gap_forecast_label', 'gap_forecast_xgb']].corr().iloc[0]['gap_forecast_xgb'])

            # Measure accuray whether needs scaling
            df_forecast['gap_forecast_label_tf'] = df_forecast['gap_forecast_label']\
                .map(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
            df_forecast['gap_forecast_naive_tf'] = df_forecast['gap_forecast_naive']\
                .map(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
            df_forecast['gap_forecast_linear_tf'] = df_forecast['gap_forecast_linear']\
                .map(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
            df_forecast['gap_forecast_xgb_tf'] = df_forecast['gap_forecast_xgb']\
                .map(lambda x: 1 if x > 0 else -1 if x < 0 else 0)

            df_forecast.replace({'gap_forecast_label_tf': {0: np.nan},
                                 'gap_forecast_naive_tf': {0: np.nan},
                                 'gap_forecast_linear_tf': {0: np.nan},
                                 'gap_forecast_xgb_tf': {0: np.nan}},
                                inplace=True)

            # Accuracy Score
            from sklearn.metrics import accuracy_score, precision_score
            df_tf = df_forecast[['gap_forecast_label_tf', 'gap_forecast_naive_tf',
                                 'gap_forecast_linear_tf', 'gap_forecast_xgb_tf']].copy()
            df_tf.dropna(inplace=True, axis=0)
            print("\n\nNaive model accuracy:")
            print(round(accuracy_score(df_tf['gap_forecast_label_tf'], df_tf['gap_forecast_naive_tf']), 3))

            print("\nLinear model accuracy:")
            print(round(accuracy_score(df_tf['gap_forecast_label_tf'], df_tf['gap_forecast_linear_tf']), 3))

            print("\nXGB model accuracy:")
            print(round(accuracy_score(df_tf['gap_forecast_label_tf'], df_tf['gap_forecast_xgb_tf']), 3))

            print("\n\nNaive model precision:")
            print(round(precision_score(df_tf['gap_forecast_label_tf'], df_tf['gap_forecast_naive_tf']), 3))

            print("\nLinear model precision:")
            print(round(precision_score(df_tf['gap_forecast_label_tf'], df_tf['gap_forecast_linear_tf']), 3))

            print("\nXGB model precision:")
            print(round(precision_score(df_tf['gap_forecast_label_tf'], df_tf['gap_forecast_xgb_tf']), 3))

            # APE Score
            df_forecast['naive_ape'] = ((df_forecast['gap_forecast_naive'] - df_forecast['gap_forecast_label']) /
                                        df_forecast['gap_forecast_label'])
            df_forecast['naive_ape'] = df_forecast['naive_ape'].map(lambda x: abs(x) if x not in [-np.inf, np.inf] else
            0 if math.isnan(x) else 0)

            df_forecast['linear_ape'] = ((df_forecast['gap_forecast_linear'] - df_forecast['gap_forecast_label']) /
                                         df_forecast['gap_forecast_label'])
            df_forecast['linear_ape'] = df_forecast['linear_ape'].map(lambda x: abs(x) if x not in [-np.inf, np.inf] else
            0 if math.isnan(x) else 0)

            df_forecast['xgb_ape'] = ((df_forecast['gap_forecast_xgb'] - df_forecast['gap_forecast_label']) /
                                      df_forecast['gap_forecast_label'])
            df_forecast['xgb_ape'] = df_forecast['xgb_ape'].map(lambda x: abs(x) if x not in [-np.inf, np.inf] else
            0 if math.isnan(x) else 0)

            # Fill NA
            df_forecast.fillna(value={'naive_ape': 0, 'linearr_ape': 0, 'xgb_ape': 0}, inplace=True)

            # Calculate Weight
            df_forecast['gap_forecast_label_abs'] = df_forecast['gap_forecast_label'].map(lambda x: abs(x))
            tot_gap = sum(df_forecast['gap_forecast_label_abs'])
            df_forecast['weight'] = df_forecast['gap_forecast_label_abs'] * 1.0 / tot_gap

            # Weighted APE
            df_forecast['naive_wape'] = df_forecast['naive_ape'] * df_forecast['weight']
            df_forecast['linear_wape'] = df_forecast['linear_ape'] * df_forecast['weight']
            df_forecast['xgb_wape'] = df_forecast['xgb_ape'] * df_forecast['weight']

            # Print
            print('\n\nModel MAPE:')
            print(df_forecast[['naive_wape', 'linear_wape',
                         'xgb_wape']].sum().reset_index())

        return

    def _write_to_realtime_gap_forecasts(self, records):

        """Use upsert feature in postgres version >= 9.5."""
        sql = """
        insert into realtime_forecasts_v2 (
            zone_id, date, hour, forecast_type, 
            forecast_name, data, created_at
        )
        values (
            :zone_id, :date, :hour, :forecast_type,
            :forecast_name, :data, :created_at
        )
        on conflict (zone_id, date, hour, forecast_type)
        where zone_id = :zone_id
        and date = :date
        and hour = :hour
        and forecast_type = :forecast_type
        do update set
            forecast_name = excluded.forecast_name,
            data = excluded.data,
            created_at = excluded.created_at
        """

        statement = scm.sql.text(sql)
        statement = statement.bindparams(scm.bindparam('data', type_=postgresql.JSONB))

        # records is a list of dict, like [{'zone_id': 1, 'whl_id': 1}, {'zone_id': 1, 'whl_id': 2}]
        with lore.io.logistics_write._engine.begin() as conn:
            conn.execute(statement, records)

        return

if __name__ == '__main__':

    f = RealTimeGapForecast(32, mode='training', running_time=datetime(2020, 7, 27), days=56)
    # f = RealTimeGapForecast(32, mode='scoring')
    f.train_score()
    f.check_model_performance()