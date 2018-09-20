import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import train_test_split
from salestools import Prior_Month
from typing import List


def SalesPredictionModel(store_id: int, sales: pd.DataFrame, fscls: List[int],
						 feature_cols: List[str], flag_log: bool = False,
						 flag_model: str = 'rf', startmonth_model: int = 201710) -> None:

	dummy = [{'log_diff': 0, 'pred': 0}]
	df_pred_all = pd.DataFrame(dummy)

	for time_filter in fscls:
		# Get prior_mn and test_mn
		prior_mn = Prior_Month(time_filter)

		# Get train, test and predict DataFrame
		train_data = sales[sales.fscl_mn_id <= prior_mn][feature_cols]
		predict_data = sales[sales.fscl_mn_id == time_filter][feature_cols]

		if flag_log:
			train_label = sales[sales.fscl_mn_id <= prior_mn]['log_VSTR_IN_CNT']
			predict_label = sales[sales.fscl_mn_id == time_filter]['log_VSTR_IN_CNT']
		else:
			train_label = sales[sales.fscl_mn_id <= prior_mn]['VSTR_IN_CNT']
			predict_label = sales[sales.fscl_mn_id == time_filter]['VSTR_IN_CNT']

		# Model
		if flag_model == 'xgb':
			model = XGBRegressor(n_jobs=-1, silent=1, subsample=0.8, eval_metric='rmse', max_depth=5,
								 n_estimator=100, gamma=0.3, min_child_weight=1)
		elif flag_model == 'rf':
			model = RandomForestRegressor(n_jobs=-1, max_depth=8, min_samples_leaf=2, n_estimators=120)

		model.fit(train_data, train_label)

		# Get prediction for each of the month
		ypred = model.predict(predict_data)
		df_pred_t = pd.DataFrame(predict_label)
		df_pred_t = df_pred_t.assign(pred=list(ypred))

		# Union to get all for each of the fscl_mn_id
		df_pred_all = pd.concat([df_pred_all, df_pred_t])

	# After prediction for all the month, get all APE and MAPE.
	df_predict = sales[sales.fscl_mn_id >= startmonth_model]
	df_predict = df_predict[['Date', 'fscl_mn_id', 'VSTR_IN_CNT', 'trn_sls_dte']]
	df_predict = df_predict.join(df_pred_all['pred'], how='left')

	if flag_log:
		df_predict['VSTR_IN_CNT_pred'] = df_predict['pred'].map(lambda x: math.exp(x))
	else:
		df_predict['VSTR_IN_CNT_pred'] = df_predict['pred']

	df_predict['ape'] = (df_predict['VSTR_IN_CNT'] - df_predict['VSTR_IN_CNT_pred']) / df_predict['VSTR_IN_CNT']
	df_predict['abs_ape'] = df_predict['ape'].map(lambda x: abs(x))

	# Save APE and MAPE to file
	if flag_log:
		ape_file = '../TrafficPrediction/APE_MAPE_Log/Error_APE_' + str(store_id) + '.csv'
		mape_file = '../TrafficPrediction/APE_MAPE_Log/Error_MAPE_' + str(store_id) + '.csv'
	else:
		ape_file = '../TrafficPrediction/APE_MAPE/Error_APE_' + str(store_id) + '.csv'
		mape_file = '../TrafficPrediction/APE_MAPE/Error_MAPE_' + str(store_id) + '.csv'

	df_predict.to_csv(ape_file)
	df_mape = df_predict[['fscl_mn_id', 'abs_ape']].groupby('fscl_mn_id').agg({'abs_ape': np.mean}).reset_index()
	df_mape.to_csv(mape_file)

	mape_mean = np.mean(df_mape.abs_ape)
	print('For %d MAPE is %.3f.'%(store_id, mape_mean))

	return mape_mean