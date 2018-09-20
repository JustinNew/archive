'''
This is the module for Holiday and Event DesignMatrix.
'''
import pandas as pd

########################################################################################################################
# This is for Observed Holiday.
def HolidayObserved(holiday: pd.DataFrame) -> pd.DataFrame:
	"""
	:param holiday: pd.DataFrame -> [['Date', 'Holiday']]
	:return: pd.DataFrame
	"""
	# Assign "New Year's Day" value.
	holiday.loc[holiday['Date'].isin(['2016-12-31']), ["Holiday"]] = "New Year's Day"

	# Observed holiday
	holiday.loc[holiday['Date'].isin(['2017-01-02']), ["Holiday"]] = "New Year's Day"
	holiday.loc[holiday['Date'].isin(['2016-12-26']), ["Holiday"]] = "Christmas Day"
	holiday.loc[holiday['Date'].isin(['2015-07-03']), ["Holiday"]] = "Independence Day"

	return holiday

########################################################################################################################
# This is for Holiday DesignMatrix.

def HolidayDesignMatrix(holiday: pd.DataFrame) -> pd.DataFrame:
	"""
	:param holiday: pd.DataFrame -> [['Date', 'Holiday']]
	:return: pd.DataFrame
	"""
	# Each Holiday as a feature.
	# Assume two columns in 'holiday' data.frame: 'Date', 'Holiday'.

	# Need to make sure the columns used for design matrix are presented.
	namelist_old = holiday.columns.tolist()

	# Create features using 'Holiday'.
	holiday_uniques = holiday['Holiday'].unique().tolist()
	holiday_uniques = [i for i in holiday_uniques if i != 'nan']

	for i in range(len(holiday_uniques)):
		suffix = holiday_uniques[i]
		holiday[suffix] = holiday['Holiday'].map(lambda x: 1 if x == suffix else 0)

	# Get the new name lists.
	namelist = holiday.columns.tolist()

	holiday_names = [i for i in namelist if i not in namelist_old]

	return holiday[['Date'] + holiday_names]

########################################################################################################################
# This is for Event DesignMatrix.

def EventDesignMatrix(calendar: pd.DataFrame) -> pd.DataFrame:
	"""
    :param calendar: pd.DataFrame -> [['Date', 'Event1_Name', 'Event1_Distribution', ...]]
	:rtype: pd.DataFrame
	"""
	# Hard coding included.
	# Create features using Event1_Name+Event1_Distribution, (E2_Name, E2_Distribution) | (E3_Name, E3_Distribution).

	# Need to make sure the columns used for design matrix are presented.
	namelist_old = calendar.columns.tolist()

	# Get 'Event1_Name' and 'Event1_Distribution' unique combinations.
	main_uniques = calendar[['Event1_Name', 'Event1_Distribution']]
	main_uniques = main_uniques.groupby(['Event1_Name', 'Event1_Distribution']).count().reset_index()
	main_uniques = main_uniques[~((main_uniques.Event1_Name == 'n/a') &
								  (main_uniques.Event1_Distribution == 'n/a'))]

	# Create features using 'Event1_Name' and 'Event1_Distribution'
	for i in range(len(main_uniques)):
		prefix = main_uniques.iloc[i, 0]
		suffix = main_uniques.iloc[i, 1]
		v_name = prefix + '_' + suffix
		calendar[v_name] = calendar[['Event1_Name',
									 'Event1_Distribution']].apply(
			lambda x: 1 if x[0] == prefix and x[1] == suffix else 0, axis=1)

	# Create features using 'Event2_Name' or 'Event3_Name'.
	E2_uniques = calendar['Event2_Name'].unique().tolist()
	E3_uniques = calendar['Event3_Name'].unique().tolist()
	E23_uniques = [i for i in list(set(E2_uniques + E3_uniques)) if i != 'n/a']

	for i in range(len(E23_uniques)):
		suffix = E23_uniques[i]
		v_name = 'E23_' + suffix
		calendar[v_name] = calendar[['Event2_Name',
									 'Event3_Name']].apply(lambda x: 1 if x[0] == suffix or x[1] == suffix else 0,
														   axis=1)

	# Create featues using 'Event2_Distribution' or 'Event3_Distribution'.
	D2_uniques = calendar['Event2_Distribution'].unique().tolist()
	D3_uniques = calendar['Event3_Distribution'].unique().tolist()
	D23_uniques = [i for i in list(set(D2_uniques + D3_uniques)) if i != 'n/a']

	for i in range(len(D23_uniques)):
		suffix = D23_uniques[i]
		v_name = 'D23_' + suffix
		calendar[v_name] = calendar[['Event2_Distribution',
									 'Event3_Distribution']].apply(
			lambda x: 1 if x[0] == suffix or x[1] == suffix else 0, axis=1)

	# Get the new name lists.
	namelist = calendar.columns.tolist()

	event_names = [i for i in namelist if i not in namelist_old]

	return calendar[['Date'] + event_names]
