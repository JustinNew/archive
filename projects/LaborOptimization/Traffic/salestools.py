import datetime

def Prior_Month(fscl_mn_id):
	year = int(fscl_mn_id / 100)
	mn = fscl_mn_id % 100

	if mn == 1:
		return (year - 1) * 100 + 11
	elif mn == 2:
		return (year - 1) * 100 + 12
	else:
		return year * 100 + mn - 2


def ToBlackFriday(x):
	if x.to_pydatetime().year == 2014:
		return (x.to_pydatetime() - datetime.datetime(2014, 11, 28)).days
	elif x.to_pydatetime().year == 2015:
		return (x.to_pydatetime() - datetime.datetime(2015, 11, 27)).days
	elif x.to_pydatetime().year == 2016:
		return (x.to_pydatetime() - datetime.datetime(2016, 11, 25)).days
	elif x.to_pydatetime().year == 2017:
		return (x.to_pydatetime() - datetime.datetime(2017, 11, 24)).days
	elif x.to_pydatetime().year == 2018:
		return (x.to_pydatetime() - datetime.datetime(2018, 11, 23)).days
	elif x.to_pydatetime().year == 2019:
		return (x.to_pydatetime() - datetime.datetime(2019, 11, 29)).days


def WeekBeforeLaborDay(x):
	if x.to_pydatetime().year == 2014:
		t = (x.to_pydatetime() - datetime.datetime(2014, 9, 1)).days
	elif x.to_pydatetime().year == 2015:
		t = (x.to_pydatetime() - datetime.datetime(2015, 9, 7)).days
	elif x.to_pydatetime().year == 2016:
		t = (x.to_pydatetime() - datetime.datetime(2016, 9, 5)).days
	elif x.to_pydatetime().year == 2017:
		t = (x.to_pydatetime() - datetime.datetime(2017, 9, 4)).days
	elif x.to_pydatetime().year == 2018:
		t = (x.to_pydatetime() - datetime.datetime(2018, 9, 3)).days
	elif x.to_pydatetime().year == 2019:
		t = (x.to_pydatetime() - datetime.datetime(2019, 9, 2)).days

	if t >= -7 and t <= -1:
		return 1
	else:
		return 0


def WeekAfterBlackFriday(x):
	if x.to_pydatetime().year == 2014:
		t = (x.to_pydatetime() - datetime.datetime(2014, 11, 28)).days
	elif x.to_pydatetime().year == 2015:
		t = (x.to_pydatetime() - datetime.datetime(2015, 11, 27)).days
	elif x.to_pydatetime().year == 2016:
		t = (x.to_pydatetime() - datetime.datetime(2016, 11, 25)).days
	elif x.to_pydatetime().year == 2017:
		t = (x.to_pydatetime() - datetime.datetime(2017, 11, 24)).days
	elif x.to_pydatetime().year == 2018:
		t = (x.to_pydatetime() - datetime.datetime(2018, 11, 23)).days
	elif x.to_pydatetime().year == 2019:
		t = (x.to_pydatetime() - datetime.datetime(2019, 11, 29)).days

	if t >= 1 and t <= 9:
		return 1
	else:
		return 0


def WeekAfterChristmas(x):
	if x.to_pydatetime().year == 2014:
		t = (x.to_pydatetime() - datetime.datetime(2014, 12, 25)).days
	elif x.to_pydatetime().year == 2015:
		t = (x.to_pydatetime() - datetime.datetime(2015, 12, 25)).days
	elif x.to_pydatetime().year == 2016:
		t = (x.to_pydatetime() - datetime.datetime(2016, 12, 25)).days
	elif x.to_pydatetime().year == 2017:
		t = (x.to_pydatetime() - datetime.datetime(2017, 12, 25)).days
	elif x.to_pydatetime().year == 2018:
		t = (x.to_pydatetime() - datetime.datetime(2018, 12, 25)).days
	elif x.to_pydatetime().year == 2019:
		t = (x.to_pydatetime() - datetime.datetime(2019, 12, 25)).days

	if t >= 1 and t <= 6:
		return 1
	else:
		return 0


def WeekAfterNewYear(x):
	if x.to_pydatetime().year == 2014:
		t = (x.to_pydatetime() - datetime.datetime(2014, 1, 1)).days
	elif x.to_pydatetime().year == 2015:
		t = (x.to_pydatetime() - datetime.datetime(2015, 1, 1)).days
	elif x.to_pydatetime().year == 2016:
		t = (x.to_pydatetime() - datetime.datetime(2016, 1, 1)).days
	elif x.to_pydatetime().year == 2017:
		t = (x.to_pydatetime() - datetime.datetime(2017, 1, 1)).days
	elif x.to_pydatetime().year == 2018:
		t = (x.to_pydatetime() - datetime.datetime(2018, 1, 1)).days
	elif x.to_pydatetime().year == 2019:
		t = (x.to_pydatetime() - datetime.datetime(2019, 1, 1)).days

	if t >= 1 and t <= 7:
		return 1
	else:
		return 0


def WeekBeforeChristmas1(x):
	if x.to_pydatetime().year == 2014:
		t = (x.to_pydatetime() - datetime.datetime(2014, 12, 25)).days
	elif x.to_pydatetime().year == 2015:
		t = (x.to_pydatetime() - datetime.datetime(2015, 12, 25)).days
	elif x.to_pydatetime().year == 2016:
		t = (x.to_pydatetime() - datetime.datetime(2016, 12, 25)).days
	elif x.to_pydatetime().year == 2017:
		t = (x.to_pydatetime() - datetime.datetime(2017, 12, 25)).days
	elif x.to_pydatetime().year == 2018:
		t = (x.to_pydatetime() - datetime.datetime(2018, 12, 25)).days
	elif x.to_pydatetime().year == 2019:
		t = (x.to_pydatetime() - datetime.datetime(2019, 12, 25)).days

	if t >= -7 and t <= -1:
		return 1
	else:
		return 0


def WeekBeforeChristmas2(x):
	if x.to_pydatetime().year == 2014:
		t = (x.to_pydatetime() - datetime.datetime(2014, 12, 25)).days
	elif x.to_pydatetime().year == 2015:
		t = (x.to_pydatetime() - datetime.datetime(2015, 12, 25)).days
	elif x.to_pydatetime().year == 2016:
		t = (x.to_pydatetime() - datetime.datetime(2016, 12, 25)).days
	elif x.to_pydatetime().year == 2017:
		t = (x.to_pydatetime() - datetime.datetime(2017, 12, 25)).days
	elif x.to_pydatetime().year == 2018:
		t = (x.to_pydatetime() - datetime.datetime(2018, 12, 25)).days
	elif x.to_pydatetime().year == 2019:
		t = (x.to_pydatetime() - datetime.datetime(2019, 12, 25)).days

	if t >= -14 and t <= -8:
		return 1
	else:
		return 0