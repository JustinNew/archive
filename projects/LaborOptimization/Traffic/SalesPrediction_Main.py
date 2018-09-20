import pandas as pd
import math, os

from salestools import ToBlackFriday, WeekBeforeLaborDay, WeekAfterBlackFriday
from salestools import WeekAfterChristmas, WeekAfterNewYear, WeekBeforeChristmas1, WeekBeforeChristmas2
from predictionmodel import SalesPredictionModel
from ingestdata import ReadInTrafficData

####################################################################################################
####################################################################################################
# Parameters to set
####################################################################################################
####################################################################################################

# Define global variables
startdate_sales = '2016-09-27'
startdate_train = '2017-11-01'
startmonth_model = 201710

# Special Days
easter_days = ['2012-04-08', '2013-03-31', '2014-04-20', '2015-04-05', '2016-03-27', '2017-04-16', '2018-04-01']
christmas_days = ['2012-12-25', '2013-12-25', '2014-12-25', '2015-12-25', '2016-12-25', '2017-12-25', '2018-12-25']
thanksgiving_days = ['2012-11-22', '2013-11-28', '2014-11-27', '2015-11-26', '2016-11-24', '2017-11-23', '2018-11-22']

# File names for sales, calendar, PO Dates and Holiday.
file_finance_calendar = './20171011 Finance Calendar.csv'
file_holiday = './Calendar Holiday Events - dates.csv'
file_sales = './sales_store1.csv'
file_po_date = './po_date_heuristic_v1.csv'
file_traffic = './Hourly_Traffic Beg 17 - Aug 18.txt'

store_list = [7, 8, 9, 10, 14, 17, 18, 19, 25, 31, 32, 34, 35, 37, 41, 43, 48, 49, 54, 55, 57, 60, 62, 63, 64, 70, 71, 72, 73, 75, 77, 79, 84, 86, 89, 91, 95, 96, 97, 101, 103, 104, 105, 106, 107, 109, 111, 112, 113, 114, 115, 117, 118, 119, 121, 123, 125, 126, 127, 129, 130, 132, 134, 136, 138, 139, 140, 141, 142, 143, 147, 148, 149, 153, 154, 155, 156, 158, 159, 160, 162, 163, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 182, 183, 184, 185, 186, 187, 188, 189, 191, 193, 195, 196, 197, 198, 202, 205, 206, 209, 210, 211, 212, 213, 214, 215, 217, 218, 219, 220, 221, 222, 224, 225, 226, 227, 228, 229, 230, 231, 232, 235, 236, 237, 239, 240, 241, 242, 244, 247, 248, 249, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 262, 263, 264, 265, 266, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 280, 281, 282, 283, 284, 285, 286, 287, 288, 290, 292, 293, 296, 297, 298, 299, 301, 302, 303, 304, 305, 306, 307, 308, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 323, 324, 325, 326, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 367, 368, 369, 370, 371, 373, 377, 379, 380, 381, 382, 383, 384, 385, 386, 388, 389, 394, 395, 397, 398, 399, 401, 403, 405, 406, 407, 408, 409, 411, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 425, 426, 427, 428, 429, 430, 431, 432, 433, 435, 436, 437, 438, 439, 440, 441, 442, 444, 445, 446, 447, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 500, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 513, 514, 515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 526, 527, 528, 529, 530, 531, 532, 533, 534, 535, 537, 538, 539, 540, 542, 544, 545, 546, 547, 548, 549, 550, 551, 552, 553, 555, 557, 559, 560, 562, 563, 564, 565, 567, 568, 569, 570, 574, 575, 576, 579, 580, 581, 583, 585, 586, 588, 589, 590, 591, 593, 595, 596, 598, 601, 602, 604, 605, 608, 609, 610, 611, 612, 613, 614, 615, 616, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 630, 631, 633, 634, 636, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 652, 653, 654, 656, 658, 659, 661, 662, 663, 664, 667, 668, 669, 673, 674, 675, 677, 678, 679, 680, 681, 682, 685, 686, 687, 689, 690, 691, 692, 693, 694, 695, 696, 697, 698, 701, 702, 703, 704, 705, 707, 708, 709, 710, 711, 712, 713, 714, 715, 716, 720, 721, 722, 723, 724, 725, 726, 727, 728, 729, 730, 731, 732, 733, 734, 735, 736, 737, 738, 739, 740, 741, 742, 743, 745, 746, 747, 748, 749, 752, 753, 755, 756, 757, 758, 759, 760, 761, 762, 763, 765, 767, 768, 769, 770, 773, 774, 776, 777, 778, 779, 780, 782, 783, 784, 785, 786, 787, 788, 790, 795, 796, 797, 798, 799, 951, 952, 953, 954, 955, 956, 957, 958, 960, 962, 963, 964, 967, 968, 969, 971, 973, 975, 976, 977, 980, 981, 982, 983, 985, 986, 987, 988, 990, 991, 992, 993, 994, 995, 997, 998, 1002, 1004, 1006, 1007, 1008, 1009, 1010, 1011, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1027, 1028, 1030, 1032, 1033, 1034, 1035, 1036, 1037, 1038, 1040, 1041, 1042, 1043, 1045, 1047, 1048, 1049, 1050, 1051, 1052, 1053, 1054, 1055, 1056, 1057, 1058, 1059, 1060, 1062, 1063, 1064, 1065, 1066, 1067, 1068, 1069, 1070, 1071, 1072, 1073, 1075, 1076, 1077, 1078, 1079, 1081, 1082, 1083, 1084, 1085, 1086, 1087, 1089, 1091, 1092, 1093, 1094, 1095, 1096, 1097, 1098, 1102, 1103, 1104, 1105, 1106, 1107, 1108, 1109, 1110, 1111, 1112, 1113, 1115, 1117, 1118, 1119, 1121, 1122, 1123, 1124, 1125, 1126, 1127, 1128, 1132, 1133, 1134, 1135, 1136, 1137, 1138, 1139, 1140, 1141, 1142, 1143, 1145, 1147, 1149, 1151, 1153, 1155, 1156, 1157, 1158, 1159, 1160, 1161, 1162, 1163, 1165, 1166, 1167, 1168, 1169, 1171, 1172, 1173, 1174, 1175, 1177, 1179, 1181, 1182, 1183, 1185, 1186, 1187, 1188, 1189, 1190, 1191, 1192, 1193, 1194, 1195, 1196, 1197, 1198, 1199, 1200, 1202, 1203, 1204, 1205, 1207, 1209, 1211, 1212, 1215, 1216, 1217, 1219, 1221, 1223, 1224, 1225, 1227, 1228, 1230, 1232, 1233, 1235, 1236, 1237, 1238, 1240, 1241, 1242, 1243, 1245, 1246, 1247, 1249, 1250, 1253, 1255, 1256, 1261, 1262, 1264, 1265, 1266, 1267, 1268, 1269, 1271, 1272, 1275, 1276, 1277, 1279, 1281, 1282, 1283, 1284, 1285, 1286, 1287, 1288, 1290, 1291, 1292, 1293, 1294, 1295, 1300, 1301, 1302, 1303, 1304, 1305, 1306, 1307, 1308, 1309, 1310, 1312, 1313, 1314, 1315, 1316, 1317, 1318, 1319, 1320, 1324, 1326, 1327, 1329, 1330, 1331, 1332, 1334, 1340, 1342, 1343, 1347, 1349, 1350, 1352, 1353, 1354, 1355, 1356, 1357, 1358, 1359, 1360, 1361, 1362, 1363, 1365, 1370, 1372, 1373, 1374, 1376, 1378, 1381, 1382, 1383, 1385, 1386, 1388, 1389, 1390, 1391, 1393, 1396, 1397, 1399, 1400, 1401, 1403, 1404, 1405, 1407, 1408, 1409, 1410, 1411, 1413, 1415, 1416, 1418, 1419, 1420, 1421, 1422, 1424, 1425, 1426, 1428, 1429, 1430, 1431, 1432, 1434, 1435, 1436, 1437, 1438, 1439, 1440, 1441, 1442, 1443, 1445, 1447, 1453, 1454, 1455, 1456, 1458, 1462, 1463, 1466, 1467, 1468, 1469, 1470, 1473, 1474, 1476, 1480, 1481, 1482, 1483, 1484, 1487, 1492, 1493, 1494, 1497, 1498, 1499, 1500, 1501, 1502, 1503, 1504, 1506, 1507, 1508, 1510, 1515, 1516, 1517, 1518, 1519, 1520, 1522, 1529, 1534, 1535, 1536, 1537, 1538, 1539, 1540, 1541, 1542, 1543, 1544, 1545]

if __name__ == '__main__':

    # Check whether the Daily Traffic data for each store exists.
    if not os.path.exists('./StoreDailyTraffic'):
        os.mkdir('./StoreDailyTraffic')

    # Read in the single traffic data file and split into store daily traffic.
	ReadInTrafficData(file_traffic)

    ####################################################################################################################
    # Read in PO Dates
    PODates = pd.read_csv(file_po_date)
    PODates = PODates[['CAL_EVNT_DTE', 'computed_po_date']]
    PODates.rename(columns={'CAL_EVNT_DTE': 'Date', 'computed_po_date': 'po_date'}, inplace=True)

    ####################################################################################################################
    # Read in Finance Calendar
    calendar = pd.read_csv(file_finance_calendar)
    calendar['Date'] = pd.to_datetime(calendar['Date'], infer_datetime_format=True)
    calendar['Date'] = calendar['Date'].astype('str')
    calendar = calendar[calendar['Date'].notnull()]

    # Fix inconsistences.
    # Replacing all 'XXX' with 'YYY'
    calendar['Event1_Discount_Type'] = calendar['Event1_Discount_Type'].map(lambda x: x.replace('off', 'Off'))

    # Replacing all space, '&'.
    calendar['Event1_Distribution'] = calendar['Event1_Distribution'].map(lambda x: x.replace(' ', ''))
    calendar['Event2_Distribution'] = calendar['Event2_Distribution'].map(lambda x: x.replace(' ', ''))
    calendar['Event3_Distribution'] = calendar['Event3_Distribution'].map(lambda x: x.replace(' ', ''))
    calendar['Event1_Name'] = calendar['Event1_Name'].map(lambda x: x.replace('&', ''))
    calendar['Event2_Name'] = calendar['Event2_Name'].map(lambda x: x.replace('&', ''))
    calendar['Event3_Name'] = calendar['Event3_Name'].map(lambda x: x.replace('&', ''))
    calendar['Event1_Name'] = calendar['Event1_Name'].map(lambda x: x.replace(' ', ''))
    calendar['Event2_Name'] = calendar['Event2_Name'].map(lambda x: x.replace(' ', ''))
    calendar['Event3_Name'] = calendar['Event3_Name'].map(lambda x: x.replace(' ', ''))

    ####################################################################################################################
    # Read in Holiday Data
    holiday = pd.read_csv(file_holiday)
    holiday = holiday[holiday.Date.notnull()][['Date', 'Holiday']]
    dates_temp = pd.date_range(start=holiday.Date.min(), end=holiday.Date.max())
    df_temp = pd.DataFrame(dates_temp, columns=['Date'])
    df_temp['Date'] = df_temp['Date'].astype('str')
    holiday = holiday.merge(df_temp, on='Date', how='right')
    holiday.sort_values(by=['Date'], inplace=True)

    ####################################################################################################################
    # Design Matrix for Events
    from designmatrix import EventDesignMatrix

    assert isinstance(calendar, pd.DataFrame)
    calendar = EventDesignMatrix(calendar)

    ####################################################################################################################
    # Design Matrix for Holiday
    from designmatrix import HolidayObserved, HolidayDesignMatrix

    assert isinstance(holiday, pd.DataFrame)
    holiday = HolidayObserved(holiday)

    holiday['Holiday'] = holiday['Holiday'].astype('str')
    holiday['Holiday'] = holiday['Holiday'].map(lambda x: x.replace("'", ""))
    holiday['Holiday'] = holiday['Holiday'].map(lambda x: x.replace(" ", ""))
    holiday['Holiday'] = holiday['Holiday'].map(lambda x: x.replace(".", ""))
    holiday = holiday[holiday['Date'].notnull()]

    holiday = HolidayDesignMatrix(holiday)

    ####################################################################################################################
    # The feature names.
    holiday_names = [i for i in holiday.columns.tolist() if i != 'Date']
    event_names = [i for i in calendar.columns.tolist() if i != 'Date']

    ####################################################################################################################
    # Get PO Dates for Sales, Calendar and Holiday

    ####################################################################################################################
    # Derived Holiday

    ####################################################################################################################
    # Read in Sales Data.
    # Train and Predict.
    dict_mapes = {}
    for store_id in store_list:
        # "Sales" is Traffic here.
        sales = pd.read_csv('./StoreDailyTraffic/DailyTraffic_' + str(store_id) + '.csv')

        # Join sales, holiday and calendar DataFrames together.
        sales = sales[['Date', 'VSTR_IN_CNT']].merge(holiday, on='Date')
        sales = sales.merge(calendar, on='Date')

        # Get the sales DataFrame ready.
        sales = sales[['Date', 'VSTR_IN_CNT'] + holiday_names + event_names]

        # Get rid of unwanted dates.
        sales = sales[~(sales['Date'].isin(easter_days + christmas_days))]
        sales = sales[sales['VSTR_IN_CNT'] > 0]

        # Get log sales and get rid of null sales date
        sales['log_VSTR_IN_CNT'] = sales['VSTR_IN_CNT'].map(lambda x: math.log(x))
        sales = sales[sales.VSTR_IN_CNT.notnull()]

        sales['trn_sls_dte'] = pd.to_datetime(sales['Date'], infer_datetime_format=True)
        sales['fscl_mn_id'] = sales['trn_sls_dte'].dt.year * 100 + sales['trn_sls_dte'].dt.month

        # Get the fscl_mn_id starting from startmonth_train until the last fscl_mn_id in the DataFrame.
        fscls_list = sales.fscl_mn_id.unique().tolist()
        fscls = [i for i in fscls_list if i >= startmonth_model]

        # Feature Engineering
        ChristmasShopDay = ['2014-12-23', '2015-12-23', '2016-12-23', '2017-12-23', '2018-12-23', '2019-12-23']
        sales['ChristShopDay'] = sales['Date'].map(lambda x: 1 if x in ChristmasShopDay else 0)
        sales['ToBlackFriday'] = sales['trn_sls_dte'].map(lambda x: ToBlackFriday(x))
        sales['WeekBeforeLaborDay'] = sales['trn_sls_dte'].map(lambda x: WeekBeforeLaborDay(x))
        sales['WeekAfterBlackFriday'] = sales['trn_sls_dte'].map(lambda x: WeekAfterBlackFriday(x))
        sales['WeekAfterChristmas'] = sales['trn_sls_dte'].map(lambda x: WeekAfterChristmas(x))
        sales['WeekAfterNewYear'] = sales['trn_sls_dte'].map(lambda x: WeekAfterNewYear(x))
        sales['WeekBeforeChristmas1'] = sales['trn_sls_dte'].map(lambda x: WeekBeforeChristmas1(x))
        sales['WeekBeforeChristmas2'] = sales['trn_sls_dte'].map(lambda x: WeekBeforeChristmas2(x))
        sales['Weekday'] = sales['trn_sls_dte'].dt.weekday
        sales['Month'] = sales['trn_sls_dte'].dt.month
        sales = sales.join(pd.get_dummies(sales.Weekday, prefix='Weekday'))
        sales = sales.join(pd.get_dummies(sales.Month, prefix='Month'))

        # Features
        day_names = ['Weekday_0', 'Weekday_1', 'Weekday_2', 'Weekday_3', 'Weekday_4', 'Weekday_5', 'Weekday_6',
                     'Month_1', 'ChristShopDay', 'WeekBeforeLaborDay', 'WeekAfterBlackFriday', 'WeekAfterChristmas',
                     'WeekAfterNewYear', 'WeekBeforeChristmas1', 'WeekBeforeChristmas2', 'ToBlackFriday']
        holiday_names = ['MemorialDay', 'IndependenceDay', 'LaborDay', 'ColumbusDay', 'BlackFriday', 'NewYearsDay',
                         'ChristmasDay', 'VeteransDay', 'Halloween', 'CyberMonday', 'GreenMonday', 'MLKJrDay',
                         'PresidentsDay', 'NewYearsEve']
        feature_cols = holiday_names + event_names + day_names

        # If feature not in sales, for example store 381 does not have Sunday Traffic data, do not include the feature.
        t_columns = sales.columns.tolist()
        feature_cols = [i for i in feature_cols if i in t_columns]

        # Train and Get MAPE
        # flag_log: True or False, whether to use logSales as dependent variable.
        # falg_model: 'rf' or 'xgb'
        '''
        SalesPredictionModel(store_id: int, sales: pd.DataFrame, fscls: List[int],
                         feature_cols: List[str], flag_log: bool = False,
                         flag_model: str = 'rf', startmonth_model: int = 201710) -> None:
        '''
        dict_mapes[store_id] = SalesPredictionModel(store_id, sales, fscls, feature_cols, False, 'rf', startmonth_model)

    # Save MAPES
    dict_mapes = {'stores': list(dict_mapes.keys()), 'mapes': list(dict_mapes.values())}
    df_mapes = pd.DataFrame(dict_mapes)
    df_mapes.to_csv('../TrafficPrediction/all_mapes.csv')

    print('Great. Done.')