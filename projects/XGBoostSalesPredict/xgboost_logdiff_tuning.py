import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split
import math

def prior_month(fscl_mn_id):
    year = int(fscl_mn_id / 100)
    mn = fscl_mn_id % 100

    if mn == 1:
        return (year - 1) * 100 + 11
    elif mn == 2:
        return (year - 1) * 100 + 12
    else:
        return year * 100 + mn - 2


def test_month(fscl_mn_id):
    year = int(fscl_mn_id / 100)
    mn = fscl_mn_id % 100

    if mn == 1:
        return (year - 1) * 100 + 12
    else:
        return year * 100 + mn - 1


def xgb_sales_predict(sales, holiday, calendar, time_names, store_id, flag):
    # Join sales, holiday and calendar DataFrames together.
    sales = sales.merge(holiday, on='Date')
    sales = sales.merge(calendar, on='Date')

    # the feature names and "po_" feature names.
    holiday_names = [i for i in holiday.columns.tolist() if i != 'Date' and i not in time_names and 'po_' not in i]
    event_names = [i for i in calendar.columns.tolist() if i != 'Date' and 'po_' not in i]

    po_holiday_names = ['po_' + i for i in holiday_names]
    po_event_names = ['po_' + i for i in event_names]

    # Get differences
    for i, j in zip(holiday_names + event_names, po_holiday_names + po_event_names):
        sales[i + '_diff'] = sales[i] - sales[j]

    # Get the sales DataFrame ready.
    sales.rename(columns={'po_date_c': 'po_date', 'kc_sales': 'net_chrgd_amt', 'po_kc_sales': 'po_net_chrgd_amt'},
                 inplace=True)
    sales = sales[['Date', 'fscl_mn_id', 'fscl_qtr_id', 'po_date', 'net_chrgd_amt', 'po_net_chrgd_amt'] +
                  [i + '_diff' for i in holiday_names + event_names] + time_names]

    # Get rid of unwanted dates.
    sales = sales[
        ~(sales['Date'].isin(easter_days + christmas_days)) & ~(sales['po_date'].isin(easter_days + christmas_days))]

    # Get log sales and log sales difference
    sales['log_net_chrgd_amt'] = sales['net_chrgd_amt'].map(lambda x: math.log(x))
    sales['log_po_net_chrgd_amt'] = sales['po_net_chrgd_amt'].map(lambda x: math.log(x))
    sales['log_diff'] = sales['log_net_chrgd_amt'] - sales['log_po_net_chrgd_amt']
    sales = sales[sales.log_po_net_chrgd_amt.notnull() & sales.log_net_chrgd_amt.notnull()]

    # Get the fscl_mn_id starting from startmonth_train until the last fscl_mn_id in the DataFrame.
    fscls_list = sales.fscl_mn_id.unique().tolist()
    fscls = [i for i in fscls_list if i >= startmonth_model]

    # Features
    excludes_cols = [k + '_diff' for k in exclude_vars_kc]
    features_cols = [j + '_diff' for j in holiday_names + event_names] + time_names
    features_cols = [i for i in features_cols if i not in excludes_cols]

    dummy = [{'log_diff': 0, 'pred': 0}]
    df_pred_all = pd.DataFrame(dummy)
    for time_filter in fscls:
        # Get prior_mn and test_mn
        prior_mn = prior_month(time_filter)
        test_mn = test_month(time_filter)

        # Get train, test and predict DataFrame
        train_data = sales[sales.fscl_mn_id <= prior_mn][features_cols]
        train_label = sales[sales.fscl_mn_id <= prior_mn]['log_diff']

        # train_data, test_data, train_label, test_label = train_test_split(train_data, train_label,
        # 																  test_size=0.05, random_state=42)

        predict_data = sales[sales.fscl_mn_id == time_filter][features_cols]
        predict_label = sales[sales.fscl_mn_id == time_filter]['log_diff']

        # Model
        if flag == 'xgb':
            model = XGBRegressor(n_jobs=-1, silent=1, subsample=0.8, eval_metric='rmse', max_depth=8,
                           n_estimator=60, gamma=0.3)
        elif flag == 'rf':
            model = RandomForestRegressor(n_jobs=-1, max_depth=8, min_samples_leaf=2, n_estimators=20)

        model.fit(train_data, train_label)

        # Get prediction for each of the month
        ypred = model.predict(predict_data)
        df_pred_t = pd.DataFrame(predict_label)
        df_pred_t = df_pred_t.assign(pred=list(ypred))

        # Union to get all for each of the fscl_mn_id
        df_pred_all = pd.concat([df_pred_all, df_pred_t])

    # After prediction for all the month, get all APE and MAPE.
    df_predict = sales[sales.fscl_mn_id >= startmonth_model]
    df_predict = df_predict[['Date', 'po_date', 'fscl_mn_id', 'fscl_qtr_id', 'net_chrgd_amt', 'po_net_chrgd_amt']]
    df_predict = df_predict.join(df_pred_all['pred'], how='left')
    df_predict['net_chrgd_amt_pred'] = df_predict[['po_net_chrgd_amt', 'pred']].apply(lambda x: x[0] * math.exp(x[1]),
                                                                                      axis=1)
    df_predict['ape'] = (df_predict['net_chrgd_amt'] - df_predict['net_chrgd_amt_pred']) / df_predict['net_chrgd_amt']
    df_predict['abs_ape'] = df_predict['ape'].map(lambda x: abs(x))

    # Save APE to file
    ape_file = '../store_forecast-master/RF_tuning_8_20_APE_MAPE/Error_APE_' + str(store_id) + '.csv'
    df_predict.to_csv(ape_file)

    # Get MAPE
    df_mape = df_predict[['fscl_mn_id', 'abs_ape']].groupby('fscl_mn_id').agg({'abs_ape': np.mean}).reset_index()
    mape_file = '../store_forecast-master/RF_tuning_8_20_APE_MAPE/Error_MAPE_' + str(store_id) + '.csv'
    df_mape.to_csv(mape_file)

    del df_mape, df_predict, df_pred_all, df_pred_t

    return

if __name__ == '__main__':

    # Define global variables
    stores_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                   29, 30, 31, 32, 33, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 50, 51, 52, 53, 54, 55, 57, 58,
                   59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 76, 78, 79, 80, 81, 82, 83, 85, 87, 88, 89,
                   90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 107, 108, 109, 110, 111, 112, 113,
                   114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134,
                   136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 147, 148, 151, 152, 153, 154, 155, 157, 159, 160, 161,
                   163, 164, 165, 166, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 181, 182, 183, 186, 187,
                   188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209,
                   210, 211, 212, 213, 214, 216, 217, 218, 219, 220, 221, 222, 223, 224, 226, 227, 228, 229, 230, 231, 232,
                   233, 235, 237, 238, 239, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256,
                   257, 258, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 271, 272, 273, 274, 275, 277, 278, 280, 281,
                   282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302,
                   303, 304, 306, 307, 308, 309, 310, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 324, 325, 326, 327,
                   328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 339, 340, 341, 342, 343, 344, 345, 346, 348, 349, 350,
                   353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374,
                   375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395,
                   396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416,
                   417, 419, 420, 421, 422, 423, 424, 425, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439,
                   440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 458, 459, 460, 461, 463, 464,
                   466, 467, 468, 469, 470, 471, 472, 474, 476, 477, 478, 479, 480, 482, 484, 485, 486, 487, 488, 489, 490,
                   492, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514,
                   515, 516, 517, 518, 519, 521, 522, 524, 526, 527, 528, 529, 531, 532, 533, 534, 535, 536, 537, 539, 541,
                   543, 544, 545, 546, 547, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559, 560, 561, 562, 563, 564,
                   565, 566, 567, 568, 569, 570, 571, 572, 573, 574, 575, 576, 577, 578, 579, 580, 581, 582, 583, 584, 585,
                   586, 587, 588, 589, 590, 591, 592, 593, 594, 596, 597, 598, 599, 600, 602, 603, 604, 605, 606, 607, 608,
                   609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 624, 625, 626, 627, 628, 629, 630,
                   631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651,
                   652, 653, 654, 655, 656, 657, 658, 659, 660, 661, 663, 664, 665, 666, 667, 668, 669, 670, 671, 672, 673,
                   674, 675, 676, 677, 678, 679, 680, 681, 682, 683, 684, 685, 687, 689, 690, 691, 692, 693, 695, 696, 698,
                   700, 701, 702, 703, 704, 705, 706, 707, 708, 710, 711, 712, 713, 714, 716, 717, 718, 719, 720, 721, 722,
                   723, 725, 726, 727, 728, 729, 730, 731, 732, 733, 734, 735, 736, 737, 738, 739, 740, 742, 743, 744, 745,
                   746, 747, 748, 749, 750, 751, 752, 753, 754, 755, 756, 758, 759, 760, 761, 762, 763, 765, 766, 767, 768,
                   769, 771, 772, 773, 774, 775, 776, 777, 778, 779, 780, 781, 782, 783, 784, 785, 786, 787, 789, 790, 793,
                   794, 795, 796, 797, 798, 799, 800, 801, 802, 803, 804, 805, 806, 807, 808, 809, 810, 811, 813, 814, 815,
                   816, 818, 819, 820, 821, 823, 824, 825, 826, 827, 828, 829, 830, 831, 832, 833, 834, 835, 836, 837, 839,
                   840, 842, 843, 844, 845, 846, 847, 848, 849, 850, 851, 852, 853, 854, 855, 856, 857, 858, 859, 860, 861,
                   862, 863, 864, 865, 866, 867, 868, 869, 870, 871, 872, 873, 874, 875, 876, 877, 878, 879, 880, 881, 883,
                   884, 885, 886, 887, 888, 889, 890, 891, 892, 893, 894, 895, 896, 897, 898, 899, 900, 901, 902, 903, 904,
                   905, 907, 908, 909, 910, 914, 915, 916, 917, 918, 919, 920, 921, 922, 923, 924, 925, 926, 927, 928, 929,
                   930, 931, 932, 933, 934, 935, 936, 937, 938, 939, 940, 941, 942, 944, 945, 946, 947, 948, 949, 950, 951,
                   952, 953, 955, 956, 957, 958, 959, 960, 961, 962, 963, 964, 965, 966, 967, 968, 969, 970, 971, 972, 973,
                   974, 975, 976, 977, 978, 980, 981, 982, 984, 985, 986, 987, 988, 990, 992, 993, 994, 995, 996, 997, 998,
                   999, 1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016,
                   1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1032, 1033,
                   1034, 1035, 1036, 1037, 1038, 1039, 1041, 1042, 1043, 1044, 1045, 1046, 1047, 1048, 1049, 1050, 1051,
                   1052, 1053, 1054, 1055, 1056, 1057, 1058, 1059, 1060, 1061, 1062, 1063, 1064, 1065, 1066, 1068, 1069,
                   1071, 1072, 1073, 1074, 1075, 1076, 1077, 1078, 1079, 1080, 1081, 1082, 1083, 1084, 1085, 1086, 1087,
                   1088, 1089, 1090, 1092, 1093, 1094, 1095, 1098, 1100, 1101, 1102, 1103, 1104, 1105, 1106, 1107, 1108,
                   1109, 1110, 1111, 1112, 1113, 1114, 1115, 1116, 1117, 1118, 1119, 1120, 1121, 1122, 1123, 1124, 1126,
                   1127, 1128, 1129, 1130, 1131, 1132, 1133, 1135, 1136, 1137, 1138, 1141, 1142, 1143, 1144, 1145, 1147,
                   1148, 1152, 1153, 1154, 1155, 1157, 1159, 1160, 1161, 1162, 1163, 1164, 1165, 1166, 1168, 1170, 1171,
                   1172, 1173, 1174, 1175, 1176, 1177, 1178, 1179, 1180, 1181, 1183, 1184, 1185, 1186, 1187, 1188, 1189]

    n = len(stores_list)
    n1 = int(n / 3)
    stores_list_1 = stores_list[:n1]
    stores_list_2 = stores_list[n1:n1 + n1]
    stores_list_3 = stores_list[n1 + n1:]

    startdate_sales = '2014-03-02'
    startdate_train = '2015-02-01'
    startmonth_model = 201602
    time_names = ['weekend_ind', 'week_bf_Christ', 'week_aftxg', 'days_aftchr',
                  'days_bflbd', 'days_aftny']
    easter_days = ['2012-04-08', '2013-03-31', '2014-04-20', '2015-04-05', '2016-03-27', '2017-04-16', '2018-04-01']
    christmas_days = ['2012-12-25', '2013-12-25', '2014-12-25', '2015-12-25', '2016-12-25', '2017-12-25', '2018-12-25']
    thanksgiving_days = ['2012-11-22', '2013-11-28', '2014-11-27', '2015-11-26', '2016-11-24', '2017-11-23', '2018-11-22']
    exclude_vars_kc = ['ElectionDay', 'GreenMonday', 'StPatricksDay', 'MothersDay', 'YomKippur',
                       'Halloween', 'FF_Limited', 'E23_Sale', 'E23_PAD', 'E23_BMSM', 'E23_Booster',
                       'E23_KCSP', 'E23_Flash', 'E23_GPO', 'E23_AssociateShop', 'D23_OnlineOnly',
                       'D23_KCOnly', 'D23_GPO', 'D23_Limited']

    holiday = pd.read_csv('../Willis/StoreWillisFinAdV1.0/store_sales_withPO/holiday_withPO.csv')
    calendar = pd.read_csv('../Willis/StoreWillisFinAdV1.0/store_sales_withPO/calendar_withPO.csv')
    # Get rid of bad column when saved into csv.
    holiday.drop('Unnamed: 0', axis=1, inplace=True)
    calendar.drop('Unnamed: 0', axis=1, inplace=True)

    for store_id in stores_list:
        sales = pd.read_csv('../Willis/StoreWillisFinAdV1.0/store_sales_withPO/sales_store_withPO_'
                            + str(store_id) + '.csv')
        # Get rid of bad column when saved into csv.
        sales.drop('Unnamed: 0', axis=1, inplace=True)

        xgb_sales_predict(sales, holiday, calendar, time_names, store_id, 'rf')

        del sales
        print('Done for ' + str(store_id))