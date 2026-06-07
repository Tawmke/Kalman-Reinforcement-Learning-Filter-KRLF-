# import library
import numpy as np
from math import radians, cos, sin, asin, sqrt
from haversine import haversine
import pickle
import pandas as pd
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
# 定位到 src 目录
# 因为 env 和 src 同级，所以是先返回上一级再进入 src
src_path = os.path.join(current_dir, "..")
sys.path.append(src_path)
# 组合导航的函数
from common.angle import Angle
from common.types import IMU, GNSS
from common.funcs import *
import yaml

def exp_average(data, expFactor=0.1):
    expRawRewards = np.zeros(data.shape)
    for i in range(data.shape[0]):
        expRaw = 0.0
        J = 0.0
        for j in range(data.shape[1]):
            J *= (1.0-expFactor)
            J += (expFactor)
            rate = expFactor/J
            expRaw = (1-rate)*expRaw
            expRaw += rate*data[i][j]
            expRawRewards[i, j] = expRaw
    return expRawRewards

def exp_average_list(data, expFactor=0.1):
    expRawRewards = np.zeros(len(data))
    expRaw = 0.0
    J = 0.0
    for j in range(len(data)):
        J *= (1.0-expFactor)
        J += (expFactor)
        rate = expFactor/J
        expRaw = (1-rate)*expRaw
        expRaw += rate*data[j]
        expRawRewards[j] = expRaw
    return expRawRewards

def geodistance(lng1,lat1,lng2,lat2):
    lng1, lat1, lng2, lat2 = map(radians, [float(lng1), float(lat1), float(lng2), float(lat2)]) # 经纬度转换成弧度
    dlon=lng2-lng1
    dlat=lat2-lat1
    a=sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    distance=2*asin(sqrt(a))*6371*1000 # 地球平均半径，6371km
    distance=round(distance/1000,3)
    return distance

def cal_distance(row):
    """
    计算两个经纬度点之间的距离
    """
    long1 = row['LongitudeDegrees_truth']
    lat1 = row['LatitudeDegrees_truth']
    long2 = row['lngDeg_RLpredict']
    lat2 = row['latDeg_RLpredict']
    long3 = row['LongitudeDegrees']
    lat3 = row['LatitudeDegrees']
    g1 = (lat1,long1)
    g2 = (lat2,long2)
    g3 = (lat3,long3)
    # g1 = (long1, lat1)
    # g2 = (long2, lat2)
    # g3 = (long3, lat3)
    ret1 = haversine(g1, g2, unit='m')
    ret2 = haversine(g1, g3, unit='m')
    result1 = "%.7f" % ret1
    result2 = "%.7f" % ret2
    return result1, result2

def cal_distance_ecef_RLKF(row,baseline_mod,blh_station):
    """
    计算两个经纬度点之间的距离
    """
    gt_lat = row[' Latitude_GT (deg)']* Angle.D2R
    gt_lon = row[' Longitude_GT (deg)']* Angle.D2R
    gt_h = row[' Ellipsoid Height_GT (m)']
    rl_lat = row['Latitude_RLpredict'] * Angle.D2R
    rl_lon = row['Longitude_RLpredict'] * Angle.D2R
    rl_h = row['Ellipsoid_Height_RLpredict']
    if baseline_mod == 'GNSS/INS':
        bl_lat = row[' Latitude (deg)']* Angle.D2R
        bl_lon = row[' Longitude (deg)']* Angle.D2R
        bl_h = row[' Ellipsoid Height (m)']

    # 计算bl误差
    rm, rn = radiusmn(blh_station[0])
    bl_pos_err = np.array([bl_lat - gt_lat, bl_lon - gt_lon, bl_h - gt_h])
    bl_pos_err = drad2dm(rm, rn, blh_station, bl_pos_err).reshape(-1)
    # 计算rl误差
    rl_pos_err = np.array([rl_lat - gt_lat, rl_lon - gt_lon, rl_h - gt_h])
    rl_pos_err = drad2dm(rm, rn, blh_station, rl_pos_err).reshape(-1)

    mse_rl_p = np.sqrt(np.sum(rl_pos_err ** 2))
    mse_bl_p = np.sqrt(np.sum(bl_pos_err ** 2))

    ll_err_rl = np.sqrt(np.sum(rl_pos_err[0:2] ** 2))
    ll_err_bl = np.sqrt(np.sum(bl_pos_err[0:2] ** 2))
    h_err_rl = np.abs(rl_pos_err[2])
    h_err_bl = np.abs(bl_pos_err[2])

    return mse_rl_p, mse_bl_p, rl_pos_err[0], rl_pos_err[1], rl_pos_err[2], bl_pos_err[0], bl_pos_err[1], bl_pos_err[2], \
           ll_err_rl, h_err_rl, ll_err_bl, h_err_bl

def cal_distance_vel_RLKF(row,baseline_mod):
    """
    计算两个经纬度点之间的距离
    """
    gt_x = row[' Velocity Y_GT (m/s)'] # 参考速度是enu的
    gt_y = row[' Velocity X_GT (m/s)']
    gt_z = -row[' Velocity Z_GT (m/s)']
    rl_x = row['Velocity_X_RLpredict']
    rl_y = row['Velocity_Y_RLpredict']
    rl_z = row['Velocity_Z_RLpredict']
    if baseline_mod == 'GNSS/INS':
        bl_x = row[' Velocity X (m/s)']
        bl_y = row[' Velocity Y (m/s)']
        bl_z = row[' Velocity Z (m/s)']

    # 计算bl误差
    bl_vel_err = np.array([bl_x - gt_x, bl_y - gt_y, bl_z - gt_z])
    # 计算rl误差
    rl_vel_err = np.array([rl_x - gt_x, rl_y - gt_y, rl_z - gt_z])

    mse_rl_v = np.sqrt(np.sum(rl_vel_err ** 2))
    mse_bl_v = np.sqrt(np.sum(bl_vel_err ** 2))

    return mse_rl_v, mse_bl_v, rl_vel_err[0], rl_vel_err[1], rl_vel_err[2], bl_vel_err[0], bl_vel_err[1], bl_vel_err[2]

def cal_distance_att_RLKF(row,baseline_mod):
    """
    计算两个经纬度点之间的距离
    """
    gt_r = row[' Roll_GT (deg)'] # 参考速度是enu的
    gt_p = row[' Pitch_GT (deg)']
    gt_h = row[' Heading_GT (deg)']
    rl_r = row['Roll_RLpredict']
    rl_p = row['Pitch_RLpredict']
    rl_h = row['Heading_RLpredict']
    if baseline_mod == 'GNSS/INS':
        bl_r = row[' Roll (deg)']
        bl_p = row[' Pitch (deg)']
        bl_h = row[' Heading (deg)']

    # 计算bl误差
    bl_att_err = np.array([bl_r - gt_r, bl_p - gt_p, bl_h - gt_h])
    if bl_att_err[2] > 180:
        bl_att_err[2] -= 360
    if bl_att_err[2] < -180:
        bl_att_err[2] += 360
    # 计算rl误差
    rl_att_err = np.array([rl_r - gt_r, rl_p - gt_p, rl_h - gt_h])
    if rl_att_err[2] > 180:
        rl_att_err[2] -= 360
    if rl_att_err[2] < -180:
        rl_att_err[2] += 360

    mse_rl_v = np.sqrt(np.sum(rl_att_err ** 2))
    mse_bl_v = np.sqrt(np.sum(bl_att_err ** 2))

    return mse_rl_v, mse_bl_v, rl_att_err[0], rl_att_err[1], rl_att_err[2], bl_att_err[0], bl_att_err[1], bl_att_err[2]

def calc_haversine(lat1, lon1, lat2, lon2):
    """Calculates the great circle distance between two points
    on the earth. Inputs are array-like and specified in decimal degrees.
    """
    RADIUS = 6_367_000
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + \
        np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    dist = 2 * RADIUS * np.arcsin(a**0.5)
    return dist

def percentile50(x):
    return np.percentile(x, 50)
def percentile95(x):
    return np.percentile(x, 95)

def get_train_score(df, gt):
    gt = gt.rename(columns={'latDeg':'latDeg_gt', 'lngDeg':'lngDeg_gt'})
    df = df.merge(gt, on=['collectionName', 'phoneName', 'millisSinceGpsEpoch'], how='inner')
    # calc_distance_error
    df['err'] = calc_haversine(df['latDeg_gt'], df['lngDeg_gt'], df['latDeg'], df['lngDeg'])
    # calc_evaluate_score
    df['phone'] = df['collectionName'] + '_' + df['phoneName']
    res = df.groupby('phone')['err'].agg([percentile50, percentile95])
    res['p50_p90_mean'] = (res['percentile50'] + res['percentile95']) / 2
    score = res['p50_p90_mean'].mean()
    return score

def recording_results_ecef_RL4KF(dir_path, data_truth_dic,trajdata_range,tripIDlist,logdirname,baseline_mod,traj_record):
    error_mean_all = 0
    rl_distances_mean_all = 0
    rl_RMSE_all, rl_ll_RMSE_all, rl_h_RMSE_all = 0,0,0
    or_distances_mean_all = 0
    or_RMSE_all, or_ll_RMSE_all, or_h_RMSE_all = 0,0,0
    error_std_all = 0
    rl_distances_std_all = 0
    or_distances_std_all = 0
    rl_ll_mean_all = 0
    or_ll_mean_all = 0
    pd_gen=False
    for train_tripIDnum in range(trajdata_range[0], trajdata_range[1] + 1):
        try:
            pd_train = data_truth_dic[tripIDlist[train_tripIDnum]]
            index_nonull = pd_train['Latitude_RLpredict'].notnull()
            pd_train = pd_train[index_nonull]
            # 提取整数历元的数据
            # 初始化准备导航状态
            config_filename = os.path.abspath(
                f'{dir_path}/dataset_Urbannav/{tripIDlist[train_tripIDnum]}/kf-gins.yaml')
            with open(config_filename, 'r', encoding='utf-8') as f:
                traj_config = yaml.load(f, Loader=yaml.FullLoader)  # 从YAML文件(filename)中加载轨迹配置数据

            step = int(traj_config["imudatarate"]/ traj_config["gnssrate"])
            pd_train = pd_train.iloc[::step]
            pd_train.reset_index(drop=True, inplace=True)

            if traj_record:
                # record rl traj
                if baseline_mod == 'GNSS/INS':
                    record_columns=['UnixTimeMillis_ref', ' Latitude (deg)', ' Longitude (deg)', ' Ellipsoid Height (m)',
                  ' Velocity X (m/s)', ' Velocity Y (m/s)', ' Velocity Z (m/s)',
                  ' Roll (deg)', ' Pitch (deg)', ' Heading (deg)',' Latitude_GT (deg)', ' Longitude_GT (deg)', ' Ellipsoid Height_GT (m)',
                  ' Velocity X_GT (m/s)', ' Velocity Y_GT (m/s)', ' Velocity Z_GT (m/s)',
                  ' Roll_GT (deg)', ' Pitch_GT (deg)', ' Heading_GT (deg)', 'Latitude_RLpredict', 'Longitude_RLpredict', 'Ellipsoid_Height_RLpredict',
                  'Velocity_X_RLpredict', 'Velocity_Y_RLpredict', 'Velocity_Z_RLpredict',
                  'Roll_RLpredict', 'Pitch_RLpredict', 'Heading_RLpredict']
                    record_columns_pos = ['UnixTimeMillis_ref', ' Latitude (deg)', ' Longitude (deg)', ' Ellipsoid Height (m)',
                                          ' Latitude_GT (deg)', ' Longitude_GT (deg)', ' Ellipsoid Height_GT (m)',
                                          'Latitude_RLpredict', 'Longitude_RLpredict', 'Ellipsoid_Height_RLpredict']
                try:
                    pd_record = pd_train[record_columns]
                except:
                    pd_record = pd_train[record_columns_pos]
                pd_record.to_csv(logdirname + f'rl_traj_{tripIDlist[train_tripIDnum].replace("/","_")}.csv', index=True)

            if baseline_mod == 'GNSS/INS':
                test = pd_train.loc[:, [' Latitude (deg)', ' Longitude (deg)', ' Ellipsoid Height (m)',
                                        ' Latitude_GT (deg)', ' Longitude_GT (deg)', ' Ellipsoid Height_GT (m)',
                                        'Latitude_RLpredict', 'Longitude_RLpredict', 'Ellipsoid_Height_RLpredict']]

            blh_station = test.iloc[0,0:3].values # 转导航坐标系初始点
            test['rl_distance'] = test.apply(lambda test: cal_distance_ecef_RLKF(test,baseline_mod,blh_station)[0], axis=1)
            test['or_distance'] = test.apply(lambda test: cal_distance_ecef_RLKF(test,baseline_mod,blh_station)[1], axis=1)
            test['error'] = test['rl_distance'].astype(float) - test['or_distance'].astype(float)
            test['count_rl_distance'] = test['rl_distance'].astype(float)
            test['count_or_distance'] = test['or_distance'].astype(float)
            test['rl_xdistance'] = test.apply(lambda test: cal_distance_ecef_RLKF(test,baseline_mod,blh_station)[2], axis=1)
            test['rl_ydistance'] = test.apply(lambda test: cal_distance_ecef_RLKF(test,baseline_mod,blh_station)[3], axis=1)
            test['rl_zdistance'] = test.apply(lambda test: cal_distance_ecef_RLKF(test,baseline_mod,blh_station)[4], axis=1)
            test['count_rl_xdistance'] = test['rl_xdistance'].astype(float)
            test['count_rl_ydistance'] = test['rl_ydistance'].astype(float)
            test['count_rl_zdistance'] = test['rl_zdistance'].astype(float)
            test['or_xdistance'] = test.apply(lambda test: cal_distance_ecef_RLKF(test,baseline_mod,blh_station)[5], axis=1)
            test['or_ydistance'] = test.apply(lambda test: cal_distance_ecef_RLKF(test,baseline_mod,blh_station)[6], axis=1)
            test['or_zdistance'] = test.apply(lambda test: cal_distance_ecef_RLKF(test,baseline_mod,blh_station)[7], axis=1)
            test['count_or_xdistance'] = test['or_xdistance'].astype(float)
            test['count_or_ydistance'] = test['or_ydistance'].astype(float)
            test['count_or_zdistance'] = test['or_zdistance'].astype(float)
            rl_lldistance = test.apply(lambda test: cal_distance_ecef_RLKF(test, baseline_mod,blh_station)[8], axis=1)
            test['rl_lldistance'] = rl_lldistance
            test['rl_hdistance'] = test.apply(lambda test: cal_distance_ecef_RLKF(test, baseline_mod,blh_station)[9], axis=1)
            test['count_rl_lldistance'] = test['rl_lldistance'].astype(float)
            test['count_rl_hdistance'] = test['rl_hdistance'].astype(float)
            or_lldistance = test.apply(lambda test: cal_distance_ecef_RLKF(test, baseline_mod,blh_station)[10], axis=1)
            test['or_lldistance'] = or_lldistance
            test['or_hdistance'] = test.apply(lambda test: cal_distance_ecef_RLKF(test, baseline_mod,blh_station)[11], axis=1)
            test['count_or_lldistance'] = test['or_lldistance'].astype(float)
            test['count_or_hdistance'] = test['or_hdistance'].astype(float)

            if pd_gen:
                error_pd.insert(error_pd.shape[1], f'{train_tripIDnum}', test['error'].describe())
                rl_distance_pd.insert(rl_distance_pd.shape[1], f'{train_tripIDnum}',
                                      test['count_rl_distance'].describe())
                or_distance_pd.insert(or_distance_pd.shape[1], f'{train_tripIDnum}',
                                      test['count_or_distance'].describe())
                tmp_dic = {'tripID':tripIDlist[train_tripIDnum], 'rl_xdistance_mean': np.mean(test['rl_xdistance']), 'rl_ydistance_mean': np.mean(test['rl_ydistance']),'rl_zdistance_mean': np.mean(test['rl_zdistance']),
                                                'rl_xdistance_std': np.std(test['rl_xdistance']),'rl_ydistance_std': np.std(test['rl_ydistance']), 'rl_zdistance_std': np.std(test['rl_zdistance']),
                                                'rl_xdistance_min': min(test['rl_xdistance']),'rl_ydistance_min': np.nanmin(test['rl_ydistance']), 'rl_zdistance_min': np.nanmin(test['rl_zdistance']),
                                                'rl_xdistance_max': np.nanmax(test['rl_xdistance']),'rl_ydistance_max': np.nanmax(test['rl_ydistance']), 'rl_zdistance_max': np.nanmax(test['rl_zdistance']),
                                                'or_xdistance_mean': np.mean(test['or_xdistance']), 'or_ydistance_mean': np.mean(test['or_ydistance']),'or_zdistance_mean': np.mean(test['or_zdistance']),
                                                'or_xdistance_std': np.std(test['or_xdistance']),'or_ydistance_std': np.std(test['or_ydistance']), 'or_zdistance_std': np.std(test['or_zdistance']),
                                                'or_xdistance_min': np.nanmin(test['or_xdistance']),'or_ydistance_min': np.nanmin(test['or_ydistance']), 'or_zdistance_min': np.nanmin(test['or_zdistance']),
                                                'or_xdistance_max': np.nanmax(test['or_xdistance']),'or_ydistance_max': np.nanmax(test['or_ydistance']), 'or_zdistance_max': np.nanmax(test['or_zdistance']),
                           'rl_llerr_mean': np.mean(test['rl_lldistance']),'rl_llerr_std': np.std(test['rl_lldistance']),'rl_llerr_min': np.nanmin(test['rl_lldistance']),'rl_llerr_max': np.nanmax(test['rl_lldistance']),
                            'rl_herr_mean': np.mean(test['rl_hdistance']),'rl_herr_std': np.std(test['rl_hdistance']),'rl_herr_min': np.nanmin(test['rl_hdistance']),'rl_herr_max': np.nanmax(test['rl_hdistance']),
                            'or_llerr_mean': np.mean(test['or_lldistance']),'or_llerr_std': np.std(test['or_lldistance']),'or_llerr_min': np.nanmin(test['or_lldistance']),'or_llerr_max': np.nanmax(test['or_lldistance']),
                            'or_herr_mean': np.mean(test['or_hdistance']),'or_herr_std': np.std(test['or_hdistance']),'or_herr_min': np.nanmin(test['or_hdistance']),'or_herr_max': np.nanmax(test['or_hdistance'])
                            }

                xyz_distance_pd.insert(xyz_distance_pd.shape[1], f'{train_tripIDnum}',
                                       pd.DataFrame.from_dict(tmp_dic, orient='index').loc[:, 0])
            else:
                error_pd = pd.DataFrame(test['error'].describe())
                error_pd = error_pd.rename(columns={'error': f'{train_tripIDnum}'})
                error_pd.index.name = 'errors'
                rl_distance_pd = pd.DataFrame(test['count_rl_distance'].describe())
                rl_distance_pd = rl_distance_pd.rename(columns={'count_rl_distance': f'{train_tripIDnum}'})
                rl_distance_pd.index.name = 'rl_distances'
                or_distance_pd = pd.DataFrame(test['count_or_distance'].describe())
                or_distance_pd = or_distance_pd.rename(columns={'count_or_distance': f'{train_tripIDnum}'})
                or_distance_pd.index.name = 'or_distances'

                tmp_dic = {'tripID':tripIDlist[train_tripIDnum], 'rl_xdistance_mean': np.mean(test['rl_xdistance']), 'rl_ydistance_mean': np.mean(test['rl_ydistance']),'rl_zdistance_mean': np.mean(test['rl_zdistance']),
                                                'rl_xdistance_std': np.std(test['rl_xdistance']),'rl_ydistance_std': np.std(test['rl_ydistance']), 'rl_zdistance_std': np.std(test['rl_zdistance']),
                                                'rl_xdistance_min': np.nanmin(test['rl_xdistance']),'rl_ydistance_min': np.nanmin(test['rl_ydistance']), 'rl_zdistance_min': np.nanmin(test['rl_zdistance']),
                                                'rl_xdistance_max': np.nanmax(test['rl_xdistance']),'rl_ydistance_max': np.nanmax(test['rl_ydistance']), 'rl_zdistance_max': np.nanmax(test['rl_zdistance']),
                                                'or_xdistance_mean': np.mean(test['or_xdistance']), 'or_ydistance_mean': np.mean(test['or_ydistance']),'or_zdistance_mean': np.mean(test['or_zdistance']),
                                                'or_xdistance_std': np.std(test['or_xdistance']),'or_ydistance_std': np.std(test['or_ydistance']), 'or_zdistance_std': np.std(test['or_zdistance']),
                                                'or_xdistance_min': np.nanmin(test['or_xdistance']),'or_ydistance_min': np.nanmin(test['or_ydistance']), 'or_zdistance_min': np.nanmin(test['or_zdistance']),
                                                'or_xdistance_max': np.nanmax(test['or_xdistance']),'or_ydistance_max': np.nanmax(test['or_ydistance']), 'or_zdistance_max': np.nanmax(test['or_zdistance']),
                           'rl_llerr_mean': np.mean(test['rl_lldistance']),'rl_llerr_std': np.std(test['rl_lldistance']),'rl_llerr_min': np.nanmin(test['rl_lldistance']),'rl_llerr_max': np.nanmax(test['rl_lldistance']),
                            'rl_herr_mean': np.mean(test['rl_hdistance']),'rl_herr_std': np.std(test['rl_hdistance']),'rl_herr_min': np.nanmin(test['rl_hdistance']),'rl_herr_max': np.nanmax(test['rl_hdistance']),
                            'or_llerr_mean': np.mean(test['or_lldistance']),'or_llerr_std': np.std(test['or_lldistance']),'or_llerr_min': np.nanmin(test['or_lldistance']),'or_llerr_max': np.nanmax(test['or_lldistance']),
                            'or_herr_mean': np.mean(test['or_hdistance']),'or_herr_std': np.std(test['or_hdistance']),'or_herr_min': np.nanmin(test['or_hdistance']),'or_herr_max': np.nanmax(test['or_hdistance'])
                           }
                xyz_distance_pd=pd.DataFrame.from_dict(tmp_dic, orient='index')
                pd_gen=True
            error_mean_all += test['error'].describe()['count'] * test['error'].describe()['mean']
            rl_distances_mean_all += test['count_rl_distance'].describe()['count'] * test['count_rl_distance'].describe()['mean']
            rl_RMSE_all += np.sum(test['count_rl_distance'] **2) # 统计 RMSE 结果
            rl_ll_RMSE_all += np.sum(test['rl_lldistance'] **2)
            rl_h_RMSE_all += np.sum(test['rl_hdistance'] **2)
            or_distances_mean_all += test['count_or_distance'].describe()['count'] * test['count_or_distance'].describe()['mean']
            or_RMSE_all += np.sum(test['count_or_distance'] **2)
            or_ll_RMSE_all += np.sum(test['or_lldistance'] **2)
            or_h_RMSE_all += np.sum(test['or_hdistance'] **2)
            error_std_all += test['error'].describe()['count'] * test['error'].describe()['std']
            rl_distances_std_all += test['count_rl_distance'].describe()['count'] * test['count_rl_distance'].describe()['std']
            or_distances_std_all += test['count_or_distance'].describe()['count'] * test['count_or_distance'].describe()['std']
            rl_ll_mean_all += test['rl_lldistance'].describe()['count'] * test['rl_lldistance'].describe()['mean']
            or_ll_mean_all += test['or_lldistance'].describe()['count'] * test['or_lldistance'].describe()['mean']

        except Exception as e:
            print(f'Trajectory {train_tripIDnum} error:{e}.')

    num_total_err = np.sum(error_pd.loc['count', :])
    num_total_rl = np.sum(rl_distance_pd.loc['count', :])
    num_total_or = np.sum(or_distance_pd.loc['count', :])
    error_min = np.min(error_pd.loc['min', :])
    error_max = np.max(error_pd.loc['max', :])
    # 计算总的 RMSE 误差
    rl_RMSE, rl_ll_RMSE, rl_h_RMSE = np.sqrt(rl_RMSE_all/num_total_err), np.sqrt(rl_ll_RMSE_all/num_total_err), np.sqrt(rl_h_RMSE_all/num_total_err)
    or_RMSE, or_ll_RMSE, or_h_RMSE = np.sqrt(or_RMSE_all/num_total_err), np.sqrt(or_ll_RMSE_all/num_total_err), np.sqrt(or_h_RMSE_all/num_total_err)
    avg_xyz_err = rl_distances_mean_all / num_total_rl
    avg_rl_llerr = rl_ll_mean_all / num_total_rl
    avg_or_llerr = or_ll_mean_all / num_total_rl
    # 保存总体数据
    error_pd.insert(error_pd.shape[1], 'Avg', [num_total_err, error_mean_all / num_total_err, error_std_all / num_total_err,
                                               error_min, 0, 0, 0, error_max])
    rl_distance_pd.insert(rl_distance_pd.shape[1], 'Avg',
                          [num_total_rl, rl_distances_mean_all / num_total_rl, rl_distances_std_all / num_total_rl,
                           np.min(rl_distance_pd.loc['min', :]), 0, 0, 0, np.max(rl_distance_pd.loc['max', :])])
    or_distance_pd.insert(or_distance_pd.shape[1], 'Avg',
                          [num_total_or, or_distances_mean_all / num_total_or, or_distances_std_all / num_total_or,
                           np.min(or_distance_pd.loc['min', :]), 0, 0, 0, np.max(or_distance_pd.loc['max', :])])
    error_pd.to_csv(logdirname + 'errors.csv', index=True)
    rl_distance_pd.to_csv(logdirname + 'rl_distances.csv', index=True)
    or_distance_pd.to_csv(logdirname + 'or_distances.csv', index=True)
    xyz_distance_pd.to_csv(logdirname + 'xyz_distances.csv', index=True)
    # 保存结果
    file_name = logdirname + f'Result_pos ratio:{(or_RMSE-rl_RMSE)/or_RMSE:.2f}_rl RMSE:{rl_RMSE:.2f}_or RMSE:{or_RMSE:.2f}.txt'
    with open(file_name, "w", encoding="utf-8") as file:
        file.write(f'Ratio:{(or_RMSE-rl_RMSE)/or_RMSE:.2f}, rl RMSE:{rl_RMSE:.2f}, or RMSE:{or_RMSE:.2f}, rl ll RMSE:{rl_ll_RMSE:.2f},'
                   f'or ll RMSE:{or_ll_RMSE:.2f}, rl h RMSE:{rl_h_RMSE:.2f}, or h RMSE:{or_h_RMSE:.2f}')

    print(
        f'Perfermances: count {num_total_err:1.0f}, compared with baseline mean: {error_mean_all / num_total_err:4.3f}+{error_std_all / num_total_err:4.3f}m, '
        f'min: {error_min:4.3f}m, max: {error_max:4.3f}m.')

    return (or_RMSE-rl_RMSE)/or_RMSE, rl_RMSE

def recording_results_ecef_RL4KF_SZ(dir_path, data_truth_dic,trajdata_range,tripIDlist,logdirname,baseline_mod,traj_record):
    error_mean_all = 0
    rl_distances_mean_all = 0
    rl_RMSE_all, rl_ll_RMSE_all, rl_h_RMSE_all = 0,0,0
    or_distances_mean_all = 0
    or_RMSE_all, or_ll_RMSE_all, or_h_RMSE_all = 0,0,0
    error_std_all = 0
    rl_distances_std_all = 0
    or_distances_std_all = 0
    rl_ll_mean_all = 0
    or_ll_mean_all = 0
    pd_gen=False
    for train_tripIDnum in range(trajdata_range[0], trajdata_range[1] + 1):
        try:
            pd_train = data_truth_dic[tripIDlist[train_tripIDnum]]
            index_nonull = pd_train['Latitude_RLpredict'].notnull()
            pd_train = pd_train[index_nonull]
            # 提取整数历元的数据
            # 初始化准备导航状态
            config_filename = os.path.abspath(
                f'{dir_path}/dataset_SZ/{tripIDlist[train_tripIDnum][:25]}/kf-gins.yaml')
            with open(config_filename, 'r', encoding='utf-8') as f:
                traj_config = yaml.load(f, Loader=yaml.FullLoader)  # 从YAML文件(filename)中加载轨迹配置数据

            step = int(traj_config["imudatarate"]/ traj_config["gnssrate"])
            pd_train = pd_train.iloc[::step]
            pd_train.reset_index(drop=True, inplace=True)

            if traj_record:
                # record rl traj
                if baseline_mod == 'GNSS/INS':
                    record_columns_pos = ['UnixTimeMillis_ref', ' Latitude (deg)', ' Longitude (deg)', ' Ellipsoid Height (m)',
                                          ' Latitude_GT (deg)', ' Longitude_GT (deg)', ' Ellipsoid Height_GT (m)',
                                          'Latitude_RLpredict', 'Longitude_RLpredict', 'Ellipsoid_Height_RLpredict']

                pd_record = pd_train[record_columns_pos]
                pd_record.to_csv(logdirname + f'rl_traj_{tripIDlist[train_tripIDnum].replace("/","_")}.csv', index=True)

            if baseline_mod == 'GNSS/INS':
                test = pd_train.loc[:, [' Latitude (deg)', ' Longitude (deg)', ' Ellipsoid Height (m)',
                                        ' Latitude_GT (deg)', ' Longitude_GT (deg)', ' Ellipsoid Height_GT (m)',
                                        'Latitude_RLpredict', 'Longitude_RLpredict', 'Ellipsoid_Height_RLpredict']]

            blh_station = test.iloc[0,0:3].values # 转导航坐标系初始点
            test['rl_distance'] = test.apply(lambda test: cal_distance_ecef_RLKF(test,baseline_mod,blh_station)[0], axis=1)
            test['or_distance'] = test.apply(lambda test: cal_distance_ecef_RLKF(test,baseline_mod,blh_station)[1], axis=1)
            test['error'] = test['rl_distance'].astype(float) - test['or_distance'].astype(float)
            test['count_rl_distance'] = test['rl_distance'].astype(float)
            test['count_or_distance'] = test['or_distance'].astype(float)
            test['rl_xdistance'] = test.apply(lambda test: cal_distance_ecef_RLKF(test,baseline_mod,blh_station)[2], axis=1)
            test['rl_ydistance'] = test.apply(lambda test: cal_distance_ecef_RLKF(test,baseline_mod,blh_station)[3], axis=1)
            test['rl_zdistance'] = test.apply(lambda test: cal_distance_ecef_RLKF(test,baseline_mod,blh_station)[4], axis=1)
            test['count_rl_xdistance'] = test['rl_xdistance'].astype(float)
            test['count_rl_ydistance'] = test['rl_ydistance'].astype(float)
            test['count_rl_zdistance'] = test['rl_zdistance'].astype(float)
            test['or_xdistance'] = test.apply(lambda test: cal_distance_ecef_RLKF(test,baseline_mod,blh_station)[5], axis=1)
            test['or_ydistance'] = test.apply(lambda test: cal_distance_ecef_RLKF(test,baseline_mod,blh_station)[6], axis=1)
            test['or_zdistance'] = test.apply(lambda test: cal_distance_ecef_RLKF(test,baseline_mod,blh_station)[7], axis=1)
            test['count_or_xdistance'] = test['or_xdistance'].astype(float)
            test['count_or_ydistance'] = test['or_ydistance'].astype(float)
            test['count_or_zdistance'] = test['or_zdistance'].astype(float)
            rl_lldistance = test.apply(lambda test: cal_distance_ecef_RLKF(test, baseline_mod,blh_station)[8], axis=1)
            test['rl_lldistance'] = rl_lldistance
            test['rl_hdistance'] = test.apply(lambda test: cal_distance_ecef_RLKF(test, baseline_mod,blh_station)[9], axis=1)
            test['count_rl_lldistance'] = test['rl_lldistance'].astype(float)
            test['count_rl_hdistance'] = test['rl_hdistance'].astype(float)
            or_lldistance = test.apply(lambda test: cal_distance_ecef_RLKF(test, baseline_mod,blh_station)[10], axis=1)
            test['or_lldistance'] = or_lldistance
            test['or_hdistance'] = test.apply(lambda test: cal_distance_ecef_RLKF(test, baseline_mod,blh_station)[11], axis=1)
            test['count_or_lldistance'] = test['or_lldistance'].astype(float)
            test['count_or_hdistance'] = test['or_hdistance'].astype(float)

            if pd_gen:
                error_pd.insert(error_pd.shape[1], f'{train_tripIDnum}', test['error'].describe())
                rl_distance_pd.insert(rl_distance_pd.shape[1], f'{train_tripIDnum}',
                                      test['count_rl_distance'].describe())
                or_distance_pd.insert(or_distance_pd.shape[1], f'{train_tripIDnum}',
                                      test['count_or_distance'].describe())
                tmp_dic = {'tripID':tripIDlist[train_tripIDnum], 'rl_xdistance_mean': np.mean(test['rl_xdistance']), 'rl_ydistance_mean': np.mean(test['rl_ydistance']),'rl_zdistance_mean': np.mean(test['rl_zdistance']),
                                                'rl_xdistance_std': np.std(test['rl_xdistance']),'rl_ydistance_std': np.std(test['rl_ydistance']), 'rl_zdistance_std': np.std(test['rl_zdistance']),
                                                'rl_xdistance_min': min(test['rl_xdistance']),'rl_ydistance_min': np.nanmin(test['rl_ydistance']), 'rl_zdistance_min': np.nanmin(test['rl_zdistance']),
                                                'rl_xdistance_max': np.nanmax(test['rl_xdistance']),'rl_ydistance_max': np.nanmax(test['rl_ydistance']), 'rl_zdistance_max': np.nanmax(test['rl_zdistance']),
                                                'or_xdistance_mean': np.mean(test['or_xdistance']), 'or_ydistance_mean': np.mean(test['or_ydistance']),'or_zdistance_mean': np.mean(test['or_zdistance']),
                                                'or_xdistance_std': np.std(test['or_xdistance']),'or_ydistance_std': np.std(test['or_ydistance']), 'or_zdistance_std': np.std(test['or_zdistance']),
                                                'or_xdistance_min': np.nanmin(test['or_xdistance']),'or_ydistance_min': np.nanmin(test['or_ydistance']), 'or_zdistance_min': np.nanmin(test['or_zdistance']),
                                                'or_xdistance_max': np.nanmax(test['or_xdistance']),'or_ydistance_max': np.nanmax(test['or_ydistance']), 'or_zdistance_max': np.nanmax(test['or_zdistance']),
                           'rl_llerr_mean': np.mean(test['rl_lldistance']),'rl_llerr_std': np.std(test['rl_lldistance']),'rl_llerr_min': np.nanmin(test['rl_lldistance']),'rl_llerr_max': np.nanmax(test['rl_lldistance']),
                            'rl_herr_mean': np.mean(test['rl_hdistance']),'rl_herr_std': np.std(test['rl_hdistance']),'rl_herr_min': np.nanmin(test['rl_hdistance']),'rl_herr_max': np.nanmax(test['rl_hdistance']),
                            'or_llerr_mean': np.mean(test['or_lldistance']),'or_llerr_std': np.std(test['or_lldistance']),'or_llerr_min': np.nanmin(test['or_lldistance']),'or_llerr_max': np.nanmax(test['or_lldistance']),
                            'or_herr_mean': np.mean(test['or_hdistance']),'or_herr_std': np.std(test['or_hdistance']),'or_herr_min': np.nanmin(test['or_hdistance']),'or_herr_max': np.nanmax(test['or_hdistance'])
                            }

                xyz_distance_pd.insert(xyz_distance_pd.shape[1], f'{train_tripIDnum}',
                                       pd.DataFrame.from_dict(tmp_dic, orient='index').loc[:, 0])
            else:
                error_pd = pd.DataFrame(test['error'].describe())
                error_pd = error_pd.rename(columns={'error': f'{train_tripIDnum}'})
                error_pd.index.name = 'errors'
                rl_distance_pd = pd.DataFrame(test['count_rl_distance'].describe())
                rl_distance_pd = rl_distance_pd.rename(columns={'count_rl_distance': f'{train_tripIDnum}'})
                rl_distance_pd.index.name = 'rl_distances'
                or_distance_pd = pd.DataFrame(test['count_or_distance'].describe())
                or_distance_pd = or_distance_pd.rename(columns={'count_or_distance': f'{train_tripIDnum}'})
                or_distance_pd.index.name = 'or_distances'

                tmp_dic = {'tripID':tripIDlist[train_tripIDnum], 'rl_xdistance_mean': np.mean(test['rl_xdistance']), 'rl_ydistance_mean': np.mean(test['rl_ydistance']),'rl_zdistance_mean': np.mean(test['rl_zdistance']),
                                                'rl_xdistance_std': np.std(test['rl_xdistance']),'rl_ydistance_std': np.std(test['rl_ydistance']), 'rl_zdistance_std': np.std(test['rl_zdistance']),
                                                'rl_xdistance_min': np.nanmin(test['rl_xdistance']),'rl_ydistance_min': np.nanmin(test['rl_ydistance']), 'rl_zdistance_min': np.nanmin(test['rl_zdistance']),
                                                'rl_xdistance_max': np.nanmax(test['rl_xdistance']),'rl_ydistance_max': np.nanmax(test['rl_ydistance']), 'rl_zdistance_max': np.nanmax(test['rl_zdistance']),
                                                'or_xdistance_mean': np.mean(test['or_xdistance']), 'or_ydistance_mean': np.mean(test['or_ydistance']),'or_zdistance_mean': np.mean(test['or_zdistance']),
                                                'or_xdistance_std': np.std(test['or_xdistance']),'or_ydistance_std': np.std(test['or_ydistance']), 'or_zdistance_std': np.std(test['or_zdistance']),
                                                'or_xdistance_min': np.nanmin(test['or_xdistance']),'or_ydistance_min': np.nanmin(test['or_ydistance']), 'or_zdistance_min': np.nanmin(test['or_zdistance']),
                                                'or_xdistance_max': np.nanmax(test['or_xdistance']),'or_ydistance_max': np.nanmax(test['or_ydistance']), 'or_zdistance_max': np.nanmax(test['or_zdistance']),
                           'rl_llerr_mean': np.mean(test['rl_lldistance']),'rl_llerr_std': np.std(test['rl_lldistance']),'rl_llerr_min': np.nanmin(test['rl_lldistance']),'rl_llerr_max': np.nanmax(test['rl_lldistance']),
                            'rl_herr_mean': np.mean(test['rl_hdistance']),'rl_herr_std': np.std(test['rl_hdistance']),'rl_herr_min': np.nanmin(test['rl_hdistance']),'rl_herr_max': np.nanmax(test['rl_hdistance']),
                            'or_llerr_mean': np.mean(test['or_lldistance']),'or_llerr_std': np.std(test['or_lldistance']),'or_llerr_min': np.nanmin(test['or_lldistance']),'or_llerr_max': np.nanmax(test['or_lldistance']),
                            'or_herr_mean': np.mean(test['or_hdistance']),'or_herr_std': np.std(test['or_hdistance']),'or_herr_min': np.nanmin(test['or_hdistance']),'or_herr_max': np.nanmax(test['or_hdistance'])
                           }
                xyz_distance_pd=pd.DataFrame.from_dict(tmp_dic, orient='index')
                pd_gen=True
            error_mean_all += test['error'].describe()['count'] * test['error'].describe()['mean']
            rl_distances_mean_all += test['count_rl_distance'].describe()['count'] * test['count_rl_distance'].describe()['mean']
            rl_RMSE_all += np.sum(test['count_rl_distance'] **2) # 统计 RMSE 结果
            rl_ll_RMSE_all += np.sum(test['rl_lldistance'] **2)
            rl_h_RMSE_all += np.sum(test['rl_hdistance'] **2)
            or_distances_mean_all += test['count_or_distance'].describe()['count'] * test['count_or_distance'].describe()['mean']
            or_RMSE_all += np.sum(test['count_or_distance'] **2)
            or_ll_RMSE_all += np.sum(test['or_lldistance'] **2)
            or_h_RMSE_all += np.sum(test['or_hdistance'] **2)
            error_std_all += test['error'].describe()['count'] * test['error'].describe()['std']
            rl_distances_std_all += test['count_rl_distance'].describe()['count'] * test['count_rl_distance'].describe()['std']
            or_distances_std_all += test['count_or_distance'].describe()['count'] * test['count_or_distance'].describe()['std']
            rl_ll_mean_all += test['rl_lldistance'].describe()['count'] * test['rl_lldistance'].describe()['mean']
            or_ll_mean_all += test['or_lldistance'].describe()['count'] * test['or_lldistance'].describe()['mean']

        except:
            print(f'Trajectory {train_tripIDnum} error.')

    num_total_err = np.sum(error_pd.loc['count', :])
    num_total_rl = np.sum(rl_distance_pd.loc['count', :])
    num_total_or = np.sum(or_distance_pd.loc['count', :])
    error_min = np.min(error_pd.loc['min', :])
    error_max = np.max(error_pd.loc['max', :])
    # 计算总的 RMSE 误差
    rl_RMSE, rl_ll_RMSE, rl_h_RMSE = np.sqrt(rl_RMSE_all/num_total_err), np.sqrt(rl_ll_RMSE_all/num_total_err), np.sqrt(rl_h_RMSE_all/num_total_err)
    or_RMSE, or_ll_RMSE, or_h_RMSE = np.sqrt(or_RMSE_all/num_total_err), np.sqrt(or_ll_RMSE_all/num_total_err), np.sqrt(or_h_RMSE_all/num_total_err)
    avg_xyz_err = rl_distances_mean_all / num_total_rl
    avg_rl_llerr = rl_ll_mean_all / num_total_rl
    avg_or_llerr = or_ll_mean_all / num_total_rl
    # 保存总体数据
    error_pd.insert(error_pd.shape[1], 'Avg', [num_total_err, error_mean_all / num_total_err, error_std_all / num_total_err,
                                               error_min, 0, 0, 0, error_max])
    rl_distance_pd.insert(rl_distance_pd.shape[1], 'Avg',
                          [num_total_rl, rl_distances_mean_all / num_total_rl, rl_distances_std_all / num_total_rl,
                           np.min(rl_distance_pd.loc['min', :]), 0, 0, 0, np.max(rl_distance_pd.loc['max', :])])
    or_distance_pd.insert(or_distance_pd.shape[1], 'Avg',
                          [num_total_or, or_distances_mean_all / num_total_or, or_distances_std_all / num_total_or,
                           np.min(or_distance_pd.loc['min', :]), 0, 0, 0, np.max(or_distance_pd.loc['max', :])])
    error_pd.to_csv(logdirname + 'errors.csv', index=True)
    rl_distance_pd.to_csv(logdirname + 'rl_distances.csv', index=True)
    or_distance_pd.to_csv(logdirname + 'or_distances.csv', index=True)
    xyz_distance_pd.to_csv(logdirname + 'xyz_distances.csv', index=True)
    # 保存结果
    file_name = logdirname + f'Result_pos ratio:{(or_RMSE-rl_RMSE)/or_RMSE:.2f}_rl RMSE:{rl_RMSE:.2f}_or RMSE:{or_RMSE:.2f}.txt'
    with open(file_name, "w", encoding="utf-8") as file:
        file.write(f'Ratio:{(or_RMSE-rl_RMSE)/or_RMSE:.2f}, rl RMSE:{rl_RMSE:.2f}, or RMSE:{or_RMSE:.2f}, rl ll RMSE:{rl_ll_RMSE:.2f},'
                   f'or ll RMSE:{or_ll_RMSE:.2f}, rl h RMSE:{rl_h_RMSE:.2f}, or h RMSE:{or_h_RMSE:.2f}')

    print(
        f'Perfermances: count {num_total_err:1.0f}, compared with baseline mean: {error_mean_all / num_total_err:4.3f}+{error_std_all / num_total_err:4.3f}m, '
        f'min: {error_min:4.3f}m, max: {error_max:4.3f}m.')

    return (or_RMSE-rl_RMSE)/or_RMSE, rl_RMSE

def recording_results_vel_RL4KF(dir_path, data_truth_dic,trajdata_range,tripIDlist,logdirname,baseline_mod,traj_record):
    error_mean_all = 0
    rl_distances_mean_all = 0
    rl_RMSE_all = 0
    or_distances_mean_all = 0
    or_RMSE_all = 0
    error_std_all = 0
    rl_distances_std_all = 0
    or_distances_std_all = 0

    pd_gen=False
    for train_tripIDnum in range(trajdata_range[0], trajdata_range[1] + 1):
        try:
            pd_train = data_truth_dic[tripIDlist[train_tripIDnum]]
            index_nonull = pd_train['Latitude_RLpredict'].notnull()
            pd_train = pd_train[index_nonull]
            # 提取整数历元的数据
            # 初始化准备导航状态
            config_filename = os.path.abspath(
                f'{dir_path}/dataset_Urbannav/{tripIDlist[train_tripIDnum]}/kf-gins.yaml')
            with open(config_filename, 'r', encoding='utf-8') as f:
                traj_config = yaml.load(f, Loader=yaml.FullLoader)  # 从YAML文件(filename)中加载轨迹配置数据

            step = int(traj_config["imudatarate"]/ traj_config["gnssrate"])
            pd_train = pd_train.iloc[::step]
            pd_train.reset_index(drop=True, inplace=True)

            if traj_record:
                # record rl traj
                if baseline_mod == 'GNSS/INS':
                    record_columns = ['UnixTimeMillis_ref', ' Latitude (deg)', ' Longitude (deg)', ' Ellipsoid Height (m)',
                  ' Velocity X (m/s)', ' Velocity Y (m/s)', ' Velocity Z (m/s)',
                  ' Roll (deg)', ' Pitch (deg)', ' Heading (deg)',' Latitude_GT (deg)', ' Longitude_GT (deg)', ' Ellipsoid Height_GT (m)',
                  ' Velocity X_GT (m/s)', ' Velocity Y_GT (m/s)', ' Velocity Z_GT (m/s)',
                  ' Roll_GT (deg)', ' Pitch_GT (deg)', ' Heading_GT (deg)', 'Latitude_RLpredict', 'Longitude_RLpredict', 'Ellipsoid_Height_RLpredict',
                  'Velocity_X_RLpredict', 'Velocity_Y_RLpredict', 'Velocity_Z_RLpredict',
                  'Roll_RLpredict', 'Pitch_RLpredict', 'Heading_RLpredict']

                pd_record = pd_train[record_columns]
                pd_record.to_csv(logdirname + f'rl_traj_{tripIDlist[train_tripIDnum].replace("/","_")}.csv', index=True)

            if baseline_mod == 'GNSS/INS':
                test = pd_train.loc[:, [' Velocity X (m/s)', ' Velocity Y (m/s)', ' Velocity Z (m/s)',
                                        ' Velocity X_GT (m/s)', ' Velocity Y_GT (m/s)', ' Velocity Z_GT (m/s)',
                                        'Velocity_X_RLpredict', 'Velocity_Y_RLpredict', 'Velocity_Z_RLpredict']]

            blh_station = test.iloc[0,0:3].values # 转导航坐标系初始点
            test['rl_distance'] = test.apply(lambda test: cal_distance_vel_RLKF(test,baseline_mod)[0], axis=1)
            test['or_distance'] = test.apply(lambda test: cal_distance_vel_RLKF(test,baseline_mod)[1], axis=1)
            test['error'] = test['rl_distance'].astype(float) - test['or_distance'].astype(float)
            test['count_rl_distance'] = test['rl_distance'].astype(float)
            test['count_or_distance'] = test['or_distance'].astype(float)
            test['rl_xdistance'] = test.apply(lambda test: cal_distance_vel_RLKF(test,baseline_mod)[2], axis=1)
            test['rl_ydistance'] = test.apply(lambda test: cal_distance_vel_RLKF(test,baseline_mod)[3], axis=1)
            test['rl_zdistance'] = test.apply(lambda test: cal_distance_vel_RLKF(test,baseline_mod)[4], axis=1)
            test['count_rl_xdistance'] = test['rl_xdistance'].astype(float)
            test['count_rl_ydistance'] = test['rl_ydistance'].astype(float)
            test['count_rl_zdistance'] = test['rl_zdistance'].astype(float)
            test['or_xdistance'] = test.apply(lambda test: cal_distance_vel_RLKF(test,baseline_mod)[5], axis=1)
            test['or_ydistance'] = test.apply(lambda test: cal_distance_vel_RLKF(test,baseline_mod)[6], axis=1)
            test['or_zdistance'] = test.apply(lambda test: cal_distance_vel_RLKF(test,baseline_mod)[7], axis=1)
            test['count_or_xdistance'] = test['or_xdistance'].astype(float)
            test['count_or_ydistance'] = test['or_ydistance'].astype(float)
            test['count_or_zdistance'] = test['or_zdistance'].astype(float)

            if pd_gen:
                error_pd.insert(error_pd.shape[1], f'{train_tripIDnum}', test['error'].describe())
                rl_distance_pd.insert(rl_distance_pd.shape[1], f'{train_tripIDnum}',
                                      test['count_rl_distance'].describe())
                or_distance_pd.insert(or_distance_pd.shape[1], f'{train_tripIDnum}',
                                      test['count_or_distance'].describe())
                tmp_dic = {'tripID':tripIDlist[train_tripIDnum], 'rl_xdistance_mean': np.mean(test['rl_xdistance']), 'rl_ydistance_mean': np.mean(test['rl_ydistance']),'rl_zdistance_mean': np.mean(test['rl_zdistance']),
                                                'rl_xdistance_std': np.std(test['rl_xdistance']),'rl_ydistance_std': np.std(test['rl_ydistance']), 'rl_zdistance_std': np.std(test['rl_zdistance']),
                                                'rl_xdistance_min': min(test['rl_xdistance']),'rl_ydistance_min': np.nanmin(test['rl_ydistance']), 'rl_zdistance_min': np.nanmin(test['rl_zdistance']),
                                                'rl_xdistance_max': np.nanmax(test['rl_xdistance']),'rl_ydistance_max': np.nanmax(test['rl_ydistance']), 'rl_zdistance_max': np.nanmax(test['rl_zdistance']),
                                                'or_xdistance_mean': np.mean(test['or_xdistance']), 'or_ydistance_mean': np.mean(test['or_ydistance']),'or_zdistance_mean': np.mean(test['or_zdistance']),
                                                'or_xdistance_std': np.std(test['or_xdistance']),'or_ydistance_std': np.std(test['or_ydistance']), 'or_zdistance_std': np.std(test['or_zdistance']),
                                                'or_xdistance_min': np.nanmin(test['or_xdistance']),'or_ydistance_min': np.nanmin(test['or_ydistance']), 'or_zdistance_min': np.nanmin(test['or_zdistance']),
                                                'or_xdistance_max': np.nanmax(test['or_xdistance']),'or_ydistance_max': np.nanmax(test['or_ydistance']), 'or_zdistance_max': np.nanmax(test['or_zdistance']),}

                xyz_distance_pd.insert(xyz_distance_pd.shape[1], f'{train_tripIDnum}',
                                       pd.DataFrame.from_dict(tmp_dic, orient='index').loc[:, 0])
            else:
                error_pd = pd.DataFrame(test['error'].describe())
                error_pd = error_pd.rename(columns={'error': f'{train_tripIDnum}'})
                error_pd.index.name = 'errors'
                rl_distance_pd = pd.DataFrame(test['count_rl_distance'].describe())
                rl_distance_pd = rl_distance_pd.rename(columns={'count_rl_distance': f'{train_tripIDnum}'})
                rl_distance_pd.index.name = 'rl_distances'
                or_distance_pd = pd.DataFrame(test['count_or_distance'].describe())
                or_distance_pd = or_distance_pd.rename(columns={'count_or_distance': f'{train_tripIDnum}'})
                or_distance_pd.index.name = 'or_distances'

                tmp_dic = {'tripID':tripIDlist[train_tripIDnum], 'rl_xdistance_mean': np.mean(test['rl_xdistance']), 'rl_ydistance_mean': np.mean(test['rl_ydistance']),'rl_zdistance_mean': np.mean(test['rl_zdistance']),
                                                'rl_xdistance_std': np.std(test['rl_xdistance']),'rl_ydistance_std': np.std(test['rl_ydistance']), 'rl_zdistance_std': np.std(test['rl_zdistance']),
                                                'rl_xdistance_min': np.nanmin(test['rl_xdistance']),'rl_ydistance_min': np.nanmin(test['rl_ydistance']), 'rl_zdistance_min': np.nanmin(test['rl_zdistance']),
                                                'rl_xdistance_max': np.nanmax(test['rl_xdistance']),'rl_ydistance_max': np.nanmax(test['rl_ydistance']), 'rl_zdistance_max': np.nanmax(test['rl_zdistance']),
                                                'or_xdistance_mean': np.mean(test['or_xdistance']), 'or_ydistance_mean': np.mean(test['or_ydistance']),'or_zdistance_mean': np.mean(test['or_zdistance']),
                                                'or_xdistance_std': np.std(test['or_xdistance']),'or_ydistance_std': np.std(test['or_ydistance']), 'or_zdistance_std': np.std(test['or_zdistance']),
                                                'or_xdistance_min': np.nanmin(test['or_xdistance']),'or_ydistance_min': np.nanmin(test['or_ydistance']), 'or_zdistance_min': np.nanmin(test['or_zdistance']),
                                                'or_xdistance_max': np.nanmax(test['or_xdistance']),'or_ydistance_max': np.nanmax(test['or_ydistance']), 'or_zdistance_max': np.nanmax(test['or_zdistance']),}
                xyz_distance_pd=pd.DataFrame.from_dict(tmp_dic, orient='index')
                pd_gen=True
            error_mean_all += test['error'].describe()['count'] * test['error'].describe()['mean']
            rl_distances_mean_all += test['count_rl_distance'].describe()['count'] * test['count_rl_distance'].describe()['mean']
            rl_RMSE_all += np.sum(test['count_rl_distance'] **2) # 统计 RMSE 结果
            or_distances_mean_all += test['count_or_distance'].describe()['count'] * test['count_or_distance'].describe()['mean']
            or_RMSE_all += np.sum(test['count_or_distance'] **2)
            error_std_all += test['error'].describe()['count'] * test['error'].describe()['std']
            rl_distances_std_all += test['count_rl_distance'].describe()['count'] * test['count_rl_distance'].describe()['std']
            or_distances_std_all += test['count_or_distance'].describe()['count'] * test['count_or_distance'].describe()['std']
        except:
            print(f'Trajectory {train_tripIDnum} error.')

    num_total_err = np.sum(error_pd.loc['count', :])
    num_total_rl = np.sum(rl_distance_pd.loc['count', :])
    num_total_or = np.sum(or_distance_pd.loc['count', :])
    error_min = np.min(error_pd.loc['min', :])
    error_max = np.max(error_pd.loc['max', :])
    # 计算总的 RMSE 误差
    rl_RMSE = np.sqrt(rl_RMSE_all/num_total_err)
    or_RMSE = np.sqrt(or_RMSE_all/num_total_err)
    avg_xyz_err = rl_distances_mean_all / num_total_rl

    # 保存总体数据
    error_pd.insert(error_pd.shape[1], 'Avg', [num_total_err, error_mean_all / num_total_err, error_std_all / num_total_err,
                                               error_min, 0, 0, 0, error_max])
    rl_distance_pd.insert(rl_distance_pd.shape[1], 'Avg',
                          [num_total_rl, rl_distances_mean_all / num_total_rl, rl_distances_std_all / num_total_rl,
                           np.min(rl_distance_pd.loc['min', :]), 0, 0, 0, np.max(rl_distance_pd.loc['max', :])])
    or_distance_pd.insert(or_distance_pd.shape[1], 'Avg',
                          [num_total_or, or_distances_mean_all / num_total_or, or_distances_std_all / num_total_or,
                           np.min(or_distance_pd.loc['min', :]), 0, 0, 0, np.max(or_distance_pd.loc['max', :])])
    error_pd.to_csv(logdirname + 'errors_vel.csv', index=True)
    rl_distance_pd.to_csv(logdirname + 'rl_vel_error.csv', index=True)
    or_distance_pd.to_csv(logdirname + 'or_vel_error.csv', index=True)
    xyz_distance_pd.to_csv(logdirname + 'xyz_vel_error.csv', index=True)
    # 保存结果
    file_name = logdirname + f'Result_vel ratio:{(or_RMSE-rl_RMSE)/or_RMSE:.2f}_rl RMSE:{rl_RMSE:.2f}_or RMSE:{or_RMSE:.2f}.txt'
    with open(file_name, "w", encoding="utf-8") as file:
        file.write(f'Ratio:{(or_RMSE-rl_RMSE)/or_RMSE:.2f}, rl vel RMSE:{rl_RMSE:.2f}, or vel RMSE:{or_RMSE:.2f}')

    print(
        f'Perfermances: count {num_total_err:1.0f}, compared with baseline mean: {error_mean_all / num_total_err:4.3f}+{error_std_all / num_total_err:4.3f}m, '
        f'min: {error_min:4.3f}m, max: {error_max:4.3f}m.')

    return (or_RMSE-rl_RMSE)/or_RMSE, rl_RMSE

def recording_results_att_RL4KF(dir_path, data_truth_dic,trajdata_range,tripIDlist,logdirname,baseline_mod,traj_record):
    error_mean_all = 0
    rl_distances_mean_all = 0
    rl_RMSE_all = 0
    or_distances_mean_all = 0
    or_RMSE_all = 0
    error_std_all = 0
    rl_distances_std_all = 0
    or_distances_std_all = 0

    pd_gen=False
    for train_tripIDnum in range(trajdata_range[0], trajdata_range[1] + 1):
        try:
            pd_train = data_truth_dic[tripIDlist[train_tripIDnum]]
            index_nonull = pd_train['Latitude_RLpredict'].notnull()
            pd_train = pd_train[index_nonull]
            # 提取整数历元的数据
            # 初始化准备导航状态
            config_filename = os.path.abspath(
                f'{dir_path}/dataset_Urbannav/{tripIDlist[train_tripIDnum]}/kf-gins.yaml')
            with open(config_filename, 'r', encoding='utf-8') as f:
                traj_config = yaml.load(f, Loader=yaml.FullLoader)  # 从YAML文件(filename)中加载轨迹配置数据

            step = int(traj_config["imudatarate"]/ traj_config["gnssrate"])
            pd_train = pd_train.iloc[::step]
            pd_train.reset_index(drop=True, inplace=True)

            if traj_record:
                # record rl traj
                if baseline_mod == 'GNSS/INS':
                    record_columns = ['UnixTimeMillis_ref', ' Latitude (deg)', ' Longitude (deg)', ' Ellipsoid Height (m)',
                  ' Velocity X (m/s)', ' Velocity Y (m/s)', ' Velocity Z (m/s)',
                  ' Roll (deg)', ' Pitch (deg)', ' Heading (deg)',' Latitude_GT (deg)', ' Longitude_GT (deg)', ' Ellipsoid Height_GT (m)',
                  ' Velocity X_GT (m/s)', ' Velocity Y_GT (m/s)', ' Velocity Z_GT (m/s)',
                  ' Roll_GT (deg)', ' Pitch_GT (deg)', ' Heading_GT (deg)', 'Latitude_RLpredict', 'Longitude_RLpredict', 'Ellipsoid_Height_RLpredict',
                  'Velocity_X_RLpredict', 'Velocity_Y_RLpredict', 'Velocity_Z_RLpredict',
                  'Roll_RLpredict', 'Pitch_RLpredict', 'Heading_RLpredict']

                pd_record = pd_train[record_columns]
                pd_record.to_csv(logdirname + f'rl_traj_{tripIDlist[train_tripIDnum].replace("/","_")}.csv', index=True)

            if baseline_mod == 'GNSS/INS':
                test = pd_train.loc[:, [' Roll (deg)', ' Pitch (deg)', ' Heading (deg)',
                                        ' Roll_GT (deg)', ' Pitch_GT (deg)', ' Heading_GT (deg)',
                                        'Roll_RLpredict', 'Pitch_RLpredict', 'Heading_RLpredict']]

            blh_station = test.iloc[0,0:3].values # 转导航坐标系初始点
            test['rl_distance'] = test.apply(lambda test: cal_distance_att_RLKF(test,baseline_mod)[0], axis=1)
            test['or_distance'] = test.apply(lambda test: cal_distance_att_RLKF(test,baseline_mod)[1], axis=1)
            test['error'] = test['rl_distance'].astype(float) - test['or_distance'].astype(float)
            test['count_rl_distance'] = test['rl_distance'].astype(float)
            test['count_or_distance'] = test['or_distance'].astype(float)
            test['rl_xdistance'] = test.apply(lambda test: cal_distance_att_RLKF(test,baseline_mod)[2], axis=1)
            test['rl_ydistance'] = test.apply(lambda test: cal_distance_att_RLKF(test,baseline_mod)[3], axis=1)
            test['rl_zdistance'] = test.apply(lambda test: cal_distance_att_RLKF(test,baseline_mod)[4], axis=1)
            test['count_rl_xdistance'] = test['rl_xdistance'].astype(float)
            test['count_rl_ydistance'] = test['rl_ydistance'].astype(float)
            test['count_rl_zdistance'] = test['rl_zdistance'].astype(float)
            test['or_xdistance'] = test.apply(lambda test: cal_distance_att_RLKF(test,baseline_mod)[5], axis=1)
            test['or_ydistance'] = test.apply(lambda test: cal_distance_att_RLKF(test,baseline_mod)[6], axis=1)
            test['or_zdistance'] = test.apply(lambda test: cal_distance_att_RLKF(test,baseline_mod)[7], axis=1)
            test['count_or_xdistance'] = test['or_xdistance'].astype(float)
            test['count_or_ydistance'] = test['or_ydistance'].astype(float)
            test['count_or_zdistance'] = test['or_zdistance'].astype(float)

            if pd_gen:
                error_pd.insert(error_pd.shape[1], f'{train_tripIDnum}', test['error'].describe())
                rl_distance_pd.insert(rl_distance_pd.shape[1], f'{train_tripIDnum}',
                                      test['count_rl_distance'].describe())
                or_distance_pd.insert(or_distance_pd.shape[1], f'{train_tripIDnum}',
                                      test['count_or_distance'].describe())
                tmp_dic = {'tripID':tripIDlist[train_tripIDnum], 'rl_xdistance_mean': np.mean(test['rl_xdistance']), 'rl_ydistance_mean': np.mean(test['rl_ydistance']),'rl_zdistance_mean': np.mean(test['rl_zdistance']),
                                                'rl_xdistance_std': np.std(test['rl_xdistance']),'rl_ydistance_std': np.std(test['rl_ydistance']), 'rl_zdistance_std': np.std(test['rl_zdistance']),
                                                'rl_xdistance_min': min(test['rl_xdistance']),'rl_ydistance_min': np.nanmin(test['rl_ydistance']), 'rl_zdistance_min': np.nanmin(test['rl_zdistance']),
                                                'rl_xdistance_max': np.nanmax(test['rl_xdistance']),'rl_ydistance_max': np.nanmax(test['rl_ydistance']), 'rl_zdistance_max': np.nanmax(test['rl_zdistance']),
                                                'or_xdistance_mean': np.mean(test['or_xdistance']), 'or_ydistance_mean': np.mean(test['or_ydistance']),'or_zdistance_mean': np.mean(test['or_zdistance']),
                                                'or_xdistance_std': np.std(test['or_xdistance']),'or_ydistance_std': np.std(test['or_ydistance']), 'or_zdistance_std': np.std(test['or_zdistance']),
                                                'or_xdistance_min': np.nanmin(test['or_xdistance']),'or_ydistance_min': np.nanmin(test['or_ydistance']), 'or_zdistance_min': np.nanmin(test['or_zdistance']),
                                                'or_xdistance_max': np.nanmax(test['or_xdistance']),'or_ydistance_max': np.nanmax(test['or_ydistance']), 'or_zdistance_max': np.nanmax(test['or_zdistance']),}

                xyz_distance_pd.insert(xyz_distance_pd.shape[1], f'{train_tripIDnum}',
                                       pd.DataFrame.from_dict(tmp_dic, orient='index').loc[:, 0])
            else:
                error_pd = pd.DataFrame(test['error'].describe())
                error_pd = error_pd.rename(columns={'error': f'{train_tripIDnum}'})
                error_pd.index.name = 'errors'
                rl_distance_pd = pd.DataFrame(test['count_rl_distance'].describe())
                rl_distance_pd = rl_distance_pd.rename(columns={'count_rl_distance': f'{train_tripIDnum}'})
                rl_distance_pd.index.name = 'rl_distances'
                or_distance_pd = pd.DataFrame(test['count_or_distance'].describe())
                or_distance_pd = or_distance_pd.rename(columns={'count_or_distance': f'{train_tripIDnum}'})
                or_distance_pd.index.name = 'or_distances'

                tmp_dic = {'tripID':tripIDlist[train_tripIDnum], 'rl_xdistance_mean': np.mean(test['rl_xdistance']), 'rl_ydistance_mean': np.mean(test['rl_ydistance']),'rl_zdistance_mean': np.mean(test['rl_zdistance']),
                                                'rl_xdistance_std': np.std(test['rl_xdistance']),'rl_ydistance_std': np.std(test['rl_ydistance']), 'rl_zdistance_std': np.std(test['rl_zdistance']),
                                                'rl_xdistance_min': np.nanmin(test['rl_xdistance']),'rl_ydistance_min': np.nanmin(test['rl_ydistance']), 'rl_zdistance_min': np.nanmin(test['rl_zdistance']),
                                                'rl_xdistance_max': np.nanmax(test['rl_xdistance']),'rl_ydistance_max': np.nanmax(test['rl_ydistance']), 'rl_zdistance_max': np.nanmax(test['rl_zdistance']),
                                                'or_xdistance_mean': np.mean(test['or_xdistance']), 'or_ydistance_mean': np.mean(test['or_ydistance']),'or_zdistance_mean': np.mean(test['or_zdistance']),
                                                'or_xdistance_std': np.std(test['or_xdistance']),'or_ydistance_std': np.std(test['or_ydistance']), 'or_zdistance_std': np.std(test['or_zdistance']),
                                                'or_xdistance_min': np.nanmin(test['or_xdistance']),'or_ydistance_min': np.nanmin(test['or_ydistance']), 'or_zdistance_min': np.nanmin(test['or_zdistance']),
                                                'or_xdistance_max': np.nanmax(test['or_xdistance']),'or_ydistance_max': np.nanmax(test['or_ydistance']), 'or_zdistance_max': np.nanmax(test['or_zdistance']),}
                xyz_distance_pd=pd.DataFrame.from_dict(tmp_dic, orient='index')
                pd_gen=True
            error_mean_all += test['error'].describe()['count'] * test['error'].describe()['mean']
            rl_distances_mean_all += test['count_rl_distance'].describe()['count'] * test['count_rl_distance'].describe()['mean']
            rl_RMSE_all += np.sum(test['count_rl_distance'] **2) # 统计 RMSE 结果
            or_distances_mean_all += test['count_or_distance'].describe()['count'] * test['count_or_distance'].describe()['mean']
            or_RMSE_all += np.sum(test['count_or_distance'] **2)
            error_std_all += test['error'].describe()['count'] * test['error'].describe()['std']
            rl_distances_std_all += test['count_rl_distance'].describe()['count'] * test['count_rl_distance'].describe()['std']
            or_distances_std_all += test['count_or_distance'].describe()['count'] * test['count_or_distance'].describe()['std']
        except:
            print(f'Trajectory {train_tripIDnum} error.')

    num_total_err = np.sum(error_pd.loc['count', :])
    num_total_rl = np.sum(rl_distance_pd.loc['count', :])
    num_total_or = np.sum(or_distance_pd.loc['count', :])
    error_min = np.min(error_pd.loc['min', :])
    error_max = np.max(error_pd.loc['max', :])
    # 计算总的 RMSE 误差
    rl_RMSE = np.sqrt(rl_RMSE_all/num_total_err)
    or_RMSE = np.sqrt(or_RMSE_all/num_total_err)
    avg_xyz_err = rl_distances_mean_all / num_total_rl

    # 保存总体数据
    error_pd.insert(error_pd.shape[1], 'Avg', [num_total_err, error_mean_all / num_total_err, error_std_all / num_total_err,
                                               error_min, 0, 0, 0, error_max])
    rl_distance_pd.insert(rl_distance_pd.shape[1], 'Avg',
                          [num_total_rl, rl_distances_mean_all / num_total_rl, rl_distances_std_all / num_total_rl,
                           np.min(rl_distance_pd.loc['min', :]), 0, 0, 0, np.max(rl_distance_pd.loc['max', :])])
    or_distance_pd.insert(or_distance_pd.shape[1], 'Avg',
                          [num_total_or, or_distances_mean_all / num_total_or, or_distances_std_all / num_total_or,
                           np.min(or_distance_pd.loc['min', :]), 0, 0, 0, np.max(or_distance_pd.loc['max', :])])
    error_pd.to_csv(logdirname + 'errors_att.csv', index=True)
    rl_distance_pd.to_csv(logdirname + 'rl_att_error.csv', index=True)
    or_distance_pd.to_csv(logdirname + 'or_att_error.csv', index=True)
    xyz_distance_pd.to_csv(logdirname + 'xyz_att_error.csv', index=True)
    # 保存结果
    file_name = logdirname + f'Result_att ratio:{(or_RMSE-rl_RMSE)/or_RMSE:.2f}_rl RMSE:{rl_RMSE:.2f}_or RMSE:{or_RMSE:.2f}.txt'
    with open(file_name, "w", encoding="utf-8") as file:
        file.write(f'Ratio:{(or_RMSE-rl_RMSE)/or_RMSE:.2f}, rl att RMSE:{rl_RMSE:.2f}, or vel RMSE:{or_RMSE:.2f}')

    print(
        f'Perfermances: count {num_total_err:1.0f}, compared with baseline mean: {error_mean_all / num_total_err:4.3f}+{error_std_all / num_total_err:4.3f}m, '
        f'min: {error_min:4.3f}m, max: {error_max:4.3f}m.')

    return (or_RMSE-rl_RMSE)/or_RMSE, rl_RMSE


def create_numbered_dir(base_path: str, prefix: str = "RecurrentPPO_") -> str:
    """
    在指定的 base_path 下创建一个自动递增编号的文件夹。
    如果 base_path 不存在，会自动创建。

    :param base_path: 父文件夹路径 (例如你的 result_folder)
    :param prefix: 新建文件夹的前缀名称
    :return: 新建文件夹的完整绝对路径
    """
    # 1. 确保父文件夹存在
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    # 2. 遍历查找已存在的最大编号
    max_num = 0
    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        # 必须是文件夹，且以指定前缀开头
        if os.path.isdir(item_path) and item.startswith(prefix):
            try:
                # 截取前缀后面的部分并转为整数
                num_str = item[len(prefix):]
                num = int(num_str)
                if num > max_num:
                    max_num = num
            except ValueError:
                # 如果后缀不是纯数字（比如 RecurrentPPO_test），则跳过
                continue

    # 3. 计算下一个编号并创建文件夹
    next_num = max_num + 1
    new_folder_name = f"{prefix}{next_num}"
    # final_dir = os.path.join(base_path, new_folder_name)
    #
    # os.makedirs(final_dir)
    return new_folder_name