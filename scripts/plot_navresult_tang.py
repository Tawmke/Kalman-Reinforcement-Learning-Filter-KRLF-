#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import sys
cur_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(cur_dir, '..')
# src_dir = os.path.join(cur_dir, '.')
sys.path.append(src_dir)
import argparse
import numpy as np
import math as m
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from datetime import datetime, timezone
from funcs import *

"""
用于绘制GNSS/INS和GNSS定位结果以及误差
创建人：唐健浩
"""

# WGS84参数
WGS84_RA = 6378137.0
WGS84_E1 = 0.00669437999013
WGS84_WIE = 7.2921151467e-5

D2R = np.pi / 180.0
R2D = 180.0 / np.pi

def load_gnss_data(filename,skiprows):
    if 'pos' in filename:
        while True:
            try:
                # rtk_pd = pd.read_table(rtk_path, sep='\s+', parse_dates={'Timestamp': [0, 1]},skiprows=rtk_skiprows)
                rtk_pd = pd.read_table(filename, sep='\s+', skiprows=skiprows)
                try:
                    ymd = rtk_pd['%'][0]  # 直到能读取到对应的数据
                    break
                except:
                    continue
            except:
                skiprows += 1
                continue

        # self.data_ = np.genfromtxt(filename, delimiter=None) # 从文本文件加载数据的强大函数，none为自动检测分隔符
        UnixTimeMillis_rtk = []
        for index in range(len(rtk_pd)):
            ymd = rtk_pd['%'][index]
            hms = rtk_pd['GPST'][index]
            time_m = epoch2time_m(ymd.replace('/', '') + hms.replace(':', '').replace('.', ''))
            UnixTimeMillis_rtk.append(f'{time_m * 1000:.0f}')
        rtk_pd['UnixTimeMillis_ref'] = UnixTimeMillis_rtk
        rtk_column = ['UnixTimeMillis_ref', 'latitude(deg)', 'longitude(deg)', 'height(m)']
        data = rtk_pd[rtk_column].values.astype(np.float64)
        data[:, 0] = (data[:, 0] - 18000) / 1000  # 转化为UTC时间

    elif 'nmea' in filename:
        gnss_data = []
        gnss_std_data = []
        max_len = 0
        max_len_std = 0
        with open(filename, 'r', encoding='utf-8', errors='replace') as file:
            for idx, line in enumerate(file):
                if 'GNZDA' in line:  # 找到当前日期
                    split_line = [item.strip() for item in line.strip().split(',')]
                    date = "".join(split_line[2:5])
                    break

            for idx, line in enumerate(file):
                if 'GNGGA' in line:
                    split_line = [item.strip() for item in line.strip().split(',')]
                    dt_str = f"{date} {split_line[1]}"
                    UnixTime = datetime.strptime(dt_str, "%d%m%Y %H%M%S.%f").replace(tzinfo=timezone.utc).timestamp()
                    split_line[1] = UnixTime
                    gnss_data.append(split_line)
                    if len(split_line) > max_len:
                        max_len = len(split_line)

                elif 'GBS' in line:
                    split_line = [item.strip() for item in line.strip().split(',')]
                    dt_str = f"{date} {split_line[1]}"
                    UnixTime = datetime.strptime(dt_str, "%d%m%Y %H%M%S.%f").replace(tzinfo=timezone.utc).timestamp()
                    split_line[1] = UnixTime
                    gnss_std_data.append(split_line)
                    if len(split_line) > max_len_std:
                        max_len_std = len(split_line)

        # 提取位置数据
        colums_gnss = ['UnixTimeMillis_ref', 'latitude(deg)', 'longitude(deg)', 'Altitude(m)', 'Geoid Separation{m}']
        index_list = [1, 2, 4, 9, 11]
        rtk_pd = list2pd(gnss_data, max_len, index_list, colums_gnss)
        rtk_pd.drop_duplicates(subset=['UnixTimeMillis_ref']).astype('float64').reset_index(drop=True)
        rtk_pd[['latitude(deg)', 'longitude(deg)']] = rtk_pd[['latitude(deg)', 'longitude(deg)']].apply(
            lambda x: x // 100 + (x % 100) / 60)
        rtk_pd['height(m)'] = rtk_pd['Altitude(m)'] + rtk_pd['Geoid Separation{m}']

        # 提取std数据
        colums_gnss = ['UnixTimeMillis_ref', 'sdn(m)', 'sde(m)', 'sdu(m)']
        index_list = [1, 2, 3, 4]
        std_pd = list2pd(gnss_std_data, max_len_std, index_list, colums_gnss)
        std_pd.drop_duplicates(subset=['UnixTimeMillis_ref']).astype('float64').reset_index(drop=True)

        # 合并数据
        merge_columns = ['UnixTimeMillis_ref']
        rtk_pd = rtk_pd.merge(std_pd, on=merge_columns, suffixes=('', ''))
        rtk_column = ['UnixTimeMillis_ref', 'latitude(deg)', 'longitude(deg)', 'height(m)']
        data = rtk_pd[rtk_column].values.astype(np.float64)


    return data

def plotNavresult(navresult_filepath,refresult_filepath):

    navresult = np.loadtxt(navresult_filepath)

    ref_column = ['GPS TOW (s)', 'UnixTimeMillis_ref', ' Latitude (deg)', ' Longitude (deg)', ' Ellipsoid Height (m)',
                  ' Velocity X (m/s)', ' Velocity Y (m/s)', ' Velocity Z (m/s)',
                  ' Roll (deg)', ' Pitch (deg)', ' Heading (deg)']
    if 'reference' in refresult_filepath:
        ref_pd = pd.read_csv(refresult_filepath)
        # GPS week/SOW to Unix
        leap_seconds = 18
        gps_to_unix_epoch_seconds = 315964800
        ref_pd['UnixTimeMillis_ref'] = ((ref_pd[' GPS Week'] * 604800 + ref_pd[
            'GPS TOW (s)'] - leap_seconds + gps_to_unix_epoch_seconds) * 1000).astype('int64') / 1000

        refresult = ref_pd[ref_column].values

    elif 'GT' in refresult_filepath:
        target_cols = [0,3,4,5,6,7,8,9,10,11,12,16,17,18]
        ref_pd = pd.read_table(refresult_filepath, sep='\s+', skiprows=2, header=None, usecols=target_cols) #
        ref_pd['Latitude (deg)'] = ref_pd.iloc[:, 1] + ref_pd.iloc[:, 2] / 60 + ref_pd.iloc[:, 3] / 3600
        ref_pd['Longitude (deg)'] = ref_pd.iloc[:, 4] + ref_pd.iloc[:, 5] / 60 + ref_pd.iloc[:, 6] / 3600
        ref_pd.drop(ref_pd.columns[[1, 2, 3, 4, 5, 6]], axis=1, inplace=True)
        ref_pd.columns = ['UnixTimeMillis_ref', ' Ellipsoid Height (m)',
                          ' Velocity X (m/s)', ' Velocity Y (m/s)',' Velocity Z (m/s)',' Roll (deg)',' Pitch (deg)',' Heading (deg)',' Latitude (deg)',' Longitude (deg)']
        ref_pd.insert(0, 'GPS TOW (s)', 0) # 加一列补充
        refresult = ref_pd[ref_column].values
        refresult[:, 10] = np.mod(refresult[:, 10], 360) # 修改0304：看是否需要
        # 载体坐标系 转 导航坐标系
        vel_body = refresult[:, 5:8] # 假设是 [右向速, 前向速, 上向速]
        vel_enu = body_to_nav_enu(vel_body, refresult[:, 10], refresult[:, 9], refresult[:, 8])
        refresult[:,5:8] = vel_enu

    refresult[:, [5, 6]] = refresult[:, [6, 5]] # 导航坐标系速度ENU转成NED下
    refresult[:, [7]] = -refresult[:, [7]]

    # 小范围内将位置转到第一个位置确定的n系
    pos = np.zeros([len(navresult), 4])
    navresult[:, 2:4] = navresult[:, 2:4] * D2R
    pos[:, 0] = navresult[:, 1]

    blh_station = navresult[0, 2:5]
    rm, rn = radiusmn(blh_station[0])

    for i in range(len(pos)):
        delta_blh = navresult[i, 2:5] - navresult[0, 2:5]
        pos[i, 1:4] = drad2dm(rm, rn, blh_station, delta_blh).reshape(1, 3)

    print('plotting estimated navigation result!')

    # 绘图
    plt.figure('horizontal position')
    plt.plot(pos[:, 2], pos[:, 1])
    plt.axis('equal')
    plt.xlabel('East [m]')
    plt.ylabel('North [m]')
    plt.title('Horizontal Position')
    plt.grid()
    plt.tight_layout()

    plt.figure('height')
    plt.plot(navresult[:, 1], navresult[:, 4],label=['GNSS/INS'])
    plt.plot(refresult[:, 1], refresult[:, 4],label=['GT'])
    plt.legend()
    plt.xlabel('Time [s]')
    plt.ylabel('Height [m]')
    plt.title('Height')
    plt.grid()
    plt.tight_layout()

    plt.figure()
    plt.plot(navresult[:, 1], navresult[:, 5:8],label=['North', 'East', 'Down'])
    plt.plot(refresult[:, 1], refresult[:, 5:8],label=['North (GT)', 'East (GT)', 'Down (GT)'])
    plt.legend()
    plt.xlabel('Time [s]')
    plt.ylabel('Velocity [m/s]')
    plt.title('Velocity')
    plt.grid()
    plt.tight_layout()

    plt.figure()
    plt.plot(navresult[:, 1], navresult[:, 8:11],label=['Roll', 'Pitch', 'Yaw'])
    plt.plot(refresult[:, 1], refresult[:, 8:11],label=['Roll (GT)', 'Pitch (GT)', 'Yaw (GT)'])
    plt.legend()
    plt.xlabel('Time [s]')
    plt.ylabel('Angle [deg]')
    plt.title('Attitude')
    plt.grid()
    plt.tight_layout()

    # plt.show()


def plotIMUerror(imuerr_filepath):

    imuerr = np.loadtxt(imuerr_filepath)
    print('plotting estimated IMU error!')

    plt.figure('gyro bias')
    plt.plot(imuerr[:, 0], imuerr[:, 1:4])
    plt.legend(['X', 'Y', 'Z'])
    plt.xlabel('Time [s]')
    plt.ylabel('Bias [deg/h]')
    plt.title('Gyroscope Bias')
    plt.grid()
    plt.tight_layout()

    plt.figure('accel bias')
    plt.plot(imuerr[:, 0], imuerr[:, 4:7])
    plt.legend(['X', 'Y', 'Z'])
    plt.xlabel('Time [s]')
    plt.ylabel('Bias [mGal]')
    plt.title('Accelerometer Bias')
    plt.grid()
    plt.tight_layout()

    plt.figure('gyro scale')
    plt.plot(imuerr[:, 0], imuerr[:, 7:10])
    plt.legend(['X', 'Y', 'Z'])
    plt.xlabel('Time [s]')
    plt.ylabel('Scale [ppm]')
    plt.title('Gyroscope Scale')
    plt.grid()
    plt.tight_layout()

    plt.figure('accel scale')
    plt.plot(imuerr[:, 0], imuerr[:, 10:13])
    plt.legend(['X', 'Y', 'Z'])
    plt.xlabel('Time [s]')
    plt.ylabel('Scale [ppm]')
    plt.title('Accelerometer Scale')
    plt.grid()
    plt.tight_layout()

    plt.show()


def plotNavError(navresult_filepath, refresult_filepath, gnsspath,skiprows,config):

    naverror, gnsserror = calcNavresultError(navresult_filepath, refresult_filepath, gnsspath,skiprows,config)
    print('calculate mavigtion result error finished!')
    print('plotting navigation error!')

    # 绘制误差曲线
    plt.figure('GNSS/INS position error')
    plt.plot(naverror[:, 1], naverror[:, 2:5], linestyle='-')
    plt.plot(gnsserror[:, 0], gnsserror[:, 1:4], linestyle='--')
    plt.legend([f'GNSS/INS (N):{np.mean(np.abs(naverror[:,2])):.2f}m', f'GNSS/INS (E):{np.mean(np.abs(naverror[:,3])):.2f}m',
                f'GNSS/INS (D):{np.mean(np.abs(naverror[:,4])):.2f}m',f'GNSS (N):{np.mean(np.abs(gnsserror[:,1])):.2f}m', f'GNSS (E):{np.mean(np.abs(gnsserror[:,2])):.2f}m',
                f'GNSS (D):{np.mean(np.abs(gnsserror[:,3])):.2f}m'])
    plt.xlabel('Time [s]')
    plt.ylabel('Error [m]')
    plt.title(f'Position Error')
    plt.grid()
    plt.ylim([-50,50])
    plt.tight_layout()

    plt.figure('velocity error')
    plt.plot(naverror[:, 1], naverror[:, 5:8])
    plt.legend([f'North:{np.mean(np.abs(naverror[:,5])):.2f}m/s', f'East:{np.mean(np.abs(naverror[:,6])):.2f}m/s', f'Down:{np.mean(np.abs(naverror[:,7])):.2f}m/s'])
    plt.xlabel('Time [s]')
    plt.ylabel('Error [m/s]')
    plt.title('Velocity Error')
    plt.grid()
    plt.tight_layout()

    plt.figure('attitude error')
    plt.plot(naverror[:, 1], naverror[:, 8:11])
    plt.legend([f'Roll:{np.mean(np.abs(naverror[:,8])):.2f} deg', f'Pitch:{np.mean(np.abs(naverror[:,9])):.2f} deg', f'Yaw:{np.mean(np.abs(naverror[:,10])):.2f} deg'])
    plt.xlabel('Time [s]')
    plt.ylabel('Error [deg]')
    plt.title('Attitude Error')
    plt.grid()
    plt.tight_layout()

    plt.show()


def plotSTD(std_filepath):

    std = np.loadtxt(std_filepath)
    print('plotting estimated STD!')

    plt.figure('position std')
    plt.plot(std[:, 0], std[:, 1:4])
    plt.legend(['North', 'East', 'Down'])
    plt.xlabel('Time [s]')
    plt.ylabel('STD [m]')
    plt.title('Position STD')
    plt.grid()
    plt.tight_layout()

    plt.figure('velocity std')
    plt.plot(std[:, 0], std[:, 4:7])
    plt.legend(['North', 'East', 'Down'])
    plt.xlabel('Time [s]')
    plt.ylabel('STD [m/s]')
    plt.title('Velocity STD')
    plt.grid()
    plt.tight_layout()

    plt.figure('attitude std')
    plt.plot(std[:, 0], std[:, 7:10])
    plt.legend(['Roll', 'Pitch', 'Yaw'])
    plt.xlabel('Time [s]')
    plt.ylabel('STD [deg]')
    plt.title('Attitude STD')
    plt.grid()
    plt.tight_layout()

    plt.figure('gyrobias std')
    plt.plot(std[:, 0], std[:, 10:13])
    plt.legend(['X', 'Y', 'Z'])
    plt.xlabel('Time [s]')
    plt.ylabel('STD [deg/h]')
    plt.title('Gyroscope Bias STD')
    plt.grid()
    plt.tight_layout()

    plt.figure('accelbias std')
    plt.plot(std[:, 0], std[:, 13:16])
    plt.legend(['X', 'Y', 'Z'])
    plt.xlabel('Time [s]')
    plt.ylabel('STD [mGal]')
    plt.title('Accelerometer Bias STD')
    plt.grid()
    plt.tight_layout()

    plt.figure('gyroscale std')
    plt.plot(std[:, 0], std[:, 16:19])
    plt.legend(['X', 'Y', 'Z'])
    plt.xlabel('Time [s]')
    plt.ylabel('STD [ppm]')
    plt.title('Gyroscope Scale STD')
    plt.grid()
    plt.tight_layout()

    plt.figure('accelscale std')
    plt.plot(std[:, 0], std[:, 19:22])
    plt.legend(['X', 'Y', 'Z'])
    plt.xlabel('Time [s]')
    plt.ylabel('STD [ppm]')
    plt.title('Accelerometer Scale STD')
    plt.grid()
    plt.tight_layout()

    plt.show()


def calcNavresultError(navresult_filepath, refresult_filepath, gnsspath,skiprows,config):
    # load gnss data
    gnssresult = load_gnss_data(gnsspath,skiprows)
    navresult = np.loadtxt(navresult_filepath)
    # edict by tang in 1205
    ref_column = ['GPS TOW (s)', 'UnixTimeMillis_ref', ' Latitude (deg)', ' Longitude (deg)', ' Ellipsoid Height (m)',
                  ' Velocity X (m/s)', ' Velocity Y (m/s)', ' Velocity Z (m/s)',
                  ' Roll (deg)', ' Pitch (deg)', ' Heading (deg)']
    if 'reference' in refresult_filepath:
        ref_pd = pd.read_csv(refresult_filepath)
        # GPS week/SOW to Unix
        leap_seconds = 18
        gps_to_unix_epoch_seconds = 315964800
        ref_pd['UnixTimeMillis_ref'] = ((ref_pd[' GPS Week'] * 604800 + ref_pd[
            'GPS TOW (s)'] - leap_seconds + gps_to_unix_epoch_seconds) * 1000).astype('int64') / 1000

        refresult = ref_pd[ref_column].values

    elif 'GT' in refresult_filepath:
        target_cols = [0,3,4,5,6,7,8,9,10,11,12,16,17,18]
        ref_pd = pd.read_table(refresult_filepath, sep='\s+', skiprows=2, header=None, usecols=target_cols) #
        ref_pd['Latitude (deg)'] = ref_pd.iloc[:, 1] + ref_pd.iloc[:, 2] / 60 + ref_pd.iloc[:, 3] / 3600
        ref_pd['Longitude (deg)'] = ref_pd.iloc[:, 4] + ref_pd.iloc[:, 5] / 60 + ref_pd.iloc[:, 6] / 3600
        ref_pd.drop(ref_pd.columns[[1, 2, 3, 4, 5, 6]], axis=1, inplace=True)
        ref_pd.columns = ['UnixTimeMillis_ref', ' Ellipsoid Height (m)',
                          ' Velocity X (m/s)', ' Velocity Y (m/s)',' Velocity Z (m/s)',' Roll (deg)',' Pitch (deg)',' Heading (deg)',' Latitude (deg)',' Longitude (deg)']
        ref_pd.insert(0, 'GPS TOW (s)', 0) # 加一列补充
        refresult = ref_pd[ref_column].values
        refresult[:, 10] = np.mod(refresult[:, 10], 360) # 修改：tang0304
        # 载体坐标系 转 导航坐标系
        vel_body = refresult[:, 5:10] # 假设是 [右向速, 前向速, 上向速]
        vel_enu = body_to_nav_enu(vel_body, refresult[:, 10], refresult[:, 9], refresult[:, 8])
        refresult[:,5:8] = vel_enu

    # 航向角平滑
    # for i in range(1, len(navresult)):
    #     if navresult[i, 10] - navresult[i - 1, 10] < -180:
    #         navresult[i:, 10] = navresult[i:, 10] + 360
    #     if navresult[i, 10] - navresult[i - 1, 10] > 180:
    #         navresult[i:, 10] = navresult[i:, 10] - 360
    #
    # for i in range(1, len(refresult)):
    #     if refresult[i, 10] - refresult[i - 1, 10] < -180:
    #         refresult[i:, 10] = refresult[i:, 10] + 360
    #     if refresult[i, 10] - refresult[i - 1, 10] > 180:
    #         refresult[i:, 10] = refresult[i:, 10] - 360

    # 找到数据重合部分，参考结果内插到测试结果
    start_time = refresult[0, 1] if refresult[0, 1] >= navresult [0, 1] else navresult [0, 1]
    end_time = refresult[-1, 1] if refresult[-1, 1] <= navresult [-1, 1] else navresult [-1, 1]
    start_index = np.argwhere(navresult[:, 1] >= start_time)[0, 0]
    end_index = np.argwhere(navresult[:, 1] <= end_time)[-1, 0]
    navresult = navresult[start_index:end_index, :]
    navresult[:, 2:4] = navresult[:, 2:4] * D2R
    refresult[:, 2:4] = refresult[:, 2:4] * D2R

    refinter = np.zeros_like(navresult)
    refinter[:, 1] = navresult[:, 1]
    for col in range(2, 11):
        refinter[:, col] = np.interp(navresult[:, 1], refresult[:, 1], refresult[:, col])

    # 补偿与GT的杆臂偏差
    GTlever = config["GTlever"]
    refinter_or = refinter.copy()
    lats_corrected, lons_corrected, atts_corrected = apply_lever_arm_compensation(refinter[:,2]/D2R,refinter[:,3]/D2R,refinter[:,4],refinter[:,10],GTlever[0],GTlever[1],GTlever[2])
    refinter[:,2], refinter[:,3], refinter[:,4] = lats_corrected* D2R, lons_corrected* D2R, atts_corrected

    # 计算误差
    naverror = np.zeros_like(navresult)
    naverror[:, 1] = navresult[:, 1]
    refinter[:, [5, 6]] = refinter[:, [6, 5]] # 速度ENU转成NED下
    refinter[:, [7]] = -refinter[:, [7]]
    # 注意heading在不同坐标系下方向可能不一样，也要注意是[-180,180]还是[0,360]
    # 特别注意：右前上坐标系的0方向是正东，前右下的0方向是正北
    # 误差计算
    naverror[:, 2:11] = navresult[:, 2:11] - refinter[:, 2:11]
    # naverror = naverror[0:220000,:] # 修改260305

    # 航向角误差处理
    for i in range(len(naverror)):
        if naverror[i, 10] > 180:
            naverror[i, 10] -= 360
        if naverror[i, 10] < -180:
            naverror[i, 10] += 360

    # 位置误差转到第一个位置确定的n系
    blh_station = navresult[0, 2:5]
    rm, rn = radiusmn(blh_station[0])
    for i in range(len(naverror)):
        naverror[i, 2:5] = drad2dm(rm, rn, blh_station, naverror[i, 2:5]).reshape(1, 3)

    # 计算gnss位置数据误差
    refresult = refinter_or.copy()# ref_pd[ref_column].values
    refresult = np.delete(refresult, 0, axis=1)
    start_time = refresult[0, 0] if refresult[0, 0] >= gnssresult[0, 0] else gnssresult [0,0]
    end_time = refresult[-1, 0] if refresult[-1, 0] <= gnssresult [-1, 0] else gnssresult [-1, 0]
    start_index = np.argwhere(gnssresult[:, 0] >= start_time)[0, 0]
    end_index = np.argwhere(gnssresult[:, 0] <= end_time)[-1, 0]
    gnssresult = gnssresult[start_index:end_index, :]
    gnssresult[:, 1:3] = gnssresult[:, 1:3] * D2R
    # refresult[:, 1:3] = refresult[:, 1:3] * D2R

    # GNSS 插值和 NAV的数量一样
    gnssinter = np.zeros_like(refresult)
    gnssinter[:, 0] = refresult[:, 0]
    for col in range(1, 4):
        gnssinter[:, col] = np.interp(refresult[:, 0], gnssresult[:, 0], gnssresult[:, col])

    # 计算误差
    gnsserror = np.zeros_like(gnssinter)
    gnsserror[:, 0] = gnssinter[:, 0]
    gnsserror[:, 1:4] = gnssinter[:, 1:4] - refresult[:, 1:4]
    # 位置误差转到第一个位置确定的n系
    blh_station = navresult[0, 2:5]
    rm, rn = radiusmn(blh_station[0])
    for i in range(len(gnsserror)):
        gnsserror[i, 1:4] = drad2dm(rm, rn, blh_station, gnsserror[i, 1:4]).reshape(1, 3)

    # gnsserror = gnsserror[0:220000, :]
    return naverror,gnsserror


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='KF-GINS') # 初始化解释器
    parser.add_argument('--conf', type=str, help='configuration file path') # 定义了一个可选的文件路径参数
    args = parser.parse_args()
    """
    可选：东京：Tokyo_Data_Odaiba Tokyo_Data_Shinjuku 
    香港新：1_UrbanNav-HK-Medium-Urban-1 2_UrbanNav-HK-Deep-Urban-1 3_UrbanNav-HK-Harsh-Urban-1
    香港旧：UrbanNav-HK-Data20190428 (IMU数据有点问题)
    """
    Dataset = '1_UrbanNav-HK-Medium-Urban-1'
    dataset_path = f'dataset_Urbannav/{Dataset}'
    src_dir += '/'
    try:
        filename = None
        if args.conf is None:
            filename = os.path.abspath(src_dir + f'./{dataset_path}/kf-gins.yaml')
        else:
            filename = args.conf
        with open(filename, 'r', encoding='utf-8') as f:
            config = yaml.load(f, Loader=yaml.FullLoader) # 从YAML文件(filename)中加载配置数据
    except Exception as e:
        print(f"Error details: {str(e)}")
        raise Exception("Failed to read configuration file. Please check the path and format of the configuration file!")

    # loadConfig(config, options)
    gnsspath = os.path.abspath(src_dir + config['gnsspath'])
    skiprows = int(config["skiprows"])

    # 导航结果和导航误差
    dataset_path = f'dataset_Urbannav/{Dataset}'
    navresult_filepath = f'../{dataset_path}/KF_GINS_Navresult_testing.nav'
    refresult_filepath = f'../{dataset_path}/{config["refname"]}'
    # 导航结果
    plotNavresult(navresult_filepath,refresult_filepath)
    # 计算并绘制导航误差
    plotNavError(navresult_filepath, refresult_filepath, gnsspath,skiprows,config)

    # 估计的IMU误差
    imuerr_filepath = '../dataset/KF_GINS_IMU_ERR.txt'
    # plotIMUerror(imuerr_filepath)

    # 估计的导航状态标准差和IMU误差标准差
    std_filepath = '../dataset/KF_GINS_STD.txt'
    # plotSTD(std_filepath)
