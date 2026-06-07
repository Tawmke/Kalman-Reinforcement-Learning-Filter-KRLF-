import math
import os
import argparse
import sys
import numpy as np
import math as m
import matplotlib.pyplot as plt
import pandas as pd
from kfgins.kf_gins_types import GINSOptions
from common.angle import Angle
import yaml

# WGS84参数
WGS84_RA = 6378137.0
WGS84_E1 = 0.00669437999013
WGS84_WIE = 7.2921151467e-5

D2R = np.pi / 180.0
R2D = 180.0 / np.pi

device_mapping = {
    "google_pixel4": "phone",
    "huawei_p40pro": "phone",
    "samsung_note8": "phone",
    "xiaomi_mi8": "phone",
    "ublox_f9p_splitter": "receiver",
    "ublox_f9p": "receiver",
    "ublox": "receiver",
    "trimble": "receiver"
}

def calculate_bearing(lat1, lon1, lat2, lon2):
    """
    计算从点 (lat1, lon1) 到点 (lat2, lon2) 的航向角（方位角）。
    输入：纬度、经度（单位为度）
    输出：航向角（单位为度，范围0-360°，相对于真北）
    """
    # 1. 将十进制度数转换为弧度
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    # 2. 计算经度差
    delta_lon_rad = lon2_rad - lon1_rad

    # 3. 使用球面正切公式计算
    # atan2(y, x) 参数顺序很重要！
    y = math.sin(delta_lon_rad) * math.cos(lat2_rad)
    x = math.cos(lat1_rad) * math.sin(lat2_rad) - \
        math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(delta_lon_rad)

    # 4. 计算初始方位角（弧度）
    bearing_rad = math.atan2(y, x)

    # 5. 将弧度转换为度，并从 (-180° ~ 180°) 调整到 (0° ~ 360°)
    bearing_deg = math.degrees(bearing_rad)
    bearing_deg = (bearing_deg + 360) % 360

    return bearing_deg

def radiusmn(lat):
    tmp = np.square(m.sin(lat))
    tmp = 1 - WGS84_E1 * tmp
    sqrttmp = np.sqrt(tmp)

    radm = WGS84_RA * (1 - WGS84_E1) / (sqrttmp * tmp)
    radn = WGS84_RA / sqrttmp
    return radm, radn


# 地理坐标系增量转成n系下坐标增量
# 参数: rm, rn 子午圈半径和卯酉圈半径; pos当前位置地理位置[rad, rad, m], drad(地理坐标系相对增量)[rad, rad, m]
# @param: return: dm, n系下增量
def drad2dm(rm, rn, pos, drad):
    dm = np.zeros([3, 1])
    dm[0] = drad[0] * (rm + pos[2])
    dm[1] = drad[1] * (rn + pos[2]) * m.cos(pos[0])
    dm[2] = -drad[2]
    return dm

def dm2drad(rm, rn, pos, dm):
    """
    n系下坐标增量转地理坐标系增量
    @param rm: 子午圈曲率半径
    @param rn: 卯酉圈曲率半径
    @param pos: 当前地理位置 [纬度(rad), 经度(rad), 高度(m)]
    @param dm: n系下增量 [北(m), 东(m), 地(m)]
    @return: drad, 地理坐标系增量 [d纬度(rad), d经度(rad), d高度(m)]
    """
    drad = np.zeros([3, 1])
    # 1. 北向增量转纬度增量: dphi = dm_N / (Rm + h)
    drad[0] = dm[0] / (rm + pos[2])
    # 2. 东向增量转经度增量: dlam = dm_E / ((Rn + h) * cos(phi))
    drad[1] = dm[1] / ((rn + pos[2]) * m.cos(pos[0]))
    # 3. 地向增量转高度增量: dh = -dm_D
    drad[2] = -dm[2]
    return drad

def epoch2time_m(ep):
    """ calculate time from epoch """
    doy = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    year = int(ep[0:4])
    mon = int(ep[4:6])
    day = int(ep[6:8])
    hour = int(ep[8:10])
    min = int(ep[10:12])
    sec = float(ep[12:-1])/100

    if year < 1970 or year > 2099 or mon < 1 or mon > 12:
        return 'error'
    days = (year-1970)*365+(year-1969)//4+doy[mon-1]+day-2
    if year % 4 == 0 and mon >= 3:
        days += 1
    time = days*86400+hour*3600+min*60+sec
    # time.sec = ep[5]-sec
    return time

def load_gnss_data(filename,skiprows):
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
    return data

def plotNavresult(navresult_filepath):

    navresult = np.loadtxt(navresult_filepath)

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
    plt.plot(navresult[:, 1], navresult[:, 4])
    plt.xlabel('Time [s]')
    plt.ylabel('Height [m]')
    plt.title('Height')
    plt.grid()
    plt.tight_layout()

    plt.figure()
    plt.plot(navresult[:, 1], navresult[:, 5:8])
    plt.legend(['North', 'East', 'Down'])
    plt.xlabel('Time [s]')
    plt.ylabel('Velocity [m/s]')
    plt.title('Velocity')
    plt.grid()
    plt.tight_layout()

    plt.figure()
    plt.plot(navresult[:, 1], navresult[:, 8:11])
    plt.legend(['Roll', 'Pitch', 'Yaw'])
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

def plotNavError(navresult_filepath, refresult_filepath, gnsspath,skiprows):

    naverror, gnsserror = calcNavresultError(navresult_filepath, refresult_filepath, gnsspath,skiprows)
    print('calculate mavigtion result error finished!')
    print('plotting navigation error!')

    # 绘制误差曲线
    plt.figure('GNSS/INS position error')
    plt.plot(naverror[:, 1], naverror[:, 2:5], linestyle='-')
    plt.plot(gnsserror[:, 0], gnsserror[:, 1:4], linestyle='--')
    plt.legend([f'GNSS/INS (N):{np.mean(np.abs(naverror[:,2])):.2f}', f'GNSS/INS (E):{np.mean(np.abs(naverror[:,3])):.2f}',
                f'GNSS/INS (D):{np.mean(np.abs(naverror[:,4])):.2f}',f'GNSS (N):{np.mean(np.abs(gnsserror[:,1])):.2f}', f'GNSS (E):{np.mean(np.abs(gnsserror[:,2])):.2f}',
                f'GNSS (D):{np.mean(np.abs(gnsserror[:,3])):.2f}'])
    plt.xlabel('Time [s]')
    plt.ylabel('Error [m]')
    plt.title(f'Position Error')
    plt.grid()
    plt.ylim([-50,50])
    plt.tight_layout()

    plt.figure('velocity error')
    plt.plot(naverror[:, 1], naverror[:, 5:8])
    plt.legend(['North', 'East', 'Down'])
    plt.xlabel('Time [s]')
    plt.ylabel('Error [m/s]')
    plt.title('Velocity Error')
    plt.grid()
    plt.tight_layout()

    plt.figure('attitude error')
    plt.plot(naverror[:, 1], naverror[:, 8:11])
    plt.legend(['Roll', 'Pitch', 'Yaw'])
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

def lord_GT(refresult_filepath,navresult,gnssresult,config):
    ref_column = ['GPS TOW (s)', 'UnixTimeMillis_ref', ' Latitude (deg)', ' Longitude (deg)', ' Ellipsoid Height (m)',
                  ' Velocity X (m/s)', ' Velocity Y (m/s)', ' Velocity Z (m/s)',
                  ' Roll (deg)', ' Pitch (deg)', ' Heading (deg)']
    if 'csv' in refresult_filepath:
        ref_pd = pd.read_csv(refresult_filepath)
        # GPS week/SOW to Unix
        leap_seconds = 18
        gps_to_unix_epoch_seconds = 315964800
        ref_pd['UnixTimeMillis_ref'] = ((ref_pd[' GPS Week'] * 604800 + ref_pd[
            'GPS TOW (s)'] - leap_seconds + gps_to_unix_epoch_seconds) * 1000).astype('int64') / 1000

        refresult = ref_pd[ref_column].values

    elif 'txt' in refresult_filepath:
        target_cols = [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 17, 18]
        ref_pd = pd.read_table(refresult_filepath, sep='\s+', skiprows=2, header=None, usecols=target_cols)  #
        ref_pd['Latitude (deg)'] = ref_pd.iloc[:, 1] + ref_pd.iloc[:, 2] / 60 + ref_pd.iloc[:, 3] / 3600
        ref_pd['Longitude (deg)'] = ref_pd.iloc[:, 4] + ref_pd.iloc[:, 5] / 60 + ref_pd.iloc[:, 6] / 3600
        ref_pd.drop(ref_pd.columns[[1, 2, 3, 4, 5, 6]], axis=1, inplace=True)
        ref_pd.columns = ['UnixTimeMillis_ref', ' Ellipsoid Height (m)',
                          ' Velocity X (m/s)', ' Velocity Y (m/s)', ' Velocity Z (m/s)', ' Roll (deg)', ' Pitch (deg)',
                          ' Heading (deg)', ' Latitude (deg)', ' Longitude (deg)']

        ref_pd.insert(0, 'GPS TOW (s)', 0)  # 加一列补充
        refresult = ref_pd[ref_column].values
        refresult[:, 10] = np.mod(refresult[:, 10], 360) # 航向角转到 [0,360]
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
    navresult[:, 2:4] = navresult[:, 2:4] # 注意，原始生成数据的经纬度是角度单位，后续算误差要转为弧度
    refresult[:, 2:4] = refresult[:, 2:4] 

    refinter = np.zeros_like(navresult)
    refinter[:, 1] = navresult[:, 1]
    for col in range(2, 11):
        refinter[:, col] = np.interp(navresult[:, 1], refresult[:, 1], refresult[:, col])

    # 补偿与GT的杆臂偏差
    GTlever = config["GTlever"]
    refinter_or = refinter.copy()
    lats_corrected, lons_corrected, atts_corrected = apply_lever_arm_compensation(refinter[:,2],refinter[:,3],refinter[:,4],refinter[:,10],GTlever[0],GTlever[1],GTlever[2])
    refinter[:,2], refinter[:,3], refinter[:,4] = lats_corrected, lons_corrected, atts_corrected

    # 计算误差
    naverror = np.zeros_like(navresult)
    naverror[:, 1] = navresult[:, 1]
    refinter_re = refinter.copy()
    refinter_re[:, [5, 6]] = refinter[:, [6, 5]] # 速度ENU转成NED下
    refinter_re[:, [7]] = -refinter[:, [7]]
    naverror[:, 2:11] = navresult[:, 2:11] - refinter_re[:, 2:11]
    naverror[:, 2:4] = naverror[:, 2:4] * D2R

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
    gnssresult[:, 1:3] = gnssresult[:, 1:3]
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
    gnsserror[:, 1:3] = gnsserror[:, 1:3] * D2R
    # 位置误差转到第一个位置确定的n系
    blh_station = navresult[0, 2:5]
    rm, rn = radiusmn(blh_station[0])
    for i in range(len(gnsserror)):
        gnsserror[i, 1:4] = drad2dm(rm, rn, blh_station, gnsserror[i, 1:4]).reshape(1, 3)

    # 计算平均误差
    nav_pos_err = np.sqrt(np.mean(naverror[:, 2] ** 2 + naverror[:, 3] ** 2 + naverror[:, 4] ** 2))
    nav_vel_err = np.sqrt(np.mean(naverror[:, 5] ** 2 + naverror[:, 6] ** 2 + naverror[:, 7] ** 2))
    nav_att_err = np.sqrt(np.mean(naverror[:, 8] ** 2 + naverror[:, 9] ** 2 + naverror[:, 10] ** 2))
    gnss_pos_err = np.sqrt(np.mean(gnsserror[:, 1] ** 2 + gnsserror[:, 2] ** 2 + gnsserror[:, 3] ** 2))

    return navresult, refinter, gnssinter, nav_pos_err,nav_vel_err,nav_att_err,gnss_pos_err

def calcNavresultError(navresult_filepath, refresult_filepath, config):
    # load gnss data
    navresult = np.loadtxt(navresult_filepath)
    # edict by tang in 1205
    ref_column = ['GPS TOW (s)', 'UnixTimeMillis_ref', ' Latitude (deg)', ' Longitude (deg)', ' Ellipsoid Height (m)',
                  ' Velocity X (m/s)', ' Velocity Y (m/s)', ' Velocity Z (m/s)',
                  ' Roll (deg)', ' Pitch (deg)', ' Heading (deg)']
    if 'csv' in refresult_filepath:
        ref_pd = pd.read_csv(refresult_filepath)
        # GPS week/SOW to Unix
        leap_seconds = 18
        gps_to_unix_epoch_seconds = 315964800
        ref_pd['UnixTimeMillis_ref'] = ((ref_pd[' GPS Week'] * 604800 + ref_pd[
            'GPS TOW (s)'] - leap_seconds + gps_to_unix_epoch_seconds) * 1000).astype('int64') / 1000

        refresult = ref_pd[ref_column].values

    elif 'txt' in refresult_filepath:
        target_cols = [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 17, 18]
        ref_pd = pd.read_table(refresult_filepath, sep='\s+', skiprows=2, header=None, usecols=target_cols)  #
        ref_pd['Latitude (deg)'] = ref_pd.iloc[:, 1] + ref_pd.iloc[:, 2] / 60 + ref_pd.iloc[:, 3] / 3600
        ref_pd['Longitude (deg)'] = ref_pd.iloc[:, 4] + ref_pd.iloc[:, 5] / 60 + ref_pd.iloc[:, 6] / 3600
        ref_pd.drop(ref_pd.columns[[1, 2, 3, 4, 5, 6]], axis=1, inplace=True)
        ref_pd.columns = ['UnixTimeMillis_ref', ' Ellipsoid Height (m)',
                          ' Velocity X (m/s)', ' Velocity Y (m/s)', ' Velocity Z (m/s)', ' Roll (deg)', ' Pitch (deg)',
                          ' Heading (deg)', ' Latitude (deg)', ' Longitude (deg)']

        ref_pd.insert(0, 'GPS TOW (s)', 0)  # 加一列补充
        refresult = ref_pd[ref_column].values

    # 航向角平滑
    for i in range(1, len(navresult)):
        if navresult[i, 10] - navresult[i - 1, 10] < -180:
            navresult[i:, 10] = navresult[i:, 10] + 360
        if navresult[i, 10] - navresult[i - 1, 10] > 180:
            navresult[i:, 10] = navresult[i:, 10] - 360

    for i in range(1, len(refresult)):
        if refresult[i, 10] - refresult[i - 1, 10] < -180:
            refresult[i:, 10] = refresult[i:, 10] + 360
        if refresult[i, 10] - refresult[i - 1, 10] > 180:
            refresult[i:, 10] = refresult[i:, 10] - 360

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
    lats_corrected, lons_corrected, atts_corrected = apply_lever_arm_compensation(refinter[:,2]/D2R,refinter[:,3]/D2R,refinter[:,4],refinter[:,10],GTlever[0],GTlever[1],GTlever[2])
    refinter[:,2], refinter[:,3], refinter[:,4] = lats_corrected* D2R, lons_corrected* D2R, atts_corrected

    # 计算误差
    naverror = np.zeros_like(navresult)
    naverror[:, 1] = navresult[:, 1]
    naverror[:, 2:11] = navresult[:, 2:11] - refinter[:, 2:11]

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

    return naverror

def apply_lever_arm_compensation(lats, lons, atts, headings, x_b=-1.5, y_b=0.0, zb=0.0):
    """
    使用 NumPy 向量化处理整列经纬度的杆臂补偿。

    参数:
    ----------
    lats     : np.array, 原始纬度列 (Degree)
    lons     : np.array, 原始经度列 (Degree)
    headings : np.array, 航向角列 (Degree, 北为0, 顺时针为正)
    x_b      : float, 杆臂在前向轴的分量 (天线在IMU后方则为 -1.5)
    y_b      : float, 杆臂在右向轴的分量 (默认 0.0)

    返回:
    ----------
    lats_imu, lons_imu : 处理后的IMU中心点经纬度
    """
    # 1. 常数定义 (WGS84 椭球体参数)
    RE = 6378137.0  # 地球长半径 (m)
    E2 = 0.00669437999013  # 第一偏心率平方

    # 2. 角度转弧度
    lat_rad = np.radians(lats)
    heading_rad = np.radians(headings)

    # 3. 计算随纬度变化的曲率半径
    # W 是计算半径的中间参数
    W = np.sqrt(1.0 - E2 * np.sin(lat_rad) ** 2)
    # Rm: 子午圈曲率半径 (用于纬度计算)
    Rm = RE * (1.0 - E2) / (W ** 3)
    # Rn: 卯酉圈曲率半径 (用于经度计算)
    Rn = RE / W

    # 4. 将载体系(Body)杆臂投影到地理系(NED)
    # delta_N: 北向位移, delta_E: 东向位移
    # 公式: NED = R_b_to_n * Body_LeverArm
    dN = x_b * np.cos(heading_rad) - y_b * np.sin(heading_rad)
    dE = x_b * np.sin(heading_rad) + y_b * np.cos(heading_rad)

    # 5. 将位移(m)转换为经纬度变化量(°)
    d_lat_deg = (dN / Rm) * (180.0 / np.pi)
    d_lon_deg = (dE / (Rn * np.cos(lat_rad))) * (180.0 / np.pi)

    # 6. 补偿计算 (IMU位置 = 天线位置 - 偏移量)
    lats_imu = lats - d_lat_deg
    lons_imu = lons - d_lon_deg
    atts_imu = atts + zb

    return lats_imu, lons_imu, atts_imu

def loadConfig(config, options: GINSOptions):
    # 读取初始位置(纬度 经度 高程)、(北向速度 东向速度 垂向速度)、姿态(欧拉角，ZYX旋转顺序, 横滚角、俯仰角、航向角)
    # load initial position(latitude longitude altitude)
    #              velocity(speeds in the directions of north, east and down)
    #              attitude(euler angle, ZYX, roll, pitch and yaw)
    vec1 = (np.array(config["initpos"])).astype(np.double)
    vec2 = (np.array(config["initvel"])).astype(np.double)
    vec3 = (np.array(config["initatt"])).astype(np.double)  # 没设置初始姿态要自己补偿

    options.initstate.pos = np.array([vec1[0], vec1[1], vec1[2]]) * Angle.D2R  # 经纬度转弧度
    options.initstate.vel = np.array([vec2[0], vec2[1], vec2[2]])
    options.initstate.euler = np.array([vec3[0], vec3[1], vec3[2]]) * Angle.D2R
    options.initstate.pos[2] *= Angle.R2D  # 高程不用

    # 读取IMU误差初始值(零偏和比例因子)
    # load initial imu error (bias and scale factor)
    vec1 = (np.array(config["initgyrbias"])).astype(np.double)
    vec2 = (np.array(config["initaccbias"])).astype(np.double)
    vec3 = (np.array(config["initgyrscale"])).astype(np.double)
    vec4 = (np.array(config["initaccscale"])).astype(np.double)

    options.initstate.imuerror.gyrbias = np.array([vec1[0], vec1[1], vec1[2]]) * Angle.D2R / 3600.0
    options.initstate.imuerror.accbias = np.array([vec2[0], vec2[1], vec2[2]]) * 1e-5
    options.initstate.imuerror.gyrscale = np.array([vec3[0], vec3[1], vec3[2]]) * 1e-6
    options.initstate.imuerror.accscale = np.array([vec4[0], vec4[1], vec4[2]]) * 1e-6

    # 读取初始位置、速度、姿态(欧拉角)的标准差
    # load initial position std, velocity std and attitude(euler angle) std
    vec1 = (np.array(config["initposstd"])).astype(np.double)
    vec2 = (np.array(config["initvelstd"])).astype(np.double)
    vec3 = (np.array(config["initattstd"])).astype(np.double)

    options.initstate_std.pos = np.array([vec1[0], vec1[1], vec1[2]])
    options.initstate_std.vel = np.array([vec2[0], vec2[1], vec2[2]])
    options.initstate_std.euler = np.array([vec3[0], vec3[1], vec3[2]]) * Angle.D2R

    # 读取IMU噪声参数
    # load imu noise parameters
    vec1 = (np.array(config["imunoise"]["arw"])).astype(np.double)
    vec2 = (np.array(config["imunoise"]["vrw"])).astype(np.double)
    vec3 = (np.array(config["imunoise"]["gbstd"])).astype(np.double)
    vec4 = (np.array(config["imunoise"]["abstd"])).astype(np.double)
    vec5 = (np.array(config["imunoise"]["gsstd"])).astype(np.double)
    vec6 = (np.array(config["imunoise"]["asstd"])).astype(np.double)

    options.imunoise.corr_time = (np.array(config["imunoise"]["corrtime"])).astype(np.double)
    options.imunoise.gyr_arw = np.array([vec1[0], vec1[1], vec1[2]])
    options.imunoise.acc_vrw = np.array([vec2[0], vec2[1], vec2[2]])
    options.imunoise.gyrbias_std = np.array([vec3[0], vec3[1], vec3[2]])
    options.imunoise.accbias_std = np.array([vec4[0], vec4[1], vec4[2]])
    options.imunoise.gyrscale_std = np.array([vec5[0], vec5[1], vec5[2]])
    options.imunoise.accscale_std = np.array([vec6[0], vec6[1], vec6[2]])

    # 读取IMU误差初始标准差,如果配置文件中没有设置，则采用IMU噪声参数中的零偏和比例因子的标准差
    # Load initial imu bias and scale std, set to bias and scale instability std if load failed
    try:
        vec1 = config['initbgstd']
    except:
        vec1 = [options.imunoise.gyrbias_std[0], options.imunoise.gyrbias_std[1], options.imunoise.gyrbias_std[2]]

    try:
        vec2 = config['initbastd']
    except:
        vec2 = [options.imunoise.accbias_std[0], options.imunoise.accbias_std[1], options.imunoise.accbias_std[2]]

    try:
        vec3 = config['initsgstd']
    except:
        vec3 = [options.imunoise.gyrscale_std[0], options.imunoise.gyrscale_std[1], options.imunoise.gyrscale_std[2]]

    try:
        vec4 = config['initsastd']
    except:
        vec4 = [options.imunoise.accscale_std[0], options.imunoise.accscale_std[1], options.imunoise.accscale_std[2]]

    # IMU初始误差转换为标准单位
    # convert initial IMU errors' units to standard units
    options.initstate_std.imuerror.gyrbias = np.array([vec1[0], vec1[1], vec1[2]]) * Angle.D2R / 3600.0
    options.initstate_std.imuerror.accbias = np.array([vec2[0], vec2[1], vec2[2]]) * 1e-5
    options.initstate_std.imuerror.gyrscale = np.array([vec3[0], vec3[1], vec3[2]]) * 1e-6
    options.initstate_std.imuerror.accscale = np.array([vec4[0], vec4[1], vec4[2]]) * 1e-6

    # IMU噪声参数转换为标准单位
    # convert imu noise parameters' units to standard units
    options.imunoise.gyr_arw *= (Angle.D2R / 60.0)
    options.imunoise.acc_vrw /= 60.0
    options.imunoise.gyrbias_std *= (Angle.D2R / 3600.0)
    options.imunoise.accbias_std *= 1e-5
    options.imunoise.gyrscale_std *= 1e-6
    options.imunoise.accscale_std *= 1e-6
    options.imunoise.corr_time *= 3600

    # GNSS天线杆臂, GNSS天线相位中心在IMU坐标系下位置
    # gnss antenna leverarm, position of GNSS antenna phase center in IMU frame
    if ("antlever" in config):
        vec1 = (np.array(config["antlever"])).astype(np.double)
        options.antlever = vec1


def calculate_dop_from_EA_AA(df):
    # 通过卫星高度角仰角估计DOP
    results = []

    # 按照你的毫秒基准 UnixTimeMillis_ref 进行分组
    for timestamp, group in df.groupby('UnixTimeMillis_ref'):

        # 同一颗卫星在不同 FRE 下 AA/EA 是一样的，必须去重
        # 否则 A 矩阵会因为线性相关导致求逆失败或结果错误
        sat_geometry = group[['AA', 'EA']].drop_duplicates()

        n_sats = len(sat_geometry)

        # 计算 DOP 至少需要 4 颗卫星
        if n_sats < 4:
            results.append({
                'UnixTimeMillis_ref': timestamp,
                'PDOP': np.nan, 'HDOP': np.nan, 'SatCount': n_sats
            })
            continue

        # 角度转弧度
        az_rad = np.radians(sat_geometry['AA'].values)
        el_rad = np.radians(sat_geometry['EA'].values)

        # 构建几何矩阵 A
        A = np.zeros((n_sats, 4))
        A[:, 0] = np.cos(el_rad) * np.sin(az_rad)  # North
        A[:, 1] = np.cos(el_rad) * np.cos(az_rad)  # East
        A[:, 2] = np.sin(el_rad)  # Up
        A[:, 3] = 1.0  # Receiver Clock Offset

        try:
            # 计算协方差矩阵 Q = (A^T * A)^-1
            Q = np.linalg.inv(A.T @ A)

            # 提取结果
            hdop = np.sqrt(Q[0, 0] + Q[1, 1])
            pdop = np.sqrt(Q[0, 0] + Q[1, 1] + Q[2, 2])

            results.append({
                'UnixTimeMillis_ref': timestamp,
                'PDOP': round(pdop, 3),
                'HDOP': round(hdop, 3),
                'SatCount': n_sats
            })
        except np.linalg.LinAlgError:
            # 如果矩阵奇异（卫星分布在一条直线上），则返回空
            results.append({
                'UnixTimeMillis_ref': timestamp,
                'PDOP': np.nan, 'HDOP': np.nan, 'SatCount': n_sats
            })

    return pd.DataFrame(results)

def get_sigma(q_mode):
    # 根据定位模式估算DOP的sigma
    if q_mode == 4: return 1.0
    if q_mode == 5: return 4.0
    if q_mode == 2: return 0.4
    if q_mode == 1: return 0.02

def body_to_nav_enu(vec_b, headings_north, pitches, rolls):
    """
    将载体坐标系 (ENU: 右-前-上) 下的向量批量转换到导航系 (ENU: 东-北-天)
    参数:
    ----------
    vec_b  : (N, 3) 矩阵 [Vx_right, Vy_front, Vz_up]
    headings_north : (N,) 数组，北向为0，顺时针为正 (单位: 度)
    pitches : (N,) 数组，抬头为正 (单位: 度)
    rolls   : (N,) 数组，右滚为正 (单位: 度)
    """

    # 1. 坐标转换：将"北向顺时针"航向角转换为"数学系东向逆时针"弧度
    # 这一步非常重要，否则方向会错位
    psi = np.radians(headings_north)
    theta = np.radians(pitches)
    phi = np.radians(rolls)

    # 预计算三角函数
    cp, sp = np.cos(psi), np.sin(psi)
    ct, st = np.cos(theta), np.sin(theta)
    cr, sr = np.cos(phi), np.sin(phi)

    # 2. 构造 ENU 旋转矩阵分量
    r11 = cp * cr - sp * st * sr
    r12 = -sp * ct
    r13 = cp * sr + sp * st * cr

    r21 = sp * cr + cp * st * sr
    r22 = cp * ct
    r23 = sp * sr - cp * st * cr

    r31 = -ct * sr
    r32 = st
    r33 = ct * cr

    # 3. 批量矩阵相乘 (V_n = C_b_n * V_b)
    v_east = r11 * vec_b[:, 0] + r12 * vec_b[:, 1] + r13 * vec_b[:, 2]
    v_north = r21 * vec_b[:, 0] + r22 * vec_b[:, 1] + r23 * vec_b[:, 2]
    v_up = r31 * vec_b[:, 0] + r32 * vec_b[:, 1] + r33 * vec_b[:, 2]

    return np.stack((-v_east, v_north, v_up), axis=1)
