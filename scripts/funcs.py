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

# WGS84参数
WGS84_RA = 6378137.0
WGS84_E1 = 0.00669437999013
WGS84_WIE = 7.2921151467e-5

D2R = np.pi / 180.0
R2D = 180.0 / np.pi

# 计算子午圈半径和卯酉圈半径
# 输入参数: lat(纬度)[rad]
def radiusmn(lat):
    tmp = np.square(m.sin(lat))
    tmp = 1 - WGS84_E1 * tmp
    sqrttmp = np.sqrt(tmp)

    radm = WGS84_RA * (1 - WGS84_E1) / (sqrttmp * tmp)
    radn = WGS84_RA / sqrttmp
    return radm, radn

def list2pd(data, max_len, index_list, colums):
    data_padded = [line + [np.nan] * (max_len - len(line)) for line in data]
    raw_pd = pd.DataFrame(data_padded)
    process_pd = raw_pd.iloc[:, index_list].replace('', np.nan).apply(pd.to_numeric, errors='coerce')
    process_pd.columns = colums
    return process_pd


# 地理坐标系增量转成n系下坐标增量
# 参数: rm, rn 子午圈半径和卯酉圈半径; pos当前位置地理位置[rad, rad, m], drad(地理坐标系相对增量)[rad, rad, m]
# @param: return: dm, n系下增量
def drad2dm(rm, rn, pos, drad):
    dm = np.zeros([3, 1])
    dm[0] = drad[0] * (rm + pos[2])
    dm[1] = drad[1] * (rn + pos[2]) * m.cos(pos[0])
    dm[2] = -drad[2]
    return dm

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


def body_to_ned_frame(vec_b, headings, pitches, rolls):
    """
    将载体系下的向量批量转换到导航系 (NED)
    vec_b  : shape 为 (N, 3) 的矩阵，包含 [Vx_b, Vy_b, Vz_b]
    其余角度为 (N,) 的数组，单位为度
    """
    # 角度转弧度
    psi = np.radians(headings)  # Yaw
    theta = np.radians(pitches)  # Pitch
    phi = np.radians(rolls)  # Roll

    # 预计算三角函数
    c_psi, s_psi = np.cos(psi), np.sin(psi)
    c_the, s_the = np.cos(theta), np.sin(theta)
    c_phi, s_phi = np.cos(phi), np.sin(phi)

    # 构造旋转矩阵的 9 个分量 (向量化)
    r11 = c_the * c_psi
    r12 = s_phi * s_the * c_psi - c_phi * s_psi
    r13 = c_phi * s_the * c_psi + s_phi * s_psi

    r21 = c_the * s_psi
    r22 = s_phi * s_the * s_psi + c_phi * c_psi
    r23 = c_phi * s_the * s_psi - s_phi * c_psi

    r31 = -s_the
    r32 = s_phi * c_the
    r33 = c_phi * c_the

    # 执行变换: V_n = C_b_n * V_b
    # vec_b[:, 0] 是 x, vec_b[:, 1] 是 y, vec_b[:, 2] 是 z
    vn_north = r11 * vec_b[:, 0] + r12 * vec_b[:, 1] + r13 * vec_b[:, 2]
    vn_east = r21 * vec_b[:, 0] + r22 * vec_b[:, 1] + r23 * vec_b[:, 2]
    vn_down = r31 * vec_b[:, 0] + r32 * vec_b[:, 1] + r33 * vec_b[:, 2]

    return np.stack((vn_north, vn_east, vn_down), axis=1)