import os
import sys
import numpy as np
import pandas as pd
cur_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(cur_dir, '..')
sys.path.append(src_dir)

from common.types import IMU

"""
用于KF-GINS组合导航IMU数据读取，IMU处理完后应该是前右下（载体）坐标系
注意：在下方next()函数中，IMU三轴数据要乘dt变成增量形式
（原本香港数据集的IMU是右前上坐标系，判断原始数据的坐标系的一个办法是看z轴的正负）
创建人：唐健浩
"""

class ImuFileLoader:
    def __init__(self, filename:str, columns:int, rate:int):
        self.dt_ = 1.0 / float(rate)
        if 'xsense' in filename or 'phone' in filename:
            imu_column = ['UnixTimeMillis_ref', ' Angular rate X (rad/s)', ' Angular rate Y (rad/s)', ' Angular rate Z (rad/s)',
                          ' Acceleration X (m/s^2)', ' Acceleration Y (m/s^2)',
                          ' Acceleration Z (m/s^2)']
            target_cols = [0, 17, 18, 19, 29, 30, 31]
            imu_pd = pd.read_csv(filename, usecols=target_cols)
            imu_pd.columns = imu_column
            imu_pd.iloc[:, 0] = imu_pd.iloc[:, 0] * 1e-6 # 对齐时间戳
            frd_imu_pd = imu_pd.copy()
            if 'phone' in filename: # 注意手机放置方向
                frd_imu_pd[' Angular rate X (rad/s)'] = imu_pd[' Angular rate Z (rad/s)']
                frd_imu_pd[' Angular rate Y (rad/s)'] = imu_pd[' Angular rate Y (rad/s)']
                frd_imu_pd[' Angular rate Z (rad/s)'] = -imu_pd[' Angular rate X (rad/s)']
                frd_imu_pd[' Acceleration X (m/s^2)'] = imu_pd[' Acceleration Z (m/s^2)']
                frd_imu_pd[' Acceleration Y (m/s^2)'] = imu_pd[' Acceleration Y (m/s^2)']
                frd_imu_pd[' Acceleration Z (m/s^2)'] = -imu_pd[' Acceleration X (m/s^2)']
            else: # 右前上 转为 前右下 坐标系
                frd_imu_pd[' Angular rate X (rad/s)'] = imu_pd[' Angular rate Y (rad/s)']
                frd_imu_pd[' Angular rate Y (rad/s)'] = imu_pd[' Angular rate X (rad/s)']
                frd_imu_pd[' Angular rate Z (rad/s)'] = -imu_pd[' Angular rate Z (rad/s)']
                frd_imu_pd[' Acceleration X (m/s^2)'] = imu_pd[' Acceleration Y (m/s^2)']
                frd_imu_pd[' Acceleration Y (m/s^2)'] = imu_pd[' Acceleration X (m/s^2)']
                frd_imu_pd[' Acceleration Z (m/s^2)'] = -imu_pd[' Acceleration Z (m/s^2)']
            self.data_ = frd_imu_pd.values

        elif 'Tokyo' or 'Data2019' in filename:
            imu_column = ['UnixTimeMillis_ref', ' Angular rate X (rad/s)', ' Angular rate Y (rad/s)',
                          ' Angular rate Z (rad/s)',
                          ' Acceleration X (m/s^2)', ' Acceleration Y (m/s^2)',
                          ' Acceleration Z (m/s^2)']
            imu_pd = pd.read_csv(filename)
            # GPS week/SOW to Unix
            leap_seconds = 18
            gps_to_unix_epoch_seconds = 315964800
            # imu_pd[' Angular rate Z (rad/s)'] = -imu_pd[' Angular rate Z (rad/s)'] # 修改：260304
            if 'Data2019' in filename: # 根据加速度z轴测量值的正负可以判断坐标系
                imu_pd['UnixTimeMillis_ref'] = ((imu_pd[' GPS Week'] * 604800 + imu_pd[
                    ' GPS TOW (s)'] - leap_seconds + gps_to_unix_epoch_seconds) * 1000).astype('int64')
                frd_imu_pd = imu_pd.copy()
                frd_imu_pd[' Angular rate X (rad/s)'] = imu_pd[' Angular rate Y (rad/s)']
                frd_imu_pd[' Angular rate Y (rad/s)'] = imu_pd[' Angular rate X (rad/s)']
                frd_imu_pd[' Angular rate Z (rad/s)'] = -imu_pd[' Angular rate Z (rad/s)']
                frd_imu_pd[' Acceleration X (m/s^2)'] = imu_pd[' Acceleration Y (m/s^2)']
                frd_imu_pd[' Acceleration Y (m/s^2)'] = imu_pd[' Acceleration X (m/s^2)']
                frd_imu_pd[' Acceleration Z (m/s^2)'] = -imu_pd[' Acceleration Z (m/s^2)']
                self.data_ = frd_imu_pd[imu_column].values
            else: # 右前上 转为 前右下 坐标系
                imu_pd['UnixTimeMillis_ref'] = ((imu_pd[' GPS Week'] * 604800 + imu_pd[
                    'GPS TOW (s)'] - leap_seconds + gps_to_unix_epoch_seconds) * 1000).astype('int64')
                self.data_ = imu_pd[imu_column].values

        self.data_[:, 0] = self.data_[:, 0] / 1000 # 最后时间戳转缓成单位为s
        self.index = 0
        self.pre_time = self.data_[0,][0]  # 读取首次时间

    def next(self):
        # 这里的IMU计算默认是前右下坐标系，数据处理时要先统一
        if self.index >= self.data_.shape[0]:
            return None
        data_ = self.data_[self.index, :]
        pre_time = self.pre_time
        imu_ = IMU()
        imu_.time = data_[0]
        imu_.dtheta = np.array(data_[1:4]) # 角速度
        imu_.dvel = np.array(data_[4:7]) #  # 加速度
        # 右前上 转为 前右下坐标系
        # R_rfu_to_frd = np.array([
        #     [0, 1, 0],  # 第一行：FRD_X = RFU_Y
        #     [1, 0, 0],  # 第二行：FRD_Y = RFU_X
        #     [0, 0, -1]  # 第三行：FRD_Z = -RFU_Z
        # ])
        # imu_.dtheta = R_rfu_to_frd @ imu_.dtheta
        # imu_.dvel = R_rfu_to_frd @ imu_.dvel

        dt = imu_.time - pre_time
        if dt < 0.1:
            imu_.dt = dt
        else:
            imu_.dt = self.dt_
        # 修改tang：原来数据集应该已经乘了dt，其他数据集也要做相应处理
        imu_.dtheta = imu_.dtheta * self.dt_ # 修改260304
        imu_.dvel =  imu_.dvel * imu_.dt

        self.index += 1
        self.pre_time = imu_.time

        return imu_

    def starttime(self):
        return self.data_[0, 0]

    def endtime(self):
        return self.data_[-1, 0]
    
    def isEof(self):
        return self.index >= self.data_.shape[0]
