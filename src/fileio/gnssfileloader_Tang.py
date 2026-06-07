import os
import sys
import numpy as np
import pandas as pd

cur_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(cur_dir, '..')
sys.path.append(src_dir)
from datetime import datetime, timezone
from pathlib import Path
from common.funcs import *
from common.types import GNSS
from common.angle import Angle
from common.timetransfer import *

"""
用于KF-GINS组合导航gnss数据读取，需要根据gnss文件来更改读取形式，读取完应该要保存gnss的经纬高位置信息
下方提供了rtklib生成的pos和nmea格式处理
创建人：唐健浩
"""

def list2pd(data, max_len, index_list, colums):
    data_padded = [line + [np.nan] * (max_len - len(line)) for line in data]
    raw_pd = pd.DataFrame(data_padded)
    process_pd = raw_pd.iloc[:, index_list].replace('', np.nan).apply(pd.to_numeric, errors='coerce')
    process_pd.columns = colums
    return process_pd

class GnssFileLoader:
    def __init__(self, filename: str, skiprows: int, config=None, save_gnss=False):
        # 两种方式导入原始GNSS的位置数据
        if 'pos' in filename:
            while True:
                try:
                    # rtk_pd = pd.read_table(rtk_path, sep='\s+', parse_dates={'Timestamp': [0, 1]},skiprows=rtk_skiprows)
                    rtk_pd = pd.read_table(filename, sep='\s+', skiprows=skiprows)
                    try:
                        ymd = rtk_pd['%'][0] # 直到能读取到对应的数据
                        break
                    except:
                        continue
                except:
                    skiprows += 1
                    continue

            UnixTimeMillis_rtk = []
            for index in range(len(rtk_pd)):
                ymd = rtk_pd['%'][index]
                hms = rtk_pd['GPST'][index]
                time_m = epoch2time_m(ymd.replace('/', '') + hms.replace(':', '').replace('.', ''))
                UnixTimeMillis_rtk.append(time_m - 18) # UTC 时间和 GPST 有18s补偿
            rtk_pd['UnixTimeMillis_ref'] = UnixTimeMillis_rtk

            rtk_column = ['UnixTimeMillis_ref', 'latitude(deg)', 'longitude(deg)', 'height(m)', 'sdn(m)', 'sde(m)','sdu(m)']
            if save_gnss:
                # 提取高度角 载噪比信息
                statpath = Path(filename).parent / config["statname"]
                gnss_feature_data = []
                max_len_feature = 0
                with open(statpath, 'r', encoding='utf-8', errors='replace') as file:
                    statlines = file.readlines()  # 一次性读入内存
                for idx, line in enumerate(statlines):
                    if '$SAT' in line:
                        split_line = [item.strip() for item in line.strip().split(',')]
                        gnss_feature_data.append(split_line)
                        if len(split_line) > max_len_feature:
                            max_len_feature = len(split_line)
                index_list = [1, 2, 4, 5, 6, 10]
                colums_gnss = ['GPS Week', 'GPS TOW (s)', 'FRE', 'AA', 'EA', 'CNR']
                gnss_feature_pd = list2pd(gnss_feature_data, max_len_feature, index_list, colums_gnss)
                leap_seconds = 18
                gps_to_unix_epoch_seconds = 315964800
                gnss_feature_pd['UnixTimeMillis_ref'] = ((gnss_feature_pd['GPS Week'] * 604800 + gnss_feature_pd[
                    'GPS TOW (s)'] - leap_seconds + gps_to_unix_epoch_seconds) * 1000).astype('int64') / 1000
                gnss_feature_pd = gnss_feature_pd.drop_duplicates().reset_index(drop=True)
                gnss_feature_pd = gnss_feature_pd[((gnss_feature_pd['EA'] >= 15) & (gnss_feature_pd['CNR'] >= 25))]
                # 计算DOP：数据没有提供DOP，只能用高度角方位角估算
                df_dop = calculate_dop_from_EA_AA(gnss_feature_pd)

                # 统计高度角的信息
                stats = gnss_feature_pd.groupby('UnixTimeMillis_ref')['CNR'].describe(percentiles=[0.25, 0.75])
                df_cnr = stats[['mean', '75%', '25%']].reset_index()
                df_cnr.columns = ['UnixTimeMillis_ref', 'CNR_Mean', 'CNR_Q75', 'CNR_Q25']
                # 统计载噪比信息
                stats = gnss_feature_pd.groupby('UnixTimeMillis_ref')['EA'].describe(percentiles=[0.25, 0.75])
                df_ea = stats[['mean', '75%', '25%']].reset_index()
                df_ea.columns = ['UnixTimeMillis_ref', 'EA_Mean', 'EA_Q75', 'EA_Q25']
                # 合并
                df_features = pd.merge(df_cnr, df_ea, on='UnixTimeMillis_ref', how='inner')
                df_features = pd.merge(df_features, df_dop, on='UnixTimeMillis_ref', how='inner')

                # 提取 卫星数
                # nmeapath = Path(filename).parent / config["nmeaname"]
                # gnss_data = []
                # max_len = 0
                # with open(nmeapath, 'r', encoding='utf-8', errors='replace') as file:
                #     filelines = file.readlines()  # 一次性读入内存
                #
                # old_date = ymd.replace('/', '') # 获取日期
                # date = old_date[6:8] + old_date[4:6] + old_date[0:4]
                # for idx, line in enumerate(filelines):
                #     if 'GGA' in line:
                #         split_line = [item.strip() for item in line.strip().split(',')]
                #         dt_str = f"{date} {split_line[1]}"
                #         UnixTime = datetime.strptime(dt_str, "%d%m%Y %H%M%S.%f").replace(tzinfo=timezone.utc).timestamp()
                #         split_line[1] = UnixTime
                #         gnss_data.append(split_line)
                #         if len(split_line) > max_len:
                #             max_len = len(split_line)
                # colums_gnss = ['UnixTimeMillis_ref', 'Satnum']
                # index_list = [1, 7]
                # nmea_pd = list2pd(gnss_data, max_len, index_list, colums_gnss)
                # nmea_pd.drop_duplicates(subset=['UnixTimeMillis_ref']).astype('float64').reset_index(drop=True)

                # 合并数据
                merge_columns = ['UnixTimeMillis_ref']
                rtk_pd = rtk_pd.merge(df_features, on=merge_columns, suffixes=('', ''))
                # rtk_pd = rtk_pd.merge(nmea_pd, on=merge_columns, suffixes=('', ''))

                rtk_column = ['UnixTimeMillis_ref', 'latitude(deg)', 'longitude(deg)', 'height(m)','sdn(m)', 'sde(m)','sdu(m)','ns',
                              'CNR_Mean','CNR_Q75','CNR_Q25','EA_Mean','EA_Q75','EA_Q25','PDOP','HDOP','Q']
            rtk_pd.dropna(inplace=True)
            self.data_ = rtk_pd[rtk_column].values.astype(np.float64)
            # self.data_[:,0] = (self.data_[:,0] - 18000) / 1000 # 转化为UTC时间，单位是s

        elif 'nmea' in filename:
            gnss_data = []
            gnss_std_data = []
            gnss_feature_data = []
            gnss_dop_data = []
            max_len = 0
            max_len_std = 0
            max_len_feature = 0
            max_len_dop = 0
            start_check = True
            with open(filename, 'r', encoding='utf-8', errors='replace') as file:
                filelines = file.readlines()  # 一次性读入内存

            for idx, line in enumerate(filelines):
                # 以下两种方式都可以获取日期，如果nmea不包含日期数据，可以直接在配置表配置
                if 'ZDA' in line: # 找到当前日期
                    split_line = [item.strip() for item in line.strip().split(',')]
                    date = "".join(split_line[2:5])
                    break
                elif 'RMC' in line:
                    split_line = [item.strip() for item in line.strip().split(',')]
                    raw_date = split_line[9]
                    date = f"{raw_date[0:2]}{raw_date[2:4]}20{raw_date[4:6]}"
                    break

            for idx, line in enumerate(filelines):
                if 'GGA' in line:
                    if start_check: # 开头的GGA的卫星信息不完全，跳过第一条定位信息
                        start_check = False
                        continue
                    split_line = [item.strip() for item in line.strip().split(',')]
                    if len(split_line[1]) == 0: # 隧道信号缺失
                        continue
                    dt_str = f"{date} {split_line[1]}"
                    UnixTime = datetime.strptime(dt_str, "%d%m%Y %H%M%S.%f").replace(tzinfo=timezone.utc).timestamp()
                    split_line[1] = UnixTime
                    gnss_data.append(split_line)
                    if len(split_line) > max_len:
                        max_len = len(split_line)

                    # 提取载噪比、高度角等信息
                    for nextline in reversed(filelines[:idx]): # 从下往上遍历
                        if 'GSV' in nextline: # 载噪比、高度角数据
                            split_line = [item.strip() for item in nextline.strip().split(',')]
                            split_line[1] = UnixTime
                            if len(split_line) >= 19:
                                split_line[19] = split_line[19].split('*')[0]  # 删除校验和
                            gnss_feature_data.append(split_line)
                            if len(split_line) > max_len_feature:
                                max_len_feature = len(split_line)

                        if 'GSA' in nextline: # DOP数据
                            split_line = [item.strip() for item in nextline.strip().split(',')]
                            split_line[1] = UnixTime
                            split_line[17] = split_line[17].split('*')[0] # 删除校验和
                            gnss_dop_data.append(split_line)
                            if len(split_line) > max_len_dop:
                                max_len_dop = len(split_line)

                        if 'GGA' in nextline:
                            break

                elif 'GST' in line:
                    split_line = [item.strip() for item in line.strip().split(',')]
                    dt_str = f"{date} {split_line[1]}"
                    UnixTime = datetime.strptime(dt_str, "%d%m%Y %H%M%S.%f").replace(tzinfo=timezone.utc).timestamp()
                    split_line[1] = UnixTime
                    split_line[-1] = split_line[-1].split('*')[0] # 删除校验和
                    gnss_std_data.append(split_line)
                    if len(split_line) > max_len_std:
                        max_len_std = len(split_line)

            # 提取位置数据
            colums_gnss = ['UnixTimeMillis_ref', 'latitude(deg)', 'longitude(deg)', 'raw_Q', 'Satnum', 'Altitude(m)', 'Geoid Separation{m}']
            index_list = [1, 2, 4, 6, 7, 9, 11]
            rtk_pd = list2pd(gnss_data, max_len, index_list, colums_gnss)
            rtk_pd.drop_duplicates(subset=['UnixTimeMillis_ref']).astype('float64').reset_index(drop=True)
            rtk_pd[['latitude(deg)', 'longitude(deg)']] = rtk_pd[['latitude(deg)', 'longitude(deg)']].apply(
                lambda x: x // 100 + (x % 100) / 60)

            valid_data = (rtk_pd['Geoid Separation{m}'].notna()) & (rtk_pd['Geoid Separation{m}'] != 0) # 判断是否有椭球高修正
            if valid_data.any():
                rtk_pd['height(m)'] = rtk_pd['Altitude(m)'] + rtk_pd['Geoid Separation{m}']
            else:
                rtk_pd['height(m)'] = rtk_pd['Altitude(m)']
                rtk_pd['Geoid Separation{m}'] = rtk_pd['Geoid Separation{m}'].fillna(0)

            nmea_to_rtklib_map = {
                4: 1,  # Fix
                5: 2,  # Float
                1: 5,  # Single
                2: 4,  # DGPS
                0: 0  # Invalid
            }
            rtk_pd['Q'] = rtk_pd['raw_Q'].map(nmea_to_rtklib_map) # 定位状态映射统一
            rtk_pd = rtk_pd.dropna(subset=['latitude(deg)']).reset_index(drop=True) # 剔除空的行

            # 提取dop数据
            colums_dop = ['UnixTimeMillis_ref', 'PDOP', 'HDOP', 'VDOP']
            index_list_dop = [1, 15, 16, 17]
            gnss_dop_pd = list2pd(gnss_dop_data, max_len_dop, index_list_dop, colums_dop).drop_duplicates(
                subset=['UnixTimeMillis_ref'], keep='first').reset_index(drop=True)

            # 提取std数据: ublox的数据会有std直接获取，手机数据和一些没有std，用DOP估算
            if gnss_std_data:
                colums_gnss = ['UnixTimeMillis_ref', 'sdn(m)', 'sde(m)','sdu(m)']
                index_list = [1, 6, 7, 8]
                std_pd = list2pd(gnss_std_data, max_len_std, index_list, colums_gnss)
                # std_pd.iloc[:, [0, 1, 2]] *= 1e-2 # 减小GNSS方差
                std_pd.drop_duplicates(subset=['UnixTimeMillis_ref']).astype('float64').reset_index(drop=True)
            else:
                std_pd = gnss_dop_pd.iloc[:, [0]].copy()
                sigma_map = {1: 0.02, 2: 0.8, 4: 3.0, 5: 5.0}
                std_pd['sigma'] = rtk_pd['Q'].map(sigma_map).fillna(8.0) # 根据定位模式估算DOP的sigma
                std_pd['sdn(m)'] = gnss_dop_pd['HDOP'] * std_pd['sigma'] * 0.707
                std_pd['sde(m)'] = gnss_dop_pd['HDOP'] * std_pd['sigma'] * 0.707
                std_pd['sdu(m)'] = gnss_dop_pd['VDOP'] * std_pd['sigma']

            rtk_column = ['UnixTimeMillis_ref', 'latitude(deg)', 'longitude(deg)', 'height(m)', 'sdn(m)', 'sde(m)','sdu(m)']
            merge_columns = ['UnixTimeMillis_ref']
            rtk_pd = rtk_pd.merge(std_pd, on=merge_columns, suffixes=('', ''))
            if save_gnss:
                # 提取卫星载噪比、高度角数据
                colums_CNR = ['UnixTimeMillis_ref','CNR1', 'CNR2', 'CNR3','CNR4']
                colums_EA = ['UnixTimeMillis_ref','EA1', 'EA2', 'EA3','EA4']
                index_list_CNR = [1, 7, 11, 15, 19]
                index_list_ea = [1, 5, 9, 13, 17]
                gnss_CNR_pd = list2pd(gnss_feature_data, max_len_feature, index_list_CNR, colums_CNR)
                gnss_EA_pd = list2pd(gnss_feature_data, max_len_feature, index_list_ea, colums_EA)
                # 剔除低载噪比和低仰角卫星
                cnr_values = gnss_CNR_pd[colums_CNR[1:]].values
                ea_values = gnss_EA_pd[colums_EA[1:]].values
                mask = (cnr_values < 25) | (ea_values < 15)
                gnss_CNR_pd.loc[:, colums_CNR[1:]] = gnss_CNR_pd[colums_CNR[1:]].mask(mask)
                gnss_EA_pd.loc[:, colums_EA[1:]] = gnss_EA_pd[colums_EA[1:]].mask(mask)
                gnss_CNR_pd[colums_CNR[1:]] = gnss_CNR_pd[colums_CNR[1:]].mask((gnss_CNR_pd[colums_CNR[1:]] > 90), np.nan) # 剔除错误数据
                gnss_EA_pd[colums_EA[1:]] = gnss_EA_pd[colums_EA[1:]].mask((gnss_EA_pd[colums_EA[1:]] > 90), np.nan) # 剔除错误数据
                # 统计载噪比特征
                df_cnr = gnss_CNR_pd.groupby('UnixTimeMillis_ref').apply(lambda x: pd.Series({
                    'CNR_Mean': x[colums_CNR[1:]].stack().mean(),  # 全局均值：反映平均信号强度
                    # 'CNR_Median': x[colums_gnss[1:]].stack().median(),  # 中位数：比均值更鲁棒，过滤掉个别强星干扰
                    'CNR_Q75': x[colums_CNR[1:]].stack().quantile(0.75),  # 75%分位数：反映当前最可靠的那颗星（顶星）
                    'CNR_Q25': x[colums_CNR[1:]].stack().quantile(0.25),  # 25%分位数：最关键！反映弱信号占比，捕捉 NLOS
                })).reset_index()
                df_ea = gnss_EA_pd.groupby('UnixTimeMillis_ref').apply(lambda x: pd.Series({
                    'EA_Mean': x[colums_EA[1:]].stack().mean(),  # 全局均值：反映平均信号强度
                    # 'CNR_Median': x[colums_gnss[1:]].stack().median(),  # 中位数：比均值更鲁棒，过滤掉个别强星干扰
                    'EA_Q75': x[colums_EA[1:]].stack().quantile(0.75),  # 75%分位数：反映当前最可靠的那颗星（顶星）
                    'EA_Q25': x[colums_EA[1:]].stack().quantile(0.25),  # 25%分位数：最关键！反映弱信号占比，捕捉 NLOS
                })).reset_index()
                df_features = pd.merge(df_cnr, df_ea, on='UnixTimeMillis_ref', how='inner')

                # 合并数据
                rtk_pd = rtk_pd.merge(df_features, on=merge_columns, suffixes=('', ''))
                rtk_pd = rtk_pd.merge(gnss_dop_pd, on=merge_columns, suffixes=('', ''))
                # 重新排序、提取numpy数据
                rtk_column = ['UnixTimeMillis_ref', 'latitude(deg)', 'longitude(deg)', 'height(m)','sdn(m)', 'sde(m)','sdu(m)','Satnum',
                              'CNR_Mean','CNR_Q75','CNR_Q25','EA_Mean','EA_Q75','EA_Q25','PDOP', 'HDOP','Q']
            rtk_pd.dropna(inplace=True)
            self.data_ = rtk_pd[rtk_column].values.astype(np.float64)

        self.index = 0

    def next(self):
        if self.index >= self.data_.shape[0]:
            return None
        data_ = self.data_[self.index, :]
        gnss_ = GNSS()
        gnss_.time = data_[0]
        gnss_.blh = np.array(data_[1:4]) # 经纬高
        gnss_.std = np.array(data_[4:7]) #
        gnss_.blh[0] *= Angle.D2R
        gnss_.blh[1] *= Angle.D2R
        self.index += 1

        return gnss_

    def starttime(self):
        return self.data_[0, 0]

    def endtime(self):
        return self.data_[-1, 0]
        
    def isEof(self):
        return self.index >= self.data_.shape[0]
