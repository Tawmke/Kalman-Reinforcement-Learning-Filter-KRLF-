import numpy as np
import pandas as pd
import sys
import pymap3d as pm
import pymap3d.vincenty as pmv
import matplotlib.pyplot as plt
import glob as gl
import scipy.optimize
from tqdm.auto import tqdm
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.spatial import distance
import os
from pathlib import Path
from scipy import signal
from scipy.signal import butter, filtfilt, buttord
from scipy.signal import medfilt
# import src.gnss_lib.coordinates as coord
import warnings
import pickle

import argparse
import yaml
import time

cur_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(cur_dir, '..')
sys.path.append(src_dir)
src_dir += '/' # 项目主路径：../kf_gins_py/

from kfgins.kf_gins_types import GINSOptions
from common.angle import Angle
from common.types import IMU, GNSS
from common.funcs_SZ import *
from fileio.gnssfileloader_SZdata import GnssFileLoader
from fileio.imufileloader_SZdata import ImuFileLoader
from kfgins.gi_engine import GIEngine
import folium
"""
可用自采集深圳数据集实现RTK/INS组合导航，该代码遍历所有轨迹读取对应参数表yaml，生成打包数据
创建人：唐健浩
"""

starting_flag = True # 设置为 False，则从某个特定traj开始遍历
load_nav_data = False # 设置为 True，可以直接导入已有导航数据
record_visualization = True # 是否需要可视化
starting_traj = 'SZ-20240729-1606-1716/RTK'
exclude_traj = ['SZ-20240728-1520-1641/RTK']

if __name__ == "__main__":
    savepath = Path(src_dir+'/dataset_SZ/')
    gnss_data_dic = {}
    data_truth_dic = {}
    imu_data_dic = {}
    trip_result = []
    for i, dirname in enumerate(tqdm(sorted(gl.glob(f'{src_dir}/dataset_SZ/*/*/')))):
        # dirname = '/mnt/sdb/home/tangjh/KF-GINS-Py-main/dataset_Urbannav/Tokyo_Data_Shinjuku/ublox' # 测试用
        # dirname = '/mnt/sdb/home/tangjh/KF-GINS-Py-main/dataset_Urbannav/1_UrbanNav-HK-Medium-Urban-1/ublox_m8t_GR/' # 测试用
        # dirname = '/mnt/sdb/home/tangjh/KF-GINS-Py-main/dataset_Urbannav/1_UrbanNav-HK-Medium-Urban-1/ublox_f9p' # 测试用
        record_nav = [] # 保存组合导航数据
        dataset_path = os.path.dirname(dirname)
        drive, equip = dirname.split('/')[-3:-1]
        tripID = f'{drive}/{equip}'
        device_type = 'receiver'
        if tripID in exclude_traj: # 排除不需要的traj
            continue
        # 可以在特定轨迹开始
        if tripID == starting_traj:
            starting_flag = True
        if not starting_flag:
            continue
        print(f'-------Processing: {tripID}----------------------')

        # S1：导入该数据的配置文件
        parser = argparse.ArgumentParser(description='KF-GINS')  # 初始化解释器
        parser.add_argument('--conf', type=str, help='configuration file path')  # 定义了一个可选的文件路径参数
        args = parser.parse_args()
        try:
            filename = None
            if args.conf is None:
                filename = os.path.abspath(f'{dirname}/kf-gins.yaml')
            else:
                filename = args.conf
            with open(filename, 'r', encoding='utf-8') as f:
                config = yaml.load(f, Loader=yaml.FullLoader)  # 从YAML文件(filename)中加载配置数据
        except Exception as e:
            print(f"Error details: {str(e)}")
            raise Exception(
                "Failed to read configuration file. Please check the path and format of the configuration file!")

        # S2：配置参数
        options = GINSOptions()
        loadConfig(config, options)

        imupath = os.path.abspath(src_dir + config['imupath'])
        gnsspath = os.path.abspath(src_dir + config['gnsspath'])
        outputpath = os.path.abspath(src_dir + config['outputpath'])
        skiprows = int(config["skiprows"])
        imudatalen = int(config["imudatalen"])
        imudatarate = int(config["imudatarate"])
        starttime = float(config["starttime"])  # 可以不用
        endtime = float(config["endtime"])

        # S3：加载GNSS文件和IMU文件
        gnssfile = GnssFileLoader(gnsspath, skiprows, config, save_gnss=True)
        imufile = ImuFileLoader(imupath, imudatalen, imudatarate)

        # S4:构建GNSS/INS推理引擎
        giengine = GIEngine(options)

        if endtime < 0:
            endtime = imufile.endtime()

        # if (endtime > 604800 or starttime < imufile.starttime() or starttime > endtime):
        #     print("Process time ERROR!")

        # S5:数据对齐, 时间移到定义的开始时间
        starttime = gnssfile.starttime()  # + 100 # 用GNSS的起始时间，不用配置表的
        imu_cur = IMU()  # IMU类包含IMU当前数据，加速度和角速度 时间等
        while True:
            imu_cur = imufile.next()
            if imu_cur.time >= starttime:
                break

        gnss = GNSS()  # GNSS类包含GNSS当前数据，经纬高和协方差等，待确定
        while True:
            gnss = gnssfile.next()
            if gnss.time >= starttime:
                break

        # S6:添加IMU和GNSS数据到GIEngine中，补偿IMU误差
        giengine.addImuData(imu_cur, True)  # 加载新时间的IMU数据，并设置是否补偿
        giengine.addGnssData(gnss)  # 加载新时间的gnss数据，并设置为可用

        process_time = time.time()

        # S7: GNSS/INS 主循环
        nav_data_path = dirname + '/KF_GINS_Navresult.nav'
        start_nav = True
        if load_nav_data: # 直接导入保存的导航数据
            if os.path.exists(nav_data_path):
                raw_result_nav = np.loadtxt(nav_data_path)
                if raw_result_nav.size != 0:
                    start_nav = False

        if start_nav:
            f_nav = open(nav_data_path, 'w')
            while True:
                # 当前IMU状态时间新于GNSS时间时，读取并添加新的GNSS数据到GIEngine
                # load new gnssdata when current state time is newer than GNSS time and add it to GIEngine
                if gnss.time < imu_cur.time and not gnssfile.isEof():  # 判断不大于数据长度
                    gnss = gnssfile.next()
                    giengine.addGnssData(gnss)

                # 读取并添加新的IMU数据到GIEngine
                # load new imudata and add it to GIEngine
                imu_cur = imufile.next()
                if imu_cur.time > endtime or imufile.isEof():  # 判断不大于IMU数据长度
                    break
                giengine.addImuData(imu_cur)

                # 处理新的IMU数据
                # process new imudata
                giengine.newImuProcess()

                timestamp = giengine.timestamp()
                navstate = giengine.getNavState()
                imuerr = navstate.imuerror

                result1 = np.array([
                    np.round(0, 9),  # 保留9位小数
                    np.round(timestamp, 9),
                    np.round(navstate.pos[0] * Angle.R2D, 9),
                    np.round(navstate.pos[1] * Angle.R2D, 9),
                    np.round(navstate.pos[2], 9),
                    np.round(navstate.vel[0], 9),
                    np.round(navstate.vel[1], 9),
                    np.round(navstate.vel[2], 9),
                    np.round(navstate.euler[0] * Angle.R2D, 9),
                    np.round(navstate.euler[1] * Angle.R2D, 9),
                    np.round(navstate.euler[2] * Angle.R2D, 9)])
                record_nav.append(result1)
                np.savetxt(f_nav, [result1], delimiter=" ", fmt="%.9f")  # 列分隔符为空格,保留9位小数

                # 创建动态进度显示
                progress = (timestamp - starttime) / (endtime - starttime) * 100.0
                sys.stdout.write('\r[{:.2f}%]'.format(progress) + str(timestamp))
                sys.stdout.flush()
            f_nav.close()
            raw_result_nav = np.array(record_nav)  # 导航结果矩阵

        # S8: 接入真值数据并将导航结果打包成dataframe，并统计导航误差
        refresult_filepath = f'{os.path.dirname(dataset_path)}/{config["refname"]}'
        navresult, refinter, nav_pos_err = lord_GT(refresult_filepath, raw_result_nav, gnssfile.data_, config) # 时间戳对齐，得到真值

        print(f'Nav: pos RMSE: {nav_pos_err}')

        pd_data = np.hstack((navresult[:,1:], refinter[:,2:]))
        column_name = ['UnixTimeMillis_ref', ' Latitude (deg)', ' Longitude (deg)', ' Ellipsoid Height (m)',
                  ' Velocity X (m/s)', ' Velocity Y (m/s)', ' Velocity Z (m/s)',
                  ' Roll (deg)', ' Pitch (deg)', ' Heading (deg)',' Latitude_GT (deg)', ' Longitude_GT (deg)', ' Ellipsoid Height_GT (m)']

        record_df = pd.DataFrame(pd_data, columns=column_name)

        if record_visualization:
            print('----------save trajectory----------')
            truth_gt = record_df[[' Latitude_GT (deg)', ' Longitude_GT (deg)']].to_numpy()
            gnssins_pd = record_df[[' Latitude (deg)', ' Longitude (deg)']].to_numpy()
            UTC_time = record_df['UnixTimeMillis_ref'].to_numpy()
            llh_gt_500 = [truth_gt[-20, 0], truth_gt[-20, 1]]
            google_satellite = 'https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}'
            m = folium.Map(location=llh_gt_500, tiles=google_satellite, attr='Google', zoom_start=18, zoom_max=25)
            print(f'Start utc={UTC_time[0]}, end utc={UTC_time[-1]}')
            for index in range(len(truth_gt)):
                if index % 100 == 0:
                    utc = UTC_time[index]
                    folium.Circle(radius=0.2, location=[truth_gt[index, 0], truth_gt[index, 1]],popup=f'Ground Truth:{index},utc:{utc}',
                                  color='yellow', fill=False).add_to(m)
                    folium.Circle(radius=0.2, location=[gnssins_pd[index, 0], gnssins_pd[index, 1]],popup=f'GNSS/INS:{index},utc:{utc}',
                                  color='cyan', fill=False).add_to(m)
            m.save(f"{dirname}/{drive}.html")

        # S9: 保存GNSS、IMU数据、导航结果
        """
        gnss数据列：['UnixTimeMillis_ref', 'latitude(deg)', 'longitude(deg)', 'height(m)','sdn(m)', 'sde(m)','sdu(m)']
        imu数据列：imu_column = ['UnixTimeMillis_ref', ' Angular rate X (rad/s)', ' Angular rate Y (rad/s)', ' Angular rate Z (rad/s)', ' Acceleration X (m/s^2)', ' Acceleration Y (m/s^2)', ' Acceleration Z (m/s^2)']
        """
        gnss_data_dic[tripID] = gnssfile.data_
        imu_data_dic[tripID] = imufile.data_
        data_truth_dic[tripID] = record_df

        # S10：特征、误差统计
        traj_stats = {
            'Traj_ID': tripID,
            'nav_pos_RMSE': nav_pos_err,
            'Satnum_max': max(gnssfile.data_[:,7]),
            'Satnum_min': min(gnssfile.data_[:,7]),
            'CNR_Mean_max': max(gnssfile.data_[:, 8]),
            'CNR_Mean_min': min(gnssfile.data_[:, 8]),
            'CNR_Q75_max': max(gnssfile.data_[:, 9]),
            'CNR_Q75_min': min(gnssfile.data_[:, 9]),
            'CNR_Q25_max': max(gnssfile.data_[:, 10]),
            'CNR_Q25_min': min(gnssfile.data_[:, 10]),
            'EA_Mean_max': max(gnssfile.data_[:, 11]),
            'EA_Mean_min': min(gnssfile.data_[:, 11]),
            'EA_Q75_max': max(gnssfile.data_[:, 12]),
            'EA_Q75_min': min(gnssfile.data_[:, 12]),
            'EA_Q25_max': max(gnssfile.data_[:, 13]),
            'EA_Q25_min': min(gnssfile.data_[:, 13]),
            'PDOP_max': max(gnssfile.data_[:, 14]),
            'PDOP_min': min(gnssfile.data_[:, 14]),
            'HDOP_max': max(gnssfile.data_[:, 15]),
            'HDOP_min': min(gnssfile.data_[:, 15]),
            'Q=1': np.mean(gnssfile.data_[:, 16] == 1),
            'Q=2': np.mean(gnssfile.data_[:, 16] == 2),
            'Q=4': np.mean(gnssfile.data_[:, 16] == 4),
            'Q=5': np.mean(gnssfile.data_[:, 16] == 5),
            'Q=0': np.mean(gnssfile.data_[:, 16] == 0),
            'Device': device_type,
        }
        trip_result.append(traj_stats)

    df_summary = pd.DataFrame(trip_result)

    # s11: 保存数据
    with open(src_dir + 'env/raw_baseline_gnssins_SZdata.pkl', 'wb') as value_file:
        pickle.dump(data_truth_dic, value_file, True)
    value_file.close()
    with open(src_dir + 'env/raw_gnss_data_SZdata.pkl', 'wb') as value_file:
        pickle.dump(gnss_data_dic, value_file, True)
    value_file.close()
    with open(src_dir + 'env/raw_imu_data_SZdata.pkl', 'wb') as value_file:
        pickle.dump(imu_data_dic, value_file, True)
    value_file.close()

    df_summary.to_csv(src_dir + 'env/raw_tripID_SZdata.csv', index=True)

    """
    gnss数据列：['UnixTimeMillis_ref', 'latitude(deg)', 'longitude(deg)', 'height(m)','sdn(m)', 'sde(m)','sdu(m)','Satnum',
                              'CNR_Mean','CNR_Q75','CNR_Q25','EA_Mean','EA_Q75','EA_Q25','PDOP','HDOP']
    imu数据列：['UnixTimeMillis_ref', ' Angular rate X (rad/s)', ' Angular rate Y (rad/s)',' Angular rate Z (rad/s)',
                          ' Acceleration X (m/s^2)', ' Acceleration Y (m/s^2)',' Acceleration Z (m/s^2)']
    """

