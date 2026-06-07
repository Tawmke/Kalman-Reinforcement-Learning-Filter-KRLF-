import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys
cur_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(cur_dir, '..')
# src_dir = os.path.join(cur_dir, '.')
sys.path.append(src_dir)
from funcs import *



dir_path = '/mnt/sdb/home/tangjh/KF-GINS-Py-main/records_values/'
plot_type = 'HK_De_ublox'
type_list = ['HK_Me_ublox']
kf_column = [' Latitude (deg)',' Longitude (deg)',' Ellipsoid Height (m)',' Velocity X (m/s)',' Velocity Y (m/s)',' Velocity Z (m/s)',' Roll (deg)',' Pitch (deg)',' Heading (deg)']
gt_column = [' Latitude_GT (deg)', ' Longitude_GT (deg)', ' Ellipsoid Height_GT (m)',
                  ' Velocity X_GT (m/s)', ' Velocity Y_GT (m/s)', ' Velocity Z_GT (m/s)',
                  ' Roll_GT (deg)', ' Pitch_GT (deg)', ' Heading_GT (deg)']
rl_colmun = ['Latitude_RLpredict','Longitude_RLpredict','Ellipsoid_Height_RLpredict','Velocity_X_RLpredict',
             'Velocity_Y_RLpredict','Velocity_Z_RLpredict','Roll_RLpredict','Pitch_RLpredict','Heading_RLpredict']

trajfolders = {}
if plot_type == 'HK_De_ublox':
    base_folder = dir_path + 'finetuning_4_testing_indomain_att_0407/HK_De_ublox_0.7_1_RMSEadv_ratio'
    trajfolders['HK_Me_ublox'] = 'lr=0.00_pos=6_SP=1.95_PC=0.0020_AP=0.00_aw=0.49/RecurrentPPO_1'
    trajfile = 'testmore_HK_Me_ublox_rl_traj_1_UrbanNav-HK-Medium-Urban-1_ublox_f9p_splitter.csv'

for type in type_list:
    pd_train = pd.read_csv(f'{base_folder}/{trajfolders[type]}/{trajfile}')
    navresult_gt = pd_train[gt_column].to_numpy()
    navresult_gt[:, [5, 6]] = navresult_gt[:, [6, 5]] # 速度ENU转成NED下
    navresult_gt[:, [7]] = -navresult_gt[:, [7]]
    navresult_kf = pd_train[kf_column].to_numpy()
    navresult_rl = pd_train[rl_colmun].to_numpy()

    # 角度制转弧度制
    navresult_gt[:, 0:2] = navresult_gt[:, 0:2] * D2R
    navresult_kf[:, 0:2] = navresult_kf[:, 0:2] * D2R
    navresult_rl[:, 0:2] = navresult_rl[:, 0:2] * D2R

    # 计算导航误差
    rl_error = navresult_rl - navresult_gt
    kf_error = navresult_kf - navresult_gt

    # 航向角误差处理
    for i in range(len(rl_error)):
        if rl_error[i, 8] > 180:
            rl_error[i, 8] -= 360
        if rl_error[i, 8] < -180:
            rl_error[i, 8] += 360
        if kf_error[i, 8] > 180:
            kf_error[i, 8] -= 360
        if kf_error[i, 8] < -180:
            kf_error[i, 8] += 360

    # 位置误差转到第一个位置确定的n系
    blh_station = navresult_kf[0, 2:5]
    rm, rn = radiusmn(blh_station[0])
    for i in range(len(rl_error)):
        rl_error[i, 0:3] = drad2dm(rm, rn, blh_station, rl_error[i, 0:3]).reshape(1, 3)
        kf_error[i, 0:3] = drad2dm(rm, rn, blh_station, kf_error[i, 0:3]).reshape(1, 3)

    # 绘制误差
    plt.figure('GNSS/INS position error')
    plt.plot(kf_error[:, 0:3], linestyle='-')
    plt.plot(rl_error[:, 0:3], linestyle='--')
    plt.legend([f'ESKF (N):{np.mean(np.abs(kf_error[:,0])):.2f}m', f'ESKF (E):{np.mean(np.abs(kf_error[:,1])):.2f}m',
                f'ESKF (D):{np.mean(np.abs(kf_error[:,2])):.2f}m',f'KRLF (N):{np.mean(np.abs(rl_error[:,0])):.2f}m', f'KRLF (E):{np.mean(np.abs(rl_error[:,1])):.2f}m',
                f'KRLF (D):{np.mean(np.abs(rl_error[:,2])):.2f}m'])
    plt.xlabel('Time [s]')
    plt.ylabel('Error [m]')
    plt.title(f'Position Error')
    plt.grid()
    # plt.ylim([-50,50])
    plt.tight_layout()

    plt.show()