import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import os
import sys
import numpy as np
cur_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(cur_dir, '..')
# src_dir = os.path.join(cur_dir, '.')
sys.path.append(src_dir)
#from funcs.funcs_KF import *
import pandas as pd
# from env.dummy_cec_env_custom import *
import gym
from stable_baselines3.common.vec_env import DummyVecEnv
#from model.ppo_recurrent import RecurrentPPO
from sb3_contrib import RecurrentPPO
# from env.env_param import *
# from funcs.PPO_SR import *
from collections import deque
from funcs import *
import time
import math

def pos_error_enu(llh_gt,llh_pre,blh_station):
    naverror = llh_pre - llh_gt
    rm, rn = radiusmn(blh_station[0])
    for i in range(len(naverror)):
        naverror[i,:] = drad2dm(rm, rn, blh_station, naverror[i,:]).reshape(1, 3)
    return np.abs(naverror)

def att_error_enu(att_gt,att_kf):
    # 只统计yaw角度误差
    naverror = att_gt - att_kf
    for i in range(len(naverror)):
        if naverror[i, 2] > 180:
            naverror[i, 2] -= 360
        if naverror[i, 2] < -180:
            naverror[i, 2] += 360

    return np.abs(naverror)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
dir_path = '/mnt/sdb/home/tangjh/KF-GINS-Py-main'
plot_path = '/mnt/sdb/home/tangjh/GSDC2023/plots/RLAKF-IMU'

plot_type = 'HK_Ha_ublox' # smurban pixel567urban
method_list = ['KRLF','KRLF']
RLmethod_list = ['KRLF']
DNNmethod_list = ['MTAKF','CNNLSTM']
linecolor_dic = {'ESKF':'k','KRLF': 'r','MTAKF':'b','GT':'g','GRU-SAC':'c','Multi-LSTMPPO':'m','RL-AKF':'m'}
linestyle_dic = {'ESKF':'-','KRLF': '--','MTAKF':'-.','CNNLSTM':(0,(3,1,1,1)),'GRU-SAC':(0,(3,1,1,1)),'Multi-LSTMPPO':':','RL-AKF':(0,(3,1,5,4))}
xy_fontsize = 22
linewidth = 3.5
patch_width = 1.5

loggerdir = {}
trajdataname = {}
trajfilename_dic = {}
if plot_type == 'HK_Ha_ublox':
    trajfilename_dic['e'] = '3_UrbanNav-HK-Harsh-Urban-1_ublox_f9p_splitter.csv'
    # 2022-10-06-21-51-us-ca-mtv-n_sm-a217m 2023-05-23-19-16-us-ca-mtv-ie2_sm-s908b
    trajfilename_dic['n'] = '3_UrbanNav-HK-Harsh-Urban-1_ublox_f9p.csv' # 2023-05-23-19-56-us-ca-mtv-ie2_sm-a600t
    trajfilename_dic['u'] = '3_UrbanNav-HK-Harsh-Urban-1_ublox_f9p.csv'
    loggerdir = {'KRLF':dir_path + '/records_values/tuning_4_testing_indomain/HK_Ha_ublox_0.7_1_RMSEadv_ratio_InHiGNSSPosAtt_PreCovPos/'
                                            'lr=0.0028_pos=7_SP=2.49_PC=0.0016_Reset=1_IR=23.349129246018737'}

    trajdataname['KRLF'] = 'testmore_HK_Ha_ublox_rl_traj_'
    trajdataname['GRU-SAC'] = 'testmore_smurban_rl_traj_'
    trajdataname['RL-AKF'] = 'testmore_smurban_rl_traj_'
    trajdataname['MTAKF'] = 'Pred_traj_'
    trajdataname['CNNLSTM'] = 'Pred_traj_'
    ylimlist = [0, 40]
    axplot = False

elif plot_type == 'HK_Ha_phone':
    trajfilename_dic['e'] = '3_UrbanNav-HK-Harsh-Urban-1_google_pixel4.csv'
    # 2022-10-06-21-51-us-ca-mtv-n_sm-a217m 2023-05-23-19-16-us-ca-mtv-ie2_sm-s908b
    trajfilename_dic['n'] = '3_UrbanNav-HK-Harsh-Urban-1_google_pixel4.csv' # 2023-05-23-19-56-us-ca-mtv-ie2_sm-a600t
    trajfilename_dic['u'] = '3_UrbanNav-HK-Harsh-Urban-1_google_pixel4.csv'
    loggerdir = {'KRLF':dir_path + '/records_values/finetuning_4_testing_indomain_att_0407/HK_Ha_phone_1_0.5_RMSEadv_ratio_InHiGNSSStaCov_PrePosAttCov/'
                                   'lr=0.0001_pos=7_SP=2.86_PC=0.0014_Reset=1_IR=14.749830545041823'}

    trajdataname['KRLF'] = 'testmore_HK_Ha_phone_rl_traj_'
    trajdataname['GRU-SAC'] = 'testmore_HK_Ha_phone_rl_traj_'
    trajdataname['RL-AKF'] = 'testmore_HK_Ha_phone_rl_traj_'
    trajdataname['MTAKF'] = 'Pred_traj_'
    trajdataname['CNNLSTM'] = 'Pred_traj_'
    ylimlist = [0, 40]
    axplot = False


plt.rc('axes', linewidth=patch_width)
directions = ['e', 'n', 'u']  # 遍历三个方向
index_dic = {'e': 0, 'n': 1, 'u': 2}
direction_name = {'e': 'East', 'n': 'North', 'u': 'Up'}
for dir in directions:
    # more tests
    check_bl_plot = True
    index = index_dic[dir]
    trajfilename = trajfilename_dic[dir]
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    if axplot:
        axins = ax.inset_axes((0.1, 0.35, 0.25, 0.5))
    #plt.figure(figsize=(14, 5))
    for method in method_list:
        folders = os.listdir(loggerdir[method])
        folders.sort()
        diserr_list = []
        # if dir == 'e' and method == 'R-KalmanAgent':
        #     folders = ['R_RecurrentPPO_2_Q_RecurrentPPO_2']
        for folder in folders:
            if ('txt' in folder) or ('csv' in folder):
                continue
            if method in RLmethod_list:
                pd_train = pd.read_csv(loggerdir[method]+'/'+folder+'/'+trajdataname[method]+trajfilename)
                llh_gt = pd_train[[' Latitude_GT (deg)',' Longitude_GT (deg)',' Ellipsoid Height_GT (m)']].to_numpy()
                llh_kf = pd_train[[' Latitude (deg)',' Longitude (deg)',' Ellipsoid Height (m)']].to_numpy()
                llh_pre = pd_train[['Latitude_RLpredict','Longitude_RLpredict','Ellipsoid_Height_RLpredict']].to_numpy()
                blh_station = llh_gt[0,:]
                llh_pre[:, 0:2] *= D2R
                llh_gt[:, 0:2] *= D2R
                llh_kf[:, 0:2] *= D2R
                pos_result_kf = pos_error_enu(llh_gt,llh_kf,blh_station)
                pos_err_kf = pos_result_kf[:,index]
                pos_result_pre = pos_error_enu(llh_gt,llh_pre,blh_station)
                diserr_list.append(pos_result_pre[:,index])
            elif method in DNNmethod_list:
                pd_train = pd.read_csv(loggerdir[method] + '/' + folder + '/' + trajdataname[method] + trajfilename)
                xyz_gt = pd_train[['E_gt','N_gt','U_gt']].to_numpy()
                xyz_kf = pd_train[['E_eskf_hfv2','N_eskf_hfv2','U_eskf_hfv2']].to_numpy()
                xyz_pre = pd_train[['E_predict','N_predict','U_predict']].to_numpy()
                disterr_kf= np.sqrt((xyz_kf[:,index]-xyz_gt[:,index])**2)
                disterr_pre= np.sqrt((xyz_pre[:,index]-xyz_gt[:,index])**2)
                diserr_list.append(disterr_pre)

        if check_bl_plot:
            check_bl_plot = False
            plt.plot(pos_err_kf, label=f'ESKF, RMSE: {np.nanmean(pos_err_kf):.2f} m',linewidth=linewidth, color=linecolor_dic['ESKF'], linestyle=linestyle_dic['ESKF'],alpha=0.6)
            if axplot:
                axins.plot(pos_err_kf, label=f'ESKF, RMSE: {np.nanmean(pos_err_kf):.2f} m',linewidth=linewidth, color=linecolor_dic['ESKF'], linestyle=linestyle_dic['ESKF'],alpha=0.6)
            continue

        try:
            diserr = np.mean(diserr_list, axis=0)
        except:
            len_min = 1000000
            idx_err_list = []
            for idx, diserr in enumerate(diserr_list):
                len_traj = len(diserr)
                if len_traj < len_min:
                    len_min = len_traj
            for idx, diserr in enumerate(diserr_list):
                if len(diserr) > len_min:
                    diserr_list[idx] = diserr[0:len_min]
            diserr = np.nanmean(diserr_list, axis=0)
        # Plot distance error
        plt.plot(diserr, label=f'{method}, RMSE: {np.nanmean(diserr):.2f} m', linewidth=linewidth, color=linecolor_dic[method], linestyle=linestyle_dic[method], alpha=0.6)
        print(f'Data len={len(diserr)}')
        if axplot:
            axins.plot(diserr, label=f'{method}, RMSE: {np.nanmean(diserr):.2f} m', linewidth=linewidth, color=linecolor_dic[method], linestyle=linestyle_dic[method], alpha=0.6)
        # plt.title('Distance error')

    if axplot:
        # 调整子坐标系的显示范围
        axins.set_xlim(xlim0, xlim1)
        axins.set_ylim(ylim0, ylim1)
        axins.tick_params(axis='x', labelsize=xy_fontsize -1 )
        axins.tick_params(axis='y', labelsize=xy_fontsize -1 )
        mark_inset(ax, axins, loc1=3, loc2=2, fc="none", ec='k', lw=1.5)
    plt.xlabel('Epoch', fontsize=xy_fontsize+5)
    # if dir == 'e':
    plt.ylabel(f'{direction_name[dir]} Error (m)', fontsize=xy_fontsize+5)
    plt.yticks(fontsize=xy_fontsize+4)
    plt.xticks(fontsize=xy_fontsize+4)
    plt.legend(fontsize=xy_fontsize-3)
    #plt.ylim(ylimlist)
    plt.grid()
    # plt.title('Positioning error in {}'.format(plot_type),fontsize=xy_fontsize)
    #y_max=np.nanmax([disterr_kf,disterr_rl,disterr_wls])
    # plt.yscale('log')
    # plt.savefig(plot_path + f'/figs/GSDC2023_{dir}_err_{plot_type}.pdf', bbox_inches='tight')
    plt.show()

plt.rc('axes', linewidth=patch_width)
directions = ['e', 'n', 'u']  # 遍历三个方向
index_dic = {'e': 0, 'n': 1, 'u': 2}
direction_name = {'e': 'roll', 'n': 'pitch', 'u': 'yaw'}
for dir in directions:
    # more tests
    check_bl_plot = True
    index = index_dic[dir]
    trajfilename = trajfilename_dic[dir]
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    if axplot:
        axins = ax.inset_axes((0.1, 0.35, 0.25, 0.5))
    #plt.figure(figsize=(14, 5))
    for method in method_list:
        folders = os.listdir(loggerdir[method])
        folders.sort()
        diserr_list = []
        # if dir == 'e' and method == 'R-KalmanAgent':
        #     folders = ['R_RecurrentPPO_2_Q_RecurrentPPO_2']
        for folder in folders:
            if ('txt' in folder) or ('csv' in folder):
                continue
            if method in RLmethod_list:
                pd_train = pd.read_csv(loggerdir[method]+'/'+folder+'/'+trajdataname[method]+trajfilename)
                att_gt = pd_train[[' Roll_GT (deg)', ' Pitch_GT (deg)', ' Heading_GT (deg)']].to_numpy()
                att_kf = pd_train[[' Roll (deg)', ' Pitch (deg)', ' Heading (deg)']].to_numpy()
                att_pre = pd_train[['Roll_RLpredict', 'Pitch_RLpredict','Heading_RLpredict']].to_numpy()
                att_err_result_kf = att_error_enu(att_gt,att_kf)
                att_err_kf = att_err_result_kf[:,index]
                att_err_pre = att_error_enu(att_gt,att_pre)
                diserr_list.append(att_err_pre[:,index])
            elif method in DNNmethod_list:
                pd_train = pd.read_csv(loggerdir[method] + '/' + folder + '/' + trajdataname[method] + trajfilename)
                xyz_gt = pd_train[['E_gt','N_gt','U_gt']].to_numpy()
                xyz_kf = pd_train[['E_eskf_hfv2','N_eskf_hfv2','U_eskf_hfv2']].to_numpy()
                xyz_pre = pd_train[['E_predict','N_predict','U_predict']].to_numpy()
                disterr_kf= np.sqrt((xyz_kf[:,index]-xyz_gt[:,index])**2)
                disterr_pre= np.sqrt((xyz_pre[:,index]-xyz_gt[:,index])**2)
                diserr_list.append(disterr_pre)

        if check_bl_plot:
            check_bl_plot = False
            plt.plot(att_err_kf, label=f'ESKF, RMSE: {np.nanmean(att_err_kf):.2f} degree',linewidth=linewidth, color=linecolor_dic['ESKF'], linestyle=linestyle_dic['ESKF'],alpha=0.6)
            if axplot:
                axins.plot(att_err_kf, label=f'ESKF, RMSE: {np.nanmean(att_err_kf):.2f} m',linewidth=linewidth, color=linecolor_dic['ESKF'], linestyle=linestyle_dic['ESKF'],alpha=0.6)
            continue

        try:
            diserr = np.mean(diserr_list, axis=0)
        except:
            len_min = 1000000
            idx_err_list = []
            for idx, diserr in enumerate(diserr_list):
                len_traj = len(diserr)
                if len_traj < len_min:
                    len_min = len_traj
            for idx, diserr in enumerate(diserr_list):
                if len(diserr) > len_min:
                    diserr_list[idx] = diserr[0:len_min]
            diserr = np.nanmean(diserr_list, axis=0)
        # Plot distance error
        plt.plot(diserr, label=f'{method}, RMSE: {np.nanmean(diserr):.2f} degree', linewidth=linewidth, color=linecolor_dic[method], linestyle=linestyle_dic[method], alpha=0.6)
        if axplot:
            axins.plot(diserr, label=f'{method}, RMSE: {np.nanmean(diserr):.2f} degree', linewidth=linewidth, color=linecolor_dic[method], linestyle=linestyle_dic[method], alpha=0.6)
        # plt.title('Distance error')

    if axplot:
        # 调整子坐标系的显示范围
        axins.set_xlim(xlim0, xlim1)
        axins.set_ylim(ylim0, ylim1)
        axins.tick_params(axis='x', labelsize=xy_fontsize -1 )
        axins.tick_params(axis='y', labelsize=xy_fontsize -1 )
        mark_inset(ax, axins, loc1=3, loc2=2, fc="none", ec='k', lw=1.5)
    plt.xlabel('Epoch', fontsize=xy_fontsize+5)
    # if dir == 'e':
    plt.ylabel(f'{direction_name[dir]} Error (degree)', fontsize=xy_fontsize+5)
    plt.yticks(fontsize=xy_fontsize+4)
    plt.xticks(fontsize=xy_fontsize+4)
    plt.legend(fontsize=xy_fontsize-3)
    #plt.ylim(ylimlist)
    plt.grid()
    # plt.title('Positioning error in {}'.format(plot_type),fontsize=xy_fontsize)
    #y_max=np.nanmax([disterr_kf,disterr_rl,disterr_wls])
    # plt.yscale('log')
    # plt.savefig(plot_path + f'/figs/GSDC2023_{dir}_err_{plot_type}.pdf', bbox_inches='tight')
    plt.show()

# 绘制位置变化
plt.rc('axes', linewidth=patch_width)
directions = ['e', 'n', 'u']  # 遍历三个方向
index_dic = {'e': 0, 'n': 1, 'u': 2}
direction_name = {'e': 'roll', 'n': 'pitch', 'u': 'yaw'}
for dir in directions:
    # more tests
    check_bl_plot = True
    index = index_dic[dir]
    trajfilename = trajfilename_dic[dir]
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    if axplot:
        axins = ax.inset_axes((0.1, 0.35, 0.25, 0.5))
    #plt.figure(figsize=(14, 5))
    for method in method_list:
        folders = os.listdir(loggerdir[method])
        folders.sort()
        diserr_list = []
        # if dir == 'e' and method == 'R-KalmanAgent':
        #     folders = ['R_RecurrentPPO_2_Q_RecurrentPPO_2']
        for folder in folders:
            if ('txt' in folder) or ('csv' in folder):
                continue
            if method in RLmethod_list:
                pd_train = pd.read_csv(loggerdir[method]+'/'+folder+'/'+trajdataname[method]+trajfilename)
                llh_gt = pd_train[[' Latitude_GT (deg)', ' Longitude_GT (deg)', ' Ellipsoid Height_GT (m)']].to_numpy()
                llh_kf = pd_train[[' Latitude (deg)', ' Longitude (deg)', ' Ellipsoid Height (m)']].to_numpy()
                llh_pre = pd_train[['Latitude_RLpredict', 'Longitude_RLpredict', 'Ellipsoid_Height_RLpredict']].to_numpy()
                llh_kf_dir = llh_kf[:,index]
                llh_gt_dir = llh_gt[:,index]
                diserr_list.append(llh_pre[:,index])
            elif method in DNNmethod_list:
                pd_train = pd.read_csv(loggerdir[method] + '/' + folder + '/' + trajdataname[method] + trajfilename)
                xyz_gt = pd_train[['E_gt','N_gt','U_gt']].to_numpy()
                xyz_kf = pd_train[['E_eskf_hfv2','N_eskf_hfv2','U_eskf_hfv2']].to_numpy()
                xyz_pre = pd_train[['E_predict','N_predict','U_predict']].to_numpy()
                disterr_kf= np.sqrt((xyz_kf[:,index]-xyz_gt[:,index])**2)
                disterr_pre= np.sqrt((xyz_pre[:,index]-xyz_gt[:,index])**2)
                diserr_list.append(disterr_pre)

        if check_bl_plot:
            check_bl_plot = False
            plt.plot(llh_kf_dir, label=f'ESKF',linewidth=linewidth, color=linecolor_dic['ESKF'], linestyle=linestyle_dic['ESKF'],alpha=0.6)
            plt.plot(llh_gt_dir, label=f'GT',linewidth=linewidth, color=linecolor_dic['GT'], linestyle=linestyle_dic['ESKF'],alpha=0.6)
            if axplot:
                axins.plot(att_err_kf, label=f'ESKF, RMSE: {np.nanmean(att_err_kf):.2f} m',linewidth=linewidth, color=linecolor_dic['ESKF'], linestyle=linestyle_dic['ESKF'],alpha=0.6)
            continue

        try:
            diserr = np.mean(diserr_list, axis=0)
        except:
            len_min = 1000000
            idx_err_list = []
            for idx, diserr in enumerate(diserr_list):
                len_traj = len(diserr)
                if len_traj < len_min:
                    len_min = len_traj
            for idx, diserr in enumerate(diserr_list):
                if len(diserr) > len_min:
                    diserr_list[idx] = diserr[0:len_min]
            diserr = np.nanmean(diserr_list, axis=0)
        # Plot distance error
        plt.plot(diserr, label=f'{method}', linewidth=linewidth, color=linecolor_dic[method], linestyle=linestyle_dic[method], alpha=0.6)
        if axplot:
            axins.plot(diserr, label=f'{method}, RMSE: {np.nanmean(diserr):.2f} degree', linewidth=linewidth, color=linecolor_dic[method], linestyle=linestyle_dic[method], alpha=0.6)
        # plt.title('Distance error')

    if axplot:
        # 调整子坐标系的显示范围
        axins.set_xlim(xlim0, xlim1)
        axins.set_ylim(ylim0, ylim1)
        axins.tick_params(axis='x', labelsize=xy_fontsize -1 )
        axins.tick_params(axis='y', labelsize=xy_fontsize -1 )
        mark_inset(ax, axins, loc1=3, loc2=2, fc="none", ec='k', lw=1.5)
    plt.xlabel('Epoch', fontsize=xy_fontsize+5)
    # if dir == 'e':
    plt.ylabel(f'{direction_name[dir]} (degree)', fontsize=xy_fontsize+5)
    plt.yticks(fontsize=xy_fontsize+4)
    plt.xticks(fontsize=xy_fontsize+4)
    plt.legend(fontsize=xy_fontsize-3)
    #plt.ylim(ylimlist)
    plt.grid()
    # plt.title('Positioning error in {}'.format(plot_type),fontsize=xy_fontsize)
    #y_max=np.nanmax([disterr_kf,disterr_rl,disterr_wls])
    # plt.yscale('log')
    # plt.savefig(plot_path + f'/figs/GSDC2023_{dir}_err_{plot_type}.pdf', bbox_inches='tight')
    plt.show()

# 绘制姿态变化
plt.rc('axes', linewidth=patch_width)
directions = ['e', 'n', 'u']  # 遍历三个方向
index_dic = {'e': 0, 'n': 1, 'u': 2}
direction_name = {'e': 'roll', 'n': 'pitch', 'u': 'yaw'}
for dir in directions:
    # more tests
    check_bl_plot = True
    index = index_dic[dir]
    trajfilename = trajfilename_dic[dir]
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    if axplot:
        axins = ax.inset_axes((0.1, 0.35, 0.25, 0.5))
    #plt.figure(figsize=(14, 5))
    for method in method_list:
        folders = os.listdir(loggerdir[method])
        folders.sort()
        diserr_list = []
        # if dir == 'e' and method == 'R-KalmanAgent':
        #     folders = ['R_RecurrentPPO_2_Q_RecurrentPPO_2']
        for folder in folders:
            if ('txt' in folder) or ('csv' in folder):
                continue
            if method in RLmethod_list:
                pd_train = pd.read_csv(loggerdir[method]+'/'+folder+'/'+trajdataname[method]+trajfilename)
                att_gt = pd_train[[' Roll_GT (deg)', ' Pitch_GT (deg)', ' Heading_GT (deg)']].to_numpy()
                att_kf = pd_train[[' Roll (deg)', ' Pitch (deg)', ' Heading (deg)']].to_numpy()
                att_pre = pd_train[['Roll_RLpredict', 'Pitch_RLpredict','Heading_RLpredict']].to_numpy()
                att_err_kf = att_kf[:,index]
                att_err_gt = att_gt[:,index]
                diserr_list.append(att_pre[:,index])
            elif method in DNNmethod_list:
                pd_train = pd.read_csv(loggerdir[method] + '/' + folder + '/' + trajdataname[method] + trajfilename)
                xyz_gt = pd_train[['E_gt','N_gt','U_gt']].to_numpy()
                xyz_kf = pd_train[['E_eskf_hfv2','N_eskf_hfv2','U_eskf_hfv2']].to_numpy()
                xyz_pre = pd_train[['E_predict','N_predict','U_predict']].to_numpy()
                disterr_kf= np.sqrt((xyz_kf[:,index]-xyz_gt[:,index])**2)
                disterr_pre= np.sqrt((xyz_pre[:,index]-xyz_gt[:,index])**2)
                diserr_list.append(disterr_pre)

        if check_bl_plot:
            check_bl_plot = False
            plt.plot(att_err_kf, label=f'ESKF',linewidth=linewidth, color=linecolor_dic['ESKF'], linestyle=linestyle_dic['ESKF'],alpha=0.6)
            plt.plot(att_err_gt, label=f'GT',linewidth=linewidth, color=linecolor_dic['GT'], linestyle=linestyle_dic['ESKF'],alpha=0.6)
            if axplot:
                axins.plot(att_err_kf, label=f'ESKF, RMSE: {np.nanmean(att_err_kf):.2f} m',linewidth=linewidth, color=linecolor_dic['ESKF'], linestyle=linestyle_dic['ESKF'],alpha=0.6)
            continue

        try:
            diserr = np.mean(diserr_list, axis=0)
        except:
            len_min = 1000000
            idx_err_list = []
            for idx, diserr in enumerate(diserr_list):
                len_traj = len(diserr)
                if len_traj < len_min:
                    len_min = len_traj
            for idx, diserr in enumerate(diserr_list):
                if len(diserr) > len_min:
                    diserr_list[idx] = diserr[0:len_min]
            diserr = np.nanmean(diserr_list, axis=0)
        # Plot distance error
        plt.plot(diserr, label=f'{method}', linewidth=linewidth, color=linecolor_dic[method], linestyle=linestyle_dic[method], alpha=0.6)
        if axplot:
            axins.plot(diserr, label=f'{method}, RMSE: {np.nanmean(diserr):.2f} degree', linewidth=linewidth, color=linecolor_dic[method], linestyle=linestyle_dic[method], alpha=0.6)
        # plt.title('Distance error')

    if axplot:
        # 调整子坐标系的显示范围
        axins.set_xlim(xlim0, xlim1)
        axins.set_ylim(ylim0, ylim1)
        axins.tick_params(axis='x', labelsize=xy_fontsize -1 )
        axins.tick_params(axis='y', labelsize=xy_fontsize -1 )
        mark_inset(ax, axins, loc1=3, loc2=2, fc="none", ec='k', lw=1.5)
    plt.xlabel('Epoch', fontsize=xy_fontsize+5)
    # if dir == 'e':
    plt.ylabel(f'{direction_name[dir]} (degree)', fontsize=xy_fontsize+5)
    plt.yticks(fontsize=xy_fontsize+4)
    plt.xticks(fontsize=xy_fontsize+4)
    plt.legend(fontsize=xy_fontsize-3)
    #plt.ylim(ylimlist)
    plt.grid()
    # plt.title('Positioning error in {}'.format(plot_type),fontsize=xy_fontsize)
    #y_max=np.nanmax([disterr_kf,disterr_rl,disterr_wls])
    # plt.yscale('log')
    # plt.savefig(plot_path + f'/figs/GSDC2023_{dir}_err_{plot_type}.pdf', bbox_inches='tight')
    plt.show()

fig = plt.figure(figsize=(11, 4))
import matplotlib.lines as mlines
ax = fig.add_subplot(111)
# 创建线条以代表图例内容
line_list = []
check = True
method_list[0] = 'ESKF'
for method in method_list:
    line = mlines.Line2D([], [], label=method, linestyle=linestyle_dic[method], color=linecolor_dic[method],linewidth=2)
    line_list.append(line)
# 添加图例
ax.legend(handles=line_list, fontsize=xy_fontsize, ncol=len(method_list))
ax.axis('off')  # 关闭坐标轴
plt.savefig(plot_path + f'/figs/GSDC2023_diserr_legend.png', bbox_inches='tight')
plt.show()
print('finish')

    # errmax = 100
    # plt.figure(figsize=(6, 4))
    # dis_err = vd_rtk
    # dis_err = np.where(dis_err > errmax, errmax, dis_err)
    # binnum = int(np.ceil(max(dis_err)))
    # plt.hist(dis_err, density=True, bins=binnum, rwidth=0.9, alpha=0.5,
    #          label='RTK, average error={:.3}m'.format(np.mean(dis_err)))
    # dis_err = vd_rl
    # dis_err = np.where(dis_err > errmax, errmax, dis_err)
    # binnum = int(np.ceil(max(dis_err)))
    # plt.hist(dis_err, density=True, bins=binnum, rwidth=0.9, alpha=0.5,
    #          label='LSTMPPO-ARAM, average error={:.3}m'.format(np.mean(dis_err)))
    # plt.grid()
    # plt.legend(fontsize=12)
    # plt.title(f'Distance error distribution of ll in {key}', fontsize=12)
    # plt.xlabel('Distance error (m)', fontsize=12)
    # plt.ylabel('Proportion', fontsize=12)
    # plt.show()
    # # plt.savefig('./figs/traj/disterr_{}.png'.format(tripIDname), bbox_inches='tight')
    # plt.savefig('{}plots/Gnss+/figs/disterr_{}_{}_{}.pdf'.format(dir_path,traj_type,test_domain,trajdata_num), bbox_inches='tight')
