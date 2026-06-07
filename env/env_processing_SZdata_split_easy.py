import gym
from gym import spaces
import random
import pickle
import os
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import glob as gl
from env.env_param import *
from scipy.spatial import distance
import matplotlib.pyplot as plt
from data_params import *
# from funcs.utilis import *
import pymap3d.vincenty as pmv

"""
该代码用于按场景分割深圳数据
创建人：唐健浩
"""
def dist_err_XYZ(true_XYZ, guess_XYZ_bl):
    tmp = true_XYZ - guess_XYZ_bl
    dist_err=np.sqrt(np.sum(tmp**2,axis=0))
    return dist_err

def vincenty_distance(llh1, llh2):
    """
    Args:
        llh1 : [latitude,longitude] (deg)
        llh2 : [latitude,longitude] (deg)
    Returns:
        d : distance between llh1 and llh2 (m)
    """
    d, az = np.array(pmv.vdist(llh1[:, 0], llh1[:, 1], llh2[:, 0], llh2[:, 1]))

    return d

#########
save_file = True # 是否保存数据
plot_dis = True
interrupt_time_thread = 50 # 中断统计时间，中断大于该时间截断为新轨迹
continue_time = 20 # 连续时间小于该参数的轨迹剔除(只针对连续情况)
labelsize = 20

######### 导入预处理数据
dir_path = '/mnt/sdb/home/tangjh/KF-GINS-Py-main/' # '/home/tangjh/smartphone-decimeter-2022/''D:/jianhao/smartphone-decimeter-2022/'
with open(dir_path+'env/raw_baseline_gnssins_SZdata.pkl', "rb") as file:
    data_truth_dic = pickle.load(file)
file.close()
# file.close()
with open(dir_path + 'env/raw_gnss_data_SZdata.pkl', "rb") as file:
    gnss_data_dic = pickle.load(file)
file.close()

########## 处理文件
trajtypelist = ['openroad','canyon','overpass','forest']
########## 统计数据
disll_type = {}
dish_type = {}
disxyz_type = {}
datatype = 'SZdata'
for type in trajtypelist:
    disll_type[type] = []
    dish_type[type] = []
    disxyz_type[type] = []

plt_folder = dir_path + f'/env/{datatype}'
if not os.path.exists(plt_folder):
    os.makedirs(plt_folder)

########## 场景划分
# load_data = True
# if load_data:
#     with open(dir_path + '/env/raw_baseline_Haige_new.pkl', "rb") as file:
#         data_truth_dic_split = pickle.load(file)
#     file.close()
#     with open(dir_path + '/env/processed_features_BDS_Haige_new_interupt.pkl', "rb") as file:
#         losfeature_split = pickle.load(file)
#     file.close()
#
#     traj_sum_df = pd.read_csv(f'{dir_path}/env/raw_tripID_BDS_Haige_new_interupt.csv')
#     tripIDlist = traj_sum_df['tripID'].values.tolist()
#     triptypelist = traj_sum_df['Type'].values.tolist()
# else:
data_truth_dic_split = {}
tripIDlist = []
triptypelist = []

for tripID,baseline in data_truth_dic.items():
    # tripID = '20260130-AM'
    # baseline = data_truth_dic[tripID]
    # baseline = baseline.reset_index(drop=True)

    for datetime in trajtype_range_dic.keys():
        if datetime in tripID:
            range_dic = trajtype_range_dic[datetime]
    # 统计中断
    interrupt_time_all = 0
    interrupt_distance_all = 0
    reftimes_org = np.array(baseline['UnixTimeMillis_ref'])  # utc时间
    for type,typerange in range_dic.items():
        if f'{tripID}-{type}' in exclude_data: # 剔除异常数据
            print(f'-------Dont Processing {type} in {tripID}--------')
            continue

        print(f'------Processing {type} in {tripID}---------')
        idx = np.where((typerange[0] <= reftimes_org) & (typerange[1] >= reftimes_org)) # 获取范围内索引
        if idx[0].size == 0:
            print(f'No data: {type} in {tripID} !!!')
            continue
        else:
            trajidx = range(int(idx[0][0]), int(idx[0][-1]))

        ################## 统计中断
        count = 0
        interrupt_UTCtimes_list = []
        for index in trajidx:
            if index == int(idx[0][0]):
                timestep_start = baseline['UnixTimeMillis_ref'][index] # start of the steptime
                prepos = np.array([baseline[' Latitude (deg)'][index],baseline[' Longitude (deg)'][index],
                                   baseline[' Ellipsoid Height (m)'][index]])
                continue
            curpos = np.array([baseline[' Latitude (deg)'][index],baseline[' Longitude (deg)'][index],baseline[' Ellipsoid Height (m)'][index]])
            dist_ll,_ = pmv.vdist(prepos[0], prepos[1], curpos[0], curpos[1])
            timestep_pre = baseline['UnixTimeMillis_ref'][index - 1]
            timestep_cur = baseline['UnixTimeMillis_ref'][index]
            interrupt_time = timestep_cur - timestep_pre
            if interrupt_time > interrupt_time_thread:
                count += 1
                if (timestep_pre - timestep_start) > continue_time:
                    interrupt_UTCtimes_list.append([timestep_start, timestep_pre])
                timestep_start = timestep_cur
                interrupt_time_all += interrupt_time
                interrupt_distance_all += dist_ll
                print(f'Count={count}, interrupt_dis={dist_ll:.2},'
                    f'interrupt_times={timestep_cur}-{timestep_pre}={interrupt_time}s')
            # 最后的idx，还要加上最后一段
            if index == int(idx[0][-1])-1:
                interrupt_UTCtimes_list.append([timestep_start, timestep_cur+1])
            prepos = curpos

        # Interrupt sequence trajectory dictionary
        for count, interrupt_UTCtime in enumerate(interrupt_UTCtimes_list):
            idx = np.where(
                (interrupt_UTCtime[0] <= reftimes_org) & (interrupt_UTCtime[1] >= reftimes_org))
            if idx[0][-1] - idx[0][0] > 1:
                balineline_split = baseline.iloc[idx]
                if len(balineline_split) < continue_time:
                    continue

                data_truth_dic_split[f'{tripID}-{type}-{count}'] = balineline_split.reset_index(drop=True)
                tripIDlist.append(f'{tripID}-{type}-{count}')
                triptypelist.append(f'{type[0:-1]}')
                #### 统计误差
                true_llh = np.array(
                    [balineline_split[' Latitude_GT (deg)'], balineline_split[' Longitude_GT (deg)'], balineline_split[' Ellipsoid Height_GT (m)']])
                guess_spp_llh = np.array(
                    [balineline_split[' Latitude (deg)'], balineline_split[' Longitude (deg)'], balineline_split[' Ellipsoid Height (m)']])

                dist_ll_spp = vincenty_distance(guess_spp_llh.T, true_llh.T)
                dist_h_spp = np.abs(guess_spp_llh[2, :] - true_llh[2, :])

                disll_type[f'{type[0:-1]}'].append(dist_ll_spp)
                dish_type[f'{type[0:-1]}'].append(dist_h_spp)


tripID_df=pd.DataFrame(tripIDlist, columns=['tripID'])
tripID_df['Type']=triptypelist
if save_file:
    with open(dir_path + f'/env/raw_baseline_SZdata_new_split.pkl', 'wb') as value_file:
        pickle.dump(data_truth_dic_split, value_file, True)
    value_file.close()
    tripID_df.to_csv(dir_path + f'env/raw_tripID_SZdata_split.csv', index=True)
    print('finish save files')

if plot_dis:
    err_max = 100
    sample_num = 0
    # for type, value in disxyz_type.items():
    #     plt.figure(figsize=(6, 5))
    #     dis_err_spp = np.concatenate(value)
    #     dis_err = np.where(dis_err_spp > err_max, err_max, dis_err_spp)
    #     binnum = int(np.ceil(max(dis_err)))
    #     plt.hist(dis_err, density=True, bins=binnum, rwidth=0.9, alpha=0.5,
    #              label='SPP, average error={:.3}m'.format(np.mean(dis_err_spp)), color='red')
    #     # plt.xlim([-0.5, 40])
    #     plt.grid()
    #     plt.legend(fontsize=labelsize-5)
    #     plt.title(f'3D distance error of ecef in {type}', fontsize=labelsize)
    #     plt.xlabel('Distance error (m)', fontsize=labelsize)
    #     plt.ylabel('Proportion', fontsize=labelsize)
    #     plt.savefig(f'{plt_folder}/3Derr_distribution_{type}.png', bbox_inches='tight')
    #
    #     plt.figure(figsize=(6, 5))
    #     plt.plot(dis_err_spp, label='SPP, average error={:.3}m'.format(np.nanmean(dis_err_spp)))
    #     plt.grid()
    #     plt.legend(fontsize=labelsize-5)
    #     plt.title(f'3D distance error of ecef in {type}', fontsize=labelsize)
    #     plt.xlabel('Step', fontsize=labelsize)
    #     plt.ylabel('Distance error (m)', fontsize=labelsize)
    #     if np.max(dis_err_spp) > 200:
    #         plt.ylim([0, 200])
    #     # plt.savefig(f'{plt_folder}/3Derr_ecef_{type}.png', bbox_inches='tight')
    #     plt.show()
    #     print(f'sample number in {type}: {len(dis_err_spp)}')
    #     sample_num += len(dis_err_spp)
    # print(f'All sample number: {sample_num}')

    for type, value in disll_type.items():
        plt.figure(figsize=(6, 5))
        ll_err_spp = np.concatenate(value)
        ll_err = np.where(ll_err_spp > err_max, err_max, ll_err_spp)
        binnum = int(np.ceil(max(ll_err)))
        plt.hist(ll_err, density=True, bins=binnum, rwidth=0.9, alpha=0.5,
                 label='GNSS/INS, average error={:.3}m'.format(np.mean(ll_err_spp)), color='red')
        # plt.xlim([-0.5, 40])
        plt.grid()
        plt.legend(fontsize=labelsize-5)
        plt.title(f'Horizontal error ecef in {type}', fontsize=labelsize)
        plt.xlabel('Horizontal error (m)', fontsize=labelsize)
        plt.ylabel('Proportion', fontsize=labelsize)
        plt.savefig(f'{plt_folder}/2Derr_distribution_{type}.png', bbox_inches='tight')

        plt.figure(figsize=(6, 5))
        plt.plot(ll_err_spp, label='GNSS/INS, average error={:.3}m'.format(np.nanmean(ll_err_spp)))
        plt.grid()
        plt.legend(fontsize=labelsize-5)
        plt.title(f'Horizontal error of ecef in {type}', fontsize=labelsize)
        plt.xlabel('Step', fontsize=labelsize)
        plt.ylabel('Horizontal error (m)', fontsize=labelsize)
        if np.max(ll_err_spp) > 200:
            plt.ylim([0, 200])
        # plt.savefig(f'{plt_folder}/2Derr_{type}.png', bbox_inches='tight')
        plt.show()

    for type, value in dish_type.items():
        plt.figure(figsize=(6, 5))
        h_err_spp = np.concatenate(value)
        h_err = np.where(h_err_spp > err_max, err_max, h_err_spp)
        binnum = int(np.ceil(max(h_err)))
        plt.hist(h_err, density=True, bins=binnum, rwidth=0.9, alpha=0.5,
                 label='GNSS/INS, average error={:.3}m'.format(np.mean(h_err_spp)), color='red')
        # plt.xlim([-0.5, 40])
        plt.grid()
        plt.legend(fontsize=labelsize-5)
        plt.title(f'Elevation error ecef in {type}', fontsize=labelsize)
        plt.xlabel('Elevation error (m)', fontsize=labelsize)
        plt.ylabel('Proportion', fontsize=labelsize)
        plt.savefig(f'{plt_folder}/herr_distribution_{type}.png', bbox_inches='tight')

        plt.figure(figsize=(6, 5))
        plt.plot(h_err_spp, label='GNSS/INS, average error={:.3}m'.format(np.nanmean(h_err_spp)))
        plt.grid()
        plt.legend(fontsize=labelsize-5)
        plt.title(f'Elevation error of ecef in {type}', fontsize=labelsize)
        plt.xlabel('Step', fontsize=labelsize)
        plt.ylabel('Elevation error (m)', fontsize=labelsize)
        if np.max(h_err_spp) > 200:
            plt.ylim([0, 200])
        # plt.savefig(f'{plt_folder}/herr_{type}.png', bbox_inches='tight')
        plt.show()
