# 强化学习定位环境构建
import gym
from gym import spaces
import random
import pickle
import os
import sys
import numpy as np
import pandas as pd
from env.env_param_SZdata import *
from scipy.spatial import distance
from scipy.linalg import block_diag
import math
import pymap3d as pm
import torch
pid = os.getpid()
print("当前程序的 PID:", pid)

current_dir = os.path.dirname(os.path.abspath(__file__))
# 定位到 src 目录
# 因为 env 和 src 同级，所以是先返回上一级再进入 src
src_path = os.path.join(current_dir, "..", "src")
sys.path.append(src_path)
# 组合导航的函数
from kfgins.kf_gins_types import GINSOptions
from common.angle import Angle
from common.types import IMU, GNSS
from common.funcs import *
from fileio.gnssfileloader_SZdata import GnssFileLoader
from fileio.imufileloader_SZdata import ImuFileLoader
from scipy.stats import chi2
from kfgins.gi_engine import GIEngine
import yaml
import src.gnss_lib.coordinates as coord

step_print = False
# 导入数据
dir_path = '/mnt/sdb/home/tangjh/KF-GINS-Py-main/'  # '/home/tangjh/smartphone-decimeter-2022/''D:/jianhao/smartphone-decimeter-2022/'
# load raw baseline data
with open(dir_path + 'env/raw_baseline_gnssins_SZdata.pkl', "rb") as file:
    data_truth_dic = pickle.load(file)
file.close()
# load raw imu data
with open(dir_path + 'env/raw_imu_data_SZdata.pkl', "rb") as file:
    data_raw_imu_dic = pickle.load(file)
file.close()
# load raw gnss data
with open(dir_path + 'env/raw_gnss_data_SZdata.pkl', "rb") as file:
    data_raw_gnss_dic = pickle.load(file)
file.close()
"""
data_raw_gnss_dic:
'UnixTimeMillis_ref', 'latitude(deg)', 'longitude(deg)', 'height(m)','sdn(m)', 'sde(m)','sdu(m)','Satnum',
'CNR_Mean','CNR_Q75','CNR_Q25','EA_Mean','EA_Q75','EA_Q25','PDOP', 'HDOP','Q'
data_raw_imu_dic:
'UnixTimeMillis_ref', ' Angular rate X (rad/s)', ' Angular rate Y (rad/s)', ' Angular rate Z (rad/s)',
' Acceleration X (m/s^2)', ' Acceleration Y (m/s^2)',' Acceleration Z (m/s^2)'
"""
record_feature = True
RANK = 21  # 状态量
NOISERANK = 18
GNSS_VALID = 1 # (s) GNSS和当前时间间隔时间内可以用当前GNSS特征

class baseEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, config, **kwargs):
        super(baseEnv, self).__init__()
        self.config = config
        self.pos_num = config["env_para"]['postraj_num']
        self.triptype = kwargs.get('triptype')
        self.tripIDlist = trip_type_maping[self.triptype]
        self.tripIDlist = [item for item in self.tripIDlist if item not in exclude_traj] # 剔除不需要的traj
        # 轨迹范围处理
        self.traj_type = kwargs.get('traj_type')
        self.trajdata_range = kwargs.get('trajdata_range')
        self.continuous_actionspace = config["env_para"]['continuous_actionspace']
        self.trajdata_sort = config["env_para"]['trajdata_sort']
        self.baseline_mod = config["env_para"]['baseline_mod']
        self.cord = config["env_para"]['cord']
        self.pos_weight = 1 # 默认位置权重为1
        self.reward_setting = config["env_para"]['reward_setting']
        self.Rlstate_reset = config["env_para"]['RL_reset_step']
        self.prdcov_mode = config["env_para"]['prdcov_mode']
        self.cumulated_reward = 0
        self.count = 0
        self.max_return = 0
        self.early_break = False #早停条件
        self.flag_finetune = kwargs.get('finetune', False) # 微调判断使用

        # 按顺序轨迹还是打乱顺序
        if self.trajdata_sort == 'sorted':
            self.tripIDnum = self.trajdata_range[0]-1
        elif self.trajdata_sort == 'randint':
            sublist = self.tripIDlist[self.trajdata_range[0]:self.trajdata_range[1]]
            random.shuffle(sublist)
            self.tripIDlist[self.trajdata_range[0]:self.trajdata_range[1]] = sublist
            self.tripIDnum = self.trajdata_range[0]-2

    def reset(self):
        # Reset the state of the environment to an initial state
        self.done = False
        self.fusing = False # 是否开始融合
        self.fusing_count = 0 # 记录融合了几次
        self.RTK_pre_state = 1  # RTK上一步解状态，默认是固定
        self.RTK_cur_state = 1 # RTK当前解状态
        self.RTK_std = np.zeros([3,1]) # RTK标准差
        if self.trajdata_sort == 'randint':
            # self.tripIDnum=random.randint(0,len(self.tripIDlist)-1)
            self.tripIDnum = random.randint(self.trajdata_range[0], self.trajdata_range[1])
        elif self.trajdata_sort == 'sorted':
            self.tripIDnum = self.tripIDnum + 1
            if self.tripIDnum > self.trajdata_range[1]:
                self.tripIDnum = self.trajdata_range[0]
                # 跑完一轮，验证提早停止条件
                self.count += 1
                self.cumulated_reward = 0
                if self.count > 5:
                    self.early_break = True
                    print("Early stop")
                if self.cumulated_reward > self.max_return:
                    self.count = 0
                    self.max_return = self.cumulated_reward

        # self.tripIDnum=tripIDnum
        # self.info['tripIDnum']=self.tripIDnum
        self.baseline = data_truth_dic[self.tripIDlist[self.tripIDnum]].copy()
        self.raw_gnss = data_raw_gnss_dic[self.tripIDlist[self.tripIDnum][:25]].copy()
        self.raw_imu = data_raw_imu_dic[self.tripIDlist[self.tripIDnum][:25]].copy()
        self.datatime = self.baseline['UnixTimeMillis_ref'].values
        self.timeend = self.baseline.loc[len(self.baseline.loc[:, 'UnixTimeMillis_ref'].values) - 1, 'UnixTimeMillis_ref']
        self.start_pos = self.baseline.loc[0, [' Latitude (deg)',' Longitude (deg)',' Ellipsoid Height (m)']].values # 初始位置，用于坐标转换
        # normalize baseline
        # self.baseline['LatitudeDegrees_norm'] = (self.baseline['LatitudeDegrees']-lat_min)/(lat_max-lat_min)
        # self.baseline['LongitudeDegrees_norm'] = (self.baseline['LongitudeDegrees']-lon_min)/(lon_max-lon_min)
        # gen pred

        # 初始化准备导航状态
        config_filename = os.path.abspath(f'{dir_path}/dataset_SZ/{self.tripIDlist[self.tripIDnum][:25]}/kf-gins.yaml')
        with open(config_filename, 'r', encoding='utf-8') as f:
            self.traj_config = yaml.load(f, Loader=yaml.FullLoader)  # 从YAML文件(filename)中加载轨迹配置数据
        options = GINSOptions() # 初始化导航状态
        loadConfig(self.traj_config, options) # 根据配置表初始化导航状态和噪声参数等

        # 加载GNSS文件和IMU文件
        self.gnssfile = GnssFileLoader(self.raw_gnss)
        self.imufile = ImuFileLoader(self.raw_imu, self.traj_config["imudatarate"])
        self.imurate = self.traj_config["imudatarate"] # imu数据频率
        self.gnssrate = 1 / (self.raw_gnss[1,0] - self.raw_gnss[0,0]) # gnss 频率

        if self.baseline_mod == 'GNSS/INS':
            self.baseline['Latitude_RLpredict'] = self.baseline[' Latitude (deg)']
            self.baseline['Longitude_RLpredict'] = self.baseline[' Longitude (deg)']
            self.baseline['Ellipsoid_Height_RLpredict'] = self.baseline[' Ellipsoid Height (m)']
            self.baseline['Velocity_X_RLpredict'] = self.baseline[' Velocity X (m/s)']
            self.baseline['Velocity_Y_RLpredict'] = self.baseline[' Velocity Y (m/s)']
            self.baseline['Velocity_Z_RLpredict'] = self.baseline[' Velocity Z (m/s)']
            self.baseline['Roll_RLpredict'] = self.baseline[' Roll (deg)']
            self.baseline['Pitch_RLpredict'] = self.baseline[' Pitch (deg)']
            self.baseline['Heading_RLpredict'] = self.baseline[' Heading (deg)']

        # Set the current step to a random point within the data frame
        # 数据对齐, 时间移到定义的开始时间
        start_data_step = int(np.ceil(len(self.baseline) * self.traj_type[0])) # 按照比例获得当前gnss索引
        # self.gnss_step = start_data_step + self.pos_num - 1 # 移动到序列后第一个点
        # gnss_time = self.raw_gnss[self.gnss_step,0] # 获取当前的gnss时间
        # nav_idx, cur_nav_time = self.find_time(self.datatime, gnss_time) # 查找当前imu时间
        # self.current_step = nav_idx - 2 # 当前的导航步已经索引到当前开始RL预测的步，环境从该步开始，current_step是针对baseline而言
        self.current_step = int(start_data_step + self.pos_num * self.imurate/self.gnssrate) -2  # self.pos_num 步后的baseline idx
        self.pre_step = int(start_data_step + (self.pos_num-1) * self.imurate/self.gnssrate)

        # 设置 初始RL独立维护的状态和协方差
        # gnss_time = self.raw_gnss[self.gnss_step-1, 0]  # 获取当前的前一步gnss时间
        # nav_idx, _ = self.find_time(self.datatime, gnss_time)
        self.RL_prestate = self.baseline.iloc[self.pre_step, 1:10].values
        self.RLcov = np.eye(self.State_Dim) * self.config["env_para"]['initial_RLcov']
        self.policy_cov_scale = self.config["env_para"]['continuous_scale_policy_cov']
        self.pre_fusion_time = self.datatime[self.pre_step] # 初次融合时间

        # 结束时间戳
        end_data_step = int(np.ceil(len(self.baseline) * self.traj_type[1]))
        self.end_step_time = self.datatime[int(end_data_step-self.imurate/self.gnssrate)] # 手动设置融合结束时间

        if self.traj_type[0] > 0:  # 只要剩下部分轨迹的定位结果
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[0:self.current_step , ['Latitude_RLpredict']] = None
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[0:self.current_step , ['Longitude_RLpredict']] = None
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[0:self.current_step , ['Ellipsoid Height_RLpredict']] = None
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[0:self.current_step , ['Velocity_X_RLpredict']] = None
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[0:self.current_step , ['Velocity_Y_RLpredict']] = None
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[0:self.current_step , ['Velocity_Z_RLpredict']] = None
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[0:self.current_step , ['Roll_RLpredict']] = None
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[0:self.current_step , ['Pitch_RLpredict']] = None
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[0:self.current_step , ['Heading_RLpredict']] = None

        # 构建GNSS/INS推理引擎
        self.giengine = KRLF_GIEngine(options)

        if self.traj_config["endtime"] < 0:
            self.endtime = self.imufile.endtime() # 默认是imu最后数据时刻是结束时间
        else:
            self.endtime = self.traj_config["endtime"]

        # 直接设置到当前的imu和gnss到引擎
        # self.imu_cur = self.imufile.set_data(self.current_step)
        # self.gnss = self.gnssfile.set_data(self.gnss_step)
        # 将系统状态和协方差递推到环境开始的步
        starttime = self.gnssfile.starttime()
        while True:
            self.imu_cur = self.imufile.next()
            if self.imu_cur.time >= starttime:
                break

        while True:
            self.gnss = self.gnssfile.next()
            if self.gnss.time >= starttime:
                break

        # 添加IMU和GNSS数据到GIEngine中，补偿IMU误差
        self.giengine.addImuData(self.imu_cur, True)  # 加载新时间的IMU数据，并设置是否补偿
        self.giengine.addGnssData(self.gnss)  # 加载新时间的gnss数据，并设置为可用

        while True: # 推算到当前的时刻（self.current_step）
            if self.giengine.timestamp() >= self.datatime[self.current_step]:
                break # 当前 current_step 还没做ins
            if self.gnss.time < self.imu_cur.time and not self.gnssfile.isEof():  # 判断不大于数据长度
                self.gnss = self.gnssfile.next()
                self.giengine.addGnssData(self.gnss)

            self.imu_cur = self.imufile.next()
            if self.imu_cur.time > self.endtime or self.imufile.isEof():  # 判断不大于IMU数据长度
                break

            self.giengine.addImuData(self.imu_cur)
            self.giengine.newImuProcess()

            progress = (self.giengine.timestamp() - starttime) / (self.datatime[self.current_step] - starttime) * 100.0
            sys.stdout.write('Reseting: \r[{:.2f}%]'.format(progress) + str(self.giengine.timestamp()))  # 创建动态进度显示
            sys.stdout.flush()

        self.innovation = 0.1 * np.ones([1,self.State_Dim])
        obs = self._next_observation()
        # must return in observation scale
        return obs

    def find_time(self, raw_imu, cur_gnss_time):
        # 1. 计算第一列（时间列）与目标时间的差值的绝对值
        absolute_diff = np.abs(raw_imu - cur_gnss_time)
        # 2. 找到差值最小的那个位置的索引
        idx = absolute_diff.argmin()
        # 3. 获取对应的元素（时间值）
        closest_time = raw_imu[idx]
        return idx, closest_time

    def _next_observation(self):
        raise NotImplementedError("please set _next_observation")

    def _normalize_his_diff(self, state):
        if self.cord == 'NED':
            state[0:2,:] = state[0:2,:] / 30
        elif self.cord == 'LLH':
            state[0:2, :] = state[0:2, :] / 1e-5
        state[2,:] = state[2,:] / 2.5
        state[3:5,:] = state[3:5,:] / 2
        state[5,:] = state[5,:] / 0.1
        state[6:8, :] = state[6:8, :] / 2
        state[8, :] = state[8, :] / 20
        return state

    def _normalize_gnss(self, gnss):
        ## max normalize
        gnss[0] = (gnss[0]) / SAT_MAX
        gnss[1] = (gnss[1] - CNR_MIN) / (CNR_MAX - CNR_MIN)
        gnss[2] = (gnss[2] - CNR_MIN) / (CNR_MAX - CNR_MIN)
        gnss[3] = (gnss[3] - CNR_MIN) / (CNR_MAX - CNR_MIN)
        gnss[4] = (gnss[4] - ELE_MIN) / (ELE_MAX - ELE_MIN)
        gnss[5] = (gnss[5] - ELE_MIN) / (ELE_MAX - ELE_MIN)
        gnss[6] = (gnss[6] - ELE_MIN) / (ELE_MAX - ELE_MIN)
        gnss[7] = 1 / gnss[7]
        gnss[8] = 1 / gnss[8] / 2
        gnss[9] = gnss[9] / 5
        return gnss

    def _normalize_state(self, pre_state):
        if self.cord == 'NED':
            pre_state[0] = pre_state[0] / 50
            pre_state[1] = pre_state[1] / 50
        elif self.cord == 'LLH':
            pre_state[0] = pre_state[0] / 30
            pre_state[1] = pre_state[1] / 120
        pre_state[2] = pre_state[2] / 5
        pre_state[3] = pre_state[3] / 5
        pre_state[4] = pre_state[4] / 5
        pre_state[5] = pre_state[5] / 1
        pre_state[6] = pre_state[6] / 360
        pre_state[7] = pre_state[7] / 360
        pre_state[8] = pre_state[8] / 360
        return pre_state

    def set_results(self, ins_result):
        target_slice = self.baseline.loc[self.current_step - ins_result.shape[0] + 1: self.current_step,['Latitude_RLpredict']]
        data_len = len(target_slice)
        if ins_result.shape[0] > data_len:
            ins_result = ins_result[:data_len,:]

        self.baseline.loc[self.current_step - ins_result.shape[0] +1: self.current_step,['Latitude_RLpredict']] = ins_result[:, 0].reshape(-1,1)
        self.baseline.loc[self.current_step - ins_result.shape[0] +1: self.current_step,['Longitude_RLpredict']] = ins_result[:, 1].reshape(-1,1)
        self.baseline.loc[self.current_step - ins_result.shape[0] +1: self.current_step,['Ellipsoid_Height_RLpredict']] = ins_result[:, 2].reshape(-1,1)
        self.baseline.loc[self.current_step - ins_result.shape[0] +1: self.current_step,['Velocity_X_RLpredict']] = ins_result[:, 3].reshape(-1,1)
        self.baseline.loc[self.current_step - ins_result.shape[0] +1: self.current_step,['Velocity_Y_RLpredict']] = ins_result[:, 4].reshape(-1,1)
        self.baseline.loc[self.current_step - ins_result.shape[0] +1: self.current_step,['Velocity_Z_RLpredict']] = ins_result[:, 5].reshape(-1,1)
        self.baseline.loc[self.current_step - ins_result.shape[0] +1: self.current_step, ['Roll_RLpredict']] = ins_result[:, 6].reshape(-1,1)
        self.baseline.loc[self.current_step - ins_result.shape[0] +1: self.current_step, ['Pitch_RLpredict']] = ins_result[:, 7].reshape(-1,1)
        self.baseline.loc[self.current_step - ins_result.shape[0] +1: self.current_step,['Heading_RLpredict']] = ins_result[:, 8].reshape(-1,1)

        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step - ins_result.shape[0] +1: self.current_step,['Latitude_RLpredict']] = ins_result[:, 0].reshape(-1,1)
        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step - ins_result.shape[0] +1: self.current_step,['Longitude_RLpredict']] = ins_result[:, 1].reshape(-1,1)
        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step - ins_result.shape[0] +1: self.current_step,['Ellipsoid_Height_RLpredict']] = ins_result[:, 2].reshape(-1,1)
        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step - ins_result.shape[0] +1: self.current_step,['Velocity_X_RLpredict']] = ins_result[:, 3].reshape(-1,1)
        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step - ins_result.shape[0] +1: self.current_step,['Velocity_Y_RLpredict']] = ins_result[:, 4].reshape(-1,1)
        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step - ins_result.shape[0] +1: self.current_step,['Velocity_Z_RLpredict']] = ins_result[:, 5].reshape(-1,1)
        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step - ins_result.shape[0] +1: self.current_step,['Roll_RLpredict']] = ins_result[:, 6].reshape(-1,1)
        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step - ins_result.shape[0] +1: self.current_step,['Pitch_RLpredict']] = ins_result[:, 7].reshape(-1,1)
        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step - ins_result.shape[0] +1: self.current_step,['Heading_RLpredict']] = ins_result[:, 8].reshape(-1,1)

    def KRLF_Process(self, dx_rl, cov_rl):
        time = self.giengine.imucur_.time
        self.giengine.timestamp_ = time

        # 如果GNSS有效，则将更新时间设置为GNSS时间
        # set update time as the gnss time if gnssdata is valid
        if self.giengine.gnssdata_.isvalid:
            updatetime = self.giengine.gnssdata_.time
        else:
            updatetime = -1

        # 判断是否需要进行GNSS更新
        # determine if we should do GNSS update
        imupre_ = self.giengine.imupre_
        imucur_ = self.giengine.imucur_
        gnssdata_ = self.giengine.gnssdata_

        res = self.giengine.isToUpdate(imupre_.time, imucur_.time, updatetime)

        if res == 0:
            raise ValueError(f"Invalid state: res is {res}. Expected res > 0 for GNSS/INS integration.")
            self.giengine.insPropagation(imupre_, imucur_)
            self._error_state_fusion(dx_rl, cov_rl)
            # self.giengine.stateFeedback()
        elif res == 1:  # 更新时间靠近imutime1
            # GNSS数据靠近上一历元，先对上一历元进行GNSS更新
            # gnssdata is near to the previous imudata, we should firstly do gnss update
            self.giengine.gnssUpdate(gnssdata_)
            self._error_state_fusion(dx_rl,cov_rl)
            self.giengine.stateFeedback()
            self.giengine.pvapre_ = self.giengine.pvacur_
            self.giengine.insPropagation(imupre_, imucur_)
        elif res == 2:  # 更新时间靠近imutime2
            # GNSS数据靠近当前历元，先对当前IMU进行状态传播
            self.giengine.insPropagation(imupre_, imucur_)
            self.giengine.gnssUpdate(gnssdata_)
            self._error_state_fusion(dx_rl, cov_rl)
            self.giengine.stateFeedback()
        else:
            # GNSS数据在两个IMU数据之间(不靠近任何一个), 将当前IMU内插到整秒时刻
            # gnssdata is near current imudata, we should firstly propagate navigation state
            midimu = IMU
            imucur_, midimu = KRLF_GIEngine.imuInterpolate(imupre_, imucur_, updatetime, midimu)

            # 对前一半IMU进行状态传播
            # propagate navigation state for the first half imudata
            self.giengine.insPropagation(imupre_, midimu)

            # 整秒时刻进行GNSS更新，并反馈系统状态
            # do GNSS position update at the whole second and feedback system states
            self.giengine.gnssUpdate(gnssdata_)
            self._error_state_fusion(dx_rl, cov_rl)
            self.giengine.stateFeedback()

            # 对后一半IMU进行状态传播
            # propagate navigation state for the second half imudata
            self.giengine.pvapre_ = self.giengine.pvacur_
            self.giengine.insPropagation(midimu, imucur_)

        # 更新上一时刻的状态和IMU数据
        # update system state and imudata at the previous epoch
        self.giengine.pvapre_ = self.giengine.pvacur_
        self.giengine.imupre_ = self.giengine.imucur_

    def _error_state_fusion(self,dx_rl,cov_rl):  # RL for KF modified in 0303
        dx_kf = self.giengine.dx_[0:self.State_Dim].copy() # 只获取 pva
        dx_rl = dx_rl.reshape([self.State_Dim,1])
        cov_kf_pva = self.giengine.Cov_[StateID.P_ID:StateID.P_ID + self.State_Dim, StateID.P_ID:StateID.P_ID + self.State_Dim].copy() # kf的pva协方差
        # I = np.identity(dx_kf.shape[0])
        # A = np.vstack((I, I))
        # N = block_diag(cov_kf_pva, cov_rl)
        # b = np.concatenate([dx_kf,dx_rl.reshape([9,1])])
        # 求解融合误差状态
        # # 1. 计算中间权阵 W = N^-1
        # try:
        #     W = np.linalg.inv(N)
        # except np.linalg.LinAlgError:
        #     # 如果 N 奇异，添加微小扰动以保持稳定
        #     W = np.linalg.inv(N + np.eye(N.shape[0]) * 1e-9)
        # # 2. 计算信息矩阵 (Normal Equation Matrix): H = A.T @ W @ A
        # AT_W = A.T @ W
        # H = AT_W @ A
        # # 3. 计算右侧观测向量: y = A.T @ W @ b
        # y = AT_W @ b
        # # 4. 求解状态增量: x = H^-1 @ y
        # dx_f = np.linalg.solve(H, y)
        K_fusion = np.linalg.solve((cov_kf_pva + cov_rl + np.eye(self.State_Dim) * 1e-12).T, cov_kf_pva.T).T
        dx_f = dx_kf + K_fusion @ (dx_rl - dx_kf)
        self.giengine.dx_[StateID.P_ID:self.State_Dim] = dx_f # 放置回原来向量

        # 融合协方差
        try:
            fusion_gain = np.linalg.solve(cov_kf_pva + cov_rl + np.eye(self.State_Dim) * 1e-12, cov_rl)
            cov_f = cov_kf_pva @ fusion_gain
        except np.linalg.LinAlgError:
            # 极端情况下退回到简单加权
            cov_f = 0.5 * (cov_kf_pva + cov_rl)

        # 3. 强制对称化与对角线保护
        cov_f = (cov_f + cov_f.T) * 0.5

        self.giengine.Cov_[StateID.P_ID:StateID.P_ID + self.State_Dim, StateID.P_ID:StateID.P_ID + self.State_Dim] = cov_f
        # 设置inovation参数
        self.innovation, Nis = compute_normalized_innovation(dx_kf, dx_rl, cov_kf_pva, cov_rl)

    def reward_calculation(self,navstate):
        if self.baseline_mod == 'GNSS/INS':
            bl_lat = self.baseline.loc[self.current_step, ' Latitude (deg)'] * Angle.D2R
            bl_lon = self.baseline.loc[self.current_step, ' Longitude (deg)'] * Angle.D2R
            bl_h = self.baseline.loc[self.current_step, ' Ellipsoid Height (m)']

        gt_lat = self.baseline.loc[self.current_step, ' Latitude_GT (deg)'] * Angle.D2R
        gt_lon = self.baseline.loc[self.current_step, ' Longitude_GT (deg)'] * Angle.D2R
        gt_h = self.baseline.loc[self.current_step, ' Ellipsoid Height_GT (m)']

        rl_lat = navstate.pos[0]
        rl_lon = navstate.pos[1]
        rl_h = navstate.pos[2]

        # 计算bl误差
        blh_station = self.start_pos[0:3]
        rm, rn = radiusmn(blh_station[0])
        bl_pos_err = np.array([bl_lat-gt_lat,bl_lon-gt_lon,bl_h-gt_h])
        bl_pos_err = drad2dm(rm, rn, blh_station, bl_pos_err).reshape(1, 3)

        # 计算rl误差
        rl_pos_err = np.array([rl_lat-gt_lat,rl_lon-gt_lon,rl_h-gt_h])
        rl_pos_err = drad2dm(rm, rn, blh_station, rl_pos_err).reshape(1, 3)

        # TODO 水平和高程是否需要加比例因子
        mse_rl_p = np.sqrt(np.sum(rl_pos_err ** 2))
        mse_bl_p = np.sqrt(np.sum(bl_pos_err ** 2))

        if self.reward_setting == 'RMSEadv': # or误差-rl误差
            reward = self.pos_weight * (mse_bl_p - mse_rl_p)
            reward = np.clip(reward, -10, 10)
        elif self.reward_setting == 'RMSEadv_ratio': # (or误差-rl误差)/or误差
            reward = self.pos_weight * (mse_bl_p - mse_rl_p)/mse_bl_p
            reward = np.clip(reward, -1.5*self.pos_weight, 1.5*self.pos_weight)
        elif self.reward_setting == 'RMSEadv_tanh': # (or误差-rl误差)/or误差
            reward = np.tanh(mse_bl_p - mse_rl_p) + self.vel_weight * np.tanh(mse_bl_v - mse_rl_v) + self.att_weight * np.tanh(mse_bl_a- mse_rl_a)
        elif self.reward_setting == 'RMSEratio_log':
            eps = 0.008
            reward = 5 * (np.log((mse_bl_p + eps) / (mse_rl_p + eps)) + self.vel_weight * np.log((mse_bl_v + eps) / (mse_rl_v + eps)) \
                     + self.att_weight * np.log((mse_bl_a + eps) / (mse_rl_a + eps)))
            reward = np.clip(reward, -10, 10)
        elif self.reward_setting == 'RMSEratio_tanh':
            reward = 10 * ( compute_reward(mse_bl_p, mse_rl_p) + self.vel_weight * compute_reward(mse_bl_v, mse_rl_v, self.vel_weight) \
                     + self.att_weight * compute_reward(mse_bl_a, mse_rl_a, self.att_weight))

        return reward, mse_rl_p

    def reward_finetune(self,navstate):
        if self.baseline_mod == 'GNSS/INS':
            bl_lat = self.baseline.loc[self.current_step, ' Latitude (deg)'] * Angle.D2R
            bl_lon = self.baseline.loc[self.current_step, ' Longitude (deg)'] * Angle.D2R
            bl_h = self.baseline.loc[self.current_step, ' Ellipsoid Height (m)']
            bl_vx = self.baseline.loc[self.current_step, ' Velocity X (m/s)']
            bl_vz = self.baseline.loc[self.current_step, ' Velocity Z (m/s)']
            bl_vy = self.baseline.loc[self.current_step, ' Velocity Y (m/s)']
            bl_roll = self.baseline.loc[self.current_step, ' Roll (deg)']
            bl_pitch = self.baseline.loc[self.current_step, ' Pitch (deg)']
            bl_heading = self.baseline.loc[self.current_step, ' Heading (deg)']

        gt_lat = self.baseline.loc[self.current_step, ' Latitude_GT (deg)'] * Angle.D2R
        gt_lon = self.baseline.loc[self.current_step, ' Longitude_GT (deg)'] * Angle.D2R
        gt_h = self.baseline.loc[self.current_step, ' Ellipsoid Height_GT (m)']
        gt_vx = self.baseline.loc[self.current_step, ' Velocity Y_GT (m/s)'] # 速度ENU转成NED下
        gt_vy = self.baseline.loc[self.current_step, ' Velocity X_GT (m/s)']
        gt_vz = -self.baseline.loc[self.current_step, ' Velocity Z_GT (m/s)']
        gt_roll = self.baseline.loc[self.current_step, ' Roll_GT (deg)']
        gt_pitch = self.baseline.loc[self.current_step, ' Pitch_GT (deg)']
        gt_heading = self.baseline.loc[self.current_step, ' Heading_GT (deg)']

        rl_lat = navstate.pos[0]
        rl_lon = navstate.pos[1]
        rl_h = navstate.pos[2]
        rl_vx = navstate.vel[0]
        rl_vy = navstate.vel[1]
        rl_vz = navstate.vel[2]
        rl_roll = navstate.euler[0] * Angle.R2D
        rl_pitch = navstate.euler[1] * Angle.R2D
        rl_heading = navstate.euler[2] * Angle.R2D

        # 计算bl误差
        blh_station = self.start_pos[0:3]
        rm, rn = radiusmn(blh_station[0])
        bl_pos_err = np.array([bl_lat-gt_lat,bl_lon-gt_lon,bl_h-gt_h])
        bl_pos_err = drad2dm(rm, rn, blh_station, bl_pos_err).reshape(1, 3)
        bl_vel_err = np.array([bl_vx-gt_vx,bl_vy-gt_vy,bl_vz-gt_vz])
        bl_att_err = np.array([bl_roll - gt_roll, bl_pitch - gt_pitch, bl_heading - gt_heading])
        if bl_att_err[2] > 180:
            bl_att_err[2] -= 360
        if bl_att_err[2] < -180:
            bl_att_err[2] += 360

        # 计算rl误差
        rl_pos_err = np.array([rl_lat-gt_lat,rl_lon-gt_lon,rl_h-gt_h])
        rl_pos_err = drad2dm(rm, rn, blh_station, rl_pos_err).reshape(1, 3)
        rl_vel_err = np.array([rl_vx-gt_vx,rl_vy-gt_vy,rl_vz-gt_vz])
        rl_att_err = np.array([rl_roll - gt_roll, rl_pitch - gt_pitch, rl_heading - gt_heading])
        if rl_att_err[2] > 180:
            rl_att_err[2] -= 360
        if rl_att_err[2] < -180:
            rl_att_err[2] += 360

        # TODO 水平和高程是否需要加比例因子
        mse_rl_p = np.sqrt(np.sum(rl_pos_err ** 2))
        mse_rl_v = np.sqrt(np.sum(rl_vel_err ** 2))
        mse_rl_a = np.sqrt(np.sum(rl_att_err ** 2))
        mse_bl_p = np.sqrt(np.sum(bl_pos_err ** 2))
        mse_bl_v = np.sqrt(np.sum(bl_vel_err ** 2))
        mse_bl_a = np.sqrt(np.sum(bl_att_err ** 2))

        if self.reward_setting == 'RMSEadv': # or误差-rl误差
            reward = 1 * (mse_bl_p - mse_rl_p) + self.vel_weight * (mse_bl_v - mse_rl_v) + self.att_weight * (mse_bl_a - mse_rl_a)
            reward = np.clip(reward, -10, 10)
        elif self.reward_setting == 'RMSEadv_ratio': # (or误差-rl误差)/or误差
            reward = 1 * (mse_bl_p - mse_rl_p)/mse_bl_p + self.vel_weight * (mse_bl_v - mse_rl_v)/mse_bl_v + self.att_weight * (mse_bl_a - mse_rl_a)/mse_bl_a
            reward = np.clip(reward, -1.5, 1.5)
        elif self.reward_setting == 'RMSEadv_tanh': # (or误差-rl误差)/or误差
            reward = np.tanh(mse_bl_p - mse_rl_p) + self.vel_weight * np.tanh(mse_bl_v - mse_rl_v) + self.att_weight * np.tanh(mse_bl_a- mse_rl_a)
        elif self.reward_setting == 'RMSEratio_log':
            eps = 0.008
            reward = 5 * (np.log((mse_bl_p + eps) / (mse_rl_p + eps)) + self.vel_weight * np.log((mse_bl_v + eps) / (mse_rl_v + eps)) \
                     + self.att_weight * np.log((mse_bl_a + eps) / (mse_rl_a + eps)))
            reward = np.clip(reward, -10, 10)
        elif self.reward_setting == 'RMSEratio_tanh':
            reward = 10 * ( compute_reward(mse_bl_p, mse_rl_p) + self.vel_weight * compute_reward(mse_bl_v, mse_rl_v, self.vel_weight) \
                     + self.att_weight * compute_reward(mse_bl_a, mse_rl_a, self.att_weight))

        return reward, mse_rl_p

    def render(self, mode='human', close=False):
        print(f'Step: {self.current_step}')
        #  print(f'reward: {self.reward}')
        print(f'total_reward: {self.total_reward}')

class Continuous_PrePosCov_InHiGNSSSatCov(baseEnv):
    # 连续动作的状态预测和协方差预测（该环境只预测位置），输入新息 历史序列 GNSS特征 状态, 输出位置预测
    metadata = {'render.modes': ['human']}
    def __init__(self, config, **kwargs):
        # def __init__(self,trajdata_range, action_scale, discrete_actionspace, reward_setting, trajdata_sort, baseline_mod):
        super(Continuous_PrePosCov_InHiGNSSSatCov, self).__init__(config,**kwargs)
        # TODO: 确认 ino和 his 的状态维度
        self.State_Dim = 3
        self.pos_weight = config["env_para"]['pos_weight']
        self.observation_space = spaces.Dict(
            {'gnss': spaces.Box(low=-1, high=1, shape=(1, 10)),
             'innovation': spaces.Box(low=0, high=1, shape=(1, self.State_Dim), dtype=np.float),
             'History': spaces.Box(low=0, high=1, shape=(1, (self.pos_num-1) * 9), dtype=np.float),
             'State': spaces.Box(low=0, high=1, shape=(1, 9), dtype=np.float),
             'Cov': spaces.Box(low=0, high=1, shape=(1, self.State_Dim), dtype=np.float)})

        # 设置动作尺度
        self.pred_scale = config["env_para"]['continuous_scale_state_pred']
        self.policy_cov_scale = config["env_para"]['continuous_scale_policy_cov']
        self.action_space = spaces.Box(low=self.continuous_actionspace[0], high=self.continuous_actionspace[1], shape=(1, 2*self.State_Dim),dtype=np.float)
        self.conv_mode = config["env_para"]['conv_mode']

    def _normalize_his_diff(self, state):
        if self.cord == 'NED':
            state[0:2,:] = state[0:2,:] / 30
        elif self.cord == 'LLH':
            state[0:2, :] = state[0:2, :] / 1e-5
        state[2,:] = state[2,:] / 2.5
        state[3:5,:] = state[3:5,:] / 2
        state[5,:] = state[5,:] / 0.1
        state[6:8, :] = state[6:8, :] / 2
        state[8, :] = state[8, :] / 20
        return state

    def _normalize_state(self, pre_state):
        if self.cord == 'NED':
            pre_state[0] = pre_state[0] / 50
            pre_state[1] = pre_state[1] / 50
        elif self.cord == 'LLH':
            pre_state[0] = pre_state[0] / 30
            pre_state[1] = pre_state[1] / 120
        pre_state[2] = pre_state[2] / 5
        pre_state[3] = pre_state[3] / 5
        pre_state[4] = pre_state[4] / 5
        pre_state[5] = pre_state[5] / 1
        pre_state[6] = pre_state[6] / 360
        pre_state[7] = pre_state[7] / 360
        pre_state[8] = pre_state[8] / 360
        return pre_state

    def _next_observation(self):
        # 在这里接入INS惯性推算，注意这里不会进行融合
        nav_result = []
        while not self.fusing: # 当开始融合时，不会再进入此循环
            if self.gnss.time < self.imu_cur.time and not self.gnssfile.isEof():  # 判断不大于数据长度
                self.gnss = self.gnssfile.next()
                self.giengine.addGnssData(self.gnss)

            # TODO: 验证回合终止条件
            if self.datatime[self.current_step] >= self.end_step_time: # 当前时间大于设置数据时间，结束回合
                self.done = True

            self.imu_cur = self.imufile.next()
            if self.imu_cur.time > self.endtime or self.imufile.isEof():  # 判断不大于IMU数据长度
                self.done = True
                break

            self.giengine.addImuData(self.imu_cur)
            res = self.giengine.check_update()
            if res == 0:
                self.giengine.newImuProcess()
                self.current_step = self.current_step + 1  # 需要注意，current_step 需要和 self.giengine.timestamp() 对应
            else:
                self.fusing = True
                break # 转移到step函数再融合

            navstate = self.giengine.getNavState()
            result = np.array([
                    np.round(navstate.pos[0] * Angle.R2D, 9),
                    np.round(navstate.pos[1] * Angle.R2D, 9),
                    np.round(navstate.pos[2], 9),
                    np.round(navstate.vel[0], 9),
                    np.round(navstate.vel[1], 9),
                    np.round(navstate.vel[2], 9),
                    np.round(navstate.euler[0] * Angle.R2D, 9),
                    np.round(navstate.euler[1] * Angle.R2D, 9),
                    np.round(navstate.euler[2] * Angle.R2D, 9)])

            nav_result.append(result)

        # TODO: 是否可以精简一下不要self.baseline
        if nav_result:
            ins_result = np.array(nav_result)
            self.set_results(ins_result) # 把新的导航结果赋值回去

        # 构建 self.pos_num -1 的历史序列拼接
        seq_len = self.pos_num * self.imurate/self.gnssrate # 根据频率获取窗口历元大小,预计大概是1s一个步长
        his_len = len(self.baseline.loc[self.current_step-seq_len: self.current_step, 'Latitude_RLpredict'].values)
        indices = np.linspace(0, his_len-1, num=self.pos_num, dtype=int)
        Lat_seq = self.baseline.loc[self.current_step-seq_len: self.current_step, 'Latitude_RLpredict'].values[indices]
        Lon_seq = self.baseline.loc[self.current_step-seq_len: self.current_step, 'Longitude_RLpredict'].values[indices]
        Hei_seq = self.baseline.loc[self.current_step-seq_len: self.current_step, 'Ellipsoid_Height_RLpredict'].values[indices]
        Vx_seq = self.baseline.loc[self.current_step-seq_len: self.current_step, 'Velocity_X_RLpredict'].values[indices]
        Vy_seq = self.baseline.loc[self.current_step-seq_len: self.current_step, 'Velocity_Y_RLpredict'].values[indices]
        Vz_seq = self.baseline.loc[self.current_step-seq_len: self.current_step, 'Velocity_Z_RLpredict'].values[indices]
        Roll_seq = self.baseline.loc[self.current_step-seq_len: self.current_step, 'Roll_RLpredict'].values[indices]
        Pitch_seq = self.baseline.loc[self.current_step-seq_len: self.current_step, 'Pitch_RLpredict'].values[indices]
        Heading_seq = self.baseline.loc[self.current_step-seq_len: self.current_step, 'Heading_RLpredict'].values[indices]
        obs = np.array([Lat_seq,Lon_seq,Hei_seq,Vx_seq,Vy_seq,Vz_seq,Roll_seq,Pitch_seq,Heading_seq])

        # 测试代码用
        # Lat_seq_gt = self.baseline.loc[self.current_step-seq_len: self.current_step, ' Latitude_GT (deg)'].values[indices]
        # Lon_seq_gt = self.baseline.loc[self.current_step-seq_len: self.current_step, ' Longitude_GT (deg)'].values[indices]
        # Hei_seq_gt = self.baseline.loc[self.current_step-seq_len: self.current_step, ' Ellipsoid Height_GT (m)'].values[indices]
        # obs_gt = np.array([Lat_seq,Lon_seq,Hei_seq])
        # obs_gt = np.diff(obs_gt, axis=1)
        # if self.cord == 'NED': # 地理坐标系转到导航坐标系
        #     blh_station = self.start_pos[0:3]
        #     rm, rn = radiusmn(blh_station[0])
        #     obs_gt[0:2, :] = obs_gt[0:2, :] * Angle.D2R
        #     for i in range(obs_gt.shape[1]):
        #         obs_gt[0:3,i] = drad2dm(rm, rn, blh_station, obs_gt[0:3,i]).reshape(3)

        # 历史状态差分
        obs = np.diff(obs, axis=1)

        # 确定坐标系
        if self.cord == 'NED': # 地理坐标系转到导航坐标系
            blh_station = self.start_pos[0:3]
            rm, rn = radiusmn(blh_station[0])
            obs[0:2, :] = obs[0:2, :] * Angle.D2R
            for i in range(obs.shape[1]):
                obs[0:3,i] = drad2dm(rm, rn, blh_station, obs[0:3,i]).reshape(3)

        obs = self._normalize_his_diff(obs)

        # gnss feature process
        # 'Satnum', 'CNR_Mean','CNR_Q75','CNR_Q25','EA_Mean','EA_Q75','EA_Q25','PDOP', 'HDOP','Q'
        obs_feature = self.gnssfile.data_[self.gnssfile.index,7:]
        obs_feature = self._normalize_gnss(obs_feature)

        # state feature process
        if self.cord == 'NED':  # 地理坐标系转到导航坐标系
            n,e,d = pm.geodetic2ned(self.RL_prestate[0], self.RL_prestate[1], self.RL_prestate[2], self.start_pos[0], self.start_pos[1], self.start_pos[2])
            pre_pos = np.array([n, e, d])
            pre_state = np.concatenate((pre_pos,self.RL_prestate[3:]))
        elif self.cord == 'LLH':
            pre_state = self.RL_prestate[0:3].copy()
        pre_state = self._normalize_state(pre_state)

        # RL cov features
        if self.prdcov_mode != "Add_mode1":
            # 获取KF方差的尺度，调整初始RL协方差尺度
            self.RLcov = np.eye(self.State_Dim) * self.config["env_para"]['initial_RLcov']
            cov_kf_pva = self.giengine.Cov_[StateID.P_ID:StateID.P_ID + self.State_Dim, StateID.P_ID:StateID.P_ID + self.State_Dim].copy()
            cov_kf_pva = np.mean(np.diag(cov_kf_pva))
            magnitude = math.floor(math.log10(cov_kf_pva))
            self.RLcov *= np.power(10.0,magnitude+self.config["env_para"]['magnitude'])

        rl_cov_diag = np.diag(self.RLcov) / self.config["env_para"]['initial_RLcov']

        # 观测合并
        obs_all = {'History': obs.reshape(1, obs.size, order='F'),
                   'gnss': obs_feature.reshape(1, 10),
                   'innovation': self.innovation.reshape(1, self.innovation.size),
                   'State': pre_state.reshape(1, pre_state.size),
                   'Cov': rl_cov_diag.reshape(1, rl_cov_diag.size)}

        return obs_all

    def step(self, action):  # modified in 3.3
        # done = (self.current_step >= len(self.baseline.loc[:, 'UnixTimeMillis'].values) * self.traj_type[-1] - (
        #     self.pos_num))
        # action for new prediction
        action = np.reshape(action, [1, 2 * self.State_Dim])
        predict_N = action[0, 0] * self.pred_scale
        predict_E = action[0, 1] * self.pred_scale
        predict_D = action[0, 2] * self.pred_scale * 1e-2

        blh_station = self.start_pos[0:3]
        rm, rn = radiusmn(blh_station[0])
        if self.cord == 'NED':
            d_pos_n = np.array([predict_N,predict_E,predict_D])
            d_pos_llh = dm2drad(rm, rn, blh_station, d_pos_n)
            d_pos_llh[0:2] = d_pos_llh[0:2] * Angle.R2D # 导航坐标系转大地
        elif self.cord == 'LLH':
            scale_llh = 1e-5
            d_pos_llh = np.array([predict_N * scale_llh,predict_E * scale_llh,predict_D])

        # 用于代码测试，不参与环境运行
        # gt_col = [' Latitude_GT (deg)',' Longitude (deg)',' Ellipsoid Height (m)',' Velocity X_GT (m/s)', ' Velocity Y_GT (m/s)', ' Velocity Z_GT (m/s)',
        #           ' Roll_GT (deg)', ' Pitch_GT (deg)', ' Heading_GT (deg)']
        # gt_lat_cur = self.baseline.loc[self.current_step + 1, gt_col].values
        # gt_lat_pre = self.baseline.loc[self.current_step - self.imurate/self.gnssrate, gt_col].values
        # d_gt = gt_lat_cur - gt_lat_pre
        # d_gt_nav = drad2dm(rm, rn, blh_station, d_gt[0:3]* Angle.D2R)

        # 预测RL状态
        navstate = self.giengine.getNavState() # 导航状态的位置和姿态都是弧度制
        self.RL_prestate[0:3] = self.RL_prestate[0:3] + d_pos_llh.reshape(3) # 单位是度

        # 计算误差状态: 注意RL预测的状态是角度制，但是预测导航状态是弧度制
        Dr = Earth.DR(navstate.pos) # 地理坐标相对位置 转 n系相对位置
        RL_pos_rad = self.RL_prestate[0:3].copy()
        RL_pos_rad[0:2] = RL_pos_rad[0:2] * Angle.D2R
        dx_rl = Dr @ (navstate.pos - RL_pos_rad)

        # 协方差调整/预测
        d_cov_rl = np.diag(self.policy_cov_scale * action[0, self.State_Dim:])
        # TODO 协方差预测不当会造成融合后对角线变为0（是因为太小？）
        if self.prdcov_mode == "Add_mode1":
            self.RLcov = self.RLcov + d_cov_rl
            self.RLcov[self.RLcov < 1e-1] = 1e-1
            pred_cov_rl = self.RLcov
        elif self.prdcov_mode == "Add_mode2":
            pred_cov_rl = self.RLcov + d_cov_rl
            pred_cov_rl[pred_cov_rl < 1e-1] = 1e-1
        elif self.prdcov_mode == "Scale_mode1":
            MIN_SCALE = 0.3
            MAX_SCALE = 3.0
            scale_factor = MIN_SCALE + (d_cov_rl + 1.0) * 0.5 * (MAX_SCALE - MIN_SCALE) # np.exp(d_cov_rl)
            scale_factor = np.clip(scale_factor, 0.3, 3)
            pred_cov_rl = self.RLcov * scale_factor

        # 确定到达融合状态，GNSS/INS可融合
        if not self.fusing:
            res = self.giengine.check_update()
            raise ValueError(f"Invalid state: res is {res}. Expected res > 0 for GNSS/INS integration.")
        self.KRLF_Process(dx_rl,pred_cov_rl)
        self.fusing = False # 融合后设为False
        self.current_step = self.current_step + 1 # 当前步加1
        self.fusing_count = self.fusing_count + 1 # 融合次数加1

        # 保存数据
        navstate = self.giengine.getNavState()
        result = np.array([
            np.round(navstate.pos[0] * Angle.R2D, 9),
            np.round(navstate.pos[1] * Angle.R2D, 9),
            np.round(navstate.pos[2], 9),
            np.round(navstate.vel[0], 9),
            np.round(navstate.vel[1], 9),
            np.round(navstate.vel[2], 9),
            np.round(navstate.euler[0] * Angle.R2D, 9),
            np.round(navstate.euler[1] * Angle.R2D, 9),
            np.round(navstate.euler[2] * Angle.R2D, 9)])
        self.set_results(result.reshape(1, -1))

        # 间隔一段步数重置RL状态
        if (self.fusing_count % self.Rlstate_reset == 0):
            # self.RL_prestate = self.baseline.iloc[self.current_step, 1:10].values
            rl_col = ['Latitude_RLpredict', 'Longitude_RLpredict', 'Ellipsoid_Height_RLpredict', 'Velocity_X_RLpredict',
                      'Velocity_Y_RLpredict', 'Velocity_Z_RLpredict', 'Roll_RLpredict', 'Pitch_RLpredict',
                      'Heading_RLpredict']
            self.RL_prestate = self.baseline.loc[self.current_step, rl_col].values

        # reward function
        if self.flag_finetune:
            reward, error = self.reward_finetune(navstate)
        else:
            reward, error = self.reward_calculation(navstate)
        self.cumulated_reward += reward

        # Execute one time step within the environment
        if self.done:
            obs = []
            # nav_result = []
            # if self.traj_type[1] < 1:
            #     while not self.fusing:  # 最后一次推理
            #         if self.gnss.time < self.imu_cur.time and not self.gnssfile.isEof():  # 判断不大于数据长度
            #             self.gnss = self.gnssfile.next()
            #             self.giengine.addGnssData(self.gnss)
            #
            #         # TODO: 验证回合终止条件
            #         self.imu_cur = self.imufile.next()
            #         if self.imu_cur.time > self.endtime or self.imufile.isEof():  # 判断不大于IMU数据长度
            #             break
            #
            #         self.giengine.addImuData(self.imu_cur)
            #         res = self.giengine.check_update()
            #         if res == 0:
            #             self.giengine.newImuProcess()
            #             self.current_step = self.current_step + 1  # 需要注意，current_step 需要和 self.giengine.timestamp() 对应
            #
            #         navstate = self.giengine.getNavState()
            #         result = np.array([
            #             np.round(navstate.pos[0] * Angle.R2D, 9),
            #             np.round(navstate.pos[1] * Angle.R2D, 9),
            #             np.round(navstate.pos[2], 9),
            #             np.round(navstate.vel[0], 9),
            #             np.round(navstate.vel[1], 9),
            #             np.round(navstate.vel[2], 9),
            #             np.round(navstate.euler[0] * Angle.R2D, 9),
            #             np.round(navstate.euler[1] * Angle.R2D, 9),
            #             np.round(navstate.euler[2] * Angle.R2D, 9)])
            #
            #         nav_result.append(result)
            #     if nav_result:
            #         ins_result = np.array(nav_result)
            #         self.set_results(ins_result)  # 把新的导航结果赋值回去
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step+1: , ['Latitude_RLpredict']] = None
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step+1: , ['Longitude_RLpredict']] = None
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step+1: , ['Ellipsoid_Height_RLpredict']] = None
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step+1: , ['Velocity_X_RLpredict']] = None
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step+1: , ['Velocity_Y_RLpredict']] = None
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step+1: , ['Velocity_Z_RLpredict']] = None
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step+1: , ['Roll_RLpredict']] = None
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step+1: , ['Pitch_RLpredict']] = None
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step+1: , ['Heading_RLpredict']] = None
        else:
            obs = self._next_observation()
        return obs, reward, self.done, {'tripIDnum': self.tripIDnum, 'current_step': self.current_step,'baseline': self.baseline,
                                   'error':error, 'tripid':self.tripIDnum, 'break': self.early_break}

class Continuous_PrePosCov_InHiGNSSPosCov_RTK(baseEnv):
    # 专门处理RTK解状态不固定环境
    metadata = {'render.modes': ['human']}
    def __init__(self, config, **kwargs):
        # def __init__(self,trajdata_range, action_scale, discrete_actionspace, reward_setting, trajdata_sort, baseline_mod):
        super(Continuous_PrePosCov_InHiGNSSPosCov_RTK, self).__init__(config,**kwargs)
        # TODO: 确认 ino和 his 的状态维度
        self.State_Dim = 3
        self.pos_weight = config["env_para"]['pos_weight']
        self.vel_weight = 0
        self.att_weight = 0
        self.observation_space = spaces.Dict(
            {'gnss': spaces.Box(low=-1, high=1, shape=(1, 10)),
             'innovation': spaces.Box(low=0, high=1, shape=(1, self.State_Dim), dtype=np.float),
             'History': spaces.Box(low=0, high=1, shape=(1, (self.pos_num-1) * self.State_Dim), dtype=np.float),
             'State': spaces.Box(low=0, high=1, shape=(1, self.State_Dim), dtype=np.float),
             'Cov': spaces.Box(low=0, high=1, shape=(1, self.State_Dim), dtype=np.float)})

        # 设置动作尺度
        self.pred_scale = config["env_para"]['continuous_scale_state_pred']
        self.action_space = spaces.Box(low=self.continuous_actionspace[0], high=self.continuous_actionspace[1], shape=(1, 2*self.State_Dim),dtype=np.float)
        self.conv_mode = config["env_para"]['conv_mode']
        self.cov_adj = config["env_para"]['cov_adj']
        self.fusion_fre = 1 # RL+KF融合频率 (Hz)

    def _normalize_his_diff(self, state):
        if self.cord == 'NED':
            state[0:2,:] = state[0:2,:] / 30
        elif self.cord == 'LLH':
            state[0:2, :] = state[0:2, :] / 1e-5
        state[2,:] = state[2,:] / 2.5
        return state * self.fusion_fre # gnss 频率不一样

    def _normalize_state(self, pre_state):
        if self.cord == 'NED':
            pre_state[0] = pre_state[0] / 50
            pre_state[1] = pre_state[1] / 50
        elif self.cord == 'LLH':
            pre_state[0] = pre_state[0] / 30
            pre_state[1] = pre_state[1] / 120
        pre_state[2] = pre_state[2] / 5
        return pre_state

    def _next_observation(self):
        # 在这里接入INS惯性推算，注意这里不会进行融合
        nav_result = []
        while not self.fusing: # 当开始融合时，不会再进入此循环
            if self.gnss.time < self.imu_cur.time and not self.gnssfile.isEof():  # 判断不大于数据长度
                self.gnss = self.gnssfile.next()
                self.giengine.addGnssData(self.gnss)

            # TODO: 验证回合终止条件
            if self.datatime[self.current_step] >= self.end_step_time: # 当前时间大于设置数据时间，结束回合
                self.done = True

            self.imu_cur = self.imufile.next()
            if self.imu_cur.time > self.endtime or self.imufile.isEof():  # 判断不大于IMU数据长度
                self.done = True
                break

            self.giengine.addImuData(self.imu_cur)
            res = self.giengine.check_update()
            cur_datatime = self.datatime[self.current_step]
            if (cur_datatime - self.pre_fusion_time) < (1/self.fusion_fre):
                self.giengine.newImuProcess()
                self.current_step = self.current_step + 1  # 需要注意，current_step 需要和 self.giengine.timestamp() 对应
            else:
                self.pre_fusion_time = cur_datatime
                self.fusing = True
                break # 转移到step函数再融合

            navstate = self.giengine.getNavState()
            result = np.array([
                    np.round(navstate.pos[0] * Angle.R2D, 9),
                    np.round(navstate.pos[1] * Angle.R2D, 9),
                    np.round(navstate.pos[2], 9),
                    np.round(navstate.vel[0], 9),
                    np.round(navstate.vel[1], 9),
                    np.round(navstate.vel[2], 9),
                    np.round(navstate.euler[0] * Angle.R2D, 9),
                    np.round(navstate.euler[1] * Angle.R2D, 9),
                    np.round(navstate.euler[2] * Angle.R2D, 9)])

            nav_result.append(result)

        # TODO: 是否可以精简一下不要self.baseline
        if nav_result:
            ins_result = np.array(nav_result)
            self.set_results(ins_result) # 把新的导航结果赋值回去

        # 构建 self.pos_num -1 的历史序列拼接
        seq_len = self.pos_num * self.imurate/self.fusion_fre # 根据频率获取窗口历元大小,预计大概是1s一个步长
        his_len = len(self.baseline.loc[self.current_step-seq_len: self.current_step, 'Latitude_RLpredict'].values)
        indices = np.linspace(0, his_len-1, num=self.pos_num, dtype=int)
        Lat_seq = self.baseline.loc[self.current_step-seq_len: self.current_step, 'Latitude_RLpredict'].values[indices]
        Lon_seq = self.baseline.loc[self.current_step-seq_len: self.current_step, 'Longitude_RLpredict'].values[indices]
        Hei_seq = self.baseline.loc[self.current_step-seq_len: self.current_step, 'Ellipsoid_Height_RLpredict'].values[indices]
        obs = np.array([Lat_seq,Lon_seq,Hei_seq])

        # 测试代码用
        # Lat_seq_gt = self.baseline.loc[self.current_step-seq_len: self.current_step, ' Latitude_GT (deg)'].values[indices]
        # Lon_seq_gt = self.baseline.loc[self.current_step-seq_len: self.current_step, ' Longitude_GT (deg)'].values[indices]
        # Hei_seq_gt = self.baseline.loc[self.current_step-seq_len: self.current_step, ' Ellipsoid Height_GT (m)'].values[indices]
        # obs_gt = np.array([Lat_seq,Lon_seq,Hei_seq])
        # obs_gt = np.diff(obs_gt, axis=1)
        # if self.cord == 'NED': # 地理坐标系转到导航坐标系
        #     blh_station = self.start_pos[0:3]
        #     rm, rn = radiusmn(blh_station[0])
        #     obs_gt[0:2, :] = obs_gt[0:2, :] * Angle.D2R
        #     for i in range(obs_gt.shape[1]):
        #         obs_gt[0:3,i] = drad2dm(rm, rn, blh_station, obs_gt[0:3,i]).reshape(3)

        # 历史状态差分
        obs = np.diff(obs, axis=1)

        # 确定坐标系
        if self.cord == 'NED': # 地理坐标系转到导航坐标系
            blh_station = self.start_pos[0:3]
            rm, rn = radiusmn(blh_station[0])
            obs[0:2, :] = obs[0:2, :] * Angle.D2R
            for i in range(obs.shape[1]):
                obs[0:3,i] = drad2dm(rm, rn, blh_station, obs[0:3,i]).reshape(3)

        obs = self._normalize_his_diff(obs)

        # gnss feature process
        # 'Satnum', 'CNR_Mean','CNR_Q75','CNR_Q25','EA_Mean','EA_Q75','EA_Q25','PDOP', 'HDOP','Q'
        if not self.gnssfile.isEof() and np.abs(self.imu_cur.time - self.gnss.time) < GNSS_VALID:
            obs_feature = self.gnssfile.data_[self.gnssfile.index,7:].copy()
            obs_feature = self._normalize_gnss(obs_feature)
            # 获取RTK状态
            self.RTK_cur_state = self.gnssfile.data_[self.gnssfile.index, 16].copy()
            if self.RTK_cur_state > 2:
                self.RTK_cur_state -= 1
            # self.RTK_std = self.gnssfile.data_[self.gnssfile.index, 4:7].copy()
        else:
            obs_feature = np.array([0.01,0.01,0.01,0.01,0.01,0.01,0.01,1,1,1.2])
            self.RTK_cur_state = 6

        # state feature process
        if self.cord == 'NED':  # 地理坐标系转到导航坐标系
            n,e,d = pm.geodetic2ned(self.RL_prestate[0], self.RL_prestate[1], self.RL_prestate[2], self.start_pos[0], self.start_pos[1], self.start_pos[2])
            pre_state = np.array([n,e,d])
        elif self.cord == 'LLH':
            pre_state = self.RL_prestate[0:3].copy()
        pre_state = self._normalize_state(pre_state)

        # 根据解状态变化手动调整方差
        if (self.RTK_cur_state - self.RTK_pre_state) > 0:
            self.RLcov = self.RLcov * self.cov_adj ** (self.RTK_cur_state - self.RTK_pre_state)
            self.policy_cov_scale *= self.cov_adj ** (self.RTK_cur_state - self.RTK_pre_state)
        elif (self.RTK_cur_state - self.RTK_pre_state) < 0:
            self.RLcov = self.RLcov / (self.cov_adj ** (self.RTK_pre_state - self.RTK_cur_state))
            self.policy_cov_scale /= (self.cov_adj ** (self.RTK_pre_state - self.RTK_cur_state))
        self.RTK_pre_state = self.RTK_cur_state
        # RL cov features
        rl_cov_diag = np.diag(self.RLcov) / self.config["env_para"]['initial_RLcov']

        # 观测合并
        obs_all = {'History': obs.reshape(1, obs.size, order='F'),
                   'gnss': obs_feature.reshape(1, 10),
                   'innovation': self.innovation.reshape(1, self.innovation.size),
                   'State': pre_state.reshape(1, pre_state.size),
                   'Cov': rl_cov_diag.reshape(1, rl_cov_diag.size)}

        return obs_all

    def step(self, action):  # modified in 3.3
        # done = (self.current_step >= len(self.baseline.loc[:, 'UnixTimeMillis'].values) * self.traj_type[-1] - (
        #     self.pos_num))
        # action for new prediction
        action = np.reshape(action, [1, 2 * self.State_Dim])
        predict_N = action[0, 0] * self.pred_scale
        predict_E = action[0, 1] * self.pred_scale
        predict_D = action[0, 2] * self.pred_scale * 1e-2

        blh_station = self.start_pos[0:3]
        rm, rn = radiusmn(blh_station[0])

        # 用于代码测试，不参与环境运行
        gt_col = [' Latitude_GT (deg)',' Longitude_GT (deg)',' Ellipsoid Height_GT (m)',' Velocity X_GT (m/s)', ' Velocity Y_GT (m/s)', ' Velocity Z_GT (m/s)',
                  ' Roll_GT (deg)', ' Pitch_GT (deg)', ' Heading_GT (deg)']
        gt_lat_cur = self.baseline.loc[self.current_step + 1, gt_col].values
        gt_lat_pre = self.baseline.loc[self.pre_step, gt_col].values
        d_gt = gt_lat_cur - gt_lat_pre
        d_gt_nav = drad2dm(rm, rn, blh_station, d_gt[0:3]* Angle.D2R)
        self.pre_step = self.current_step

        if self.cord == 'NED':
            d_pos_n = d_gt_nav # np.array([predict_N,predict_E,predict_D])
            d_pos_llh = dm2drad(rm, rn, blh_station, d_pos_n)
            d_pos_llh[0:2] = d_pos_llh[0:2] * Angle.R2D # 导航坐标系转大地
        elif self.cord == 'LLH':
            scale_llh = 1e-5
            d_pos_llh = np.array([predict_N * scale_llh,predict_E * scale_llh,predict_D])

        # 预测RL状态
        navstate = self.giengine.getNavState() # 导航状态的位置和姿态都是弧度制
        self.RL_prestate[0:3] = self.RL_prestate[0:3] + d_pos_llh.reshape(3) # 单位是度

        # 计算误差状态: 注意RL预测的状态是角度制，但是预测导航状态是弧度制
        Dr = Earth.DR(navstate.pos) # 地理坐标相对位置 转 n系相对位置
        RL_pos_rad = gt_lat_cur[0:3] # self.RL_prestate[0:3].copy()
        RL_pos_rad[0:2] = RL_pos_rad[0:2] * Angle.D2R
        dx_rl = Dr @ (navstate.pos - RL_pos_rad)

        # 协方差调整/预测
        d_cov_rl = np.diag(self.policy_cov_scale * action[0, self.State_Dim:])
        # TODO 协方差预测不当会造成融合后对角线变为0（是因为太小？）
        if self.prdcov_mode == "Add_mode1":
            self.RLcov = self.RLcov + d_cov_rl
            self.RLcov[self.RLcov < 1e-3] = 1e-3
            pred_cov_rl = self.RLcov
        elif self.prdcov_mode == "Add_mode2":
            pred_cov_rl = self.RLcov + d_cov_rl
            pred_cov_rl[pred_cov_rl < 1e-3] = 1e-3

        # 确定到达融合状态，GNSS/INS可融合
        if not self.fusing:
            res = self.giengine.check_update()
            raise ValueError(f"Invalid state: res is {res}. Expected res > 0 for GNSS/INS integration.")
        self.KRLF_Process(dx_rl,pred_cov_rl)
        self.fusing = False # 融合后设为False
        self.current_step = self.current_step + 1 # 当前步加1
        self.fusing_count = self.fusing_count + 1 # 融合次数加1

        # 保存数据
        navstate = self.giengine.getNavState()
        result = np.array([
            np.round(navstate.pos[0] * Angle.R2D, 9),
            np.round(navstate.pos[1] * Angle.R2D, 9),
            np.round(navstate.pos[2], 9),
            np.round(navstate.vel[0], 9),
            np.round(navstate.vel[1], 9),
            np.round(navstate.vel[2], 9),
            np.round(navstate.euler[0] * Angle.R2D, 9),
            np.round(navstate.euler[1] * Angle.R2D, 9),
            np.round(navstate.euler[2] * Angle.R2D, 9)])
        self.set_results(result.reshape(1, -1))

        # 间隔一段步数重置RL状态
        if (self.fusing_count % self.Rlstate_reset == 0):
            rl_col = ['Latitude_RLpredict','Longitude_RLpredict','Ellipsoid_Height_RLpredict','Velocity_X_RLpredict',
                      'Velocity_Y_RLpredict','Velocity_Z_RLpredict','Roll_RLpredict','Pitch_RLpredict','Heading_RLpredict']
            self.RL_prestate = self.baseline.loc[self.current_step, rl_col].values

        # reward function
        if self.flag_finetune:
            reward, error = self.reward_finetune(navstate)
        else:
            reward, error = self.reward_calculation(navstate)
        self.cumulated_reward += reward

        # Execute one time step within the environment
        if self.done:
            obs = []
            # nav_result = []
            # if self.traj_type[1] < 1:
            #     while not self.fusing:  # 最后一次推理
            #         if self.gnss.time < self.imu_cur.time and not self.gnssfile.isEof():  # 判断不大于数据长度
            #             self.gnss = self.gnssfile.next()
            #             self.giengine.addGnssData(self.gnss)
            #
            #         # TODO: 验证回合终止条件
            #         self.imu_cur = self.imufile.next()
            #         if self.imu_cur.time > self.endtime or self.imufile.isEof():  # 判断不大于IMU数据长度
            #             break
            #
            #         self.giengine.addImuData(self.imu_cur)
            #         res = self.giengine.check_update()
            #         if res == 0:
            #             self.giengine.newImuProcess()
            #             self.current_step = self.current_step + 1  # 需要注意，current_step 需要和 self.giengine.timestamp() 对应
            #
            #         navstate = self.giengine.getNavState()
            #         result = np.array([
            #             np.round(navstate.pos[0] * Angle.R2D, 9),
            #             np.round(navstate.pos[1] * Angle.R2D, 9),
            #             np.round(navstate.pos[2], 9),
            #             np.round(navstate.vel[0], 9),
            #             np.round(navstate.vel[1], 9),
            #             np.round(navstate.vel[2], 9),
            #             np.round(navstate.euler[0] * Angle.R2D, 9),
            #             np.round(navstate.euler[1] * Angle.R2D, 9),
            #             np.round(navstate.euler[2] * Angle.R2D, 9)])
            #
            #         nav_result.append(result)
            #     if nav_result:
            #         ins_result = np.array(nav_result)
            #         self.set_results(ins_result)  # 把新的导航结果赋值回去
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step+1: , ['Latitude_RLpredict']] = None
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step+1: , ['Longitude_RLpredict']] = None
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step+1: , ['Ellipsoid_Height_RLpredict']] = None
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step+1: , ['Velocity_X_RLpredict']] = None
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step+1: , ['Velocity_Y_RLpredict']] = None
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step+1: , ['Velocity_Z_RLpredict']] = None
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step+1: , ['Roll_RLpredict']] = None
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step+1: , ['Pitch_RLpredict']] = None
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step+1: , ['Heading_RLpredict']] = None
        else:
            obs = self._next_observation()
        return obs, reward, self.done, {'tripIDnum': self.tripIDnum, 'current_step': self.current_step,'baseline': self.baseline,
                                   'error':error, 'tripid':self.tripIDnum, 'break': self.early_break}




class GnssFileLoader:
    def __init__(self, gnss_data):
        self.data_ = gnss_data.copy()
        self.index = 0

    def next(self):
        if self.index >= self.data_.shape[0]:
            return None
        data_ = self.data_[self.index, :]
        gnss_ = GNSS()
        gnss_.time = data_[0]
        gnss_.blh = np.array(data_[1:4])  # 经纬高
        gnss_.std = np.array(data_[4:7])  #
        gnss_.blh[0] *= Angle.D2R
        gnss_.blh[1] *= Angle.D2R
        self.index += 1
        return gnss_

    def set_data(self,idx):
        self.index = idx
        if self.index >= self.data_.shape[0]:
            return None
        data_ = self.data_[self.index, :]
        gnss_ = GNSS()
        gnss_.time = data_[0]
        gnss_.blh = np.array(data_[1:4])  # 经纬高
        gnss_.std = np.array(data_[4:7])  #
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

class ImuFileLoader:
    def __init__(self, imu_data, rate):
        self.data_ = imu_data.copy()
        self.index = 0
        self.pre_time = self.data_[0,][0]  # 读取首次时间
        self.dt_ = 1.0 / float(rate)

    def next(self):
        # 这里的IMU计算默认是前右下坐标系，数据处理时要先统一
        if self.index >= self.data_.shape[0]:
            return None
        data_ = self.data_[self.index, :]
        pre_time = self.pre_time
        imu_ = IMU()
        imu_.time = data_[0]
        imu_.dtheta = np.array(data_[1:4])  # 角速度
        imu_.dvel = np.array(data_[4:7])  # # 加速度

        dt = imu_.time - pre_time
        if dt < 0.1:
            imu_.dt = dt
        else:
            imu_.dt = self.dt_
        # 修改tang：原来数据集应该已经乘了dt，其他数据集也要做相应处理
        imu_.dtheta = imu_.dtheta * imu_.dt
        imu_.dvel = imu_.dvel * imu_.dt

        self.index += 1
        self.pre_time = imu_.time

        return imu_

    def set_data(self, idx):
        self.index = idx
        if self.index >= self.data_.shape[0]:
            return None

        data_ = self.data_[self.index, :]
        pre_time = self.pre_time
        imu_ = IMU()
        imu_.time = data_[0]
        imu_.dtheta = np.array(data_[1:4])  # 角速度
        imu_.dvel = np.array(data_[4:7])  # # 加速度

        dt = imu_.time - pre_time
        if dt < 0.1:
            imu_.dt = dt
        else:
            imu_.dt = self.dt_

        imu_.dtheta = imu_.dtheta * imu_.dt
        imu_.dvel = imu_.dvel * imu_.dt

        self.index += 1
        self.pre_time = imu_.time
        return imu_

    def starttime(self):
        return self.data_[0, 0]

    def endtime(self):
        return self.data_[-1, 0]

    def isEof(self):
        return self.index >= self.data_.shape[0]

def update_attitude(euler_old, d_euler):
    """
    更新姿态（输入都要是弧度计算）
    :param euler_old: 旧欧拉角（弧度）
    :param d_euler: 欧拉角更新增量（弧度）
    :return: 新欧拉角（弧度）
    """
    q_old = RU.euler2quaternion( np.flip(euler_old, axis=0))
    q_delta = RU.rotvec2quaternion(d_euler)
    R_new = Rotation.from_quat(q_delta) * Rotation.from_quat(q_old)  # 姿态补偿
    q_new = R_new.as_quat()
    cbn = RU.quaternion2matrix(q_new)
    euler_new = RU.matrix2euler(cbn)
    return euler_new

def calculate_phi_error(euler_pred, euler_nominal):
    """
    计算欧拉角误差
    euler_pred: 真实欧拉角 [roll, pitch, yaw] (单位: 度)
    euler_nominal: 系统当前欧拉角 [roll, pitch, yaw] (单位: 度)
    返回: 3维旋转向量误差 (单位: 弧度)
    """
    q_true = Rotation.from_euler('zyx', [euler_pred[2], euler_pred[1], euler_pred[0]], degrees=True)
    q_nom = Rotation.from_euler('zyx', [euler_nominal[2], euler_nominal[1], euler_nominal[0]], degrees=True)

    # 计算误差旋转 q_err，满足 q_true = q_err * q_nom (左乘更新)
    q_err = q_true * q_nom.inv()

    # 转回旋转向量 (dx_phi)
    return q_err.as_rotvec()

def compute_reward(bl, rl, weight=1.0, eps=1e-6):
    # 计算对数比例，反映提升倍数
    ratio_log = np.log((bl + eps) / (rl + eps))
    # 使用 tanh 压缩到 [-1, 1]，保证梯度平滑且不爆炸
    return np.tanh(ratio_log)


def compute_normalized_innovation(dx_kf, dx_rl, cov_kf, cov_rl):
    """
    复现公式: (H * P_R * H^T + P_K)^-1/2 * (dx_K - H^T * dx_R)
    """
    # 1. 计算残差向量 vt (假设 H 已经转置匹配维度)
    # 如果 H 是观测矩阵，dx_rl 在观测空间，则为 dx_kf - H.T @ dx_rl
    vt = dx_kf - dx_rl

    # 2. 计算合成协方差矩阵 S
    S = cov_rl + cov_kf

    # 3. 计算 S 的 -1/2 次幂
    # 方法 A: 使用 scipy.linalg.sqrtm (最通用但稍慢)
    # S_inv_half = inv(sqrtm(S))

    # 方法 B: 使用特征值分解 (更稳定，适合对称正定矩阵)
    eigenvalues, eigenvectors = np.linalg.eigh(S)
    # 确保特征值为正（处理类似 -5e-9 的数值噪声）
    eigenvalues = np.maximum(eigenvalues, 1e-12)
    # S^-1/2 = Q * Lambda^-1/2 * Q^T
    S_inv_half = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T

    # 4. 最终计算
    result = S_inv_half @ vt
    Nis = vt.T @ result
    return result, Nis