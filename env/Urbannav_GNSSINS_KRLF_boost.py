# 强化学习定位环境构建
import gym
from gym import spaces
import random
import pickle
import os
import sys
import numpy as np
import pandas as pd
from env.env_param import *
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
from fileio.gnssfileloader_Tang import GnssFileLoader
from fileio.imufileloader_tang import ImuFileLoader
from scipy.stats import chi2
from kfgins.gi_engine import GIEngine
import yaml
import src.gnss_lib.coordinates as coord

"""
香港GNSS/INS环境，加速改进版
"""

step_print = False
# 导入数据
dir_path = '/mnt/sdb/home/tangjh/KF-GINS-Py-main/'  # '/home/tangjh/smartphone-decimeter-2022/''D:/jianhao/smartphone-decimeter-2022/'
# load raw baseline data
with open(dir_path + 'env/raw_baseline_gnssins_Urbannav.pkl', "rb") as file:
    data_truth_dic = pickle.load(file)
file.close()
# load raw imu data
with open(dir_path + 'env/raw_imu_data_Urbannav.pkl', "rb") as file:
    data_raw_imu_dic = pickle.load(file)
file.close()
# load raw gnss data
with open(dir_path + 'env/raw_gnss_data_Urbannav.pkl', "rb") as file:
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
    """
    260531修改：给高度奖励设置比例因子
    """
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
        self.vel_weight = config["env_para"]['vel_weight']
        self.att_weight = config["env_para"]['att_weight']
        self.reward_setting = config["env_para"]['reward_setting']
        self.Rlstate_reset = config["env_para"]['RL_reset_step']
        self.prdcov_mode = config["env_para"]['prdcov_mode']
        self.cumulated_reward = 0
        self.count = 0
        self.max_return = 0
        self.early_break = False #早停条件
        # 微调专用变量
        self.flag_finetune = kwargs.get('finetuning',False) # 微调判断使用
        self.config_tuning = kwargs.get('config_tuning',None) # 微调参数表
        self.dvel_list = []
        self.dcbn_list = []
        if self.flag_finetune:
            self.pos_weight = self.config_tuning["finetuning_settings"]['pos_weight']
            self.vel_weight = self.config_tuning["finetuning_settings"]['vel_weight']
            self.att_weight = self.config_tuning["finetuning_settings"]['att_weight']
            self.nis_weight = self.config_tuning["finetuning_settings"]['nis_weight']

            # 按顺序轨迹还是打乱顺序
        if self.trajdata_sort == 'sorted':
            self.tripIDnum = self.trajdata_range[0]
        elif self.trajdata_sort == 'randint':
            sublist = self.tripIDlist[self.trajdata_range[0]:self.trajdata_range[1]]
            random.shuffle(sublist)
            self.tripIDlist[self.trajdata_range[0]:self.trajdata_range[1]] = sublist
            self.tripIDnum = self.trajdata_range[0]-2

        # ⚠️改进260519：设置baseline列名
        self.cols_rl_predict = [
            'Latitude_RLpredict', 'Longitude_RLpredict', 'Ellipsoid_Height_RLpredict',
            'Velocity_X_RLpredict', 'Velocity_Y_RLpredict', 'Velocity_Z_RLpredict',
            'Roll_RLpredict', 'Pitch_RLpredict', 'Heading_RLpredict'
        ]
        self.cols_gt = [
            ' Latitude_GT (deg)', ' Longitude_GT (deg)', ' Ellipsoid Height_GT (m)',
            ' Velocity Y_GT (m/s)', ' Velocity X_GT (m/s)', ' Velocity Z_GT (m/s)',
            ' Roll_GT (deg)', ' Pitch_GT (deg)', ' Heading_GT (deg)'
        ] # 这里顺便做了坐标转换
        self.cols_bl = [
            ' Latitude (deg)', ' Longitude (deg)', ' Ellipsoid Height (m)',
            ' Velocity X (m/s)', ' Velocity Y (m/s)', ' Velocity Z (m/s)',
            ' Roll (deg)', ' Pitch (deg)', ' Heading (deg)'
        ]
        self.multipliers = np.array([Angle.R2D, Angle.R2D, 1.0, 1.0, 1.0, 1.0, Angle.R2D, Angle.R2D, Angle.R2D])

    def reset(self):
        # Reset the state of the environment to an initial state
        self.done = False
        self.fusing = False # 是否开始融合
        self.fusing_count = 0 # 记录融合了几次
        self.RTK_pre_state = 1  # RTK上一步解状态，默认是固定
        self.RTK_cur_state = 1 # RTK当前解状态
        self.RTK_std = np.zeros([3,1]) # RTK标准差
        # 微调修改260607：清空微调用速度差分list
        self.dvel_list = []
        self.dcbn_list = []
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
                if self.cumulated_reward > self.max_return:
                    self.count = 0
                    self.max_return = self.cumulated_reward

        # self.tripIDnum=tripIDnum
        # self.info['tripIDnum']=self.tripIDnum
        self.baseline = data_truth_dic[self.tripIDlist[self.tripIDnum]].copy()
        self.raw_gnss = data_raw_gnss_dic[self.tripIDlist[self.tripIDnum]].copy()
        self.raw_imu = data_raw_imu_dic[self.tripIDlist[self.tripIDnum]].copy()
        self.datatime = self.baseline['UnixTimeMillis_ref'].values
        self.timeend = self.baseline.loc[len(self.baseline.loc[:, 'UnixTimeMillis_ref'].values) - 1, 'UnixTimeMillis_ref']
        self.start_pos = self.baseline.loc[0, [' Latitude (deg)',' Longitude (deg)',' Ellipsoid Height (m)']].values # 初始位置，用于坐标转换
        # self.start_pos[0:2] *= Angle.D2R

        # normalize baseline
        # self.baseline['LatitudeDegrees_norm'] = (self.baseline['LatitudeDegrees']-lat_min)/(lat_max-lat_min)
        # self.baseline['LongitudeDegrees_norm'] = (self.baseline['LongitudeDegrees']-lon_min)/(lon_max-lon_min)
        # gen pred

        # 初始化准备导航状态
        config_filename = os.path.abspath(f'{dir_path}/dataset_Urbannav/{self.tripIDlist[self.tripIDnum]}/kf-gins.yaml')
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
            # ⚠️改进260519：简化赋值设计
            self.baseline[self.cols_rl_predict] = self.baseline[self.cols_bl].values

        # ⚠️改进260519：创建三个缓冲区
        self.baseline_buffer = self.baseline[self.cols_rl_predict].values.astype(np.float64)  # RL预测，动态读写
        self.gt_buffer = self.baseline[self.cols_gt].values.astype(np.float64)  # 真值，只读
        self.bl_buffer = self.baseline[self.cols_bl].values.astype(np.float64)

        # Set the current step to a random point within the data frame
        # 数据对齐, 时间移到定义的开始时间
        start_data_step = int(np.ceil(len(self.baseline) * self.traj_type[0])) # 按照比例获得当前gnss索引
        # self.gnss_step = start_data_step + self.pos_num - 1 # 移动到序列后第一个点
        # gnss_time = self.raw_gnss[self.gnss_step,0] # 获取当前的gnss时间
        # nav_idx, cur_nav_time = self.find_time(self.datatime, gnss_time) # 查找当前imu时间
        # self.current_step = nav_idx - 2 # 当前的导航步已经索引到当前开始RL预测的步，环境从该步开始，current_step是针对baseline而言
        self.current_step = int(start_data_step + self.pos_num * self.imurate/self.gnssrate) -2  # self.pos_num 步后的baseline idx
        # ⚠️改进260519：增加全局初始时间
        self.start_step = self.current_step
        self.pre_step = int(start_data_step + (self.pos_num-1) * self.imurate/self.gnssrate)

        # 设置 初始RL独立维护的状态和协方差
        # gnss_time = self.raw_gnss[self.gnss_step-1, 0]  # 获取当前的前一步gnss时间
        # nav_idx, _ = self.find_time(self.datatime, gnss_time)
        # ⚠️改进260519：直接从缓冲区拿，不用管 Pandas 原表结构怎么变，绝对安全
        self.RL_prestate = self.baseline_buffer[self.pre_step, :].copy()
        self.RLcov = np.eye(self.State_Dim) * self.config["env_para"]['initial_RLcov']
        self.policy_cov_scale = self.config["env_para"]['continuous_scale_policy_cov']
        self.pre_fusion_time = self.datatime[self.pre_step] # 初次融合时间

        # 结束时间戳
        end_data_step = int(np.ceil(len(self.baseline) * self.traj_type[1]))
        self.end_step_time = self.datatime[int(end_data_step-self.imurate/self.gnssrate)]

        # if self.traj_type[0] > 0:  # 只要剩下部分轨迹的定位结果
        #     data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[0:self.current_step, self.cols_rl_predict] = np.nan

        # 构建GNSS/INS推理引擎
        self.giengine = KRLF_GIEngine(options)

        if self.traj_config["endtime"] < 0:
            self.endtime = self.imufile.endtime()
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
            dvel,dcbn = self.giengine.newImuProcess()
            self.dvel_list.append(dvel)
            self.dcbn_list.append(dcbn)

            progress = (self.giengine.timestamp() - starttime) / (self.datatime[self.current_step] - starttime) * 100.0
            sys.stdout.write('Reseting: \r[{:.2f}%]'.format(progress) + str(self.giengine.timestamp()))  # 创建动态进度显示
            sys.stdout.flush()

        self.innovation = 0.1 * np.ones([1,self.State_Dim])
        self.innovation_correct = 0.1 * np.ones([1,3])
        self.NIS = 0 # 信息
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
        # ⚠️改进修正260519：仅对 NumPy 缓冲区进行快速切片赋值
        start_idx = self.current_step - ins_result.shape[0] + 1
        end_idx = self.current_step + 1
        data_len = end_idx - start_idx
        if ins_result.shape[0] > data_len:
            ins_result = ins_result[:data_len,:]

        self.baseline_buffer[start_idx:end_idx, :] = ins_result
        # data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step - ins_result.shape[0] +1: self.current_step,['Latitude_RLpredict']] = ins_result[:, 0].reshape(-1,1)
        # data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step - ins_result.shape[0] +1: self.current_step,['Longitude_RLpredict']] = ins_result[:, 1].reshape(-1,1)
        # data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step - ins_result.shape[0] +1: self.current_step,['Ellipsoid_Height_RLpredict']] = ins_result[:, 2].reshape(-1,1)
        # data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step - ins_result.shape[0] +1: self.current_step,['Velocity_X_RLpredict']] = ins_result[:, 3].reshape(-1,1)
        # data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step - ins_result.shape[0] +1: self.current_step,['Velocity_Y_RLpredict']] = ins_result[:, 4].reshape(-1,1)
        # data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step - ins_result.shape[0] +1: self.current_step,['Velocity_Z_RLpredict']] = ins_result[:, 5].reshape(-1,1)
        # data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step - ins_result.shape[0] +1: self.current_step,['Roll_RLpredict']] = ins_result[:, 6].reshape(-1,1)
        # data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step - ins_result.shape[0] +1: self.current_step,['Pitch_RLpredict']] = ins_result[:, 7].reshape(-1,1)
        # data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step - ins_result.shape[0] +1: self.current_step,['Heading_RLpredict']] = ins_result[:, 8].reshape(-1,1)

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
            dvel, dcbn = self.giengine.insPropagation(imupre_, imucur_)
        elif res == 2:  # 更新时间靠近imutime2
            # GNSS数据靠近当前历元，先对当前IMU进行状态传播
            dvel, dcbn = self.giengine.insPropagation(imupre_, imucur_)
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
            middvel_f, middcbn_f = self.giengine.insPropagation(imupre_, midimu)

            # 整秒时刻进行GNSS更新，并反馈系统状态
            # do GNSS position update at the whole second and feedback system states
            self.giengine.gnssUpdate(gnssdata_)
            self._error_state_fusion(dx_rl, cov_rl)
            self.giengine.stateFeedback()

            # 对后一半IMU进行状态传播
            # propagate navigation state for the second half imudata
            self.giengine.pvapre_ = self.giengine.pvacur_
            middvel_b, middcbn_b = self.giengine.insPropagation(midimu, imucur_)

            # 计算全程增量
            dvel = middvel_f + middvel_b
            dcbn = middcbn_f @ middcbn_b


        # 更新上一时刻的状态和IMU数据
        # update system state and imudata at the previous epoch
        self.giengine.pvapre_ = self.giengine.pvacur_
        self.giengine.imupre_ = self.giengine.imucur_
        return dvel, dcbn

    def _error_state_fusion(self,dx_rl,cov_rl):  # RL for KF modified in 0303
        dx_kf = self.giengine.dx_[0:self.State_Dim].copy() # 只获取 pva
        dx_rl = dx_rl.reshape([self.State_Dim,1])
        cov_kf_pva = self.giengine.Cov_[StateID.P_ID:StateID.P_ID + self.State_Dim, StateID.P_ID:StateID.P_ID + self.State_Dim].copy() # kf的pva协方差
        # 设置inovation参数
        self.innovation, self.Nis = compute_normalized_innovation(dx_kf, dx_rl, cov_kf_pva, cov_rl)
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

    def compute_innovation(self):
        # IMU位置转到GNSS天线相位中心位置
        # convert IMU position to GNSS antenna phase center position
        gnssdata = self.giengine.gnssdata_

        Dr_inv = Earth.DRi(self.giengine.pvacur_.pos)  # n系相对位置转地理坐标相对位置
        Dr = Earth.DR(self.giengine.pvacur_.pos)  # # 地理坐标相对位置转n系相对位置
        antenna_pos = self.giengine.pvacur_.pos + Dr_inv @ self.giengine.pvacur_.att.cbn @ self.giengine.options_.antlever

        # GNSS位置测量新息
        # compute GNSS position innovation
        dz = Dr @ (antenna_pos - gnssdata.blh)  # 相当于单位变成米？

        # 构造GNSS位置观测矩阵
        # construct GNSS position measurement matrix
        H_gnsspos = np.zeros((3, self.giengine.Cov_.shape[0]))
        H_gnsspos[0:3, StateID.P_ID:StateID.P_ID + 3] = np.identity(3)
        H_gnsspos[0:3, StateID.PHI_ID:StateID.PHI_ID + 3] = RU.skewSymmetric(
            self.giengine.pvacur_.att.cbn @ self.giengine.options_.antlever)  # StateID.PHI 姿态ID

        dz = dz.reshape(3, 1)
        innovation = dz - H_gnsspos @ self.giengine.dx_
        return innovation[0:3]

    def reward_calculation(self,navstate):
        if self.baseline_mod == 'GNSS/INS':
            # 💥 修正260519：从专门的原始基线缓冲区 (bl_buffer) 中读取，绝对不会被 RL_predict 污染
            bl_lat = self.bl_buffer[self.current_step, 0] * Angle.D2R
            bl_lon = self.bl_buffer[self.current_step, 1] * Angle.D2R
            bl_h = self.bl_buffer[self.current_step, 2]
            bl_vx = self.bl_buffer[self.current_step, 3]
            bl_vy = self.bl_buffer[self.current_step, 4]
            bl_vz = self.bl_buffer[self.current_step, 5]
            bl_roll = self.bl_buffer[self.current_step, 6]
            bl_pitch = self.bl_buffer[self.current_step, 7]
            bl_heading = self.bl_buffer[self.current_step, 8]

        gt_lat = self.gt_buffer[self.current_step, 0] * Angle.D2R
        gt_lon = self.gt_buffer[self.current_step, 1] * Angle.D2R
        gt_h = self.gt_buffer[self.current_step, 2]
        gt_vx = self.gt_buffer[self.current_step, 3]
        gt_vy = self.gt_buffer[self.current_step, 4]
        gt_vz = -self.gt_buffer[self.current_step, 5]  # enu 转 ned
        gt_roll = self.gt_buffer[self.current_step, 6]
        gt_pitch = self.gt_buffer[self.current_step, 7]
        gt_heading = self.gt_buffer[self.current_step, 8]

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
            reward = self.pos_weight * (mse_bl_p - mse_rl_p) + self.vel_weight * (mse_bl_v - mse_rl_v) + self.att_weight * (mse_bl_a - mse_rl_a)
            reward = np.clip(reward, -10, 10)
        elif self.reward_setting == 'RMSEadv_ratio': # (or误差-rl误差)/or误差
            reward = self.pos_weight * (mse_bl_p - mse_rl_p)/mse_bl_p + self.vel_weight * (mse_bl_v - mse_rl_v)/mse_bl_v + self.att_weight * (mse_bl_a - mse_rl_a)/mse_bl_a
            reward = np.clip(reward, -1.5* (self.pos_weight+self.vel_weight+self.att_weight), 1.5*(self.pos_weight+self.vel_weight+self.att_weight))
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

    def reward_finetune(self):
        dt = self.config_tuning['finetuning_settings']['dt'] # 微调计算奖励时间间隔
        start_idx = self.current_step - dt
        end_idx = self.current_step
        # 这里的数值代表“当误差达到这个值时，单项奖励降到约 0.36 (即 exp(-1))”
        scale_p = 2.0  # 位置容忍误差 (米)
        scale_v = 0.5  # 速度容忍误差 (米/秒)
        scale_att = 0.05  # 姿态容忍误差 (弧度，0.05rad 约 2.8度)

        # 位置差分计算
        pre_pos = self.baseline_buffer[self.current_step - dt,0:3].copy()
        cur_pos = self.baseline_buffer[self.current_step,0:3].copy()
        pre_pos[0:2] *= Angle.D2R
        cur_pos[0:2] *= Angle.D2R
        velocities = self.baseline_buffer[start_idx:end_idx, 3:6]
        step_dts = self.datatime[start_idx + 1: end_idx + 1] - self.datatime[start_idx: end_idx]
        # 向量化直接计算理论增量
        dpos_all = np.sum(velocities * step_dts[:, np.newaxis], axis=0)
        p_res = cur_pos - pre_pos - Earth.DRi(pre_pos) @ dpos_all
        # 计算n系误差
        rm, rn = radiusmn(pre_pos[0])
        r_p_res_n = drad2dm(rm, rn, pre_pos, p_res).reshape(1, 3)
        r_p_res_n = np.sqrt(np.sum(r_p_res_n ** 2))
        score_p = np.exp(-(r_p_res_n / scale_p))

        # 速度计算
        pre_vel = self.baseline_buffer[self.current_step - dt,3:6].copy()
        cur_vel = self.baseline_buffer[self.current_step,3:6].copy()
        dvel_all = np.sum(self.dvel_list[-dt:], axis=0)
        r_v_res = cur_vel - pre_vel - dvel_all
        r_v_res = np.sqrt(np.sum(r_v_res ** 2))
        score_v = np.exp(- (r_v_res / scale_v))

        # 姿态计算
        pre_att = self.baseline_buffer[self.current_step - dt,6:9].copy()
        cur_att = self.baseline_buffer[self.current_step, 6:9].copy()
        pre_cbn = RU.euler2matrix(pre_att * Angle.D2R)
        cur_cbn = RU.euler2matrix(cur_att * Angle.D2R)
        delta_C_bn = np.eye(3)
        for dcbn in self.dcbn_list[-dt:]:
            delta_C_bn = delta_C_bn @ dcbn

        delta_R_actual = pre_cbn.T @ cur_cbn
        error_matrix = delta_R_actual @ delta_C_bn.T
        trace = np.trace(error_matrix)
        cos_theta = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)  # 截断处理：防止 numpy 浮点误差导致数值超出 [-1, 1]，引发 arccos 报错
        # 获取纯粹的角度误差 (弧度制，范围 0 到 pi)
        r_att_res = np.arccos(cos_theta)
        r_att_res = np.sqrt(np.sum(r_att_res ** 2))
        score_att = np.exp(- (r_att_res / scale_att))

        # 卡方NIS
        r_nis = np.log1p(self.NIS)/ np.log1p(3.0)
        score_nis = np.exp(- max(0, r_nis - 1.0))

        reward = self.pos_weight * score_p + self.vel_weight * score_v + self.att_weight * score_att + self.nis_weight * score_nis
        reward = reward * 3.0 - 1.5 # 尽量映射到[-1.5,1.5]范围内

        # 极端情况惩罚：如果发生了严重的发散，给一个明确的负反馈
        if r_p_res_n > 10.0 or r_nis > 20.0:
            reward -= 1.0  # 给予截断性惩罚

        return reward, r_p_res_n

    def render(self, mode='human', close=False):
        print(f'Step: {self.current_step}')
        #  print(f'reward: {self.reward}')
        print(f'total_reward: {self.total_reward}')

class Continuous_PreStaCov_InHiGNSSStaCov(baseEnv):
    # 连续动作的状态预测和协方差预测（该环境只预测PVA），输入新息 历史序列 GNSS特征
    metadata = {'render.modes': ['human']}
    def __init__(self, config, **kwargs):
        # def __init__(self,trajdata_range, action_scale, discrete_actionspace, reward_setting, trajdata_sort, baseline_mod):
        super(Continuous_PreStaCov_InHiGNSSStaCov, self).__init__(config,**kwargs)
        # TODO: 确认 ino和 his 的状态维度
        self.State_Dim = 9
        self.pos_weight = 1
        self.observation_space = spaces.Dict(
            {'gnss': spaces.Box(low=-1, high=1, shape=(1, 10)),
             'innovation': spaces.Box(low=0, high=1, shape=(1, self.State_Dim), dtype=np.float),
             'History': spaces.Box(low=0, high=1, shape=(1, (self.pos_num-1) * self.State_Dim), dtype=np.float),
             'State': spaces.Box(low=0, high=1, shape=(1, self.State_Dim), dtype=np.float),
             'Cov': spaces.Box(low=0, high=1, shape=(1, self.State_Dim), dtype=np.float)})

        # 设置动作尺度
        self.pred_scale = config["env_para"]['continuous_scale_state_pred']
        self.vel_scale = config["env_para"]['continuous_scale_vel_pred']
        self.att_scale = config["env_para"]['continuous_scale_att_pred']
        self.policy_cov_scale = config["env_para"]['continuous_scale_policy_cov']
        self.action_space = spaces.Box(low=self.continuous_actionspace[0], high=self.continuous_actionspace[1], shape=(1, 9+9),dtype=np.float)
        self.conv_mode = config["env_para"]['conv_mode']

    def _next_observation(self):
        # 在这里拼入INS纯捷联计算？
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
            try:
                self.set_results(ins_result) # 把新的导航结果赋值回去
            except:
                pass

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

        # 当前状态拼接
        # if self.baseline_mod == 'GNSS/INS':
        #     col_rl = self.baseline.columns.get_loc('Latitude_RLpredict')
        #     current_state = self.baseline.iloc[self.current_step-1, col_rl:col_rl+9].values
        #     obs = np.column_stack((obs, current_state))
        # 历史状态差分
        obs = np.diff(obs, axis=1)
        # 航向角误差处理
        for i in range(obs.shape[1]):
            if obs[8, i] > 180:
                obs[8, i] -= 360
            if obs[8, i] < -180:
                obs[8, i] += 360

        # 确定坐标系
        if self.cord == 'NED': # 地理坐标系转到导航坐标系
            blh_station = self.start_pos[0:3]
            rm, rn = radiusmn(blh_station[0])
            obs[0:2, :] = obs[0:2, :] * D2R
            for i in range(obs.shape[1]):
                obs[0:3,i] = drad2dm(rm, rn, blh_station, obs[0:3,i]).reshape(3)

        obs = self._normalize_his_diff(obs)

        # gnss feature process
        # 'Satnum', 'CNR_Mean','CNR_Q75','CNR_Q25','EA_Mean','EA_Q75','EA_Q25','PDOP', 'HDOP','Q'
        obs_feature = self.gnssfile.data_[self.gnssfile.index,7:]
        obs_feature = self._normalize_gnss(obs_feature)

        # state feature process
        if self.cord == 'NED':  # 地理坐标系转到导航坐标系
            n,e,d = pm.geodetic2ned(self.RL_prestate[0], self.RL_prestate[1], self.RL_prestate[2], self.start_pos[0]*Angle.R2D, self.start_pos[1]*Angle.R2D, self.start_pos[2])
            pre_pos = np.array([n,e,d])
            pre_state = np.concatenate((pre_pos,self.RL_prestate[3:]))
        elif self.cord == 'LLH':
            pre_state = self.RL_prestate.copy()
        pre_state = self._normalize_state(pre_state)

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
        action = np.reshape(action, [1, 18])
        predict_N = action[0, 0] * self.pred_scale
        predict_E = action[0, 1] * self.pred_scale
        predict_D = action[0, 2] * self.pred_scale * 1e-2
        predict_vx = action[0, 3] * self.vel_scale
        predict_vy = action[0, 4] * self.vel_scale
        predict_vz = action[0, 5] * self.vel_scale * 1e-2
        predict_roll = action[0, 6] * self.att_scale * 1e-2
        predict_pitch = action[0, 7] * self.att_scale * 1e-2
        predict_yaw = action[0, 8] * self.att_scale

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
        self.RL_prestate[3:6] = self.RL_prestate[3:6] + np.array([predict_vx,predict_vy,predict_vz])
        # 姿态预测
        d_euler_rad = np.array([predict_roll,predict_pitch,predict_yaw]) * Angle.D2R
        self.RL_prestate[6:9] = update_attitude(self.RL_prestate[6:9]* Angle.D2R, d_euler_rad) * Angle.R2D

        # 计算误差状态: 注意RL预测的状态是角度制，但是预测导航状态是弧度制
        Dr = Earth.DR(navstate.pos) # 地理坐标相对位置 转 n系相对位置
        RL_pos_rad = self.RL_prestate[0:3].copy()
        RL_pos_rad[0:2] = RL_pos_rad[0:2] * Angle.D2R
        dx_rl_p = Dr @ (navstate.pos - RL_pos_rad)
        dx_rl_v = navstate.vel - self.RL_prestate[3:6]
        dx_rl_a = calculate_phi_error(self.RL_prestate[6:9], navstate.euler * Angle.R2D)
        dx_rl = np.concatenate((dx_rl_p, dx_rl_v, dx_rl_a)) # 得到rl的误差状态估计，已经是符合估计误差状态的坐标系和单位

        # 协方差调整/预测
        d_cov_rl = np.diag(self.policy_cov_scale * action[0, 9:])
        if self.prdcov_mode == "Add_mode1":
            self.RLcov = self.RLcov + d_cov_rl
            self.RLcov[self.RLcov < 1e-6] = 1e-6
            pred_cov_rl = self.RLcov
        elif self.prdcov_mode == "Add_mode2":
            pred_cov_rl = self.RLcov + d_cov_rl
            pred_cov_rl[pred_cov_rl < 1e-6] = 1e-6

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
            self.RL_prestate = self.baseline.iloc[self.current_step, 1:10].values

        # reward function
        reward, error = self.reward_calculation(navstate)
        self.cumulated_reward += reward

        # Execute one time step within the environment
        if self.done:
            obs = []
            nav_result = []
            if self.traj_type[1] < 1:
                while not self.fusing:  # 最后一次推理
                    if self.gnss.time < self.imu_cur.time and not self.gnssfile.isEof():  # 判断不大于数据长度
                        self.gnss = self.gnssfile.next()
                        self.giengine.addGnssData(self.gnss)

                    # TODO: 验证回合终止条件
                    self.imu_cur = self.imufile.next()
                    if self.imu_cur.time > self.endtime or self.imufile.isEof():  # 判断不大于IMU数据长度
                        break

                    self.giengine.addImuData(self.imu_cur)
                    res = self.giengine.check_update()
                    if res == 0:
                        self.giengine.newImuProcess()
                        self.current_step = self.current_step + 1  # 需要注意，current_step 需要和 self.giengine.timestamp() 对应

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
                if nav_result:
                    ins_result = np.array(nav_result)
                    self.set_results(ins_result)  # 把新的导航结果赋值回去
        else:
            obs = self._next_observation()
        return obs, reward, self.done, {'tripIDnum': self.tripIDnum, 'current_step': self.current_step,'baseline': self.baseline,
                                   'error':error, 'tripid':self.tripIDnum, 'break': self.early_break}  # self.info#, {}# , 'data_truth_dic':data_truth_dic

class Continuous_PrePosAttCov_InHiGNSSPosCov(baseEnv):
    # 连续动作的状态预测和协方差预测（该环境只预测位置），输入新息 历史序列 GNSS特征 位置状态, 输出位置预测
    metadata = {'render.modes': ['human']}
    def __init__(self, config, **kwargs):
        # def __init__(self,trajdata_range, action_scale, discrete_actionspace, reward_setting, trajdata_sort, baseline_mod):
        super(Continuous_PrePosAttCov_InHiGNSSPosCov, self).__init__(config,**kwargs)
        # TODO: 确认 ino和 his 的状态维度
        self.State_Dim = 6
        self.pos_weight = config["env_para"]['pos_weight']
        self.vel_weight = 0
        self.threshold = chi2.ppf(0.99, self.State_Dim)
        self.h_scale = config.get("env_para", {}).get("h_scale", 0.01)
        self.n_sigma = config.get("env_para", {}).get("n_sigma", 3.0)
        self.observation_space = spaces.Dict(
            {'gnss': spaces.Box(low=-1, high=1, shape=(1, 10)),
             'innovation': spaces.Box(low=0, high=1, shape=(1, self.State_Dim), dtype=np.float),
             'History': spaces.Box(low=0, high=1, shape=(1, (self.pos_num-1) * self.State_Dim), dtype=np.float),
             'State': spaces.Box(low=0, high=1, shape=(1, self.State_Dim), dtype=np.float),
             'Cov': spaces.Box(low=0, high=1, shape=(1, self.State_Dim), dtype=np.float)})

        # 设置动作尺度
        self.pred_scale = config["env_para"]['continuous_scale_state_pred']
        self.policy_cov_scale = config["env_para"]['continuous_scale_policy_cov']
        self.att_scale = config["env_para"]['continuous_scale_att_pred']
        self.action_space = spaces.Box(low=self.continuous_actionspace[0], high=self.continuous_actionspace[1], shape=(1, 2*self.State_Dim+2*self.State_Dim),dtype=np.float)
        self.conv_mode = config["env_para"]['conv_mode']

    def _normalize_his_diff(self, state):
        if self.cord == 'NED':
            state[0:2,:] = state[0:2,:] / 30
        elif self.cord == 'LLH':
            state[0:2, :] = state[0:2, :] / 1e-5
        state[2,:] = state[2,:] / 2.5
        state[3:5, :] = state[3:5, :] / 2
        state[5, :] = state[5, :] / 20
        return state

    def _normalize_state(self, pre_state):
        if self.cord == 'NED':
            pre_state[0] = pre_state[0] / 50
            pre_state[1] = pre_state[1] / 50
        elif self.cord == 'LLH':
            pre_state[0] = pre_state[0] / 30
            pre_state[1] = pre_state[1] / 120
        pre_state[2] = pre_state[2] / 5
        pre_state[3] = pre_state[3] / 360
        pre_state[4] = pre_state[4] / 360
        pre_state[5] = pre_state[5] / 360
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
            # 提取 navstate 的值拼接成 1D 数组
            raw_state = np.concatenate((navstate.pos, navstate.vel, navstate.euler))
            # 一次性乘法和 round
            result = np.round(raw_state * self.multipliers, 9)
            nav_result.append(result)

        # TODO: 是否可以精简一下不要self.baseline
        if nav_result:
            ins_result = np.array(nav_result)
            try:
                self.set_results(ins_result) # 把新的导航结果赋值回去
            except:
                pass

        # 💥 修正260519：改进切片提取操作
        # 构建 self.pos_num -1 的历史序列拼接
        seq_len = int(self.pos_num * self.imurate / self.gnssrate)
        start_idx = self.current_step - seq_len
        end_idx = self.current_step + 1
        indices = np.linspace(start_idx, end_idx - 1, num=self.pos_num, dtype=int)
        obs_seq = self.baseline_buffer[indices][:, [0, 1, 2, 6, 7, 8]]
        obs = obs_seq.T
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
            n,e,d = pm.geodetic2ned(self.RL_prestate[0], self.RL_prestate[1], self.RL_prestate[2], self.start_pos[0]*Angle.R2D, self.start_pos[1]*Angle.R2D, self.start_pos[2])
            pre_pos = np.array([n,e,d])
            pre_state = np.concatenate((pre_pos, self.RL_prestate[6:])) # 拼接位置和姿态状态
        elif self.cord == 'LLH':
            pre_state = self.RL_prestate[0:3].copy()
        pre_state = self._normalize_state(pre_state)

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
        predict_roll = action[0, 3] * self.att_scale * 1e-2
        predict_pitch = action[0, 4] * self.att_scale * 1e-2
        predict_yaw = action[0, 5] * self.att_scale

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
        # 姿态预测
        d_euler_rad = np.array([predict_roll,predict_pitch,predict_yaw]) * Angle.D2R
        self.RL_prestate[6:9] = update_attitude(self.RL_prestate[6:9] * Angle.D2R, d_euler_rad) * Angle.R2D

        # 计算误差状态: 注意RL预测的状态是角度制，但是预测导航状态是弧度制
        Dr = Earth.DR(navstate.pos) # 地理坐标相对位置 转 n系相对位置
        RL_pos_rad = self.RL_prestate[0:3].copy()
        RL_pos_rad[0:2] = RL_pos_rad[0:2] * Angle.D2R
        dx_rl_p = Dr @ (navstate.pos - RL_pos_rad)
        dx_rl_a = calculate_phi_error(self.RL_prestate[6:9], navstate.euler * Angle.R2D)
        dx_rl = np.concatenate((dx_rl_p, dx_rl_a)) # 得到rl的误差状态估计，已经是符合估计误差状态的坐标系和单位

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

        # 确定到达融合状态，GNSS/INS可融合
        if not self.fusing:
            res = self.giengine.check_update()
            raise ValueError(f"Invalid state: res is {res}. Expected res > 0 for GNSS/INS integration.")
        self.KRLF_Process(dx_rl,pred_cov_rl)
        self.fusing = False # 融合后设为False
        self.current_step = self.current_step + 1 # 当前步加1
        self.fusing_count = self.fusing_count + 1 # 融合次数加1

        # 修改 260519：保存数据
        navstate = self.giengine.getNavState()
        # 提取 navstate 的值拼接成 1D 数组
        raw_state = np.concatenate((navstate.pos, navstate.vel, navstate.euler))
        # 一次性乘法和 round
        result = np.round(raw_state * self.multipliers, 9)
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
            # 💥 修正260519：回合结束时，一次性同步缓存回 Pandas
            self.baseline[self.cols_rl_predict] = self.baseline_buffer
            # 1. 覆盖外部 data_truth_dic 跑完的所有数据
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[:, self.cols_rl_predict] = self.baseline_buffer
            # 2. 清除未来多余的残余数据 (设为 np.nan)
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step + 1:, self.cols_rl_predict] = np.nan
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[0:self.start_step, self.cols_rl_predict] = np.nan
        else:
            obs = self._next_observation()
        return obs, reward, self.done, {'tripIDnum': self.tripIDnum, 'current_step': self.current_step,'baseline': self.baseline,
                                   'error':error, 'tripid':self.tripIDnum, 'break': self.early_break}  # self.info#, {}# , 'data_truth_dic':data_truth_dic

    def _error_state_fusion(self,dx_rl,cov_rl):  # RL for KF modified in 0303
        indices = [0, 1, 2, 6, 7, 8]
        dx_kf = self.giengine.dx_[indices].copy() # 只获取 pa
        dx_rl = dx_rl.reshape([self.State_Dim,1])
        cov_kf_p = self.giengine.Cov_[StateID.P_ID:StateID.P_ID + 3, StateID.P_ID:StateID.P_ID + 3].copy() # kf的pva协方差
        cov_kf_a = self.giengine.Cov_[StateID.PHI_ID:StateID.PHI_ID + 3, StateID.PHI_ID:StateID.PHI_ID + 3].copy() # kf的pva协方差
        cov_kf_pva = block_diag(cov_kf_p, cov_kf_a)
        # 设置inovation参数
        self.innovation, Nis = compute_normalized_innovation(dx_kf, dx_rl, cov_kf_pva, cov_rl)
        K_fusion = np.linalg.solve((cov_kf_pva + cov_rl + np.eye(self.State_Dim) * 1e-12).T, cov_kf_pva.T).T
        # 卡方检验
        if self.n_sigma * self.threshold < Nis:
            K_fusion = K_fusion * 1
            cov_rl_scale = 1
            # print(f"No RL update: NIS={Nis}")
        else:
            cov_rl_scale = 1

        dx_f = dx_kf + K_fusion @ (dx_rl - dx_kf)
        self.giengine.dx_[StateID.P_ID:StateID.P_ID+3] = dx_f[0:3] # 放置回原来向量
        self.giengine.dx_[StateID.PHI_ID:StateID.PHI_ID+3] = dx_f[3:6] # 放置回原来向量

        # 融合协方差
        try:
            fusion_gain = np.linalg.solve(cov_kf_pva + cov_rl * cov_rl_scale + np.eye(self.State_Dim) * 1e-12, cov_rl* cov_rl_scale)
            cov_f = cov_kf_pva @ fusion_gain
        except np.linalg.LinAlgError:
            # 极端情况下退回到简单加权
            cov_f = 0.5 * (cov_kf_pva + cov_rl * cov_rl_scale)

        # 3. 强制对称化与对角线保护
        cov_f = (cov_f + cov_f.T) * 0.5

        self.giengine.Cov_[StateID.P_ID:StateID.P_ID + 3, StateID.P_ID:StateID.P_ID + 3] = cov_f[0:3,0:3]
        self.giengine.Cov_[StateID.PHI_ID:StateID.PHI_ID + 3, StateID.PHI_ID:StateID.PHI_ID + 3] = cov_f[3:6,3:6]

class DA_PrePosCov_PosAttCorrectCov_InHiGNSSSatCov(baseEnv):
    # 连续动作的状态预测和协方差预测（该环境只预测位置），输入新息 历史序列 GNSS特征 状态, 输出位置预测，两个agent同时训练
    metadata = {'render.modes': ['human']}
    def __init__(self, config_agent, **kwargs):
        # def __init__(self,trajdata_range, action_scale, discrete_actionspace, reward_setting, trajdata_sort, baseline_mod):
        super(DA_PrePosCov_PosAttCorrectCov_InHiGNSSSatCov, self).__init__(config_agent,**kwargs)
        # TODO: 确认 ino和 his 的状态维度
        self.State_Dim = 6
        # self.pos_weight = config_agent["env_para"].get('pos_weight', 1)
        self.threshold = chi2.ppf(0.999, self.State_Dim)
        self.h_scale = config_agent.get("env_para", {}).get("h_scale", 0.01)
        self.n_sigma = config_agent.get("env_para", {}).get("n_sigma", 3.0)
        self.observation_space = spaces.Dict(
            {'gnss': spaces.Box(low=-1, high=1, shape=(1, 10)),
             'innovation': spaces.Box(low=0, high=1, shape=(1, self.State_Dim), dtype=np.float),
             'History': spaces.Box(low=0, high=1, shape=(1, (self.pos_num-1) * self.State_Dim), dtype=np.float),
             'State': spaces.Box(low=0, high=1, shape=(1, self.State_Dim), dtype=np.float),
             'Cov': spaces.Box(low=0, high=1, shape=(1, self.State_Dim), dtype=np.float),
             # A2 观测
             'KF_P': spaces.Box(low=0, high=1, shape=(1, self.State_Dim), dtype=np.float),
             'innova_cor': spaces.Box(low=0, high=1, shape=(1, 3), dtype=np.float),})

        # 设置动作尺度(A1)
        self.pred_scale = config_agent["env_para"]['continuous_scale_state_pred']
        self.pred_att_scale = config_agent["env_para"]['continuous_scale_att_pred']
        self.policy_cov_scale = config_agent["env_para"]['continuous_scale_policy_cov']
        # 设置动作尺度(A2)
        self.correct_scale = config_agent["env_para"]['continuous_scale_state_correct']
        self.correct_att_scale = config_agent["env_para"]['continuous_scale_att_correct']
        self.policy_P_scale = config_agent["env_para"]['continuous_scale_P_cov']
        self.corcov_mode =  config_agent["env_para"]['corcov_mode']

        # 动作维度修改：双智能体环境下为两个智能体动作维度之和
        self.action_space = spaces.Box(low=self.continuous_actionspace[0], high=self.continuous_actionspace[1], shape=(1, 2*self.State_Dim+2*self.State_Dim),dtype=np.float)

    def _normalize_his_diff(self, state):
        if self.cord == 'NED':
            state[0:2,:] = state[0:2,:] / 30
        elif self.cord == 'LLH':
            state[0:2, :] = state[0:2, :] / 1e-5
        state[2,:] = state[2,:] / 2.5
        state[3:5, :] = state[3:5, :] / 2
        state[5, :] = state[5, :] / 20
        return state

    def _normalize_state(self, pre_state):
        if self.cord == 'NED':
            pre_state[0] = pre_state[0] / 50
            pre_state[1] = pre_state[1] / 50
        elif self.cord == 'LLH':
            pre_state[0] = pre_state[0] / 30
            pre_state[1] = pre_state[1] / 120
        pre_state[2] = pre_state[2] / 5
        pre_state[3] = pre_state[3] / 360
        pre_state[4] = pre_state[4] / 360
        pre_state[5] = pre_state[5] / 360
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
                dvel,dcbn = self.giengine.newImuProcess()
                self.dvel_list.append(dvel)
                self.dcbn_list.append(dcbn)
                self.current_step = self.current_step + 1  # 需要注意，current_step 需要和 self.giengine.timestamp() 对应
            else:
                self.fusing = True
                break # 转移到step函数再融合

            navstate = self.giengine.getNavState()
            # 提取 navstate 的值拼接成 1D 数组
            raw_state = np.concatenate((navstate.pos, navstate.vel, navstate.euler))
            # 一次性乘法和 round
            result = np.round(raw_state * self.multipliers, 9)
            nav_result.append(result)

        # TODO: 是否可以精简一下不要self.baseline
        if nav_result:
            ins_result = np.array(nav_result)
            self.set_results(ins_result) # 把新的导航结果赋值回去

        # 构建 self.pos_num -1 的历史序列拼接
        seq_len = int(self.pos_num * self.imurate / self.gnssrate)
        start_idx = self.current_step - seq_len
        end_idx = self.current_step + 1
        indices = np.linspace(start_idx, end_idx - 1, num=self.pos_num, dtype=int)
        obs_seq = self.baseline_buffer[indices][:, [0, 1, 2, 6, 7, 8]]
        obs = obs_seq.T
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
        obs_feature = self.gnssfile.data_[self.gnssfile.index,7:].copy()
        obs_feature = self._normalize_gnss(obs_feature)

        # state feature process
        if self.cord == 'NED':  # 地理坐标系转到导航坐标系
            n,e,d = pm.geodetic2ned(self.RL_prestate[0], self.RL_prestate[1], self.RL_prestate[2], self.start_pos[0], self.start_pos[1], self.start_pos[2])
            pre_pos = np.array([n,e,d])
            pre_state = np.concatenate((pre_pos, self.RL_prestate[6:])) # 拼接位置和姿态状态
        elif self.cord == 'LLH':
            pre_state = self.RL_prestate[0:3].copy()
        pre_state = self._normalize_state(pre_state)

        # RL cov features
        rl_cov_diag = np.diag(self.RLcov) / self.config["env_para"]['initial_RLcov']

        # KF P features
        cov_kf_p = self.giengine.Cov_[StateID.P_ID:StateID.P_ID + 3, StateID.P_ID:StateID.P_ID + 3].copy() # kf的pva协方差
        cov_kf_a = self.giengine.Cov_[StateID.PHI_ID:StateID.PHI_ID + 3, StateID.PHI_ID:StateID.PHI_ID + 3].copy() # kf的pva协方差
        diag_p = np.diag(cov_kf_p) / 10.0
        diag_a = np.diag(cov_kf_a) * 10.0
        KF_P = np.concatenate((diag_p, diag_a))

        # 观测合并
        obs_all = {'History': obs.reshape(1, obs.size, order='F'),
                   'gnss': obs_feature.reshape(1, 10),
                   'innovation': self.innovation.reshape(1, self.innovation.size),
                   'State': pre_state.reshape(1, pre_state.size),
                   'Cov': rl_cov_diag.reshape(1, rl_cov_diag.size),
                   'KF_P': KF_P.reshape(1, KF_P.size),
                   'innova_cor': self.innovation_correct.reshape(1, self.innovation_correct.size)}

        return obs_all

    def step(self, action):  # modified in 3.3
        # done = (self.current_step >= len(self.baseline.loc[:, 'UnixTimeMillis'].values) * self.traj_type[-1] - (
        #     self.pos_num))
        # action for new prediction
        action = np.reshape(action, [1, 2 * self.State_Dim+2 * self.State_Dim])
        blh_station = self.start_pos[0:3]
        rm, rn = radiusmn(blh_station[0])

        ############# A1 动作处理 #################
        predict_N = action[0, 0] * self.pred_scale
        predict_E = action[0, 1] * self.pred_scale
        predict_D = action[0, 2] * self.pred_scale * 1e-2
        predict_roll = action[0, 3] * self.pred_att_scale * 1e-2
        predict_pitch = action[0, 4] * self.pred_att_scale * 1e-2
        predict_yaw = action[0, 5] * self.pred_att_scale

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
        # 姿态预测
        d_euler_rad = np.array([predict_roll,predict_pitch,predict_yaw]) * Angle.D2R
        self.RL_prestate[6:9] = update_attitude(self.RL_prestate[6:9] * Angle.D2R, d_euler_rad) * Angle.R2D

        # 计算误差状态: 注意RL预测的状态是角度制，但是预测导航状态是弧度制
        Dr = Earth.DR(navstate.pos) # 地理坐标相对位置 转 n系相对位置
        RL_pos_rad = self.RL_prestate[0:3].copy()
        RL_pos_rad[0:2] = RL_pos_rad[0:2] * Angle.D2R
        dx_rl_p = Dr @ (navstate.pos - RL_pos_rad)
        dx_rl_a = calculate_phi_error(self.RL_prestate[6:9], navstate.euler * Angle.R2D)
        dx_rl = np.concatenate((dx_rl_p, dx_rl_a)) # 得到rl的误差状态估计，已经是符合估计误差状态的坐标系和单位

        # 协方差调整/预测
        d_cov_rl = np.diag(self.policy_cov_scale * action[0, self.State_Dim:2*self.State_Dim])
        # TODO 协方差预测不当会造成融合后对角线变为0（是因为太小？）
        if self.prdcov_mode == "Add_mode1":
            self.RLcov = self.RLcov + d_cov_rl
            self.RLcov[self.RLcov < 1e-1] = 1e-1
            pred_cov_rl = self.RLcov
        elif self.prdcov_mode == "Add_mode2":
            pred_cov_rl = self.RLcov + d_cov_rl
            pred_cov_rl[pred_cov_rl < 1e-1] = 1e-1

        # 确定到达融合状态，GNSS/INS可融合
        if not self.fusing:
            res = self.giengine.check_update()
            raise ValueError(f"Invalid state: res is {res}. Expected res > 0 for GNSS/INS integration.")
        dvel, dcbn = self.KRLF_Process(dx_rl,pred_cov_rl)
        self.dvel_list.append(dvel)
        self.dcbn_list.append(dcbn)
        self.fusing = False # 融合后设为False
        self.current_step = self.current_step + 1 # 当前步加1
        self.fusing_count = self.fusing_count + 1 # 融合次数加1

        ############# A2 动作处理 #################
        correct_N = action[0, 2 * self.State_Dim + 0] * self.correct_scale
        correct_E = action[0, 2 * self.State_Dim + 1] * self.correct_scale
        correct_D = action[0, 2 * self.State_Dim + 2] * self.correct_scale * 2
        cor_pos_n = np.array([correct_N, correct_E, correct_D])
        # action for att correction
        correct_roll = action[0, 2 * self.State_Dim + 3] * self.correct_att_scale * 1e-2
        correct_pitch = action[0, 2 * self.State_Dim + 4] * self.correct_att_scale * 1e-2
        correct_yaw = action[0, 2 * self.State_Dim + 5] * self.correct_att_scale
        cor_att = np.array([correct_roll, correct_pitch, correct_yaw])

        # 位置修正
        navstate = self.giengine.getNavState()
        cor_pos_llh = dm2drad(rm, rn, blh_station, cor_pos_n)
        cor_pva_llh = navstate.pos + cor_pos_llh.reshape(-1)
        self.giengine.pvacur_.pos = cor_pva_llh

        # 姿态修正
        cor_euler_rad = cor_att * Angle.D2R
        cor_att = update_attitude(navstate.euler, cor_euler_rad)
        self.giengine.pvacur_.att.euler = cor_att
        self.giengine.pvacur_.cbn = RU.euler2matrix(cor_att)  # 姿态角转旋转矩阵
        self.giengine.pvacur_.qbn = RU.euler2quaternion(np.flip(cor_att, axis=0))  # 姿态角转四元数矩阵

        # 位置 P 调整
        cov_kf_pva = self.giengine.Cov_[StateID.P_ID:StateID.P_ID + 3, StateID.P_ID:StateID.P_ID + 3].copy()
        d_cov_rl = np.diag(self.policy_P_scale * action[0, 3 * self.State_Dim:3 * self.State_Dim + 3])
        # TODO 协方差预测不当会造成融合后对角线变为0
        if self.corcov_mode == "Add_mode1":
            cov_kf_pva = cov_kf_pva + d_cov_rl
            cov_kf_pva[cov_kf_pva < 1e-1] = 1e-1
            pred_cov_rl = cov_kf_pva
        elif self.corcov_mode == "Scale_mode1":
            MIN_SCALE = 0.1
            MAX_SCALE = 10.0
            scale_factor = np.exp(d_cov_rl)  # np.exp(d_cov_rl)
            scale_factor = np.clip(scale_factor, MIN_SCALE, MAX_SCALE)
            pred_cov_rl = cov_kf_pva * scale_factor

        self.giengine.Cov_[StateID.P_ID:StateID.P_ID + 3, StateID.P_ID:StateID.P_ID + 3] = pred_cov_rl
        self.innovation_correct = self.compute_innovation()

        # 姿态 P 调整
        cov_kf_pva = self.giengine.Cov_[StateID.PHI_ID:StateID.PHI_ID + 3, StateID.PHI_ID:StateID.PHI_ID + 3].copy()
        d_cov_rl = np.diag(self.policy_P_scale * action[0, 3 * self.State_Dim + 3:])
        # TODO 协方差预测不当会造成融合后对角线变为0
        if self.corcov_mode == "Add_mode1":
            cov_kf_pva = cov_kf_pva + d_cov_rl
            cov_kf_pva[cov_kf_pva < 1e-1] = 1e-1
            pred_cov_rl = cov_kf_pva
        elif self.corcov_mode == "Scale_mode1":
            MIN_SCALE = 0.1
            MAX_SCALE = 10.0
            scale_factor = np.exp(d_cov_rl)  # np.exp(d_cov_rl)
            scale_factor = np.clip(scale_factor, MIN_SCALE, MAX_SCALE)
            pred_cov_rl = cov_kf_pva * scale_factor

        self.giengine.Cov_[StateID.PHI_ID:StateID.PHI_ID + 3, StateID.PHI_ID:StateID.PHI_ID + 3] = pred_cov_rl

        ############### 保存数据 #################
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
            self.RL_prestate = self.baseline.iloc[self.current_step, 1:10].values
            # rl_col = ['Latitude_RLpredict', 'Longitude_RLpredict', 'Ellipsoid_Height_RLpredict', 'Velocity_X_RLpredict',
            #           'Velocity_Y_RLpredict', 'Velocity_Z_RLpredict', 'Roll_RLpredict', 'Pitch_RLpredict',
            #           'Heading_RLpredict']
            # self.RL_prestate = self.baseline.loc[self.current_step, rl_col].values

        # reward function
        if self.flag_finetune:
            reward, error = self.reward_finetune()
        else:
            reward, error = self.reward_calculation(navstate)
        self.cumulated_reward += reward

        # Execute one time step within the environment
        obs = self._next_observation()
        if self.done:
            obs = []
            # 💥 修正260519：回合结束时，一次性同步缓存回 Pandas
            self.baseline[self.cols_rl_predict] = self.baseline_buffer
            # 1. 覆盖外部 data_truth_dic 跑完的所有数据
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[:, self.cols_rl_predict] = self.baseline_buffer
            # 2. 清除未来多余的残余数据 (设为 np.nan)
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step + 1:, self.cols_rl_predict] = np.nan
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[0:self.start_step, self.cols_rl_predict] = np.nan

        return obs, reward, self.done, {'tripIDnum': self.tripIDnum, 'current_step': self.current_step,'baseline': self.baseline,
                                   'error':error, 'tripid':self.tripIDnum, 'break': self.early_break}  # self.info#, {}# , 'data_truth_dic':data_truth_dic

    def _error_state_fusion(self,dx_rl,cov_rl):  # RL for KF modified in 0303
        indices = [0, 1, 2, 6, 7, 8]
        dx_kf = self.giengine.dx_[indices].copy() # 只获取 pa
        dx_rl = dx_rl.reshape([self.State_Dim,1])
        cov_kf_p = self.giengine.Cov_[StateID.P_ID:StateID.P_ID + 3, StateID.P_ID:StateID.P_ID + 3].copy() # kf的pva协方差
        cov_kf_a = self.giengine.Cov_[StateID.PHI_ID:StateID.PHI_ID + 3, StateID.PHI_ID:StateID.PHI_ID + 3].copy() # kf的pva协方差
        cov_kf_pva = block_diag(cov_kf_p, cov_kf_a)
        # 设置inovation参数
        self.innovation, self.NIS = compute_normalized_innovation(dx_kf, dx_rl, cov_kf_pva, cov_rl)
        K_fusion = np.linalg.solve((cov_kf_pva + cov_rl + np.eye(self.State_Dim) * 1e-12).T, cov_kf_pva.T).T
        # 卡方检验
        # if self.n_sigma * self.threshold < Nis:
        #     K_fusion = K_fusion * 1
        #     cov_rl_scale = 1
        #     # print(f"No RL update: NIS={Nis}")
        # else:
        #     cov_rl_scale = 1

        cov_rl_scale = 1
        dx_f = dx_kf + K_fusion @ (dx_rl - dx_kf)
        self.giengine.dx_[StateID.P_ID:StateID.P_ID+3] = dx_f[0:3] # 放置回原来向量
        self.giengine.dx_[StateID.PHI_ID:StateID.PHI_ID+3] = dx_f[3:6] # 放置回原来向量

        # 融合协方差
        try:
            fusion_gain = np.linalg.solve(cov_kf_pva + cov_rl * cov_rl_scale + np.eye(self.State_Dim) * 1e-12, cov_rl* cov_rl_scale)
            cov_f = cov_kf_pva @ fusion_gain
        except np.linalg.LinAlgError:
            # 极端情况下退回到简单加权
            cov_f = 0.5 * (cov_kf_pva + cov_rl * cov_rl_scale)

        # 3. 强制对称化与对角线保护
        cov_f = (cov_f + cov_f.T) * 0.5

        self.giengine.Cov_[StateID.P_ID:StateID.P_ID + 3, StateID.P_ID:StateID.P_ID + 3] = cov_f[0:3,0:3]
        self.giengine.Cov_[StateID.PHI_ID:StateID.PHI_ID + 3, StateID.PHI_ID:StateID.PHI_ID + 3] = cov_f[3:6,3:6]

class Continuous_PrePosAttCov_InHiGNSSStaCov(baseEnv):
    # 连续动作的状态预测和协方差预测（该环境只预测位置和姿态），输入新息 历史状态序列 GNSS特征 状态, 输出位置预测
    metadata = {'render.modes': ['human']}
    def __init__(self, config, **kwargs):
        # def __init__(self,trajdata_range, action_scale, discrete_actionspace, reward_setting, trajdata_sort, baseline_mod):
        super(Continuous_PrePosAttCov_InHiGNSSStaCov, self).__init__(config,**kwargs)
        # TODO: 确认 ino和 his 的状态维度
        self.State_Dim = 6
        self.pos_weight = config["env_para"]['pos_weight']
        self.vel_weight = 0
        self.threshold = chi2.ppf(0.999, self.State_Dim)
        self.h_scale = config.get("env_para", {}).get("h_scale", 0.01)
        self.n_sigma = config.get("env_para", {}).get("n_sigma", 3.0)
        self.observation_space = spaces.Dict(
            {'gnss': spaces.Box(low=-1, high=1, shape=(1, 10)),
             'innovation': spaces.Box(low=0, high=1, shape=(1, self.State_Dim), dtype=np.float),
             'History': spaces.Box(low=0, high=1, shape=(1, (self.pos_num-1) * 9), dtype=np.float),
             'State': spaces.Box(low=0, high=1, shape=(1, 9), dtype=np.float),
             'Cov': spaces.Box(low=0, high=1, shape=(1, self.State_Dim), dtype=np.float)})

        # 设置动作尺度
        self.pred_scale = config["env_para"]['continuous_scale_state_pred']
        self.policy_cov_scale = config["env_para"]['continuous_scale_policy_cov']
        self.att_scale = config["env_para"]['continuous_scale_att_pred']
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
            try:
                self.set_results(ins_result) # 把新的导航结果赋值回去
            except:
                pass

        # 构建 self.pos_num -1 的历史序列拼接
        seq_len = int(self.pos_num * self.imurate/self.gnssrate) # 根据频率获取窗口历元大小,预计大概是1s一个步长
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

        # 当前状态拼接
        # if self.baseline_mod == 'GNSS/INS':
        #     col_rl = self.baseline.columns.get_loc('Latitude_RLpredict')
        #     current_state = self.baseline.iloc[self.current_step-1, col_rl:col_rl+9].values
        #     obs = np.column_stack((obs, current_state))
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
            n,e,d = pm.geodetic2ned(self.RL_prestate[0], self.RL_prestate[1], self.RL_prestate[2], self.start_pos[0]*Angle.R2D, self.start_pos[1]*Angle.R2D, self.start_pos[2])
            pre_pos = np.array([n,e,d])
            pre_state = np.concatenate((pre_pos,self.RL_prestate[3:]))
        elif self.cord == 'LLH':
            pre_state = self.RL_prestate[0:3].copy()
        pre_state = self._normalize_state(pre_state)

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
        predict_D = action[0, 2] * self.pred_scale * self.h_scale
        predict_roll = action[0, 3] * self.att_scale * 1e-2
        predict_pitch = action[0, 4] * self.att_scale * 1e-2
        predict_yaw = action[0, 5] * self.att_scale

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
        # 姿态预测
        d_euler_rad = np.array([predict_roll,predict_pitch,predict_yaw]) * Angle.D2R
        self.RL_prestate[6:9] = update_attitude(self.RL_prestate[6:9] * Angle.D2R, d_euler_rad) * Angle.R2D

        # 计算误差状态: 注意RL预测的状态是角度制，但是预测导航状态是弧度制
        Dr = Earth.DR(navstate.pos) # 地理坐标相对位置 转 n系相对位置
        RL_pos_rad = self.RL_prestate[0:3].copy()
        RL_pos_rad[0:2] = RL_pos_rad[0:2] * Angle.D2R
        dx_rl_p = Dr @ (navstate.pos - RL_pos_rad)
        dx_rl_a = calculate_phi_error(self.RL_prestate[6:9], navstate.euler * Angle.R2D)
        dx_rl = np.concatenate((dx_rl_p, dx_rl_a)) # 得到rl的误差状态估计，已经是符合估计误差状态的坐标系和单位

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
            MAX_SCALE = 2.0
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
        else:
            rl_col = ['Velocity_X_RLpredict','Velocity_Y_RLpredict', 'Velocity_Z_RLpredict']
            self.RL_prestate[3:6] = self.baseline.loc[self.current_step, rl_col].values

        # reward function
        if self.flag_finetune:
            reward, error = self.reward_finetune(navstate)
        else:
            reward, error = self.reward_calculation(navstate)
        self.cumulated_reward += reward

        # Execute one time step within the environment
        if self.done:
            obs = []
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step+1: , ['Latitude_RLpredict']] = None
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step+1: , ['Longitude_RLpredict']] = None
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step+1: , ['Ellipsoid Height_RLpredict']] = None
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step+1: , ['Velocity_X_RLpredict']] = None
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step+1: , ['Velocity_Y_RLpredict']] = None
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step+1: , ['Velocity_Z_RLpredict']] = None
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step+1: , ['Roll_RLpredict']] = None
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step+1: , ['Pitch_RLpredict']] = None
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step+1: , ['Heading_RLpredict']] = None
        else:
            obs = self._next_observation()
        return obs, reward, self.done, {'tripIDnum': self.tripIDnum, 'current_step': self.current_step,'baseline': self.baseline,
                                   'error':error, 'tripid':self.tripIDnum, 'break': self.early_break}  # self.info#, {}# , 'data_truth_dic':data_truth_dic

    def _error_state_fusion(self,dx_rl,cov_rl):  # RL for KF modified in 0303
        indices = [0, 1, 2, 6, 7, 8]
        dx_kf = self.giengine.dx_[indices].copy() # 只获取 pa
        dx_rl = dx_rl.reshape([self.State_Dim,1])
        cov_kf_p = self.giengine.Cov_[StateID.P_ID:StateID.P_ID + 3, StateID.P_ID:StateID.P_ID + 3].copy() # kf的pva协方差
        cov_kf_a = self.giengine.Cov_[StateID.PHI_ID:StateID.PHI_ID + 3, StateID.PHI_ID:StateID.PHI_ID + 3].copy() # kf的pva协方差
        cov_kf_pva = block_diag(cov_kf_p, cov_kf_a)
        # 设置inovation参数
        self.innovation, Nis = compute_normalized_innovation(dx_kf, dx_rl, cov_kf_pva, cov_rl)
        K_fusion = np.linalg.solve((cov_kf_pva + cov_rl + np.eye(self.State_Dim) * 1e-12).T, cov_kf_pva.T).T
        # 卡方检验
        if self.n_sigma * self.threshold < Nis:
            K_fusion = K_fusion * 0.1
            cov_rl_scale = 100
            print(f"No RL update: NIS={Nis}")
        else:
            cov_rl_scale = 1

        dx_f = dx_kf + K_fusion @ (dx_rl - dx_kf)
        self.giengine.dx_[StateID.P_ID:StateID.P_ID+3] = dx_f[0:3] # 放置回原来向量
        self.giengine.dx_[StateID.PHI_ID:StateID.PHI_ID+3] = dx_f[3:6] # 放置回原来向量

        # 融合协方差
        try:
            fusion_gain = np.linalg.solve(cov_kf_pva + cov_rl*cov_rl_scale + np.eye(self.State_Dim) * 1e-12, cov_rl*cov_rl_scale)
            cov_f = cov_kf_pva @ fusion_gain
        except np.linalg.LinAlgError:
            # 极端情况下退回到简单加权
            cov_f = 0.5 * (cov_kf_pva + cov_rl*cov_rl_scale)

        # 3. 强制对称化与对角线保护
        cov_f = (cov_f + cov_f.T) * 0.5

        self.giengine.Cov_[StateID.P_ID:StateID.P_ID + 3, StateID.P_ID:StateID.P_ID + 3] = cov_f[0:3,0:3]
        self.giengine.Cov_[StateID.PHI_ID:StateID.PHI_ID + 3, StateID.PHI_ID:StateID.PHI_ID + 3] = cov_f[3:6,3:6]

class Continuous_PrePosCov_InHiGNSSPosCov(baseEnv):
    # 连续动作的状态预测和协方差预测（该环境只预测位置），输入新息 历史序列 GNSS特征 状态, 输出位置预测
    metadata = {'render.modes': ['human']}
    def __init__(self, config, **kwargs):
        # def __init__(self,trajdata_range, action_scale, discrete_actionspace, reward_setting, trajdata_sort, baseline_mod):
        super(Continuous_PrePosCov_InHiGNSSPosCov, self).__init__(config,**kwargs)
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
        self.policy_cov_scale = config["env_para"]['continuous_scale_policy_cov']
        self.action_space = spaces.Box(low=self.continuous_actionspace[0], high=self.continuous_actionspace[1], shape=(1, 2*self.State_Dim),dtype=np.float)
        self.conv_mode = config["env_para"]['conv_mode']

    def _normalize_his_diff(self, state):
        if self.cord == 'NED':
            state[0:2,:] = state[0:2,:] / 30
        elif self.cord == 'LLH':
            state[0:2, :] = state[0:2, :] / 1e-5
        state[2,:] = state[2,:] / 2.5
        return state * self.gnssrate # gnss 频率不一样

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
        obs_feature = self.gnssfile.data_[self.gnssfile.index,7:]
        obs_feature = self._normalize_gnss(obs_feature)

        # state feature process
        if self.cord == 'NED':  # 地理坐标系转到导航坐标系
            n,e,d = pm.geodetic2ned(self.RL_prestate[0], self.RL_prestate[1], self.RL_prestate[2], self.start_pos[0]*Angle.R2D, self.start_pos[1]*Angle.R2D, self.start_pos[2])
            pre_state = np.array([n,e,d])
        elif self.cord == 'LLH':
            pre_state = self.RL_prestate[0:3].copy()
        pre_state = self._normalize_state(pre_state)

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

class Continuous_CorrectPosAttCov_InHiGNSSPosCov(baseEnv):
    # 连续动作的状态预测和协方差预测（该环境只预测位置），输入新息 历史序列 GNSS特征 状态, 输出位置预测
    metadata = {'render.modes': ['human']}
    def __init__(self, config, **kwargs):
        # def __init__(self,trajdata_range, action_scale, discrete_actionspace, reward_setting, trajdata_sort, baseline_mod):
        super(Continuous_CorrectPosAttCov_InHiGNSSPosCov, self).__init__(config,**kwargs)
        # TODO: 确认 ino和 his 的状态维度
        self.State_Dim = 6
        self.pos_weight = config["env_para"]['pos_weight']
        self.vel_weight = 0
        self.observation_space = spaces.Dict(
            {'gnss': spaces.Box(low=-1, high=1, shape=(1, 10)),
             'innovation': spaces.Box(low=0, high=1, shape=(1, 3), dtype=np.float),
             'History': spaces.Box(low=0, high=1, shape=(1, (self.pos_num-1) * self.State_Dim), dtype=np.float),
             'State': spaces.Box(low=0, high=1, shape=(1, self.State_Dim), dtype=np.float),
             'Cov': spaces.Box(low=0, high=1, shape=(1, self.State_Dim), dtype=np.float)})

        # 设置动作尺度
        self.pred_scale = config["env_para"]['continuous_scale_state_pred']
        self.policy_cov_scale = config["env_para"]['continuous_scale_policy_cov']
        self.att_scale = config["env_para"]['continuous_scale_att_pred']
        self.action_space = spaces.Box(low=self.continuous_actionspace[0], high=self.continuous_actionspace[1], shape=(1, 2*self.State_Dim),dtype=np.float)
        self.conv_mode = config["env_para"]['conv_mode']

    def _normalize_his_diff(self, state):
        if self.cord == 'NED':
            state[0:2,:] = state[0:2,:] / 30
        elif self.cord == 'LLH':
            state[0:2, :] = state[0:2, :] / 1e-5
        state[2,:] = state[2,:] / 2.5
        state[3:5, :] = state[3:5, :] / 2
        state[5, :] = state[5, :] / 20
        return state

    def _normalize_state(self, pre_state):
        if self.cord == 'NED':
            pre_state[0] = pre_state[0] / 50
            pre_state[1] = pre_state[1] / 50
        elif self.cord == 'LLH':
            pre_state[0] = pre_state[0] / 30
            pre_state[1] = pre_state[1] / 120
        pre_state[2] = pre_state[2] / 5
        pre_state[3] = pre_state[3] / 360
        pre_state[4] = pre_state[4] / 360
        pre_state[5] = pre_state[5] / 360
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
            # 提取 navstate 的值拼接成 1D 数组
            raw_state = np.concatenate((navstate.pos, navstate.vel, navstate.euler))
            # 一次性乘法和 round
            result = np.round(raw_state * self.multipliers, 9)
            nav_result.append(result)

        # TODO: 是否可以精简一下不要self.baseline
        if nav_result:
            ins_result = np.array(nav_result)
            self.set_results(ins_result) # 把新的导航结果赋值回去

        # 💥 修正260519：改进切片提取操作
        # 构建 self.pos_num -1 的历史序列拼接
        seq_len = int(self.pos_num * self.imurate / self.gnssrate)
        start_idx = self.current_step - seq_len
        end_idx = self.current_step + 1
        indices = np.linspace(start_idx, end_idx - 1, num=self.pos_num, dtype=int)
        obs_seq = self.baseline_buffer[indices][:, [0, 1, 2, 6, 7, 8]]
        obs = obs_seq.T
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
            n,e,d = pm.geodetic2ned(self.RL_prestate[0], self.RL_prestate[1], self.RL_prestate[2], self.start_pos[0]*Angle.R2D, self.start_pos[1]*Angle.R2D, self.start_pos[2])
            pre_pos = np.array([n,e,d])
            pre_state = np.concatenate((pre_pos, self.RL_prestate[6:])) # 拼接位置和姿态状态
        elif self.cord == 'LLH':
            pre_state = self.RL_prestate[0:3].copy()
        pre_state = self._normalize_state(pre_state)

        # KF P cov features
        cov_kf_p = self.giengine.Cov_[StateID.P_ID:StateID.P_ID + 3, StateID.P_ID:StateID.P_ID + 3].copy() # kf的pva协方差
        cov_kf_a = self.giengine.Cov_[StateID.PHI_ID:StateID.PHI_ID + 3, StateID.PHI_ID:StateID.PHI_ID + 3].copy() # kf的pva协方差
        diag_p = np.diag(cov_kf_p) / 10.0
        diag_a = np.diag(cov_kf_a) * 10.0
        rl_cov_diag = np.concatenate((diag_p, diag_a))

        # 观测合并
        obs_all = {'History': obs.reshape(1, obs.size, order='F'),
                   'gnss': obs_feature.reshape(1, 10),
                   'innovation': self.innovation_correct.reshape(1, self.innovation_correct.size),
                   'State': pre_state.reshape(1, pre_state.size),
                   'Cov': rl_cov_diag.reshape(1, rl_cov_diag.size)}

        return obs_all

    def step(self, action):  # modified in 3.3
        # action for pos correction
        action = np.reshape(action, [1, -1])
        predict_N = action[0, 0] * self.pred_scale
        predict_E = action[0, 1] * self.pred_scale
        predict_D = action[0, 2] * self.pred_scale * 2
        d_pos_n = np.array([predict_N, predict_E, predict_D])
        # action for att correction
        predict_roll = action[0, 3] * self.att_scale * 1e-2
        predict_pitch = action[0, 4] * self.att_scale * 1e-2
        predict_yaw = action[0, 5] * self.att_scale
        d_att = np.array([predict_roll, predict_pitch, predict_yaw])

        # 确定到达融合状态，GNSS/INS可融合
        if not self.fusing:
            res = self.giengine.check_update()
            raise ValueError(f"Invalid state: res is {res}. Expected res > 0 for GNSS/INS integration.")
        self.KRLF_Process()
        self.fusing = False # 融合后设为False
        self.current_step = self.current_step + 1 # 当前步加1
        self.fusing_count = self.fusing_count + 1 # 融合次数加1

        # 位置修正
        blh_station = self.start_pos[0:3]
        rm, rn = radiusmn(blh_station[0])
        navstate = self.giengine.getNavState()

        d_pos_llh = dm2drad(rm, rn, blh_station, d_pos_n)
        cor_pva_llh = navstate.pos + d_pos_llh.reshape(-1)
        self.giengine.pvacur_.pos = cor_pva_llh

        # 姿态修正
        d_euler_rad = d_att * Angle.D2R
        cor_att = update_attitude(navstate.euler, d_euler_rad)
        self.giengine.pvacur_.att.euler = cor_att
        self.giengine.pvacur_.cbn = RU.euler2matrix(cor_att)  # 姿态角转旋转矩阵
        self.giengine.pvacur_.qbn = RU.euler2quaternion(np.flip(cor_att, axis=0))  # 姿态角转四元数矩阵

        # 位置 P 调整
        cov_kf_pva = self.giengine.Cov_[StateID.P_ID:StateID.P_ID + 3, StateID.P_ID:StateID.P_ID + 3].copy()
        d_cov_rl = np.diag(self.policy_cov_scale * action[0, self.State_Dim:self.State_Dim+3])
        # TODO 协方差预测不当会造成融合后对角线变为0
        if self.prdcov_mode == "Add_mode1":
            cov_kf_pva = cov_kf_pva + d_cov_rl
            # 正确的做法：提取对角线 -> 裁剪对角线 -> 填回原矩阵
            diag_elements = np.diag(cov_kf_pva)
            diag_elements[diag_elements < 1e-1] = 1e-1
            np.fill_diagonal(cov_kf_pva, diag_elements)
            pred_cov_rl = cov_kf_pva
        elif self.prdcov_mode == "Scale_mode1":
            MIN_SCALE = 0.1
            MAX_SCALE = 10.0
            scale_factor = np.exp(d_cov_rl)  # np.exp(d_cov_rl)
            scale_factor = np.clip(scale_factor, MIN_SCALE, MAX_SCALE)
            pred_cov_rl = cov_kf_pva * scale_factor

        self.giengine.Cov_[StateID.P_ID:StateID.P_ID + 3, StateID.P_ID:StateID.P_ID + 3] = pred_cov_rl
        self.innovation_correct = self.compute_innovation()

        # 姿态 P 调整
        cov_kf_pva = self.giengine.Cov_[StateID.PHI_ID:StateID.PHI_ID + 3, StateID.PHI_ID:StateID.PHI_ID + 3].copy()
        d_cov_rl = np.diag(self.policy_cov_scale * action[0, self.State_Dim+3:])
        # TODO 协方差预测不当会造成融合后对角线变为0
        if self.prdcov_mode == "Add_mode1":
            cov_kf_pva = cov_kf_pva + d_cov_rl
            # 正确的做法：提取对角线 -> 裁剪对角线 -> 填回原矩阵
            diag_elements = np.diag(cov_kf_pva)
            diag_elements[diag_elements < 1e-1] = 1e-1
            np.fill_diagonal(cov_kf_pva, diag_elements)
            pred_cov_rl = cov_kf_pva
        elif self.prdcov_mode == "Scale_mode1":
            MIN_SCALE = 0.1
            MAX_SCALE = 10.0
            scale_factor = np.exp(d_cov_rl)  # np.exp(d_cov_rl)
            scale_factor = np.clip(scale_factor, MIN_SCALE, MAX_SCALE)
            pred_cov_rl = cov_kf_pva * scale_factor

        self.giengine.Cov_[StateID.PHI_ID:StateID.PHI_ID + 3, StateID.PHI_ID:StateID.PHI_ID + 3] = pred_cov_rl

        # 💥 修改 260519：保存数据
        navstate = self.giengine.getNavState()
        # 提取 navstate 的值拼接成 1D 数组
        raw_state = np.concatenate((navstate.pos, navstate.vel, navstate.euler))
        # 一次性乘法和 round
        result = np.round(raw_state * self.multipliers, 9)
        self.set_results(result.reshape(1, -1))

        # reward function
        if self.flag_finetune:
            reward, error = self.reward_finetune(navstate)
        else:
            reward, error = self.reward_calculation(navstate)
        self.cumulated_reward += reward

        # Execute one time step within the environment
        obs = self._next_observation()
        if self.done:
            # obs = []
            # 💥 修正260519：回合结束时，一次性同步缓存回 Pandas
            self.baseline[self.cols_rl_predict] = self.baseline_buffer
            # 1. 覆盖外部 data_truth_dic 跑完的所有数据
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[:, self.cols_rl_predict] = self.baseline_buffer
            # 2. 清除未来多余的残余数据 (设为 np.nan)
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step + 1:, self.cols_rl_predict] = np.nan
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[0:self.start_step, self.cols_rl_predict] = np.nan

        return obs, reward, self.done, {'tripIDnum': self.tripIDnum, 'current_step': self.current_step,'baseline': self.baseline,
                                   'error':error, 'tripid':self.tripIDnum, 'break': self.early_break}

    def KRLF_Process(self):
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
            # self.giengine.stateFeedback()
        elif res == 1:  # 更新时间靠近imutime1
            # GNSS数据靠近上一历元，先对上一历元进行GNSS更新
            # gnssdata is near to the previous imudata, we should firstly do gnss update
            self.giengine.gnssUpdate(gnssdata_)
            self.giengine.stateFeedback()
            self.giengine.pvapre_ = self.giengine.pvacur_
            self.giengine.insPropagation(imupre_, imucur_)
        elif res == 2:  # 更新时间靠近imutime2
            # GNSS数据靠近当前历元，先对当前IMU进行状态传播
            self.giengine.insPropagation(imupre_, imucur_)
            self.giengine.gnssUpdate(gnssdata_)
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
            self.giengine.stateFeedback()

            # 对后一半IMU进行状态传播
            # propagate navigation state for the second half imudata
            self.giengine.pvapre_ = self.giengine.pvacur_
            self.giengine.insPropagation(midimu, imucur_)

        # 更新上一时刻的状态和IMU数据
        # update system state and imudata at the previous epoch
        self.giengine.pvapre_ = self.giengine.pvacur_
        self.giengine.imupre_ = self.giengine.imucur_

    def compute_innovation(self):
        # IMU位置转到GNSS天线相位中心位置
        # convert IMU position to GNSS antenna phase center position
        gnssdata = self.giengine.gnssdata_

        Dr_inv = Earth.DRi(self.giengine.pvacur_.pos)  # n系相对位置转地理坐标相对位置
        Dr = Earth.DR(self.giengine.pvacur_.pos)  # # 地理坐标相对位置转n系相对位置
        antenna_pos = self.giengine.pvacur_.pos + Dr_inv @ self.giengine.pvacur_.att.cbn @ self.giengine.options_.antlever

        # GNSS位置测量新息
        # compute GNSS position innovation
        dz = Dr @ (antenna_pos - gnssdata.blh)  # 相当于单位变成米？

        # 构造GNSS位置观测矩阵
        # construct GNSS position measurement matrix
        H_gnsspos = np.zeros((3, self.giengine.Cov_.shape[0]))
        H_gnsspos[0:3, StateID.P_ID:StateID.P_ID + 3] = np.identity(3)
        H_gnsspos[0:3, StateID.PHI_ID:StateID.PHI_ID + 3] = RU.skewSymmetric(
            self.giengine.pvacur_.att.cbn @ self.giengine.options_.antlever)  # StateID.PHI 姿态ID

        dz = dz.reshape(3, 1)
        innovation = dz - H_gnsspos @ self.giengine.dx_
        return innovation[0:3]

class Continuous_CorrectPosCov_InHiGNSSPosCov(baseEnv):
    # 连续动作的状态预测和协方差预测（该环境只预测位置），输入新息 历史序列 GNSS特征 状态, 输出位置预测
    metadata = {'render.modes': ['human']}
    def __init__(self, config, **kwargs):
        # def __init__(self,trajdata_range, action_scale, discrete_actionspace, reward_setting, trajdata_sort, baseline_mod):
        super(Continuous_CorrectPosCov_InHiGNSSPosCov, self).__init__(config,**kwargs)
        # TODO: 确认 ino和 his 的状态维度
        self.State_Dim = 3
        self.pos_weight = config["env_para"]['pos_weight']
        self.vel_weight = 0
        self.att_weight = 0
        self.observation_space = spaces.Dict(
            {'gnss': spaces.Box(low=-1, high=1, shape=(1, 10)),
             'innovation': spaces.Box(low=0, high=1, shape=(1, 3), dtype=np.float),
             'History': spaces.Box(low=0, high=1, shape=(1, (self.pos_num-1) * self.State_Dim), dtype=np.float),
             'State': spaces.Box(low=0, high=1, shape=(1, self.State_Dim), dtype=np.float),
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
        return state * self.gnssrate # gnss 频率不一样

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
            if res == 0:
                self.giengine.newImuProcess()
                self.current_step = self.current_step + 1  # 需要注意，current_step 需要和 self.giengine.timestamp() 对应
            else:
                self.fusing = True
                break # 转移到step函数再融合

            navstate = self.giengine.getNavState()
            # 提取 navstate 的值拼接成 1D 数组
            raw_state = np.concatenate((navstate.pos, navstate.vel, navstate.euler))
            # 一次性乘法和 round
            result = np.round(raw_state * self.multipliers, 9)
            nav_result.append(result)

        # TODO: 是否可以精简一下不要self.baseline
        if nav_result:
            ins_result = np.array(nav_result)
            self.set_results(ins_result) # 把新的导航结果赋值回去

        # 💥 修正260519：改进切片提取操作
        # 构建 self.pos_num -1 的历史序列拼接
        seq_len = int(self.pos_num * self.imurate / self.gnssrate)
        start_idx = self.current_step - seq_len
        end_idx = self.current_step + 1
        indices = np.linspace(start_idx, end_idx - 1, num=self.pos_num, dtype=int)
        obs_seq = self.baseline_buffer[indices][:, [0, 1, 2]]
        obs = obs_seq.T
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
            n,e,d = pm.geodetic2ned(self.RL_prestate[0], self.RL_prestate[1], self.RL_prestate[2], self.start_pos[0]*Angle.R2D, self.start_pos[1]*Angle.R2D, self.start_pos[2])
            pre_state = np.array([n,e,d])
        elif self.cord == 'LLH':
            pre_state = self.RL_prestate[0:3].copy()
        pre_state = self._normalize_state(pre_state)

        # KF P cov features
        cov_kf_p = self.giengine.Cov_[StateID.P_ID:StateID.P_ID + 3, StateID.P_ID:StateID.P_ID + 3].copy() # kf的pva协方差
        rl_cov_diag = np.diag(cov_kf_p) / 10.0

        # 观测合并
        obs_all = {'History': obs.reshape(1, obs.size, order='F'),
                   'gnss': obs_feature.reshape(1, 10),
                   'innovation': self.innovation_correct.reshape(1, self.innovation_correct.size),
                   'State': pre_state.reshape(1, pre_state.size),
                   'Cov': rl_cov_diag.reshape(1, rl_cov_diag.size)}

        return obs_all

    def step(self, action):  # modified in 3.3
        # action for pos correction
        action = np.reshape(action, [1, -1])
        predict_N = action[0, 0] * self.pred_scale
        predict_E = action[0, 1] * self.pred_scale
        predict_D = action[0, 2] * self.pred_scale * 2
        d_pos_n = np.array([predict_N, predict_E, predict_D])

        # 确定到达融合状态，GNSS/INS可融合
        if not self.fusing:
            res = self.giengine.check_update()
            raise ValueError(f"Invalid state: res is {res}. Expected res > 0 for GNSS/INS integration.")
        self.KRLF_Process()
        self.fusing = False # 融合后设为False
        self.current_step = self.current_step + 1 # 当前步加1
        self.fusing_count = self.fusing_count + 1 # 融合次数加1

        # 位置修正
        blh_station = self.start_pos[0:3]
        rm, rn = radiusmn(blh_station[0])
        navstate = self.giengine.getNavState()

        d_pos_llh = dm2drad(rm, rn, blh_station, d_pos_n)
        cor_pva_llh = navstate.pos + d_pos_llh.reshape(-1)
        self.giengine.pvacur_.pos = cor_pva_llh

        # 位置 P 调整
        cov_kf_pva = self.giengine.Cov_[StateID.P_ID:StateID.P_ID + 3, StateID.P_ID:StateID.P_ID + 3].copy()
        d_cov_rl = np.diag(self.policy_cov_scale * action[0, self.State_Dim:self.State_Dim+3])
        # TODO 协方差预测不当会造成融合后对角线变为0
        if self.prdcov_mode == "Add_mode1":
            cov_kf_pva = cov_kf_pva + d_cov_rl
            # 正确的做法：提取对角线 -> 裁剪对角线 -> 填回原矩阵
            diag_elements = np.diag(cov_kf_pva)
            diag_elements[diag_elements < 1e-1] = 1e-1
            np.fill_diagonal(cov_kf_pva, diag_elements)
            pred_cov_rl = cov_kf_pva
        elif self.prdcov_mode == "Scale_mode1":
            MIN_SCALE = 0.1
            MAX_SCALE = 10.0
            scale_factor = np.exp(d_cov_rl)  # np.exp(d_cov_rl)
            scale_factor = np.clip(scale_factor, MIN_SCALE, MAX_SCALE)
            pred_cov_rl = cov_kf_pva * scale_factor

        self.giengine.Cov_[StateID.P_ID:StateID.P_ID + 3, StateID.P_ID:StateID.P_ID + 3] = pred_cov_rl
        self.innovation_correct = self.compute_innovation()

        # 修改 260519：保存数据
        navstate = self.giengine.getNavState()
        # 提取 navstate 的值拼接成 1D 数组
        raw_state = np.concatenate((navstate.pos, navstate.vel, navstate.euler))
        # 一次性乘法和 round
        result = np.round(raw_state * self.multipliers, 9)
        self.set_results(result.reshape(1, -1))

        # reward function
        if self.flag_finetune:
            reward, error = self.reward_finetune(navstate)
        else:
            reward, error = self.reward_calculation(navstate)
        self.cumulated_reward += reward

        # Execute one time step within the environment
        if self.done:
            obs = []
            # 💥 修正260519：回合结束时，一次性同步缓存回 Pandas
            self.baseline[self.cols_rl_predict] = self.baseline_buffer
            # 1. 覆盖外部 data_truth_dic 跑完的所有数据
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[:, self.cols_rl_predict] = self.baseline_buffer
            # 2. 清除未来多余的残余数据 (设为 np.nan)
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step + 1:, self.cols_rl_predict] = np.nan
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[0:self.start_step, self.cols_rl_predict] = np.nan
        else:
            obs = self._next_observation()
        return obs, reward, self.done, {'tripIDnum': self.tripIDnum, 'current_step': self.current_step,'baseline': self.baseline,
                                   'error':error, 'tripid':self.tripIDnum, 'break': self.early_break}

    def KRLF_Process(self):
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
            # self.giengine.stateFeedback()
        elif res == 1:  # 更新时间靠近imutime1
            # GNSS数据靠近上一历元，先对上一历元进行GNSS更新
            # gnssdata is near to the previous imudata, we should firstly do gnss update
            self.giengine.gnssUpdate(gnssdata_)
            self.giengine.stateFeedback()
            self.giengine.pvapre_ = self.giengine.pvacur_
            self.giengine.insPropagation(imupre_, imucur_)
        elif res == 2:  # 更新时间靠近imutime2
            # GNSS数据靠近当前历元，先对当前IMU进行状态传播
            self.giengine.insPropagation(imupre_, imucur_)
            self.giengine.gnssUpdate(gnssdata_)
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
            self.giengine.stateFeedback()

            # 对后一半IMU进行状态传播
            # propagate navigation state for the second half imudata
            self.giengine.pvapre_ = self.giengine.pvacur_
            self.giengine.insPropagation(midimu, imucur_)

        # 更新上一时刻的状态和IMU数据
        # update system state and imudata at the previous epoch
        self.giengine.pvapre_ = self.giengine.pvacur_
        self.giengine.imupre_ = self.giengine.imucur_

    def compute_innovation(self):
        # IMU位置转到GNSS天线相位中心位置
        # convert IMU position to GNSS antenna phase center position
        gnssdata = self.giengine.gnssdata_

        Dr_inv = Earth.DRi(self.giengine.pvacur_.pos)  # n系相对位置转地理坐标相对位置
        Dr = Earth.DR(self.giengine.pvacur_.pos)  # # 地理坐标相对位置转n系相对位置
        antenna_pos = self.giengine.pvacur_.pos + Dr_inv @ self.giengine.pvacur_.att.cbn @ self.giengine.options_.antlever

        # GNSS位置测量新息
        # compute GNSS position innovation
        dz = Dr @ (antenna_pos - gnssdata.blh)  # 相当于单位变成米？

        # 构造GNSS位置观测矩阵
        # construct GNSS position measurement matrix
        H_gnsspos = np.zeros((3, self.giengine.Cov_.shape[0]))
        H_gnsspos[0:3, StateID.P_ID:StateID.P_ID + 3] = np.identity(3)
        H_gnsspos[0:3, StateID.PHI_ID:StateID.PHI_ID + 3] = RU.skewSymmetric(
            self.giengine.pvacur_.att.cbn @ self.giengine.options_.antlever)  # StateID.PHI 姿态ID

        dz = dz.reshape(3, 1)
        innovation = dz - H_gnsspos @ self.giengine.dx_
        return innovation[0:3]

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
            n,e,d = pm.geodetic2ned(self.RL_prestate[0], self.RL_prestate[1], self.RL_prestate[2], self.start_pos[0]*Angle.R2D, self.start_pos[1]*Angle.R2D, self.start_pos[2])
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
    Nis = result.T @ result
    return result, Nis