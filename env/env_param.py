# normalize parameters
# for 170 datasets lat [37.7770066, 34.1405127] lon [-118.315262, -122.4331823]
import os
import sys
from enum import IntEnum
import numpy as np
from scipy.spatial.transform import Rotation
cur_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(cur_dir, '..', "src")
sys.path.append(src_dir)
import pandas as pd
from common.types import IMU, GNSS
from kfgins.kf_gins_types import *
from common.rotation import RotationUtils as RU
from kfgins.insmech_tang import INSMech as INS
from common.earth import Earth
# 卫星特征参数
SAT_MAX = 30
CNR_MAX = 50
CNR_MIN = 25
ELE_MAX = 80
ELE_MIN = 15


CLIGHT = 299_792_458   # speed of light (m/s)
RE_WGS84 = 6_378_137
OMGE = 7.2921151467E-5  # earth angular velocity (IS-GPS) (rad/s)
g_acc = 9.80665 # Gravitational acceleration
E_2 = 0.00669437999014 # 第一偏心率

a = 6378137 # 长半轴（单位：米）
b = 6356752.3142
esq = 6.69437999014 * 0.001
e1sq = 6.73949674228 * 0.001

dir_path = '/mnt/sdb/home/tangjh/KF-GINS-Py-main/'
traj_sum_df = pd.read_csv(f'{dir_path}/env/raw_tripID_summary.csv')

# 轨迹标签分类
traj_HK_MeDe_ublox = ['1_UrbanNav-HK-Medium-Urban-1/ublox_f9p',
'1_UrbanNav-HK-Medium-Urban-1/ublox_f9p_splitter',
'2_UrbanNav-HK-Deep-Urban-1/ublox_f9p',
'2_UrbanNav-HK-Deep-Urban-1/ublox_splitter'] # 香港 medium and deep urban with ublox
traj_HK_Ha_ublox = ['3_UrbanNav-HK-Harsh-Urban-1/ublox_f9p',
'3_UrbanNav-HK-Harsh-Urban-1/ublox_f9p_splitter'] #  香港 harsh urban with ublox
traj_HK_Tu_ublox = ['4_UrbanNav-HK-Tunnel-1/ublox_f9p',
'4_UrbanNav-HK-Tunnel-1/ublox_f9p_splitter'] #  香港 tunnel urban with ublox
traj_HK_MeDeHa_ublox = ['1_UrbanNav-HK-Medium-Urban-1/ublox_f9p',
'1_UrbanNav-HK-Medium-Urban-1/ublox_f9p_splitter',
'2_UrbanNav-HK-Deep-Urban-1/ublox_f9p',
'2_UrbanNav-HK-Deep-Urban-1/ublox_splitter',
'3_UrbanNav-HK-Harsh-Urban-1/ublox_f9p',
'3_UrbanNav-HK-Harsh-Urban-1/ublox_f9p_splitter']
traj_HK_DeHa_ublox = [ '3_UrbanNav-HK-Harsh-Urban-1/ublox_f9p_splitter',
'2_UrbanNav-HK-Deep-Urban-1/ublox_f9p',
'2_UrbanNav-HK-Deep-Urban-1/ublox_splitter',
'3_UrbanNav-HK-Harsh-Urban-1/ublox_f9p',
]
traj_HK_Me = traj_sum_df[traj_sum_df['Traj_ID'].str.contains('Medium', na=False)]['Traj_ID'].values.tolist()
traj_HK_De = traj_sum_df[traj_sum_df['Traj_ID'].str.contains('Deep', na=False)]['Traj_ID'].values.tolist()
traj_HK_Ha = traj_sum_df[traj_sum_df['Traj_ID'].str.contains('Harsh', na=False)]['Traj_ID'].values.tolist()
traj_HK_Tu = traj_sum_df[traj_sum_df['Traj_ID'].str.contains('Tunnel', na=False)]['Traj_ID'].values.tolist()
traj_HK_Me_phone = traj_sum_df[(traj_sum_df['Traj_ID'].str.contains('Medium', na=False))
                               & ~(traj_sum_df['Traj_ID'].str.contains('ublox', na=False))]['Traj_ID'].values.tolist()
traj_HK_Me_ublox = traj_sum_df[(traj_sum_df['Traj_ID'].str.contains('Medium', na=False))
                               & (traj_sum_df['Traj_ID'].str.contains('ublox', na=False))]['Traj_ID'].values.tolist()
traj_HK_De_ublox = traj_sum_df[(traj_sum_df['Traj_ID'].str.contains('Deep', na=False))
                               & (traj_sum_df['Traj_ID'].str.contains('ublox', na=False))]['Traj_ID'].values.tolist()
traj_HK_Ha_ublox = traj_sum_df[(traj_sum_df['Traj_ID'].str.contains('Harsh', na=False))
                               & (traj_sum_df['Traj_ID'].str.contains('ublox', na=False))]['Traj_ID'].values.tolist()
traj_HK_Ha_phone = ['3_UrbanNav-HK-Harsh-Urban-1/google_pixel4','3_UrbanNav-HK-Harsh-Urban-1/huawei_p40pro']

traj_HK_Me_ublox_f9p = ['1_UrbanNav-HK-Medium-Urban-1/ublox_f9p']
traj_HK_Ha_ublox_f9p = ['3_UrbanNav-HK-Harsh-Urban-1/ublox_f9p']

traj_TK_Od = traj_sum_df[(traj_sum_df['Traj_ID'].str.contains('Odaiba', na=False))]['Traj_ID'].values.tolist()
traj_TK_Sh = traj_sum_df[(traj_sum_df['Traj_ID'].str.contains('Shinjuku', na=False))]['Traj_ID'].values.tolist()
traj_TK_Sh_ublox = ['Tokyo_Data_Shinjuku/ublox']
traj_TK_Sh_trimble = ['Tokyo_Data_Shinjuku/trimble']
traj_TK_Od_ublox = ['Tokyo_Data_Odaiba/ublox']
traj_TK_Od_trimble = ['Tokyo_Data_Odaiba/trimble']

exclude_traj = ['2_UrbanNav-HK-Deep-Urban-1/xiaomi_mi8','1_UrbanNav-HK-Medium-Urban-1/samsung_note8',
                '1_UrbanNav-HK-Medium-Urban-1/huawei_p40pro']

trip_type_maping={
    'HK_MeDe_ublox': traj_HK_MeDe_ublox, # 切换轨迹时候会不平稳
    'HK_Ha_ublox': traj_HK_Ha_ublox,
    'HK_Ha_phone': traj_HK_Ha_phone,
    'HK_Tu_ublox': traj_HK_Tu_ublox,
    'HK_MeDeHa_ublox': traj_HK_MeDeHa_ublox,
    'HK_DeHa_ublox': traj_HK_DeHa_ublox,
    'HK_Me': traj_HK_Me,
    'HK_De': traj_HK_De,
    'HK_Ha': traj_HK_Ha,
    'HK_Me_phone': traj_HK_Me_phone,
    'HK_Me_ublox': traj_HK_Me_ublox,
    'HK_Me_ublox_A2': traj_HK_Me_ublox,
    'HK_Me_ublox_Dagent': traj_HK_Me_ublox,
    'HK_De_ublox': traj_HK_De_ublox,
    'HK_De_ublox_A2': traj_HK_De_ublox,
    'HK_De_ublox_Dagent': traj_HK_De_ublox,
    'HK_Me_ublox_f9p': traj_HK_Me_ublox_f9p,
    'HK_Ha_ublox_f9p': traj_HK_Ha_ublox_f9p,
    'TK_Od': traj_TK_Od,
    'TK_Sh': traj_TK_Sh,
    'TK_Sh_ublox': traj_TK_Sh_ublox,
    'TK_Sh_trimble': traj_TK_Sh_trimble,
    'TK_Od_ublox':traj_TK_Od_ublox,
    'TK_Od_trimble':traj_TK_Od_trimble
}


class StateID(IntEnum):
    P_ID = 0
    V_ID = 3
    PHI_ID = 6
    BG_ID = 9
    BA_ID = 12
    SG_ID = 15
    SA_ID = 18


class NoiseID(IntEnum):
    VRW_ID = 0
    ARW_ID = 3
    BGSTD_ID = 6  # 陀螺仪零偏std
    BASTD_ID = 9  # 加速度计零偏std
    SGSTD_ID = 12  # 陀螺仪尺度因子std
    SASTD_ID = 15  # 陀螺仪尺度因子std


class KRLF_GIEngine:
    #  更新时间对齐误差，IMU状态和观测信息误差小于它则认为两者对齐
    TIME_ALIGN_ERR = 0.001

    # Kalman滤波相关
    RANK = 21  # 状态量
    NOISERANK = 18

    def __init__(self, options: GINSOptions):
        self.options_ = GINSOptions()
        self.timestamp_ = 0.0

        # IMU和GNSS原始数据
        self.imupre_ = IMU()
        self.imucur_ = IMU()
        self.gnssdata_ = GNSS()

        # IMU状态（位置、速度、姿态和IMU误差）
        self.pvacur_ = PVA()
        self.pvapre_ = PVA()
        self.imuerror_ = ImuError()  # 偏差，尺度

        # Kalman滤波相关
        self.Cov_ = np.zeros((KRLF_GIEngine.RANK, KRLF_GIEngine.RANK))
        self.Qc_ = np.zeros((KRLF_GIEngine.NOISERANK, KRLF_GIEngine.NOISERANK))
        self.dx_ = np.zeros((KRLF_GIEngine.RANK, 1))

        self.options_ = options
        # options.print_options()
        self.timestamp_ = 0.0
        imunoise = self.options_.imunoise

        # 初始化系统噪声阵(Q?)
        # initialize noise matrix
        self.Qc_[NoiseID.ARW_ID:NoiseID.ARW_ID + 3, NoiseID.ARW_ID:NoiseID.ARW_ID + 3] = np.diag(
            np.square(imunoise.gyr_arw))
        self.Qc_[NoiseID.VRW_ID:NoiseID.VRW_ID + 3, NoiseID.VRW_ID:NoiseID.VRW_ID + 3] = np.diag(
            np.square(imunoise.acc_vrw))
        self.Qc_[NoiseID.BGSTD_ID:NoiseID.BGSTD_ID + 3,
        NoiseID.BGSTD_ID:NoiseID.BGSTD_ID + 3] = 2 / imunoise.corr_time * np.diag(np.square(imunoise.gyrbias_std))
        self.Qc_[NoiseID.BASTD_ID:NoiseID.BASTD_ID + 3,
        NoiseID.BASTD_ID:NoiseID.BASTD_ID + 3] = 2 / imunoise.corr_time * np.diag(np.square(imunoise.accbias_std))
        self.Qc_[NoiseID.SGSTD_ID:NoiseID.SGSTD_ID + 3,
        NoiseID.SGSTD_ID:NoiseID.SGSTD_ID + 3] = 2 / imunoise.corr_time * np.diag(np.square(imunoise.gyrscale_std))
        self.Qc_[NoiseID.SASTD_ID:NoiseID.SASTD_ID + 3,
        NoiseID.SASTD_ID:NoiseID.SASTD_ID + 3] = 2 / imunoise.corr_time * np.diag(np.square(imunoise.accscale_std))

        # 设置系统状态(位置、速度、姿态和IMU误差)初值和初始协方差
        # set initial state (position, velocity, attitude and IMU error) and covariance
        initstate = self.options_.initstate
        initstate_std = self.options_.initstate_std
        self.initialize(initstate, initstate_std)

    def initialize(self, initstate: NavState, initstate_std: NavState):
        # 初始化位置、速度、姿态
        # initialize position, velocity and attitude
        self.pvacur_.pos = initstate.pos
        self.pvacur_.vel = initstate.vel
        self.pvacur_.att.euler = initstate.euler
        self.pvacur_.att.cbn = RU.euler2matrix(initstate.euler)  # 姿态角转旋转矩阵
        self.pvacur_.att.qbn = RU.euler2quaternion(np.flip(initstate.euler, axis=0))  # 姿态角转四元数矩阵

        # 初始化IMU误差
        # initialize imu error
        self.imuerror_ = initstate.imuerror

        # 给上一时刻状态赋同样的初值
        # set the same value to the previous state
        p = self.pvacur_
        self.pvapre_ = p

        # 初始化协方差
        # initialize covariance
        imuerror_std = initstate_std.imuerror  # 初始化状态协方差
        self.Cov_[StateID.P_ID:StateID.P_ID + 3, StateID.P_ID:StateID.P_ID + 3] = \
            np.diag(np.square(initstate_std.pos))
        self.Cov_[StateID.V_ID:StateID.V_ID + 3, StateID.V_ID:StateID.V_ID + 3] = \
            np.diag(np.square(initstate_std.vel))
        self.Cov_[StateID.PHI_ID:StateID.PHI_ID + 3, StateID.PHI_ID:StateID.PHI_ID + 3] = \
            np.diag(np.square(initstate_std.euler))
        self.Cov_[StateID.BG_ID:StateID.BG_ID + 3, StateID.BG_ID:StateID.BG_ID + 3] = \
            np.diag(np.square(imuerror_std.gyrbias))
        self.Cov_[StateID.BA_ID:StateID.BA_ID + 3, StateID.BA_ID:StateID.BA_ID + 3] = \
            np.diag(np.square(imuerror_std.accbias))
        self.Cov_[StateID.SG_ID:StateID.SG_ID + 3, StateID.SG_ID:StateID.SG_ID + 3] = \
            np.diag(np.square(imuerror_std.gyrscale))
        self.Cov_[StateID.SA_ID:StateID.SA_ID + 3, StateID.SA_ID:StateID.SA_ID + 3] = \
            np.diag(np.square(imuerror_std.accscale))

    def check_update(self):
        # 如果GNSS有效，则将更新时间设置为GNSS时间
        # set update time as the gnss time if gnssdata is valid
        if self.gnssdata_.isvalid:
            updatetime = self.gnssdata_.time
        else:
            updatetime = -1

        # 判断是否需要进行GNSS更新
        # determine if we should do GNSS update
        imupre_ = self.imupre_
        imucur_ = self.imucur_

        res = self.isToUpdate(imupre_.time, imucur_.time, updatetime)
        return res

    def newImuProcess(self):
        # 当前IMU时间作为系统当前状态时间
        # set current IMU time as the current state time
        time = self.imucur_.time
        self.timestamp_ = time

        # 如果GNSS有效，则将更新时间设置为GNSS时间
        # set update time as the gnss time if gnssdata is valid
        if self.gnssdata_.isvalid:
            updatetime = self.gnssdata_.time
        else:
            updatetime = -1

        # 判断是否需要进行GNSS更新
        # determine if we should do GNSS update
        imupre_ = self.imupre_
        imucur_ = self.imucur_
        gnssdata_ = self.gnssdata_
        pvacur_ = self.pvacur_

        res = self.isToUpdate(imupre_.time, imucur_.time, updatetime)

        if res == 0:  # 更新时间不在imutimt1和imutime2之间，且不靠近任何一个
            # 只传播导航状态
            # only propagate navigation state
            dvel, dcbn = self.insPropagation(imupre_, imucur_)
        elif res == 1:  # 更新时间靠近imutime1
            # GNSS数据靠近上一历元，先对上一历元进行GNSS更新
            # gnssdata is near to the previous imudata, we should firstly do gnss update
            self.gnssUpdate(gnssdata_)
            self.stateFeedback()
            self.pvapre_ = self.pvacur_
            dvel, dcbn = self.insPropagation(imupre_, imucur_)
        elif res == 2:  # 更新时间靠近imutime2
            # GNSS数据靠近当前历元，先对当前IMU进行状态传播
            dvel, dcbn = self.insPropagation(imupre_, imucur_)
            self.gnssUpdate(gnssdata_)
            self.stateFeedback()
        else:
            # GNSS数据在两个IMU数据之间(不靠近任何一个), 将当前IMU内插到整秒时刻
            # gnssdata is near current imudata, we should firstly propagate navigation state
            midimu = IMU
            imucur_, midimu = KRLF_GIEngine.imuInterpolate(imupre_, imucur_, updatetime, midimu)

            # 对前一半IMU进行状态传播
            # propagate navigation state for the first half imudata
            middvel_f, middcbn_f = self.insPropagation(imupre_, midimu)

            # 整秒时刻进行GNSS更新，并反馈系统状态
            # do GNSS position update at the whole second and feedback system states
            self.gnssUpdate(gnssdata_)
            self.stateFeedback()

            # 对后一半IMU进行状态传播
            # propagate navigation state for the second half imudata
            self.pvapre_ = self.pvacur_
            middvel_b, middcbn_b = self.insPropagation(midimu, imucur_)

            # 计算全程增量
            dvel = middvel_f + middvel_b
            dcbn = middcbn_f @ middcbn_b

        # 检查协方差矩阵对角线元素是否大于0
        self.checkCov()

        # 更新上一时刻的状态和IMU数据
        # update system state and imudata at the previous epoch
        self.pvapre_ = self.pvacur_
        self.imupre_ = self.imucur_

        return dvel,dcbn

    def checkCov(self):
        for i in range(KRLF_GIEngine.RANK):
            if self.Cov_[i, i] < 0:
                # print(f"Covariance is negative at {self.timestamp_:.10f}, cov[{i}]={self.Cov_[i, i]} !")
                self.Cov_[i, i] = 1e-6
                # raise ValueError("Covariance is negative")
                # exit(os.EXIT_FAILURE)

    def imuCompensate(self, imu: IMU) -> IMU:
        # 补偿IMU零偏误差
        # compensate the imu bias
        imu.dtheta -= self.imuerror_.gyrbias * imu.dt
        imu.dvel -= self.imuerror_.accbias * imu.dt

        # 补偿IMU比例因子误差
        # compensate the imu scale
        one = np.ones(3)
        gyrscale = one + self.imuerror_.gyrscale
        accscale = one + self.imuerror_.accscale
        imu.dtheta = np.multiply(imu.dtheta, np.reciprocal(gyrscale))  # 计算数组元素倒数
        imu.dvel = np.multiply(imu.dvel, np.reciprocal(accscale))
        return imu

    def insPropagation(self, imupre: IMU, imucur: IMU):
        # 对当前IMU数据(imucur)补偿误差, 上一IMU数据(imupre)已经补偿过了
        # compensate imu error to 'imucur', 'imupre' has been compensated
        imucur = self.imuCompensate(imucur)

        # IMU状态更新(机械编排算法)
        # update imustate(mechanization)
        pvapre_ = self.pvapre_
        self.pvacur_, dvel, dcbn = INS.insMech(pvapre_, imupre, imucur)

        # 系统噪声传播，姿态误差采用phi角误差模型，初始化Phi阵(状态转移矩阵)，F阵，Qd阵(传播噪声阵)，G阵(噪声驱动阵)
        # system noise propagate, phi-angle error model for attitude error, initialize Phi (state transition), F matrix, Qd(propagation noise) and G(noise driven) matrix
        Phi = np.identity(KRLF_GIEngine.RANK)
        F = np.zeros((KRLF_GIEngine.RANK, KRLF_GIEngine.RANK))
        Qd = np.zeros((KRLF_GIEngine.RANK, KRLF_GIEngine.RANK))
        G = np.zeros((KRLF_GIEngine.RANK, KRLF_GIEngine.NOISERANK))

        # 使用上一历元状态计算状态转移矩阵
        # compute state transition matrix using the previous state
        rmrn = Earth.meridianPrimeVerticalRadius(pvapre_.pos[0])
        gravity = Earth.gravity(pvapre_.pos)
        wie_n = np.array([Earth.WGS84_WIE * np.cos(pvapre_.pos[0]), 0, -Earth.WGS84_WIE * np.sin(pvapre_.pos[0])])
        wen_n = np.array([pvapre_.vel[1] / (rmrn[1] + pvapre_.pos[2]), -pvapre_.vel[0] / (rmrn[0] + pvapre_.pos[2]),
                          -pvapre_.vel[1] * np.tan(pvapre_.pos[0]) / (rmrn[1] + pvapre_.pos[2])])
        rmh = rmrn[0] + pvapre_.pos[2]
        rnh = rmrn[1] + pvapre_.pos[2]
        accel = imucur.dvel / imucur.dt
        omega = imucur.dtheta / imucur.dt

        # 位置误差 文档没具体公式
        # position error
        temp = np.zeros((3, 3))
        temp[0, 0] = -pvapre_.vel[2] / rmh
        temp[0, 2] = pvapre_.vel[0] / rmh
        temp[1, 0] = pvapre_.vel[1] * np.tan(pvapre_.pos[0]) / rnh
        temp[1, 1] = -(pvapre_.vel[2] + pvapre_.vel[0] * np.tan(pvapre_.pos[0])) / rnh
        temp[1, 2] = pvapre_.vel[1] / rnh
        F[StateID.P_ID:StateID.P_ID + 3, StateID.P_ID:StateID.P_ID + 3] = temp
        F[StateID.P_ID:StateID.P_ID + 3, StateID.V_ID:StateID.V_ID + 3] = np.identity(3)

        # 速度误差
        # velocity error
        temp = np.zeros((3, 3))
        temp[0, 0] = -2 * pvapre_.vel[1] * Earth.WGS84_WIE * np.cos(pvapre_.pos[0]) / rmh - np.power(pvapre_.vel[1],
                                                                                                     2) / rmh / rnh / np.power(
            np.cos(pvapre_.pos[0]), 2)
        temp[0, 2] = pvapre_.vel[0] * pvapre_.vel[2] / rmh / rmh - np.power(pvapre_.vel[1], 2) * np.tan(
            pvapre_.pos[0]) / rnh / rnh
        temp[1, 0] = 2 * Earth.WGS84_WIE * (
                    pvapre_.vel[0] * np.cos(pvapre_.pos[0]) - pvapre_.vel[2] * np.sin(pvapre_.pos[0])) / rmh + \
                     pvapre_.vel[0] * pvapre_.vel[1] / rmh / rnh / np.power(np.cos(pvapre_.pos[0]), 2)
        temp[1, 2] = (pvapre_.vel[1] * pvapre_.vel[2] + pvapre_.vel[0] * pvapre_.vel[1] * np.tan(
            pvapre_.pos[0])) / rnh / rnh
        temp[2, 0] = 2 * Earth.WGS84_WIE * pvapre_.vel[1] * np.sin(pvapre_.pos[0]) / rmh
        temp[2, 2] = -np.power(pvapre_.vel[1], 2) / rnh / rnh - np.power(pvapre_.vel[0],
                                                                         2) / rmh / rmh + 2 * gravity / (
                                 np.sqrt(rmrn[0] * rmrn[1]) + pvapre_.pos[2])
        F[StateID.V_ID:StateID.V_ID + 3, StateID.P_ID:StateID.P_ID + 3] = temp
        temp = np.zeros((3, 3))
        temp[0, 0] = pvapre_.vel[2] / rmh
        temp[0, 1] = -2 * (Earth.WGS84_WIE * np.sin(pvapre_.pos[0]) + pvapre_.vel[1] * np.tan(pvapre_.pos[0]) / rnh)
        temp[0, 2] = pvapre_.vel[0] / rmh
        temp[1, 0] = 2 * Earth.WGS84_WIE * np.sin(pvapre_.pos[0]) + pvapre_.vel[1] * np.tan(pvapre_.pos[0]) / rnh
        temp[1, 1] = (pvapre_.vel[2] + pvapre_.vel[0] * np.tan(pvapre_.pos[0])) / rnh
        temp[1, 2] = 2 * Earth.WGS84_WIE * np.cos(pvapre_.pos[0]) + pvapre_.vel[1] / rnh
        temp[2, 0] = -2 * pvapre_.vel[0] / rmh
        temp[2, 1] = -2 * (Earth.WGS84_WIE * np.cos(pvapre_.pos[0]) + pvapre_.vel[1] / rnh)
        F[StateID.V_ID:StateID.V_ID + 3, StateID.V_ID:StateID.V_ID + 3] = temp
        F[StateID.V_ID:StateID.V_ID + 3, StateID.PHI_ID:StateID.PHI_ID + 3] = RU.skewSymmetric(pvapre_.att.cbn @ accel)
        F[StateID.V_ID:StateID.V_ID + 3, StateID.BA_ID:StateID.BA_ID + 3] = pvapre_.att.cbn
        F[StateID.V_ID:StateID.V_ID + 3, StateID.SA_ID:StateID.SA_ID + 3] = pvapre_.att.cbn @ np.diag(accel)

        # 姿态误差
        # attitude error
        temp = np.zeros((3, 3))
        temp[0, 0] = -Earth.WGS84_WIE * np.sin(pvapre_.pos[0]) / rmh
        temp[0, 2] = pvapre_.vel[1] / rnh / rnh
        temp[1, 2] = -pvapre_.vel[0] / rmh / rmh
        temp[2, 0] = -Earth.WGS84_WIE * np.cos(pvapre_.pos[0]) / rmh - pvapre_.vel[1] / rmh / rnh / np.power(
            np.cos(pvapre_.pos[0]), 2)
        temp[2, 2] = -pvapre_.vel[1] * np.tan(pvapre_.pos[0]) / rnh / rnh
        F[StateID.PHI_ID:StateID.PHI_ID + 3, StateID.P_ID:StateID.P_ID + 3] = temp
        temp = np.zeros((3, 3))
        temp[0, 1] = 1 / rnh
        temp[1, 0] = -1 / rmh
        temp[2, 1] = -np.tan(pvapre_.pos[0]) / rnh
        F[StateID.PHI_ID:StateID.PHI_ID + 3, StateID.V_ID:StateID.V_ID + 3] = temp
        F[StateID.PHI_ID:StateID.PHI_ID + 3, StateID.PHI_ID:StateID.PHI_ID + 3] = -RU.skewSymmetric(wie_n + wen_n)
        F[StateID.PHI_ID:StateID.PHI_ID + 3, StateID.BG_ID:StateID.BG_ID + 3] = -pvapre_.att.cbn
        F[StateID.PHI_ID:StateID.PHI_ID + 3, StateID.SG_ID:StateID.SG_ID + 3] = -pvapre_.att.cbn @ np.diag(omega)

        # IMU零偏误差和比例因子误差，建模成一阶高斯-马尔科夫过程
        # imu bias error and scale error, modeled as the first-order Gauss-Markov process
        F[StateID.BG_ID:StateID.BG_ID + 3,
        StateID.BG_ID:StateID.BG_ID + 3] = -1 / self.options_.imunoise.corr_time * np.identity(3)
        F[StateID.BA_ID:StateID.BA_ID + 3,
        StateID.BA_ID:StateID.BA_ID + 3] = -1 / self.options_.imunoise.corr_time * np.identity(3)
        F[StateID.SG_ID:StateID.SG_ID + 3,
        StateID.SG_ID:StateID.SG_ID + 3] = -1 / self.options_.imunoise.corr_time * np.identity(3)
        F[StateID.SA_ID:StateID.SA_ID + 3,
        StateID.SA_ID:StateID.SA_ID + 3] = -1 / self.options_.imunoise.corr_time * np.identity(3)

        # 系统噪声驱动矩阵
        # system noise driven matrix
        G[StateID.V_ID:StateID.V_ID + 3, NoiseID.VRW_ID:NoiseID.VRW_ID + 3] = pvapre_.att.cbn
        G[StateID.PHI_ID:StateID.PHI_ID + 3, NoiseID.ARW_ID:NoiseID.ARW_ID + 3] = pvapre_.att.cbn
        G[StateID.BG_ID:StateID.BG_ID + 3, NoiseID.BGSTD_ID:NoiseID.BGSTD_ID + 3] = np.identity(3)
        G[StateID.BA_ID:StateID.BA_ID + 3, NoiseID.BASTD_ID:NoiseID.BASTD_ID + 3] = np.identity(3)
        G[StateID.SG_ID:StateID.SG_ID + 3, NoiseID.SGSTD_ID:NoiseID.SGSTD_ID + 3] = np.identity(3)
        G[StateID.SA_ID:StateID.SA_ID + 3, NoiseID.SASTD_ID:NoiseID.SASTD_ID + 3] = np.identity(3)

        # 状态转移矩阵
        # compute the state transition matrix
        Phi = Phi + F * imucur.dt

        # 计算系统传播噪声
        # compute system propagation noise
        Qd = G @ self.Qc_ @ G.T * imucur.dt
        Qd = (Phi @ Qd @ Phi.T + Qd) / 2

        # EKF预测传播系统协方差和系统误差状态
        # do EKF predict to propagate covariance and error state
        self.EKFPredict(Phi, Qd)
        return dvel, dcbn

    # TODO 原版GNSS更新貌似有问题，dx会在第一次更新无故增大
    def gnssUpdate(self, gnssdata: GNSS):
        # IMU位置转到GNSS天线相位中心位置
        # convert IMU position to GNSS antenna phase center position
        Dr_inv = Earth.DRi(self.pvacur_.pos)  # n系相对位置转地理坐标相对位置
        Dr = Earth.DR(self.pvacur_.pos)  # # 地理坐标相对位置转n系相对位置
        antenna_pos = self.pvacur_.pos + Dr_inv @ self.pvacur_.att.cbn @ self.options_.antlever

        # GNSS位置测量新息
        # compute GNSS position innovation
        dz = Dr @ (antenna_pos - gnssdata.blh)  # 相当于单位变成米？

        # 构造GNSS位置观测矩阵
        # construct GNSS position measurement matrix
        H_gnsspos = np.zeros((3, self.Cov_.shape[0]))
        H_gnsspos[0:3, StateID.P_ID:StateID.P_ID + 3] = np.identity(3)
        H_gnsspos[0:3, StateID.PHI_ID:StateID.PHI_ID + 3] = RU.skewSymmetric(
            self.pvacur_.att.cbn @ self.options_.antlever)  # StateID.PHI 姿态ID

        # 位置观测噪声阵
        # construct measurement noise matrix
        R_gnsspos = np.diag(np.multiply(gnssdata.std, gnssdata.std))

        # EKF更新协方差和误差状态
        # do EKF update to update covariance and error state
        dz = dz.reshape(3, 1)
        self.EKFUpdate(dz, H_gnsspos, R_gnsspos)

        # GNSS更新之后设置为不可用
        # Set GNSS invalid after update
        self.gnssdata_.isvalid = False

    def isToUpdate(self, imutime1: float, imutime2: float, updatetime: float) -> int:
        if np.abs(imutime1 - updatetime) < KRLF_GIEngine.TIME_ALIGN_ERR:
            # 更新时间靠近imutime1, （gnss_t, imu_pre_t, imu_cur_t）
            # updatetime is near to imutime1
            return 1
        elif np.abs(imutime2 - updatetime) <= KRLF_GIEngine.TIME_ALIGN_ERR:
            # 更新时间靠近imutime2
            # updatetime is near to imutime2
            return 2
        elif imutime1 < updatetime and updatetime < imutime2:
            # 更新时间在imutime1和imutime2之间, 但不靠近任何一个
            # updatetime is between imutime1 and imutime2, but not near to either
            return 3
        else:
            # 更新时间不在imutimt1和imutime2之间，且不靠近任何一个
            # updatetime is not bewteen imutime1 and imutime2, and not near to either.
            return 0

    def EKFPredict(self, Phi, Qd):
        # assert Phi.shape[0] == self.Cov_.shape[0]
        # assert Qd.shape[0] == self.Cov_.shape[0]

        # 传播系统协方差和误差状态
        # propagate system covariance and error state
        self.Cov_ = Phi @ self.Cov_ @ Phi.transpose() + Qd
        self.dx_ = Phi @ self.dx_
        # 防止奇异
        # self.Cov_ = (self.Cov_ + self.Cov_.transpose()) * 0.5

    def EKFUpdate(self, dz: np.ndarray, H: np.ndarray, R: np.ndarray):
        assert H.shape[1] == self.Cov_.shape[0]
        assert dz.shape[0] == H.shape[0]
        assert dz.shape[0] == R.shape[0]
        # assert  dz.shape[1] == 1

        # 计算Kalman增益
        # compute Kalman Gain
        temp = H @ self.Cov_ @ H.transpose() + R
        K = self.Cov_ @ H.transpose() @ np.linalg.inv(temp)

        # 更新系统误差状态和协方差
        # update system error state and covariance
        I = np.identity(self.Cov_.shape[0])
        I = I - K @ H

        # 如果每次更新后都进行状态反馈，则更新前dx_一直为0，下式可以简化为：dx_ = K * dz;
        # if state feedback is performed after every update, dx_ is always zero before the update
        # the following formula can be simplified as : dx_ = K * dz;
        or_Cov = self.Cov_.copy()
        self.dx_ = self.dx_ + K @ (dz - H @ self.dx_)
        self.Cov_ = I @ self.Cov_ @ I.transpose() + K @ R @ K.transpose()
        # 防止奇异
        # self.Cov_ = (self.Cov_ + self.Cov_.transpose()) * 0.5

    def stateFeedback(self):
        # 位置误差反馈
        # posisiton error feedback
        delta_r = np.concatenate(self.dx_[StateID.P_ID:StateID.P_ID + 3, 0:1]) # 位置误差状态是导航坐标系的
        Dr_inv = Earth.DRi(self.pvacur_.pos)  # n系相对位置转地理坐标相对位置
        self.pvacur_.pos -= Dr_inv @ delta_r # 转换（逆）阵乘n系的误差得到地理坐标的误差

        # 速度误差反馈
        # velocity error feedback
        vectemp = np.concatenate(self.dx_[StateID.V_ID:StateID.V_ID + 3, 0:1])
        self.pvacur_.vel -= vectemp

        # 姿态误差反馈
        # attitude error feedback
        vectemp = np.concatenate(self.dx_[StateID.PHI_ID:StateID.PHI_ID + 3, 0:1])
        qpn = RU.rotvec2quaternion(vectemp)
        qqq = Rotation.from_quat(qpn) * Rotation.from_quat(self.pvacur_.att.qbn)  # 姿态补偿
        self.pvacur_.att.qbn = qqq.as_quat()
        self.pvacur_.att.cbn = RU.quaternion2matrix(self.pvacur_.att.qbn)
        self.pvacur_.att.euler = RU.matrix2euler(self.pvacur_.att.cbn)

        # IMU零偏误差反馈
        # IMU bias error feedback
        vectemp = np.concatenate(self.dx_[StateID.BG_ID:StateID.BG_ID + 3, 0:1])
        self.imuerror_.gyrbias += vectemp
        vectemp = np.concatenate(self.dx_[StateID.BA_ID:StateID.BA_ID + 3, 0:1])
        self.imuerror_.accbias += vectemp

        # IMU比例因子误差反馈
        # IMU sacle error feedback
        vectemp = np.concatenate(self.dx_[StateID.SG_ID:StateID.SG_ID + 3, 0:1])
        self.imuerror_.gyrscale += vectemp
        vectemp = np.concatenate(self.dx_[StateID.SA_ID:StateID.SA_ID + 3, 0:1])
        self.imuerror_.accscale += vectemp

        # 误差状态反馈到系统状态后,将误差状态清零
        # set 'dx' to zero after feedback error state to system state
        self.dx_[:, :] = 0

    def addImuData(self, imu: IMU, compensate=False):
        i = self.imucur_
        self.imupre_ = i
        self.imucur_ = imu
        if compensate:
            self.imucur_ = self.imuCompensate(imu)
        return self.imucur_

    def addGnssData(self, gnss: GNSS):
        self.gnssdata_ = gnss
        self.gnssdata_.isvalid = True

    @staticmethod
    def imuInterpolate(imu1: IMU, imu2: IMU, timestamp: float, midimu: IMU, ):
        if imu1.time > timestamp or imu2.time < timestamp:  # timestamp 是gnss更新时间
            return
        lamda = (timestamp - imu1.time) / (imu2.time - imu1.time)
        midimu.time = timestamp
        midimu.dtheta = imu2.dtheta * lamda
        midimu.dvel = imu2.dvel * lamda
        midimu.dt = timestamp - imu1.time
        imu2.dtheta = imu2.dtheta - midimu.dtheta
        imu2.dvel = imu2.dvel - midimu.dvel
        imu2.dt = imu2.dt - midimu.dt
        return imu2, midimu

    def timestamp(self) -> float:
        return self.timestamp_

    def getNavState(self) -> NavState:
        state = NavState()
        state.pos = self.pvacur_.pos
        state.vel = self.pvacur_.vel
        state.euler = self.pvacur_.att.euler
        state.imuerror = self.imuerror_
        return state

    def getCovariance(self) -> np.ndarray:
        return self.Cov_
