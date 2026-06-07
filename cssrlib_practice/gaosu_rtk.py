"""
 static test for RTK
"""
# import matplotlib
# matplotlib.use('TkAgg')  # 或者 'TkAgg'
import matplotlib.pyplot as plt
import numpy as np
import sys
import pandas as pd

import cssrlib.rinex as rn
import cssrlib.gnss as gn
from cssrlib.rtk import rtkpos
from cssrlib.gnss import rSigRnx
from cssrlib.peph import atxdec, searchpcv

# 基本路径设置
bdir = '/mnt/sdb/home/chenlj/code/position/RTK_practice/cssrlib/data/gaosu/'
# navfile = bdir+'SEPT078M.21P'
# obsfile = bdir+'SEPT078M1.21O'
# basefile = bdir+'3034078M1.21O'
navfile = bdir+'TD1050-0915PR-Gaosu-VLGchezai-0823-RTCM-10Hz.nav'
obsfile = bdir+'TD1050-0915PR-Gaosu-VLGchezai-0823-RTCM-10Hz.obs'
basefile = bdir+'base-Gaosu-0823-RTCM.obs'
atxfile = bdir+"test.atx"
np.set_printoptions(legacy='1.25')  # 新增, 使得SciView视图不显示数值类型, 只显示数值 Chen251014

# 读取真值文件并转换为GPST时间
df_ref = pd.read_csv(bdir+'0823_LCI_AM_reference_nmea_50Hz.csv')
ref_pd_retiming = df_ref.copy()
ref_pd_retiming['UnixTimeMillis_org'] = ref_pd_retiming['UnixTimeMillis_ref']
ref_mtimes = np.array(df_ref['UnixTimeMillis_ref'], dtype='float') + 18000  # UTC转GPST
# ref_mtimes = [f'{i:.0f}' for i in ref_mtimes]
ref_mtimes = np.round(ref_mtimes).astype(int)
ref_pd_retiming['UnixTimeMillis_ref'] = ref_mtimes

# xyz_ref = [-3962108.673,   3381309.574,   3668678.638]  # 用户真值/enu坐标系原点
# pos_ref = gn.ecef2pos(xyz_ref)  # ECEF to LLH
#
# Define signals to be processed
#
# sigs = [rSigRnx("GC1C"), rSigRnx("GC2W"),   # 第一个字符：系统标识；第二个字符：观测类型；
#         rSigRnx("EC1C"), rSigRnx("EC5Q"),   # 第三个字符：频带号；第四个字符：跟踪属性。
#         rSigRnx("GL1C"), rSigRnx("GL2W"), rSigRnx("GS1C"), rSigRnx("GS2W"),
#         rSigRnx("EL1C"), rSigRnx("EL5Q"), rSigRnx("ES1C"), rSigRnx("ES5Q")]
#
# sigsb = [rSigRnx("GC1C"), rSigRnx("GC2W"),
#          rSigRnx("EC1X"), rSigRnx("EC5X"),
#          rSigRnx("GL1C"), rSigRnx("GL2W"), rSigRnx("GS1C"), rSigRnx("GS2W"),
#          rSigRnx("EL1X"), rSigRnx("EL5X"), rSigRnx("ES1X"), rSigRnx("ES5X")]

# 根据obs文件调整观测信号类型
sigs = [rSigRnx("GC1C"), rSigRnx("GL1C"), rSigRnx("GS1C"),
        rSigRnx("EC1C"), rSigRnx("EL1C"), rSigRnx("ES1C")]

sigsb = [rSigRnx("GC1C"), rSigRnx("GL1C"), rSigRnx("GS1C"),
         rSigRnx("EC1C"), rSigRnx("EL1C"), rSigRnx("ES1C")]

# rover
dec = rn.rnxdec()
dec.setSignals(sigs)
nav = gn.Nav(nf=1)  # nav.pmode默认为1，即默认为kinematic; nav.nf默认为2，即默认跟踪每颗卫星的2个频率（双频接收机）
dec.decode_nav(navfile, nav)
dec.autoSubstituteSignals()  # 自动寻找同一频段的替代信号, 新增 Chen251021

# base
# rb_enu = [23.152245124, 113.419960459, 19.0009]
# nav.rb = gn.pos2ecef(rb_enu, isdeg=True)
nav.rb = [-2332148.0735,  5384129.2676,  2492238.1077]  # 高速场景基站真值

decb = rn.rnxdec()
decb.setSignals(sigsb)
decb.decode_obsh(basefile)
dec.decode_obsh(obsfile)
dec.autoSubstituteSignals()  # 新增 Chen251021

# Load the antenna data for the satellites and receivers
dec.ant = "{:16s}{:4s}".format("JAVRINGANT_DM", "SCIS")
decb.ant = "{:16s}{:4s}".format("TRM59800.80", "NONE")

atx = atxdec()
atx.readpcv(atxfile)
# atx.readngspcv(ngsantfile)

# Set antenna PCO/PCV data
nav.rcv_ant = searchpcv(atx.pcvr, dec.ant,  dec.ts)
nav.rcv_ant_b = searchpcv(atx.pcvr, decb.ant, decb.ts)

if nav.rcv_ant is None:
    print("ERROR: no rover antenna found!")
    sys.exit(1)

if nav.rcv_ant_b is None:
    print("ERROR: no base antenna found!")
    sys.exit(1)

# Initialize the variables for position and the RTK configuration parameters
rtk = rtkpos(nav, dec.pos, logfile='test_rtk.log')  # dec.pos是由观测文件提供的用户位置初始值
rr = dec.pos

# Run RTK positioning
# nep = 60*2  # 历元数
# 改用为根据观测文件自调节时间长度 Chen251031
_, nepoch_r = dec.get_obs_timestamps(obsfile)
_, nepoch_b = dec.get_obs_timestamps(basefile)
nep = min(int(nepoch_r/10), nepoch_b)

t = np.zeros(nep)
smode = np.zeros(nep, dtype=int)
dist_enu_rtk = np.zeros((nep, 5))
dist_err_xyz_rtk = np.zeros((nep, 4))
rtk_ecef = np.zeros((nep, 3))
true_ecef = np.zeros((nep, 3))

i = 0
t_ref = gn.gtime_t()
for ne in range(nep):
    obs, obsb = rn.sync_obs(dec, decb, dt_th=0.01)  # 基准站与用户观测频率不一致，需提高时间戳匹配精度 Chen251024
    if ne == 0:
        t0 = nav.t = obs.t
    rtk.process(obs, obsb=obsb)
    t[ne] = gn.timediff(nav.t, t0)
    sol = nav.xa[0:3] if nav.smode == 4 else nav.x[0:3]

    # if gn.time2str(obs.t) == '2022-08-23 03:44:39':
    #     a = 1

    # 寻找对齐时间戳  新增Chen251028
    while True:
        t_ref_i = ref_pd_retiming['UnixTimeMillis_ref'][i]
        t_ref.time = t_ref_i // 1000  # 整数秒部分
        t_ref.sec = (t_ref_i % 1000) / 1000.0  # 小数秒部分（毫秒转换为秒）
        dt_th = 0.001
        dt = gn.timediff(obs.t, t_ref)
        if np.abs(dt) <= dt_th:
            break
        if dt > dt_th:
            i += 1
        # elif dt < dt_th:
        #     obs = dec.decode_obs()

    # 根据对齐的时间戳提取相应数据
    llh_ref = ref_pd_retiming.loc[ref_pd_retiming['UnixTimeMillis_ref'] == t_ref_i,
                                 ['LatitudeDegrees', 'LongitudeDegrees', 'AltitudeMeters']].values.reshape(-1)
    pos_ref = llh_ref.copy()
    pos_ref[0] = pos_ref[0] / 180 * np.pi
    pos_ref[1] = pos_ref[1] / 180 * np.pi
    xyz_ref = ref_pd_retiming.loc[ref_pd_retiming['UnixTimeMillis_ref'] == t_ref_i,
                                                 ['ecefX', 'ecefY', 'ecefZ']].values.reshape(-1)
    # if i == 36400:
    #     pos_ref0 = pos_ref

    dist_enu_rtk[ne, :3] = gn.ecef2enu(pos_ref, sol-xyz_ref)                                # 记录enu分轴误差
    dist_enu_rtk[ne, 3] = np.sqrt(np.sum((gn.ecef2enu(pos_ref, sol-xyz_ref))**2, axis=0))   # 记录enu三维误差
    dist_enu_rtk[ne, 4] = np.sqrt(dist_enu_rtk[ne, 0]**2+dist_enu_rtk[ne, 1]**2)            # 记录enu水平误差
    dist_err_xyz_rtk[ne, :3] = sol-xyz_ref                                                  # 记录ecef分轴误差
    dist_err_xyz_rtk[ne, 3] = np.sqrt(np.sum((sol-xyz_ref)**2, axis=0))                     # 记录ecef三维误差
    rtk_ecef[ne, :] = sol
    true_ecef[ne, :] = xyz_ref
    smode[ne] = nav.smode

    # Log to standard output
    sys.stdout.write('\r {} ENU {:7.4f} {:7.4f} {:7.4f}, 2D {:6.4f}, mode {:1d}'
                     .format(gn.time2str(obs.t),
                             dist_enu_rtk[ne, 0], dist_enu_rtk[ne, 1], dist_enu_rtk[ne, 2],
                             np.sqrt(dist_enu_rtk[ne, 0]**2+dist_enu_rtk[ne, 1]**2),
                             smode[ne]))

dec.fobs.close()
decb.fobs.close()

# 计算误差均值
percent = 0
if percent:
    score_xyz_rtk = np.mean([np.quantile(dist_err_xyz_rtk[:, 3], 0.50),
                             np.quantile(dist_err_xyz_rtk[:, 3], 0.95)])
    score_en_rtk = np.mean([np.quantile(dist_enu_rtk[:, 4], 0.50),
                            np.quantile(dist_enu_rtk[:, 4], 0.95)])
    score_h_rtk = np.mean([np.quantile(np.abs(dist_enu_rtk[:, 2]), 0.50),
                           np.quantile(np.abs(dist_enu_rtk[:, 2]), 0.95)])
else:
    score_xyz_rtk = np.mean(dist_err_xyz_rtk[:, 3])
    score_en_rtk = np.mean(dist_enu_rtk[:, 4])
    score_h_rtk = np.mean(np.abs(dist_enu_rtk[:, 2]))

print(f'\n3D_mean_error: {np.mean(dist_err_xyz_rtk[:, 3])}')
print(f'2Den_mean_error: {np.mean(dist_enu_rtk[:, 4])}')

# 绘制误差曲线
fig_type = 1
ylim = 0.02
if fig_type == 1:
    plt.plot(t, dist_enu_rtk[:, :4])
    plt.xticks(np.arange(0, nep+1, step=30))
    plt.ylabel('position error [m]')
    plt.xlabel('time[s]')
    plt.legend(['east', 'north', 'up', 'error_3D'])
    plt.title(f'error_3D: {score_xyz_rtk:.3f} m\n'
              f'error_2D: {score_en_rtk:.3f} m | '
              f'error_h: {score_h_rtk:.3f} m')
    plt.grid()
    # plt.axis([0, ne, -ylim, ylim])
    plt.savefig(f'/mnt/sdb/home/chenlj/code/position/RTK_practice/cssrlib_practice/output'
                f'/gaosu.png')
    plt.close()
    # plt.show()
else:
    plt.plot(dist_enu_rtk[:, 0], dist_enu_rtk[:, 1])
    plt.xlabel('easting [m]')
    plt.ylabel('northing [m]')
    plt.grid()
    plt.axis([-ylim, ylim, -ylim, ylim])
    plt.show()

# 绘制三维轨迹
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot3D(true_ecef[:, 0], true_ecef[:, 1], true_ecef[:, 2],
          label='Ground Truth', c='b', linewidth=2)
ax.plot3D(rtk_ecef[:, 0], rtk_ecef[:, 1], rtk_ecef[:, 2],
          label='RTK Estimated', c='r', linestyle='--')
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_zlabel('Z [m]')
ax.legend()
plt.title(f'3D Trajectory Comparison(epoch: {nep})')
plt.savefig(f'/mnt/sdb/home/chenlj/code/position/RTK_practice/cssrlib_practice/output'
            f'/gaosu_trajectory.png')
plt.close()
a = 1
