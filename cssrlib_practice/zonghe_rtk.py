"""
 static test for RTK
"""
# import matplotlib
# matplotlib.use('TkAgg')  # 或者 'TkAgg'
import matplotlib.pyplot as plt
import numpy as np
import sys
import pandas as pd
import pymap3d.vincenty as pmv

import cssrlib.rinex as rn
import cssrlib.gnss as gn
from cssrlib.rtk import rtkpos
from cssrlib.gnss import rSigRnx
from cssrlib.peph import atxdec, searchpcv

# 基本路径设置
bdir = '/mnt/sdb/home/chenlj/code/position/RTK_practice/cssrlib/data/zonghe/'
# navfile = bdir+'SEPT078M.21P'
# obsfile = bdir+'SEPT078M1.21O'
# basefile = bdir+'3034078M1.21O'
navfile = bdir+'TD1050-0915PR-gz-zonghe-VLG-0823-RTCM-10Hz.nav'
obsfile = bdir+'TD1050-0915PR-gz-zonghe-VLG-0823-RTCM-10Hz.obs'
basefile = bdir+'base-GZ-zonghe-20220823-RTCM-1Hz.obs'
atxfile = bdir+"test.atx"
np.set_printoptions(legacy='1.25')  # 新增, 使得SciView视图不显示数值类型, 只显示数值 Chen251014

# 读取真值文件并转换为GPST时间
df_ref = pd.read_csv(bdir+'0823_LCI_PM_reference_nmea.csv')
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
        # rSigRnx("CC2I"), rSigRnx("CL2I"), rSigRnx("CS2I")]

sigsb = [rSigRnx("GC1C"), rSigRnx("GL1C"), rSigRnx("GS1C"),
         rSigRnx("EC1C"), rSigRnx("EL1C"), rSigRnx("ES1C")]
         # rSigRnx("CC2I"), rSigRnx("CL2I"), rSigRnx("CS2I")]

# rover
dec = rn.rnxdec()
dec.setSignals(sigs)
nav = gn.Nav(nf=1)  # nav.pmode默认为1，即默认为kinematic; nav.nf默认为2，即默认跟踪每颗卫星的2个频率（双频接收机）
dec.decode_nav(navfile, nav)
dec.autoSubstituteSignals()  # 自动寻找同一频段的替代信号, 新增 Chen251021

# base
# rb_enu = [23.152245130, 113.324368452, 19.0009]
# nav.rb = gn.pos2ecef(rb_enu, isdeg=True)
nav.rb = [-2323161.9760,  5388012.7169,  2492238.1084]  # 综合场景基站真值

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
rtk_ecef = np.zeros((nep, 7))
t_ref = gn.gtime_t()

for ne in range(nep):
    obs, obsb = rn.sync_obs(dec, decb, dt_th=0.01)  # 基准站与用户观测频率不一致，需提高时间戳匹配精度 Chen251024
    if ne == 0:
        t0 = nav.t = obs.t
    rtk.process(obs, obsb=obsb)
    t[ne] = gn.timediff(nav.t, t0)
    sol = nav.xa[0:3] if nav.smode == 4 else nav.x[0:3]
    smode[ne] = nav.smode

    # 记录结果
    rtk_ecef[ne, 0] = int((obs.t.time + obs.t.sec) * 1000)  # 精确到毫秒
    rtk_ecef[ne, 1:4] = sol
    rtk_ecef[ne, 4:] = gn.ecef2pos(sol)

dec.fobs.close()
decb.fobs.close()

# 与真值做比较
column_names = ['UnixTimeMillis_ref', 'ecefX_rtk', 'ecefY_rtk', 'ecefZ_rtk',
                'latitude(deg)', 'longitude(deg)', 'height(m)']
merge_columns = ['UnixTimeMillis_ref']
rtk_ecef_df = pd.DataFrame(rtk_ecef, columns=column_names)
rtk_ecef_df['UnixTimeMillis_ref'] = rtk_ecef_df['UnixTimeMillis_ref'].astype('int64')
rtk_ecef_df['latitude(deg)'] = rtk_ecef_df['latitude(deg)'] / np.pi * 180
rtk_ecef_df['longitude(deg)'] = rtk_ecef_df['longitude(deg)'] / np.pi * 180
baseline = rtk_ecef_df.merge(ref_pd_retiming, on=merge_columns, suffixes=('_rtk', ''))  # 对齐时间戳

true_XYZ = np.array([baseline['ecefX'], baseline['ecefY'], baseline['ecefZ']])
true_llh = np.array([baseline['LatitudeDegrees'], baseline['LongitudeDegrees'], baseline['AltitudeMeters']])
rtk_XYZ = np.array([baseline['ecefX_rtk'], baseline['ecefY_rtk'], baseline['ecefZ_rtk']])
rtk_llh = np.array([baseline['latitude(deg)'], baseline['longitude(deg)'], baseline['height(m)']])

dist_err_xyz_rtk = np.sqrt(np.sum((true_XYZ-rtk_XYZ)**2, axis=0))
dist_ll_rtk, _ = np.array(pmv.vdist(rtk_llh.T[:, 0], rtk_llh.T[:, 1], true_llh.T[:, 0], true_llh.T[:, 1]))
dist_h_rtk = np.abs(rtk_llh[2, :] - true_llh[2, :])
dist_llh_rtk = np.vstack((rtk_llh - true_llh, dist_err_xyz_rtk))

# 计算误差均值
percent = 0
if percent:
    score_xyz_rtk = np.mean([np.quantile(dist_err_xyz_rtk, 0.50),
                             np.quantile(dist_err_xyz_rtk, 0.95)])
    score_ll_rtk = np.mean([np.quantile(dist_ll_rtk, 0.50),
                            np.quantile(dist_ll_rtk, 0.95)])
    score_h_rtk = np.mean([np.quantile(np.abs(dist_h_rtk), 0.50),
                           np.quantile(np.abs(dist_h_rtk), 0.95)])
else:
    score_xyz_rtk = np.mean(dist_err_xyz_rtk)
    score_ll_rtk = np.mean(dist_ll_rtk)
    score_h_rtk = np.mean(np.abs(dist_h_rtk))

print(f'\n3D_mean_error: {np.mean(dist_err_xyz_rtk)}')
print(f'2Dll_mean_error: {np.mean(dist_ll_rtk)}')

# 绘制误差曲线
fig_type = 1
ylim = 0.02
if fig_type == 1:
    plt.plot(t, dist_llh_rtk.T)
    plt.xticks(np.arange(0, nep+1, step=30))
    plt.ylabel('position error [m]')
    plt.xlabel('time[s]')
    plt.legend(['latitude', 'longitude', 'height', 'error_3D'])
    plt.title(f'error_3D: {score_xyz_rtk:.3f} m\n'
              f'error_2D: {score_ll_rtk:.3f} m | '
              f'error_h: {score_h_rtk:.3f} m')
    plt.grid()
    # plt.axis([0, ne, -ylim, ylim])
    plt.savefig(f'/mnt/sdb/home/chenlj/code/position/RTK_practice/cssrlib_practice/output'
                f'/zonghe.png')
    plt.close()
    # plt.show()
else:
    plt.plot(dist_llh_rtk[0, :], dist_llh_rtk[1, :])
    plt.xlabel('latitude [deg]')
    plt.ylabel('longitude [deg]')
    plt.grid()
    plt.axis([-ylim, ylim, -ylim, ylim])
    plt.show()

# 绘制三维轨迹
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot3D(true_XYZ[:, 0], true_XYZ[:, 1], true_XYZ[:, 2],
          label='Ground Truth', c='b', linewidth=2)
ax.plot3D(rtk_ecef[:, 0], rtk_ecef[:, 1], rtk_ecef[:, 2],
          label='RTK Estimated', c='r', linestyle='--')
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_zlabel('Z [m]')
ax.legend()
plt.title(f'3D Trajectory Comparison(epoch: {nep})')
plt.savefig(f'/mnt/sdb/home/chenlj/code/position/RTK_practice/cssrlib_practice/output'
            f'/zonghe_trajectory.png')
plt.close()
a = 1
