import pandas as pd
import numpy as np
import cssrlib.gnss as gn
import cssrlib.rinex as rn
from tqdm import tqdm
from cssrlib.ephemeris import satposs
from cssrlib.gnss import sat2id, sat2prn, geodist, satazel
from cssrlib.gnss import time2str
from cssrlib.gnss import uTYP,  rSigRnx
np.set_printoptions(legacy='1.25')

"""
该程序直接处理rinex文件生成卫星测量值和卫星位置的csv文件
创建人：陈立基
"""

def build_signal_list(sys_char, freq_list):
    """自动生成类似 GC1C, GL1C, GS1C, GD1C 这种形式的信号描述"""

    obs_types = ["C", "L", "S", "D"]  # 伪距/载波相位/CN0/多普勒
    sigs = []

    for freq in freq_list:            # 例如 ["1C","2W","5Q"]
        for typ in obs_types:         # C/L/S/D
            sigs.append(rSigRnx(f"{sys_char}{typ}{freq}"))

    return sigs


# 基本路径设置
bdir = '/mnt/sdb/home/tangjh/KF-GINS-Py-main/dataset_Urbannav/1_UrbanNav-HK-Medium-Urban-1/'
navfile = bdir+'hkst137c.21n'
obsfile = bdir+'UrbanNav-HK-Medium-Urban-1.ublox.f9p.obs'

# 根据obs文件调整观测信号类型
systems = {
    'G': ['1C', '2W', '5Q'],    # GPS
    'E': ['1C', '5Q', '7Q'],    # Galileo
    'C': ['2I', '7I', '6I'],    # BDS
    'R': ['1C', '2C']           # Glonass
}

# 读取导航文件
dec = rn.rnxdec()
nav = gn.Nav()
dec.decode_nav(navfile, nav)

# 循环处理单系统信号
system_csv_list = []  # 存储每个系统生成的 CSV 文件路径
for sys_char, freq_list in systems.items():
    print(f"\n====== 开始处理系统 {sys_char}（{len(freq_list)} 频） ======")

    # 设置该系统的信号
    sigs_sys = build_signal_list(sys_char, freq_list)

    dec = rn.rnxdec()  # 每次循环需要重新初始化一下
    dec.setSignals(sigs_sys)
    dec.decode_obsh(obsfile)
    dec.autoSubstituteSignals()

    enph_list = []
    gnss_list = []
    _, nep = dec.get_obs_timestamps(obsfile)  # 根据观测文件自调节时间长度
    for ne in tqdm(range(nep), desc="处理历元"):
        # 获取观测文件
        obs = dec.decode_obs()
        if obs is None:
            continue

        # 获取当前历元时间
        utc_time = gn.gpst2utc(obs.t)   # 根据头文件信息, 原始时间为GPST时间
        unix_millis = int((utc_time.time + utc_time.sec) * 1000)  # 精确到毫秒

        # 获取卫星位置, 速度等
        rs, vs, dts, svh, nsat = satposs(obs, nav)

        # 找出健康卫星的索引 (svh == 0)
        valid_idx = np.where(svh == 0)[0]

        # 如果没有健康卫星，则跳过该历元
        if len(valid_idx) == 0:
            print(f'{time2str(obs.t)}时刻无可见健康卫星')
            continue

        # 提取健康卫星的数据
        healthySat_rs = rs[valid_idx, :]
        healthySat_vs = vs[valid_idx, :]
        healthySat_dts = dts[valid_idx]
        healthySat_id = obs.sat[valid_idx]
        healthyRawP = obs.P[valid_idx]
        healthyRawL = obs.L[valid_idx]
        healthyRawD = obs.D[valid_idx]
        healthyRawS = obs.S[valid_idx]

        # 为每颗健康卫星构建一行数据，并加入总列表
        for i, sat in enumerate(healthySat_id):     # 遍历卫星
            sys, _ = sat2prn(sat)
            rnx_sigs = dec.getSignals(sys, uTYP.L)

            # # 高度角（需提供接收机位置rr_）
            # _, ele = geodist(healthySat_rs[i, :], rr_)
            #
            # # 方位角（需提供接收机位置rr_）
            # pos = gn.ecef2pos(rr_)
            # _, azi = satazel(pos, ele)

            for j, sig in enumerate(rnx_sigs):      # 遍历系统内不同频段

                gnss_list = [
                    unix_millis,            # 0. 时间
                    sat2id(sat),            # 1. 卫星类型/gnss_sv_id
                    sat,                    # 2. 卫星编号/Svid
                    sys.name,               # 3. 星座类型/gnss_id
                    healthyRawP[i, j],      # 4. 原始伪距/RawPseudorangeMeters
                    healthyRawL[i, j],      # 5. 原始载波相位/carrier_phase
                    healthyRawD[i, j],      # 6. 原始多普勒/raw_doppler_hz
                    healthyRawS[i, j],      # 7. 载噪比/Cn0DbHz
                    sig.str(),              # 8. 观测信号类型/obs_signal_type
                    *healthySat_rs[i],      # 9.10.11. 卫星位置
                    *healthySat_vs[i],      # 12.13.14. 卫星速度
                    healthySat_dts[i],       # 15. 卫星时钟偏置（卫星钟差）
                    # ele                     # 16. 高度角
                    # azi                     # 17. 方位角
                ]

                enph_list.append(gnss_list)
    dec.fobs.close()

    # 创建列名
    column_names = ['utcTimeMillis', 'gnss_sv_id', 'Svid', 'gnss_id', 'RawPseudorangeMeters', 'carrier_phase',
                    'raw_doppler_hz', 'Cn0DbHz', 'obs_signal_type',
                    'SvPositionXEcefMeters', 'SvPositionYEcefMeters', 'SvPositionZEcefMeters',
                    'SvVelocityXEcefMetersPerSecond', 'SvVelocityYEcefMetersPerSecond', 'SvVelocityZEcefMetersPerSecond',
                    'SvClockBiasSeconds']

    # 转换为DataFrame并保存为CSV
    enph_df = pd.DataFrame(enph_list, columns=column_names)
    csv_sys = f"{bdir}output_{sys_char}.csv"
    enph_df.to_csv(csv_sys, index=False)
    system_csv_list.append(csv_sys)

# 所有系统的信息拼接在一起
df_all = pd.concat(
    [pd.read_csv(fp) for fp in system_csv_list],
    ignore_index=True
)

# 重新排序
df_all.sort_values(by=['utcTimeMillis', 'Svid'], inplace=True)

# 保存为csv文件
merged_csv = f"{bdir}2512336b_25O.csv"
df_all.to_csv(merged_csv, index=False)

a = 1
