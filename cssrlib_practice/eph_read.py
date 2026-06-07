from cssrlib.ephemeris import satposs
from cssrlib.gnss import rSigRnx
import cssrlib.gnss as gn
import cssrlib.rinex as rn

# 基本路径设置
bdir = '/mnt/sdb/home/chenlj/code/position/RTK_practice/cssrlib/data/zonghe/'
navfile = bdir+'TD1050-0915PR-gz-zonghe-VLG-0823-RTCM-10Hz.nav'
obsfile = bdir+'TD1050-0915PR-gz-zonghe-VLG-0823-RTCM-10Hz.obs'

# 根据obs文件调整观测信号类型
sigs = [rSigRnx("GC1C"), rSigRnx("GL1C"), rSigRnx("GS1C"),
        rSigRnx("EC1C"), rSigRnx("EL1C"), rSigRnx("ES1C")]
        # rSigRnx("CC2I"), rSigRnx("CL2I"), rSigRnx("CS2I")]

# rover
dec = rn.rnxdec()
dec.setSignals(sigs)
nav = gn.Nav(nf=1)  # nav.pmode默认为1，即默认为kinematic; nav.nf默认为2，即默认跟踪每颗卫星的2个频率（双频接收机）
dec.decode_nav(navfile, nav)
dec.autoSubstituteSignals()  # 自动寻找同一频段的替代信号, 新增 Chen251021
dec.decode_obsh(obsfile)

# 改用为根据观测文件自调节时间长度 Chen251031
_, nepoch_r = dec.get_obs_timestamps(obsfile)
nep = nepoch_r
for ne in range(nep):
    obs = dec.decode_obs()

    # 获取卫星位置, 速度等
    rs, vs, dts, svh, nsat = satposs(obs, nav)
    a = 1
