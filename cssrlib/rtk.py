"""
module for RTK positioning

"""

from cssrlib.pppssr import pppos
import numpy as np
from copy import deepcopy
from cssrlib.ephemeris import satposs


class rtkpos(pppos):
    """ class for RTK processing """

    def __init__(self, nav, pos0=np.zeros(3), logfile=None):
        """ initialize variables for PPP-RTK """

        # trop, iono from cssr
        # phase windup model is local/regional
        super().__init__(nav=nav, pos0=pos0, logfile=logfile,
                         trop_opt=0, iono_opt=0, phw_opt=0)

        self.nav.eratio = np.ones(self.nav.nf)*100  # [-] factor 用于高度角模型中的频率相关误差放大因子  原: 50
        self.nav.err = [0, 0.01, 0.005]/np.sqrt(2)  # [m] sigma  用于高度角模型中的a, b取值  原: [0, 0.01, 0.005]
        self.nav.sig_p0 = 30.0  # [m]  # 注1: PPP/PPP-RTK为100; 注2: 这里虽然改了, 但后续似乎没有用到
        self.nav.thresar = 1.5  # AR acceptance threshold  2.0 --> 1.5 Chen251029

        # 0:float-ppp,1:continuous,2:instantaneous,3:fix-and-hold
        self.nav.armode = 3     # AR is enabled  原: 1 Chen251030

        # 新增
        self.pmode = 1  # 0: static, 1: kinematic  改为静态 Chen251027
        self.nav.parmode = 1  # 1: normal, 2: PAR（部分模糊度固定策略）   原: 2 Chen251029
        self.nav.sig_n0 = 1.0  # [cyc] 周   原: 30.0 Chen251029
        self.nav.sig_qp = 1.0 / np.sqrt(1)  # [m/sqrt(s)]    原: 0.01 Chen251028
        self.nav.sig_qv = 1.0 / np.sqrt(1)  # [m/s/sqrt(s)]    原: 1.0 Chen251028

    def base_process(self, obs, obsb, rs, dts, svh):
        """ processing for base station in RTK """
        rsb, vsb, dtsb, svhb, _ = satposs(obsb, self.nav)
        yr, er, elr = self.zdres(
            obsb, None, None, rsb, vsb, dtsb, self.nav.rb, 0)  # rtype: 0 - 基准站; 1 - 流动站.

        # Editing observations (base/rover)
        sat_ed_r = self.qcedit(obsb, rsb, dtsb, svhb, rr=self.nav.rb)
        sat_ed_u = self.qcedit(obs, rs, dts, svh)

        # define common satellite between base and rover
        sat_ed = np.intersect1d(sat_ed_u, sat_ed_r, True)
        ir = np.intersect1d(obsb.sat, sat_ed, True, True)[1]  # 共视卫星在基准站观测数据中的索引
        iu = np.intersect1d(obs.sat, sat_ed, True, True)[1]   # 共视卫星在流动站观测数据中的索引
        ns = len(iu)

        y = np.zeros((ns*2, self.nav.nf*2))  # 非差残差
        e = np.zeros((ns*2, 3))

        y[ns:, :] = yr[ir, :]
        e[ns:, :] = er[ir, :]

        obs_ = deepcopy(obs)
        obs_.sat = obs.sat[iu]
        obs_.L = obs.L[iu, :]-obsb.L[ir, :]  # 站间单差
        obs_.P = obs.P[iu, :]-obsb.P[ir, :]

        return y, e, iu, obs_
