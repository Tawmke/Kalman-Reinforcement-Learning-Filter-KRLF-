import numpy as np
import pandas as pd
import os
import sys
cur_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(cur_dir, '..')
sys.path.append(src_dir)

from env.SZdata_GNSSINS_KRLF import * # RL环境
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C
from model.ppo_recurrent_ATF1_AKF import RecurrentPPO
from model.ppo_recurrent_ATF1_AKF_FT import RecurrentPPO_FT # 微调模型
from funcs_rl.utilis_eskf import *
from model.model_ATF_KF import *
import time
import argparse
import yaml
"""
KRLF强化学习环境用于深圳数据集，训练时候不区分场景
"""
if __name__ == "__main__":
    # 可选环境：'SZ_canyon_RTK'
    triptype = 'SZ_RTK'
    conf_name = f'KRLF_{triptype}.yaml'
    flag_finetune = False
    parser = argparse.ArgumentParser(description='KRLF')  # 初始化解释器
    parser.add_argument('--conf', type=str, help='configuration file path')  # 定义了一个可选的文件路径参数
    args = parser.parse_args()
    try:
        filename = None
        if args.conf is None:
            filename = os.path.abspath(f'{dir_path}/src/configs/{conf_name}')
        else:
            filename = args.conf
        with open(filename, 'r', encoding='utf-8') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)  # 从YAML文件(filename)中加载配置数据
    except Exception as e:
        print(f"Error details: {str(e)}")
        raise Exception(
            "Failed to read configuration file. Please check the path and format of the configuration file!")

    os.environ["CUDA_VISIBLE_DEVICES"] = config['system']['gpu_id']

    # 获取训练列表和测试类型
    triptype = config['training_settings']['triptype']
    tripIDlist = trip_type_maping[triptype]
    tripIDlist = [item for item in tripIDlist if item not in exclude_traj]  # 剔除不需要的traj
    moreteststypelist = config['testing_dataset_setting'][triptype]  # 测试类型

    # 设置是否使用所有轨迹做训练
    if config['training_settings']['alltraj']: # 全轨迹按比例划分训练测试
        ratio = 1
        traj_type_target_train = config['training_settings']['traj_type_target_train']  # 轨迹数据的比例
        traj_type_target_test = config['training_settings']['traj_type_target_test']
        trajdata_range = [0, len(tripIDlist) - 1]
    else:
        ratio = 0.5  # 数据集所有数据一半训练一半测试
        traj_type_target_train = [0, 1]  # 轨迹数据的比例
        traj_type_target_test = [0, 1]
        trajdata_range =  [0, int(np.ceil(len(tripIDlist) * ratio))]

    # 只测试不训练，导入模型路径设置
    if config["testing_settings"]['onlytesting']:
        testdate = config["testing_settings"]['testdate']
        model_basefolder = config["testing_settings"]['model_basefolder']
        model_basefolder=f'{dir_path}/records_values/{testdate}/{model_basefolder}'
        if config["testing_settings"]['model_name_list']:
            model_folderlist = config["testing_settings"]['model_name_list']  # only for testing
        else:
            model_folderlist=os.listdir(model_basefolder)
            model_folderlist.sort()

    # 开启训练模式
    if config["testing_settings"]['onlytesting'] == False:
        # 构建环境输入参数
        kwargs = {
            'traj_type': traj_type_target_train, # 每条轨迹的取样比例
            'trajdata_range': trajdata_range, # 使用所有轨迹的比例
            'triptype': triptype
        }
        running_date = config['training_settings']['running_date']
        training_stepnum = config['training_settings']['training_stepnum']
        envmod = config['env_para']['envmod']
        learning_rate = config['training_settings']['learning_rate']
        posnum = config['env_para']['postraj_num']
        ent_coef = config['model_para']['ent_coef']
        n_steps = config['model_para']['n_steps']
        batch_size = config['model_para']['batch_size']
        n_epochs = config['model_para']['n_epochs']
        vf_coef = config['model_para']['vf_coef']
        reward_setting = config['env_para']['reward_setting']
        RL_reset_step = config['env_para']['RL_reset_step']
        scale_state_pred = config['env_para']['continuous_scale_state_pred']
        scale_policy_cov = config['env_para']['continuous_scale_policy_cov']
        initial_RLcov = config['env_para']['initial_RLcov']

        if envmod == 'InHiGNSSSatCov_PreCovPos':
            env = DummyVecEnv([lambda: Continuous_PrePosCov_InHiGNSSSatCov(config, **kwargs)])
            encoder = CustomATF1_GnssHisInnStateCov

        tensorboard_log = f'{dir_path}records_values/{running_date}/{triptype}_{traj_type_target_train[1]}_{ratio}_{reward_setting}_{envmod}/' \
                          f'lr={learning_rate:.4f}_pos={posnum}_SP={scale_state_pred:.4f}_PC={scale_policy_cov:.4f}_IR={initial_RLcov:2f}'

        obs = env.reset()
        policy_dim = 0
        for key, value in obs.items():
            policy_dim += obs[key].shape[-1]
        policy_kwargs = dict(features_extractor_class=encoder,  features_extractor_kwargs=dict(features_dim=policy_dim),
                             ATF_trig=config['model_para']['networkmod'], net_arch=config['model_para']['net_archppo'])

        model = RecurrentPPO(
            "MlpLstmPolicy", env, verbose=2, policy_kwargs=policy_kwargs, tensorboard_log=tensorboard_log, learning_rate=learning_rate, ent_coef=ent_coef,
            n_steps=n_steps, batch_size=batch_size, n_epochs=n_epochs, vf_coef=vf_coef)
        model.learn(total_timesteps=training_stepnum, eval_log_path=tensorboard_log)

        #print and save training results
        logdirname=model.logger.dir+'/train_'
        # logdirname='./'
        print('Training finished.')

        #record model
        # params=model.get_parameters()
        model.save(model.logger.dir+f"/Save_model_{config['env_para']['reward_setting']}_trainingnum{training_stepnum:0.1e}_lr{learning_rate:0.1e}")
        recording_results_vel_RL4KF(dir_path, data_truth_dic, trajdata_range, tripIDlist, logdirname,config["env_para"]["baseline_mod"],
                                     traj_record=config["training_settings"]["traj_record"])
        recording_results_att_RL4KF(dir_path, data_truth_dic, trajdata_range, tripIDlist, logdirname,config["env_para"]["baseline_mod"],
                                     traj_record=config["training_settings"]["traj_record"])
        recording_results_ecef_RL4KF(dir_path, data_truth_dic,trajdata_range,tripIDlist,logdirname,config["env_para"]["baseline_mod"],
                                     traj_record=config["training_settings"]["traj_record"])
        # 保存参数表
        with open(f'{logdirname}config.yaml', 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True, sort_keys=False)

        # 进行模型测试
        for testtype in moreteststypelist:
            print(f'more test for {testtype} env begin here')
            tripIDlist_test = trip_type_maping[testtype]
            tripIDlist_test = [item for item in tripIDlist_test if item not in exclude_traj]

            if config['training_settings']['alltraj']:
                more_test_trajrange = [0, len(tripIDlist_test) - 1] # 所有测试轨迹进入测试
            else:
                more_test_trajrange = [int(np.ceil(len(tripIDlist_test)*ratio))+1, len(tripIDlist_test) - 1] # 后ratio半的测试轨迹进入测试

            if testtype == triptype:
                traj_type = traj_type_target_test  # 独立同分布测试
            else:
                traj_type = [0, 1]  # 域外分布测试范围

            test_trajlist = range(more_test_trajrange[0],more_test_trajrange[-1]+1) # range(more_test_trajrange[0],more_test_trajrange[-1]+1)#[0,1,2,3,4,5]
            for test_traj in test_trajlist:
                test_trajdata_range = [test_traj, test_traj]
                test_kwargs = {
                    'traj_type': traj_type,  # 每条轨迹的取样比例
                    'trajdata_range': test_trajdata_range,  # 使用所有轨迹的比例
                    'triptype': testtype,
                    'finetune': flag_finetune
                }

                if envmod == 'InHiGNSSSta_PreCovSta':
                    scale_state_pred = config['env_para']['continuous_scale_state_pred']
                    scale_policy_cov = config['env_para']['continuous_scale_policy_cov']
                    env = DummyVecEnv([lambda: Continuous_PreStaCov_InHiGNSSStaCov(config, **test_kwargs)])

                elif envmod == 'InHiGNSSPos_PreCovPos':
                    scale_state_pred = config['env_para']['continuous_scale_state_pred']
                    scale_policy_cov = config['env_para']['continuous_scale_policy_cov']
                    env = DummyVecEnv([lambda: Continuous_PrePosCov_InHiGNSSPosCov(config, **test_kwargs)])

                elif envmod == 'InHiGNSSPosAtt_PreCovPos':
                    scale_state_pred = config['env_para']['continuous_scale_state_pred']
                    scale_policy_cov = config['env_para']['continuous_scale_policy_cov']
                    env = DummyVecEnv([lambda: Continuous_PrePosAttCov_InHiGNSSPosCov(config, **test_kwargs)])

                elif envmod == 'InHiGNSSStaCov_PrePosAttCov':
                    scale_state_pred = config['env_para']['continuous_scale_state_pred']
                    scale_policy_cov = config['env_para']['continuous_scale_policy_cov']
                    env = DummyVecEnv([lambda: Continuous_PrePosAttCov_InHiGNSSStaCov(config, **test_kwargs)])

                elif envmod == 'InHiGNSSPos_PreCovPos_RTK':
                    scale_state_pred = config['env_para']['continuous_scale_state_pred']
                    scale_policy_cov = config['env_para']['continuous_scale_policy_cov']
                    env = DummyVecEnv([lambda: Continuous_PrePosCov_InHiGNSSPosCov_RTK(config, **test_kwargs)])

                obs = env.reset()
                maxiter = 100000
                for iter in range(maxiter):
                    if iter == 0:  # reset state for a perid of iterations
                        action, _states = model.predict(obs, deterministic=True)
                    else:
                        action, _states = model.predict(obs, deterministic=True, state=_states)
                    obs, rewards, done, info = env.step(action)
                    tmp = info[0]['tripIDnum']
                    if iter <= 1 or iter % 50 == 0:
                        # print(f'Iter {:.1f} reward is {:.2e}'.format(iter, rewards))
                        print(f'Iter {iter}, traj {tmp} reward is {rewards}')
                    elif done:
                        print(f'Iter {iter}, traj {tmp} reward is {rewards}, done')
                        break

            logdirname=model.logger.dir + f'/testmore_{testtype}_'
            recording_results_vel_RL4KF(dir_path, data_truth_dic,[test_trajlist[0],test_trajlist[-1]],tripIDlist_test,logdirname,
                                         config["env_para"]["baseline_mod"],traj_record=False)
            recording_results_ecef_RL4KF(dir_path, data_truth_dic,[test_trajlist[0],test_trajlist[-1]],tripIDlist_test,logdirname,
                                         config["env_para"]["baseline_mod"],traj_record=config["training_settings"]["traj_record"])
            recording_results_att_RL4KF(dir_path, data_truth_dic,[test_trajlist[0],test_trajlist[-1]],tripIDlist_test,logdirname,
                                         config["env_para"]["baseline_mod"],traj_record=False)

        print('More Test for different phonetype finished.')

    elif config["testing_settings"]['onlytesting']:
        for model_folder in model_folderlist:
            if config["testing_settings"]['Twoagent_testing']:
                break
            model_sepfolderlist=os.listdir(f'{model_basefolder}/{model_folder}') # PPO_1
            model_sepfolderlist.sort()

            for model_sepfolder in model_sepfolderlist:
                process_trig = False
                if ('csv' not in model_sepfolder) and ('txt' not in model_sepfolder):
                    model_filelist=os.listdir(f'{model_basefolder}/{model_folder}/{model_sepfolder}')
                    model_filelist.sort()
                    for model_file in model_filelist:
                        if 'Save_model' in model_file:
                            model_filename=model_file
                            process_trig = True
                        elif 'yaml' in model_file:
                            conf_name = model_file

                # 配置环境参数
                try:
                    filename = os.path.abspath(f'{model_basefolder}/{model_folder}/{model_sepfolder}/{conf_name}')
                    with open(filename, 'r', encoding='utf-8') as f:
                        config = yaml.load(f, Loader=yaml.FullLoader)  # 从YAML文件(filename)中加载配置数据
                except Exception as e:
                    print(f"Error details: {str(e)}")
                    continue
                    # raise Exception(
                    #     "Failed to read configuration file. Please check the path and format of the configuration file!")

                # 导入模型和环境
                if process_trig:
                    model_loggerdir=f'{model_basefolder}/{model_folder}/{model_sepfolder}'
                    t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                    print(f'{model_loggerdir}, {t}')
                    model_filenamepath=f'{model_loggerdir}/{model_filename}'
                    # 基本参数
                    kwargs = {
                        'traj_type': traj_type_target_train,  # 每条轨迹的取样比例
                        'trajdata_range': trajdata_range,  # 使用所有轨迹的比例
                        'triptype': triptype
                    }
                    envmod = config['env_para']['envmod']

                    if envmod == 'InHiGNSSPos_PreCovPos':
                        env = DummyVecEnv([lambda: Continuous_PrePosCov_InHiGNSSPosCov(config, **kwargs)])
                        encoder = CustomATF1_GnssHisInnStateCov

                    elif envmod == 'InHiGNSSPosAtt_PreCovPos':
                        env = DummyVecEnv([lambda: Continuous_PrePosAttCov_InHiGNSSPosCov(config, **kwargs)])
                        encoder = CustomATF1_GnssHisInnStateCov

                    obs = env.reset()
                    policy_dim = 0
                    for key, value in obs.items():
                        policy_dim += obs[key].shape[-1]
                    policy_kwargs = dict(features_extractor_class=encoder,features_extractor_kwargs=dict(features_dim=policy_dim),
                                         ATF_trig=config['model_para']['networkmod'],net_arch=config['model_para']['net_archppo'])

                    model = RecurrentPPO("MlpLstmPolicy", env, policy_kwargs=policy_kwargs)
                    model.policy.features_extractor.attention1.attwts.weight = torch.nn.Parameter(model.policy.features_extractor.attention1.attwts.weight.squeeze())
                    model.policy.features_extractor.attention2.attwts.weight = torch.nn.Parameter(model.policy.features_extractor.attention2.attwts.weight.squeeze())
                    model.policy.features_extractor.attention3.attwts.weight = torch.nn.Parameter(model.policy.features_extractor.attention3.attwts.weight.squeeze())
                    model.policy.features_extractor.attention4.attwts.weight = torch.nn.Parameter(model.policy.features_extractor.attention4.attwts.weight.squeeze())
                    model.policy.features_extractor.attention5.attwts.weight = torch.nn.Parameter(model.policy.features_extractor.attention5.attwts.weight.squeeze())

                    model.load(model_filenamepath,env=env)

                    # 进入测试环境
                    for testtype in moreteststypelist:
                        print(f'more test for {testtype} env begin here')
                        tripIDlist_test = trip_type_maping[testtype]
                        tripIDlist_test = [item for item in tripIDlist_test if item not in exclude_traj]

                        if config['training_settings']['alltraj']:
                            more_test_trajrange = [0, len(tripIDlist_test) - 1]  # 所有测试轨迹进入测试
                        else:
                            more_test_trajrange = [int(np.ceil(len(tripIDlist_test) * ratio)) + 1,
                                                   len(tripIDlist_test) - 1]  # 后ratio半的测试轨迹进入测试

                        if testtype == triptype:
                            traj_type = traj_type_target_test  # 独立同分布测试
                        else:
                            traj_type = [0, 1]  # 域外分布测试范围

                        test_trajlist = range(more_test_trajrange[0],more_test_trajrange[-1]+1)  # range(more_test_trajrange[0],more_test_trajrange[-1]+1)#[0,1,2,3,4,5]
                        for test_traj in test_trajlist:
                            test_trajdata_range = [test_traj, test_traj]
                            test_kwargs = {
                                'traj_type': traj_type,  # 每条轨迹的取样比例
                                'trajdata_range': test_trajdata_range,  # 使用所有轨迹的比例
                                'triptype': testtype
                            }

                            if envmod == 'InHiGNSSSta_PreCovSta':
                                env = DummyVecEnv([lambda: Continuous_PreStaCov_InHiGNSSStaCov(config, **test_kwargs)])
                            elif envmod == 'InHiGNSSPos_PreCovPos':
                                env = DummyVecEnv([lambda: Continuous_PrePosCov_InHiGNSSPosCov(config, **test_kwargs)])
                            elif envmod == 'InHiGNSSPosAtt_PreCovPos':
                                env = DummyVecEnv([lambda: Continuous_PrePosAttCov_InHiGNSSPosCov(config, **test_kwargs)])
                            elif envmod == 'InHiGNSSStaCov_PrePosAttCov':
                                env = DummyVecEnv([lambda: Continuous_PrePosAttCov_InHiGNSSStaCov(config, **test_kwargs)])

                            obs = env.reset()
                            maxiter = 100000
                            for iter in range(maxiter):
                                if iter == 0:  # reset state for a perid of iterations
                                    action, _states = model.predict(obs, deterministic=True)
                                else:
                                    action, _states = model.predict(obs, deterministic=True, state=_states)
                                obs, rewards, done, info = env.step(action)
                                tmp = info[0]['tripIDnum']
                                if iter <= 1 or iter % 50 == 0:
                                    # print(f'Iter {:.1f} reward is {:.2e}'.format(iter, rewards))
                                    print(f'Iter {iter}, traj {tmp} reward is {rewards}')
                                elif done:
                                    print(f'Iter {iter}, traj {tmp} reward is {rewards}, done')
                                    break

                        logdirname = model_loggerdir + f'/testmore_{testtype}_'

                        recording_results_vel_RL4KF(dir_path, data_truth_dic,[test_trajlist[0],test_trajlist[-1]],tripIDlist_test,logdirname,
                                                     config["env_para"]["baseline_mod"],traj_record=False)
                        recording_results_ecef_RL4KF(dir_path, data_truth_dic,[test_trajlist[0],test_trajlist[-1]],tripIDlist_test,logdirname,
                                                     config["env_para"]["baseline_mod"],traj_record=config["training_settings"]["traj_record"])
                        recording_results_att_RL4KF(dir_path, data_truth_dic,[test_trajlist[0],test_trajlist[-1]],tripIDlist_test,logdirname,
                                                     config["env_para"]["baseline_mod"],traj_record=False)

        print('only test finish!')

    if Twoagent_testing:
        R_model_sepfolderlist=os.listdir(f'{R_model_basefolder}/{R_model_folder}') # PPO_1
        R_model_sepfolderlist.sort()
        Q_model_sepfolderlist=os.listdir(f'{Q_model_basefolder}/{Q_model_folder}') # PPO_1
        Q_model_sepfolderlist.sort()
        # model_sepfolderlist=['RecurrentPPO_2']
        if f'lr=' in R_model_folder and f'{posnum_test}' in R_model_folder:
                continuous_Xaction_scale = float(R_model_folder.split('XAS=')[1].split('_RMSEadv_re')[0])
                continuous_Vaction_scale = float(Q_model_folder.split('VAS=')[1].split('_RMSEadv_re')[0])
                QS = float(Q_model_folder.split('_QS=')[1].split('_VAS')[0])
                RS = float(R_model_folder.split('_RS=')[1].split('_XAS')[0])
                noise_scale_dic = {'process': QS,'measurement': RS}
        else:
            print('R_model_folder error')

        QR_model_logger = f'{QR_model_basefolder}/lr={learning_rate_list[0]}_pos={posnum_test}_QS={QS}_RS={RS}_XAS={continuous_Xaction_scale}_VAS={continuous_Vaction_scale}_twoagent'
        if not os.path.exists(QR_model_logger):
            os.makedirs(QR_model_logger)

        for R_model_sepfolder in R_model_sepfolderlist:
            for Q_model_sepfolder in Q_model_sepfolderlist:
                R_process_trig = False
                Q_process_trig = False
                if ('csv' not in R_model_sepfolder) and ('txt' not in R_model_sepfolder):
                    R_model_filelist=os.listdir(f'{R_model_basefolder}/{R_model_folder}/{R_model_sepfolder}')
                    R_model_filelist.sort()
                    for R_model_file in R_model_filelist:
                        if networkmod in R_model_file:
                            R_model_filename=R_model_file
                            R_process_trig = True
                            break
                        else:
                            R_process_trig = False

                if ('csv' not in Q_model_sepfolder) and ('txt' not in Q_model_sepfolder):
                    Q_model_filelist=os.listdir(f'{Q_model_basefolder}/{Q_model_folder}/{Q_model_sepfolder}')
                    Q_model_filelist.sort()
                    for Q_model_file in Q_model_filelist:
                        if networkmod in Q_model_file:
                            Q_model_filename = Q_model_file
                            Q_process_trig = True
                            break
                        else:
                            Q_process_trig = False

                if R_process_trig and Q_process_trig:
                    R_model_loggerdir = f'{R_model_basefolder}/{R_model_folder}/{R_model_sepfolder}'
                    t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                    print(f'{R_model_loggerdir}, {t}')
                    R_model_filenamepath=f'{R_model_loggerdir}/{R_model_filename}'

                    Q_model_loggerdir = f'{Q_model_basefolder}/{Q_model_folder}/{Q_model_sepfolder}'
                    t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                    print(f'{Q_model_loggerdir}, {t}')
                    Q_model_filenamepath=f'{Q_model_loggerdir}/{Q_model_filename}'

                    if networkmod in {'continuous_lstmATF1'}:
                        env = DummyVecEnv([lambda: GPSPosition_continuous_lospos_convQR_onlyRNallcorrect(trajdata_range,traj_type_target_train,triptype,continuous_action_scale, continuous_actionspace,
                                           reward_setting,trajdata_sort, baseline_mod,posnum_test,noise_scale_dic,conv_corr,allcorrect=allcorrect)])
                        obs = env.reset()
                        features_dim_gnss = obs['gnss'].shape[-1]
                        features_dim_pos = obs['pos'].shape[-1]
                        features_dim_R = obs['R_noise'].shape[-1]
                        features_dim_ino = obs['innovation'].shape[-1]

                        encoder = CustomATF1_AKFRL_losposcovRinnovation  # innovation
                        policy_dim = features_dim_gnss + features_dim_pos + features_dim_R + features_dim_ino
                        net_arch = [network_unit, network_unit]
                        policy_kwargs = dict(features_extractor_class=encoder,features_extractor_kwargs=dict(features_dim=policy_dim),ATF_trig=networkmod, net_arch=net_arch)
                        R_model = RecurrentPPO("MlpLstmPolicy", env, policy_kwargs=policy_kwargs)
                        R_model.policy.features_extractor.attention1.attwts.weight = torch.nn.Parameter(R_model.policy.features_extractor.attention1.attwts.weight.squeeze())
                        R_model.policy.features_extractor.attention2.attwts.weight = torch.nn.Parameter(R_model.policy.features_extractor.attention2.attwts.weight.squeeze())
                        R_model.policy.features_extractor.attention3.attwts.weight = torch.nn.Parameter(R_model.policy.features_extractor.attention3.attwts.weight.squeeze())
                        R_model.policy.features_extractor.attention4.attwts.weight = torch.nn.Parameter(R_model.policy.features_extractor.attention4.attwts.weight.squeeze())
                        R_model.load(R_model_filenamepath, env=env)

                        env = DummyVecEnv([lambda: GPSPosition_continuous_lospos_convQR_onlyQNallcorrect(trajdata_range,traj_type_target_train,triptype,continuous_Vaction_scale,continuous_actionspace,
                                                reward_setting,trajdata_sort,baseline_mod,posnum_test,noise_scale_dic,conv_corr,allcorrect=allcorrect)])
                        obs = env.reset()
                        features_dim_gnss = obs['gnss'].shape[-1]
                        features_dim_pos = obs['pos'].shape[-1]
                        features_dim_Q = obs['Q_noise'].shape[-1]
                        features_dim_ino = obs['innovation'].shape[-1]
                        encoder = CustomATF1_AKFRL_poscovQinnovation
                        policy_dim = features_dim_pos + features_dim_Q + features_dim_ino
                        net_arch = [network_unit, network_unit]
                        policy_kwargs = dict(features_extractor_class=encoder,features_extractor_kwargs=dict(features_dim=policy_dim),ATF_trig=networkmod, net_arch=net_arch)
                        Q_model = RecurrentPPO("MlpLstmPolicy", env, policy_kwargs=policy_kwargs)
                        Q_model.policy.features_extractor.attention1.attwts.weight = torch.nn.Parameter(Q_model.policy.features_extractor.attention1.attwts.weight.squeeze())
                        Q_model.policy.features_extractor.attention2.attwts.weight = torch.nn.Parameter(Q_model.policy.features_extractor.attention2.attwts.weight.squeeze())
                        Q_model.policy.features_extractor.attention3.attwts.weight = torch.nn.Parameter(Q_model.policy.features_extractor.attention3.attwts.weight.squeeze())
                        Q_model.load(Q_model_filenamepath,env=env)

                    # more tests
                    if moretests:
                        for testtype in moreteststypelist:
                            print(f'more test for {testtype} env begin here')
                            if testtype == 'highway':
                                tripIDlist_test = traj_highway
                            elif testtype == 'urban':
                                tripIDlist_test = traj_urban
                            elif testtype == 'xiaomiurban':
                                tripIDlist_test = traj_xiaomiurban
                            elif testtype == 'pixelurban':
                                tripIDlist_test = traj_pixelurban
                            elif testtype == 'pixel567urban':
                                tripIDlist_test = traj_pixel567urban
                            elif testtype == 'smurban':
                                tripIDlist_test = traj_smurban

                            if alltraj:
                                more_test_trajrange = [0, len(tripIDlist) - 1]
                            else:
                                more_test_trajrange = [int(np.ceil(len(tripIDlist_test) * ratio)) + 1,len(tripIDlist_test) - 1]

                            if testtype == triptype:
                                traj_type = traj_type_target_test  # 独立同分布测试
                            else:
                                traj_type = [0, 1]  # 域外分布测试范围

                            test_trajlist = range(more_test_trajrange[0], more_test_trajrange[-1] + 1)  # [0,1,2,3,4,5]
                            for test_traj in test_trajlist:
                                test_trajdata_range = [test_traj, test_traj]
                                if networkmod in continuous_lists:
                                    env = DummyVecEnv([lambda: GPSPosition_continuous_lospos_convQR_QRNallcorrect(test_trajdata_range, traj_type, testtype, continuous_Xaction_scale,
                                        continuous_Vaction_scale,continuous_actionspace, reward_setting, trajdata_sort, baseline_mod, posnum_test,
                                        noise_scale_dic, conv_corr, interrupt_dic=interrupt_dic, allcorrect=allcorrect)])

                                obs = env.reset()
                                maxiter = 100000
                                for iter in range(maxiter):
                                    if iter  == 0:  # reset state for a perid of iterations % 10
                                        R_action, _Rstates = R_model.predict(obs, deterministic=True)
                                        Q_action, _Qstates = Q_model.predict(obs, deterministic=True)
                                    else:
                                        R_action, _Rstates = R_model.predict(obs, deterministic=True, state=_Rstates)
                                        Q_action, _Qstates = Q_model.predict(obs, deterministic=True, state=_Qstates)
                                    action = np.concatenate((Q_action,R_action), axis=1)
                                    obs, rewards, done, info = env.step(action)
                                    tmp = info[0]['tripIDnum']
                                    if iter <= 1 or iter % 100 == 0:
                                        # print(f'Iter {:.1f} reward is {:.2e}'.format(iter, rewards))
                                        print(f'Iter {iter}, traj {tmp} reward is {rewards}')
                                    elif rewards == 0:
                                        # print(f'Iter {:.1f} reward is {:.2e}'.format(iter, rewards))
                                        print(f'Iter {iter}, traj {tmp} reward is {rewards}')
                                    elif done:
                                        print(f'Iter {iter}, traj {tmp} reward is {rewards}, done')
                                        break

                            model_loggerdir = QR_model_logger + f'/R_{R_model_sepfolder}_Q_{Q_model_sepfolder}'
                            if not os.path.exists(model_loggerdir):
                                os.makedirs(model_loggerdir)
                            logdirname = model_loggerdir + f'/testmore_{testtype}_'
                            if interrupt_dic is not None:
                                inter_time = interrupt_dic['time']
                                logdirname = model_loggerdir + f'/testmore_{testtype}_interrupt{inter_time}_'
                            recording_results_ecef_RL4KF(dir_path, data_truth_dic, [test_trajlist[0], test_trajlist[-1]],tripIDlist_test, logdirname, baseline_mod, traj_record=True)

        print('only test finish!')