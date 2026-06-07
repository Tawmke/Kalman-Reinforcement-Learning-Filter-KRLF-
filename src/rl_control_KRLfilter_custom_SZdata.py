import numpy as np
import pandas as pd
import os
import sys
cur_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(cur_dir, '..')
sys.path.append(src_dir)

from env.SZdata_GNSSINS_KRLF_split import * # RL环境
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
KRLF强化学习环境用于深圳数据集，训练和测试区分四种场景
"""
if __name__ == "__main__":
    # 可选环境：'SZ_canyon_RTK' SZ_forest_RTK SZ_overpass_RTK SZ_openroad_RTK SZ_openroad_RTK_A2
    triptype_traning = 'SZ_openroad_RTK_A2'
    conf_name = f'KRLF_{triptype_traning}.yaml'
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
            model_folderlist.sort(reverse=True)

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

        elif envmod == 'InHiGNSSSatCov_POSCorrectCov':
            env = DummyVecEnv([lambda: Continuous_POSCorrectCov_InHiGNSSPosCov(config, **kwargs)])
            encoder = CustomATF1_GnssHisInnStateCov

        tensorboard_log = f'{dir_path}records_values/{running_date}/{triptype_traning}_{traj_type_target_train[1]}_{ratio}_{reward_setting}_{envmod}/' \
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
        recording_results_ecef_RL4KF_SZ(dir_path, data_truth_dic,trajdata_range,tripIDlist,logdirname,config["env_para"]["baseline_mod"],
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

                if envmod == 'InHiGNSSSatCov_PreCovPos':
                    env = DummyVecEnv([lambda: Continuous_PrePosCov_InHiGNSSSatCov(config, **test_kwargs)])
                elif envmod == 'InHiGNSSSatCov_POSCorrectCov':
                    env = DummyVecEnv([lambda: Continuous_POSCorrectCov_InHiGNSSPosCov(config, **test_kwargs)])

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
            recording_results_ecef_RL4KF_SZ(dir_path, data_truth_dic,[test_trajlist[0],test_trajlist[-1]],tripIDlist_test,logdirname,
                                         config["env_para"]["baseline_mod"],traj_record=config["training_settings"]["traj_record"])

        print('More Test for different phonetype finished.')

    elif config["testing_settings"]['onlytesting']:
        print("---------Only for testing mode-------------")
        for model_folder in model_folderlist:
            try:
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
                        # 选择环境导入
                        if envmod == 'InHiGNSSSatCov_PreCovPos':
                            env = DummyVecEnv([lambda: Continuous_PrePosCov_InHiGNSSSatCov(config, **kwargs)])
                            encoder = CustomATF1_GnssHisInnStateCov
                        elif envmod == 'InHiGNSSSatCov_POSCorrectCov':
                            env = DummyVecEnv([lambda: Continuous_POSCorrectCov_InHiGNSSPosCov(config, **kwargs)])
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
                                more_test_trajrange = [int(np.ceil(len(tripIDlist_test) * ratio)) + 1,len(tripIDlist_test) - 1]  # 后ratio半的测试轨迹进入测试

                            if testtype == triptype:
                                traj_type = [0, 1] # traj_type_target_test  # 独立同分布测试
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

                                if envmod == 'InHiGNSSSatCov_PreCovPos':
                                    env = DummyVecEnv([lambda: Continuous_PrePosCov_InHiGNSSSatCov(config, **test_kwargs)])
                                elif envmod == 'InHiGNSSSatCov_POSCorrectCov':
                                    env = DummyVecEnv([lambda: Continuous_POSCorrectCov_InHiGNSSPosCov(config, **test_kwargs)])

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

                            recording_results_ecef_RL4KF_SZ(dir_path, data_truth_dic,[test_trajlist[0],test_trajlist[-1]],tripIDlist_test,logdirname,
                                                         config["env_para"]["baseline_mod"],traj_record=config["training_settings"]["traj_record"])

            except Exception as e:
                print(f"Error details: {str(e)}")
                continue

        print('only test finish!')

    if config["testing_settings"]['Twoagent_testing']:
        # 导入双智能体模型的路径和相关参数
        test_triptype_list = config["testing_settings"]['test_triptype']
        for test_triptype in test_triptype_list:
            agent_1_basefolder = f'{dir_path}/records_values/{testdate}/{config["testing_settings"]["Dagent_file"][test_triptype]["agent_1_basefolder"]}'
            agent_2_basefolder = f'{dir_path}/records_values/{testdate}/{config["testing_settings"]["Dagent_file"][test_triptype]["agent_2_basefolder"]}'
            agent_1_folder = config["testing_settings"]["Dagent_file"][test_triptype]["agent_1_folder"]
            agent_2_folder = config["testing_settings"]["Dagent_file"][test_triptype]["agent_2_folder"]
            QR_basefolder = 'source=pixel567urban_0.7_1_losposconvQR_QRcorrect_fullobs_kf_continuous_lstmATF1'
            QR_model_basefolder = f'{dir_path}/records_values/{testdate}/{config["testing_settings"]["Dagent_file"][test_triptype]["Dual_agent_basefolder"]}'
            MA_envmode = config["testing_settings"]["MA_envmode"]
            # agent 运行次数模型导入
            agent_1_sepfolderlist=os.listdir(f'{agent_1_basefolder}/{agent_1_folder}') # PPO_1
            agent_1_sepfolderlist.sort()
            agent_2_sepfolderlist=os.listdir(f'{agent_2_basefolder}/{agent_2_folder}') # PPO_1
            agent_2_sepfolderlist.sort()
            # model_sepfolderlist=['RecurrentPPO_2']

            # 设置双智能体保存路径
            QR_model_logger = f'{QR_model_basefolder}/A1={agent_1_folder}_A2={agent_2_folder}'
            if not os.path.exists(QR_model_logger):
                os.makedirs(QR_model_logger)

            for agent_1_sepfolder in agent_1_sepfolderlist:
                for agent_2_sepfolder in agent_2_sepfolderlist:
                    agent_1_process_trig = False
                    agent_2_process_trig = False
                    if ('csv' not in agent_1_sepfolder) and ('txt' not in agent_1_sepfolder):
                        agent_1_filelist=os.listdir(f'{agent_1_basefolder}/{agent_1_folder}/{agent_1_sepfolder}')
                        agent_1_filelist.sort()
                        for agent_1_file in agent_1_filelist:
                            if 'Save_model' in agent_1_file:
                                agent_1_filename = agent_1_file
                                agent_1_process_trig = True
                            elif 'yaml' in agent_1_file:
                                conf_name_agent_1 = agent_1_file

                    if ('csv' not in agent_2_sepfolder) and ('txt' not in agent_2_sepfolder):
                        agent_2_filelist=os.listdir(f'{agent_2_basefolder}/{agent_2_folder}/{agent_2_sepfolder}')
                        agent_2_filelist.sort()
                        for agent_2_file in agent_2_filelist:
                            if 'Save_model' in agent_2_file:
                                agent_2_filename = agent_2_file
                                agent_2_process_trig = True
                            elif 'yaml' in agent_2_file:
                                conf_name_agent_2 = agent_2_file

                    # 导入config参数配置
                    try:
                        filename = os.path.abspath(f'{agent_1_basefolder}/{agent_1_folder}/{agent_1_sepfolder}/{conf_name_agent_1}')
                        with open(filename, 'r', encoding='utf-8') as f:
                            config_agent_1 = yaml.load(f, Loader=yaml.FullLoader)  # 从YAML文件(filename)中加载配置数据

                        filename = os.path.abspath(f'{agent_2_basefolder}/{agent_2_folder}/{agent_2_sepfolder}/{conf_name_agent_2}')
                        with open(filename, 'r', encoding='utf-8') as f:
                            config_agent_2 = yaml.load(f, Loader=yaml.FullLoader)  # 从YAML文件(filename)中加载配置数据
                    except Exception as e:
                        print(f"Error details: {str(e)}")
                        continue

                    if agent_1_process_trig and agent_2_process_trig:
                        agent_1_loggerdir = f'{agent_1_basefolder}/{agent_1_folder}/{agent_1_sepfolder}'
                        t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                        print(f'{agent_1_loggerdir}, {t}')
                        agent_1_filenamepath=f'{agent_1_loggerdir}/{agent_1_filename}'

                        agent_2_loggerdir = f'{agent_2_basefolder}/{agent_2_folder}/{agent_2_sepfolder}'
                        t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                        print(f'{agent_2_loggerdir}, {t}')
                        agent_2_filenamepath=f'{agent_2_loggerdir}/{agent_2_filename}'

                        kwargs = {
                            'traj_type': traj_type_target_train,  # 每条轨迹的取样比例
                            'trajdata_range': trajdata_range,  # 使用所有轨迹的比例
                            'triptype': triptype
                        }

                        env = DummyVecEnv([lambda: Continuous_PrePosCov_InHiGNSSSatCov(config_agent_1, **kwargs)])
                        obs = env.reset()
                        encoder = CustomATF1_GnssHisInnStateCov  # innovation
                        policy_dim = 0
                        for key, value in obs.items():
                            policy_dim += obs[key].shape[-1]
                        policy_kwargs = dict(features_extractor_class=encoder,features_extractor_kwargs=dict(features_dim=policy_dim),
                                       ATF_trig=config['model_para']['networkmod'], net_arch=config['model_para']['net_archppo'])
                        agent_1 = RecurrentPPO("MlpLstmPolicy", env, policy_kwargs=policy_kwargs)
                        agent_1.policy.features_extractor.attention1.attwts.weight = torch.nn.Parameter(agent_1.policy.features_extractor.attention1.attwts.weight.squeeze())
                        agent_1.policy.features_extractor.attention2.attwts.weight = torch.nn.Parameter(agent_1.policy.features_extractor.attention2.attwts.weight.squeeze())
                        agent_1.policy.features_extractor.attention3.attwts.weight = torch.nn.Parameter(agent_1.policy.features_extractor.attention3.attwts.weight.squeeze())
                        agent_1.policy.features_extractor.attention4.attwts.weight = torch.nn.Parameter(agent_1.policy.features_extractor.attention4.attwts.weight.squeeze())
                        agent_1.policy.features_extractor.attention5.attwts.weight = torch.nn.Parameter(agent_1.policy.features_extractor.attention5.attwts.weight.squeeze())
                        agent_1.load(agent_1_filenamepath, env=env)

                        env = DummyVecEnv([lambda: Continuous_POSCorrectCov_InHiGNSSPosCov(config_agent_2, **kwargs)])
                        obs = env.reset()
                        encoder = CustomATF1_GnssHisInnPosP_A2
                        policy_dim = 0
                        for key, value in obs.items():
                            policy_dim += obs[key].shape[-1]
                        policy_kwargs = dict(features_extractor_class=encoder,features_extractor_kwargs=dict(features_dim=policy_dim),
                                             ATF_trig=config['model_para']['networkmod'],net_arch=config['model_para']['net_archppo'])
                        agent_2 = RecurrentPPO("MlpLstmPolicy", env, policy_kwargs=policy_kwargs)
                        agent_2.policy.features_extractor.attention1.attwts.weight = torch.nn.Parameter(agent_2.policy.features_extractor.attention1.attwts.weight.squeeze())
                        agent_2.policy.features_extractor.attention2.attwts.weight = torch.nn.Parameter(agent_2.policy.features_extractor.attention2.attwts.weight.squeeze())
                        agent_2.policy.features_extractor.attention3.attwts.weight = torch.nn.Parameter(agent_2.policy.features_extractor.attention3.attwts.weight.squeeze())
                        agent_2.policy.features_extractor.attention4.attwts.weight = torch.nn.Parameter(agent_2.policy.features_extractor.attention4.attwts.weight.squeeze())
                        agent_2.policy.features_extractor.attention5.attwts.weight = torch.nn.Parameter(agent_2.policy.features_extractor.attention5.attwts.weight.squeeze())
                        agent_2.load(agent_2_filenamepath, env=env)

                        # more tests
                        testtype = test_triptype
                        print(f'more test for {testtype} env begin here')
                        tripIDlist_test = trip_type_maping[testtype]
                        tripIDlist_test = [item for item in tripIDlist_test if item not in exclude_traj]

                        if config['training_settings']['alltraj']:
                            more_test_trajrange = [0, len(tripIDlist_test) - 1]  # 所有测试轨迹进入测试
                        else:
                            more_test_trajrange = [int(np.ceil(len(tripIDlist_test) * ratio)) + 1,
                                                   len(tripIDlist_test) - 1]  # 后ratio半的测试轨迹进入测试

                        if testtype == triptype:
                            traj_type = [0, 1]  # traj_type_target_test  # 独立同分布测试
                        else:
                            traj_type = [0, 1]  # 域外分布测试范围

                        test_trajlist = range(more_test_trajrange[0], more_test_trajrange[-1] + 1)  # range(more_test_trajrange[0],more_test_trajrange[-1]+1)#[0,1,2,3,4,5]
                        for test_traj in test_trajlist:
                            test_trajdata_range = [test_traj, test_traj]
                            test_kwargs = {
                                'traj_type': traj_type,  # 每条轨迹的取样比例
                                'trajdata_range': test_trajdata_range,  # 使用所有轨迹的比例
                                'triptype': testtype
                            }

                            if MA_envmode == 'InHiGNSSSatCov_PreCovPos-POSCorrectCov':
                                env = DummyVecEnv([lambda: Continuous_PrePosCov_POSCorrectCov_InHiGNSSSatCov(config_agent_1, config_agent_2, **test_kwargs)])

                            agent_1.policy.observation_space = env.observation_space
                            agent_2.policy.observation_space = env.observation_space

                            obs = env.reset()
                            maxiter = 100000
                            for iter in range(maxiter):
                                if iter == 0:  # reset state for a perid of iterations
                                    a1_action, _a1states = agent_1.predict(obs, deterministic=True)
                                    a2_action, _a2states = agent_2.predict(obs, deterministic=True)
                                else:
                                    a1_action, _a1states = agent_1.predict(obs, deterministic=True, state=_a1states)
                                    a2_action, _a2states = agent_2.predict(obs, deterministic=True, state=_a2states)

                                action = np.concatenate((a1_action, a2_action), axis=1)
                                obs, rewards, done, info = env.step(action)
                                tmp = info[0]['tripIDnum']
                                if iter <= 1 or iter % 50 == 0:
                                    # print(f'Iter {:.1f} reward is {:.2e}'.format(iter, rewards))
                                    print(f'Iter {iter}, traj {tmp} reward is {rewards}')
                                elif done:
                                    print(f'Iter {iter}, traj {tmp} reward is {rewards}, done')
                                    break

                        model_loggerdir = QR_model_logger + f'/A1_{agent_1_sepfolder}_A2_{agent_2_sepfolder}'
                        if not os.path.exists(model_loggerdir):
                            os.makedirs(model_loggerdir)

                        logdirname = model_loggerdir + f'/testmore_{testtype}_'

                        recording_results_ecef_RL4KF_SZ(dir_path, data_truth_dic, [test_trajlist[0], test_trajlist[-1]],
                                                        tripIDlist_test, logdirname,config["env_para"]["baseline_mod"],
                                                        traj_record=config["training_settings"]["traj_record"])

        print('only test finish!')