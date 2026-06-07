import numpy as np
import pandas as pd
import os
import sys
cur_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(cur_dir, '..')
sys.path.append(src_dir)
from stable_baselines3.common.utils import get_schedule_fn
from env.Urbannav_GNSSINS_KRLF_boost import * # RL环境
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C
from model.ppo_recurrent_D_agent import RecurrentPPO
from model.ppo_recurrent_ATF1_AKF_FT import RecurrentPPO_FT # 微调模型
from funcs_rl.utilis_eskf import *
from model.model_ATF_KF import *
import time
import argparse
import yaml

"""
双智能体联合训练
"""

if __name__ == "__main__":
    # 可选环境：'HK_Me_ublox_Dagent'
    triptype = 'HK_De_ublox_Dagent'
    print(f"-----Env mode: {triptype}-------")
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
    if config["testing_settings"]['onlytesting'] or config["finetuning_settings"]['finetuning']:
        testdate = config["testing_settings"]['testdate']
        model_basefolder = config["testing_settings"]['model_basefolder']
        model_basefolder=f'{dir_path}/records_values/{testdate}/{model_basefolder}'
        if config["testing_settings"]['model_name_list']:
            model_folderlist = config["testing_settings"]['model_name_list']  # only for testing
        else:
            model_folderlist=os.listdir(model_basefolder)
            model_folderlist.sort()

    # 构建训练环境输入参数
    kwargs = {
        'traj_type': traj_type_target_train,  # 每条轨迹的取样比例
        'trajdata_range': trajdata_range,  # 使用所有轨迹的比例
        'triptype': triptype}

    # 开启训练模式
    if (config["testing_settings"]['onlytesting'] == False) and (config["finetuning_settings"]['finetuning'] == False):
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
        scale_att_pred = config['env_para']['continuous_scale_att_pred']
        scale_state_correct = config['env_para']['continuous_scale_state_correct']
        scale_att_correct = config['env_para']['continuous_scale_att_correct']
        scale_policy_cov = config['env_para']['continuous_scale_policy_cov']
        initial_RLcov = config['env_para']['initial_RLcov']

        if envmod == 'InHiGNSSSatCov_PreCovPos-PosAttCorrectCov':
            env = DummyVecEnv([lambda: DA_PrePosCov_PosAttCorrectCov_InHiGNSSSatCov(config, **kwargs)])
            a1_encoder = CustomATF1_GnssHisInnStateCov
            a2_encoder = CustomATF1_GnssHisInnPosAttP

        tensorboard_log = f'{dir_path}records_values/{running_date}/{triptype}_{traj_type_target_train[1]}_{ratio}_{reward_setting}_{envmod}/' \
                          f'lr={learning_rate:.4f}_pos={posnum}_SP={scale_state_pred:.2f}_SC={scale_state_correct:.2f}_' \
                          f'AP={scale_att_pred:.4f}_AC={scale_att_correct:.4f}'

        obs = env.reset()
        features_dim_gnss = obs['gnss'].shape[-1]
        features_dim_inn = obs['innovation'].shape[-1]
        features_dim_his = obs['History'].shape[-1]
        features_dim_sta = obs['State'].shape[-1]
        features_dim_cov = obs['Cov'].shape[-1]
        features_dim_P = obs['KF_P'].shape[-1]
        features_dim_ico = obs['innova_cor'].shape[-1]
        dim_A1 = features_dim_gnss + features_dim_inn + features_dim_his + features_dim_sta + features_dim_cov
        dim_A2 = features_dim_gnss + features_dim_ico + features_dim_his + features_dim_sta + features_dim_P

        policy_kwargs_A1 = dict(features_extractor_class=a1_encoder,  features_extractor_kwargs=dict(features_dim=dim_A1),
                             ATF_trig=config['model_para']['networkmod'], net_arch=config['model_para']['net_archppo'])
        policy_kwargs_A2 = dict(features_extractor_class=a2_encoder,  features_extractor_kwargs=dict(features_dim=dim_A2),
                             ATF_trig=config['model_para']['networkmod'], net_arch=config['model_para']['net_archppo'])
        policy_kwargs_dic = {'A1_policy': policy_kwargs_A1, 'A2_policy': policy_kwargs_A2}

        model = RecurrentPPO(
            "MlpLstmPolicy", env, verbose=2, policy_kwargs_dic=policy_kwargs_dic, tensorboard_log=tensorboard_log, learning_rate=learning_rate, ent_coef=ent_coef,
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

                if envmod == 'InHiGNSSSatCov_PreCovPos-PosAttCorrectCov':
                    env = DummyVecEnv([lambda: DA_PrePosCov_PosAttCorrectCov_InHiGNSSSatCov(config, **kwargs)])

                obs = env.reset()
                maxiter = 100000
                for iter in range(maxiter):
                    if iter == 0:  # reset state for a perid of iterations
                        action, _state_A1, _state_A2 = model.predict(obs, deterministic=True)
                    else:
                        action, _state_A1, _state_A2 = model.predict(obs, deterministic=True, state_A1=_state_A1,state_A2=_state_A2)
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
        print(f"----------Only testing for {triptype} setting---------")
        for model_folder in model_folderlist:
            if config["finetuning_settings"]['finetuning']:
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

                    envmod = config['env_para']['envmod']

                    if envmod == 'InHiGNSSSatCov_PreCovPos-PosAttCorrectCov':
                        env = DummyVecEnv([lambda: DA_PrePosCov_PosAttCorrectCov_InHiGNSSSatCov(config, **kwargs)])
                        a1_encoder = CustomATF1_GnssHisInnStateCov
                        a2_encoder = CustomATF1_GnssHisInnPosAttP

                    obs = env.reset()
                    features_dim_gnss = obs['gnss'].shape[-1]
                    features_dim_inn = obs['innovation'].shape[-1]
                    features_dim_his = obs['History'].shape[-1]
                    features_dim_sta = obs['State'].shape[-1]
                    features_dim_cov = obs['Cov'].shape[-1]
                    features_dim_P = obs['KF_P'].shape[-1]
                    features_dim_ico = obs['innova_cor'].shape[-1]
                    dim_A1 = features_dim_gnss + features_dim_inn + features_dim_his + features_dim_sta + features_dim_cov
                    dim_A2 = features_dim_gnss + features_dim_ico + features_dim_his + features_dim_sta + features_dim_P

                    policy_kwargs_A1 = dict(features_extractor_class=a1_encoder,features_extractor_kwargs=dict(features_dim=dim_A1),
                                            ATF_trig=config['model_para']['networkmod'],net_arch=config['model_para']['net_archppo'])
                    policy_kwargs_A2 = dict(features_extractor_class=a2_encoder, features_extractor_kwargs=dict(features_dim=dim_A2),
                                            ATF_trig=config['model_para']['networkmod'],net_arch=config['model_para']['net_archppo'])
                    policy_kwargs_dic = {'A1_policy': policy_kwargs_A1, 'A2_policy': policy_kwargs_A2}

                    model = RecurrentPPO("MlpLstmPolicy", env, policy_kwargs_dic=policy_kwargs_dic)
                    model.policy_Q.features_extractor.attention1.attwts.weight = torch.nn.Parameter(model.policy_Q.features_extractor.attention1.attwts.weight.squeeze())
                    model.policy_Q.features_extractor.attention2.attwts.weight = torch.nn.Parameter(model.policy_Q.features_extractor.attention2.attwts.weight.squeeze())
                    model.policy_Q.features_extractor.attention3.attwts.weight = torch.nn.Parameter(model.policy_Q.features_extractor.attention3.attwts.weight.squeeze())
                    model.policy_Q.features_extractor.attention4.attwts.weight = torch.nn.Parameter(model.policy_Q.features_extractor.attention4.attwts.weight.squeeze())
                    model.policy_Q.features_extractor.attention5.attwts.weight = torch.nn.Parameter(model.policy_Q.features_extractor.attention5.attwts.weight.squeeze())
                    model.policy_R.features_extractor.attention1.attwts.weight = torch.nn.Parameter(model.policy_R.features_extractor.attention1.attwts.weight.squeeze())
                    model.policy_R.features_extractor.attention2.attwts.weight = torch.nn.Parameter(model.policy_R.features_extractor.attention2.attwts.weight.squeeze())
                    model.policy_R.features_extractor.attention3.attwts.weight = torch.nn.Parameter(model.policy_R.features_extractor.attention3.attwts.weight.squeeze())
                    model.policy_R.features_extractor.attention4.attwts.weight = torch.nn.Parameter(model.policy_R.features_extractor.attention4.attwts.weight.squeeze())
                    model.policy_R.features_extractor.attention5.attwts.weight = torch.nn.Parameter(model.policy_R.features_extractor.attention5.attwts.weight.squeeze())

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

                            if envmod == 'InHiGNSSSatCov_PreCovPos-PosAttCorrectCov':
                                env = DummyVecEnv(
                                    [lambda: DA_PrePosCov_PosAttCorrectCov_InHiGNSSSatCov(config, **test_kwargs)])

                            obs = env.reset()
                            maxiter = 100000
                            for iter in range(maxiter):
                                if iter == 0:  # reset state for a perid of iterations
                                    action, _state_A1, _state_A2 = model.predict(obs, deterministic=True)
                                else:
                                    action, _state_A1, _state_A2 = model.predict(obs, deterministic=True,state_A1=_state_A1, state_A2=_state_A2)
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
                                                     config["env_para"]["baseline_mod"],traj_record=True)
                        recording_results_att_RL4KF(dir_path, data_truth_dic,[test_trajlist[0],test_trajlist[-1]],tripIDlist_test,logdirname,
                                                     config["env_para"]["baseline_mod"],traj_record=False)

        print('only test finish!')

    elif config["finetuning_settings"]['finetuning']:
        print(f"----------Fine tuning for {triptype} setting---------")
        for model_folder in model_folderlist:
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
                        config_train = yaml.load(f, Loader=yaml.FullLoader)  # 从YAML文件(filename)中加载配置数据
                except Exception as e:
                    print(f"Error details: {str(e)}")
                    continue
                    # raise Exception(
                    #     "Failed to read configuration file. Please check the path and format of the configuration file!")

                # 导入模型和环境
                if process_trig:
                    # 获取训练模型路径
                    model_loggerdir=f'{model_basefolder}/{model_folder}/{model_sepfolder}'
                    t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                    print(f'{model_loggerdir}, {t}')
                    model_filenamepath=f'{model_loggerdir}/{model_filename}'

                    # 获取微调模型参数
                    learning_rate = config['finetuning_settings']['learning_rate']
                    vf_coef = config['finetuning_settings']['vf_coef']
                    n_steps = config['finetuning_settings']['n_steps']
                    clip_range = config['finetuning_settings']['clip_range']
                    n_epochs = config['finetuning_settings']['n_epochs']
                    kf_coef = config['finetuning_settings']['kf_coef']
                    gae_lambda = config['finetuning_settings']['gae_lambda']

                    # 新增微调数据记录文件夹
                    loger_folder = f'/Finetune_model/FT_lr={learning_rate:.2f}_nsteps={n_steps:.2f}_clip_range={clip_range:.2f}_kf={kf_coef:.2f}'
                    result_folder = model_loggerdir + loger_folder
                    if not os.path.exists(result_folder):
                        os.makedirs(result_folder)
                    # final_log_dir = create_numbered_dir(result_folder, prefix="FT_RecurrentPPO_")  # 重复实验文件夹
                    # final_log_dir = result_folder + '/FT_RecurrentPPO'

                    # 重导入构建训练环境
                    envmod = config_train['env_para']['envmod']
                    kwargs['finetuning'] = True
                    kwargs['config_tuning'] = config # 补充微调参数
                    if envmod == 'InHiGNSSSatCov_PreCovPos-PosAttCorrectCov':
                        env = DummyVecEnv([lambda: DA_PrePosCov_PosAttCorrectCov_InHiGNSSSatCov(config_train, **kwargs)])
                        a1_encoder = CustomATF1_GnssHisInnStateCov
                        a2_encoder = CustomATF1_GnssHisInnPosAttP

                    obs = env.reset()

                    print("-----加载预训练模型-----")
                    features_dim_gnss = obs['gnss'].shape[-1]
                    features_dim_inn = obs['innovation'].shape[-1]
                    features_dim_his = obs['History'].shape[-1]
                    features_dim_sta = obs['State'].shape[-1]
                    features_dim_cov = obs['Cov'].shape[-1]
                    features_dim_P = obs['KF_P'].shape[-1]
                    features_dim_ico = obs['innova_cor'].shape[-1]
                    dim_A1 = features_dim_gnss + features_dim_inn + features_dim_his + features_dim_sta + features_dim_cov
                    dim_A2 = features_dim_gnss + features_dim_ico + features_dim_his + features_dim_sta + features_dim_P

                    policy_kwargs_A1 = dict(features_extractor_class=a1_encoder,features_extractor_kwargs=dict(features_dim=dim_A1),
                                            ATF_trig=config_train['model_para']['networkmod'],net_arch=config_train['model_para']['net_archppo'])
                    policy_kwargs_A2 = dict(features_extractor_class=a2_encoder, features_extractor_kwargs=dict(features_dim=dim_A2),
                                            ATF_trig=config_train['model_para']['networkmod'],net_arch=config_train['model_para']['net_archppo']) # 微调修改0605：增加log_std_init初始化
                    policy_kwargs_dic = {'A1_policy': policy_kwargs_A1, 'A2_policy': policy_kwargs_A2, 'finetune': True} # 设置开启微调

                    model = RecurrentPPO("MlpLstmPolicy", env, kf_coef=kf_coef, verbose=2, n_steps=n_steps,
                                         batch_size=n_steps, policy_kwargs_dic=policy_kwargs_dic, tensorboard_log=None)

                    with torch.no_grad():
                        model.policy_Q.features_extractor.attention1.attwts.weight = torch.nn.Parameter(model.policy_Q.features_extractor.attention1.attwts.weight.squeeze())
                        model.policy_Q.features_extractor.attention2.attwts.weight = torch.nn.Parameter(model.policy_Q.features_extractor.attention2.attwts.weight.squeeze())
                        model.policy_Q.features_extractor.attention3.attwts.weight = torch.nn.Parameter(model.policy_Q.features_extractor.attention3.attwts.weight.squeeze())
                        model.policy_Q.features_extractor.attention4.attwts.weight = torch.nn.Parameter(model.policy_Q.features_extractor.attention4.attwts.weight.squeeze())
                        model.policy_Q.features_extractor.attention5.attwts.weight = torch.nn.Parameter(model.policy_Q.features_extractor.attention5.attwts.weight.squeeze())
                        model.policy_R.features_extractor.attention1.attwts.weight = torch.nn.Parameter(model.policy_R.features_extractor.attention1.attwts.weight.squeeze())
                        model.policy_R.features_extractor.attention2.attwts.weight = torch.nn.Parameter(model.policy_R.features_extractor.attention2.attwts.weight.squeeze())
                        model.policy_R.features_extractor.attention3.attwts.weight = torch.nn.Parameter(model.policy_R.features_extractor.attention3.attwts.weight.squeeze())
                        model.policy_R.features_extractor.attention4.attwts.weight = torch.nn.Parameter(model.policy_R.features_extractor.attention4.attwts.weight.squeeze())
                        model.policy_R.features_extractor.attention5.attwts.weight = torch.nn.Parameter(model.policy_R.features_extractor.attention5.attwts.weight.squeeze())

                    model.load(model_filenamepath, env=env)

                    # 重置 策略 的优化器
                    model.policy_Q.optimizer = model.policy_Q.optimizer_class(
                        model.policy_Q.parameters(),lr=3e-3,
                        **model.policy_Q.optimizer_kwargs)

                    # 重置 policy_R 的优化器
                    model.policy_R.optimizer = model.policy_R.optimizer_class(
                        model.policy_R.parameters(),lr=3e-3,
                        **model.policy_R.optimizer_kwargs)

                    # 1. 遍历 Q 网络的全部子模块，寻找 LSTM/RNN 并压平
                    for module in model.policy_Q.modules():
                        if isinstance(module, torch.nn.RNNBase):  # RNNBase 是 LSTM/GRU/RNN 的父类
                            module.flatten_parameters()

                    # 2. 遍历 R 网络的全部子模块，寻找 LSTM/RNN 并压平
                    for module in model.policy_R.modules():
                        if isinstance(module, torch.nn.RNNBase):
                            module.flatten_parameters()

                    # 手动降低模型方差
                    with torch.no_grad():
                        torch.nn.init.constant_(model.policy_Q.log_std, -3.0)
                        torch.nn.init.constant_(model.policy_R.log_std, -3.0)

                    print("-----Pre-training critic model start-----")
                    model.learn(total_timesteps=500, train_critic=True)
                    print("-----Pre-training critic model end-----")

                    # 重置记录路径
                    model.tensorboard_log = result_folder

                    # 重置 策略 的优化器
                    model.policy_Q.optimizer = model.policy_Q.optimizer_class(
                        model.policy_Q.parameters(),lr=learning_rate,
                        **model.policy_Q.optimizer_kwargs)

                    # 重置 policy_R 的优化器
                    model.policy_R.optimizer = model.policy_R.optimizer_class(
                        model.policy_R.parameters(),lr=learning_rate,
                        **model.policy_R.optimizer_kwargs)

                    # 重新导入训练参数
                    model.vf_coef = vf_coef
                    model.n_steps = n_steps
                    model.n_epochs = n_epochs
                    model.clip_range_val = clip_range  # 保存个数值
                    model.clip_range = get_schedule_fn(clip_range)
                    # 同步更新策略子模块中的 clip_range (因为你的 train 函数里用的是策略里的值)
                    model.policy_Q.clip_range = model.clip_range
                    model.policy_R.clip_range = model.clip_range
                    model.gae_lambda = gae_lambda  # 推荐降到 0.90 或 0.85

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
                                'triptype': testtype,
                                'finetuning': True,
                                'config_tuning': config
                            }

                            if envmod == 'InHiGNSSSatCov_PreCovPos-PosAttCorrectCov':
                                # 这里必须导入训练环境的参数
                                env = DummyVecEnv(
                                    [lambda: DA_PrePosCov_PosAttCorrectCov_InHiGNSSSatCov(config_train, **test_kwargs)])

                            print("-----Fine tuning Policy model-----")
                            model.set_env(env)
                            model.learn(total_timesteps=20000, eval_log_path=result_folder, reset_num_timesteps=True) # 回合结束即停止

                        logdirname = model.logger.dir + f'/finetuning_{testtype}_'

                        recording_results_vel_RL4KF(dir_path, data_truth_dic,[test_trajlist[0],test_trajlist[-1]],tripIDlist_test,logdirname,
                                                     config["env_para"]["baseline_mod"],traj_record=False)
                        recording_results_ecef_RL4KF(dir_path, data_truth_dic,[test_trajlist[0],test_trajlist[-1]],tripIDlist_test,logdirname,
                                                     config["env_para"]["baseline_mod"],traj_record=True)
                        recording_results_att_RL4KF(dir_path, data_truth_dic,[test_trajlist[0],test_trajlist[-1]],tripIDlist_test,logdirname,
                                                     config["env_para"]["baseline_mod"],traj_record=False)

                        model.save(model.logger.dir + f"/Save_model_{config['env_para']['reward_setting']}_lr{learning_rate:0.1e}")

                        # 保存参数表
                        with open(f'{logdirname}config_FT.yaml', 'w', encoding='utf-8') as f:
                            yaml.dump(config, f, allow_unicode=True, sort_keys=False)

        print('only test finish!')