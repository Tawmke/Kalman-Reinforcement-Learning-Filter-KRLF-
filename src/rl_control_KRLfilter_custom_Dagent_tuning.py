import numpy as np
import pandas as pd
import os
import sys
cur_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(cur_dir, '..')
sys.path.append(src_dir)
import multiprocessing
from stable_baselines3.common.utils import get_schedule_fn
from env.SZdata_GNSSINS_KRLF_split_boost import * # RL环境
from env.Urbannav_GNSSINS_KRLF_boost import DA_PrePosCov_PosAttCorrectCov_InHiGNSSSatCov # RL环境
from env.env_param import trip_type_maping as trip_type_maping_HK
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C
from model.ppo_recurrent_D_agent import RecurrentPPO
from funcs_rl.utilis_eskf import *
from model.model_ATF_KF import *
import time
import argparse
import yaml
import optuna
import optuna.visualization as vis
import copy
import warnings

# 精准忽略关于 lr_scheduler.step() 的警告
warnings.filterwarnings(
    "ignore", message=r".*Detected call of `lr_scheduler.step()`.*"
)
warnings.filterwarnings(
    "ignore", message=r".*RNN module weights are not part of.*"
)
"""
KRLF 调参用
"""
def main_process(config, trial_number=None):
    os.environ["CUDA_VISIBLE_DEVICES"] = config['system']['gpu_id']
    triptype = config['training_settings']['triptype']
    moreteststypelist = config['testing_dataset_setting'][triptype]  # 测试类型

    # 设置是否使用所有轨迹做训练
    # if config['training_settings']['alltraj']: # 全轨迹按比例划分训练测试
    #     ratio = 1
    #     traj_type_target_train = config['training_settings']['traj_type_target_train']  # 轨迹数据的比例
    #     traj_type_target_test = config['training_settings']['traj_type_target_test']
    #     trajdata_range = [0, len(tripIDlist) - 1]
    # else:
    #     ratio = 0.5  # 数据集所有数据一半训练一半测试
    #     traj_type_target_train = [0, 1]  # 轨迹数据的比例
    #     traj_type_target_test = [0, 1]
    #     trajdata_range =  [0, int(np.ceil(len(tripIDlist) * ratio))]

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


    # 开启训练模式
    if config["finetuning_settings"]['finetuning'] == False:
        running_date = config['training_settings']['running_date']
        training_stepnum = config['training_settings']['training_stepnum']
        learning_rate = config['training_settings']['learning_rate']
        envmod = config['env_para']['envmod']
        posnum = config['env_para']['postraj_num']
        reward_setting = config['env_para']['reward_setting']
        RL_reset_step = config['env_para']['RL_reset_step']
        ent_coef = config['model_para']['ent_coef']
        n_steps = config['model_para']['n_steps']
        batch_size = config['model_para']['batch_size']
        n_epochs = config['model_para']['n_epochs']
        vf_coef = config['model_para']['vf_coef']
        clip_range = config['model_para']['clip_range']
        clip_range_vf = config['model_para']['clip_range_vf']
        scale_state_pred = config['env_para']['continuous_scale_state_pred']
        scale_state_correct = config['env_para']['continuous_scale_state_correct']
        scale_policy_cov = config['env_para']['continuous_scale_policy_cov']
        scale_P_cov = config['env_para']['continuous_scale_P_cov']
        initial_RLcov = config['env_para']['initial_RLcov']
        magnitude = config['env_para']['magnitude']

        if envmod == 'InHiGNSSSatCov_PreCovPos-PosCorrectCov':
            env = DummyVecEnv([lambda: DA_PrePosCov_PosCorrectCov_InHiGNSSSatCov(config, **kwargs)])
            a1_encoder = CustomATF1_GnssHisInnStateCov
            a2_encoder = CustomATF1_GnssHisInnPosAttP

        tensorboard_log = f'{dir_path}records_values/{running_date}/{triptype}_{traj_type_target_train[1]}_{ratio}_{reward_setting}_{envmod}/' \
                          f'SP={scale_state_pred:.3f}_RLC={scale_policy_cov:.3f}_SC={scale_state_correct:.2f}_P={scale_P_cov:.2f}_IR={initial_RLcov:2f}'

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

        model = RecurrentPPO(
            "MlpLstmPolicy", env, verbose=0, policy_kwargs_dic=policy_kwargs_dic, tensorboard_log=tensorboard_log, learning_rate=learning_rate, ent_coef=ent_coef,
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
        testing_result = 0
        for testtype in moreteststypelist:
            print(f'more test for {testtype} env begin here')
            tripIDlist_test = trip_type_maping[testtype]
            tripIDlist_test = [item for item in tripIDlist_test if item not in exclude_traj]

            if config['training_settings']['alltraj']:
                more_test_trajrange = [0, len(tripIDlist_test) - 1] # 所有测试轨迹进入测试
            else:
                more_test_trajrange = [int(np.ceil(len(tripIDlist_test)*ratio))+1, len(tripIDlist_test) - 1] # 后ratio半的测试轨迹进入测试

            if testtype == triptype:
                traj_type = [0, 1] # traj_type_target_test  # 独立同分布测试
            else:
                traj_type = [0, 1]  # 域外分布测试范围

            test_trajlist = range(more_test_trajrange[0],more_test_trajrange[-1]+1) #[0,1,2,3,4,5]
            for test_traj in test_trajlist:
                test_trajdata_range = [test_traj, test_traj]
                test_kwargs = {
                    'traj_type': traj_type,  # 每条轨迹的取样比例
                    'trajdata_range': test_trajdata_range,  # 使用所有轨迹的比例
                    'triptype': testtype
                }

                if envmod == 'InHiGNSSSatCov_PreCovPos-PosCorrectCov':
                    env = DummyVecEnv([lambda: DA_PrePosCov_PosCorrectCov_InHiGNSSSatCov(config, **test_kwargs)])

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
            test_pos_ratio, test_rl_RMSE, test_mae_ratio, test_rl_mae = recording_results_ecef_RL4KF_SZ(dir_path, data_truth_dic,[test_trajlist[0],test_trajlist[-1]],tripIDlist_test,logdirname,
                                         config["env_para"]["baseline_mod"],traj_record=config["training_settings"]["traj_record"])

            testing_result += test_pos_ratio # 可以选择MAE或者RMSE调参
            if testtype == triptype:
                indomain_result = test_pos_ratio

        print('More Test for different phonetype finished.')

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
                    initial_logstd = config['finetuning_settings']['initial_logstd']
                    dt = config['finetuning_settings']['dt']
                    pos_w = config['finetuning_settings']['pos_weight']
                    gae_lambda = config['finetuning_settings']['gae_lambda']
                    # 修改 261022：使用调参配置中的熵系数，确保 Optuna 调整的参数真正生效。
                    ent_coef = config['finetuning_settings']['ent_coef']

                    # 新增微调数据记录文件夹
                    loger_folder = f'/Finetune_model/FT_nsteps={n_steps:.2f}_clip_range={clip_range:.2f}_kf={kf_coef:.2f}_nstep={n_steps}_dt={dt}_pw={pos_w:.2f}' \
                                   f'_logstd={initial_logstd:.1f}_gae={gae_lambda:.2f}'
                    # 修改 261022：目录中的超参数经过格式化会舍入，追加 trial 编号避免不同 trial 共用目录。
                    # if trial_number is not None:
                    #     loger_folder += f'_trial{trial_number}'
                    result_folder = model_loggerdir + loger_folder
                    if not os.path.exists(result_folder):
                        os.makedirs(result_folder)
                    # final_log_dir = create_numbered_dir(result_folder, prefix="FT_RecurrentPPO_")  # 重复实验文件夹
                    # final_log_dir = result_folder + '/FT_RecurrentPPO'

                    # 重导入构建训练环境
                    envmod = config_train['env_para']['envmod']
                    # 获取训练列表和测试类型
                    triptype = config_train['training_settings']['triptype']
                    traj_type_target_train = config_train['training_settings']['traj_type_target_train']  # 轨迹数据的比例
                    tripIDlist = trip_type_maping_HK[triptype]
                    tripIDlist = [item for item in tripIDlist if item not in exclude_traj]  # 剔除不需要的traj
                    trajdata_range = [0, len(tripIDlist) - 1]
                    # 构建训练环境输入参数
                    kwargs = {
                        'traj_type': traj_type_target_train,  # 每条轨迹的取样比例
                        'trajdata_range': trajdata_range,  # 使用所有轨迹的比例
                        'triptype': triptype}
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

                    model = RecurrentPPO("MlpLstmPolicy", env, kf_coef=kf_coef, ent_coef=ent_coef, verbose=0, n_steps=n_steps,
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
                        torch.nn.init.constant_(model.policy_Q.log_std, initial_logstd)
                        torch.nn.init.constant_(model.policy_R.log_std, initial_logstd)

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
                    model.n_steps = n_steps
                    model.n_epochs = n_epochs
                    model.clip_range_val = clip_range  # 保存个数值
                    model.clip_range = get_schedule_fn(clip_range)
                    # 同步更新策略子模块中的 clip_range (因为你的 train 函数里用的是策略里的值)
                    model.policy_Q.clip_range = model.clip_range
                    model.policy_R.clip_range = model.clip_range
                    model.gae_lambda = gae_lambda  # 推荐降到 0.90 或 0.85

                    ########### 进入测试环境 ###########
                    envmod = config['env_para']['envmod']
                    # 修改 261021：迁移测试使用当前 HK 模型保存的完整 env_para，
                    # 保持观测窗口、初始化协方差及动作尺度与模型训练时一致。
                    config_migration = copy.deepcopy(config)
                    config_migration['env_para'] = copy.deepcopy(config_train['env_para'])

                    print("-----Fine tuning Policy model-----")
                    testing_result = 0
                    # 修改 261022：整个 trial 只在第一次 learn 时创建 TensorBoard run 目录；
                    # 后续轨迹和测试类型继续复用该目录，避免生成多个 RecurrentPPO 子目录。
                    first_policy_learn = True
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
                            traj_type = [0, 1]  # 独立同分布测试
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

                            # 修改 261022：按当前 SZ 测试配置的 envmod 选择迁移环境，而非 HK 模型的 envmod。
                            if envmod == 'InHiGNSSSatCov_PreCovPos-PosCorrectCov':
                                # 这里必须导入训练环境的参数
                                env = DummyVecEnv(
                                    [lambda: DA_HK2SZ_PrePosCov_PosCorrectCov_InHiGNSSSatCov(config_migration, **test_kwargs)])

                            model.set_env(env)
                            model.learn(total_timesteps=20000, eval_log_path=result_folder,
                                        reset_num_timesteps=first_policy_learn)  # 回合结束即停止
                            first_policy_learn = False

                        logdirname = model.logger.dir + f'/finetuning_{testtype}_'

                        test_pos_ratio, test_rl_RMSE, test_mae_ratio, test_rl_mae = recording_results_ecef_RL4KF_SZ(dir_path, data_truth_dic,[test_trajlist[0],test_trajlist[-1]],tripIDlist_test,logdirname,
                                                     config["env_para"]["baseline_mod"],traj_record=True)

                        # model.save(model.logger.dir + f"/Save_model_{config['env_para']['reward_setting']}_lr{learning_rate:0.1e}")

                        # 保存参数表
                        with open(f'{logdirname}config_FT.yaml', 'w', encoding='utf-8') as f:
                            yaml.dump(config, f, allow_unicode=True, sort_keys=False)

                        testing_result += test_pos_ratio
                        if testtype == triptype:
                            indomain_result = test_pos_ratio

        print('only test finish!')

    return testing_result / len(moreteststypelist)

def objective(trial, mode_list, config_tuning_base,finetune):
    config_tuning = copy.deepcopy(config_tuning_base)  # 一个很致命的bug，n_jobs > 1 时，多线程之间参数表会相互影响
    ## 噪声参数
    if config_tuning["training_settings"]["triptype"] == 'SZ_openroad_RTK_Dagent' and finetune == False:
        if 'env_param' in mode_list:
            # A1 专属参数
            continuous_scale_state_pred = trial.suggest_float("continuous_scale_state_pred", 0.01,2.0, log=True)
            config_tuning["env_para"]["continuous_scale_state_pred"] = continuous_scale_state_pred
            continuous_scale_policy_cov = trial.suggest_float("continuous_scale_policy_cov", 1.0e-3,1.0e-1, log=True)
            config_tuning["env_para"]["continuous_scale_policy_cov"] = continuous_scale_policy_cov
            initial_RLcov = trial.suggest_float("initial_RLcov", 2, 20, log=False)
            config_tuning["env_para"]["initial_RLcov"] = initial_RLcov
            RL_reset_step = trial.suggest_int("RL_reset_step", 1, 3, log=False)
            config_tuning["env_para"]["RL_reset_step"] = RL_reset_step
            # A2 专属参数
            continuous_scale_state_correct = trial.suggest_float("continuous_scale_state_correct", 0.02,1.2, log=True)
            config_tuning["env_para"]["continuous_scale_state_correct"] = continuous_scale_state_correct
            continuous_scale_P_cov = trial.suggest_float("continuous_scale_P_cov", 1.0e-3,5.0e-2, log=True)
            config_tuning["env_para"]["continuous_scale_P_cov"] = continuous_scale_P_cov

        if 'model_param' in mode_list:
            postraj_num = trial.suggest_int("postraj_num", 5, 8, log=False)
            config_tuning["env_para"]["postraj_num"] = postraj_num
            ent_coef = trial.suggest_float("ent_coef", 0, 0.1, log=False)
            config_tuning["model_para"]["ent_coef"] = ent_coef
            vf_coef = trial.suggest_float("vf_coef", 0.1, 0.3, log=False)
            config_tuning["model_para"]["vf_coef"] = vf_coef
            clip_range = trial.suggest_float("clip_range", 0.05, 0.2, log=False)
            config_tuning["model_para"]["clip_range"] = clip_range

        if 'model_training' in mode_list:
            buffer_options = [128]
            # n_steps = trial.suggest_categorical("reply_buffer_size", buffer_options)
            # config_tuning["model_para"]["n_steps"] = n_steps
            learning_rate = trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True)
            config_tuning["training_settings"]["learning_rate"] = learning_rate

        if 'vel_att_param' in mode_list:
            # continuous_scale_att_pred = trial.suggest_float("continuous_scale_att_pred", 1.0e-3,5.0e-1, log=True) # 1.0e-3,5.0e-1
            # config_tuning["env_para"]["continuous_scale_att_pred"] = continuous_scale_att_pred
            continuous_scale_att_correct = trial.suggest_float("continuous_scale_att_correct", 0.05,0.5, log=True)
            config_tuning["env_para"]["continuous_scale_att_correct"] = continuous_scale_att_correct
            att_weight = trial.suggest_float("att_weight", 0,1, log=False) # 0,1
            config_tuning["env_para"]["att_weight"] = att_weight

    elif finetune == True:
        if 'env_param' in mode_list:
            pos_weight = trial.suggest_float("pos_weight", 0,1, log=False)
            config_tuning["finetuning_settings"]["pos_weight"] = pos_weight
            vel_weight = trial.suggest_float("vel_weight", 0,1, log=False)
            config_tuning["finetuning_settings"]["vel_weight"] = vel_weight
            att_weight = trial.suggest_float("att_weight", 0,1, log=False)
            config_tuning["finetuning_settings"]["att_weight"] = att_weight
            nis_weight = trial.suggest_float("nis_weight", 0, 1, log=False)
            config_tuning["finetuning_settings"]["nis_weight"] = nis_weight
            dt = trial.suggest_int("dt", 1, 50, log=True)
            config_tuning["finetuning_settings"]["dt"] = dt

        if 'model_param' in mode_list:
            kf_coef = trial.suggest_float("kf_coef", 0, 1, log=False)
            config_tuning["finetuning_settings"]["kf_coef"] = kf_coef
            clip_range = trial.suggest_float("clip_range", 0, 0.2, log=False)
            config_tuning["finetuning_settings"]["clip_range"] = clip_range
            initial_logstd = trial.suggest_float("initial_logstd", -5, -1, log=False)
            config_tuning["finetuning_settings"]["initial_logstd"] = initial_logstd
            gae_lambda = trial.suggest_float("gae_lambda", 0.9, 0.99, log=False )
            config_tuning["finetuning_settings"]["gae_lambda"] = gae_lambda
            # 新增
            ent_coef = trial.suggest_float("ent_coef", 0, 0.1, log=False)
            config_tuning["finetuning_settings"]["ent_coef"] = ent_coef

        if 'model_training' in mode_list:
            buffer_options = [16, 32, 64]
            n_steps = trial.suggest_categorical("n_steps", buffer_options)
            config_tuning["finetuning_settings"]["n_steps"] = n_steps
            config_tuning["finetuning_settings"]["batch_size"] = n_steps
            learning_rate = trial.suggest_float("learning_rate", 5e-6, 2e-4, log=True)
            config_tuning["finetuning_settings"]["learning_rate"] = learning_rate

    # 2. 运行你的算法
    # 假设你有一个函数执行定位并返回位置误差 RMS
    # 修改 261022：以 trial 编号区分输出，避免参数目录舍入造成结果混写。
    ll_error = main_process(config_tuning, trial_number=trial.number)

    # 3. 返回误差，Optuna 会尝试将其最小化
    return ll_error

def tuning(mode_list,config_tuning,triptype_training,finetune):
    print(f'{config_tuning["training_settings"]["triptype"]} for {mode_list} start !')
    save_name = 'records_values/'
    triptype = config_tuning['training_settings']['triptype']
    running_date = config_tuning['training_settings']['running_date']
    # 1. 设置一个统一的实验名称
    study_name = f"tuning_{running_date}_{triptype_training}_{finetune}_forest"
    # 2. 使用 sqlite 数据库保存进度（当前目录下会生成一个 my_tuning.db 文件）
    storage_name = f"sqlite:///my_tuning_{running_date}_{triptype_training}_{finetune}_forest.db"
    time.sleep(random.uniform(0, 3))  # 随机休眠防止撞车
    try:
        # 尝试建库
        study = optuna.create_study(
            study_name=study_name,
            direction="maximize",
            storage=storage_name,
            load_if_exists=True
        )
    except Exception as e:  # 捕获到 SQLite 的 Unique constraint 报错
        print(f"发现建库冲突，等待 2 秒后重试...")
        time.sleep(2)
        # 别人已经建好了，这次直接 load_if_exists=True 进去就行
        study = optuna.create_study(
            study_name=study_name,
            direction="maximize",
            storage=storage_name,
            load_if_exists=True
        )
    study.optimize(lambda trial: objective(trial, mode_list, config_tuning,finetune), n_trials=50,gc_after_trial=True,show_progress_bar=True, n_jobs=1)
    print("最优参数: ", study.best_params)
    df = study.trials_dataframe()
    # 保存为 CSV
    tune_param = "_".join(mode_list)
    vis.plot_optimization_history(study).show()
    df.to_csv(dir_path + save_name + f"/optimization_results_{running_date}_{triptype}_{tune_param}.csv", index=False, encoding='utf-8-sig')

if __name__ == "__main__":
    traj_training_list = ['SZ_De_RTK_Dagent'] # SZ_openroad_RTK_Dagent
    gpu_list = ["0","1","2","3"] # ,"0","1","2","3"
    task_list = []
    for idx in range(len(gpu_list)):
        for triptype_training in traj_training_list:
            gpu_id = gpu_list[idx]
            conf_name = f'KRLF_{triptype_training}.yaml' # 读取相应的参数表
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

            config["system"]["gpu_id"] = gpu_id
            finetune = config["finetuning_settings"]["finetuning"]
            # config["training_settings"]["triptype"] = triptype

            mode_list_1 = ['env_param','model_training'] # 'env_param','model_param','model_training','vel_att_param'
            process_1 = multiprocessing.Process(target=tuning, args=(mode_list_1,config,triptype_training,finetune))
            task_list.append(process_1)
            # mode_list_2 = ['env_param','model_param','model_training']
            # process_2 = multiprocessing.Process(target=tuning, args=(mode_list_2, config,))
            # task_list.append(process_2)
            # mode_list_3 = ['env_param','model_param','model_training']
            # process_3 = multiprocessing.Process(target=tuning, args=(mode_list_3, config,))
            # task_list.append(process_3)
            process_1.start()
            # process_2.start()
            # process_3.start()

    # for t in task_list:
    #     t.start()

    for t in task_list:
        t.join()
