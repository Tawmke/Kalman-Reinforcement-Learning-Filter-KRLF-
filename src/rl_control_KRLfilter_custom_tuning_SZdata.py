import numpy as np
import pandas as pd
import os
import sys
cur_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(cur_dir, '..')
sys.path.append(src_dir)
import multiprocessing
from env.SZdata_GNSSINS_KRLF_split_boost import * # RL环境
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C
from model.ppo_recurrent_ATF1_AKF import RecurrentPPO
from funcs_rl.utilis_eskf import *
from model.model_ATF_KF import *
import time
import argparse
import yaml
import optuna
import optuna.visualization as vis
import copy
"""
KRLF 调参用
"""
def main_process(config):
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
    if True:
        # 构建环境输入参数
        kwargs = {
            'traj_type': traj_type_target_train, # 每条轨迹的取样比例
            'trajdata_range': trajdata_range, # 使用所有轨迹的比例
            'triptype': triptype
        }
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
        scale_policy_cov = config['env_para']['continuous_scale_policy_cov']
        initial_RLcov = config['env_para']['initial_RLcov']
        magnitude = config['env_para']['magnitude']

        if envmod == 'InHiGNSSSatCov_PreCovPos':
            env = DummyVecEnv([lambda: Continuous_PrePosCov_InHiGNSSSatCov(config, **kwargs)])
            encoder = CustomATF1_GnssHisInnStateCov
        elif envmod == 'InHiGNSSSatCov_POSCorrectCov':
            env = DummyVecEnv([lambda: Continuous_POSCorrectCov_InHiGNSSPosCov(config, **kwargs)])
            encoder = CustomATF1_GnssHisInnStateCov

        tensorboard_log = f'{dir_path}records_values/{running_date}/{triptype}_{traj_type_target_train[1]}_{ratio}_{reward_setting}_{envmod}/' \
                          f'lr={learning_rate:.4f}_mag={magnitude}_SP={scale_state_pred:.2f}_PC={scale_policy_cov:.4f}_Reset={RL_reset_step}_IR={initial_RLcov:.2f}'

        obs = env.reset()
        policy_dim = 0
        for key, value in obs.items():
            policy_dim += obs[key].shape[-1]
        policy_kwargs = dict(features_extractor_class=encoder,  features_extractor_kwargs=dict(features_dim=policy_dim),
                             ATF_trig=config['model_para']['networkmod'], net_arch=config['model_para']['net_archppo'])

        model = RecurrentPPO(
            "MlpLstmPolicy", env, verbose=0, policy_kwargs=policy_kwargs, tensorboard_log=tensorboard_log, learning_rate=learning_rate, ent_coef=ent_coef,
            n_steps=n_steps, batch_size=batch_size, n_epochs=n_epochs, vf_coef=vf_coef, clip_range=clip_range)
        model.learn(total_timesteps=training_stepnum, eval_log_path=tensorboard_log)

        #print and save training results
        logdirname=model.logger.dir+'/train_'
        # logdirname='./'
        print('Training finished.')

        #record model
        # params=model.get_parameters()
        model.save(model.logger.dir+f"/Save_model_{config['env_para']['reward_setting']}_trainingnum{training_stepnum:0.1e}_lr{learning_rate:0.1e}")
        train_pos_ratio, train_rl_RMSE = recording_results_ecef_RL4KF_SZ(dir_path, data_truth_dic,trajdata_range,tripIDlist,logdirname,config["env_para"]["baseline_mod"],
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
            test_pos_ratio, test_rl_RMSE = recording_results_ecef_RL4KF_SZ(dir_path, data_truth_dic,[test_trajlist[0],test_trajlist[-1]],tripIDlist_test,logdirname,
                                         config["env_para"]["baseline_mod"],traj_record=config["training_settings"]["traj_record"])

            testing_result += test_pos_ratio
            if testtype == triptype:
                indomain_result = test_pos_ratio

        print('More Test for different phonetype finished.')

    return testing_result / len(moreteststypelist)

def objective(trial, mode_list, config_tuning_base):
    config_tuning = copy.deepcopy(config_tuning_base)  # 一个很致命的bug，n_jobs > 1 时，多线程之间参数表会相互影响
    ## 噪声参数
    if config_tuning["training_settings"]["triptype"] == 'SZ_canyon_RTK':
        if 'env_param' in mode_list:
            continuous_scale_state_pred = trial.suggest_float("continuous_scale_state_pred", 5e-2,2.5, log=True)
            config_tuning["env_para"]["continuous_scale_state_pred"] = continuous_scale_state_pred
            continuous_scale_policy_cov = trial.suggest_float("continuous_scale_policy_cov", 5.0e-3,5.0e-2, log=True)
            config_tuning["env_para"]["continuous_scale_policy_cov"] = continuous_scale_policy_cov
            initial_RLcov = trial.suggest_float("initial_RLcov", 6, 16, log=False)
            config_tuning["env_para"]["initial_RLcov"] = initial_RLcov
            RL_reset_step = trial.suggest_int("RL_reset_step", 1, 4, log=False)
            config_tuning["env_para"]["RL_reset_step"] = RL_reset_step
            magnitude_options = [0,0.5,1]
            magnitude = trial.suggest_categorical("magnitude", magnitude_options)
            config_tuning["env_para"]["magnitude"] = magnitude

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
            learning_rate = trial.suggest_float("learning_rate", 5e-5, 5e-3, log=True)
            config_tuning["training_settings"]["learning_rate"] = learning_rate

    elif config_tuning["training_settings"]["triptype"] == 'SZ_openroad_RTK':
        if 'env_param' in mode_list:
            continuous_scale_state_pred = trial.suggest_float("continuous_scale_state_pred", 5e-2,3.0, log=True)
            config_tuning["env_para"]["continuous_scale_state_pred"] = continuous_scale_state_pred
            continuous_scale_policy_cov = trial.suggest_float("continuous_scale_policy_cov", 5.0e-3,1.0e-1, log=True)
            config_tuning["env_para"]["continuous_scale_policy_cov"] = continuous_scale_policy_cov
            initial_RLcov = trial.suggest_float("initial_RLcov", 5, 30, log=False)
            config_tuning["env_para"]["initial_RLcov"] = initial_RLcov
            RL_reset_step = trial.suggest_int("RL_reset_step", 1, 2, log=False)
            config_tuning["env_para"]["RL_reset_step"] = RL_reset_step
            # magnitude_options = [0,0.5,1]
            # magnitude = trial.suggest_categorical("magnitude", magnitude_options)
            # config_tuning["env_para"]["magnitude"] = magnitude

        if 'model_param' in mode_list:
            # postraj_num = trial.suggest_int("postraj_num", 5, 8, log=False)
            # config_tuning["env_para"]["postraj_num"] = postraj_num
            # ent_coef = trial.suggest_float("ent_coef", 0, 0.1, log=False)
            # config_tuning["model_para"]["ent_coef"] = ent_coef
            # vf_coef = trial.suggest_float("vf_coef", 0.1, 0.3, log=False)
            # config_tuning["model_para"]["vf_coef"] = vf_coef
            clip_range = trial.suggest_float("clip_range", 0.05, 0.2, log=False)
            config_tuning["model_para"]["clip_range"] = clip_range

        if 'model_training' in mode_list:
            # buffer_options = [128]
            # n_steps = trial.suggest_categorical("reply_buffer_size", buffer_options)
            # config_tuning["model_para"]["n_steps"] = n_steps
            learning_rate = trial.suggest_float("learning_rate", 8e-5, 1e-3, log=True)
            config_tuning["training_settings"]["learning_rate"] = learning_rate

    elif config_tuning["training_settings"]["triptype"] == 'SZ_openroad_RTK_A2':
        if 'env_param' in mode_list:
            continuous_scale_state_pred = trial.suggest_float("continuous_scale_state_pred", 5e-2,0.8, log=True)
            config_tuning["env_para"]["continuous_scale_state_pred"] = continuous_scale_state_pred
            continuous_scale_policy_cov = trial.suggest_float("continuous_scale_policy_cov", 5.0e-3,1.0e-1, log=True)
            config_tuning["env_para"]["continuous_scale_policy_cov"] = continuous_scale_policy_cov

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
            # buffer_options = [128]
            # n_steps = trial.suggest_categorical("reply_buffer_size", buffer_options)
            # config_tuning["model_para"]["n_steps"] = n_steps
            learning_rate = trial.suggest_float("learning_rate", 8e-5, 2e-3, log=True)
            config_tuning["training_settings"]["learning_rate"] = learning_rate

    # 2. 运行你的算法
    # 假设你有一个函数执行定位并返回位置误差 RMS
    ll_error = main_process(config_tuning)

    # 3. 返回误差，Optuna 会尝试将其最小化
    return ll_error

def tuning(mode_list,config_tuning,triptype_training):
    print(f'{config_tuning["training_settings"]["triptype"]} for {mode_list} start !')
    save_name = 'records_values/'
    running_date = config_tuning['training_settings']['running_date']
    # 1. 设置一个统一的实验名称
    study_name = "multi_gpu_tuning"
    # 2. 使用 sqlite 数据库保存进度（当前目录下会生成一个 my_tuning.db 文件）
    storage_name = f"sqlite:///my_tuning_{running_date}_{triptype_training}.db"
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
    study.optimize(lambda trial: objective(trial, mode_list, config_tuning), n_trials=50,gc_after_trial=True,show_progress_bar=True, n_jobs=1)
    print("最优参数: ", study.best_params)
    df = study.trials_dataframe()
    # 保存为 CSV
    tune_param = "_".join(mode_list)
    vis.plot_optimization_history(study).show()
    df.to_csv(dir_path + save_name + f"/optimization_results_{running_date}_{triptype}_{tune_param}.csv", index=False, encoding='utf-8-sig')

if __name__ == "__main__":
    traj_training_list = ['SZ_openroad_RTK'] # ,SZ_canyon_RTK SZ_forest_RTK SZ_openroad_RTK SZ_openroad_RTK_A2
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
            # config["training_settings"]["triptype"] = triptype

            mode_list_1 = ['env_param','model_param','model_training'] # 'env_param','model_param','model_training','vel_att_param'
            process_1 = multiprocessing.Process(target=tuning, args=(mode_list_1,config,triptype_training))
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