---
author: Jianhao Tang
  - "2025-05-22: Jianhao Tang  初始文档创建"
---

# Python版本KF-GINS 实现 KRLF算法（更新中）

## 前提说明
***实验室资料使用严禁外传！！***

## 简介


[KF-GINS](https://github.com/i2Nav-WHU/KF-GINS) 实现了武大经典的GNSS/INS松组合导航解算，其包含了相应的 [C++版本](https://github.com/i2Nav-WHU/KF-GINS) 和 [Matlab版本](https://github.com/i2Nav-WHU/KF-GINS-Matlab) 。
本项目使用了开源的[香港数据集(UrbanNavDataset)](https://github.com/IPNL-POLYU/UrbanNavDataset?tab=readme-ov-file)以及使用了北斗智芯团队自建的深圳数据集。有需要使用同学可参考该框架进行RL+组合导航的创新实验验证。

> **注意：该项目并非 [KFGINS](https://github.com/i2Nav-WHU/KFGINS) 官方版本。**

## 代码结构
```
project-root/
├── README.md                   # 本文件
├── dataset_cpt/                # 武大提供的原始数据集，本项目不使用
├── dataset_SZ/                 # 深圳原始数据集
├── dataset_Urbannav/           # 原始UrbanNavDataset
├── env/                        # RL环境代码
├── gnss_lib/                   # 一些卫星定位函数
├── src/                        # py执行代码
├── ── configs/                 # 模型的参数配置文件
├── ── kf_gins_SZdata_processing.py           # GNSS/INS数据处理（深圳）
├── ── kf gins SZdata.py                      # 基础GNSS/INS实现（深圳）
├── ── rl_control_KRLfilter_custom_SZdata.py  # 基础KRLF执行代码（深圳）
├── ── rl_control_KRLfilter_custom_tuning_SZdata.py  # 基础KRLF调参代码（深圳）
├── ...
```

## 1. 使用说明

### 1.1 安装依赖
本项目使用的环境要求不高，可使用[environment.yml](environment.yml)进行一步到位的配置（可能会有一些多余的库），或者根据缺失的包一步一步进行配置，或者参考服务器85中dingweizu文件夹的环境配置。需要注意，stable_baseline3的版本只能是1.6.2，python版本为3.8.10。

### 1.2 快速开始
1. 运行代码[rl_control_KRLfilter_custom_SZdata.py](./src/rl_control_KRLfilter_custom_SZdata.py)或[rl_control_KRLfilter_custom.py](./src/rl_control_KRLfilter_custom.py)分别进行香港和深圳数据集的实验，通过修改`triptype_traning`参数可以切换对应实验的训练环境。
2. 在本项目设置中，A2表示Agent-2（位置/姿态状态修正），没有写的是Agent-1（位置/姿态状态预测）
3. 在纯测试阶段，可以设置`onlytesting=True`和`Twoagent_testing=True`进行双智能体联合工作（待更新）

```python
# 可选环境：SZ_canyon_RTK SZ_forest_RTK SZ_overpass_RTK SZ_openroad_RTK SZ_openroad_RTK_A2
triptype_traning = 'SZ_openroad_RTK_A2'
```
### 1.3 参数设置
在`./src/configs`的文件夹下，有对应`triptype`的参数配置文件

### 1.4 超参数搜索
1. 运行代码[rl_control_KRLfilter_custom_tuning_SZdata.py](./src/rl_control_KRLfilter_custom_tuning_SZdata.py)或[rl_control_KRLfilter_custom_tuning.py](./src/rl_control_KRLfilter_custom_tuning.py)分别进行香港和深圳数据集的超参数调优。
2. 通过修改`traj_list`参数可以切换对应实验的训练环境。
3. 通过修改`objective`函数里各参数的范围可以设置不同参数的搜索范围。

### 1.5 数据处理
1. 运行代码[kf_gins_SZdata_processing.py](./src/kf_gins_SZdata_processing.py)或[kf_gins_data_processing.py](./src/kf_gins_data_processing.py)分别进行香港和深圳数据集的打包处理。
2. 如果项目已提供`env/`环境中已处理好的pkl和csv文件，则不需要再重新运行，如有需要可自己再处理
