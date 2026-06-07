
import os
import re
from pathlib import Path
import numpy as np
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import math

def analyze_ratio(file_path,keywords):
    """
    分析最优ratio参数
    :param file_path:
    :return:
    """
    base_path = Path(file_path)
    # results_map 存储格式: { folder_A_name: [ratio_val1, ratio_val2, ...] }
    results_map = {}
    # full_text_map 存储格式: { folder_A_name: "完整的txt内容字符串" }
    full_text_map = {}

    # 1. 遍历文件夹 A (不同参数)
    if not base_path.exists():
        print(f"路径不存在: {file_path}")
        return

    for folder_a in base_path.iterdir():
        if not folder_a.is_dir():
            continue

        ratios_in_a = []

        # 2. 遍历文件夹 B (重复实验次数)
        for folder_b in folder_a.iterdir():
            if not folder_b.is_dir():
                continue

            # --- 提取 txt 中的 Ratio ---
            candidate_files = []
            for txt_file in folder_b.glob("*.txt"):
                if all(key in txt_file.name for key in keywords):
                    candidate_files.append(txt_file)

            if candidate_files:
                # key=lambda x: x.stat().st_mtime 表示按修改时间排序
                # reverse=True 表示降序，即第一个就是最新的文件
                candidate_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                # 3. 只取最新的一条进行处理
                latest_file = candidate_files[0]
                try:
                    with open(latest_file, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if not content:
                            continue
                        match = re.search(r"Ratio:\s*(-?[\d.]+)", content)
                        if match:
                            ratio_val = float(match.group(1))
                            ratios_in_a.append(ratio_val)
                            # 记录该文件夹 A 下的一个代表性完整结果（用于最后展示）
                            full_text_map[folder_a.name] = content
                except Exception as e:
                    print(f"读取TXT文件出错 {latest_file}: {e}")
            else:
                print("未找到符合关键字条件的TXT文件")

            # 3. 筛选文件名同时含有 'ratio' 和 'testmore' 的 txt 文件
            # for txt_file in folder_b.glob("*.txt"):
            #     fname = txt_file.name
            #     if all(key in fname for key in keywords):
            #         try:
            #             with open(txt_file, 'r', encoding='utf-8') as f:
            #                 content = f.read().strip()
            #                 if not content:
            #                     continue
            #
            #                 # 4. 正则匹配 Ratio 数值
            #                 # 匹配 "Ratio:" 后面跟着的浮点数
            #                 match = re.search(r"Ratio:([\d.]+)", content)
            #                 if match:
            #                     ratio_val = float(match.group(1))
            #                     ratios_in_a.append(ratio_val)
            #                     # 记录该文件夹 A 下的一个代表性完整结果（用于最后展示）
            #                     full_text_map[folder_a.name] = content
            #         except Exception as e:
            #             print(f"解析文件 {txt_file} 时出错: {e}")

        # 如果该参数文件夹 A 下找到了有效数据，存入 map
        if ratios_in_a:
            results_map[folder_a.name] = ratios_in_a

    # 5. 计算平均值并提取最优参数
    if not results_map:
        print("未找到同时包含 'ratio' 和 'testmore' 的有效结果文件。")
        return

    # 计算每个文件夹 A 的平均 Ratio
    avg_scores = {name: np.mean(vals) for name, vals in results_map.items()}

    # 找到平均值最大的文件夹 A
    best_folder_a = max(avg_scores, key=avg_scores.get)
    max_avg_ratio = avg_scores[best_folder_a]

    # --- 最终打印 ---
    print("=" * 60)
    print(f"项目分析报告")
    print("=" * 60)
    print(f"🏆 表现最优参数: {best_folder_a}")
    print(f"📈 最大平均 Ratio 值: {max_avg_ratio:.6f}")
    print(f"总计参与计算的样本数: {len(results_map[best_folder_a])}")
    print("-" * 60)
    print(f"该路径下的原始记录样例:")
    print(f"\"{full_text_map[best_folder_a]}\"")
    print("=" * 60)


def extract_and_plot_results(file_path, krlf_yaml_path,param_type, target_param, keywords):
    base_path = Path(file_path)
    data_records = []

    # 1. 检查基础路径
    if not base_path.exists():
        print(f"路径不存在: {file_path}")
        return

    print("开始扫描文件并提取数据...")

    # ================= 1. 读取 KRLF.yaml 获取排除基准值 =================
    exclude_value = None
    try:
        with open(krlf_yaml_path, 'r', encoding='utf-8') as f:
            krlf_config = yaml.safe_load(f)
            # 解析嵌套结构：krlf_config[param_type][target_param]
            if krlf_config and param_type in krlf_config and target_param in krlf_config[param_type]:
                exclude_value = float(krlf_config[param_type][target_param])
                print(f"✅ 成功加载基准配置: 将排除 {target_param} == {exclude_value} 的数据。")
            else:
                print(f"⚠️ KRLF.yaml 中未找到对应的 [{param_type}][{target_param}]，不执行过滤。")
    except Exception as e:
        print(f"❌ 读取 KRLF.yaml 失败: {e}，不执行过滤。")

    print("\n开始扫描文件并提取数据...")

    # 2. 遍历文件夹 A (不同参数组)
    for folder_a in base_path.iterdir():
        if not folder_a.is_dir():
            continue

        # 3. 遍历文件夹 B (重复实验次数)
        for folder_b in folder_a.iterdir():
            if not folder_b.is_dir():
                continue

            ratio_val = None
            param_val = None

            # --- 提取 txt 中的 Ratio ---
            candidate_files = []
            for txt_file in folder_b.glob("*.txt"):
                if all(key in txt_file.name for key in keywords):
                    candidate_files.append(txt_file)

            # 2. 如果找到了符合条件的文件，按修改时间从晚到早排序
            if candidate_files:
                # key=lambda x: x.stat().st_mtime 表示按修改时间排序
                # reverse=True 表示降序，即第一个就是最新的文件
                candidate_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                # 3. 只取最新的一条进行处理
                latest_file = candidate_files[0]
                try:
                    with open(latest_file, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        match = re.search(r"Ratio:\s*(-?[\d.]+)", content)
                        if match:
                            ratio_val = float(match.group(1))
                            print(f"成功从最新文件 {folder_a} 提取 Ratio: {ratio_val}") # latest_file.name
                except Exception as e:
                    print(f"读取TXT文件出错 {latest_file}: {e}")
            else:
                print("未找到符合关键字条件的TXT文件")

            # --- 提取 yaml 中的参数 ---
            yaml_file = folder_b / 'train_config.yaml'
            if yaml_file.exists():
                try:
                    with open(yaml_file, 'r', encoding='utf-8') as f:
                        config = yaml.safe_load(f)
                        # 假设参数在第一层级。如果是嵌套的，例如 config['model']['scale']，请相应修改
                        if config and target_param in config[param_type]:
                            param_val = float(config[param_type][target_param])
                except Exception as e:
                    print(f"读取YAML文件出错 {yaml_file}: {e}")

            # --- 如果两者都成功提取，则记录这一组数据 ---
            if ratio_val is not None and param_val is not None:
                if exclude_value is not None and math.isclose(param_val, exclude_value, rel_tol=1e-6):
                    continue  # 满足条件，直接跳过，不录入数据

                data_records.append({
                    'Folder_A': folder_a.name,
                    'Folder_B': folder_b.name,
                    target_param: param_val,
                    'Ratio': ratio_val * 100
                })

    # 4. 数据整理与绘图准备
    if not data_records:
        print("未提取到任何有效数据，请检查路径或文件格式。")
        return

    # 转换为 Pandas DataFrame，方便处理和绘图
    df = pd.DataFrame(data_records)

    # 按照参数值从小到大排序
    df = df.sort_values(by=target_param).reset_index(drop=True)

    print(f"\n成功提取 {len(df)} 条有效记录。")
    print(df.head())  # 打印前几行预览一下

    # ================= 3. 自动检测量级跨度 =================
    # 获取严格大于 0 的最小值和最大值
    positive_vals = df[df[target_param] > 0][target_param]
    if not positive_vals.empty:
        x_min = positive_vals.min()
        x_max = positive_vals.max()
        # 如果最大值是最小值的 50 倍以上，开启对数坐标
        use_log_scale = (x_max / x_min) >= 50
    else:
        use_log_scale = False

    # ================= 3. 数据整理 =================
    df = pd.DataFrame(data_records)
    df = df.sort_values(by=target_param).reset_index(drop=True)

    # ================= 💥 新增功能：寻找最优参数 =================
    # 1. 计算每个参数对应的平均 Ratio
    avg_df = df.groupby(target_param)['Ratio'].mean().reset_index()

    # 2. 找到平均值最大的一行
    best_row_avg = avg_df.loc[avg_df['Ratio'].idxmax()]
    best_param_avg = best_row_avg[target_param]
    max_ratio_avg = best_row_avg['Ratio']

    # 3. 找出所有实验中单次跑出的绝对最大值（供参考）
    best_row_abs = df.loc[df['Ratio'].idxmax()]
    best_param_abs = best_row_abs[target_param]
    max_ratio_abs = best_row_abs['Ratio']

    print("\n" + "=" * 50)
    print("🏆 最优结果分析")
    print("=" * 50)
    print(f"🔹 最佳参数值 ({target_param}): {best_param_avg}")
    print(f"🔹 对应的最大平均 Ratio: {max_ratio_avg:.4f}")
    print(f"   (注: 单次实验碰巧跑出的绝对最高 Ratio 为 {max_ratio_abs:.4f}，发生于参数 {best_param_abs} 时)")
    print("=" * 50 + "\n")

    # ================= 5. 开始绘制美观的曲线图 =================

    # 设置 Seaborn 的美化主题和高分辨率
    sns.set_theme(style="whitegrid", palette="muted")
    plt.figure(figsize=(10, 6), dpi=150)

    # 使用 lineplot。当 X 轴同一个值有多个 Y 值时，
    # Seaborn 会自动计算平均值，并画出代表方差的半透明误差带 (Confidence Interval)
    ax = sns.lineplot(
        data=df,
        x=target_param,
        y='Ratio',
        marker='o',  # 在数据点上画圆圈
        markersize=8,  # 圆圈大小
        linewidth=2.5,  # 线条粗细
        ci=95  # 显示95%置信区间，若不需要误差带可改为 errorbar=None
    )
    # 用一个红色的星号在图上把最优平均值标出来
    plt.scatter([best_param_avg], [max_ratio_avg], color='red', marker='*', s=300, zorder=5, label='Best Average Ratio')
    plt.annotate(
        f'({best_param_avg:.4f}, {max_ratio_avg:.2f})',  # 文本内容，y值保留两位小数
        xy=(best_param_avg, max_ratio_avg),  # 锚点坐标 (星星的位置)
        xytext=(15, 5),  # 文本相对于锚点的偏移 (向右15像素，向上5像素)
        textcoords='offset points',  # 声明使用像素偏移，防止对数坐标系下文字变形
        color='red',  # 字体颜色
        fontsize=14,  # 字体大小
        fontweight='bold',  # 加粗
        ha='left',  # 水平左对齐
        va='bottom'  # 垂直底部对齐
    )
    plt.legend(fontsize=18)

    # 如果检测到跨度过大，设置为对数坐标 (Log Scale)
    if use_log_scale:
        ax.set_xscale('log')
        x_label = f"{target_param} (Log Scale)"
        print(f"检测到 {target_param} 量级跨度较大，已自动切换为对数坐标轴。")
    else:
        x_label = target_param

    # 图表细节美化
    # plt.title(f'Effect of {target_param} on Ratio', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel(target_param, fontsize=18, labelpad=10)
    plt.ylabel('Position RMSE Ratio (%)', fontsize=18, labelpad=10)

    # 修改刻度字体大小
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # 移除顶部和右侧的边框线，让图表看起来更干净（学术风）
    sns.despine()

    # 保存高清图片并显示
    output_img = f"trend_plot_{target_param}.png"
    # plt.savefig(output_img, bbox_inches='tight')
    print(f"\n图表已生成并保存为: {output_img}")

    plt.show()

if __name__ == "__main__":
    dir_path = '/mnt/sdb/home/tangjh/KF-GINS-Py-main/records_values/'
    triptype = 'HK_De_ublox_DA_FT'
    if triptype == 'HK_Ha_ublox':
        record_path = 'tuning_4_testing_indomain/HK_Ha_ublox_0.7_1_RMSEadv_ratio_InHiGNSSPosAtt_PreCovPos'  # tuning_4_testing_indomain finetuning_4_testing_indomain_0403
    elif triptype == 'HK_De_ublox':
        record_path = 'finetuning_4_testing_indomain_att_0407/HK_De_ublox_0.7_1_RMSEadv_ratio'
    elif triptype == 'HK_De_ublox_A2':
        record_path = 'finetuning_4_testing_indomain_att_0407/HK_De_ublox_A2_0.7_1_RMSEadv_ratio_InHiGNSSPos_CorrectCovPosAtt'
    elif triptype == 'HK_De_ublox_DA':
        record_path = 'finetuning_4_testing_indomain_att_0407/HK_De_ublox_Dagent_0.7_1_RMSEadv_ratio_InHiGNSSSatCov_PreCovPos-PosAttCorrectCov'
    elif triptype == 'HK_De_ublox_DA_FT':
        record_path = 'finetuning_4_testing_indomain_att_0407/HK_De_ublox_Dagent_0.7_1_RMSEadv_ratio_InHiGNSSSatCov_PreCovPos-PosAttCorrectCov/' \
                      'lr=0.0005_pos=6_SP=1.95_SC=0.40_AP=0.0170_AC=0.0864_AW=0.89/RecurrentPPO_1'
    elif triptype == 'HK_Me_ublox':
        record_path = 'finetuning_4_testing_indomain_att_0407/HK_Me_ublox_0.7_1_RMSEadv_ratio'
    elif triptype == 'HK_Me_ublox_A2':
        record_path = 'finetuning_4_testing_indomain_att_0407/HK_Me_ublox_A2_0.7_1_RMSEadv_ratio_InHiGNSSPos_CorrectCovPosAtt'
    elif triptype == 'HK_Me_ublox_DA':
        record_path = 'finetuning_4_testing_indomain_att_0407/HK_Me_ublox_Dagent_0.7_1_RMSEadv_ratio_InHiGNSSSatCov_PreCovPos-PosAttCorrectCov'
    elif triptype == 'SZ_openroad_RTK_A2':
        record_path = 'tuning_4_testing_indomain_SZ/SZ_openroad_RTK_A2_0.7_1_RMSEadv_ratio_InHiGNSSSatCov_POSCorrectCov'
    elif triptype == 'SZ_openroad_RTK':
        record_path = 'tuning_4_testing_indomain_SZ/SZ_openroad_RTK_0.7_1_RMSEadv_ratio_InHiGNSSSatCov_PreCovPos'

    if 'FT' in triptype:
        keywords = ['ratio', 'finetuning', 'pos']  # ,'forest' testmore ,'openroad'
    else:
        keywords = ['ratio', 'testmore','pos'] # ,'forest' testmore ,'openroad'
    file_path = dir_path + record_path
    # 分析最优参数
    analyze_ratio(file_path,keywords)
    # 读取某类参数的曲线
    param_type = 'env_para'
    target_param = "continuous_scale_state_correct" # continuous_scale_state_pred
    krlf_yaml_path = f"/mnt/sdb/home/tangjh/KF-GINS-Py-main/src/KRLF_{triptype}.yaml"
    extract_and_plot_results(file_path, krlf_yaml_path, param_type, target_param, keywords)