import shap
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.projections import register_projection
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

# 设置全局字体为Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 18

# 自定义绿-白-红渐变（可调整颜色深浅）
colors = ["#27AE60", "#FFFFFF", "#E74C3C"]  # 深绿→白→深红
n_bins = 100  # 渐变平滑度
cmap_green_white_red = LinearSegmentedColormap.from_list(
    "GreenWhiteRed", colors, N=n_bins
)

def plot_awakening_radar(pred_labels, true_labels):
    """
    Plot awakening level radar chart to compare predicted values and true values
    
    Parameters:
    pred_labels: list/np.array, 5 predicted awakening level values
    true_labels: list/np.array, 5 true awakening level values
    """
    # Validate input length
    if len(pred_labels) != 5 or len(true_labels) != 5:
        raise ValueError("Both predicted labels and true labels must contain exactly 5 awakening level values!")
    
    # Define radar chart projection (fix Spine initialization parameter issue)
    def radar_factory(num_vars, frame='circle'):
        theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)
        class RadarAxes(PolarAxes):
            name = 'radar'
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.set_theta_zero_location('N')

            def fill(self, *args, closed=True, **kwargs):
                return super().fill(closed=closed, *args, **kwargs)

            def plot(self, *args, **kwargs):
                lines = super().plot(*args, **kwargs)
                for line in lines:
                    self._close_line(line)

            def _close_line(self, line):
                x, y = line.get_data()
                if x[0] != x[-1]:
                    x = np.concatenate((x, [x[0]]))
                    y = np.concatenate((y, [y[0]]))
                    line.set_data(x, y)

            def set_varlabels(self, labels):
                self.set_thetagrids(np.degrees(theta), labels)

            def _gen_axes_patch(self):
                if frame == 'circle':
                    return Circle((0.5, 0.5), 0.5)
                elif frame == 'polygon':
                    return RegularPolygon((0.5, 0.5), num_vars, radius=.5, edgecolor="k")
                else:
                    raise ValueError("unknown value for 'frame': %s" % frame)

            def _gen_axes_spines(self):
                if frame == 'circle':
                    # Fix core issue: explicitly specify spine_type parameter (adapt to new Matplotlib version)
                    spine = Spine(axes=self, spine_type='circle', path=Path.unit_circle())
                    spine.set_transform(Affine2D().scale(.5).translate(.5, .5) + self.transAxes)
                    return {'polar': spine}
                else:
                    return PolarAxes._gen_axes_spines(self)

        register_projection(RadarAxes)
        return theta

    # Define 5 awakening level dimension labels (modify according to actual needs)
    labels = ['SO', 'HS', 'LC', 'MC', 'SC']
    N = len(labels)
    theta = radar_factory(N, frame='circle')

    # Initialize canvas
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='radar'))
    fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)

    # Set radar chart value range (automatically adapt to min/max values for aesthetic appearance)
    all_data = np.concatenate([pred_labels, true_labels])
    vmin, vmax = np.min(all_data), np.max(all_data)
    # 扩大数值范围，为数值标注留出空间
    ax.set_ylim(vmin - 0.2*(vmax-vmin), vmax + 0.2*(vmax-vmin))

    # Draw grid lines
    ax.set_rgrids(np.linspace(vmin, vmax, 5), angle=0, fontsize=12)
    ax.set_varlabels(labels)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Draw true and predicted value curves
    # True values: blue solid line + translucent fill
    ax.plot(theta, true_labels, color='#1f77b4', linewidth=2, label='True Arousal Level')
    ax.fill(theta, true_labels, color='#1f77b4', alpha=0.25)
    # Predicted values: orange dashed line + translucent fill
    ax.plot(theta, pred_labels, color='#ff7f0e', linewidth=2, linestyle='--', label='Predicted Arousal Level')
    ax.fill(theta, pred_labels, color='#ff7f0e', alpha=0.25)

    # ---------------------- 核心新增：添加数值标注 ----------------------
    # 定义标注样式参数
    fontsize = 12
    true_color = '#1f77b4'  # 真实值颜色
    pred_color = '#ff7f0e'  # 预测值颜色
    offset = 0.05 * (vmax - vmin)  # 数值标注偏移量，避免与线条重叠

    # 为每个维度添加真实值标注
    for angle, value in zip(theta, true_labels):
        # 标注位置在真实值基础上轻微外移，避免遮挡
        ax.text(angle, value - offset, f'{value:.2f}', 
                ha='center', va='center', fontsize=fontsize, color=true_color,
                fontweight='bold')
    
    # 为每个维度添加预测值标注（偏移量更大，避免与真实值标注重叠）
    for angle, value in zip(theta, pred_labels):
        ax.text(angle, value + 2*offset, f'{value:.2f}', 
                ha='center', va='center', fontsize=fontsize, color=pred_color,
                fontweight='bold')

    # Set legend and title
    ax.legend(loc='upper left', fontsize=13)
    plt.tight_layout()
    # Display the chart
    plt.show()

# 2. 定义一个预测包装函数
def model_predict_function(X):
    """
    一个包装函数，用于将 TabularModel 适配到 SHAP 的接口。
    参数:
    X (np.ndarray 或 pd.DataFrame): 特征数据。SHAP 会传递一个 NumPy 数组进来。
    返回:
    np.ndarray: 模型的预测概率或原始输出。
    """
    # SHAP 的 KernelExplainer 会传递 NumPy 数组，而 TabularModel 的 predict 方法需要 DataFrame
    # 因此，我们需要先将 NumPy 数组转换回 Pandas DataFrame
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=X_test.columns)
    
    # 使用 TabularModel 的 predict 方法进行预测
    # predict 方法返回一个字典，包含了 'prediction' 和 'probabilities' 等键
    # 深度学习模型
    prediction_result = model.predict(X)
    # SHAP 需要模型的原始输出（logits）或概率。
    # 对于分类任务，通常使用 'probabilities'
    # 对于回归任务，使用 'prediction'
    # 根据你的任务类型选择正确的键
    # 以下这行代码针对前4个pytorch模型使用
    return prediction_result["consciousness_prediction"]

def shap_summary_top_features(shap_values, X, top_n=2, show=True, save=True):
    """
    绘制前 top_n 个重要特征的 SHAP 汇总图，并按重要性排序。
    
    参数:
    - shap_values: SHAP 值数组，形状 (n_samples, n_features)
    - X: 原始特征矩阵 (DataFrame 或 ndarray)
    - top_n: 要显示的特征数量
    - show: 是否直接显示图形
    """
    # 如果是 DataFrame，提取特征名
    if isinstance(X, pd.DataFrame):
        feature_names = X.columns.tolist()
    else:
        feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
    feature_names = ['Eyes' if ele == 'open_one\'s_eyes' else ele for ele in feature_names]
    feature_names = ['Motor' if ele == 'motion' else ele for ele in feature_names]
    feature_names = ['Verbal' if ele == 'language' else ele for ele in feature_names]
    
    # 计算每个特征的 SHAP 重要性（绝对值均值）
    shap_importance = np.abs(shap_values).mean(axis=0)
    
    # 按重要性排序，取前 top_n 个特征的索引
    sorted_idx = np.argsort(shap_importance)[::-1][:top_n]
    # 筛选出对应的 SHAP 值和特征
    shap_values_top = shap_values[:, sorted_idx]
    X_top = X.iloc[:, sorted_idx] if isinstance(X, pd.DataFrame) else X[:, sorted_idx]
    feature_names_top = [feature_names[i] for i in sorted_idx]
    
    # colors = [
    #         (0.1, 0.2, 0.5),   # 深蓝（冷色，负贡献）
    #         (0.3, 0.4, 0.7),   # 中度蓝
    #         (0.7, 0.3, 0.3),   # 中度红
    #         (0.5, 0.1, 0.1)    # 深红（暖色，正贡献）
    #     ]

    # # 创建自定义 colormap
    # custom_coolwarm = LinearSegmentedColormap.from_list(
    #     "custom_coolwarm", colors, N=256
    # )
    
    # 绘制汇总图
    shap.summary_plot(
        shap_values_top,
        X_top,
        feature_names=feature_names_top,
        cmap='coolwarm',
        show=False
    )
    fig = plt.gcf()
    fig.set_size_inches(5, 4)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    if show:
        plt.show()
    if save:
        plt.tight_layout()
        # plt.savefig(r'experiment3_record\\exp1_image\\7_3_rf_minus1_abnormal_shap_summary.jpg', dpi=1000)
        # plt.savefig(r'experiment3_record\\exp1_image\\7_3_lgb_midim_abnormal_shap_summary.jpg', dpi=1000)
        plt.savefig(r'experiment3_record\\exp1_image\\7_3_rf_similarity_abnormal_shap_summary.jpg', dpi=1000)
    
if __name__ == '__main__':
    # 对实验1中的全量指标+缺项GCS进行SHAP分析 ==> 对测试集进行分析 ==> 解释模型为什么这样预测
    # 加载gcs训练集
    # df_train = pd.read_csv(r'data_base\\00_minus_db1\\full_assessment\\7-3\\train.csv')
    # model = pickle.load(open(r"model_pth\\exp_1\\00_minus_db1\\7-3\\seed_230\\rf_model.pkl", "rb"))
    # df_test = pd.read_csv(r'data_base\\00_minus_db1\\full_assessment\\7-3\\test.csv')
    # df_test = pd.read_csv(r'data_base\\00_minus_db1\\normal_assessment\\6_4_test_unfilled.csv', index_col=0)
    # df_test = pd.read_csv(r'data_base\\00_minus_db1\\abnormal_assessment\\6_4_test_filled.csv', index_col=0)
    
    # df_train = pd.read_csv(r'data_base\\01_midim_db\\full_assessment\\7-3\\train.csv')
    # model = pickle.load(open(r"model_pth\\exp_1\\01_midim_db\\7-3\\seed_230\\rf_model.pkl", "rb"))
    # df_test = pd.read_csv(r'data_base\\01_midim_db\\full_assessment\\7-3\\test.csv')
    # df_test = pd.read_csv(r'data_base\\01_midim_db\\normal_assessment\\7_3_test_unfilled.csv', index_col=0)
    # df_test = pd.read_csv(r'data_base\\01_midim_db\\abnormal_assessment\\7_3_test_filled.csv', index_col=0)
    
    df_train = pd.read_csv(r'data_base\\03_cos_similarity_db\\full_assessment\\7-3\\train.csv')
    model = pickle.load(open(r"model_pth\\exp_1\\03_cos_similarity_db\\7-3\\seed_230\\rf_model.pkl", "rb"))
    # df_test = pd.read_csv(r'data_base\\03_cos_similarity_db\\full_assessment\\7-3\\test.csv')
    # df_test = pd.read_csv(r'data_base\\03_cos_similarity_db\\normal_assessment\\7_3_test_unfilled.csv', index_col=0)
    df_test = pd.read_csv(r'data_base\\03_cos_similarity_db\\abnormal_assessment\\7_3_test_filled.csv', index_col=0)
    
    select_columns = ['INP_NO', 'Age', 'SEX', 'heart_rate', 'breathing', 'Blood_oxygen_saturation', 
                      'Blood_pressure_high', 'Blood_pressure_low', 'Left_pupil_size', 'Right_pupil_size',
                      'consciousness', 'ChartTime']
    X_train = df_train.drop(columns=select_columns)
    # 背景数据-解释数据：300-300，修改为背景数据-解释数据： 500-3000
    X_train_sample = shap.sample(X_train, 10000)  # 采样500个样本作为背景数据
    # X_test = df_test.iloc[:, -5:-1]
    # 先不去掉consciousness
    X_test = df_test.drop(columns=[x for x in select_columns if x != 'consciousness'])
    X_test_sample = shap.sample(X_test, 10000)  # 采样500个样本作为解释数据
    y_test_samaple = X_test_sample['consciousness']
    # X_test_sample去掉consciousness
    X_test_sample = X_test_sample.drop(columns=['consciousness'])
    # 计算模型预测的类别
    y_pred = model.predict(X_test_sample)
    # 统计真实标签和预测标签中不同类别的数量
    true_counts = np.bincount(y_test_samaple)
    pred_counts = np.bincount(y_pred)
    # 绘制预测值与真实值的对比图
    # plot_awakening_radar(true_counts, pred_counts)
    # 替换X_test_sample中的open_one's_eyes为Eyes
    X_test_sample = X_test_sample.rename(columns={'open_one\'s_eyes': 'Eyes'})
    # 替换X_test_sample中的motion为Motor
    X_test_sample = X_test_sample.rename(columns={'motion': 'Motor'})
    # 替换X_test_sample中的language为Verbal
    X_test_sample = X_test_sample.rename(columns={'language': 'Verbal'})
    # Random Forest
    explainer = shap.TreeExplainer(model, X_train_sample)
    # LightGBM
    # explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_sample)
    ###############################################################################
    shap_values = np.array(shap_values)  # 转换为NumPy数组，形状为 (类别数, 样本数, 特征数)
    # 根据特征选择shap值
    shap_values = shap_values[:, :, :3]
    X_test_sample = X_test_sample.iloc[:, :3]
    plt.figure(figsize=(5, 4))
    shap.summary_plot(
        shap_values[4],
        X_test_sample,
        cmap='coolwarm',
        show=False
    )
    fig = plt.gcf()
    fig.set_size_inches(5, 4)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    # plt.show()
    plt.savefig(r'experiment3_record\\exp1_image\\7_3_rf_similarity_abnormal_class4_shap_summary.jpg', dpi=1000)
    exit()
    # # 遍历y_pred, 根据y_pred保留每个特征在每个样本的SHAP值
    # shap_values = shap_values[y_pred, np.arange(len(y_pred)), :]
    # feature_shap_importance = np.abs(shap_values).mean(axis=0)
    # print(feature_shap_importance)
    # # 获取前3个最重要特征的索引
    # top3_indices = np.argsort(feature_shap_importance)[-3:]
    # top3_features = X_test_sample.columns[top3_indices]
    # # 获取前3个最重要特征的SHAP值
    # top3_shap_values = shap_values[:, top3_indices]
    # ###############################################################################
    # plt.figure(figsize=(5, 4))
    # shap.summary_plot(
    #     top3_shap_values,
    #     X_test_sample[top3_features],
    #     cmap='coolwarm',
    #     show=False
    # )
    # # shap_summary_top_features(shap_values, X_test_sample, top_n=3, show=True, save=False)
    # fig = plt.gcf()
    # fig.set_size_inches(5, 4)
    # plt.xticks(fontsize=16)
    # plt.yticks(fontsize=16)
    # # plt.savefig(r'exp_img\\exp1_all_metrics_7_3\\7-3_RF_shap_summary.jpg', dpi=1000)
    # plt.tight_layout()
    # plt.show()
    # exit()
    
    # # 模型1: 7-3-danet
    # model = pickle.load(open(r"model_pth\\exp1\\all_metrics\\7-3\\DANet.pkl", "rb"))
    # # # 这个地方使用训练集子集作为背景数据, 第二项为基准数据集|背景数据集
    # explainer = shap.PermutationExplainer(model_predict_function, X_train_sample)
    # shap_values = explainer(X_test_sample)
    # # # 使用测试集作为输入及特征，测试集为要解释的数据
    # shap.summary_plot(shap_values, X_test_sample, cmap="coolwarm", show=False)
    # plt.tight_layout()
    # plt.savefig(r'exp_img\\exp1_all_metrics_7_3\\7-3_danet_shap_summary.jpg', dpi=1000)
    # plt.close()
    
    # # 模型2: 7-3-tabnet
    # model = pickle.load(open(r"model_pth\\exp1\\all_metrics\\7-3\\TabNet.pkl", "rb"))
    # # # 这个地方使用训练集子集作为背景数据, 第二项为基准数据集|背景数据集
    # explainer = shap.PermutationExplainer(model_predict_function, X_train_sample)
    # shap_values = explainer(X_test_sample)
    # # # 使用测试集作为输入及特征，测试集为要解释的数据
    # shap.summary_plot(shap_values, X_test_sample, cmap="coolwarm", show=False)
    # plt.tight_layout()
    # plt.savefig(r'exp_img\\exp1_all_metrics_7_3\\7-3_tabnet_shap_summary.jpg', dpi=1000)
    # plt.close()
    
    # # 模型3: 7-3-fttransformer
    # model = pickle.load(open(r"model_pth\\exp1\\all_metrics\\7-3\\FTTransformer.pkl", "rb"))
    # # # 这个地方使用训练集子集作为背景数据, 第二项为基准数据集|背景数据集
    # explainer = shap.PermutationExplainer(model_predict_function, X_train_sample)
    # shap_values = explainer(X_test_sample)
    # # # 使用测试集作为输入及特征，测试集为要解释的数据
    # shap.summary_plot(shap_values, X_test_sample, cmap="coolwarm", show=False)
    # plt.tight_layout()
    # plt.savefig(r'exp_img\\exp1_all_metrics_7_3\\7-3_fttransformer_shap_summary.jpg', dpi=1000)
    # plt.close()
    
    # # 模型4： 7-3-tabtransformer
    # model = pickle.load(open(r"model_pth\\exp1\\all_metrics\\7-3\\TabTransformer.pkl", "rb"))
    # # # 这个地方使用训练集子集作为背景数据, 第二项为基准数据集|背景数据集
    # explainer = shap.PermutationExplainer(model_predict_function, X_train_sample)
    # shap_values = explainer(X_test_sample)
    # # # 使用测试集作为输入及特征，测试集为要解释的数据
    # shap.summary_plot(shap_values, X_test_sample, cmap="coolwarm", show=False)
    # plt.tight_layout()
    # plt.savefig(r'exp_img\\exp1_all_metrics_7_3\\7-3_tabtransformer_shap_summary.jpg', dpi=1000)
    # plt.close()
    # exit()
    
    # # 模型5： 7-3-lightgbm
    # model = pickle.load(open(r"model_pth\\exp1\\all_metrics\\7-3\\lgb.pkl", "rb"))
    # # # 这个地方使用训练集子集作为背景数据, 第二项为基准数据集|背景数据集
    # # explainer = shap.PermutationExplainer(model.predict, X_train_sample)
    # explainer = shap.TreeExplainer(model, X_train_sample)
    # shap_values = explainer.shap_values(X_test_sample)
    # shap_values = np.array(shap_values)  # 转换为NumPy数组，形状为 (类别数, 样本数, 特征数)
    # # 取 “每个样本最终预测类别” 对应的 SHAP 值（代表特征对 “最终预测结果” 的贡献）
    # y_pred = model.predict(X_test_sample)
    # # 原始shap_values维度是 (类别数, 样本数, 特征数)，需调整索引顺序为 (样本数, 类别数, 特征数)
    # shap_values_transposed = shap_values.transpose(1, 0, 2)  # 形状变为 (100, 5, 10)
    # # 按预测类别索引提取SHAP值，最终形状 (100, 10)
    # shap_values = shap_values_transposed[np.arange(len(y_pred)), y_pred, :]
    # # # 使用测试集作为输入及特征，测试集为要解释的数据
    # shap.summary_plot(shap_values, X_test_sample, cmap="coolwarm", show=False)
    # plt.tight_layout()
    # plt.savefig(r'exp_img\\exp1_all_metrics_7_3\\7-3_LightGBM_shap_summary.jpg', dpi=1000)
    # plt.close()
    
    # # 模型6： 7-3-xgboost
    # model = pickle.load(open(r"model_pth\\exp1\\all_metrics\\7-3\\xgboost.pkl", "rb"))
    # # # 这个地方使用训练集子集作为背景数据, 第二项为基准数据集|背景数据集
    # # explainer = shap.PermutationExplainer(model.predict, X_train_sample)
    # explainer = shap.TreeExplainer(model, X_train_sample)
    # shap_values = explainer.shap_values(X_test_sample)
    # shap_values = np.array(shap_values)  # 转换为NumPy数组，形状为 (类别数, 样本数, 特征数)
    # # 取 “每个样本最终预测类别” 对应的 SHAP 值（代表特征对 “最终预测结果” 的贡献）
    # y_pred = model.predict(X_test_sample)
    # # 原始shap_values维度是 (类别数, 样本数, 特征数)，需调整索引顺序为 (样本数, 类别数, 特征数)
    # shap_values_transposed = shap_values.transpose(1, 0, 2)  # 形状变为 (100, 5, 10)
    # # 按预测类别索引提取SHAP值，最终形状 (100, 10)
    # shap_values = shap_values_transposed[np.arange(len(y_pred)), y_pred, :]
    # # # 使用测试集作为输入及特征，测试集为要解释的数据
    # shap.summary_plot(shap_values, X_test_sample, cmap="coolwarm", show=False)
    # plt.tight_layout()
    # plt.savefig(r'exp_img\\exp1_all_metrics_7_3\\7-3_XGBoost_shap_summary.jpg', dpi=1000)
    # plt.close()
    
    # 模型7： 7-3-randomforest
    model = pickle.load(open(r"model_pth\\exp_1\\03_cos_similarity_db\\7-3\\seed_230\\rf_model.pkl", "rb"))
    # # 绘制特征重要性柱状图
    # importances = model.feature_importances_
    # # 4. 按重要性排序
    # indices = np.argsort(importances)[::-1]
    # sorted_features = [X_train.columns[i] for i in indices]
    # sorted_importances = [importances[i] for i in indices]
    # sorted_features = sorted_features[:3]
    # sorted_importances = sorted_importances[:3]
    # sorted_features.reverse()
    # sorted_importances.reverse()
    # # 6. 顶刊配色（Nature 常用配色）
    # colors = ['#1f2937', '#374151', '#4b5563']  # 深色系
    # # 7. 绘制水平条形图
    # plt.figure(figsize=(8, 4))
    # bars = plt.barh(range(len(sorted_features)), sorted_importances, color=colors)

    # # 添加数值标签
    # for bar in bars:
    #     width = bar.get_width()
    #     plt.text(width + 0.005, bar.get_y() + bar.get_height()/2,
    #             f'{width:.4f}', ha='left', va='center')
    # plt.barh(range(len(sorted_features)), sorted_importances, color=colors)
    # # 设置 x 轴范围
    # plt.xlim(0, max(sorted_importances) * 1.2)  # 留出空间给数值标签
    # plt.yticks(range(len(sorted_features)), sorted_features)
    # plt.xlabel("Feature Importance")
    # plt.ylabel("Feature")
    # # plt.tight_layout()
    # # plt.show()
    # # 保存为pdf
    # # plt.savefig(r'experiment3_record\\exp1_image\\7-3_RF_origin_full_summary.pdf', dpi=1000)
    # # plt.close()
    # exit()
    # # 这个地方使用训练集子集作为背景数据, 第二项为基准数据集|背景数据集
    # explainer = shap.PermutationExplainer(model.predict, X_train_sample)
    explainer = shap.TreeExplainer(model, X_train_sample)
    shap_values = explainer.shap_values(X_test_sample)
    shap_values = np.array(shap_values)  # 转换为NumPy数组，形状为 (类别数, 样本数, 特征数)
    # 取 “每个样本最终预测类别” 对应的 SHAP 值（代表特征对 “最终预测结果” 的贡献）
    y_pred = model.predict(X_test_sample)
    # 原始shap_values维度是 (类别数, 样本数, 特征数)，需调整索引顺序为 (样本数, 类别数, 特征数)
    shap_values_transposed = shap_values.transpose(1, 0, 2)  # 形状变为 (100, 5, 10)
    # 按预测类别索引提取SHAP值，最终形状 (100, 10)
    shap_values = shap_values_transposed[np.arange(len(y_pred)), y_pred, :]
    # # 使用测试集作为输入及特征，测试集为要解释的数据
    shap.summary_plot(shap_values, X_test_sample, cmap="coolwarm", max_display=3, show=False)
    plt.tight_layout()
    # plt.savefig(r'exp_img\\exp1_all_metrics_7_3\\7-3_RF_shap_summary.jpg', dpi=1000)
    # plt.close()
    plt.show()
    
    # # 模型8： 7-3-CatBoost
    # model = pickle.load(open(r"model_pth\\exp1\\all_metrics\\7-3\\catboost.pkl", "rb"))
    # # # 这个地方使用训练集子集作为背景数据, 第二项为基准数据集|背景数据集
    # explainer = shap.PermutationExplainer(model.predict, X_train_sample)
    # shap_values = explainer(X_test_sample)
    # # 使用测试集作为输入及特征，测试集为要解释的数据
    # shap.summary_plot(shap_values, X_test_sample, cmap="coolwarm", show=False)
    # plt.tight_layout()
    # plt.savefig(r'exp_img\\exp1_all_metrics_7_3\\7-3_catboost_shap_summary.jpg', dpi=1000)
    # plt.close()
    
    # # 模型9： 7-3-MLP
    # model = pickle.load(open(r"model_pth\\exp1\\all_metrics\\7-3\\mlp.pkl", "rb"))
    # # # 这个地方使用训练集子集作为背景数据, 第二项为基准数据集|背景数据集
    # explainer = shap.PermutationExplainer(model.predict, X_train_sample)
    # shap_values = explainer(X_test_sample)
    # # 使用测试集作为输入及特征，测试集为要解释的数据
    # shap.summary_plot(shap_values, X_test_sample, cmap="coolwarm", show=False)
    # plt.tight_layout()
    # plt.savefig(r'exp_img\\exp1_all_metrics_7_3\\7-3_mlp_shap_summary.jpg', dpi=1000)
    # plt.close()
