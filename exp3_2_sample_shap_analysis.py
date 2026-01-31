import shap
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
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

def plot_pred_true_comparison(true_labels, pred_labels):
    """
    绘制真实标签 vs 预测标签的对比图
    正确预测用绿点(○)标记，错误预测用红叉(×)标记
    
    Parameters:
    -----------
    true_labels : array-like (list/numpy array)
        真实标签数组（可以是整数/字符串类型）
    pred_labels : array-like (list/numpy array)
        预测标签数组（需与真实标签长度、类型一致）
    
    Returns:
    --------
    None (直接显示图表，并打印统计信息)
    """
    # ===================== 1. 参数校验 =====================
    # 转换为numpy数组，方便后续操作
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)
    
    # 检查长度是否一致
    if len(true_labels) != len(pred_labels):
        raise ValueError("Error: true_labels and pred_labels must have the same length!")
    
    # 检查是否为空
    if len(true_labels) == 0:
        raise ValueError("Error: Input arrays cannot be empty!")
    
    # ===================== 2. 处理标签（兼容字符串/数字） =====================
    # 获取所有唯一标签，用于坐标轴刻度
    all_labels = np.unique(np.concatenate([true_labels, pred_labels]))
    # 创建标签到索引的映射（兼容字符串标签）
    label_to_idx = {label: idx for idx, label in enumerate(all_labels)}
    
    # 将标签转换为索引（方便绘图）
    true_idx = np.array([label_to_idx[label] for label in true_labels])
    pred_idx = np.array([label_to_idx[label] for label in pred_labels])
    
    # ===================== 3. 绘图设置 =====================
    plt.figure(figsize=(6, 6))
    # 重置matplotlib字体（确保英文显示正常）
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # ===================== 4. 绘制标记点 =====================
    for t, p, t_label, p_label in zip(true_idx, pred_idx, true_labels, pred_labels):
        if t_label == p_label:
            # 正确预测：绿点（带黑色边框更醒目）
            plt.scatter(t, p, color='green', marker='o', s=120, 
                        edgecolors='black', linewidth=1, zorder=3)
        else:
            # 错误预测：红叉（加粗线条）
            plt.scatter(t, p, color='red', marker='x', s=120, 
                        linewidth=2, zorder=3)
    
    # ===================== 5. 图表美化 =====================
    # 添加完美预测对角线
    min_idx = 0
    max_idx = len(all_labels) - 1
    plt.plot([min_idx, max_idx], [min_idx, max_idx], 'k--', alpha=0.5, label='Perfect Prediction Line')
    
    # 设置坐标轴
    plt.xlabel('True Labels', fontsize=14, fontweight='medium')
    plt.ylabel('Predicted Labels', fontsize=14, fontweight='medium')
    
    # 设置刻度（显示原始标签，而非索引）
    plt.xticks(range(len(all_labels)), all_labels, ha='right')
    plt.yticks(range(len(all_labels)), all_labels)
    
    # 添加网格、图例、调整布局
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left', fontsize=13)
    plt.tight_layout()
    
    # 显示图表
    plt.savefig(r'experiment3_record\\exp2_image\\pred_true_comparison.jpg', dpi=1000, bbox_inches='tight')
    
    # ===================== 6. 输出统计信息 =====================
    correct = np.sum(true_labels == pred_labels)
    total = len(true_labels)
    accuracy = correct / total
    
    print("="*50)
    print(f"Total samples: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Incorrect predictions: {total - correct}")
    print(f"Prediction accuracy: {accuracy:.4f} ({accuracy:.2%})")
    print("="*50)

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

def shap_summary_top_features(shap_values, X, top_n=2, show=True):
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
    feature_names = ['Language' if ele == 'language' else ele for ele in feature_names]
    
    # 计算每个特征的 SHAP 重要性（绝对值均值）
    shap_importance = np.abs(shap_values).mean(axis=0)
    
    # 按重要性排序，取前 top_n 个特征的索引
    sorted_idx = np.argsort(shap_importance)[::-1][:top_n]
    
    # 筛选出对应的 SHAP 值和特征
    shap_values_top = shap_values[:, sorted_idx]
    X_top = X.iloc[:, sorted_idx] if isinstance(X, pd.DataFrame) else X[:, sorted_idx]
    feature_names_top = [feature_names[i] for i in sorted_idx]
    
    # 绘制汇总图
    shap.summary_plot(
        shap_values_top,
        X_top,
        feature_names=feature_names_top,
        show=False,
        cmap="coolwarm"
    )
    
    plt.tight_layout()
    if show:
        plt.show()
    
if __name__ == '__main__':
    df_train = pd.read_csv(r'data_base\\03_cos_similarity_db\\full_assessment\\7-3\\train.csv')
    df_test = pd.read_csv(r'data_base\\03_cos_similarity_db\\full_assessment\\7-3\\test.csv')
    # df_test = pd.read_csv(r'data_base\\03_cos_similarity_db\\normal_assessment\\7_3_test_unfilled.csv')
    select_columns = ['INP_NO', 'Age', 'SEX', 'heart_rate', 'breathing', 'Blood_oxygen_saturation', 
                      'Blood_pressure_high', 'Blood_pressure_low', 'Left_pupil_size', 'Right_pupil_size', 'language', 'language_filled',
                      'consciousness', 'ChartTime']
    X_train = df_train.drop(columns=select_columns)
    # X_train = df_train.iloc[:, -5:-1]
    # 背景数据-解释数据：300-300，修改为背景数据-解释数据： 500-3000
    X_train_sample = shap.sample(X_train, 1000)  # 采样200个样本作为背景数据
    # X_test = df_test.iloc[:, -5:-1]
    X_test = df_test.drop(columns=[x for x in select_columns if x != 'consciousness'])
    # X_test_sample = shap.sample(X_test, 3)  # 采样500个样本作为解释数据
    # 从consciousness分别为0，1，2，3，4中选择5个样本
    X_test_sample = pd.concat([X_test[X_test['consciousness'] == i].sample(2, random_state=0) for i in range(5)], axis=0)
    y_test_sample = X_test_sample['consciousness']
    print('选择的样本数据：')
    print(X_test_sample)
    # X_test_sample去掉consciousness
    X_test_sample = X_test_sample.drop(columns=['consciousness'])
    model = pickle.load(open(r"model_pth\\exp_2\\02_similarity\\03_eyes_motion\\7-3\\seed_230\\rf_model.pkl", "rb"))
    model_pred = model.predict(X_test_sample)
    print('模型预测结果：')
    print(model_pred)
    # 统计真实标签和预测标签中不同类别的数量
    # plot_pred_true_comparison(y_test_sample, model_pred)
    # 获取错误预测的样本索引
    misclassifid = np.where(model_pred != y_test_sample)[0]
    # plot_label_comparison(y_test_sample, model_pred)
    # true_counts = np.bincount(y_test_sample)
    # pred_counts = np.bincount(model_pred)
    # plot_awakening_radar(pred_counts, true_counts)
    # 替换X_test_sample中的open_one's_eyes为Eyes
    X_test_sample = X_test_sample.rename(columns={'open_one\'s_eyes': 'Eyes'})
    # 替换X_test_sample中的motion为Motor
    X_test_sample = X_test_sample.rename(columns={'motion': 'Motor'})
    selected_class = 4
    
    # # 这个地方使用训练集子集作为背景数据, 第二项为基准数据集|背景数据集
    explainer = shap.TreeExplainer(model, X_train_sample)
    shap_values = explainer.shap_values(X_test_sample)
    shap_values = np.array(shap_values)
    
    # 选择特征
    # 选择你关心的两个特征
    selected_features = ["Eyes", "Motor"]
    expected_value = explainer.expected_value[selected_class]
    # 从特征矩阵中筛选
    X_selected = X_test_sample[selected_features]
    # 从 SHAP 值中筛选（根据特征名的位置）
    feature_indices = [list(X_test_sample.columns).index(f) for f in selected_features]
    shap_values_selected = shap_values[:, :, feature_indices]
    shap.decision_plot(
        expected_value,
        shap_values_selected[selected_class],
        features=X_selected,
        highlight=misclassifid,
        feature_names=['Eyes', 'Motor'],
        show=False
    )
    plt.tight_layout()
    # plt.show()
    exit()
    plt.savefig(r'experiment3_record\\exp2_image\\7_3_rf_class_4.jpg', dpi=1000)
    # plt.close()
    exit()
    # # 模型1: 7-3-danet
    # model = pickle.load(open(r"model_pth\\exp2\\vitials_eyes_motion\\7-3\\DANet.pkl", "rb"))
    # # # 这个地方使用训练集子集作为背景数据, 第二项为基准数据集|背景数据集
    # explainer = shap.PermutationExplainer(model_predict_function, X_train_sample)
    # shap_values = explainer(X_test_sample)
    # # # 使用测试集作为输入及特征，测试集为要解释的数据
    # shap.summary_plot(shap_values, X_test_sample, cmap="coolwarm", show=False)
    # plt.tight_layout()
    # plt.savefig(r'exp_img\\exp2_eyes_motion_7_3\\7-3_danet_shap_summary.jpg', dpi=1000)
    # plt.close()
    
    # # 模型2: 7-3-tabnet
    # model = pickle.load(open(r"model_pth\\exp2\\vitials_eyes_motion\\7-3\\TabNet.pkl", "rb"))
    # # # 这个地方使用训练集子集作为背景数据, 第二项为基准数据集|背景数据集
    # explainer = shap.PermutationExplainer(model_predict_function, X_train_sample)
    # shap_values = explainer(X_test_sample)
    # # # 使用测试集作为输入及特征，测试集为要解释的数据
    # shap.summary_plot(shap_values, X_test_sample, cmap="coolwarm", show=False)
    # plt.tight_layout()
    # plt.savefig(r'exp_img\\exp2_eyes_motion_7_3\\7-3_tabnet_shap_summary.jpg', dpi=1000)
    # plt.close()
    
    # # 模型3: 7-3-fttransformer
    # model = pickle.load(open(r"model_pth\\exp2\\vitials_eyes_motion\\7-3\\FTTransformer.pkl", "rb"))
    # # # 这个地方使用训练集子集作为背景数据, 第二项为基准数据集|背景数据集
    # explainer = shap.PermutationExplainer(model_predict_function, X_train_sample)
    # shap_values = explainer(X_test_sample)
    # # # 使用测试集作为输入及特征，测试集为要解释的数据
    # shap.summary_plot(shap_values, X_test_sample, cmap="coolwarm", show=False)
    # plt.tight_layout()
    # plt.savefig(r'exp_img\\exp2_eyes_motion_7_3\\7-3_fttransformer_shap_summary.jpg', dpi=1000)
    # plt.close()
    
    model = pickle.load(open(r"model_pth\\exp_2\\02_similarity\\03_eyes_motion\\6-4\\seed_230\\rf_model.pkl", "rb"))
    # # 这个地方使用训练集子集作为背景数据, 第二项为基准数据集|背景数据集
    explainer = shap.TreeExplainer(model, X_train_sample)
    shap_values = explainer.shap_values(X_test_sample)
    # 睁眼评分
    shap.summary_plot(shap_values[4], X_test_sample, cmap="coolwarm", show=False, max_display=2)
    # shap_values = np.array(shap_values)  # 转换为NumPy数组，形状为 (类别数, 样本数, 特征数)
    # # 取 “每个样本最终预测类别” 对应的 SHAP 值（代表特征对 “最终预测结果” 的贡献）
    # y_pred = model.predict(X_test_sample)
    # # 原始shap_values维度是 (类别数, 样本数, 特征数)，需调整索引顺序为 (样本数, 类别数, 特征数)
    # shap_values_transposed = shap_values.transpose(1, 0, 2)  # 形状变为 (100, 5, 10)
    # # 按预测类别索引提取SHAP值，最终形状 (100, 10)
    # shap_values = shap_values_transposed[np.arange(len(y_pred)), y_pred, :]
    # # # 使用测试集作为输入及特征，测试集为要解释的数据
    # # shap.summary_plot(shap_values, X_test_sample, cmap="coolwarm", max_display=2, show=False)
    # shap.dependence_plot(
    #     ind=0,  # 第0个特征（按SHAP重要性排序）
    #     shap_values=shap_values,
    #     features=X_test_sample
    # )
    plt.tight_layout()
    # plt.savefig(r'exp_img\\exp1_all_metrics_7_3\\7-3_RF_shap_summary.jpg', dpi=1000)
    # plt.close()
    plt.show()
    
    # plt.savefig(r'exp_img\\exp2_eyes_motion_7_3\\7-3_tabtransformer_shap_summary.jpg', dpi=1000)
    # plt.close()
    exit()
    
    # # 模型5： 7-3-lightgbm
    # model = pickle.load(open(r"model_pth\\exp2\\vitials_eyes_motion\\7-3\\lgb.pkl", "rb"))
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
    # plt.savefig(r'exp_img\\exp2_eyes_motion_7_3\\7-3_LightGBM_shap_summary.jpg', dpi=1000)
    # plt.close()
    
    # # 模型6： 7-3-xgboost
    # model = pickle.load(open(r"model_pth\\exp2\\vitials_eyes_motion\\7-3\\xgboost.pkl", "rb"))
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
    # plt.savefig(r'exp_img\\exp2_eyes_motion_7_3\\7-3_XGBoost_shap_summary.jpg', dpi=1000)
    # plt.close()
    
    # # 模型7： 7-3-randomforest
    # model = pickle.load(open(r"model_pth\\exp2\\vitials_eyes_motion\\7-3\\rf_model.pkl", "rb"))
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
    # plt.savefig(r'exp_img\\exp2_eyes_motion_7_3\\7-3_RF_shap_summary.jpg', dpi=1000)
    # plt.close()
    
    # # 模型8： 7-3-CatBoost
    # model = pickle.load(open(r"model_pth\\exp2\\vitials_eyes_motion\\7-3\\catboost.pkl", "rb"))
    # # # 这个地方使用训练集子集作为背景数据, 第二项为基准数据集|背景数据集
    # explainer = shap.PermutationExplainer(model.predict, X_train_sample)
    # shap_values = explainer(X_test_sample)
    # # 使用测试集作为输入及特征，测试集为要解释的数据
    # shap.summary_plot(shap_values, X_test_sample, cmap="coolwarm", show=False)
    # plt.tight_layout()
    # plt.savefig(r'exp_img\\exp2_eyes_motion_7_3\\7-3_catboost_shap_summary.jpg', dpi=1000)
    # plt.close()
    
    # # 模型9： 7-3-MLP
    # model = pickle.load(open(r"model_pth\\exp2\\vitials_eyes_motion\\7-3\\mlp.pkl", "rb"))
    # # # 这个地方使用训练集子集作为背景数据, 第二项为基准数据集|背景数据集
    # explainer = shap.PermutationExplainer(model.predict, X_train_sample)
    # shap_values = explainer(X_test_sample)
    # # 使用测试集作为输入及特征，测试集为要解释的数据
    # shap.summary_plot(shap_values, X_test_sample, cmap="coolwarm", show=False)
    # plt.tight_layout()
    # plt.savefig(r'exp_img\\exp2_eyes_motion_7_3\\7-3_mlp_shap_summary.jpg', dpi=1000)
    # plt.close()
