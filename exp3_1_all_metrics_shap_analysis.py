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

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 18

# Custom Green-White-Red Gradient (Adjustable Color Intensity)
colors = ["#27AE60", "#FFFFFF", "#E74C3C"]  # Dark green → White → Dark red
n_bins = 100  # Gradient Smoothness
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
    # Expand the numerical range to allow space for value annotations.
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

    # Define annotation style parameters
    fontsize = 12
    true_color = '#1f77b4'  # True Color
    pred_color = '#ff7f0e'  # Predicted Value Color
    offset = 0.05 * (vmax - vmin)  # Offset numerical annotations to avoid overlapping with lines.

    # Add ground truth labels for each dimension
    for angle, value in zip(theta, true_labels):
        # The annotation position is slightly offset from the actual value to avoid obstruction.
        ax.text(angle, value - offset, f'{value:.2f}', 
                ha='center', va='center', fontsize=fontsize, color=true_color,
                fontweight='bold')
    
    # Add prediction labels for each dimension (with larger offsets to avoid overlapping with true value labels)
    for angle, value in zip(theta, pred_labels):
        ax.text(angle, value + 2*offset, f'{value:.2f}', 
                ha='center', va='center', fontsize=fontsize, color=pred_color,
                fontweight='bold')

    # Set legend and title
    ax.legend(loc='upper left', fontsize=13)
    plt.tight_layout()
    # Display the chart
    plt.show()

# 2. Define a prediction wrapper function
def model_predict_function(X):
    """
    A wrapper function that adapts TabularModel to SHAP's interface.
    Parameters:
    X (np.ndarray or pd.DataFrame): Feature data. SHAP passes in a NumPy array.
    Returns:
    np.ndarray: The model's predicted probabilities or raw outputs.
    """
    # SHAP's KernelExplainer passes NumPy arrays, while TabularModel's predict method requires a DataFrame.
    # Therefore, we first need to convert the NumPy array back into a Pandas DataFrame.
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=X_test.columns)
    
    # # Use the predict method of TabularModel for prediction
    # The predict method returns a dictionary containing keys such as ‘prediction’ and ‘probabilities’.
    # Deep Learning Model
    prediction_result = model.predict(X)
    # SHAP requires the model's raw outputs (logits) or probabilities.
    # For classification tasks, ‘probabilities’ are typically used.
    # For regression tasks, use ‘prediction’.
    return prediction_result["consciousness_prediction"]

def shap_summary_top_features(shap_values, X, top_n=2, show=True, save=True):
    """
    Plot a summary SHAP map of the top_n most important features, sorted by importance.
    
    Parameters:
    - shap_values: SHAP value array, shape (n_samples, n_features)
    - X: Raw feature matrix (DataFrame or ndarray)
    - top_n: Number of features to display
    - show: Whether to display the graph directly
    """
    # If it is a DataFrame, extract feature names.
    if isinstance(X, pd.DataFrame):
        feature_names = X.columns.tolist()
    else:
        feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
    feature_names = ['Eyes' if ele == 'open_one\'s_eyes' else ele for ele in feature_names]
    feature_names = ['Motor' if ele == 'motion' else ele for ele in feature_names]
    feature_names = ['Verbal' if ele == 'language' else ele for ele in feature_names]
    
    # Compute the SHAP importance (mean absolute value) for each feature.
    shap_importance = np.abs(shap_values).mean(axis=0)
    
    # Sort by importance and take the indices of the top_n features.
    sorted_idx = np.argsort(shap_importance)[::-1][:top_n]
    # Filter out the corresponding SHAP values and features
    shap_values_top = shap_values[:, sorted_idx]
    X_top = X.iloc[:, sorted_idx] if isinstance(X, pd.DataFrame) else X[:, sorted_idx]
    feature_names_top = [feature_names[i] for i in sorted_idx]
    
    # colors = [
    #         (0.1, 0.2, 0.5),   
    #         (0.3, 0.4, 0.7),   
    #         (0.7, 0.3, 0.3),   
    #         (0.5, 0.1, 0.1)    
    #     ]

    # custom_coolwarm = LinearSegmentedColormap.from_list(
    #     "custom_coolwarm", colors, N=256
    # )
    
    # Create a summary chart
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
    X_train_sample = shap.sample(X_train, 10000)  
    # X_test = df_test.iloc[:, -5:-1]
    X_test = df_test.drop(columns=[x for x in select_columns if x != 'consciousness'])
    X_test_sample = shap.sample(X_test, 10000) 
    y_test_samaple = X_test_sample['consciousness']
    X_test_sample = X_test_sample.drop(columns=['consciousness'])
    y_pred = model.predict(X_test_sample)
    # Count the number of different categories in the actual labels and predicted labels.
    true_counts = np.bincount(y_test_samaple)
    pred_counts = np.bincount(y_pred)
    # Plot a comparison chart of predicted values versus actual values
    # plot_awakening_radar(true_counts, pred_counts)
    # Replace “open_one's_eyes” with “Eyes” in X_test_sample.
    X_test_sample = X_test_sample.rename(columns={'open_one\'s_eyes': 'Eyes'})
    # Replace “motion” with “Motor” in X_test_sample.
    X_test_sample = X_test_sample.rename(columns={'motion': 'Motor'})
    # Replace the language in X_test_sample with Verbal
    X_test_sample = X_test_sample.rename(columns={'language': 'Verbal'})
    # Random Forest
    explainer = shap.TreeExplainer(model, X_train_sample)
    # LightGBM
    # explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_sample)
    ###############################################################################
    shap_values = np.array(shap_values)  # Convert to a NumPy array with shape (number of classes, number of samples, number of features)
    # Selecting SHAP values based on features
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
    # # Iterate through y_pred, retaining the SHAP value for each feature in each sample based on y_pred.
    # shap_values = shap_values[y_pred, np.arange(len(y_pred)), :]
    # feature_shap_importance = np.abs(shap_values).mean(axis=0)
    # print(feature_shap_importance)
    # # Obtain the indices of the top 3 most important features
    # top3_indices = np.argsort(feature_shap_importance)[-3:]
    # top3_features = X_test_sample.columns[top3_indices]
    # # Obtain the SHAP values for the top 3 most important features
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
    
    # # Model 1: 7-3-danet
    # model = pickle.load(open(r"model_pth\\exp1\\all_metrics\\7-3\\DANet.pkl", "rb"))
    # explainer = shap.PermutationExplainer(model_predict_function, X_train_sample)
    # shap_values = explainer(X_test_sample)
    # shap.summary_plot(shap_values, X_test_sample, cmap="coolwarm", show=False)
    # plt.tight_layout()
    # plt.savefig(r'exp_img\\exp1_all_metrics_7_3\\7-3_danet_shap_summary.jpg', dpi=1000)
    # plt.close()
    
    # # Model 2: 7-3-tabnet
    # model = pickle.load(open(r"model_pth\\exp1\\all_metrics\\7-3\\TabNet.pkl", "rb"))
    # explainer = shap.PermutationExplainer(model_predict_function, X_train_sample)
    # shap_values = explainer(X_test_sample)
    # shap.summary_plot(shap_values, X_test_sample, cmap="coolwarm", show=False)
    # plt.tight_layout()
    # plt.savefig(r'exp_img\\exp1_all_metrics_7_3\\7-3_tabnet_shap_summary.jpg', dpi=1000)
    # plt.close()
    
    # # Model 3: 7-3-fttransformer
    # model = pickle.load(open(r"model_pth\\exp1\\all_metrics\\7-3\\FTTransformer.pkl", "rb"))
    # explainer = shap.PermutationExplainer(model_predict_function, X_train_sample)
    # shap_values = explainer(X_test_sample)
    # shap.summary_plot(shap_values, X_test_sample, cmap="coolwarm", show=False)
    # plt.tight_layout()
    # plt.savefig(r'exp_img\\exp1_all_metrics_7_3\\7-3_fttransformer_shap_summary.jpg', dpi=1000)
    # plt.close()
    
    # # Model 4： 7-3-tabtransformer
    # model = pickle.load(open(r"model_pth\\exp1\\all_metrics\\7-3\\TabTransformer.pkl", "rb"))
    # explainer = shap.PermutationExplainer(model_predict_function, X_train_sample)
    # shap_values = explainer(X_test_sample)
    # shap.summary_plot(shap_values, X_test_sample, cmap="coolwarm", show=False)
    # plt.tight_layout()
    # plt.savefig(r'exp_img\\exp1_all_metrics_7_3\\7-3_tabtransformer_shap_summary.jpg', dpi=1000)
    # plt.close()
    # exit()
    
    # # Model 5： 7-3-lightgbm
    # model = pickle.load(open(r"model_pth\\exp1\\all_metrics\\7-3\\lgb.pkl", "rb"))
    # # explainer = shap.PermutationExplainer(model.predict, X_train_sample)
    # explainer = shap.TreeExplainer(model, X_train_sample)
    # shap_values = explainer.shap_values(X_test_sample)
    # shap_values = np.array(shap_values)  # Convert to a NumPy array with shape (number of classes, number of samples, number of features)
    # # Obtain the SHAP value corresponding to the “final predicted category for each sample” (representing the feature's contribution to the “final prediction result”).
    # y_pred = model.predict(X_test_sample)
    # # The original shap_values dimensions are (number of classes, number of samples, number of features), requiring adjustment to the index order: (number of samples, number of classes, number of features).
    # shap_values_transposed = shap_values.transpose(1, 0, 2) 
    # # Indexing SHAP values by prediction category
    # shap_values = shap_values_transposed[np.arange(len(y_pred)), y_pred, :]
    # shap.summary_plot(shap_values, X_test_sample, cmap="coolwarm", show=False)
    # plt.tight_layout()
    # plt.savefig(r'exp_img\\exp1_all_metrics_7_3\\7-3_LightGBM_shap_summary.jpg', dpi=1000)
    # plt.close()
    
    # # Model 6： 7-3-xgboost
    # model = pickle.load(open(r"model_pth\\exp1\\all_metrics\\7-3\\xgboost.pkl", "rb"))
    # # explainer = shap.PermutationExplainer(model.predict, X_train_sample)
    # explainer = shap.TreeExplainer(model, X_train_sample)
    # shap_values = explainer.shap_values(X_test_sample)
    # shap_values = np.array(shap_values)  # Convert to a NumPy array with shape (number of classes, number of samples, number of features)
    # # Take the SHAP value corresponding to the “final predicted category for each sample” (representing the feature's contribution to the “final prediction result”).
    # y_pred = model.predict(X_test_sample)
    # # The original shap_values dimensions are (number of classes, number of samples, number of features), requiring adjustment to the index order: (number of samples, number of classes, number of features).
    # shap_values_transposed = shap_values.transpose(1, 0, 2)
    # # Indexing SHAP values by prediction category
    # shap_values = shap_values_transposed[np.arange(len(y_pred)), y_pred, :]
    # shap.summary_plot(shap_values, X_test_sample, cmap="coolwarm", show=False)
    # plt.tight_layout()
    # plt.savefig(r'exp_img\\exp1_all_metrics_7_3\\7-3_XGBoost_shap_summary.jpg', dpi=1000)
    # plt.close()
    
    # Model 7： 7-3-randomforest
    model = pickle.load(open(r"model_pth\\exp_1\\03_cos_similarity_db\\7-3\\seed_230\\rf_model.pkl", "rb"))
    # importances = model.feature_importances_
    # indices = np.argsort(importances)[::-1]
    # sorted_features = [X_train.columns[i] for i in indices]
    # sorted_importances = [importances[i] for i in indices]
    # sorted_features = sorted_features[:3]
    # sorted_importances = sorted_importances[:3]
    # sorted_features.reverse()
    # sorted_importances.reverse()
    # colors = ['#1f2937', '#374151', '#4b5563']
    # plt.figure(figsize=(8, 4))
    # bars = plt.barh(range(len(sorted_features)), sorted_importances, color=colors)

    # for bar in bars:
    #     width = bar.get_width()
    #     plt.text(width + 0.005, bar.get_y() + bar.get_height()/2,
    #             f'{width:.4f}', ha='left', va='center')
    # plt.barh(range(len(sorted_features)), sorted_importances, color=colors)
    # plt.xlim(0, max(sorted_importances) * 1.2)  # Leave space for numerical labels
    # plt.yticks(range(len(sorted_features)), sorted_features)
    # plt.xlabel("Feature Importance")
    # plt.ylabel("Feature")
    # # plt.tight_layout()
    # # plt.show()
    # # plt.savefig(r'experiment3_record\\exp1_image\\7-3_RF_origin_full_summary.pdf', dpi=1000)
    # # plt.close()
    # exit()
    
    # explainer = shap.PermutationExplainer(model.predict, X_train_sample)
    explainer = shap.TreeExplainer(model, X_train_sample)
    shap_values = explainer.shap_values(X_test_sample)
    shap_values = np.array(shap_values)  # Convert to a NumPy array with shape (number of classes, number of samples, number of features)
    # Take the SHAP value corresponding to the “final predicted category for each sample” (representing the feature's contribution to the “final prediction result”).
    y_pred = model.predict(X_test_sample)
    # The original shap_values dimensions are (number of classes, number of samples, number of features), requiring adjustment to the index order: (number of samples, number of classes, number of features).
    shap_values_transposed = shap_values.transpose(1, 0, 2)
    # Indexing SHAP values by prediction category
    shap_values = shap_values_transposed[np.arange(len(y_pred)), y_pred, :]
    shap.summary_plot(shap_values, X_test_sample, cmap="coolwarm", max_display=3, show=False)
    plt.tight_layout()
    # plt.savefig(r'exp_img\\exp1_all_metrics_7_3\\7-3_RF_shap_summary.jpg', dpi=1000)
    # plt.close()
    plt.show()
    
    # # Model 8： 7-3-CatBoost
    # model = pickle.load(open(r"model_pth\\exp1\\all_metrics\\7-3\\catboost.pkl", "rb"))
    # explainer = shap.PermutationExplainer(model.predict, X_train_sample)
    # shap_values = explainer(X_test_sample)
    # shap.summary_plot(shap_values, X_test_sample, cmap="coolwarm", show=False)
    # plt.tight_layout()
    # plt.savefig(r'exp_img\\exp1_all_metrics_7_3\\7-3_catboost_shap_summary.jpg', dpi=1000)
    # plt.close()
    
    # # Model 9： 7-3-MLP
    # model = pickle.load(open(r"model_pth\\exp1\\all_metrics\\7-3\\mlp.pkl", "rb"))
    # explainer = shap.PermutationExplainer(model.predict, X_train_sample)
    # shap_values = explainer(X_test_sample)
    # shap.summary_plot(shap_values, X_test_sample, cmap="coolwarm", show=False)
    # plt.tight_layout()
    # plt.savefig(r'exp_img\\exp1_all_metrics_7_3\\7-3_mlp_shap_summary.jpg', dpi=1000)
    # plt.close()
