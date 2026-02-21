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

# Set the global font to Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 18

# Custom green-white-red gradient (adjustable color intensity)
colors = ["#27AE60", "#FFFFFF", "#E74C3C"]
n_bins = 100  # gradient smoothness
cmap_green_white_red = LinearSegmentedColormap.from_list(
    "GreenWhiteRed", colors, N=n_bins
)

def plot_pred_true_comparison(true_labels, pred_labels):
    """
    Plot a comparison of true labels vs. predicted labels
    Correct predictions marked with green dots(○),
    incorrect predictions marked with red corsses(×).
    
    Parameters:
    -----------
    true_labels : array-like (list/numpy array)
    pred_labels : array-like (list/numpy array)
    
    Returns:
    --------
    None (Display charts directly and print statistical information)
    """
    # ===================== 1. Parameter Validation =====================
    # Convert to a NumPy array for easier subsequent operations
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)
    
    # Check if the lengths are consistent
    if len(true_labels) != len(pred_labels):
        raise ValueError("Error: true_labels and pred_labels must have the same length!")
    
    # Check if it is empty
    if len(true_labels) == 0:
        raise ValueError("Error: Input arrays cannot be empty!")
    
    # ===================== 2. Processing Tags (Compatible with Strings/Numbers) =====================
    # Retrieve all unique labels for axis tick marks
    all_labels = np.unique(np.concatenate([true_labels, pred_labels]))
    # Create a mapping from tags to indexes (compatible with string tags)
    label_to_idx = {label: idx for idx, label in enumerate(all_labels)}
    
    # Convert labels to indices (for easier plotting)
    true_idx = np.array([label_to_idx[label] for label in true_labels])
    pred_idx = np.array([label_to_idx[label] for label in pred_labels])
    
    # ===================== 3. Drawing Settings =====================
    plt.figure(figsize=(6, 6))
    # Reset matplotlib fonts (ensure proper display of English text)
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # ===================== 4. Plot marker points =====================
    for t, p, t_label, p_label in zip(true_idx, pred_idx, true_labels, pred_labels):
        if t_label == p_label:
            # Correct prediction: Green dot (with black border for enhanced visibility)
            plt.scatter(t, p, color='green', marker='o', s=120, 
                        edgecolors='black', linewidth=1, zorder=3)
        else:
            # Error prediction: Red cross (bold line)
            plt.scatter(t, p, color='red', marker='x', s=120, 
                        linewidth=2, zorder=3)
    
    # ===================== 5. Chart Enhancement =====================
    # Add Perfect Diagonal Prediction
    min_idx = 0
    max_idx = len(all_labels) - 1
    plt.plot([min_idx, max_idx], [min_idx, max_idx], 'k--', alpha=0.5, label='Perfect Prediction Line')
    
    # Set axes
    plt.xlabel('True Labels', fontsize=14, fontweight='medium')
    plt.ylabel('Predicted Labels', fontsize=14, fontweight='medium')
    
    # Set scale (display original labels instead of indices)
    plt.xticks(range(len(all_labels)), all_labels, ha='right')
    plt.yticks(range(len(all_labels)), all_labels)
    
    # Add grids, legends, and adjust layouts
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left', fontsize=13)
    plt.tight_layout()
    
    # Display Chart
    plt.savefig(r'experiment3_record\\exp2_image\\pred_true_comparison.jpg', dpi=1000, bbox_inches='tight')
    
    # ===================== 6. Output statistical information =====================
    correct = np.sum(true_labels == pred_labels)
    total = len(true_labels)
    accuracy = correct / total
    
    print("="*50)
    print(f"Total samples: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Incorrect predictions: {total - correct}")
    print(f"Prediction accuracy: {accuracy:.4f} ({accuracy:.2%})")
    print("="*50)

# 2. Define a prediction wrapper function
def model_predict_function(X):
    """
    A wrapper function for adapting TabularModel to SHAP's interface.
     Parameters:
    X (np.ndarray or pd.DataFrame): Feature data. SHAP passes in a NumPy array.
    Returns:
    np.ndarray: The model's predicted probabilities or raw outputs.
    """
    #  SHAP's KernelExplainer passes NumPy arrays, while TabularModel's predict method requires a DataFrame.
    # Therefore, we first need to convert the NumPy array back into a Pandas DataFrame.
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=X_test.columns)
    
    # Use the predict method of TabularModel to make predictions.
    # The predict method returns a dictionary containing keys such as ‘prediction’ and ‘probabilities’.
    # Deep learning model 
    prediction_result = model.predict(X)
    # SHAP requires the model's raw outputs (logits) or probabilities.
    # For classification tasks, ‘probabilities’ are typically used.
    # For regression tasks, use ‘prediction’.
    return prediction_result["consciousness_prediction"]

def shap_summary_top_features(shap_values, X, top_n=2, show=True):
    """
    Plot the SHAP summary map for the top n most important features, sorted by importance.
    
    Parameters:
    - shap_values: SHAP value array，shape (n_samples, n_features)
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
    feature_names = ['Language' if ele == 'language' else ele for ele in feature_names]
    
    # Compute the SHAP importance (mean absolute value) for each feature.
    shap_importance = np.abs(shap_values).mean(axis=0)
    
    # Sort by importance and take the indices of the top_n features.
    sorted_idx = np.argsort(shap_importance)[::-1][:top_n]
    
    # Filter out the corresponding SHAP values and features
    shap_values_top = shap_values[:, sorted_idx]
    X_top = X.iloc[:, sorted_idx] if isinstance(X, pd.DataFrame) else X[:, sorted_idx]
    feature_names_top = [feature_names[i] for i in sorted_idx]
    
    # Create a summary chart
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
    # Background Data - Interpretive Data: 300-300, modified to Background Data - Interpretive Data: 500-3000
    X_train_sample = shap.sample(X_train, 1000)  # Collect 1000 samples as background data
    # X_test = df_test.iloc[:, -5:-1]
    X_test = df_test.drop(columns=[x for x in select_columns if x != 'consciousness'])
    # X_test_sample = shap.sample(X_test, 3)  # Sampling 500 samples as explanatory data
    # Select 5 samples from consciousness levels 0, 1, 2, 3, and 4.
    X_test_sample = pd.concat([X_test[X_test['consciousness'] == i].sample(2, random_state=0) for i in range(5)], axis=0)
    y_test_sample = X_test_sample['consciousness']
    print('Selected sample data：')
    print(X_test_sample)
    # X_test_sample remove consciousness
    X_test_sample = X_test_sample.drop(columns=['consciousness'])
    model = pickle.load(open(r"model_pth\\exp_2\\02_similarity\\03_eyes_motion\\7-3\\seed_230\\rf_model.pkl", "rb"))
    model_pred = model.predict(X_test_sample)
    print('Model detection result:')
    print(model_pred)
    # Count the number of different categories in the actual labels and predicted labels.
    # plot_pred_true_comparison(y_test_sample, model_pred)
    # Retrieve the sample index for the error prediction
    misclassifid = np.where(model_pred != y_test_sample)[0]
    # plot_label_comparison(y_test_sample, model_pred)
    # true_counts = np.bincount(y_test_sample)
    # pred_counts = np.bincount(model_pred)
    # plot_awakening_radar(pred_counts, true_counts)
    # Replace "open_one's_eyes" with "Eyes" in X_test_sample
    X_test_sample = X_test_sample.rename(columns={'open_one\'s_eyes': 'Eyes'})
    # Replace "motion" with "Motor" in X_test_sample.
    X_test_sample = X_test_sample.rename(columns={'motion': 'Motor'})
    selected_class = 4
    
    # # This location uses a subset of the training data as background data, with the second item being the benchmark dataset | background dataset.
    explainer = shap.TreeExplainer(model, X_train_sample)
    shap_values = explainer.shap_values(X_test_sample)
    shap_values = np.array(shap_values)
    
    selected_features = ["Eyes", "Motor"]
    expected_value = explainer.expected_value[selected_class]
    # Filtering from the feature matrix
    X_selected = X_test_sample[selected_features]
    # Filtering based on SHAP values (according to feature name positions)
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
    # # Model 1: 7-3-danet
    # model = pickle.load(open(r"model_pth\\exp2\\vitials_eyes_motion\\7-3\\DANet.pkl", "rb"))
    # explainer = shap.PermutationExplainer(model_predict_function, X_train_sample)
    # shap_values = explainer(X_test_sample)
    # shap.summary_plot(shap_values, X_test_sample, cmap="coolwarm", show=False)
    # plt.tight_layout()
    # plt.savefig(r'exp_img\\exp2_eyes_motion_7_3\\7-3_danet_shap_summary.jpg', dpi=1000)
    # plt.close()
    
    # # Model 2: 7-3-tabnet
    # model = pickle.load(open(r"model_pth\\exp2\\vitials_eyes_motion\\7-3\\TabNet.pkl", "rb"))
    # explainer = shap.PermutationExplainer(model_predict_function, X_train_sample)
    # shap_values = explainer(X_test_sample)
    # shap.summary_plot(shap_values, X_test_sample, cmap="coolwarm", show=False)
    # plt.tight_layout()
    # plt.savefig(r'exp_img\\exp2_eyes_motion_7_3\\7-3_tabnet_shap_summary.jpg', dpi=1000)
    # plt.close()
    
    # # Model 3: 7-3-fttransformer
    # model = pickle.load(open(r"model_pth\\exp2\\vitials_eyes_motion\\7-3\\FTTransformer.pkl", "rb"))
    # explainer = shap.PermutationExplainer(model_predict_function, X_train_sample)
    # shap_values = explainer(X_test_sample)
    # shap.summary_plot(shap_values, X_test_sample, cmap="coolwarm", show=False)
    # plt.tight_layout()
    # plt.savefig(r'exp_img\\exp2_eyes_motion_7_3\\7-3_fttransformer_shap_summary.jpg', dpi=1000)
    # plt.close()
    
    model = pickle.load(open(r"model_pth\\exp_2\\02_similarity\\03_eyes_motion\\6-4\\seed_230\\rf_model.pkl", "rb"))
    explainer = shap.TreeExplainer(model, X_train_sample)
    shap_values = explainer.shap_values(X_test_sample)
    shap.summary_plot(shap_values[4], X_test_sample, cmap="coolwarm", show=False, max_display=2)
    # shap_values = np.array(shap_values)  # Convert to a NumPy array with shape (number of classes, number of samples, number of features)
    # # Take the SHAP value corresponding to the “final predicted category for each sample” (representing the feature's contribution to the “final prediction result”).
    # y_pred = model.predict(X_test_sample)
    # # The original shap_values dimensions are (number of classes, number of samples, number of features), requiring adjustment to the index order: (number of samples, number of classes, number of features).
    # shap_values_transposed = shap_values.transpose(1, 0, 2)  
    # # Indexing SHAP values by prediction category
    # shap_values = shap_values_transposed[np.arange(len(y_pred)), y_pred, :]
    # # shap.summary_plot(shap_values, X_test_sample, cmap="coolwarm", max_display=2, show=False)
    # shap.dependence_plot(
    #     ind=0,  # Feature 0 (ranked by SHAP importance)
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
    
    # # Model 5： 7-3-lightgbm
    # model = pickle.load(open(r"model_pth\\exp2\\vitials_eyes_motion\\7-3\\lgb.pkl", "rb"))
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
    # plt.savefig(r'exp_img\\exp2_eyes_motion_7_3\\7-3_LightGBM_shap_summary.jpg', dpi=1000)
    # plt.close()
    
    # # Model 6： 7-3-xgboost
    # model = pickle.load(open(r"model_pth\\exp2\\vitials_eyes_motion\\7-3\\xgboost.pkl", "rb"))
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
    # plt.savefig(r'exp_img\\exp2_eyes_motion_7_3\\7-3_XGBoost_shap_summary.jpg', dpi=1000)
    # plt.close()
    
    # # Model 7： 7-3-randomforest
    # model = pickle.load(open(r"model_pth\\exp2\\vitials_eyes_motion\\7-3\\rf_model.pkl", "rb"))
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
    # plt.savefig(r'exp_img\\exp2_eyes_motion_7_3\\7-3_RF_shap_summary.jpg', dpi=1000)
    # plt.close()
    
    # # Model 8： 7-3-CatBoost
    # model = pickle.load(open(r"model_pth\\exp2\\vitials_eyes_motion\\7-3\\catboost.pkl", "rb"))
    # explainer = shap.PermutationExplainer(model.predict, X_train_sample)
    # shap_values = explainer(X_test_sample)
    # shap.summary_plot(shap_values, X_test_sample, cmap="coolwarm", show=False)
    # plt.tight_layout()
    # plt.savefig(r'exp_img\\exp2_eyes_motion_7_3\\7-3_catboost_shap_summary.jpg', dpi=1000)
    # plt.close()
    
    # # Model 9： 7-3-MLP
    # model = pickle.load(open(r"model_pth\\exp2\\vitials_eyes_motion\\7-3\\mlp.pkl", "rb"))
    # explainer = shap.PermutationExplainer(model.predict, X_train_sample)
    # shap_values = explainer(X_test_sample)
    # shap.summary_plot(shap_values, X_test_sample, cmap="coolwarm", show=False)
    # plt.tight_layout()
    # plt.savefig(r'exp_img\\exp2_eyes_motion_7_3\\7-3_mlp_shap_summary.jpg', dpi=1000)
    # plt.close()
