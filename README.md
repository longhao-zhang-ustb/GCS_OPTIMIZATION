# Optimization Study on Glasgow Defect Detection

## Description

This is a code implementation that utilizes machine learning and data imputation techniques to improve the detection of Glasgow Coma Scale (GCS) assessment failures. We focused our research on addressing issues such as assessment failures in patients with artificial airways and the simplification of GCS scoring. Our study employed a restricted-access dataset, which can be downloaded upon obtaining data access permissions. Our research confirms the effectiveness of machine learning and data imputation in enhancing the detection of GCS assessment failures. If you have any other questions, feel free to contact me.

## Dataset Information

This research utilizes a restricted-access dataset (Critical care database comprising patients with infection at Zigong Fourth People's Hospital), and access to the resources is granted only after completing relevant training and signing the data usage agreement. We have completed the Collaborative Institutional Training Initiative (Record ID: 64076481) training and signed the data usage agreement, thereby exempting us from secondary review by the institutional ethics review board. The address for requesting raw data is https://physionet.org/content/icu-infection-zigong-fourth/1.1/ (accessed January 17, 2026).

## Code Information

### Data Preprocessing Related Files

utils_save/zigong_data_prepro_no_gcs.py: Used for performing preprocessing tasks such as patient record screening, handling outliers, and label encoding.<br>utils_save/00_dataset_split.py: Implementation Code for the DM-N1 Filling Method.<br>utils_save/01_dataset_split.py: Implementation Code for the DM-M Filling Method.<br>utils_save/03_dataset_split.py: Implementation Code for the DM-S Filling Method.<br>utils_save/check_data.py: Verify the number of patients in the training and test sets to ensure no overlap between them, thereby preventing data leakage.<br>

### Algorithm Related Files

main.py: Program entry point file.<br>requirements.txt: Includes the dependency environment required for program execution.<br>train_zigong.py: Model Definition and Training.<br>evaluate_zigong.py: Load the trained model for performance evaluation.<br>exp3_1_all_metrics_shap_analysis.py: Perform SHAP analysis on the test results of patients with assessment failure.<br>exp3_2_sample_shap_analysis.py: Performing SHAP analysis on experimental results for simplified GCS scoring.<br>

### Drawing Function Related Files

draw_pic/plot_draw_age_sex.py: Plotting Function for Demographic Characteristics of Statistical Data (Figure 2).<br>draw_pic/plot_gcs_arousal_heatmap.py: Plotting function for the correlation between GCS total scores and arousal levels (Figure 3).<br>draw_pic/plot_gcs_cos.py: Plotting function for the correlation between combinations of different GCS scores and corresponding levels of consciousness (Figure 4).<br>draw_pic/plot_bar.py: Plot of detection performance for different GCS simplification items on arousal level (Figure 5).<br>

## Usage Instructions

Step1: Create a virtual Python environment using Anaconda.<br>Step2: Run `pip install -r requirements.txt` to install environment dependencies.<br>Step3: Run `python zigong_data_prepro_no_gcs.py` to perform patient screening.<br>Step4: Run utils_save/00_dataset_split.py, utils_save/01_dataset_split.py, and utils_save/03_dataset_split.py separately to obtain the training and test sets for the three padding methods.<br>Step5: Run main.py to complete data loading and perform Experiment 1 and Experiment 2.<br>Step6: Run exp3_1_all_metrics_shap_analysis.py and exp3_2_sample_shap_analysis.py to complete Experiment 3.<br>The plotting functions draw_pic/plot_draw_age_sex.py, draw_pic/plot_gcs_arousal_heatmap.py, and draw_pic/plot_gcs_cos.py are used for statistical data visualization and can be run independently after data acquisition. draw_pic/plot_bar.py can be used for plotting upon obtaining experimental results.

## Requirements

Please refer to the requirements.txt. The specific installation method is as follows: `pip install -r requirements.txt`.

## Methodology

Please refer to the manuscript.

## Citations

1. Xu, P., Chen, L., & Zhang, Z. (2022). Critical care database comprising patients with infection at Zigong Fourth People's Hospital (version 1.1). PhysioNet. RRID:SCR_007345. https://doi.org/10.13026/xpt9-z726.
2. Xu P, Chen L, Zhu Y, Yu S, Chen R, Huang W, Wu F, Zhang Z. Critical Care Database Comprising Patients With Infection. Front Public Health. 2022 Mar 17;10:852410. doi: 10.3389/fpubh.2022.852410. PMID: 35372245; PMCID: PMC8968758.
3. Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215–e220. RRID:SCR_007345.

## License & Contribution Guidelines

**License (for files):** [PhysioNet Credentialed Health Data License 1.5.0](https://physionet.org/content/icu-infection-zigong-fourth/view-license/1.1/)<br>**Data Use Agreement:** [PhysioNet Credentialed Health Data Use Agreement 1.5.0](https://physionet.org/content/icu-infection-zigong-fourth/view-dua/1.1/)<br>**Required training:** [CITI Data or Specimens Only Research](https://physionet.org/content/icu-infection-zigong-fourth/view-required-training/1.1/#1)<br>
