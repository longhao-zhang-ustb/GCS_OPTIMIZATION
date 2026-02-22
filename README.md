# Optimization Study on Glasgow Defect Detection

## Description

This is a code implementation that utilizes machine learning and data imputation techniques to improve the detection of Glasgow Coma Scale (GCS) assessment failures. We focused our research on addressing issues such as assessment failures in patients with artificial airways and the simplification of GCS scoring. Our study employed a restricted-access dataset, which can be downloaded upon obtaining data access permissions. Our research confirms the effectiveness of machine learning and data imputation in enhancing the detection of GCS assessment failures. If you have any other questions, feel free to contact me.

## Dataset Information

This research utilizes a restricted-access dataset (Database name: Critical care database comprising patients with infection at Zigong Fourth People's Hospital), and access to the resources is granted only after completing relevant training and signing the data usage agreement. We have completed the Collaborative Institutional Training Initiative (Record ID: 64076481) training and signed the data usage agreement, thereby exempting us from secondary review by the institutional ethics review board. The address for requesting raw data is https://physionet.org/content/icu-infection-zigong-fourth/1.1/ (accessed January 17, 2026).

## Code Information
**main.py:** <br>
**requirements.txt:** <br>
**train_zigong.py:** <br>
**evaluate_zigong.py:** <br>
**exp3_1_all_metrics_shap_analysis.py:** <br>
**exp3_2_sample_shap_analysis.py:** <br>
**zigong_data_prepro_no_gcs.py:** <br>
**utils_save/00_dataset_split.py:** <br>
**utils_save/01_dataset_split.py:** <br>
**utils_save/03_dataset_split.py:** <br>
**utils_save/check_data.py:** <br>
**draw_pic/plot_draw_age_sex.py:** <br>
**draw_pic/plot_gcs_cos.py:** <br>
**draw_pic/plot_gcs_arousal_heatmap.py:** <br>
**draw_pic/plot_bar.py:** <br>
## Usage Instructions



## Requirements

Please refer to the requirements.txt. The specific installation method is as follows: pip install -r requirements.txt.

## Methodology

Please refer to the manuscript.

## Citations

1. Xu, P., Chen, L., & Zhang, Z. (2022). Critical care database comprising patients with infection at Zigong Fourth People's Hospital (version 1.1). PhysioNet. RRID:SCR_007345. https://doi.org/10.13026/xpt9-z726.
2. Xu P, Chen L, Zhu Y, Yu S, Chen R, Huang W, Wu F, Zhang Z. Critical Care Database Comprising Patients With Infection. Front Public Health. 2022 Mar 17;10:852410. doi: 10.3389/fpubh.2022.852410. PMID: 35372245; PMCID: PMC8968758.
3. Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215–e220. RRID:SCR_007345.

## License & Contribution Guidelines

[PhysioNet Credentialed Health Data License 1.5.0](https://physionet.org/content/icu-infection-zigong-fourth/view-license/1.1/) <br>
[PhysioNet Credentialed Health Data Use Agreement 1.5.0](https://physionet.org/content/icu-infection-zigong-fourth/view-dua/1.1/) <br>
[CITI Data or Specimens Only Research](https://physionet.org/content/icu-infection-zigong-fourth/view-required-training/1.1/#1) <br>
