import pandas as pd
import numpy as np
from scipy import stats

def calc_eye_score(method):
    if method == '1→无反应' or method == '1→反应' or method == '1无反应':
        return 1
    elif method == '2→疼痛睁眼' or method == '2→痛睁眼':
        return 2
    elif method == '3→呼唤睁眼' or method == '3→唤睁眼':
        return 3
    elif method == '4→自动睁眼' or method == '4唤睁眼':
        return 4
    else:
        return np.nan

def calc_motion_score(method):
    if method == '1→无反应':
        return 1
    elif method == '2→疼痛伸直':
        return 2
    elif method == '3→疼痛屈曲':
        return 3
    elif method == '4→疼痛躲避':
        return 4
    elif method == '5→疼痛定位':
        return 5
    elif method == '6→遵嘱运动':
        return 6
    elif method == 'S→运动障碍':
        return -1
    else:
        return np.nan

def calc_language_score(method):
    if method == '1→无反应':
        return 1
    elif method == '2→只发声':
        return 2
    elif method == '3→语言不确切' or  method == '→语言不确切':
        return 3
    elif method == '4→语言不正确':
        return 4
    elif method == '5→语言正确':
        return 5
    elif method == 'T→人工气道' or method == '不能发音' or method == '→人工气道' or method == 'T→人工道' or method == '工气道' or method == '人工气道':
        return -1
    else:
        return np.nan


if __name__ == '__main__':
    # Read data
    df = pd.read_csv('zigong_data/dtNursingChart.csv', low_memory=False)
    # Remove rows where ChartTime is null
    df = df.dropna(subset=['ChartTime'])
    df_status = pd.read_csv('zigong_data/dtICD.csv', low_memory=False)
    # Determine whether INP_NO in df_status corresponds uniquely to Status_Discharge.
    # inp_no_status_counts = df_status.groupby('INP_NO')['Status_Discharge'].nunique()
    # inp_no_with_multiple_status = inp_no_status_counts[inp_no_status_counts > 1]
    # Remove rows where Status_Discharge is null
    df_status = df_status.dropna(subset=['Status_Discharge'])
    # Retrieve the INP_NO and Status_Discharge columns
    df_status = df_status[['INP_NO', 'Status_Discharge']]
    # Group by INP_NO. If Status_Discharge contains Dead, delete all records corresponding to that INP_NO within the group.
    df_status = df_status[~df_status['INP_NO'].isin(df_status[df_status['Status_Discharge'].str.contains('Dead', case=False, na=False)]['INP_NO'])]
    # Remove duplicates of INP_NO in df_status
    inp_no_alive = df_status['INP_NO'].unique()
    # The results are saved as a list.
    inp_no_alive = list(inp_no_alive)
    # Count the number of different values in Status_Discharge
    # Count the number of missing values in all columns
    # missing_rates = df.isna().mean()
    # for column, missing_rate in missing_rates.items(): 
    #     print(f'Column: {column}, Missing Rate: {missing_rate:.2%}')
    # Select the required columns
    selected_columns = ['INP_NO', 'heart_rate', 'breathing', 'Blood_oxygen_saturation',
                        'Blood_pressure_high', 'Blood_pressure_low', 'Left_pupil_size', 
                        'Right_pupil_size', 'open_one\'s_eyes', 'motion', 'language', 
                        'consciousness', 'ChartTime']
    df_selected = df[selected_columns]
    # Determine whether the INP_NO in df_selected exists in inp_no_alive. If it exists, retain it; otherwise, delete it.
    df_selected = df_selected[df_selected['INP_NO'].isin(inp_no_alive)]
    # Only one duplicate row is retained in `df_selected`.
    df_selected = df_selected.drop_duplicates(keep='first')
    # Statistical Missing Rate
    missing_rates = df_selected.isna().mean()
    for column, missing_rate in missing_rates.items():
        print(f'Column: {column}, Missing Rate: {missing_rate:.2%}')
    # Remove rows containing missing values
    df_selected = df_selected.dropna()
    for state, count in df_selected['language'].value_counts(sort=True, ascending=True).items():
        print(f'  State: {state}, Count: {count}')
    # Add GCS score
    # Convert the characters before the arrow in the open_one‘s_eyes column to numeric values, replacing any unconvertible characters with nan.
    df_selected['open_one\'s_eyes'] = df_selected['open_one\'s_eyes'].apply(calc_eye_score)
    # Retain the characters preceding the arrow in the motion column and convert them to numeric values. Use nan for values that cannot be converted.
    df_selected['motion'] = df_selected['motion'].apply(calc_motion_score)
    # Retain characters before the arrow in the language column and convert them to numeric values. Use nan for values that cannot be converted.
    df_selected['language'] = df_selected['language'].apply(calc_language_score)
    # Remove rows containing missing values
    df_cleaned = df_selected.dropna()
    print(f'Original shape: {df_selected.shape}, Cleaned shape: {df_cleaned.shape}')
    # Calculate the missing rate
    missing_rates = df_cleaned.isna().mean()
    for column, missing_rate in missing_rates.items():
        print(f'Column: {column}, Missing Rate: {missing_rate:.2%}')
    # Remove non-numeric rows
    df_cleaned = df_cleaned[pd.to_numeric(df_cleaned['Left_pupil_size'], errors='coerce').notnull()]
    df_cleaned = df_cleaned[pd.to_numeric(df_cleaned['Right_pupil_size'], errors='coerce').notnull()]
    df_cleaned['Left_pupil_size'] = df_cleaned['Left_pupil_size'].astype(float)
    df_cleaned['Right_pupil_size'] = df_cleaned['Right_pupil_size'].astype(float)
    # Remove abnormal vital sign measurements
    df_cleaned = df_cleaned[(df_cleaned['heart_rate'] >= 0) & \
                            (df_cleaned['heart_rate'] <= 350) & \
                            (df_cleaned['breathing'] >= 0) & \
                            (df_cleaned['breathing'] <= 80) & \
                            (df_cleaned['Blood_oxygen_saturation'] >= 0) & \
                            (df_cleaned['Blood_oxygen_saturation'] <= 100) & \
                            (df_cleaned['Blood_pressure_high'] >= 0) & \
                            (df_cleaned['Blood_pressure_high'] <= 300) & \
                            (df_cleaned['Blood_pressure_low'] >= 0) & \
                            (df_cleaned['Blood_pressure_low'] <= 200) & \
                            (df_cleaned['Left_pupil_size'] >= 0) & \
                            (df_cleaned['Left_pupil_size'] <= 10) & \
                            (df_cleaned['Right_pupil_size'] >= 0) & \
                            (df_cleaned['Right_pupil_size'] <= 10)]
    consciousness_counts = df_cleaned['consciousness'].value_counts()
    print('Consciousness state counts:')
    for state, count in consciousness_counts.items():
        print(f'  State: {state}, Count: {count}')
    # Rows with values retained for light coma, moderate coma, and severe coma
    valid_states = ['清醒', '嗜睡', '昏睡', '浅昏迷', '中昏迷', '深昏迷']
    df_filtered = df_cleaned[df_cleaned['consciousness'].isin(valid_states)]
    print(f'After filtering valid states, shape: {df_filtered.shape}')
    consciousness_counts_filtered = df_filtered['consciousness'].value_counts()
    print('Filtered consciousness state counts:')
    for state, count in consciousness_counts_filtered.items():
        print(f'  State: {state}, Count: {count}')
    # Encode the consciousness column
    state_encoding = {'清醒': 0, '嗜睡':1, '昏睡': 1, '浅昏迷': 2, '中昏迷': 3, '深昏迷': 4}
    df_filtered['consciousness'] = df_filtered['consciousness'].map(state_encoding)
    consciousness_counts_encoded = df_filtered['consciousness'].value_counts()
    print('Encoded consciousness state counts:')
    for state, count in consciousness_counts_encoded.items():
        print(f'  State: {state}, Count: {count}')
    # patient_num = df_filtered['INP_NO'].value_counts()
    print(f'Before filtering age >= 18, shape: {df_filtered.shape}')
    # Cascade Patient Basic Information Form
    df_patient_info = pd.read_csv(r'zigong_data\dtBaseline.csv')
    df_filtered = pd.merge(df_filtered, df_patient_info[['INP_NO', 'Age', 'SEX']], on='INP_NO', how='left')
    # Move the Age and Gender columns after the INP_NO column.
    cols = df_filtered.columns.tolist()
    cols.remove('Age')
    cols.remove('SEX')
    cols.insert(cols.index('INP_NO') + 1, 'Age')
    cols.insert(cols.index('INP_NO') + 2, 'SEX')
    df_filtered = df_filtered[cols]
    # Identify patients whose age is no greater than 18.
    df_filtered = df_filtered[df_filtered['Age'] >= 18]
    # Filter records where all fields are identical
    df_filtered = df_filtered.drop_duplicates()
    # Print the number of filtered patients
    df_patient = df_filtered.drop_duplicates(subset='INP_NO', keep='first')
    print(f'After filtering patients age >= 18, shape: {df_patient.shape}')
    # Count the number of each consciousness state
    consciousness_counts = df_filtered['consciousness'].value_counts()
    print('Consciousness state counts:')
    for state, count in consciousness_counts.items():
        print(f'  State: {state}, Count: {count}')
    # Save the final processed data
    df_filtered.to_csv('zigong_data/20251227_final_processed_data.csv', index=False)
