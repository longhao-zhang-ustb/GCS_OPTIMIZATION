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
    # 读取数据
    df = pd.read_csv('zigong_data/dtNursingChart.csv', low_memory=False)
    # 去掉其中ChartTime为空的行
    df = df.dropna(subset=['ChartTime'])
    df_status = pd.read_csv('zigong_data/dtICD.csv', low_memory=False)
    # 查找df_status中1种INP_NO是否仅对应1种Status_Discharge
    # inp_no_status_counts = df_status.groupby('INP_NO')['Status_Discharge'].nunique()
    # inp_no_with_multiple_status = inp_no_status_counts[inp_no_status_counts > 1]
    # 去掉Status_Discharge为空的行
    df_status = df_status.dropna(subset=['Status_Discharge'])
    # 获取INP_NO与Status_Discharge两列
    df_status = df_status[['INP_NO', 'Status_Discharge']]
    # 按照INP_NO分组，如果Status_Discharge包含Dead，则删掉该组INP_NO对应的全部记录
    df_status = df_status[~df_status['INP_NO'].isin(df_status[df_status['Status_Discharge'].str.contains('Dead', case=False, na=False)]['INP_NO'])]
    # 将df_status中的INP_NO去重
    inp_no_alive = df_status['INP_NO'].unique()
    # 结果保存为一个列表
    inp_no_alive = list(inp_no_alive)
    # 统计下Status_Discharge中不同取值的数量
    # 统计所有列的缺失值
    # missing_rates = df.isna().mean()
    # for column, missing_rate in missing_rates.items(): 
    #     print(f'Column: {column}, Missing Rate: {missing_rate:.2%}')
    # 选择需要的列
    selected_columns = ['INP_NO', 'heart_rate', 'breathing', 'Blood_oxygen_saturation',
                        'Blood_pressure_high', 'Blood_pressure_low', 'Left_pupil_size', 
                        'Right_pupil_size', 'open_one\'s_eyes', 'motion', 'language', 
                        'consciousness', 'ChartTime']
    df_selected = df[selected_columns]
    # 判断df_selected中的INP_NO是否存在于inp_no_alive中，存在则保留，否则删除
    df_selected = df_selected[df_selected['INP_NO'].isin(inp_no_alive)]
    # df_selected中重复行仅保留1条记录
    df_selected = df_selected.drop_duplicates(keep='first')
    # 统计缺失率
    missing_rates = df_selected.isna().mean()
    for column, missing_rate in missing_rates.items():
        print(f'Column: {column}, Missing Rate: {missing_rate:.2%}')
    # 去掉包含缺失值的行
    df_selected = df_selected.dropna()
    for state, count in df_selected['language'].value_counts(sort=True, ascending=True).items():
        print(f'  State: {state}, Count: {count}')
    # 添加GCS评分
    # 将open_ones_eyes列保留箭头前的字符并转为数值型，不能转换的用nan表示
    df_selected['open_one\'s_eyes'] = df_selected['open_one\'s_eyes'].apply(calc_eye_score)
    # 将motion列保留箭头前的字符并转为数值型，不能转换的用nan表示
    df_selected['motion'] = df_selected['motion'].apply(calc_motion_score)
    # 将language列保留箭头前的字符并转为数值型，不能转换的用nan表示
    df_selected['language'] = df_selected['language'].apply(calc_language_score)
    # 去掉包含缺失值的行
    df_cleaned = df_selected.dropna()
    print(f'Original shape: {df_selected.shape}, Cleaned shape: {df_cleaned.shape}')
    # 统计缺失率
    missing_rates = df_cleaned.isna().mean()
    for column, missing_rate in missing_rates.items():
        print(f'Column: {column}, Missing Rate: {missing_rate:.2%}')
    # 去掉非数值型的行
    df_cleaned = df_cleaned[pd.to_numeric(df_cleaned['Left_pupil_size'], errors='coerce').notnull()]
    df_cleaned = df_cleaned[pd.to_numeric(df_cleaned['Right_pupil_size'], errors='coerce').notnull()]
    df_cleaned['Left_pupil_size'] = df_cleaned['Left_pupil_size'].astype(float)
    df_cleaned['Right_pupil_size'] = df_cleaned['Right_pupil_size'].astype(float)
    # 去掉异常的生命体征测量值
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
    # 统计各个意识状态的数量
    consciousness_counts = df_cleaned['consciousness'].value_counts()
    print('Consciousness state counts:')
    for state, count in consciousness_counts.items():
        print(f'  State: {state}, Count: {count}')
    # 保留值为浅昏迷，中昏迷，深昏迷的行
    valid_states = ['清醒', '嗜睡', '昏睡', '浅昏迷', '中昏迷', '深昏迷']
    df_filtered = df_cleaned[df_cleaned['consciousness'].isin(valid_states)]
    print(f'After filtering valid states, shape: {df_filtered.shape}')
    consciousness_counts_filtered = df_filtered['consciousness'].value_counts()
    print('Filtered consciousness state counts:')
    for state, count in consciousness_counts_filtered.items():
        print(f'  State: {state}, Count: {count}')
    # 对consciousness列进行编码
    # state_encoding = {'清醒': 0, '嗜睡':1, '昏睡': 1, '浅昏迷': 2, '中昏迷': 3, '深昏迷': 4}
    state_encoding = {'清醒': 0, '嗜睡':1, '昏睡': 2, '浅昏迷': 3, '中昏迷': 4, '深昏迷': 5}
    df_filtered['consciousness'] = df_filtered['consciousness'].map(state_encoding)
    # 打印编码后的各个意识状态的数量
    consciousness_counts_encoded = df_filtered['consciousness'].value_counts()
    print('Encoded consciousness state counts:')
    for state, count in consciousness_counts_encoded.items():
        print(f'  State: {state}, Count: {count}')
    # 打印患者的数目
    # patient_num = df_filtered['INP_NO'].value_counts()
    print(f'Before filtering age >= 18, shape: {df_filtered.shape}')
    # 级联患者基本信息表
    df_patient_info = pd.read_csv(r'zigong_data\dtBaseline.csv')
    df_filtered = pd.merge(df_filtered, df_patient_info[['INP_NO', 'Age', 'SEX']], on='INP_NO', how='left')
    # 将年龄, 性别列调整到INP_NO列后面
    cols = df_filtered.columns.tolist()
    cols.remove('Age')
    cols.remove('SEX')
    cols.insert(cols.index('INP_NO') + 1, 'Age')
    cols.insert(cols.index('INP_NO') + 2, 'SEX')
    df_filtered = df_filtered[cols]
    # 查找其中年龄不大于18的患者
    df_filtered = df_filtered[df_filtered['Age'] >= 18]
    # 过滤所有字段都相同的记录
    df_filtered = df_filtered.drop_duplicates()
    # 打印过滤后的患者数目
    df_patient = df_filtered.drop_duplicates(subset='INP_NO', keep='first')
    print(f'After filtering patients age >= 18, shape: {df_patient.shape}')
    # 统计各个意识状态的数量
    consciousness_counts = df_filtered['consciousness'].value_counts()
    print('Consciousness state counts:')
    for state, count in consciousness_counts.items():
        print(f'  State: {state}, Count: {count}')
    exit()
    """
    State: 2, Count: 204443 ==>浅昏迷
    State: 1, Count: 187110 ==>嗜睡+昏睡
    State: 3, Count: 92298  ==>中昏迷
    State: 0, Count: 33331  ==>清醒
    State: 4, Count: 21327  ==>深昏迷
    """
    # 保存最终处理的数据
    df_filtered.to_csv('zigong_data/20251227_final_processed_data.csv', index=False)
