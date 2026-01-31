import pandas as pd

def check_if_equal():
    df = pd.read_csv(r'zigong_data/dtBaseline.csv')
    # 读取PATIENT_ID列，存入列表
    patient_ids = df['PATIENT_ID'].tolist()
    # 打印数量
    print(len(patient_ids))
    # 去重
    patient_ids = list(set(patient_ids))
    # 打印数量
    print(len(patient_ids))
    # 读取INP_NO列，存入列表
    inp_nos = df['INP_NO'].tolist()
    # 打印数量
    print(len(inp_nos))
    # 去重
    inp_nos = list(set(inp_nos))
    # 打印数量
    print(len(inp_nos))
    # 方法二：检查后发现患者ID和住院ID是一一对应的关系
    # 检查PATIENT_ID和INP_NO是否一一对应
    # 遍历患者ID
    for patient_id in patient_ids:
        # 从DataFrame中筛选出当前患者ID对应的住院ID
        inp_no = df[df['PATIENT_ID'] == patient_id]['INP_NO'].tolist()[0]
        # 如果住院ID不在住院ID列表中，打印患者ID和住院ID
        if inp_no not in inp_nos:
            print(patient_id, inp_no)

# 检查训练集和测试集中患者的数量
def check_patient_counts():
    df_train = pd.read_csv(r'data_base/03_cos_similarity_db/full_assessment/7-3/train.csv')
    df_test = pd.read_csv(r'data_base/03_cos_similarity_db/full_assessment/7-3/test.csv')
    train_patient_ids = df_train['INP_NO'].unique()
    test_patient_ids = df_test['INP_NO'].unique()
    print(f'Number of unique patients in training set: {len(train_patient_ids)}')
    print(f'Number of unique patients in testing set: {len(test_patient_ids)}')

if __name__ == '__main__':
    check_if_equal()
    check_patient_counts()
