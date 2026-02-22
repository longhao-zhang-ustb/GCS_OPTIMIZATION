import pandas as pd

def check_if_equal():
    df = pd.read_csv(r'zigong_data/dtBaseline.csv')
    # Read the PATIENT_ID column and store it in a list.
    patient_ids = df['PATIENT_ID'].tolist()
    print(len(patient_ids))
    # Deduplication
    patient_ids = list(set(patient_ids))
    print(len(patient_ids))
    # Read the INP_NO column and store it in a list.
    inp_nos = df['INP_NO'].tolist()
    print(len(inp_nos))
    # Deduplication
    inp_nos = list(set(inp_nos))
    print(len(inp_nos))
    # Method 2: Upon verification, it was found that PATIENT_ID and INP_NO corresponded one-to-one.
    # Iterate through patient IDs
    for patient_id in patient_ids:
        # Filter the INP_NO corresponding to the current PATIENT_ID from the DataFrame.
        inp_no = df[df['PATIENT_ID'] == patient_id]['INP_NO'].tolist()[0]
        if inp_no not in inp_nos:
            print(patient_id, inp_no)

# Check the number of patients in the training set and test set
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
