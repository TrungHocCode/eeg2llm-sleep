import pandas as pd
#FIXFILE_DIAG1
# Đọc dữ liệu từ file DIAGNOSIS.csv
df_diag = pd.read_csv('DIAGNOSIS.csv')
# Lọc các cột cần giữ lại
columns_needed_diag = [
    'STUDY_ENC_ID',
    'STUDY_PAT_ID',
    'DX_START_DATETIME',
    'DX_END_DATETIME',
    'DX_NAME'
]
# Giữ lại các cột cần thiết
filtered_diag_df = df_diag[columns_needed_diag]
# Lưu lại kết quả nếu cần
filtered_diag_df.to_csv('filtered_diagnosis.csv', index=False)

#FIXFILE_DIAG2
# Đọc dữ liệu đã lọc
df_diag = pd.read_csv('filtered_diagnosis.csv')

# Đưa cột DX_NAME về dạng chữ thường để so sánh dễ hơn
df_diag['DX_NAME_lower'] = df_diag['DX_NAME'].str.lower()

# Hàm phân loại bệnh
def classify_diag(name):
    if pd.isna(name):
        return 'normal'
    if 'seizure' in name:
        return 'seizure'
    elif 'sleep disorder' in name:
        return 'sleep disorders'
    elif 'cerebral infarction' in name:
        return 'cerebral infarction'
    else:
        return 'normal'

# Áp dụng hàm để tạo cột mới
df_diag['diag'] = df_diag['DX_NAME_lower'].apply(classify_diag)
# Xóa cột phụ nếu không cần
df_diag = df_diag.drop(columns=['DX_NAME_lower'])
# Lưu lại file kết quả
df_diag.to_csv('diagnosis_with_diag.csv', index=False)


#FIXFILE_MEDICATION
# Đọc dữ liệu từ file CSV gốc
df = pd.read_csv('medication.csv')

# Các cột cần giữ lại
columns_needed = [
    'STUDY_ENC_ID',
    'STUDY_PAT_ID',
    'MED_START_DATETIME',
    'MED_END_DATETIME',
    'MED_ORDER_DATETIME',
    'MED_TAKEN_DATETIME',
    'FREQUENCY',
    'EFF_DRUG_DOSE_SOURCE_VALUE',
    'DRUG_DOSE_UNIT',
    'MEDICATION_DESCR'
]

# Lọc dữ liệu với các cột cần thiết
filtered_df = df[columns_needed]

# Chuyển MED_TAKEN_DATETIME về kiểu datetime để đảm bảo xử lý chính xác
filtered_df['MED_TAKEN_DATETIME'] = pd.to_datetime(filtered_df['MED_TAKEN_DATETIME'], errors='coerce')

# Bỏ các dòng không có MED_TAKEN_DATETIME
filtered_df = filtered_df[filtered_df['MED_TAKEN_DATETIME'].notna()]

# Lưu kết quả đã lọc và làm sạch
filtered_df.to_csv('filtered_medication.csv', index=False)

diag_df = pd.read_csv('diagnosis_with_diag.csv')
filtered_df = pd.read_csv('data_filtered_modified.csv')
diag_df['diag'] = filtered_df['diag']

diag_df['diag'] = diag_df['diag'].replace({
    'Động kinh': 'seizure',
    'Rối loạn giấc ngủ': 'sleep disorders',
    'Nhồi máu não': 'cerebral infarction'
})

# Lưu kết quả
diag_df.to_csv('diagnosis_diag_updated.csv', index=False)


### CREATE INPUT-OUTPUT FILE FOR INTENT CLASSIFICATION ###
# Load data
df_diag = pd.read_csv("diagnosis_diag_updated.csv", parse_dates=["DX_START_DATETIME", "DX_END_DATETIME"])
df_med = pd.read_csv("filtered_medication.csv", parse_dates=["MED_ORDER_DATETIME"])
df_demo = pd.read_csv("DEMONGRAPHIC.csv")
df_meas = pd.read_csv("MEASUREMENT.csv", parse_dates=["MEAS_RECORDED_DATETIME"])

# Disease mapping
DISEASE_MAP = {
    "seizure": "seizure",
    "cerebral infarction": "cerebral infarction",
    "sleep disorders": "sleep disorders"
}

# Output containers
selected_patients = set()
results = []

def safe(val):
    return val if pd.notna(val) else "None"

def get_diagnosis_name(enc_id):
    dx_names = df_diag[df_diag['STUDY_ENC_ID'] == enc_id]['DX_NAME'].unique()
    return ", ".join([safe(x) for x in dx_names])

def get_measurement(pat_id, enc_id):
    sub = df_meas[(df_meas['STUDY_PAT_ID'] == pat_id) & (df_meas['STUDY_ENC_ID'] == enc_id)]
    for _, row in sub.iterrows():
        mtype = safe(row['MEAS_TYPE'])
        if mtype in ["BMI", "BMIPCT"]:
            return f"{mtype}: {safe(row['MEAS_VALUE_NUMBER'])}"
        else:
            return f"{mtype}: {safe(row['MEAS_VALUE_TEXT'])}"
    return "None"

def get_medications(enc_id):
    meds = df_med[df_med['STUDY_ENC_ID'] == enc_id]
    unique_drugs = []
    seen = set()
    for _, row in meds.iterrows():
        name = safe(row['MEDICATION_DESCR'])
        if name in seen:
            continue
        seen.add(name)
        freq = safe(row['FREQUENCY'])
        dose = safe(row['EFF_DRUG_DOSE_SOURCE_VALUE'])
        unit = safe(row['DRUG_DOSE_UNIT'])
        dosage = f"{dose} {unit}" if dose != "None" and unit != "None" else "None"
        unique_drugs.append((name, freq, dosage))
        if len(unique_drugs) >= 5:
            break
    return unique_drugs

def process_group(diag_label, limit):
    global selected_patients
    count = 0
    sub_diag = df_diag[df_diag['diag'] == diag_label]
    for pat_id in sub_diag['STUDY_PAT_ID'].unique():
        if pat_id in selected_patients:
            continue
        pat_sub = sub_diag[sub_diag['STUDY_PAT_ID'] == pat_id]
        for _, dx_row in pat_sub.iterrows():
            enc_id = dx_row['STUDY_ENC_ID']
            diagnosis = DISEASE_MAP.get(diag_label, "normal")
            dx_names = get_diagnosis_name(enc_id)
            demo = df_demo[df_demo['STUDY_PAT_ID'] == pat_id]
            gender = safe(demo.iloc[0]['GENDER_DESCR']) if not demo.empty else "None"
            age = safe(demo.iloc[0]['PEDS_GEST_AGE_NUM_DAYS']) if not demo.empty else "None"
            meas = get_measurement(pat_id, enc_id)
            meds = get_medications(enc_id)
            if not meds and diag_label != "normal":
                continue

            result = f"Cặp: {len(results) + 1}\nInput:\n- Patient Information:\n"
            result += f"- Diagnose: {diagnosis}\n"
            result += f"- Past_diseases: {dx_names}\n"
            result += f"- Days-Age: {age}\n"
            result += f"- Gender: {gender}\n"
            result += f"- Meas-type-value: {meas}\n"
            result += "Output:\n"

            if diag_label != "normal":
                result += "Treatment Regimen:\n"
                for drug, freq, dose in meds:
                    result += f"- Drugs Name: {drug}\n"
                    result += f"- Usage time: {freq}\n"
                    result += f"- Dosage: {dose}\n\n"

            results.append(result)
            selected_patients.add(pat_id)
            count += 1
            break
        if count >= limit:
            break

def process_normal(limit):
    all_ids = set(df_demo['STUDY_PAT_ID'])
    diseased_ids = set(df_diag['STUDY_PAT_ID'])
    normal_ids = list(all_ids - diseased_ids - selected_patients)
    count = 0
    for pat_id in normal_ids:
        demo = df_demo[df_demo['STUDY_PAT_ID'] == pat_id]
        gender = safe(demo.iloc[0]['GENDER_DESCR']) if not demo.empty else "None"
        age = safe(demo.iloc[0]['PEDS_GEST_AGE_NUM_DAYS']) if not demo.empty else "None"
        meas = get_measurement(pat_id, None)
        result = f"Cặp: {len(results) + 1}\nInput:\n- Patient Information:\n"
        result += f"- Diagnose: normal\n"
        result += f"- Past_diseases: None\n"
        result += f"- Days-Age: {age}\n"
        result += f"- Gender: {gender}\n"
        result += f"- Meas-type-value: {meas}\n"
        result += "Output:\n"
        results.append(result)
        selected_patients.add(pat_id)
        count += 1
        if count >= limit:
            break

# Process each disease group
process_group("cerebral infarction", 500)
process_group("sleep disorders", 500)
process_group("seizure", 500)
process_normal(500)

# Save to file
with open("input_output.txt", "w", encoding="utf-8") as f:
    for entry in results:
        f.write(entry + "-" * 40 + "\n")

print("Đã tạo xong file input_output.txt với đầy đủ định dạng.")