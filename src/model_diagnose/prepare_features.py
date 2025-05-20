import pandas as pd
import os

#Ham lay ma benh nhan (study_pat_id) tu ten file tsv
def get_study_pat_id(file_path):
    # Lấy tên file từ đường dẫn
    file_name = os.path.basename(file_path)
    file_name = os.path.splitext(file_name)[0]
    parts = file_name.split('_')
    return int(parts[0]), int(parts[1])

#Ham phan loai giai doan + su kien
def classify_sleep_stages(desc):
    main_desc=["Sleep stage W","Sleep stage N1","Sleep stage N2","Sleep stage N3","Sleep stage R","Obstructive Apnea","Obstructive Hypopnea","Mixed Apnea","Central Apnea","Oxygen Desaturation","EGG Arousal","Hypopnea"]
    if desc in main_desc:
        return desc 
    return "Other"

#Duyet toan bo file tsv de tinh cac gia tri thoi gian, tan suat va gop lai thanh 1 dataframe hoan chinh
data_dir="data/sleep_data/tsv"
file_list=os.listdir(data_dir)    
dfs=[]
for file in file_list:
    file_path=os.path.join(data_dir,file)
    study_pat_id,sleep_study_id=get_study_pat_id(file_path)
    df = pd.read_csv(file_path, sep="\t")

    df['sleep_stages'] = df['description'].apply(classify_sleep_stages)

    total_duration = df["duration"].sum()
    total_hours = total_duration / 3600

    grouped = df.groupby("sleep_stages").agg(
        Count=('duration', 'count'),  # Tần suất sự kiện
        Max_duration=('duration', 'max'),      # Thời gian tối đa
        Mean_duration=('duration', 'mean'),    # Thời gian trung bình
    ).reset_index()

    # Tính tần suất sự kiện mỗi giờ
    grouped["Frequency"] = grouped["Count"] / total_hours

    # Tính tỷ lệ phần trăm (số lần xuất hiện / tổng số lần xuất hiện * 100)
    total_events = grouped["Count"].sum()
    grouped["Percentage"] = (grouped["Count"] / total_events) * 100

    # Thêm STUDY_PAT_ID
    grouped["STUDY_PAT_ID"] = study_pat_id 
    dfs.append(grouped)
    
sleep_data=pd.concat(dfs,ignore_index=True)
sleep_data.to_csv('../data/processed/features.csv')
# Merge 2 df lai voi nhau

features_df=pd.read_csv("data/processed/data_features.csv")
features_list=["Sleep stage W","Sleep stage N1","Sleep stage N2","Sleep stage N3","Sleep stage R","Obstructive Apnea","Obstructive Hypopnea","Mixed Apnea","Central Apnea","Oxygen Desaturation","EGG Arousal","Hypopnea"]
metric_columns = [
    'Count', 'Max_duration', 'Mean_duration', 
     'Frequency', 'Percentage'
]
agg_df = features_df.groupby(['STUDY_PAT_ID','sleep_stages'])[metric_columns].mean().reset_index()

# Tạo dictionary cho mỗi sự kiện/giai đoạn của mỗi bệnh nhân
def create_feature_dict(row):
    return {
        'Count': row['Count'],
        'Max_duration': row['Max_duration'],
        'Mean_duration': row['Mean_duration'],
        'Frequency': row['Frequency'],
        'Percentage': row['Percentage']
    }

# Áp dụng hàm để tạo dictionary
agg_df['feature_dict'] = agg_df.apply(create_feature_dict, axis=1)

# Pivot dữ liệu để mỗi bệnh nhân là một dòng, mỗi sự kiện/giai đoạn là một cột
pivot_df = agg_df.pivot_table(
    index='STUDY_PAT_ID',
    columns='sleep_stages',
    values='feature_dict',
    aggfunc='first'
).reset_index()

# Đảm bảo có đúng 12 cột đặc trưng
# Nếu thiếu, thêm cột với giá trị là dictionary rỗng
for feature in features_list:
    if feature not in pivot_df.columns:
        pivot_df[feature] = [{}] * len(pivot_df)

# Chỉ giữ lại các cột cần thiết (STUDY_PAT_ID và 12 đặc trưng)
final_df = pivot_df[['STUDY_PAT_ID'] + features_list]

# Lưu kết quả
label_df=pd.read_csv('data/processed/DIAGNOSIS_cleaned.csv')
final_df=pd.merge(final_df,label_df,on='STUDY_PAT_ID')
final_df.to_csv('data/processed/diagnosis_sleep_data', index=False)




