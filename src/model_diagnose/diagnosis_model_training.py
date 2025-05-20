import pandas as pd
import ast
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE    
from sklearn.metrics import accuracy_score
#Doc du lieu tu file csv theo chunk
processed_chunks = []
for chunk in pd.read_csv('data/processed/diagnosis_sleep_data.csv', chunksize=100000,low_memory=False):
    drop_columns=['Unnamed: 0','STUDY_PAT_ID','DX_CODE','DX_NAME']
    chunk=chunk.drop(drop_columns,axis=1)
    feature_cols = [col for col in chunk.columns if col != 'diag']
    new_cols = []
    for col in feature_cols:
        # Nếu dòng đó NaN → coi như dict rỗng {}
        def safe_parse(x):
            if pd.isna(x):
                return {}
            if isinstance(x, str):
                try:
                    return ast.literal_eval(x)
                except:
                    return {}
            return {}  # Nếu x không phải string thì cũng trả về dict rỗng luôn
        chunk[col] = chunk[col].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else ({})
        )
        temp_df = pd.DataFrame({
            f'{col}_count': chunk[col].apply(lambda x: x.get('Count', 0)),
            f'{col}_mean_duration': chunk[col].apply(lambda x: x.get('Mean_duration', 0)),
            f'{col}_max_duration': chunk[col].apply(lambda x: x.get('Max_duration', 0)),
            f'{col}_frequency': chunk[col].apply(lambda x: x.get('Frequency', 0)),
            f'{col}_percentage': chunk[col].apply(lambda x: x.get('Percentage', 0)),
        })
        
        # Thêm DataFrame phụ vào list
        new_cols.append(temp_df)

    chunk = pd.concat([chunk] + new_cols, axis=1)
    chunk = chunk.drop(columns=feature_cols)
    chunk = chunk.dropna(subset=['diag'])
    processed_chunks.append(chunk)
full_df = pd.concat(processed_chunks, ignore_index=True)

# Tách feature và label
X = full_df.drop(columns=['diag'])
y = full_df['diag']
#Xử lí mất cân bằng dữ liệu
X['diag'] = y 
# Chia lấy các hàng có giá trị 'diag' là 0, 1, 2, 3
df_majority = X[X['diag'] == 0.0]
df_minority_class_1 = X[X['diag'] == 1.0]
df_minority_class_2 = X[X['diag'] == 2.0]
df_minority_class_3 = X[X['diag'] == 3.0]

# Giảm mẫu lớp 0 để khớp với kích thước của lớp 1
df_majority_downsampled = df_majority.sample(n=len(df_minority_class_1), random_state=42)

# Kết hợp lại các lớp đã xử lý
df_balanced = pd.concat([df_majority_downsampled, df_minority_class_1,df_minority_class_2, df_minority_class_3])

# Trộn ngẫu nhiên lại dữ liệu
df_balanced = df_balanced.sample(frac=1, random_state=42)

# Chia dữ liệu thành X và y
y_balanced = df_balanced['diag']
X_balanced = df_balanced.drop(columns=['diag'])

#Sử dụng SMOTE để tạo mẫu cho lớp thiểu số
sm=SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X_balanced , y_balanced)
# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Mô hình huấn luyện
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred = model.predict(X_test)
train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc = accuracy_score(y_test, model.predict(X_test))
print(f"Train Accuracy: {train_acc:.2f}")
print(f"Test Accuracy: {test_acc:.2f}")
# Đánh giá mô hình
print(classification_report(y_test, y_pred))

from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np

train_sizes, train_scores, test_scores = learning_curve(
    model, X, y, cv=5, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 10))

train_mean = train_scores.mean(axis=1)
test_mean = test_scores.mean(axis=1)

plt.plot(train_sizes, train_mean, 'o-', label="Training score")
plt.plot(train_sizes, test_mean, 'o-', label="Validation score")
plt.xlabel("Training Set Size")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Learning Curve")
plt.show()
