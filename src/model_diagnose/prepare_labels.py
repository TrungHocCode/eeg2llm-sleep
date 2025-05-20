import pandas as pd
import re


#Doc data
df=pd.read_csv('data/raw/DIAGNOSIS.csv')
#Bo di nhung cot k su dung
drop_columns=['CLASS_OF_PROBLEM','CHRONIC_YN','PROV_ID','DX_ALT_CODE','DX_ENC_TYPE','DX_SOURCE_TYPE','STUDY_ENC_ID','DX_CODE_TYPE','Unnamed: 0.1','Unnamed: 0']
df=df.drop(drop_columns,axis=1)
df=df.dropna()
#Ham phan loai benh
def classify_diag(dx_code):
    if pd.isna(dx_code):
        return 0
    dx_code = str(dx_code)
    if re.match(r'^(?:G40\.[0-9]|G40\.901|G40\.909|R56\.0|R56\.1|R56\.9|345\.(?:0[0-9]|[1-8][0-9]|9[0-1])|780\.39|780\.31)', dx_code):
        return 1
    elif re.match(r'^(?:G47\.[0-4]|G47\.[8-9]|780\.5[0-9]|307\.4[0-9])', dx_code):
        return 2
    elif re.match(r'^(?:I63\.[0-9]|433\.\d{2}|434\.\d{2}|436)', dx_code):
        return 3
    return 0

df['diag']=df['DX_CODE'].apply(classify_diag)

print(df.head())
df.to_csv('data/processed/DIAGNOSIS_cleaned.csv')