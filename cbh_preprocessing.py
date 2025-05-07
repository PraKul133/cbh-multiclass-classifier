import pandas as pd
import re
from sklearn.utils import resample

# Define standard target class names
target_classes = ['Sexual', 'Religion', 'Political', 'Troll', 'Threats', 'Ethnicity', 'Vocational', 'Racism']

# Step 1: Load the dataset
df = pd.read_csv("Balanced_CBH_Dataset_Preprocessed.csv")

# Step 2: Clean the text column
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@\w+|\#','', text)
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['Text'] = df['Text'].apply(clean_text)

# Step 3: Drop empty or very short text
df.dropna(subset=['Text', 'Types'], inplace=True)
df = df[df['Text'].str.len() > 5]

# Step 4: Normalize and filter target classes
def normalize_label(label):
    label = label.lower().strip()
    for target in target_classes:
        if target.lower() in label:
            return target
    return None

df['Types'] = df['Types'].apply(normalize_label)
df = df[df['Types'].notnull()]  # keep only target classes

# ✅ Step 4.1: Add binary label for ROC-AUC analysis (Threat vs Non-Threat)
df['Threat_Label'] = df['Types'].apply(lambda x: 1 if x == 'Threats' else 0)

# Step 5: Sample exactly 500 examples per class
target_samples = 500
balanced_df = pd.DataFrame()

for label in target_classes:
    class_df = df[df['Types'] == label]
    if len(class_df) < target_samples:
        resampled = resample(class_df, replace=True, n_samples=target_samples, random_state=42)
    else:
        resampled = resample(class_df, replace=False, n_samples=target_samples, random_state=42)
    balanced_df = pd.concat([balanced_df, resampled])

# Step 6: Shuffle and save
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
balanced_df.to_csv("Final_CBH_Balanced_500_Each_Class.csv", index=False)

print("✅ Final dataset saved as 'Final_CBH_Balanced_500_Each_Class.csv'")
print("📊 Final class distribution:\n", balanced_df['Types'].value_counts())
print("🔐 Threat vs Non-Threat distribution:\n", balanced_df['Threat_Label'].value_counts())
