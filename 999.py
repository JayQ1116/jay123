from ucimlrepo import fetch_ucirepo
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno  # For visualizing missing values
# 1. Load the dataset
chronic_kidney_disease = fetch_ucirepo(id=336)

# 2. Get features and target
X = chronic_kidney_disease.data.features
y = chronic_kidney_disease.data.targets

# 3. Combine features and target for unified analysis
df = pd.concat([X, y], axis=1)
# 2. Get features and target
X = chronic_kidney_disease.data.features
y = chronic_kidney_disease.data.targets

# 3. Combine features and target for unified analysis
df = pd.concat([X, y], axis=1)

# **Modification: Binary classification for target variable**
df['class'] = df['class'].apply(lambda x: 'ckd' if x == 'ckd' else 'notckd').astype('category')

# ===== Figure 1: Data Preview =====
n_rows, n_cols = df.head().shape
fig_width = 18  # Set width to 18 inches
fig_height = max(8, n_rows * 1.2)
fig1, ax = plt.subplots(figsize=(16, 5))
ax.axis('off')
table = ax.table(cellText=df.head().values,
                colLabels=df.columns,
                cellLoc='center',
                loc='center')

table.set_fontsize(28) # Set font size to 28
table.scale(1, 2.5)
plt.title("Data Preview", fontsize=14)
plt.tight_layout(pad=0)
plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.05)
plt.show()




# Manually specify continuous variables
continuous_vars = ['age', 'bp', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wbcc', 'rbcc']

# Force these variables to be numeric
for var in continuous_vars:
    if var in df.columns:
        df[var] = pd.to_numeric(df[var], errors='coerce')  # Set non-convertible values to NaN

# Manually specify categorical variables
categorical_vars = ['sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane', 'class']

# Ensure these variables are categorical
for var in categorical_vars:
    if var in df.columns:
        df[var] = df[var].astype('category')

# Reclassify variables
cat_cols = df.select_dtypes(include=['category']).columns
num_cols = df.select_dtypes(include=['float64', 'int64']).columns

print("\nCategorical variables:", list(cat_cols))
print("Numerical variables:", list(num_cols))

# Ensure 'su' is in categorical variables
cat_cols = list(cat_cols)
num_cols = list(num_cols)

var_types = pd.DataFrame({
    'Variable': cat_cols + num_cols,
    'Type': ['Categorical'] * len(cat_cols) + ['Continuous'] * len(num_cols)
})


def type_color(t):
    return '#FFA07A' if t == 'Categorical' else '#87CEFA'  # 橙色和蓝色

colors = [type_color(t) for t in var_types['Type']]


fig, ax = plt.subplots(figsize=(8, len(var_types)*0.4))
ax.axis('off')

# Create a table with variable names and types
table = ax.table(
    cellText=var_types.values,
    colLabels=var_types.columns,
    cellLoc='center',
    loc='center',
    colColours=['lightgrey', 'lightgrey'],
)

# Set the color for each cell based on variable type
for i, color in enumerate(colors):
    table[(i+1, 1)].set_facecolor(color)

# Adjust table properties
table.auto_set_font_size(False)
table.set_fontsize(12)
table.auto_set_column_width([0, 1])

plt.title("Variables and Their Types (Colored by Type)", fontsize=14)
plt.savefig('figure.png', dpi=300)
plt.show()


# Check if 'su' is correctly classified
if 'su' in cat_cols:
    print("\n'su' is correctly classified as a categorical variable.")
else:
    print("\nWarning: 'su' is not classified as a categorical variable.")

# 4. Basic structure of the dataset
print("Data shape:", df.shape)
print("\nFirst 5 rows of data:")
print(df.head())

print("\nData types and missing info:")
print(df.info())

# 5. Count missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# 6. Visualize missing values
plt.figure(figsize=(10, 4))
msno.bar(df)
plt.title("Missing Values - Bar Chart")
plt.savefig('figure.png', dpi=300)
plt.show()

plt.figure(figsize=(10, 4))
msno.matrix(df)
plt.title("Missing Values - Matrix")
plt.savefig('figure.png', dpi=300)
plt.show()
# 7. Separate categorical and numerical variables
cat_cols = df.select_dtypes(include=['object']).columns
num_cols = df.select_dtypes(include=['float64', 'int64']).columns

print("\nCategorical variables:", list(cat_cols))
print("Numerical variables:", list(num_cols))

# 8. Summary statistics for numerical features
print("\nSummary statistics for numerical features:")
print(df[num_cols].describe())

# 9. Frequency counts for categorical variables
for col in cat_cols:
    print(f"\nFrequency distribution of {col}:")
    print(df[col].value_counts(dropna=False))

# 10. Target variable distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='class', data=df)
plt.title("Target Variable: CKD Class Distribution")
plt.savefig('figure.png', dpi=300)
plt.show()

# 11. Boxplots for numerical features by CKD class (to detect outliers)
plt.figure(figsize=(15, 12))
for i, feature in enumerate(num_cols):
    plt.subplot(4, 4, i+1)
    sns.boxplot(x='class', y=feature, data=df)
    plt.title(f'{feature} by CKD Class - Boxplot')
plt.tight_layout()
plt.savefig('figure.png', dpi=300)
plt.show()

# 12. Heatmap of correlations between numerical features
plt.figure(figsize=(12, 10))
corr = df[num_cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap of Numerical Features")
plt.savefig('figure.png', dpi=300)
plt.show()

# 15. Boxplots for all continuous variables by class
plt.figure(figsize=(15, 12))
for i, col in enumerate(num_cols):
    plt.subplot((len(num_cols) + 2) // 3, 3, i + 1)
    sns.boxplot(x='class', y=col, data=df)
    plt.title(f'{col} by class')
plt.tight_layout()
plt.savefig('figure.png', dpi=300)
plt.show()

# Histograms for all continuous variables
plt.figure(figsize=(18, 12))
for i, col in enumerate(num_cols):
    plt.subplot((len(num_cols) + 2) // 3, 3, i + 1)
    sns.histplot(df[col], kde=True, bins=30)  # Histogram with KDE
    plt.title(f'{col} - Histogram')
plt.tight_layout()
plt.savefig('figure.png', dpi=300)
plt.show()


# Reconfirm categorical variables
cat_cols = df.select_dtypes(include=['category']).columns

print("\nUnique values in 'dm':")
print(df['dm'].unique())
# Clean 'dm' column values
df['dm'] = df['dm'].replace({'\tno': 'no', '\tyes': 'yes', ' yes': 'yes', '': None})
df['dm'] = df['dm'].astype('category')
# 16. Frequency plots for all categorical variables
plt.figure(figsize=(18, 12))
for i, col in enumerate(cat_cols):
    plt.subplot((len(cat_cols) + 2) // 3, 3, i + 1)
    df[col].value_counts(dropna=False).plot(kind='bar')
    plt.title(f'{col} (Frequency)')
plt.tight_layout()
plt.savefig('figure.png', dpi=300)
plt.show()


# 17. Bar plots for all categorical variables grouped by class
plt.figure(figsize=(18, 12))
for i, col in enumerate(cat_cols):
    plt.subplot((len(cat_cols) + 2) // 3, 3, i + 1)
    sns.countplot(x=col, hue='class', data=df)
    plt.title(f'{col} by class')
    plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('figure.png', dpi=300)
plt.show()


# 18. Variable description summary and export
default_desc = {
    'age': 'Age of the patient', 'bp': 'Blood pressure', 'sg': 'Specific gravity',
    'al': 'Albumin', 'su': 'Sugar', 'rbc': 'Red blood cells', 'pc': 'Pus cell',
    'pcc': 'Pus cell clumps', 'ba': 'Bacteria', 'bgr': 'Blood glucose random',
    'bu': 'Blood urea', 'sc': 'Serum creatinine', 'sod': 'Sodium', 'pot': 'Potassium',
    'hemo': 'Hemoglobin', 'pcv': 'Packed cell volume', 'wc': 'White blood cell count',
    'rc': 'Red blood cell count', 'htn': 'Hypertension', 'dm': 'Diabetes mellitus',
    'cad': 'Coronary artery disease', 'appet': 'Appetite', 'pe': 'Pedal edema',
    'ane': 'Anemia', 'class': 'Chronic kidney disease diagnosis label'
}

def get_var_type(series):
    return 'Continuous' if pd.api.types.is_numeric_dtype(series) else 'Categorical'

summary = pd.DataFrame({
    'Variable Name': df.columns,
    'Sample Count (#)': df.notna().sum().values,
    'Missing Count (Na)': df.isna().sum().values,
    'Type': [get_var_type(df[col]) for col in df.columns],
    'Description': [default_desc.get(col, 'NA') for col in df.columns]
})

# Print summary
print(summary)

# Plot variable description table
fig, ax = plt.subplots(figsize=(12, len(summary) * 0.5))  # Dynamically adjust table height
ax.axis('off')  # Turn off axis

# Create table
table = ax.table(cellText=summary.values,
                 colLabels=summary.columns,
                 cellLoc='center',
                 loc='center')

# Adjust table style
table.auto_set_font_size(False)
table.set_fontsize(10)
table.auto_set_column_width(col=list(range(len(summary.columns))))

plt.title("Variable Description Summary", fontsize=14)
plt.savefig('figure.png', dpi=300)
plt.show()


# Save variable description to CSV
summary.to_csv('variable_summary.csv', index=False, encoding='utf-8-sig')
print("Variable description has been saved to 'variable_summary.csv'.")

# Check for anomalies in categorical variables
for col in cat_cols:
    print(f"\nColumn: {col}")
    print(df[col].value_counts(dropna=False))


import matplotlib.pyplot as plt
import missingno as msno
import pandas as pd
from ucimlrepo import fetch_ucirepo  
from sklearn.impute import KNNImputer
import pandas as pd
from sklearn import set_config
set_config(display='text')

chronic_kidney_disease = fetch_ucirepo(id=336)
df = pd.concat([chronic_kidney_disease.data.features, chronic_kidney_disease.data.targets], axis=1)  # 补全括号

# ========== Modification: Target values ​​are encoded as 0 and 1 ==========
df['class'] = df['class'].apply(lambda x: 1 if x == 'ckd' else 0)

# ========== 1. Backup the original dataframe ==========

df_original = df.copy()

# ========== 2. Missing value handling ==========

# KNN imputation for numerical columns
knn_imputer = KNNImputer(n_neighbors=5)
df[num_cols] = knn_imputer.fit_transform(df[num_cols])

# Mode imputation for categorical columns
for col in cat_cols:
    if df[col].isnull().sum() > 0:
        mode_val = df[col].mode()[0]
        df[col].fillna(mode_val, inplace=True)

# ========== 3. Missing value visualization (Before vs After) ==========

for col in cat_cols:
    if df[col].isnull().sum() > 0:
        mode_val = df[col].mode()[0]
        df[col].fillna(mode_val, inplace=True)

# ========== 3. Missing value visualization (Before vs After) ==========

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

msno.matrix(df_original, ax=axes[0, 0])
axes[0, 0].set_title('Missing Value Matrix - Before Imputation')

msno.bar(df_original, ax=axes[0, 1])
axes[0, 1].set_title('Missing Values Per Column - Before Imputation')

msno.matrix(df, ax=axes[1, 0])
axes[1, 0].set_title('Missing Value Matrix - After Imputation')

msno.bar(df, ax=axes[1, 1])
axes[1, 1].set_title('Missing Values Per Column - After Imputation')

plt.tight_layout()
plt.savefig('figure.png', dpi=300)
plt.show()

#========== 4. Comparison before and after One-Hot ==========

# ========== Modification: Remove the target value 'class' from the categorical variable ===========
categorical_vars = ['sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']

# 1. Count the number of unique values ​​of each categorical variable (determine the number of new columns after One-Hot)
category_unique_counts = {col: df[col].nunique() for col in categorical_vars}

# 2. One-Hot 인코딩 수행(다중공선성을 피하기 위해 drop_first=True)
df_encoded = pd.get_dummies(df, columns=categorical_vars, drop_first=True)

# 3. 각 원래 범주형 변수를 확장한 후 열의 수를 계산합니다.
encoded_columns_per_var = {}
for col in categorical_vars:
    encoded_columns = [c for c in df_encoded.columns if c.startswith(col + '_')]
    encoded_columns_per_var[col] = len(encoded_columns)

# 4. 각 변수에 대한 확장된 세부 정보를 보여주는 쌓인 막대형 차트를 그립니다.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

# 하위 그림 1: 인코딩 전후 총 기능 수 비교
ax1.bar(['Before Encoding', 'After Encoding'],
        [len(categorical_vars), df_encoded.shape[1] - (df.shape[1] - len(categorical_vars))],
        color=['skyblue', 'salmon'])
ax1.set_title("Total Features Before vs After One-Hot Encoding")
ax1.set_ylabel("Number of Features")

# 하위 그림 2: 각 범주형 변수에 대한 확장된 세부 정보
categories = list(encoded_columns_per_var.keys())
before = [1] * len(categories)  
after = [encoded_columns_per_var[col] for col in categories]

ax2.bar(categories, before, label='Before Encoding', color='skyblue')
ax2.bar(categories, after, bottom=before, label='After Encoding', color='salmon')
ax2.set_title("Feature Expansion per Categorical Variable")
ax2.set_ylabel("Number of Features")
ax2.tick_params(axis='x', rotation=45)
ax2.legend()

plt.tight_layout()
plt.savefig('figure.png', dpi=300)
plt.show()
# ========== 5. 표준화, 정규화, 비교 차트 ==========
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# df가 원래 DataFrame이고, continuous_vars가 정의되어 있다고 가정합니다.
continuous_vars = ['age', 'bp', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wbcc', 'rbcc']

# 1. 전처리: 원본 데이터 백업 및 누락된 값 처리
df_clean = df.copy()
df_clean[continuous_vars] = df_clean[continuous_vars].apply(pd.to_numeric, errors='coerce')
df_clean[continuous_vars] = df_clean[continuous_vars].fillna(df_clean[continuous_vars].median())

# 2. 표준화 및 정규화
scaler_standard = StandardScaler()
df_standardized = df_clean.copy()
df_standardized[continuous_vars] = scaler_standard.fit_transform(df_clean[continuous_vars])

scaler_minmax = MinMaxScaler()
df_normalized = df_clean.copy()
df_normalized[continuous_vars] = scaler_minmax.fit_transform(df_clean[continuous_vars])

# 3. 치료 전후 모든 연속변수의 비교 차트를 그린다(최적화 크기)
plt.figure(figsize=(18, 18))  
for i, col in enumerate(continuous_vars, 1):
   # 원시 데이터
    plt.subplot(len(continuous_vars), 3, 3*i-2)
    sns.histplot(df_clean[col], kde=True, color='blue', bins=15) 
    plt.title(f'Original {col}\nRange: [{df_clean[col].min():.1f}, {df_clean[col].max():.1f}]', fontsize=8)
    plt.xlabel('') 

    # 표준화 후
    plt.subplot(len(continuous_vars), 3, 3*i-1)
    sns.histplot(df_standardized[col], kde=True, color='green', bins=15)
    plt.title(f'Standardized {col}\n(Mean=0, Std=1)', fontsize=8)
    plt.xlabel('')

    # 정규화 후
    plt.subplot(len(continuous_vars), 3, 3*i)
    sns.histplot(df_normalized[col], kde=True, color='red', bins=15)
    plt.title(f'Normalized {col}\nRange: [0, 1]', fontsize=8)
    plt.xlabel('')

plt.tight_layout(pad=1.5)  
plt.savefig('figure.png', dpi=300)
plt.show()


# 4. 연속 변수의 분포 비교 차트(최적화 크기)
plt.figure(figsize=(18, 6))  
df_melted = pd.concat([
    df_clean[continuous_vars].assign(Processing='Original'),
    df_standardized[continuous_vars].assign(Processing='Standardized'),
    df_normalized[continuous_vars].assign(Processing='Normalized')
])
df_melted = pd.melt(df_melted, id_vars='Processing', var_name='Variable', value_name='Value')

sns.boxplot(x='Variable', y='Value', hue='Processing', data=df_melted,
            palette={'Original': 'blue', 'Standardized': 'green', 'Normalized': 'red'},
            width=0.6)  
plt.title('Distribution Comparison: Original vs Standardized vs Normalized', fontsize=10)
plt.xticks(rotation=45, fontsize=8)  
plt.yticks(fontsize=8)
plt.xlabel('Variable', fontsize=9)
plt.ylabel('Value', fontsize=9)
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig('figure.png', dpi=300)
plt.show()
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE
# 2. Extract features and target variable
X = df.drop('class', axis=1)
y = df['class']

# 3. One-Hot Encoding (apply to all non-numeric columns, drop_first=True to avoid multicollinearity)
X_encoded = pd.get_dummies(X, drop_first=True)
df_encoded = pd.get_dummies(df_clean, drop_first=True)
# 4. Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# ---------- Handling Class Imbalance ----------
def visualize_class_distribution(y_values, title):
    """Plot class distribution"""
    plt.figure(figsize=(8, 6))
    sns.countplot(x=y_values, palette='pastel')
    plt.title(title)
    plt.savefig('figure.png', dpi=300)
    plt.show()


# ① Under Sampling
rus = RandomUnderSampler(random_state=42)
X_train_under, y_train_under = rus.fit_resample(X_train, y_train)
visualize_class_distribution(y_train_under, "Target Distribution After Under Sampling")

# ② Over Sampling
ros = RandomOverSampler(random_state=42)
X_train_over, y_train_over = ros.fit_resample(X_train, y_train)
visualize_class_distribution(y_train_over, "Target Distribution After Over Sampling")

# ③ SMOTE（Synthetic Minority Over-sampling Technique）
sm = SMOTE(random_state=42)
X_train_smote, y_train_smote = sm.fit_resample(X_train, y_train)
visualize_class_distribution(y_train_smote, "Target Distribution After SMOTE Sampling")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.model_selection import validation_curve, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
warnings.filterwarnings('ignore')

# 결과 폴더 생성
os.makedirs('result', exist_ok=True)

# 다중 프로세스 문제를 피하기 위해 환경 변수를 설정하세요
os.environ['LOKY_MAX_CPU_COUNT'] = '1'

#고급 스타일 설정
plt.style.use('default')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['axes.edgecolor'] = 'black'

# 색상 구성표 정의
model_colors = {
    'Random Forest': '#4A90E2',
    'Logistic Regression': '#50C878',
    'SVM': '#FFB347',
    'Decision Tree': '#FF6B6B',
    'XGBoost': '#9B59B6'
}

# 결과 파일 초기화
result_file = 'result/result.txt'
with open(result_file, 'w', encoding='utf-8') as f:
    f.write("머신러닝 모델 분석 결과\n")
    f.write("="*80 + "\n\n")

def log_result(text, print_text=True):
    """결과를 동시에 인쇄하고 파일에 저장"""
    if print_text:
        print(text)
    with open(result_file, 'a', encoding='utf-8') as f:
        f.write(text + "\n")

log_result(" 완전한 머신 러닝 Pipeline 분석을 시작하세요...")

# ==================== Part I: Initial Random Forest Analysis ====================
log_result("\n" + "="*80)
log_result("Part I: Random Forest Baseline Analysis Using All Features")
log_result("="*80)

# Separate features and target variable
X = df_encoded.drop(['class'], axis=1)
if '' in X.columns:
    X = X.drop([''], axis=1)
y = df_encoded['class']

log_result(f"Number of features: {X.shape[1]}")
log_result(f"Number of samples: {X.shape[0]}")

# Split train and test set (using all features)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Train Random Forest with all features
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Prediction
y_pred = rf.predict(X_test)
y_pred_proba = rf.predict_proba(X_test)[:, 1]

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
log_result(f"Model accuracy using all features: {accuracy:.4f}")

# Get feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

# Create advanced dashboard
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
fig.suptitle('Random Forest Analysis Dashboard', fontsize=22, fontweight='bold', y=0.98)

# Color scheme
primary_color = '#4A90E2'
secondary_color = '#FF6B6B'
accent_color = '#50C878'

# 1. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
            cbar=False, square=True,
            annot_kws={'fontsize': 20, 'fontweight': 'bold'},
            linewidths=3, linecolor='black')

ax1.set_title('Confusion Matrix', fontsize=18, fontweight='bold', pad=20)
ax1.set_xlabel('Predicted', fontsize=16, fontweight='bold')
ax1.set_ylabel('Actual', fontsize=16, fontweight='bold')
ax1.tick_params(colors='black', which='both', labelsize=14)
ax1.set_facecolor('#FAFAFA')

for spine in ax1.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(2)

# 2. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

ax2.plot(fpr, tpr, color=primary_color, lw=4,
         label=f'ROC Curve (AUC = {roc_auc:.3f})',
         marker='o', markersize=6, markevery=10)
ax2.plot([0, 1], [0, 1], color='gray', lw=3, linestyle='--',
         alpha=0.8, label='Random Classifier')

ax2.fill_between(fpr, tpr, alpha=0.2, color=primary_color)

ax2.set_xlim([-0.02, 1.02])
ax2.set_ylim([-0.02, 1.05])
ax2.set_xlabel('False Positive Rate', fontsize=16, fontweight='bold')
ax2.set_ylabel('True Positive Rate', fontsize=16, fontweight='bold')
ax2.set_title('ROC Curve Analysis', fontsize=18, fontweight='bold', pad=20)
ax2.legend(loc="lower right", frameon=True, fancybox=True, shadow=True)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_facecolor('#FAFAFA')

ax2.text(0.6, 0.3, f'AUC = {roc_auc:.3f}', fontsize=16, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8))

for spine in ax2.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(2)

# 3. Feature Importance (Top 20)
top_20_importance = feature_importance.head(20)

colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_20_importance)))
bars = ax3.barh(range(len(top_20_importance)), top_20_importance['importance'],
                color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

for i, (bar, value) in enumerate(zip(bars, top_20_importance['importance'])):
    ax3.text(value + max(top_20_importance['importance']) * 0.01, i, f'{value:.3f}',
             va='center', fontsize=12, fontweight='bold')

ax3.set_yticks(range(len(top_20_importance)))
ax3.set_yticklabels(top_20_importance['feature'], fontsize=12)
ax3.set_xlabel('Feature Importance', fontsize=16, fontweight='bold')
ax3.set_title('Top 20 Feature Importance', fontsize=18, fontweight='bold', pad=20)
ax3.grid(True, axis='x', alpha=0.3, linestyle='--')
ax3.set_facecolor('#FAFAFA')
ax3.invert_yaxis()

for spine in ax3.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(2)

if len(top_20_importance) > 0:
    ax3.axhline(y=0, color='red', linewidth=3, alpha=0.7)

plt.tight_layout()
plt.subplots_adjust(top=0.90)
plt.savefig('result/random_forest_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

# Record baseline results
log_result("\nClassification Report:")
log_result(classification_report(y_test, y_pred))
log_result(f"ROC AUC Score: {roc_auc:.4f}")

log_result("\nTop 20 most important features:")
log_result(str(feature_importance.head(20)))

top_20_features = feature_importance.head(20)['feature'].tolist()
log_result(f"\nTop 20 feature names: {top_20_features}")

# ==================== Part II: Multi-Model Optimization Analysis ====================
log_result("\n" + "="*80)
log_result("Part II: Multi-Model Optimization Analysis Using Selected Features")
log_result("="*80)

# Use the top 20 selected features
X_selected = df_encoded[top_20_features]
y = df_encoded['class']

log_result(f"Number of features used: {X_selected.shape[1]}")
log_result(f"Number of samples: {X_selected.shape[0]}")
log_result(f"Target variable distribution:\n{y.value_counts()}")

# Split train and test set
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42, stratify=y)

# ==================== Stage 1: Initial Grid Search ====================
log_result("\n" + "="*80)
log_result("Stage 1: Initial Grid Search for Rough Best Parameters")
log_result("="*80)

# Define models
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

# Initial grid search (coarse tuning)
initial_param_grids = {
    "Random Forest": {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5]
    },
    "Logistic Regression": {
        'classifier__C': [0.1, 1, 10],
        'classifier__penalty': ['l1', 'l2'],
        'classifier__solver': ['liblinear']
    },
    "SVM": {
        'classifier__C': [0.1, 1, 10],
        'classifier__kernel': ['rbf', 'linear']
    },
    "Decision Tree": {
        'classifier__max_depth': [5, 10, 20, None],
        'classifier__min_samples_split': [2, 5, 10]
    },
    "XGBoost": {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [3, 6, 9],
        'classifier__learning_rate': [0.01, 0.1, 0.2]
    }
}

# Store initial results
initial_results = {}
initial_best_params = {}

# Initial grid search for each model
for name, model in models.items():
    log_result(f"\nInitial search: {name}")

    # Create Pipeline
    if name in ["Logistic Regression", "SVM"]:
        pipeline = ImbPipeline([
            ('smote', SMOTE(random_state=42)),
            ('scaler', StandardScaler()),
            ('classifier', model)
        ])
    else:
        pipeline = ImbPipeline([
            ('smote', SMOTE(random_state=42)),
            ('classifier', model)
        ])

    # Grid search
    grid_search = GridSearchCV(pipeline, initial_param_grids[name], cv=5, scoring='accuracy', n_jobs=1)
    grid_search.fit(X_train, y_train)

    initial_best_params[name] = grid_search.best_params_
    initial_results[name] = grid_search.best_score_

    log_result(f"  Initial best params: {grid_search.best_params_}")
    log_result(f"  Initial best CV score: {grid_search.best_score_:.4f}")

# ==================== Stage 2: Validation Curve Fine-tuning of Key Parameters ====================
log_result("\n" + "="*80)
log_result("Stage 2: Validation Curve Fine-tuning of Key Parameters")
log_result("="*80)

# Define the most important parameter and fine-tuning range for each model
key_params = {
    "Random Forest": {
        'param_name': 'classifier__n_estimators',
        'param_range': [50, 100, 150, 200, 250, 300, 350, 400],
        'description': 'Number of Trees (n_estimators)'
    },
    "Logistic Regression": {
        'param_name': 'classifier__C',
        'param_range': [0.01, 0.1, 0.5, 1, 2, 5, 10, 20],
        'description': 'Regularization Strength (C)'
    },
    "SVM": {
        'param_name': 'classifier__C',
        'param_range': [0.01, 0.1, 0.5, 1, 2, 5, 10, 20],
        'description': 'Regularization Strength (C)'
    },
    "Decision Tree": {
        'param_name': 'classifier__max_depth',
        'param_range': [3, 5, 7, 10, 15, 20, 25, 30],
        'description': 'Max Depth (max_depth)',
        'special_handling': True
    },
    "XGBoost": {
        'param_name': 'classifier__n_estimators',
        'param_range': [50, 100, 150, 200, 250, 300, 350, 400],
        'description': 'Number of Trees (n_estimators)'
    }
}

# Store validation curve results and final parameters
final_params = {}
validation_results = {}

# Create validation curve plots
fig, axes = plt.subplots(2, 3, figsize=(22, 16))
fig.suptitle('Validation Curves for Key Parameters', fontsize=22, fontweight='bold', y=0.98)

for i, (name, model) in enumerate(models.items()):
    row, col = i // 3, i % 3
    ax = axes[row, col]

    log_result(f"\nFine-tuning parameter: {name} - {key_params[name]['description']}")

    # Create pipeline based on initial best params
    if name in ["Logistic Regression", "SVM"]:
        base_pipeline = ImbPipeline([
            ('smote', SMOTE(random_state=42)),
            ('scaler', StandardScaler()),
            ('classifier', model)
        ])
    else:
        base_pipeline = ImbPipeline([
            ('smote', SMOTE(random_state=42)),
            ('classifier', model)
        ])

    # Set initial best params (except the one to be tuned)
    base_params = initial_best_params[name].copy()
    param_to_tune = key_params[name]['param_name']
    if param_to_tune in base_params:
        del base_params[param_to_tune]

    base_pipeline.set_params(**base_params)

    # Compute validation curve
    param_range = key_params[name]['param_range']
    train_scores, val_scores = validation_curve(
        base_pipeline, X_train, y_train,
        param_name=param_to_tune, param_range=param_range,
        cv=5, scoring='accuracy', n_jobs=1
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    # Plot validation curve
    ax.fill_between(param_range, train_mean - train_std, train_mean + train_std,
                    alpha=0.2, color=model_colors[name])
    ax.fill_between(param_range, val_mean - val_std, val_mean + val_std,
                    alpha=0.2, color='orange')

    ax.plot(param_range, train_mean, 'o-', color=model_colors[name],
            linewidth=3, markersize=8, label='Training Score')
    ax.plot(param_range, val_mean, 's-', color='orange',
            linewidth=3, markersize=8, label='Validation Score')

    # Find best parameter value
    best_idx = np.argmax(val_mean)
    best_param_value = param_range[best_idx]
    best_score = val_mean[best_idx]

    # For Decision Tree, if a very large value is selected, set to None
    if name == "Decision Tree" and best_param_value >= 25:
        final_best_param_value = None
        display_value = f"{best_param_value} (→None)"
    else:
        final_best_param_value = best_param_value
        display_value = str(best_param_value)

    # Annotate best point
    ax.annotate(f'Best: {display_value}\nScore: {best_score:.3f}',
                xy=(best_param_value, best_score),
                xytext=(10, 10), textcoords='offset points',
                fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

    # Store final params
    final_params[name] = base_params.copy()
    final_params[name][param_to_tune] = final_best_param_value
    validation_results[name] = best_score

    ax.set_xlabel(key_params[name]['description'], fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=14, fontweight='bold')
    ax.set_title(f'{name}', fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='lower right', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_facecolor('#FAFAFA')
    ax.tick_params(labelsize=12)

    # Add border
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(2)

    log_result(f"  Best {key_params[name]['description']}: {final_best_param_value}")
    log_result(f"  Validation score: {best_score:.4f}")

# Hide extra subplots
axes[1, 2].set_visible(False)

plt.tight_layout()
plt.subplots_adjust(top=0.93)
plt.savefig('result/validation_curves.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

# ==================== Stage 3: Train Models with Final Parameters ====================
log_result("\n" + "="*80)
log_result("Stage 3: Train Models with Final Parameters")
log_result("="*80)

# Store final results
final_results = {}
final_models = {}

for name, model in models.items():
    log_result(f"\nFinal training: {name}")

    # Create Pipeline
    if name in ["Logistic Regression", "SVM"]:
        final_pipeline = ImbPipeline([
            ('smote', SMOTE(random_state=42)),
            ('scaler', StandardScaler()),
            ('classifier', model)
        ])
    else:
        final_pipeline = ImbPipeline([
            ('smote', SMOTE(random_state=42)),
            ('classifier', model)
        ])

    # Set final params
    final_pipeline.set_params(**final_params[name])

    # Train model
    final_pipeline.fit(X_train, y_train)

    # Predict
    y_pred = final_pipeline.predict(X_test)
    y_pred_proba = final_pipeline.predict_proba(X_test)[:, 1]
    test_accuracy = accuracy_score(y_test, y_pred)

    # Cross-validation
    cv_scores = cross_val_score(final_pipeline, X_train, y_train, cv=5, scoring='accuracy')

    # Store results
    final_results[name] = {
        'final_params': final_params[name],
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'test_accuracy': test_accuracy,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'pipeline': final_pipeline
    }
    final_models[name] = final_pipeline

    log_result(f"  Final params: {final_params[name]}")
    log_result(f"  Cross-validation: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    log_result(f"  Test accuracy: {test_accuracy:.4f}")

# ==================== Results Comparison Analysis ====================
log_result("\n" + "="*80)
log_result("Comparison of Parameter Optimization Effects")
log_result("="*80)

# Create comparison table
comparison_df = pd.DataFrame({
    'Model': list(models.keys()),
    'Initial_CV_Score': [initial_results[name] for name in models.keys()],
    'Validation_CV_Score': [validation_results[name] for name in models.keys()],
    'Final_Test_Score': [final_results[name]['test_accuracy'] for name in models.keys()],
    'Improvement': [validation_results[name] - initial_results[name] for name in models.keys()]
})

log_result("\nComparison of parameter optimization effects:")
log_result(str(comparison_df.round(4)))

# Calculate detailed evaluation metrics
detailed_metrics = {}
for name, result in final_results.items():
    y_pred = result['y_pred']
    detailed_metrics[name] = {
        'Accuracy': result['test_accuracy'],
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred)
    }

metrics_df = pd.DataFrame(detailed_metrics).T
log_result("\nFinal model detailed metrics:")
log_result(str(metrics_df.round(4)))

# Visualization of optimization effect (only before/after comparison, improvement plot removed)
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

# Before and after optimization comparison
x_pos = np.arange(len(comparison_df))
width = 0.35

bars1 = ax.bar(x_pos - width/2, comparison_df['Initial_CV_Score'], width,
        label='Initial Grid Search', alpha=0.8, color='lightblue', edgecolor='black')
bars2 = ax.bar(x_pos + width/2, comparison_df['Final_Test_Score'], width,
        label='Final Optimized', alpha=0.8, color='lightgreen', edgecolor='black')

# Annotate values
for i, (initial, final) in enumerate(zip(comparison_df['Initial_CV_Score'], comparison_df['Final_Test_Score'])):
    ax.text(i - width/2, initial + 0.005, f'{initial:.3f}', ha='center', va='bottom', fontweight='bold')
    ax.text(i + width/2, final + 0.005, f'{final:.3f}', ha='center', va='bottom', fontweight='bold')

ax.set_xlabel('Models', fontsize=14, fontweight='bold')
ax.set_ylabel('Accuracy', fontsize=14, fontweight='bold')
ax.set_title('Before vs After Parameter Optimization', fontsize=16, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
ax.legend(loc='lower left')
ax.grid(True, axis='y', alpha=0.3)
ax.set_facecolor('#FAFAFA')

# Add border
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(2)

plt.tight_layout()
plt.savefig('result/parameter_optimization_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

# Set results to final_results
results = final_results

# ==================== Full Visualization Analysis ====================
log_result("\n" + "="*80)
log_result("Start generating full visualization analysis")
log_result("="*80)

# ==================== Figure 1: Confusion Matrix (2×3) ====================
fig1, axes = plt.subplots(2, 3, figsize=(22, 14))
fig1.suptitle('Confusion Matrix Analysis', fontsize=22, fontweight='bold', y=0.98)

model_names = list(results.keys())
for i, name in enumerate(model_names):
    row, col = i // 3, i % 3
    ax = axes[row, col]

    cm = confusion_matrix(y_test, results[name]['y_pred'])

    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                ax=ax, cbar=False, square=True,
                annot_kws={'fontsize': 18, 'fontweight': 'bold'},
                linewidths=2, linecolor='black')

    ax.set_title(f'{name}', fontsize=18, fontweight='bold', pad=25)
    ax.set_xlabel('Predicted', fontsize=16, fontweight='bold')
    ax.set_ylabel('Actual', fontsize=16, fontweight='bold')
    ax.tick_params(colors='black', which='both', labelsize=14)

    # Add border
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(2)

# Hide extra subplot
axes[1, 2].set_visible(False)

plt.tight_layout()
plt.subplots_adjust(top=0.93)
plt.savefig('result/confusion_matrices.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

# ==================== Figure 2: Learning Curves (2×3) ====================
fig2, axes = plt.subplots(2, 3, figsize=(22, 16))
fig2.suptitle('Learning Curves Analysis', fontsize=22, fontweight='bold', y=0.98)

for i, name in enumerate(model_names):
    row, col = i // 3, i % 3
    ax = axes[row, col]

    # Create simple pipeline for learning curve
    if name in ["Logistic Regression", "SVM"]:
        simple_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', models[name])
        ])
    else:
        simple_pipeline = Pipeline([
            ('classifier', models[name])
        ])

    # Set optimal parameters (remove smote prefix)
    simple_params = {}
    for key, value in final_params[name].items():
        if key.startswith('classifier__'):
            simple_params[key] = value
    simple_pipeline.set_params(**simple_params)

    # Compute learning curve
    train_sizes, train_scores, val_scores = learning_curve(
        simple_pipeline, X_train, y_train, cv=5, n_jobs=1,
        train_sizes=np.linspace(0.1, 1.0, 8), scoring='accuracy'
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    # Plot learning curve
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                    alpha=0.2, color=model_colors[name])
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                    alpha=0.2, color='orange')

    line1 = ax.plot(train_sizes, train_mean, 'o-', color=model_colors[name],
                    linewidth=3, markersize=8, label='Training Score')
    line2 = ax.plot(train_sizes, val_mean, 's-', color='orange',
                    linewidth=3, markersize=8, label='Validation Score')

    # Annotate final values
    ax.annotate(f'{train_mean[-1]:.3f}',
                xy=(train_sizes[-1], train_mean[-1]),
                xytext=(5, 5), textcoords='offset points',
                fontsize=12, fontweight='bold')
    ax.annotate(f'{val_mean[-1]:.3f}',
                xy=(train_sizes[-1], val_mean[-1]),
                xytext=(5, -15), textcoords='offset points',
                fontsize=12, fontweight='bold')

    ax.set_title(f'{name}', fontsize=18, fontweight='bold', pad=25)
    ax.set_xlabel('Training Set Size', fontsize=16, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=16, fontweight='bold')
    ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True, fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_facecolor('#FAFAFA')
    ax.tick_params(labelsize=14)

    # Add border
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(2)

# Hide extra subplot
axes[1, 2].set_visible(False)

plt.tight_layout()
plt.subplots_adjust(top=0.93)
plt.savefig('result/learning_curves.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

# ==================== Figure 3-6: Feature Importance (Individual Large Plots) ====================
# Random Forest Feature Importance
fig3, ax = plt.subplots(1, 1, figsize=(16, 12))

name = 'Random Forest'
pipeline = results[name]['pipeline']
importances = pipeline.named_steps['classifier'].feature_importances_

# Create feature importance DataFrame
feature_imp_df = pd.DataFrame({
    'feature': top_20_features,
    'importance': importances
}).sort_values('importance', ascending=True)

# Plot horizontal bar chart
bars = ax.barh(range(len(feature_imp_df)), feature_imp_df['importance'],
               color=model_colors[name], alpha=0.8, edgecolor='black', linewidth=1.5)

# Annotate values
for j, (bar, value) in enumerate(zip(bars, feature_imp_df['importance'])):
    ax.text(value + max(feature_imp_df['importance']) * 0.01, j, f'{value:.3f}',
            va='center', fontsize=14, fontweight='bold')

ax.set_yticks(range(len(feature_imp_df)))
ax.set_yticklabels(feature_imp_df['feature'], fontsize=16)
ax.set_xlabel('Feature Importance', fontsize=18, fontweight='bold')
ax.set_title(f'{name} - Feature Importance Analysis', fontsize=20, fontweight='bold', pad=25)
ax.grid(True, axis='x', alpha=0.3, linestyle='--')
ax.set_facecolor('#FAFAFA')
ax.tick_params(labelsize=16)

# Add border
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(2)

plt.tight_layout()
plt.savefig('result/feature_importance_rf.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

# Logistic Regression Feature Importance
fig4, ax = plt.subplots(1, 1, figsize=(16, 12))

name = 'Logistic Regression'
pipeline = results[name]['pipeline']
importances = np.abs(pipeline.named_steps['classifier'].coef_[0])

# Create feature importance DataFrame
feature_imp_df = pd.DataFrame({
    'feature': top_20_features,
    'importance': importances
}).sort_values('importance', ascending=True)

# Plot horizontal bar chart
bars = ax.barh(range(len(feature_imp_df)), feature_imp_df['importance'],
               color=model_colors[name], alpha=0.8, edgecolor='black', linewidth=1.5)

# Annotate values
for j, (bar, value) in enumerate(zip(bars, feature_imp_df['importance'])):
    ax.text(value + max(feature_imp_df['importance']) * 0.01, j, f'{value:.3f}',
            va='center', fontsize=14, fontweight='bold')

ax.set_yticks(range(len(feature_imp_df)))
ax.set_yticklabels(feature_imp_df['feature'], fontsize=16)
ax.set_xlabel('Coefficient Magnitude', fontsize=18, fontweight='bold')
ax.set_title(f'{name} - Feature Importance Analysis', fontsize=20, fontweight='bold', pad=25)
ax.grid(True, axis='x', alpha=0.3, linestyle='--')
ax.set_facecolor('#FAFAFA')
ax.tick_params(labelsize=16)

# Add border
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(2)

plt.tight_layout()
plt.savefig('result/feature_importance_lr.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

# Decision Tree Feature Importance
fig5, ax = plt.subplots(1, 1, figsize=(16, 12))

name = 'Decision Tree'
pipeline = results[name]['pipeline']
importances = pipeline.named_steps['classifier'].feature_importances_

# Create feature importance DataFrame
feature_imp_df = pd.DataFrame({
    'feature': top_20_features,
    'importance': importances
}).sort_values('importance', ascending=True)

# Plot horizontal bar chart
bars = ax.barh(range(len(feature_imp_df)), feature_imp_df['importance'],
               color=model_colors[name], alpha=0.8, edgecolor='black', linewidth=1.5)

# Annotate values
for j, (bar, value) in enumerate(zip(bars, feature_imp_df['importance'])):
    ax.text(value + max(feature_imp_df['importance']) * 0.01, j, f'{value:.3f}',
            va='center', fontsize=14, fontweight='bold')

ax.set_yticks(range(len(feature_imp_df)))
ax.set_yticklabels(feature_imp_df['feature'], fontsize=16)
ax.set_xlabel('Feature Importance', fontsize=18, fontweight='bold')
ax.set_title(f'{name} - Feature Importance Analysis', fontsize=20, fontweight='bold', pad=25)
ax.grid(True, axis='x', alpha=0.3, linestyle='--')
ax.set_facecolor('#FAFAFA')
ax.tick_params(labelsize=16)

# Add border
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(2)

plt.tight_layout()
plt.savefig('result/feature_importance_dt.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

# XGBoost Feature Importance
fig6, ax = plt.subplots(1, 1, figsize=(16, 12))

name = 'XGBoost'
pipeline = results[name]['pipeline']
importances = pipeline.named_steps['classifier'].feature_importances_

# Create feature importance DataFrame
feature_imp_df = pd.DataFrame({
    'feature': top_20_features,
    'importance': importances
}).sort_values('importance', ascending=True)

# Plot horizontal bar chart
bars = ax.barh(range(len(feature_imp_df)), feature_imp_df['importance'],
               color=model_colors[name], alpha=0.8, edgecolor='black', linewidth=1.5)

# Annotate values
for j, (bar, value) in enumerate(zip(bars, feature_imp_df['importance'])):
    ax.text(value + max(feature_imp_df['importance']) * 0.01, j, f'{value:.3f}',
            va='center', fontsize=14, fontweight='bold')

ax.set_yticks(range(len(feature_imp_df)))
ax.set_yticklabels(feature_imp_df['feature'], fontsize=16)
ax.set_xlabel('Feature Importance', fontsize=18, fontweight='bold')
ax.set_title(f'{name} - Feature Importance Analysis', fontsize=20, fontweight='bold', pad=25)
ax.grid(True, axis='x', alpha=0.3, linestyle='--')
ax.set_facecolor('#FAFAFA')
ax.tick_params(labelsize=16)

# Add border
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(2)

plt.tight_layout()
plt.savefig('result/feature_importance_xgb.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

# ==================== Figure 7: ROC Curve Comparison ====================
fig7, ax = plt.subplots(1, 1, figsize=(14, 12))

# Plot ROC curves
for name, result in results.items():
    fpr, tpr, _ = roc_curve(y_test, result['y_pred_proba'])
    roc_auc = auc(fpr, tpr)

    ax.plot(fpr, tpr, color=model_colors[name], lw=4,
            label=f'{name} (AUC = {roc_auc:.3f})', marker='o', markersize=6, markevery=5)

# Plot diagonal
ax.plot([0, 1], [0, 1], 'k--', lw=3, alpha=0.7, label='Random Classifier')

ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.05])
ax.set_xlabel('False Positive Rate', fontsize=18, fontweight='bold')
ax.set_ylabel('True Positive Rate', fontsize=18, fontweight='bold')
ax.set_title('ROC Curves Comparison', fontsize=20, fontweight='bold', pad=25)
ax.legend(loc="lower right", frameon=True, fancybox=True, shadow=True, fontsize=14)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_facecolor('#FAFAFA')
ax.tick_params(labelsize=16)

# Add border
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(2)

plt.tight_layout()
plt.savefig('result/roc_curves.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

# ==================== Figure 8: Model Performance Comparison (2×2) ====================
fig8, axes = plt.subplots(2, 2, figsize=(18, 14))
fig8.suptitle('Model Performance Comparison', fontsize=22, fontweight='bold', y=0.98)

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

for metric, pos in zip(metrics, positions):
    ax = axes[pos]

    values = metrics_df[metric].values
    model_names_list = list(metrics_df.index)

    # Create bar chart
    bars = ax.bar(range(len(model_names_list)), values,
                  color=[model_colors[name] for name in model_names_list],
                  alpha=0.8, edgecolor='black', linewidth=2)

    # Annotate values
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{value:.3f}', ha='center', va='bottom',
                fontsize=14, fontweight='bold')

    ax.set_ylabel(metric, fontsize=16, fontweight='bold')
    ax.set_title(f'{metric} Comparison', fontsize=18, fontweight='bold', pad=20)
    ax.set_xticks(range(len(model_names_list)))
    ax.set_xticklabels(model_names_list, rotation=45, ha='right', fontsize=12)
    ax.set_ylim(0, max(values) * 1.15)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.set_facecolor('#FAFAFA')
    ax.tick_params(labelsize=14)

    # Add border
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(2)

plt.tight_layout()
plt.subplots_adjust(top=0.93)
plt.savefig('result/performance_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

log_result("\nAll charts have been generated and saved:")
log_result("random_forest_analysis.png - Random Forest Baseline Analysis")
log_result("validation_curves.png - Validation Curves")
log_result("parameter_optimization_comparison.png - Parameter Optimization Comparison")
log_result("confusion_matrices.png - Confusion Matrices")
log_result("learning_curves.png - Learning Curves")
log_result("feature_importance_rf.png - Random Forest Feature Importance")
log_result("feature_importance_lr.png - Logistic Regression Feature Importance")
log_result("feature_importance_dt.png - Decision Tree Feature Importance")
log_result("feature_importance_xgb.png - XGBoost Feature Importance")
log_result("roc_curves.png - ROC Curve Comparison")
log_result("performance_comparison.png - Performance Comparison")

# Print detailed results
log_result("\n" + "="*80)
log_result("Detailed Analysis Results")
log_result("="*80)

for name, result in results.items():
    log_result(f"\n{name}:")
    log_result(f"  Final params: {result['final_params']}")
    log_result(f"  Cross-validation score: {result['cv_mean']:.4f}")
    log_result(f"  Test set accuracy: {result['test_accuracy']:.4f}")
    log_result(f"  Classification report:")
    report = classification_report(y_test, result['y_pred'])
    # Add indentation
    for line in report.split('\n'):
        log_result(f"    {line}")

log_result("\nAnalysis complete! All results have been saved to the result directory.")

