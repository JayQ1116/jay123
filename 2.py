import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
from sklearn.impute import KNNImputer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from ucimlrepo import fetch_ucirepo
import os
from PIL import Image
import warnings
import joblib
import io
from pymongo import MongoClient

# MongoDB configuration
# Replace <Qujy031023streamlit > with actual password
MONGO_URI = "mongodb+srv://jay:Qujy031023@cluster0.pn1wsxr.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
MONGO_DB = "ckd_ml"
MONGO_COLL = "models"

def get_mongo_collection():
    client = MongoClient(MONGO_URI)
    db = client[MONGO_DB]
    return db[MONGO_COLL]

def save_model_to_mongo(model_dict, model_name="ckd_ensemble"):
    
    coll = get_mongo_collection()
    buf = io.BytesIO()
    joblib.dump(model_dict, buf)
    buf.seek(0)
    coll.delete_many({"model_name": model_name})
    coll.insert_one({
        "model_name": model_name,
        "model_blob": buf.read(),
    })

def load_model_from_mongo(model_name="ckd_ensemble"):
  
    coll = get_mongo_collection()
    doc = coll.find_one({"model_name": model_name})
    if doc:
        buf = io.BytesIO(doc["model_blob"])
        model_dict = joblib.load(buf)
        return model_dict
    return None


def save_user_input_to_mongo(input_data):
    """Save user input data to MongoDB user_inputs collection"""
    client = MongoClient(MONGO_URI)
    db = client[MONGO_DB]
    user_inputs_coll = db["user_inputs"]
    user_inputs_coll.insert_one(input_data)

def load_all_user_inputs_from_mongo():
    """Load all user inputs from MongoDB user_inputs collection"""
    client = MongoClient(MONGO_URI)
    db = client[MONGO_DB]
    user_inputs_coll = db["user_inputs"]
    return list(user_inputs_coll.find())


warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="CKD Intelligent Diagnosis System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Main container */
    .main {
        padding: 0rem 1rem;
        background-color: #f8f9fa;
    }

    /* Header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }

    .main-header h1 {
        font-size: 3rem;
        margin: 0;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }

    .main-header p {
        font-size: 1.2rem;
        margin-top: 0.5rem;
        opacity: 0.9;
    }

    /* Sub header */
    .sub-header {
        color: #2c3e50;
        font-size: 2rem;
        font-weight: 600;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #667eea;
    }

    /* Prediction box */
    .prediction-box {
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        text-align: center;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    .prediction-box:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(0,0,0,0.15);
    }

    .success-box {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        border: none;
        color: #1a5f3f;
    }

    .danger-box {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        border: none;
        color: #8b0000;
    }

    .prediction-box h4 {
        font-size: 1.5rem;
        margin-bottom: 1rem;
        font-weight: 700;
    }

    .prediction-box p {
        font-size: 1.1rem;
        margin: 0.5rem 0;
    }

    /* Metric card */
    .metric-card {
        background: white;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 5px 20px rgba(0,0,0,0.08);
        transition: transform 0.3s ease;
    }

    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
    }

    .metric-card h3 {
        color: #7c8798;
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
    }

    .metric-card h2 {
        color: #2c3e50;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }

    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        border-radius: 10px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(102, 126, 234, 0.4);
    }

    /* Info box */
    .info-box {
        background: linear-gradient(135deg, #e0f2fe 0%, #cffafe 100%);
        border-left: 5px solid #0284c7;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Global variables
continuous_vars = ['age', 'bp', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wbcc', 'rbcc']
categorical_vars = ['sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']

categorical_values = {
    'sg': [1.005, 1.010, 1.015, 1.020, 1.025],
    'al': [0, 1, 2, 3, 4, 5],
    'su': [0, 1, 2, 3, 4, 5],
    'rbc': ['normal', 'abnormal'],
    'pc': ['normal', 'abnormal'],
    'pcc': ['present', 'notpresent'],
    'ba': ['present', 'notpresent'],
    'htn': ['yes', 'no'],
    'dm': ['yes', 'no'],
    'cad': ['yes', 'no'],
    'appet': ['good', 'poor'],
    'pe': ['yes', 'no'],
    'ane': ['yes', 'no']
}

# Model colors
model_colors = {
    'Random Forest': '#4A90E2',
    'Logistic Regression': '#50C878',
    'SVM': '#FFB347',
    'Decision Tree': '#FF6B6B',
    'XGBoost': '#9B59B6'
}

# Set matplotlib style
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

# Environment variable
os.environ['LOKY_MAX_CPU_COUNT'] = '1'


# Cache data loading and processing
@st.cache_data
def load_and_process_data():
    """Load and process data following original code logic"""

    # First data load (for exploratory analysis)
    chronic_kidney_disease = fetch_ucirepo(id=336)
    X = chronic_kidney_disease.data.features
    y = chronic_kidney_disease.data.targets
    df_first = pd.concat([X, y], axis=1)

    # Binary classification for target variable (first time: ckd/notckd)
    df_first['class'] = df_first['class'].apply(lambda x: 'ckd' if x == 'ckd' else 'notckd').astype('category')

    # Manually specify continuous variables as numeric
    for var in continuous_vars:
        if var in df_first.columns:
            df_first[var] = pd.to_numeric(df_first[var], errors='coerce')

    # Manually specify categorical variables (including class)
    categorical_vars_with_class = categorical_vars + ['class']
    for var in categorical_vars_with_class:
        if var in df_first.columns:
            df_first[var] = df_first[var].astype('category')

    # Re-classify variables
    cat_cols = df_first.select_dtypes(include=['category']).columns
    num_cols = df_first.select_dtypes(include=['float64', 'int64']).columns

    # Second data load (for model training)
    chronic_kidney_disease = fetch_ucirepo(id=336)
    df = pd.concat([chronic_kidney_disease.data.features, chronic_kidney_disease.data.targets], axis=1)

    # Encode target values as 0 and 1
    df['class'] = df['class'].apply(lambda x: 1 if x == 'ckd' else 0)

    # Backup original data
    df_original = df.copy()

    # Clean 'dm' column outliers
    if 'dm' in df.columns:
        df['dm'] = df['dm'].replace({'\tno': 'no', '\tyes': 'yes', ' yes': 'yes', '': None})
        df['dm'] = df['dm'].astype('category')

    # Handle missing values
    # KNN imputation for numerical columns
    knn_imputer = KNNImputer(n_neighbors=5)
    df[num_cols] = knn_imputer.fit_transform(df[num_cols])

    # Mode imputation for categorical columns
    for col in cat_cols:
        if col != 'class' and df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)

    # One-hot encoding (excluding class)
    df_encoded = pd.get_dummies(df, columns=categorical_vars, drop_first=True)

    # Separate features and target
    X = df_encoded.drop(['class'], axis=1)
    if '' in X.columns:
        X = X.drop([''], axis=1)
    y = df_encoded['class']

    # Process df_clean for standardization display
    df_clean = df.copy()
    df_clean[continuous_vars] = df_clean[continuous_vars].apply(pd.to_numeric, errors='coerce')
    df_clean[continuous_vars] = df_clean[continuous_vars].fillna(df_clean[continuous_vars].median())

    return df, df_encoded, X, y, df_original, df_clean


# Core function for model training (without UI elements)
@st.cache_resource
def train_models_core(_X, _y, _df_encoded):
    """Core logic for model training"""
    # Create copies to avoid read-only issues
    X = _X.copy()
    y = _y.copy()
    df_encoded = _df_encoded.copy()

    # Part 1: Initial Random Forest Analysis
    # Use all features
    X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_all, y_train_all)

    # Get feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    # Get top 20 features
    top_20_features = feature_importance.head(20)['feature'].tolist()

    # Part 2: Multi-model optimization with selected features
    X_selected = df_encoded[top_20_features]
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.3, random_state=42, stratify=y
    )

    # Define models
    models = {
        "Random Forest": RandomForestClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42)
    }

    # Stage 1: Initial grid search
    initial_param_grids = {
        "Random Forest": {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [None, 10, 20],
            'classifier__min_samples_split': [2, 5]
        },
        "Logistic Regression": {
            'classifier__C': [0.1, 1, 10],
            'classifier__penalty': ['l1'],
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

    initial_results = {}
    initial_best_params = {}

    for name, model in models.items():
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

        grid_search = GridSearchCV(pipeline, initial_param_grids[name], cv=5, scoring='accuracy', n_jobs=1)
        grid_search.fit(X_train, y_train)

        initial_best_params[name] = grid_search.best_params_
        initial_results[name] = grid_search.best_score_

    # Stage 2: Validation curve fine-tuning
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
            'description': 'Max Depth'
        },
        "XGBoost": {
            'param_name': 'classifier__n_estimators',
            'param_range': [50, 100, 150, 200, 250, 300, 350, 400],
            'description': 'Number of Trees (n_estimators)'
        }
    }

    final_params = {}
    validation_results = {}

    for name, model in models.items():
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

        base_params = initial_best_params[name].copy()
        param_to_tune = key_params[name]['param_name']
        if param_to_tune in base_params:
            del base_params[param_to_tune]

        base_pipeline.set_params(**base_params)

        param_range = key_params[name]['param_range']
        train_scores, val_scores = validation_curve(
            base_pipeline, X_train, y_train,
            param_name=param_to_tune, param_range=param_range,
            cv=5, scoring='accuracy', n_jobs=1
        )

        val_mean = np.mean(val_scores, axis=1)
        best_idx = np.argmax(val_mean)
        best_param_value = param_range[best_idx]

        if name == "Decision Tree" and best_param_value >= 25:
            final_best_param_value = None
        else:
            final_best_param_value = best_param_value

        final_params[name] = base_params.copy()
        final_params[name][param_to_tune] = final_best_param_value
        validation_results[name] = val_mean[best_idx]

    # Stage 3: Final training
    final_results = {}
    final_models = {}

    for name, model in models.items():
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

        final_pipeline.set_params(**final_params[name])
        final_pipeline.fit(X_train, y_train)

        y_pred = final_pipeline.predict(X_test)
        y_pred_proba = final_pipeline.predict_proba(X_test)[:, 1]

        final_models[name] = final_pipeline

    return {
        'final_models': final_models,
        'top_20_features': top_20_features,
        'X_test': X_test,
        'y_test': y_test
    }


# Preprocess input data
def preprocess_input(input_data, top_20_features, df_encoded_columns):
    """Preprocess input data"""
    df = pd.DataFrame([input_data])

    # Process continuous variables
    for var in continuous_vars:
        if var in df.columns:
            df[var] = pd.to_numeric(df[var], errors='coerce')

    # One-hot encoding
    df_encoded = pd.get_dummies(df, columns=categorical_vars, drop_first=True)

    # Ensure all required columns exist
    for col in df_encoded_columns:
        if col not in df_encoded.columns and col != 'class':
            df_encoded[col] = 0

    # Select only the columns needed by the model, ensuring consistent order
    df_final = df_encoded[df_encoded_columns[df_encoded_columns != 'class']]

    # Select top_20_features
    if set(top_20_features).issubset(set(df_final.columns)):
        df_final = df_final[top_20_features]

    return df_final



# Main application
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏥 CKD Intelligent Diagnosis System</h1>
        <p>Machine Learning Based CKD Prediction & Analysis Platform</p>
    </div>
    """, unsafe_allow_html=True)

    
    # Load data
    with st.spinner('Loading data...'):
        df, df_encoded, X, y, df_original, df_clean = load_and_process_data()

    # Sidebar
    with st.sidebar:
        st.markdown("### 🎛️ Control Panel")

        with st.expander("ℹ️ System Introduction"):
            st.markdown("""
            This system strictly follows the original code logic:
            1. Data preprocessing & missing value imputation
            2. Feature selection (Top 20)
            3. Three-stage model optimization
            4. Multi-model ensemble prediction

            5 Algorithms used:
            - 🌲 Random Forest
            - 📊 Logistic Regression
            - 🎯 SVM
            - 🌳 Decision Tree
            - 🚀 XGBoost
            """)

        mode = st.radio(
            "Select Function Module",
            ["🏠 Home", "🤖 Model Training", "🔮 Prediction", "📊 Performance Analysis"]
        )

        if mode == "🔮 Prediction":
            st.markdown("### 📝 Sample Selection")
            sample_option = st.selectbox(
                "Select Input Method",
                ["Manual Input", "Random CKD Patient Sample", "Random Healthy Sample"]
            )

            if sample_option != "Manual Input":
                if st.button("🎲 Extract Random Sample"):
                    # Select sample from actual dataset
                    if 'CKD Patient' in sample_option:
                        ckd_samples = df[df['class'] == 1]
                        random_idx = np.random.choice(ckd_samples.index)
                    else:
                        healthy_samples = df[df['class'] == 0]
                        random_idx = np.random.choice(healthy_samples.index)

                    random_sample = df.loc[random_idx].to_dict()
                    random_sample.pop('class', None)
                    st.session_state['random_sample'] = random_sample
                    st.success("✅ Real sample extracted from dataset!")

    # Home
    if mode == "🏠 Home":
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>Total Samples</h3>
                <h2>{}</h2>
            </div>
            """.format(len(df)), unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3>CKD Patients</h3>
                <h2>{}</h2>
            </div>
            """.format((df['class'] == 1).sum()), unsafe_allow_html=True)

        with col3:
            st.markdown("""
            <div class="metric-card">
                <h3>Healthy</h3>
                <h2>{}</h2>
            </div>
            """.format((df['class'] == 0).sum()), unsafe_allow_html=True)

        with col4:
            st.markdown("""
            <div class="metric-card">
                <h3>Features</h3>
                <h2>{}</h2>
            </div>
            """.format(len(X.columns)), unsafe_allow_html=True)

        # Dataset Overview
        st.markdown('<h2 class="sub-header">📊 Dataset Overview</h2>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            # Class distribution pie chart
            fig, ax = plt.subplots(figsize=(8, 6))
            class_counts = df['class'].value_counts()
            colors = ['#84fab0', '#fa709a']
            labels = ['Healthy', 'CKD']

            wedges, texts, autotexts = ax.pie(class_counts.values, labels=labels, colors=colors,
                                              autopct='%1.1f%%', startangle=90,
                                              textprops={'fontsize': 14, 'fontweight': 'bold'})

            for autotext in autotexts:
                autotext.set_color('white')

            ax.set_title('Class Distribution', fontsize=16, fontweight='bold', pad=20)
            st.pyplot(fig)

        with col2:
            # Feature statistics
            fig, ax = plt.subplots(figsize=(8, 6))

            # Display number of features after processing
            original_features = len(continuous_vars) + len(categorical_vars)
            encoded_features = len(X.columns)

            feature_stats = ['Original Features', 'After Encoding']
            feature_counts = [original_features, encoded_features]
            colors = ['#667eea', '#764ba2']

            bars = ax.bar(feature_stats, feature_counts, color=colors, alpha=0.8)

            for bar, count in zip(bars, feature_counts):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                        f'{count}', ha='center', va='bottom',
                        fontsize=16, fontweight='bold')

            ax.set_ylabel('Number of Features', fontsize=14)
            ax.set_title('Feature Dimension Changes', fontsize=16, fontweight='bold', pad=20)
            ax.set_ylim(0, max(feature_counts) * 1.2)
            st.pyplot(fig)

    # Model Training
    elif mode == "🤖 Model Training":
        st.markdown('<h2 class="sub-header">🤖 Model Training & Optimization</h2>', unsafe_allow_html=True)

        # Train Model button on a single row
        if st.button("🚀 Train Model", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()
            status_text.text('Training model, please wait...')
            results = train_models_core(X, y, df_encoded)
            st.session_state['trained'] = True
            st.session_state['results'] = results
            st.session_state['df_encoded_columns'] = df_encoded.columns
            progress_bar.empty()
            status_text.empty()
            st.success("✅ Model training completed!")

        # Save/Load/Retrain buttons on the next row, side by side
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("💾 Save Model to MongoDB", use_container_width=True):
                if 'results' in st.session_state:
                    save_model_to_mongo(st.session_state['results'])
                    st.success("✅ Model saved to MongoDB!")
                else:
                    st.warning("Please train the model first!")
        with col2:
            if st.button("☁️ Load Model from MongoDB", use_container_width=True):
                loaded = load_model_from_mongo()
                if loaded:
                    st.session_state['trained'] = True
                    st.session_state['results'] = loaded
                    st.success("✅ Model loaded from MongoDB!")
                else:
                    st.warning("No model found in MongoDB!")
        with col3:
            if st.button("🔄 Retrain Model (Overwrite MongoDB)", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()
                status_text.text('Retraining model...')

                user_inputs = load_all_user_inputs_from_mongo()
                if user_inputs:
                    user_inputs_df = pd.DataFrame(user_inputs)
                    if '_id' in user_inputs_df.columns:
                        user_inputs_df = user_inputs_df.drop(columns=['_id'])
                    # Merge with original data
                    df_aug = pd.concat([df, user_inputs_df], ignore_index=True)

                    # 1. Convert continuous variables to numeric
                    for var in continuous_vars:
                        if var in df_aug.columns:
                            df_aug[var] = pd.to_numeric(df_aug[var], errors='coerce')
                    # 2. Convert categorical variables to category type
                    for var in categorical_vars:
                        if var in df_aug.columns:
                            df_aug[var] = df_aug[var].astype('category')

                    # 3. Impute missing values (same as load_and_process_data)
                    from sklearn.impute import KNNImputer
                    knn_imputer = KNNImputer(n_neighbors=5)
                    num_cols = df_aug.select_dtypes(include=['float64', 'int64']).columns
                    df_aug[num_cols] = knn_imputer.fit_transform(df_aug[num_cols])

                    # Categorical: fill with mode
                    cat_cols = df_aug.select_dtypes(include=['category']).columns
                    for col in cat_cols:
                        if col != 'class' and df_aug[col].isnull().sum() > 0:
                            mode_val = df_aug[col].mode()[0]
                            df_aug[col].fillna(mode_val, inplace=True)

                    # 4. One-hot encoding
                    df_aug_encoded = pd.get_dummies(df_aug, columns=categorical_vars, drop_first=True)
                    X_aug = df_aug_encoded.drop(['class'], axis=1)
                    y_aug = df_aug_encoded['class']

                    # 5. Check for any remaining NaN
                    if X_aug.isnull().any().any() or y_aug.isnull().any():
                        st.error("❌ Data still contains missing values after preprocessing. Please check input format.")
                        progress_bar.empty()
                        status_text.empty()
                        st.stop()

                    results = train_models_core(X_aug, y_aug, df_aug_encoded)
                else:
                    results = train_models_core(X, y, df_encoded)

                save_model_to_mongo(results)
                st.session_state['trained'] = True
                st.session_state['results'] = results
                st.session_state['df_encoded_columns'] = df_encoded.columns
                progress_bar.empty()
                status_text.empty()
                st.success("✅ Model retrained and saved to MongoDB!")

    # Prediction
    elif mode == "🔮 Prediction":
        st.markdown('<h2 class="sub-header">🔮 Patient Diagnosis Prediction</h2>', unsafe_allow_html=True)

        if 'trained' not in st.session_state:
            st.warning("⚠️ Please complete model training first!")
            st.stop()

        # Get training results
        results = st.session_state['results']
        final_models = results['final_models']
        top_20_features = results['top_20_features']
        df_encoded_columns = st.session_state['df_encoded_columns']

        # Create input form
        with st.form("prediction_form"):
            input_data = {}

            # Get random sample (if any)
            random_sample = st.session_state.get('random_sample', {})

            # Continuous variables input
            st.markdown("#### 🔢 Physiological Indicators (Continuous)")
            cont_cols = st.columns(3)
            for i, var in enumerate(continuous_vars):
                with cont_cols[i % 3]:
                    default_value = random_sample.get(var, 0.0) if random_sample else 0.0
                    input_data[var] = st.number_input(
                        f"{var}",
                        value=float(default_value),
                        format="%.2f",
                        help=f"Enter value for {var}"
                    )

            # Categorical variables input
            st.markdown("#### 📋 Clinical Indicators (Categorical)")
            cat_cols = st.columns(3)
            for i, var in enumerate(categorical_vars):
                with cat_cols[i % 3]:
                    if random_sample and var in random_sample:
                        default_value = random_sample[var]
                        if default_value in categorical_values[var]:
                            default_index = categorical_values[var].index(default_value)
                        else:
                            default_index = 0
                    else:
                        default_index = 0

                    input_data[var] = st.selectbox(
                        f"{var}",
                        options=categorical_values[var],
                        index=default_index,
                        help=f"Select value for {var}"
                    )

            # Submit button
            submitted = st.form_submit_button("🔍 Start Diagnosis", use_container_width=True)

            if submitted:
                # Save user input to MongoDB
                save_user_input_to_mongo(input_data)
                # Preprocess data
                processed_data = preprocess_input(input_data, top_20_features, df_encoded_columns)

                # Make predictions
                st.markdown("### 🎯 Diagnosis Results")

                results_cols = st.columns(len(final_models))
                predictions = []

                for i, (name, model) in enumerate(final_models.items()):
                    with results_cols[i]:
                        prediction = model.predict(processed_data)[0]
                        probability = model.predict_proba(processed_data)[0]

                        predictions.append(prediction)

                        if prediction == 1:
                            st.markdown(f"""
                            <div class="prediction-box danger-box">
                                <h4>{name}</h4>
                                <p><strong>Diagnosis Result</strong></p>
                                <p style="font-size: 1.5rem;">🔴 CKD Positive</p>
                                <p>Confidence: {probability[1]:.1%}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="prediction-box success-box">
                                <h4>{name}</h4>
                                <p><strong>Diagnosis Result</strong></p>
                                <p style="font-size: 1.5rem;">🟢 Healthy</p>
                                <p>Confidence: {probability[0]:.1%}</p>
                            </div>
                            """, unsafe_allow_html=True)

                # Overall diagnosis recommendation
                ckd_count = sum(predictions)
                st.markdown("---")
                if ckd_count >= 3:
                    st.error(
                        "⚠️ **Overall Diagnosis**: Majority of models ({}/{}) predict CKD positive. Please seek medical attention for detailed examination.".format(
                            ckd_count, len(predictions)))
                else:
                    st.success(
                        "✅ **Overall Diagnosis**: Majority of models ({}/{}) predict healthy status. Regular check-ups are still recommended.".format(
                            len(predictions) - ckd_count, len(predictions)))

    # Performance Analysis
    elif mode == "📊 Performance Analysis":
        st.markdown('<h2 class="sub-header">📊 Model Performance Analysis</h2>', unsafe_allow_html=True)

        # Create tabs for different analysis categories
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🎯 Baseline Analysis",
            "📈 Model Optimization",
            "🔍 Model Evaluation",
            "⭐ Feature Importance",
            "📊 Performance Comparison"
        ])

        with tab1:
            st.markdown("### Random Forest Baseline Analysis")
            if os.path.exists("result/random_forest_analysis.png"):
                st.image("result/random_forest_analysis.png", use_container_width=True)
            else:
                st.warning("Random Forest analysis image not found")

        with tab2:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Validation Curves")
                if os.path.exists("result/validation_curves.png"):
                    st.image("result/validation_curves.png", use_container_width=True)
                else:
                    st.warning("Validation curves image not found")

            with col2:
                st.markdown("### Parameter Optimization")
                if os.path.exists("result/parameter_optimization_comparison.png"):
                    st.image("result/parameter_optimization_comparison.png", use_container_width=True)
                else:
                    st.warning("Parameter optimization image not found")

        with tab3:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Confusion Matrices")
                if os.path.exists("result/confusion_matrices.png"):
                    st.image("result/confusion_matrices.png", use_container_width=True)
                else:
                    st.warning("Confusion matrices image not found")

            with col2:
                st.markdown("### Learning Curves")
                if os.path.exists("result/learning_curves.png"):
                    st.image("result/learning_curves.png", use_container_width=True)
                else:
                    st.warning("Learning curves image not found")

        with tab4:
            st.markdown("### Feature Importance Analysis")

            # Create 2x2 grid for feature importance plots
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Random Forest")
                if os.path.exists("result/feature_importance_rf.png"):
                    st.image("result/feature_importance_rf.png", use_container_width=True)
                else:
                    st.warning("RF feature importance image not found")

                st.markdown("#### Decision Tree")
                if os.path.exists("result/feature_importance_dt.png"):
                    st.image("result/feature_importance_dt.png", use_container_width=True)
                else:
                    st.warning("DT feature importance image not found")

            with col2:
                st.markdown("#### Logistic Regression")
                if os.path.exists("result/feature_importance_lr.png"):
                    st.image("result/feature_importance_lr.png", use_container_width=True)
                else:
                    st.warning("LR feature importance image not found")

                st.markdown("#### XGBoost")
                if os.path.exists("result/feature_importance_xgb.png"):
                    st.image("result/feature_importance_xgb.png", use_container_width=True)
                else:
                    st.warning("XGB feature importance image not found")

        with tab5:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### ROC Curves")
                if os.path.exists("result/roc_curves.png"):
                    st.image("result/roc_curves.png", use_container_width=True)
                else:
                    st.warning("ROC curves image not found")

            with col2:
                st.markdown("### Performance Metrics")
                if os.path.exists("result/performance_comparison.png"):
                    st.image("result/performance_comparison.png", use_container_width=True)
                else:
                    st.warning("Performance comparison image not found")


if __name__ == "__main__":
    main()