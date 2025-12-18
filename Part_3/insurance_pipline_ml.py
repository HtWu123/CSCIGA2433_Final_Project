# insurance_pipeline_final.py


import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# -------- CONFIG --------
# 假设 csv 文件和脚本在同一个目录下
DATA_PATH = 'data/medical_insurance_final.csv' 
SAVE_DIR = Path("./saved_models")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42

# -------- LOAD DATA --------
if not os.path.exists(DATA_PATH):
    print(f"❌ 错误: 找不到文件 {DATA_PATH}")
    print("请确保 csv 文件和这个 python 脚本在同一个文件夹里！")
    exit() # 如果找不到文件直接退出

df = pd.read_csv(DATA_PATH)
print("Loaded data shape:", df.shape)

# -------- BASIC CLEAN & FEATURE ENGINEERING --------
df.columns = [c.strip() for c in df.columns]

# normalize smoker values and create smoker_num
if 'smoker' in df.columns:
    df['smoker_norm'] = df['smoker'].astype(str).str.strip().str.lower()
    smoker_map = {'never': 0.0, 'former': 1.0, 'current': 2.0}
    df['smoker_num'] = df['smoker_norm'].map(smoker_map).fillna(0.0).astype(float)
else:
    df['smoker_num'] = 0.0

# disease columns -> total_chronic_diseases
disease_cols = [
    'hypertension','diabetes','asthma','copd','cardiovascular_disease',
    'cancer_history','kidney_disease','liver_disease','arthritis','mental_health'
]
existing_diseases = [c for c in disease_cols if c in df.columns]
for c in existing_diseases:
    df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
df['total_chronic_diseases'] = df[existing_diseases].sum(axis=1) if existing_diseases else 0

# fill categorical missing
for c in ['region','urban_rural','alcohol_freq','sex']:
    if c in df.columns:
        df[c] = df[c].fillna('UNKNOWN')

# numeric median fill
numeric_candidates = [
    'age','bmi','deductible','annual_premium','monthly_premium','copay',
    'systolic_bp','diastolic_bp','ldl','hba1c',
    'visits_last_year','medication_count','risk_score','claims_count','avg_claim_amount',
    'annual_medical_cost','total_claims_paid'
]
for c in numeric_candidates:
    if c in df.columns:
        if df[c].dtype.kind in 'biufc':
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(df[c].median())
        else:
            df[c] = df[c].fillna(df[c].mode()[0] if not df[c].isnull().all() else 0)

# construct is_high_risk if missing
if 'is_high_risk' not in df.columns:
    if 'annual_medical_cost' in df.columns:
        thr = df['annual_medical_cost'].median()
        df['is_high_risk'] = (df['annual_medical_cost'] > thr).astype(int)
    else:
        df['is_high_risk'] = 0

# -------- SELECT FEATURES --------
FEATURES = [
    'age','sex','bmi','smoker_num','alcohol_freq',
    'systolic_bp','diastolic_bp','ldl','hba1c',
    'region','urban_rural',
    'deductible','copay','visits_last_year','medication_count',
    'risk_score','total_chronic_diseases','claims_count','avg_claim_amount'
]
FEATURES = [c for c in FEATURES if c in df.columns]
print("Using features:", FEATURES)

TARGET_PREMIUM = 'annual_premium'
TARGET_RISK = 'is_high_risk'

# -------- PREPROCESSOR --------
numeric_cols = [c for c in FEATURES if df[c].dtype.kind in 'biufc']
cat_cols = [c for c in FEATURES if c not in numeric_cols]

try:
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
except TypeError:
    ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='UNKNOWN')),
    ('ohe', ohe)
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_pipeline, numeric_cols),
        ('cat', cat_pipeline, cat_cols)
    ],
    remainder='drop'
)

preprocessor.fit(df[FEATURES])

# -------- TRAIN & SAVE --------
if not (SAVE_DIR / "premium_pipe.joblib").exists():
    print("🚀 开始训练模型 (本地)...")
    
    # 1. Premium Model
    X = df[FEATURES].copy()
    y = df[TARGET_PREMIUM].copy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    
    lgb_reg = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=31, random_state=RANDOM_STATE, n_jobs=-1)
    premium_pipe = Pipeline([('preproc', preprocessor), ('reg', lgb_reg)])
    premium_pipe.fit(X_train, y_train)
    joblib.dump(premium_pipe, SAVE_DIR / "premium_pipe.joblib")
    print("✅ Premium Model 训练并保存完毕。")

    # 2. Risk Model
    risk_features = [c for c in ['age','bmi','smoker_num','total_chronic_diseases'] if c in df.columns]
    Xr = df[risk_features].copy()
    yr = df[TARGET_RISK].copy()
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr, yr, test_size=0.2, random_state=RANDOM_STATE, stratify=yr)
    
    scaler_risk = StandardScaler()
    Xr_train_s = scaler_risk.fit_transform(Xr_train)
    lgb_clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=31, random_state=RANDOM_STATE, n_jobs=-1)
    lgb_clf.fit(Xr_train_s, yr_train)
    
    joblib.dump(lgb_clf, SAVE_DIR / "risk_model.joblib")
    joblib.dump(scaler_risk, SAVE_DIR / "risk_scaler.joblib")
    print("✅ Risk Model 训练并保存完毕。")
    
    # 3. Save Feature Defaults (为了后续网页版不用读 csv)
    feature_defaults = {}
    for f in FEATURES:
        if df[f].dtype.kind in 'biufc':
            feature_defaults[f] = float(df[f].median())
        else:
            feature_defaults[f] = str(df[f].mode()[0])
    joblib.dump(feature_defaults, SAVE_DIR / "feature_defaults.joblib")
    print("✅ Feature Defaults 保存完毕。")

else:
    print("⚡ 模型已存在，跳过训练步骤 (如果要重新训练，请删除 saved_models 文件夹里的文件)")


# -------- INFERENCE FUNCTIONS --------
PLAN_MULTIPLIERS = {"EPO": 0.90, "HMO": 0.95, "POS": 1.00, "PPO": 1.15}

def business_rule_recommend(user_info, risk_prob, plan_prices, preference=None):
    pref = preference or {}
    priority = pref.get('priority', 'balanced')
    budget = pref.get('budget_monthly', None)

    scores = {}
    # 简单的规则逻辑
    for plan, price in plan_prices.items():
        monthly = price['monthly_premium']
        score = 1.0 / (monthly + 1e-9) # 越便宜分越高
        if risk_prob > 0.6 and plan == 'PPO': score += 0.5 # 高风险推荐 PPO
        scores[plan] = score

    recommended = max(scores.items(), key=lambda x: x[1])[0]
    return recommended, scores

def get_insurance_quotes(user_info, preference=None, show_cards=True):
    # 加载模型
    try:
        premium_pipe = joblib.load(SAVE_DIR / "premium_pipe.joblib")
        risk_model = joblib.load(SAVE_DIR / "risk_model.joblib")
        risk_scaler = joblib.load(SAVE_DIR / "risk_scaler.joblib")
        defaults = joblib.load(SAVE_DIR / "feature_defaults.joblib")
    except FileNotFoundError:
        print("❌ 错误: 找不到模型文件，请先运行脚本进行训练！")
        return {}

    ui = user_info.copy()
    
    # 处理 smoker
    if 'smoker' in ui and 'smoker_num' not in ui:
        ui['smoker_num'] = {'never':0.0,'former':1.0,'current':2.0}.get(str(ui['smoker']).strip().lower(), 0.0)
    
    # 填充缺失值 (使用 defaults)
    for f in FEATURES:
        if f not in ui:
            ui[f] = defaults.get(f, 0)

    # 预测保费
    user_df = pd.DataFrame([{k: ui[k] for k in FEATURES}])
    base_annual = float(premium_pipe.predict(user_df)[0])
    base_annual = max(base_annual, 0.0)

    # 预测风险
    risk_features = [c for c in ['age','bmi','smoker_num','total_chronic_diseases'] if c in df.columns]
    Xr = pd.DataFrame([{k: ui[k] for k in risk_features}])
    Xr_s = risk_scaler.transform(Xr)
    risk_prob = float(risk_model.predict_proba(Xr_s)[0][1])

    # 计算不同计划价格
    plan_prices = {}
    for plan, mul in PLAN_MULTIPLIERS.items():
        ann = base_annual * mul
        plan_prices[plan] = {'annual_premium': float(ann), 'monthly_premium': float(ann/12.0)}

    recommended_plan, scores = business_rule_recommend(ui, risk_prob, plan_prices, preference)

    if show_cards:
        print(f"\n=== 结果: 风险 {risk_prob:.2%} | 推荐 {recommended_plan} ===")
        for p, v in plan_prices.items():
            print(f"{p}: ${v['monthly_premium']:.2f}/月")

    return {
        'risk_probability': risk_prob,
        'recommended_plan': recommended_plan,
        'plans': plan_prices
    }

if __name__ == "__main__":
    print("\n--- 本地测试 ---")
    sample = {'age': 50, 'sex': 'Female', 'bmi': 23.5, 'smoker': 'Never'}
    get_insurance_quotes(sample)