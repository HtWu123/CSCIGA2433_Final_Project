import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# -------- CONFIG --------
DATA_PATH = 'data/medical_insurance_final.csv' 
SAVE_DIR = Path("./saved_models")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42

# 定义特征列表 (全局变量，方便外部调用)
NUMERIC_FEATURES = ['age', 'bmi', 'smoker_num', 'total_chronic_diseases', 
                   'systolic_bp', 'diastolic_bp', 'visits_last_year']
CATEGORICAL_FEATURES = ['sex', 'region', 'alcohol_freq']
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_PREMIUM = 'annual_premium' 
TARGET_RISK = 'risk_score'

def train_model(new_data_df=None):
    """
    核心训练函数。
    new_data_df: 来自数据库的新数据 (Pandas DataFrame)
    """
    print("🔄 开始模型训练流程...")
    
    # 1. 加载原始 CSV 数据
    if os.path.exists(DATA_PATH):
        df_csv = pd.read_csv(DATA_PATH)
        df_csv.columns = [c.strip() for c in df_csv.columns]
    else:
        df_csv = pd.DataFrame()
        print("⚠️ 警告: 原始 CSV 文件未找到，将仅使用新数据训练。")

    # 2. 合并新数据 (如果有)
    if new_data_df is not None and not new_data_df.empty:
        print(f"📥 合并新数据: {len(new_data_df)} 条记录")
        # 确保新数据列名与 CSV 一致
        # 假设 MongoDB 数据已经清理好列名
        df = pd.concat([df_csv, new_data_df], ignore_index=True)
    else:
        df = df_csv

    if df.empty:
        return "❌ 训练失败: 没有数据可用"

    # 3. 特征工程 (处理 smoker)
    # 确保 smoker 列存在
    if 'smoker' in df.columns:
        df['smoker_norm'] = df['smoker'].astype(str).str.strip().str.lower()
        smoker_map = {'never': 0, 'former': 1, 'current': 2}
        df['smoker_num'] = df['smoker_norm'].map(smoker_map).fillna(0)
    elif 'smoker_norm' in df.columns: # 如果数据库直接存了 smoker_norm
        # 处理可能的映射
        smoker_map = {'never': 0, 'former': 1, 'current': 2}
        # 如果已经是数字就不动，如果是字符串就映射
        if df['smoker_norm'].dtype == 'object':
             df['smoker_num'] = df['smoker_norm'].map(smoker_map).fillna(0)
        else:
             df['smoker_num'] = df['smoker_norm']
    else:
        df['smoker_num'] = 0

    # 补全缺失列
    for col in NUMERIC_FEATURES:
        if col not in df.columns: df[col] = 0
    for col in CATEGORICAL_FEATURES:
        if col not in df.columns: df[col] = 'unknown'
        
    # 如果缺少目标值 (Target)，为了演示，我们可以用现有模型预测一个填进去，或者直接丢弃这些行
    # 这里为了作业简单，我们假设新数据没有 Target (因为是未标记数据)，
    # **但在真实场景中，我们需要等待数据有了结果(Label)才能训练**。
    # **作业变通方法**: 我们只用带 Label 的数据训练。
    # 也就是：只合并那些 is_customer=True 且有 purchase_price 的数据，或者直接用全量数据做演示。
    
    # 为了保证代码不报错，我们给缺失的 Target 填默认值 (仅供演示代码跑通)
    if TARGET_PREMIUM not in df.columns: df[TARGET_PREMIUM] = 3000
    if TARGET_RISK not in df.columns: df[TARGET_RISK] = 0
    
    df[TARGET_PREMIUM] = df[TARGET_PREMIUM].fillna(df[TARGET_PREMIUM].mean())
    df[TARGET_RISK] = df[TARGET_RISK].fillna(0)

    # 4. 预处理管道
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERIC_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES)
        ])

    # 5. 训练模型 1: 保费预测
    X = df[FEATURES]
    y_premium = df[TARGET_PREMIUM]

    premium_pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', lgb.LGBMRegressor(random_state=RANDOM_STATE))
    ])
    premium_pipe.fit(X, y_premium)

    # 6. 训练模型 2: 风险分类
    # 简化处理：假设 Risk > 0.5 是高风险
    y_risk = (df[TARGET_RISK] > 0.5).astype(int)

    risk_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', lgb.LGBMClassifier(random_state=RANDOM_STATE))
    ])
    risk_pipeline.fit(X, y_risk)

    # 7. 保存模型
    joblib.dump(premium_pipe, SAVE_DIR / "premium_pipe.joblib")
    joblib.dump(risk_pipeline, SAVE_DIR / "risk_model.joblib")
    joblib.dump(FEATURES, SAVE_DIR / "features_list.joblib")

    # 保存默认值
    defaults = df[FEATURES].median(numeric_only=True).to_dict()
    cat_defaults = df[CATEGORICAL_FEATURES].mode().iloc[0].to_dict()
    defaults.update(cat_defaults)
    joblib.dump(defaults, SAVE_DIR / "feature_defaults.joblib")

    print("✅ The ML model has been retrained and saved.")
    return "Success"

if __name__ == "__main__":
    # 如果直接运行脚本，不带新数据训练
    train_model()