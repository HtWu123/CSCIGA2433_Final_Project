from flask import Flask, render_template, request, redirect, url_for, session, flash
from pymongo import MongoClient
from bson.objectid import ObjectId
import pandas as pd
import datetime
import joblib
from pathlib import Path
import certifi
import os
from dotenv import load_dotenv

# === 导入训练函数 ===
from insurance_pipline_ml import train_model

app = Flask(__name__)
app.secret_key = "super_secret_key_for_session"
load_dotenv()
# ================= CONFIG =================
CONNECTION_STRING = os.getenv("MONGODB_URI")
DB_NAME = "insurance_app_db"
USERS_COL = "customers" 
CLAIMS_COL = "claims"

# --- 商业定价策略配置 ---
MARKET_CORRECTION_FACTOR = 4.0 
BASE_OPERATING_COST = 500.0     
TARGET_PROFIT_MARGIN = 0.20     

# ================= DB CONNECTION =================
try:
    client = MongoClient(CONNECTION_STRING, tlsCAFile=certifi.where())
    db = client[DB_NAME]
    users_collection = db[USERS_COL]
    claims_collection = db[CLAIMS_COL]
    print("✅ Connected to MongoDB Atlas")
except Exception as e:
    print(f"⚠️ DB Connection Error: {e}")
    users_collection = None
    claims_collection = None

# ================= LOAD MODELS FUNCTION =================
SAVE_DIR = Path("./saved_models")
premium_pipe = None
risk_pipe = None
defaults = {}
features_list = []

def load_models():
    global premium_pipe, risk_pipe, defaults, features_list
    try:
        premium_pipe = joblib.load(SAVE_DIR / "premium_pipe.joblib")
        risk_pipe = joblib.load(SAVE_DIR / "risk_model.joblib")
        defaults = joblib.load(SAVE_DIR / "feature_defaults.joblib")
        features_list = joblib.load(SAVE_DIR / "features_list.joblib")
        print("✅ ML Models Loaded/Reloaded")
    except Exception as e:
        print(f"⚠️ Models not found or error: {e}")

# 启动时加载一次
load_models()

# ================= HELPER FUNCTIONS =================
def calculate_commercial_price(base_risk_cost, plan_multiplier, risk_prob):
    risk_loading = 1.0 + (risk_prob * 0.8)
    estimated_claims_cost = base_risk_cost * MARKET_CORRECTION_FACTOR * plan_multiplier * risk_loading
    total_cost = estimated_claims_cost + BASE_OPERATING_COST
    final_price = total_cost / (1 - TARGET_PROFIT_MARGIN)
    return round(final_price, 2)

def get_recommendations(user_data):
    if not premium_pipe: return {}, "Standard"
    input_df = pd.DataFrame([user_data])
    for col in features_list:
        if col not in input_df.columns: input_df[col] = defaults.get(col, 0)

    raw_base_cost = max(float(premium_pipe.predict(input_df)[0]), 50.0)
    risk_prob = float(risk_pipe.predict_proba(input_df)[0][1])

    plans = {
        'Basic': {'multiplier': 0.7, 'desc': 'Essential coverage for healthy individuals.'},
        'Standard': {'multiplier': 1.0, 'desc': 'Balanced coverage for most families.'},
        'Premium': {'multiplier': 1.5, 'desc': 'All-inclusive VIP coverage.'}
    }
    
    results = {}
    recommended = 'Standard'
    if risk_prob < 0.3: recommended = 'Basic'
    elif risk_prob > 0.7: recommended = 'Premium'

    for name, info in plans.items():
        annual_price = calculate_commercial_price(raw_base_cost, info['multiplier'], risk_prob)
        results[name] = {
            'monthly': annual_price / 12,
            'annual': annual_price,
            'desc': info['desc'],
            'deductible': 5000 if name == 'Basic' else (1000 if name == 'Standard' else 500),
            'copay': 50 if name == 'Basic' else (30 if name == 'Standard' else 10)
        }
    return results, recommended

# ================= AUTH ROUTES =================

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        if users_collection is not None:
            user = users_collection.find_one({'email': email, 'password': password})
            if user:
                session['user_id'] = str(user['_id'])
                session['role'] = 'user'
                session['user_name'] = user.get('email')
                return redirect(url_for('user_dashboard'))
            else:
                flash("Invalid email or password", "error")
    return render_template('login.html')

@app.route('/admin_login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        password = request.form.get('password')
        if password == "admin123":
            session['role'] = 'admin'
            return redirect(url_for('admin_stats'))
        else:
            flash("Invalid admin password", "error")
    return render_template('admin_login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

# ================= ML UPDATE ROUTE (NEW) =================
@app.route('/retrain', methods=['POST'])
def retrain_models():
    if session.get('role') != 'admin':
        return redirect(url_for('admin_login'))
    
    try:
        # 1. 从 MongoDB 获取最新数据
        new_users = list(users_collection.find())
        
        if not new_users:
            flash("数据库中没有新数据，无需更新。", "warning")
            return redirect(url_for('admin_stats'))

        # 2. 转换为 DataFrame
        df_new = pd.DataFrame(new_users)
        
        # 3. 列名映射 (App存的字段 -> 模型需要的字段)
        # 模型需要: smoker, total_chronic_diseases
        # App存了: smoker_norm, diseases (参见 index 路由)
        if 'smoker_norm' in df_new.columns:
            df_new['smoker'] = df_new['smoker_norm'] 
        if 'diseases' in df_new.columns:
            df_new['total_chronic_diseases'] = df_new['diseases']

        # 4. 调用训练脚本
        train_model(df_new)
        
        # 5. 重新加载模型到内存
        load_models()
        
        flash(f"模型更新成功！使用了 {len(df_new)} 条新数据。", "success")
    except Exception as e:
        flash(f"模型更新失败: {str(e)}", "error")
        print(e)
        
    return redirect(url_for('admin_stats'))

# ================= USER ROUTES =================

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            user_data = {
                'email': request.form.get('email'),
                'password': request.form.get('password'),
                'age': int(request.form.get('age')),
                'sex': request.form.get('sex'),
                'bmi': float(request.form.get('bmi')),
                'smoker_norm': request.form.get('smoker'), # 对应 ML 中的 smoker
                'smoker_num': {'never':0, 'former':1, 'current':2}.get(request.form.get('smoker'), 0),
                'smoking_years': int(request.form.get('smoking_years', 0)),
                'alcohol_freq': request.form.get('alcohol_freq'),
                'systolic_bp': int(request.form.get('systolic_bp')),
                'diastolic_bp': int(request.form.get('diastolic_bp')),
                'visits_last_year': int(request.form.get('visits_last_year')),
                'diseases': int(request.form.get('diseases')), # 对应 ML 中的 total_chronic_diseases
                'region': request.form.get('region'),
                'newsletter_opt_in': 'newsletter' in request.form,
                'is_customer': False, 
                'created_at': datetime.datetime.now()
            }
            
            user_id = "offline"
            if users_collection is not None:
                if users_collection.find_one({'email': user_data['email']}):
                    flash("该邮箱已被注册，请直接登录", "error")
                    return redirect(url_for('login'))
                
                res = users_collection.insert_one(user_data)
                user_id = str(res.inserted_id)
                session['user_id'] = user_id
                session['role'] = 'user'
                session['user_name'] = user_data['email']

            plans, rec = get_recommendations(user_data)
            return render_template('results.html', details=plans, rec=rec, user_id=user_id, user_name=user_data['email'])

        except Exception as e:
            return f"Error: {e}"

    return render_template('index.html')

@app.route('/buy', methods=['POST'])
def buy():
    plan = request.form['plan']
    price = request.form['price']
    user_id = request.form['user_id']
    
    if users_collection:
        users_collection.update_one(
            {'_id': ObjectId(user_id)},
            {'$set': {
                'is_customer': True, 
                'purchased_plan': plan,
                'monthly_premium': float(price),
                'purchase_date': datetime.datetime.now()
            }}
        )
    return redirect(url_for('user_dashboard'))

@app.route('/dashboard')
def user_dashboard():
    if 'user_id' not in session: return redirect(url_for('login'))
    user = users_collection.find_one({'_id': ObjectId(session['user_id'])})
    my_claims = list(claims_collection.find({'user_id': session['user_id']}))
    return render_template('user_dashboard.html', user=user, claims=my_claims)

@app.route('/claim', methods=['GET', 'POST'])
def claim():
    if 'user_id' not in session: return redirect(url_for('login'))
    if request.method == 'POST':
        amount = float(request.form.get('amount'))
        reason = request.form.get('reason')
        user = users_collection.find_one({'_id': ObjectId(session['user_id'])})
        
        if user and user.get('is_customer'):
            claim_doc = {
                'user_id': session['user_id'],
                'email': user['email'],
                'plan_type': user.get('purchased_plan'),
                'amount': amount,
                'reason': reason,
                'status': 'Pending Review',
                'date': datetime.datetime.now()
            }
            claims_collection.insert_one(claim_doc)
            flash("The claim application has been submitted!", "success")
            return redirect(url_for('user_dashboard'))
        else:
            flash("You haven't purchased insurance yet.", "error")
    return render_template('claims.html')

@app.route('/admin')
def admin_stats():
    if session.get('role') != 'admin': return redirect(url_for('admin_login'))
    all_users = list(users_collection.find().sort("created_at", -1))

    # Stats Logic
    pipeline_revenue = [{'$match': {'is_customer': True}}, {'$group': {'_id': '$purchased_plan', 'customer_count': {'$sum': 1}, 'total_monthly_revenue': {'$sum': '$monthly_premium'}}}]
    pipeline_claims = [{'$group': {'_id': '$plan_type', 'total_payout': {'$sum': '$amount'}}}]
    revenue_data = list(users_collection.aggregate(pipeline_revenue))
    claims_data = list(claims_collection.aggregate(pipeline_claims))
    
    stats = {}
    total_rev, total_profit = 0, 0
    for r in revenue_data:
        stats[r['_id']] = {'customers': r['customer_count'], 'revenue': r['total_monthly_revenue']*12, 'payout': 0, 'profit': 0, 'margin': 0}
    for c in claims_data:
        if c['_id'] in stats: stats[c['_id']]['payout'] = c['total_payout']
    for plan, data in stats.items():
        op_cost = data['customers'] * BASE_OPERATING_COST
        data['profit'] = data['revenue'] - data['payout'] - op_cost
        if data['revenue'] > 0: data['margin'] = round((data['profit']/data['revenue'])*100, 2)
        total_rev += data['revenue']
        total_profit += data['profit']

    return render_template('admin.html', stats=stats, total_rev=total_rev, total_profit=total_profit, all_users=all_users)

if __name__ == '__main__':
    app.run(debug=True, port=5000)