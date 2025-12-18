from flask import Flask, render_template, request, redirect, url_for, session, flash
from pymongo import MongoClient
from bson.objectid import ObjectId
import pandas as pd
import numpy as np
import datetime
import joblib
from pathlib import Path
import certifi
import sys
import os
from dotenv import load_dotenv

# === 导入训练函数 ===
# 确保 insurance_pipline_ml.py 在同一目录下
from insurance_pipline_ml import train_model

# 1. 加载 .env 文件中的变量
load_dotenv()

app = Flask(__name__)
app.secret_key = "super_secret_key_for_session" # 生产环境中应设置更复杂的随机字符串

# ================= CONFIG =================
# 从环境变量获取连接字符串
CONNECTION_STRING = os.getenv('MONGODB_URI')

DB_NAME = "insurance_app_db"
USERS_COL = "customers" 
CLAIMS_COL = "claims"
CSV_PATH = 'data/medical_insurance_final.csv' 

# --- 商业定价策略配置 ---
# 调整后的温和参数，模拟真实商业保险
MARKET_CORRECTION_FACTOR = 1.2  # 市场校准系数 (上浮 20%)
BASE_OPERATING_COST = 50.0      # 每单固定运营成本
TARGET_PROFIT_MARGIN = 0.15     # 目标利润率 (15%)

# ================= DB CONNECTION (ROBUST) =================
users_collection = None
claims_collection = None

print("----------------------------------------------------------------")
print(f"🔄 正在尝试连接 MongoDB Atlas...")

if not CONNECTION_STRING:
    print("❌ fail cannot find MONGODB_URI！")
    print("please make sure you have a .env file with MONGODB_URI.")
    sys.exit(1)

try:
    # 强制检查 dnspython 是否存在
    import dns.resolver
except ImportError:
    print("\n❌ fail cannot find dnspython！")
    print("MongoDB SRV needs dnspython to resolve SRV records.")
    print("please do this: pip install dnspython\n")

try:
    # 强制使用 SSL (tls=True) 并指定 CA 证书
    # serverSelectionTimeoutMS 设置为 5000ms (5秒)，避免长时间卡死
    client = MongoClient(CONNECTION_STRING, tlsCAFile=certifi.where(), serverSelectionTimeoutMS=5000)
    
    # 测试连接 (这一步会真正触发网络请求)
    client.admin.command('ping')
    
    db = client[DB_NAME]
    users_collection = db[USERS_COL]
    claims_collection = db[CLAIMS_COL]
    print("✅ successfully connected to MongoDB Atlas!")
    
except Exception as e:
    print("\n" + "="*50)
    print("❌ db fail！")
    print(f"error detail: {e}")
    print("-" * 30)
    print("💡 suggestion: ")
    print("1. Check your .env file for correct MONGODB_URI.")
    print("2. Ensure you have installed 'dnspython'.")
    print("3. Verify your IP is whitelisted in MongoDB Atlas.")
    print("="*50 + "\n")

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
        print("✅ ML Models Loaded")
    except Exception as e:
        print(f"⚠️ ML loading warning: {e}")
        print("if first time running this project make sure run python insurance_pipline_ml.py")

# 启动时先加载一次
load_models()

# ================= HELPER FUNCTIONS =================
def calculate_commercial_price(base_risk_cost, plan_multiplier, risk_prob):
    """
    商业定价核心逻辑：
    价格 = (基础风险成本 * 市场系数 * 计划倍率 * 风险调整 + 运营成本) / (1 - 利润率)
    """
    risk_loading = 1.0 + (risk_prob * 0.5) # 风险越高，倍率越高
    
    # 1. 预估赔付成本 (Estimated Claims Cost)
    estimated_claims_cost = base_risk_cost * MARKET_CORRECTION_FACTOR * plan_multiplier * risk_loading
    
    # 2. 总成本 (Total Cost = Claims + Operating)
    total_cost = estimated_claims_cost + BASE_OPERATING_COST
    
    # 3. 最终售价 (含利润) -> Price = Cost / (1 - Margin)
    final_price = total_cost / (1 - TARGET_PROFIT_MARGIN)
    
    return round(final_price, 2)

def get_recommendations(user_data):
    if not premium_pipe: return {}, "Standard"
    
    # 准备 DataFrame
    input_df = pd.DataFrame([user_data])
    
    # 填充缺失列
    for col in features_list:
        if col not in input_df.columns:
            input_df[col] = defaults.get(col, 0)

    # 预测基础纯保费 (来自 ML 的纯风险预测)
    raw_base_cost = max(float(premium_pipe.predict(input_df)[0]), 50.0)
    
    # 预测风险概率
    risk_prob = float(risk_pipe.predict_proba(input_df)[0][1])

    # 套餐逻辑
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
        # 使用商业定价函数
        annual_price = calculate_commercial_price(raw_base_cost, info['multiplier'], risk_prob)
        
        results[name] = {
            'monthly': annual_price / 12,
            'annual': annual_price,
            'desc': info['desc'],
            'deductible': 5000 if name == 'Basic' else (1000 if name == 'Standard' else 500),
            'copay': 50 if name == 'Basic' else (30 if name == 'Standard' else 10)
        }
        
    return results, recommended

# ================= AUTH ROUTES (Login/Logout) =================

@app.route('/login', methods=['GET', 'POST'])
def login():
    if users_collection is None:
        return "❌ Database connection failed.", 500

    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        try:
            user = users_collection.find_one({'email': email, 'password': password})
            if user:
                session['user_id'] = str(user['_id'])
                session['role'] = 'user'
                session['user_name'] = user.get('email')
                return redirect(url_for('user_dashboard'))
            else:
                flash("email or password error", "error")
        except Exception as e:
            return f"Database Error during Login: {e}", 500

    return render_template('login.html')

@app.route('/admin_login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        password = request.form.get('password')
        if password == "admin123": # 硬编码管理员密码
            session['role'] = 'admin'
            return redirect(url_for('admin_stats'))
        else:
            flash("admin password error", "error")
    return render_template('admin_login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

# ================= ML UPDATE ROUTE (Retrain) =================
@app.route('/retrain', methods=['POST'])
def retrain_models():
    # 鉴权
    if session.get('role') != 'admin':
        return redirect(url_for('admin_login'))
    
    if users_collection is None:
        flash("Database connection failed.", "error")
        return redirect(url_for('admin_stats'))

    try:
        # 1. 获取新数据
        new_users = list(users_collection.find())
        if not new_users:
            flash("no new data", "warning")
            return redirect(url_for('admin_stats'))

        df_new = pd.DataFrame(new_users)
        
        # 2. 字段映射 (数据库字段 -> 模型字段)
        if 'smoker_norm' in df_new.columns:
            df_new['smoker'] = df_new['smoker_norm'] 
        if 'diseases' in df_new.columns:
            df_new['total_chronic_diseases'] = df_new['diseases']

        # ==========================================
        # 核心逻辑：价格逆向还原 (De-Commercialization)
        # 防止将高额售价作为成本价训练，导致价格螺旋上升
        # ==========================================
        if 'monthly_premium' in df_new.columns:
            print("reloading the data...")
            
            def reverse_price(row):
                # 如果用户没买，或者数据缺失，返回 NaN
                if not row.get('is_customer') or pd.isna(row.get('monthly_premium')):
                    return np.nan
                
                # 1. 拿到年化商业售价
                final_price = row['monthly_premium'] * 12
                
                # 2. 逆向推导：剥离利润和运营成本
                # 原公式: Final = (Total_Cost) / (1 - Margin)
                # Total_Cost = Claims_Cost + Op_Cost
                
                # 第一步：还原含成本的总价
                total_cost = final_price * (1 - TARGET_PROFIT_MARGIN)
                
                # 第二步：扣除固定运营费
                claims_cost = total_cost - BASE_OPERATING_COST
                
                # 第三步：除以市场系数 (还原到原始数据集的水平)
                # 假设平均 Risk Loading 为 1.0 (简化)
                base_pure_cost = claims_cost / MARKET_CORRECTION_FACTOR
                
                return max(base_pure_cost, 50.0) # 设个底线

            # 创建 'annual_premium' 列用于训练
            df_new['annual_premium'] = df_new.apply(reverse_price, axis=1)
            
            # 打印有效数据量
            valid_prices = df_new['annual_premium'].dropna()
            print(f"new records: {len(valid_prices)}")
            
        else:
            print("⚠️no monthly_premium in data, skipping price de-commercialization.")

        # 3. 训练模型
        train_model(df_new)
        
        # 4. 重新加载模型到内存
        load_models()
        
        flash(f"Model updated successfully! {len(df_new)} new records have been de-commercialized and used for training.", "success")
        
    except Exception as e:
        flash(f"Model update failed: {str(e)}", "error")
        print(f"Retrain Error: {e}")
        
    return redirect(url_for('admin_stats'))

# ================= USER ROUTES =================

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # 1. 收集表单数据
            user_data = {
                'email': request.form.get('email'),
                'age': int(request.form.get('age')),
                'sex': request.form.get('sex'),
                'bmi': float(request.form.get('bmi')),
                'smoker_norm': request.form.get('smoker'), 
                'smoker_num': {'never':0, 'former':1, 'current':2}.get(request.form.get('smoker'), 0),
                'smoking_years': int(request.form.get('smoking_years', 0)),
                'alcohol_freq': request.form.get('alcohol_freq'),
                'systolic_bp': int(request.form.get('systolic_bp')),
                'diastolic_bp': int(request.form.get('diastolic_bp')),
                'visits_last_year': int(request.form.get('visits_last_year')),
                'diseases': int(request.form.get('diseases')),
                'region': request.form.get('region'),
                'newsletter_opt_in': 'newsletter' in request.form,
                'created_at': datetime.datetime.now()
            }
            
            password_input = request.form.get('password')
            if password_input:
                user_data['password'] = password_input

            user_id = "offline"
            
            if users_collection is not None:
                # 场景 A: 用户已登录 (Session 有 ID) -> 更新数据
                if 'user_id' in session:
                    user_id = session['user_id']
                    # 更新除了 _id 以外的字段
                    users_collection.update_one({'_id': ObjectId(user_id)}, {'$set': user_data})
                    session['user_name'] = user_data['email'] # 更新 Session 中的邮箱
                
                # 场景 B: 邮箱已存在但未登录 -> 提示登录
                elif users_collection.find_one({'email': user_data['email']}):
                    flash("This email has been registered! Please log in first, then click 'Get Quote' on the Dashboard to update your information.", "warning")
                    return redirect(url_for('login'))
                
                # 场景 C: 新用户 -> 插入数据
                else:
                    user_data['is_customer'] = False
                    if not password_input: user_data['password'] = "123456" # 默认密码
                    
                    res = users_collection.insert_one(user_data)
                    user_id = str(res.inserted_id)
                    
                    # 自动登录
                    session['user_id'] = user_id
                    session['role'] = 'user'
                    session['user_name'] = user_data['email']

            # 计算报价
            plans, rec = get_recommendations(user_data)
            return render_template('results.html', details=plans, rec=rec, user_id=user_id, user_name=user_data['email'])

        except Exception as e:
            return f"Error processing data: {e}"

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
    # 购买后跳转到 Dashboard 查看状态
    return redirect(url_for('user_dashboard'))

@app.route('/dashboard')
def user_dashboard():
    # 必须登录才能看
    if 'user_id' not in session: return redirect(url_for('login'))
    
    if users_collection is None: return "DB Connection Error", 500

    user = users_collection.find_one({'_id': ObjectId(session['user_id'])})
    my_claims = list(claims_collection.find({'user_id': session['user_id']}))
    
    return render_template('user_dashboard.html', user=user, claims=my_claims)

@app.route('/claim', methods=['GET', 'POST'])
def claim():
    if 'user_id' not in session: return redirect(url_for('login'))
    
    if request.method == 'POST':
        amount = float(request.form.get('amount'))
        reason = request.form.get('reason')
        if users_collection:
            user = users_collection.find_one({'_id': ObjectId(session['user_id'])})
            
            # 只有正式客户能理赔
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
                flash("Claim application submitted successfully!", "success")
                return redirect(url_for('user_dashboard'))
            else:
                flash("You have not purchased insurance, so you cannot file a claim.", "error")
                
    return render_template('claims.html')

# ================= ADMIN ROUTES =================

@app.route('/admin')
def admin_stats():
    # 必须管理员登录
    if session.get('role') != 'admin': return redirect(url_for('admin_login'))
    
    if users_collection is None:
        return "Database connection failed.", 500
        
    # 获取所有用户用于列表展示
    all_users = list(users_collection.find().sort("created_at", -1))

    # 聚合管道：计算收入
    pipeline_revenue = [
        {'$match': {'is_customer': True}}, 
        {'$group': {
            '_id': '$purchased_plan', 
            'customer_count': {'$sum': 1}, 
            'total_monthly_revenue': {'$sum': '$monthly_premium'}
        }}
    ]
    
    # 聚合管道：计算理赔
    pipeline_claims = [
        {'$group': {
            '_id': '$plan_type', 
            'total_payout': {'$sum': '$amount'}
        }}
    ]
    
    revenue_data = list(users_collection.aggregate(pipeline_revenue))
    claims_data = list(claims_collection.aggregate(pipeline_claims))
    
    stats = {}
    total_rev, total_profit = 0, 0
    
    # 整理数据
    for r in revenue_data:
        stats[r['_id']] = {
            'customers': r['customer_count'], 
            'revenue': r['total_monthly_revenue'] * 12, # 年化收入
            'payout': 0, 
            'profit': 0, 
            'margin': 0
        }
        
    for c in claims_data:
        if c['_id'] in stats: 
            stats[c['_id']]['payout'] = c['total_payout']
            
    # 计算利润率
    for plan, data in stats.items():
        op_cost = data['customers'] * BASE_OPERATING_COST
        data['profit'] = data['revenue'] - data['payout'] - op_cost
        
        if data['revenue'] > 0: 
            data['margin'] = round((data['profit'] / data['revenue']) * 100, 2)
            
        total_rev += data['revenue']
        total_profit += data['profit']

    return render_template('admin.html', 
                           stats=stats, 
                           total_rev=total_rev, 
                           total_profit=total_profit, 
                           all_users=all_users)

if __name__ == '__main__':
    print("🚀 App running on http://127.0.0.1:5000")
    app.run(debug=True, port=5000)