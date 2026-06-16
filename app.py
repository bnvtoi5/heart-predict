# Demo: https://your-demo-link.streamlit.app 
# Source Code: https://github.com/bnvtoi5/heart-predict
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Khởi tạo các mô hình và bộ gộp Ensemble
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from xgboost import XGBClassifier

# 1. CẤU HÌNH GIAO DIỆN WIDE VÀ TIÊU ĐỀ
st.set_page_config(page_title="Web App Demo", layout="wide")
st.markdown("<h1 style='color: red;'>🫀 Web App Demo</h1>", unsafe_allow_html=True)

# Định nghĩa thứ tự cột thuộc tính chuẩn khớp với file CSV của thầy
COLUMNS_ORDER = [
    'age', 'trestbps', 'chol', 'thalach', 'oldpeak', 
    'sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'
]

# Tham số phân phối thực tế để chuẩn hóa ngầm dữ liệu thô từ giao diện
UCI_STATS = {
    'age': {'mean': 54.366, 'std': 9.082},
    'trestbps': {'mean': 131.623, 'std': 17.538},
    'chol': {'mean': 246.264, 'std': 51.830},
    'thalach': {'mean': 149.646, 'std': 22.905},
    'oldpeak': {'mean': 1.039, 'std': 1.161},
    'cp': {'min_val': 1.0, 'max_val': 4.0},
    'restecg': {'min_val': 0.0, 'max_val': 2.0},
    'slope': {'min_val': 1.0, 'max_val': 3.0},
    'ca': {'min_val': 0.0, 'max_val': 3.0},
    'thal': {'min_val': 3.0, 'max_val': 7.0}
}

def normalize_user_input(raw_dict):
    scaled_list = []
    for col in COLUMNS_ORDER:
        val = raw_dict[col]
        if col in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']:
            norm_val = (val - UCI_STATS[col]['mean']) / UCI_STATS[col]['std']
        elif col in ['cp', 'restecg', 'slope', 'ca', 'thal']:
            norm_val = (val - UCI_STATS[col]['min_val']) / (UCI_STATS[col]['max_val'] - UCI_STATS[col]['min_val'])
        else:
            norm_val = val
        scaled_list.append(norm_val)
    return np.array([scaled_list])

# 2. HÀM TRAIN MÔ HÌNH KHI KHỞI ĐỘNG (Thêm mô hình Ensemble Soft Voting)
@st.cache_resource
def train_models_on_startup():
    try:
        train_path = os.path.join("data", "raw_train.csv")
        train_df = pd.read_csv(train_path)
        
        X_train = train_df[COLUMNS_ORDER].values
        y_train = train_df['target'].values
        
        # Định nghĩa các mô hình cơ sở
        base_models = [
            ('Decision Tree', DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42)),
            ('k-NN', KNeighborsClassifier(n_neighbors=7, weights='distance')),
            ('Naive Bayes', GaussianNB(var_smoothing=1e-2)),
            ('Random Forest', RandomForestClassifier(n_estimators=150, max_depth=4, min_samples_leaf=4, random_state=42)),
            ('AdaBoost', AdaBoostClassifier(n_estimators=50, learning_rate=0.1, random_state=42)),
            ('Gradient Boosting', GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)),
            ('XGBoost', XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, eval_metric='logloss', random_state=42))
        ]
        
        # Tạo mô hình Ensemble bằng phương pháp Soft Voting
        ensemble_model = VotingClassifier(estimators=base_models, voting='soft')
        
        # Tạo từ điển chứa tất cả bao gồm cả mô hình tích hợp cuối cùng
        models = dict(base_models)
        models['Ensemble (Soft Voting)'] = ensemble_model
        
        # Huấn luyện toàn bộ mô hình trên tập Train
        for name, model in models.items():
            model.fit(X_train, y_train)
            
        return models, None
    except Exception as e:
        return None, str(e)

models_dict, error_msg = train_models_on_startup()

if error_msg:
    st.error(f"❌ Không thể huấn luyện dữ liệu của thầy. Lỗi: {error_msg}")
    st.stop()

# --- KHÔNG THAY ĐỔI: KHO MẪU BỆNH NHÂN CỦA THẦY ---
examples_data = {
    "Example 1 (No Heart Disease)": {
        'age': 58, 'sex': 1.0, 'cp': 2.0, 'trestbps': 130, 'chol': 250, 
        'fbs': 0.0, 'restecg': 1.0, 'thalach': 150, 'exang': 0.0, 
        'oldpeak': 1.0, 'slope': 1.0, 'ca': 0.0, 'thal': 3.0
    },
    "Example 2 (Heart Disease)": {
        'age': 63, 'sex': 1.0, 'cp': 4.0, 'trestbps': 145, 'chol': 233, 
        'fbs': 1.0, 'restecg': 0.0, 'thalach': 120, 'exang': 1.0, 
        'oldpeak': 2.3, 'slope': 3.0, 'ca': 2.0, 'thal': 7.0
    }
}

if "age" not in st.session_state:
    st.session_state.age = 58
    st.session_state.sex = 1.0
    st.session_state.cp = 2.0
    st.session_state.trestbps = 130
    st.session_state.chol = 250
    st.session_state.fbs = 0.0
    st.session_state.restecg = 1.0
    st.session_state.thalach = 150
    st.session_state.exang = 0.0
    st.session_state.oldpeak = 1.0
    st.session_state.slope = 1.0
    st.session_state.ca = 0.0
    st.session_state.thal = 3.0

def handle_example_change():
    selected = st.session_state.example_select
    if selected in examples_data:
        data = examples_data[selected]
        st.session_state.age = int(data['age'])
        st.session_state.sex = float(data['sex'])
        st.session_state.cp = float(data['cp'])
        st.session_state.trestbps = int(data['trestbps'])
        st.session_state.chol = int(data['chol'])
        st.session_state.fbs = float(data['fbs'])
        st.session_state.restecg = float(data['restecg'])
        st.session_state.thalach = int(data['thalach'])
        st.session_state.exang = float(data['exang'])
        st.session_state.oldpeak = float(data['oldpeak'])
        st.session_state.slope = float(data['slope'])
        st.session_state.ca = float(data['ca'])
        st.session_state.thal = float(data['thal'])

col_left, col_right = st.columns([1.2, 1])

# --- CỘT BÊN TRÁI: NHẬP LIỆU ---
with col_left:
    st.markdown("### 📂 Enter Patient Features")
    with st.container(border=True):
        row1_col1, row1_col2, row1_col3, row1_col4 = st.columns(4)
        with row1_col1: age = st.number_input("age (years)", min_value=1, max_value=120, key="age")
        with row1_col2: sex = st.selectbox("sex (0=female, 1=male)", options=[0.0, 1.0], index=[0.0, 1.0].index(st.session_state.sex), key="sex")
        with row1_col3: cp = st.selectbox("cp (chest pain type 1..4)", options=[1.0, 2.0, 3.0, 4.0], index=[1.0, 2.0, 3.0, 4.0].index(st.session_state.cp), key="cp")
        with row1_col4: trestbps = st.number_input("trestbps (resting BP mmHg)", key="trestbps")

    with st.container(border=True):
        row2_col1, row2_col2, row2_col3, row2_col4 = st.columns(4)
        with row2_col1: chol = st.number_input("chol (serum cholesterol mg/dl)", key="chol")
        with row2_col2: fbs = st.selectbox("fbs (>120 mg/dl? 1/0)", options=[0.0, 1.0], index=[0.0, 1.0].index(st.session_state.fbs), key="fbs")
        with row2_col3: restecg = st.selectbox("restecg (0..2)", options=[0.0, 1.0, 2.0], index=[0.0, 1.0, 2.0].index(st.session_state.restecg), key="restecg")
        with row2_col4: thalach = st.number_input("thalach (max heart rate)", key="thalach")

    with st.container(border=True):
        row3_col1, row3_col2, row3_col3, row3_col4 = st.columns(4)
        with row3_col1: exang = st.selectbox("exang (exercise angina 1/0)", options=[0.0, 1.0], index=[0.0, 1.0].index(st.session_state.exang), key="exang")
        with row3_col2: oldpeak = st.number_input("oldpeak (ST depression)", step=0.1, key="oldpeak")
        with row3_col3: slope = st.selectbox("slope (1..3)", options=[1.0, 2.0, 3.0], index=[1.0, 2.0, 3.0].index(st.session_state.slope), key="slope")
        with row3_col4: ca = st.selectbox("ca (major vessels 0..3)", options=[0.0, 1.0, 2.0, 3.0], index=[0.0, 1.0, 2.0, 3.0].index(st.session_state.ca), key="ca")

    with st.container(border=True):
        thal = st.selectbox("thal (3=normal, 6=fixed, 7=reversible)", options=[3.0, 6.0, 7.0], index=[3.0, 6.0, 7.0].index(st.session_state.thal), key="thal")
    
    st.write("") 
    bot_col1, bot_col2 = st.columns([1.2, 1])
    with bot_col1:
        with st.container(border=True): 
            st.selectbox("Select Example Patient", options=["-- Tự nhập thủ công --", "Example 1 (No Heart Disease)", "Example 2 (Heart Disease)"], key="example_select", on_change=handle_example_change)
    with bot_col2:
        st.write(" ") 
        predict_btn = st.button("🔍 Predict", use_container_width=True, type="primary")

# --- CỘT BÊN PHẢI: CO GIÃN ĐỘNG CHIỀU CAO THEO ĐỘ TỰ TIN (PREDICTION CONFIDENCE) ---
with col_right:
    st.markdown("### 📊 Model Predictions Overview")
    st.markdown("Model Predictions")
    
    if "has_predicted" not in st.session_state:
        st.session_state.has_predicted = False

    if predict_btn:
        st.session_state.has_predicted = True
        
        raw_user_features = {
            'age': age, 'trestbps': trestbps, 'chol': chol, 'thalach': thalach, 'oldpeak': oldpeak,
            'sex': sex, 'cp': cp, 'fbs': fbs, 'restecg': restecg, 'exang': exang, 'slope': slope, 'ca': ca, 'thal': thal
        }
        
        input_data_scaled = normalize_user_input(raw_user_features)
        
        model_names = list(models_dict.keys())
        bar_colors = []
        bar_texts = []
        confidences = []
        
        # 3. Lấy dự đoán nhãn và ĐỘ TỰ TIN (Mức độ phần trăm tin cậy) từ hàm predict_proba
        for name in model_names:
            model = models_dict[name]
            
            # Lấy mảng xác suất [P(Nhãn 0), P(Nhãn 1)]
            prob = model.predict_proba(input_data_scaled)[0]
            pred = np.argmax(prob) # Lấy nhãn có xác suất cao hơn làm kết quả dự đoán
            conf_val = prob[pred]  # Lấy đúng giá trị xác suất lớn nhất đó làm chiều cao cột
            
            confidences.append(conf_val)
            
            if pred == 1:
                bar_colors.append("#d62728") # Màu đỏ nếu đoán có bệnh
                bar_texts.append("⚠️ Heart Disease")
            else:
                bar_colors.append("#2ca02c") # Màu xanh nếu đoán không bệnh
                bar_texts.append("🗹 No Heart Disease")
                
        # Lưu toàn bộ kết quả tính toán động vào trạng thái session_state
        st.session_state.bar_colors = bar_colors
        st.session_state.bar_texts = bar_texts
        st.session_state.confidences = confidences

    # Vẽ biểu đồ động bằng Matplotlib trùng khớp hoàn toàn với cấu trúc ảnh của thầy
    if st.session_state.has_predicted:
        model_names = list(models_dict.keys())
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Thiết lập viền đen dày hơn cho cột cuối Ensemble để giống hệt thầy
        bars = ax.bar(
            model_names, 
            st.session_state.confidences, 
            color=st.session_state.bar_colors, 
            edgecolor='black', 
            linewidth=1
        )
        bars[-1].set_linewidth(2.5) # Cột cuối (Ensemble) có viền đen đậm hơn giống thầy
        
        # Chèn chữ lồng bên trong và xoay dọc 270 độ
        for bar, text_label in zip(bars, st.session_state.bar_texts):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height / 2.0,
                text_label, 
                ha='center', 
                va='center', 
                color='white', 
                weight='bold', 
                size=11, 
                rotation=270
            )
            
        # Ghi số phần trăm độ tự tin lên đỉnh mỗi thanh bar (Ví dụ: 100%, 71%, 88%...)
        ax.bar_label(bars, fmt=lambda x: f'{x:.0%}', padding=5, weight='bold', size=11, color='black')
        
        # Đặt tên nhãn trục theo đúng biểu đồ của thầy
        ax.set_ylabel('Prediction Confidence', weight='bold', size=12)
        ax.set_xlabel('Model', weight='bold', size=12, labelpad=15)
        ax.set_ylim(0, 1.15)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        
        # Xoay chữ tên mô hình ở trục hoành hướng đi xuống giống thầy
        plt.xticks(rotation=-30, ha='left', weight='bold')
        plt.tight_layout()
        
        # Đẩy biểu đồ lên Streamlit
        st.pyplot(fig)
    else:
        # Trạng thái ban đầu khi chưa nhấn nút Predict
        st.info("💡 Vui lòng thiết lập các thông số ở cột bên trái, sau đó nhấn nút **🔍 Predict** để tiến hành chạy mô hình tính toán và xem kết quả.")
