import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="심장질환 예측 시스템",
    page_icon="🫀",
    layout="centered"
)

def build_model():
    return nn.Sequential(
        nn.Linear(15, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(128, 2),
    )

@st.cache_resource
def load_model():
    model = build_model()
    model.load_state_dict(
        torch.load('binary_mlp_final.pth', map_location='cpu')
    )
    model.eval()
    return model

# 학습 데이터 원본 기준 표준화 파라미터 (역산)
RAW_PARAMS = {
    'age':      {'mean': 54.4,  'std': 9.1},
    'trestbps': {'mean': 131.6, 'std': 17.6},
    'chol':     {'mean': 199.1, 'std': 110.0},
    'thalch':   {'mean': 137.5, 'std': 25.9},
    'oldpeak':  {'mean': 0.89,  'std': 1.07},
}

FEATURE_COLS = [
    'age', 'sex', 'trestbps', 'chol', 'fbs', 'thalch', 'exang', 'oldpeak',
    'cp_atypical angina', 'cp_non-anginal', 'cp_typical angina',
    'restecg_normal', 'restecg_st-t abnormality',
    'slope_flat', 'slope_upsloping'
]

def preprocess(inputs):
    row = {}

    # 연속형 표준화
    for col in ['age', 'trestbps', 'chol', 'thalch']:
        row[col] = (inputs[col] - RAW_PARAMS[col]['mean']) / RAW_PARAMS[col]['std']

    # oldpeak: log1p 후 표준화
    oldpeak_log = np.log1p(max(inputs['oldpeak'], 0))
    row['oldpeak'] = (oldpeak_log - RAW_PARAMS['oldpeak']['mean']) / RAW_PARAMS['oldpeak']['std']

    # 이진형
    row['sex']   = 1.0 if inputs['sex'] == 'Male' else 0.0
    row['fbs']   = 1.0 if inputs['fbs'] else 0.0
    row['exang'] = 1.0 if inputs['exang'] else 0.0

    # cp One-Hot (기준: asymptomatic)
    cp_map = {
        'atypical angina': (1,0,0), 'non-anginal': (0,1,0),
        'typical angina':  (0,0,1), 'asymptomatic': (0,0,0),
    }
    v = cp_map[inputs['cp']]
    row['cp_atypical angina'] = float(v[0])
    row['cp_non-anginal']     = float(v[1])
    row['cp_typical angina']  = float(v[2])

    # restecg One-Hot (기준: lv hypertrophy)
    restecg_map = {
        'normal': (1,0), 'st-t abnormality': (0,1), 'lv hypertrophy': (0,0),
    }
    v = restecg_map[inputs['restecg']]
    row['restecg_normal']           = float(v[0])
    row['restecg_st-t abnormality'] = float(v[1])

    # slope One-Hot (기준: downsloping)
    slope_map = {
        'flat': (1,0), 'upsloping': (0,1), 'downsloping': (0,0),
    }
    v = slope_map[inputs['slope']]
    row['slope_flat']      = float(v[0])
    row['slope_upsloping'] = float(v[1])

    x = np.array([row[col] for col in FEATURE_COLS], dtype=np.float32)
    return torch.FloatTensor(x).unsqueeze(0)


# ===== UI =====
st.title("🫀 심장질환 예측 시스템")
st.caption("환자의 임상 정보를 입력하면 심장질환 유무를 예측합니다.")
st.warning("⚠️ 본 시스템은 연구용 모델로, 실제 임상 진단을 대체할 수 없습니다.")

st.divider()
st.subheader("환자 정보 입력")

col1, col2 = st.columns(2)

with col1:
    age      = st.number_input("나이 (age)", min_value=1, max_value=120, value=55)
    sex      = st.selectbox("성별 (sex)", ["Male", "Female"])
    cp       = st.selectbox(
        "흉통 유형 (cp)",
        ["asymptomatic", "typical angina", "atypical angina", "non-anginal"],
        help="asymptomatic: 무증상 · typical angina: 전형적 협심증 · "
             "atypical angina: 비전형 협심증 · non-anginal: 비협심증성"
    )
    trestbps = st.number_input("안정시 혈압 (trestbps, mmHg)", min_value=50, max_value=250, value=130)
    chol     = st.number_input("혈청 콜레스테롤 (chol, mg/dl)", min_value=100, max_value=600, value=240)
    fbs      = st.checkbox("공복 혈당 > 120 mg/dl (fbs)", value=False)
    restecg  = st.selectbox(
        "안정시 심전도 (restecg)",
        ["normal", "st-t abnormality", "lv hypertrophy"],
        help="normal: 정상 · st-t abnormality: ST-T 이상 · lv hypertrophy: 좌심실 비대"
    )

with col2:
    thalch  = st.number_input("최대 심박수 (thalch, bpm)", min_value=60, max_value=250, value=150)
    exang   = st.checkbox("운동 유발성 협심증 (exang)", value=False)
    oldpeak = st.number_input(
        "ST 하강 수치 (oldpeak)", min_value=0.0, max_value=10.0,
        value=1.0, step=0.1,
        help="운동으로 인한 안정시 대비 ST 분절 하강 수치"
    )
    slope   = st.selectbox(
        "최대 운동 ST 기울기 (slope)",
        ["flat", "upsloping", "downsloping"],
        help="flat: 평탄 · upsloping: 상향 · downsloping: 하향"
    )

st.divider()

if st.button("예측하기", type="primary", use_container_width=True):
    try:
        model = load_model()
    except FileNotFoundError:
        st.error("모델 파일을 찾을 수 없습니다. 'model/binary_mlp_final.pth' 경로를 확인해주세요.")
        st.stop()

    inputs = {
        'age': age, 'sex': sex, 'cp': cp,
        'trestbps': trestbps, 'chol': chol, 'fbs': fbs,
        'restecg': restecg, 'thalch': thalch, 'exang': exang,
        'oldpeak': oldpeak, 'slope': slope,
    }

    x = preprocess(inputs)

    with torch.no_grad():
        logits = model(x)
        proba  = torch.softmax(logits, dim=1).squeeze()
        pred   = logits.argmax(dim=1).item()

    prob_normal  = proba[0].item()
    prob_disease = proba[1].item()

    st.subheader("예측 결과")

    if pred == 0:
        st.success("✅ 정상 — 심장질환 가능성이 낮습니다.")
    else:
        st.error("⚠️ 심장질환 의심 — 전문의 진료를 권장합니다.")

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric(label="정상 확률", value=f"{prob_normal:.1%}")
    with col_b:
        st.metric(label="질환 확률", value=f"{prob_disease:.1%}")

    st.progress(prob_disease, text=f"질환 위험도  {prob_disease:.1%}")

    with st.expander("입력값 확인"):
        summary = {
            "나이": age, "성별": sex, "흉통유형": cp,
            "혈압(mmHg)": trestbps, "콜레스테롤(mg/dl)": chol,
            "공복혈당>120": "예" if fbs else "아니오",
            "심전도": restecg, "최대심박수": thalch,
            "운동협심증": "예" if exang else "아니오",
            "ST하강": oldpeak, "ST기울기": slope,
        }
        st.table(pd.DataFrame(summary.items(), columns=["항목", "값"]))

    st.caption("※ 본 예측 결과는 연구용이며 의학적 진단을 대체하지 않습니다.")
