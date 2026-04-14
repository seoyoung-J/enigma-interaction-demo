import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from pathlib import Path

# =========================
# paths
# =========================
# 스크립트 위치 기반 프로젝트 루트
BASE_DIR = Path(__file__).resolve().parent

# enigma_artifacts: 프로젝트 루트 기준 상대경로
ARTIFACT_DIR = BASE_DIR / "enigma_artifacts"

# enigma_outputs: 프로젝트 루트 기준 상대경로
OUTPUT_DIR = BASE_DIR / "enigma_outputs"
CSV_DIR = OUTPUT_DIR / "csv"
EDA_DIR = CSV_DIR / "eda"
SUMMARY_DIR = CSV_DIR / "summary"

# 로컬 데이터 디렉토리 (프로젝트 루트 기반 상대 경로)
FRAMES_DIR = BASE_DIR / "data" / "frames"
FEATURE_DIR = BASE_DIR / "data" / "exported_features" / "dinov2_vitg14"
ANNOTATION_PATH = BASE_DIR / "data" / "ENIGMA-51_annotations_master.json"

device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# model
# =========================
class SequenceClassifier(nn.Module):
    def __init__(
        self,
        input_dim,
        num_classes,
        model_type="LSTM",
        hidden_size=128,
        num_layers=1,
        dropout=0.3,
        bidirectional=False,
        head_type="linear"
    ):
        """
        Args:
            head_type: "linear", "mlp", or "layernorm_relu_linear"
        """
        super().__init__()
        self.model_type = model_type
        self.bidirectional = bidirectional
        self.head_type = head_type

        rnn_cls = {"GRU": nn.GRU, "LSTM": nn.LSTM}[model_type]
        self.rnn = rnn_cls(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )

        out_dim = hidden_size * (2 if bidirectional else 1)
        
        if head_type == "linear":
            self.head = nn.Linear(out_dim, num_classes)
        elif head_type == "mlp":
            self.head = nn.Sequential(
                nn.Linear(out_dim, 256),
                nn.ReLU(),
                nn.Linear(256, num_classes)
            )
        elif head_type == "layernorm_relu_linear":
            self.head = nn.Sequential(
                nn.LayerNorm(out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, num_classes)
            )
        else:
            raise ValueError(f"Unknown head_type: {head_type}")

    def forward(self, x):
        out, hidden = self.rnn(x)
        if self.model_type == "LSTM":
            hidden = hidden[0]  # h_n only

        if self.bidirectional:
            feat = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            feat = hidden[-1]

        return self.head(feat)

# =========================
# utils
# =========================
def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def infer_head_type_and_out_dim(state_dict):
    """
    state_dict의 head.* 키를 분석하여 head_type과 out_dim 추론
    
    Returns:
        (head_type, out_dim) 튜플
    """
    head_keys = sorted([k for k in state_dict.keys() if k.startswith("head.")])
    
    if not head_keys:
        return "linear", None
    
    head_0_weight = state_dict.get("head.0.weight")
    if head_0_weight is None:
        return "linear", None
    
    weight_shape = head_0_weight.shape
    
    # 1D shape: LayerNorm의 weight
    if len(weight_shape) == 1:
        out_dim = weight_shape[0]
        # head.0: LayerNorm, head.1: ReLU(implicit), head.2: Linear
        if "head.2.weight" in state_dict:
            return "layernorm_relu_linear", out_dim
        else:
            return "linear", out_dim
    
    # 2D shape: Linear의 weight (out_features, in_features)
    elif len(weight_shape) == 2:
        out_dim = weight_shape[1]  # in_features
        # head.0: Linear, head.1: ReLU(implicit), head.2: Linear
        if "head.2.weight" in state_dict:
            return "mlp", out_dim
        else:
            return "linear", out_dim
    
    return "linear", None

@st.cache_resource
def load_artifact_bundle(task_dir_name: str):
    """
    enigma_artifacts/{task_dir_name} 에서 아티팩트를 로드합니다.
    
    Args:
        task_dir_name: "interaction", "current_pair", 또는 "future_pair"
    
    Returns:
        (model, scaler, label_info, config) 튜플
    """
    task_dir = ARTIFACT_DIR / task_dir_name
    
    # 디렉토리 존재 여부 확인
    if not task_dir.exists():
        raise FileNotFoundError(f"폴더를 찾을 수 없음: {task_dir}")
    
    # 필요한 파일들
    config_path = task_dir / "config.json"
    label_info_path = task_dir / "label_info.json"
    scaler_path = task_dir / "scaler.pkl"
    model_path = task_dir / "model.pt"
    
    # 각 파일 존재 여부 확인
    missing_files = []
    for fpath, fname in [
        (config_path, "config.json"),
        (label_info_path, "label_info.json"),
        (scaler_path, "scaler.pkl"),
        (model_path, "model.pt"),
    ]:
        if not fpath.exists():
            missing_files.append(fname)
    
    if missing_files:
        raise FileNotFoundError(
            f"[{task_dir_name}] 폴더에서 다음 파일(들)이 부족합니다:\n"
            f"- {', '.join(missing_files)}\n"
            f"폴더 경로: {task_dir}"
        )
    
    # 파일 로드
    config = load_json(config_path)
    label_info = load_json(label_info_path)
    scaler = joblib.load(scaler_path)
    
    # state_dict 먼저 로드
    state_dict = torch.load(model_path, map_location=device)
    
    # state_dict 분석하여 head_type 추론
    head_type, inferred_out_dim = infer_head_type_and_out_dim(state_dict)
    
    # 로그 출력: head 구조 정보
    print(f"\n{'='*60}")
    print(f"[{task_dir_name}] 모델 로드")
    print(f"{'='*60}")
    head_keys = sorted([k for k in state_dict.keys() if k.startswith("head.")])
    print(f"Head keys and shapes:")
    for key in head_keys:
        print(f"  {key:30} {state_dict[key].shape}")
    print(f"Inferred head_type: {head_type}")
    print(f"Inferred out_dim: {inferred_out_dim}")
    print(f"{'='*60}\n")
    
    # 모델 생성
    model = SequenceClassifier(
        input_dim=config["input_dim"],
        num_classes=config["num_classes"],
        model_type="LSTM" if config["model"] in ["LSTM", "BiLSTM"] else "GRU",
        hidden_size=config["hidden_size"],
        num_layers=1,
        dropout=config["dropout"],
        bidirectional=config["model"] == "BiLSTM",
        head_type=head_type
    )
    
    # state_dict 적용
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    return model, scaler, label_info, config

def transform_sequence_array(X_seq: np.ndarray, scaler):
    # X_seq: [B, T, D]
    B, T, D = X_seq.shape
    X2 = X_seq.reshape(B * T, D)
    X2 = scaler.transform(X2)
    return X2.reshape(B, T, D)

def predict_one_sequence(model, scaler, label_info, seq_np: np.ndarray):
    seq_np = transform_sequence_array(seq_np, scaler)
    x = torch.tensor(seq_np, dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))

    idx_to_label = label_info.get("idx_to_label", {})
    pred_label = idx_to_label.get(str(pred_idx), str(pred_idx))

    return pred_idx, pred_label, probs

def find_feature_path(frame_key: str):
    p = FEATURE_DIR / f"{frame_key}.npy"
    return p if p.exists() else None

def load_feature(frame_key: str):
    p = find_feature_path(frame_key)
    if p is None:
        return None

    arr = np.load(p)
    arr = np.squeeze(arr)   # (1,1536) -> (1536,)

    return arr

def make_seq_from_frame_keys(frame_keys):
    feats = []
    for fk in frame_keys:
        arr = load_feature(fk)
        if arr is None:
            return None

        if arr.ndim != 1:
            raise ValueError(f"{fk}.npy shape error: {arr.shape}")

        feats.append(arr.astype(np.float32))

    # [1, T, D]
    return np.stack(feats, axis=0)[None, ...]

def find_frame_image(frame_key: str):
    candidates = [
        FRAMES_DIR / f"{frame_key}.jpg",
        FRAMES_DIR / f"{frame_key}.png",
        FRAMES_DIR / f"{frame_key}.jpeg",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None

def format_prob_table(probs, label_info):
    idx_to_label = label_info.get("idx_to_label", {})
    rows = []
    for i, p in enumerate(probs):
        rows.append({
            "class_idx": i,
            "label": idx_to_label.get(str(i), str(i)),
            "probability": float(p)
        })
    df = pd.DataFrame(rows).sort_values("probability", ascending=False).reset_index(drop=True)
    return df

# =========================
# data
# =========================
@st.cache_data
def load_frames_df():
    try:
        df = pd.read_csv(EDA_DIR / "eda_frames.csv")
        return df
    except FileNotFoundError as e:
        st.error(f"eda_frames.csv not found: {EDA_DIR / 'eda_frames.csv'}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading eda_frames.csv: {str(e)}")
        st.stop()

@st.cache_data
def load_single_df():
    try:
        df = pd.read_csv(SUMMARY_DIR / "single_interaction_frames.csv")
        return df
    except FileNotFoundError:
        st.warning("⚠️ single_interaction_frames.csv 파일이 없습니다. 앱이 계속 실행됩니다.")
        return pd.DataFrame()
    except Exception as e:
        st.warning(f"⚠️ single_interaction_frames.csv 로드 중 오류 발생: {str(e)}")
        return pd.DataFrame()

# =========================
# app
# =========================
st.set_page_config(page_title="ENIGMA Interaction Demo", layout="wide")

st.title("산업 작업 상호작용 분류 데모")
st.caption("최근 4개 프레임을 기준으로 상호작용 유무 → 현재 pair → 미래 pair를 예측합니다.")

frames_df = load_frames_df()
single_df = load_single_df()

# sidebar
st.sidebar.header("입력 선택")

video_ids = sorted(frames_df["video_id"].astype(str).unique().tolist())
selected_video = st.sidebar.selectbox("비디오 선택", video_ids)

video_df = frames_df[frames_df["video_id"].astype(str) == selected_video].copy()
video_df = video_df.sort_values("timestamp").reset_index(drop=True)

if len(video_df) < 4:
    st.error("선택한 비디오의 프레임 수가 4개 미만입니다.")
    st.stop()

selected_end_idx = st.sidebar.slider(
    "현재 프레임 위치 선택",
    min_value=3,
    max_value=len(video_df) - 1,
    value=min(10, len(video_df) - 1),
    step=1
)

seq_len = 4
seq_df = video_df.iloc[selected_end_idx - seq_len + 1:selected_end_idx + 1].copy()
seq_frame_keys = seq_df["frame_key"].tolist()

st.sidebar.markdown("### 선택된 4개 frame_key")
for fk in seq_frame_keys:
    st.sidebar.write(fk)

run_btn = st.sidebar.button("예측 실행", width="stretch")

# main info
col_left, col_right = st.columns([1.1, 1.2])

with col_left:
    st.subheader("선택 정보")
    st.write("video_id:", selected_video)
    st.dataframe(seq_df[["frame_key", "timestamp", "num_interactions"]], width="stretch")

with col_right:
    st.subheader("선택된 프레임")
    img_cols = st.columns(seq_len)

    for col, fk in zip(img_cols, seq_frame_keys):
        img_path = find_frame_image(fk)
        with col:
            st.caption(fk)
            if img_path is not None:
                st.image(str(img_path), width="stretch")
            else:
                st.warning("이미지 없음")

if run_btn:
    seq_np = make_seq_from_frame_keys(seq_frame_keys)

    if seq_np is None:
        st.error("선택한 frame_key 중 일부에 대응하는 npy feature를 찾지 못했습니다.")
        st.stop()

    # artifacts 로드
    try:
        interaction_model, interaction_scaler, interaction_label_info, interaction_config = load_artifact_bundle("interaction")
        current_model, current_scaler, current_label_info, current_config = load_artifact_bundle("current_pair")
        future_model, future_scaler, future_label_info, future_config = load_artifact_bundle("future_pair")
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()
    except Exception as e:
        st.error(f"❌ 아티팩트 로드 중 오류 발생:\n{str(e)}")
        st.stop()

    st.markdown("---")
    st.subheader("예측 결과")

    # interaction prediction
    inter_idx, inter_label, inter_probs = predict_one_sequence(
        interaction_model,
        interaction_scaler,
        interaction_label_info,
        seq_np
    )

    r1, r2 = st.columns([0.9, 1.1])

    with r1:
        st.markdown("### 1) 상호작용 유무")
        st.success(f"예측 결과: **{inter_label}**")

    with r2:
        st.markdown("### 상호작용 유무 확률")
        inter_prob_df = format_prob_table(inter_probs, interaction_label_info)
        st.dataframe(inter_prob_df, width="stretch")

    if inter_idx == 1:
        # current pair
        cur_idx, cur_label, cur_probs = predict_one_sequence(
            current_model,
            current_scaler,
            current_label_info,
            seq_np
        )

        # future pair
        fut_idx, fut_label, fut_probs = predict_one_sequence(
            future_model,
            future_scaler,
            future_label_info,
            seq_np
        )

        st.markdown("---")
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("### 2) 현재 pair 분류")
            st.success(f"예측 pair: **{cur_label}**")
            cur_prob_df = format_prob_table(cur_probs, current_label_info)
            st.dataframe(cur_prob_df, width="stretch")

        with c2:
            st.markdown("### 3) 미래 pair 분류")
            st.info(f"다음 시점 예측 pair: **{fut_label}**")
            fut_prob_df = format_prob_table(fut_probs, future_label_info)
            st.dataframe(fut_prob_df, width="stretch")
    else:
        st.info("상호작용 없음으로 판정되어 pair 분류는 생략했습니다.")